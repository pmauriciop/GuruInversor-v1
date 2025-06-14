#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento Incremental - GuruInversor

Sistema avanzado para entrenamiento incremental de modelos LSTM que permite:
- Reentrenamiento automático con nuevos datos
- Preservación del conocimiento previo
- Detección y manejo de deriva del modelo
- Versionado automático de modelos
"""

import os
import sys
import logging
import joblib
import shutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Importar componentes del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.trainer import LSTMTrainer, TrainingConfig
from ml.model_evaluator import ModelEvaluator, EvaluationConfig
from ml.preprocessor import DataProcessor, ProcessingConfig
from data.collector import YahooFinanceCollector
from database.connection import get_database

logger = logging.getLogger(__name__)


@dataclass
class IncrementalConfig:
    """Configuración para entrenamiento incremental."""
    
    # Configuración de reentrenamiento
    retrain_threshold_days: int = 7  # Días antes de considerar reentrenamiento
    min_new_samples: int = 50  # Mínimo de nuevas muestras para reentrenar
    max_model_age_days: int = 30  # Edad máxima del modelo antes de forzar reentrenamiento
    
    # Configuración de deriva del modelo
    performance_degradation_threshold: float = 0.15  # Umbral de degradación (15%)
    drift_detection_window: int = 100  # Ventana para detectar deriva
    
    # Configuración de versionado
    max_model_versions: int = 5  # Máximo número de versiones a mantener
    backup_before_retrain: bool = True  # Respaldar modelo antes de reentrenar
    
    # Configuración de datos incrementales
    incremental_data_overlap: int = 20  # Días de solapamiento con datos anteriores
    validation_split: float = 0.2  # Porcentaje para validación
    
    # Rutas
    models_directory: str = "ml/results/models"
    incremental_logs_directory: str = "ml/results/incremental_logs"
    
    def __post_init__(self):
        """Crear directorios necesarios."""
        Path(self.models_directory).mkdir(parents=True, exist_ok=True)
        Path(self.incremental_logs_directory).mkdir(parents=True, exist_ok=True)


class IncrementalTrainer:
    """
    Entrenador incremental avanzado para modelos LSTM.
    
    Maneja el reentrenamiento automático, detección de deriva,
    y versionado de modelos para mantener el rendimiento óptimo.
    """
    
    def __init__(self, config: IncrementalConfig = None):
        """
        Inicializar entrenador incremental.
          Args:
            config: Configuración para entrenamiento incremental
        """
        self.config = config or IncrementalConfig()
        self.db_manager = get_database()
        self.db = self.db_manager  # Alias para compatibilidad
        self.collector = YahooFinanceCollector()
        self.processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        
        # Estados internos
        self.model_registry = {}  # Registro de modelos por ticker
        self.performance_history = {}  # Historial de rendimiento
        
        # Configurar logging específico
        self._setup_logging()
        logger.info("🔄 IncrementalTrainer inicializado")
        logger.info(f"📁 Directorio de modelos: {self.config.models_directory}")
        logger.info(f"📊 Umbral de degradación: {self.config.performance_degradation_threshold:.1%}")
    
    def _setup_logging(self):
        """Configurar logging específico para entrenamiento incremental."""
        log_file = Path(self.config.incremental_logs_directory) / f"incremental_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Evitar duplicar handlers
        handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
            for h in logger.handlers
        )
        
        if not handler_exists:
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                
                logger.addHandler(file_handler)
            except Exception as e:
                # Si hay problemas con el logging de archivo, usar solo consola
                logger.warning(f"No se pudo configurar logging de archivo: {e}")
    
    def check_retrain_need(self, ticker: str) -> Dict[str, Any]:
        """
        Verificar si un modelo necesita reentrenamiento.
        
        Args:
            ticker: Símbolo del ticker
            
        Returns:
            Diccionario con información sobre necesidad de reentrenamiento
        """
        logger.info(f"🔍 Verificando necesidad de reentrenamiento para {ticker}")
        
        result = {
            'ticker': ticker,
            'needs_retrain': False,
            'reasons': [],
            'model_info': {},
            'recommendations': []
        }
        
        try:
            # 1. Verificar existencia del modelo
            model_path = self._get_model_path(ticker)
            if not model_path.exists():
                result['needs_retrain'] = True
                result['reasons'].append('No existe modelo previo')
                result['recommendations'].append('Entrenar modelo inicial')
                return result
            
            # 2. Cargar información del modelo
            model_info = self._load_model_info(ticker)
            result['model_info'] = model_info
            
            # 3. Verificar edad del modelo
            model_age = self._calculate_model_age(model_info)
            if model_age > self.config.max_model_age_days:
                result['needs_retrain'] = True
                result['reasons'].append(f'Modelo demasiado antiguo ({model_age} días)')
                result['recommendations'].append('Reentrenamiento por edad')
            
            # 4. Verificar disponibilidad de nuevos datos
            new_data_info = self._check_new_data_availability(ticker, model_info)
            if new_data_info['new_samples'] >= self.config.min_new_samples:
                result['needs_retrain'] = True
                result['reasons'].append(f'Nuevos datos disponibles ({new_data_info["new_samples"]} muestras)')
                result['recommendations'].append('Entrenamiento incremental')
            
            # 5. Verificar degradación del rendimiento
            performance_check = self._check_performance_degradation(ticker)
            if performance_check['degraded']:
                result['needs_retrain'] = True
                result['reasons'].append(f'Degradación del rendimiento ({performance_check["degradation"]:.1%})')
                result['recommendations'].append('Reentrenamiento completo')
            
            # 6. Logging de resultados
            if result['needs_retrain']:
                logger.warning(f"⚠️ {ticker} necesita reentrenamiento: {', '.join(result['reasons'])}")
            else:
                logger.info(f"✅ {ticker} no necesita reentrenamiento")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error verificando necesidad de reentrenamiento para {ticker}: {e}")
            result['needs_retrain'] = True
            result['reasons'].append(f'Error en verificación: {e}')
            return result
    
    def retrain_model(self, ticker: str, retrain_type: str = 'incremental') -> Dict[str, Any]:
        """
        Reentrenar modelo para un ticker específico.
        
        Args:
            ticker: Símbolo del ticker
            retrain_type: Tipo de reentrenamiento ('incremental' o 'complete')
            
        Returns:
            Diccionario con resultados del reentrenamiento
        """
        logger.info(f"🔄 Iniciando reentrenamiento {retrain_type} para {ticker}")
        
        result = {
            'ticker': ticker,
            'retrain_type': retrain_type,
            'success': False,
            'start_time': datetime.now(),
            'end_time': None,
            'old_performance': {},
            'new_performance': {},
            'improvement': {},
            'model_version': None,
            'errors': []
        }
        
        try:
            # 1. Backup del modelo existente si es necesario
            if self.config.backup_before_retrain:
                backup_result = self._backup_current_model(ticker)
                if not backup_result['success']:
                    result['errors'].append('Error en backup del modelo')
                    return result
            
            # 2. Obtener rendimiento del modelo actual
            if retrain_type == 'incremental':
                result['old_performance'] = self._get_current_performance(ticker)
            
            # 3. Recolectar y preparar datos
            data_result = self._prepare_training_data(ticker, retrain_type)
            if not data_result['success']:
                result['errors'].extend(data_result['errors'])
                return result
            
            # 4. Configurar entrenador
            trainer = self._setup_trainer(ticker, retrain_type)
            
            # 5. Entrenar modelo
            if retrain_type == 'incremental':
                training_result = self._perform_incremental_training(
                    trainer, ticker, data_result['data']
                )
            else:
                training_result = self._perform_complete_training(
                    trainer, ticker, data_result['data']
                )
            
            if not training_result['success']:
                result['errors'].extend(training_result['errors'])
                return result
            
            # 6. Evaluar nuevo modelo
            evaluation_result = self._evaluate_retrained_model(
                ticker, training_result['model'], data_result['data']
            )
            result['new_performance'] = evaluation_result['metrics']
            
            # 7. Calcular mejora
            if retrain_type == 'incremental' and result['old_performance']:
                result['improvement'] = self._calculate_improvement(
                    result['old_performance'], result['new_performance']
                )
            
            # 8. Guardar modelo y metadatos
            save_result = self._save_retrained_model(
                ticker, training_result['model'], result
            )
            result['model_version'] = save_result['version']
            
            # 9. Actualizar registro
            self._update_model_registry(ticker, result)
            
            result['success'] = True
            result['end_time'] = datetime.now()
            
            duration = (result['end_time'] - result['start_time']).total_seconds()
            logger.info(f"✅ Reentrenamiento {retrain_type} completado para {ticker} en {duration:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en reentrenamiento para {ticker}: {e}")
            result['errors'].append(str(e))
            result['end_time'] = datetime.now()
            return result
    
    def batch_retrain(self, tickers: List[str], force: bool = False) -> Dict[str, Any]:
        """
        Reentrenar múltiples modelos en lote.
        
        Args:
            tickers: Lista de tickers para verificar/reentrenar
            force: Forzar reentrenamiento incluso si no es necesario
            
        Returns:
            Diccionario con resultados del reentrenamiento en lote
        """
        logger.info(f"🔄 Iniciando reentrenamiento en lote para {len(tickers)} tickers")
        
        batch_result = {
            'total_tickers': len(tickers),
            'retrained': [],
            'skipped': [],
            'failed': [],
            'summary': {},
            'start_time': datetime.now(),
            'end_time': None
        }
        
        for ticker in tickers:
            try:
                # Verificar necesidad de reentrenamiento
                if not force:
                    check_result = self.check_retrain_need(ticker)
                    if not check_result['needs_retrain']:
                        batch_result['skipped'].append({
                            'ticker': ticker,
                            'reason': 'No necesita reentrenamiento'
                        })
                        continue
                
                # Determinar tipo de reentrenamiento
                retrain_type = 'incremental' if not force else 'complete'
                
                # Realizar reentrenamiento
                retrain_result = self.retrain_model(ticker, retrain_type)
                
                if retrain_result['success']:
                    batch_result['retrained'].append(retrain_result)
                else:
                    batch_result['failed'].append(retrain_result)
                
            except Exception as e:
                logger.error(f"❌ Error procesando {ticker} en lote: {e}")
                batch_result['failed'].append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        # Generar resumen
        batch_result['summary'] = {
            'retrained_count': len(batch_result['retrained']),
            'skipped_count': len(batch_result['skipped']),
            'failed_count': len(batch_result['failed']),
            'success_rate': len(batch_result['retrained']) / len(tickers) if tickers else 0
        }
        
        batch_result['end_time'] = datetime.now()
        duration = (batch_result['end_time'] - batch_result['start_time']).total_seconds()
        
        logger.info(f"✅ Reentrenamiento en lote completado en {duration:.1f}s")
        logger.info(f"📊 Resultados: {batch_result['summary']['retrained_count']} reentrenados, "
                   f"{batch_result['summary']['skipped_count']} omitidos, "
                   f"{batch_result['summary']['failed_count']} fallidos")
        
        return batch_result
    
    def _get_model_path(self, ticker: str) -> Path:
        """Obtener ruta del modelo para un ticker."""
        return Path(self.config.models_directory) / f"{ticker}_model.keras"
    
    def _load_model_info(self, ticker: str) -> Dict[str, Any]:
        """Cargar información del modelo."""
        info_path = Path(self.config.models_directory) / f"{ticker}_model_info.json"
        if info_path.exists():
            import json
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _calculate_model_age(self, model_info: Dict[str, Any]) -> int:
        """Calcular edad del modelo en días."""
        if not model_info or 'created_date' not in model_info:
            return float('inf')
        
        created_date = datetime.fromisoformat(model_info['created_date'])
        return (datetime.now() - created_date).days
    
    def _check_new_data_availability(self, ticker: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verificar disponibilidad de nuevos datos."""
        result = {'new_samples': 0, 'last_data_date': None}
        
        try:
            # Obtener fecha de últimos datos del modelo
            last_training_date = model_info.get('last_training_date')
            if not last_training_date:
                return result
            
            last_date = datetime.fromisoformat(last_training_date)
            
            # Obtener datos más recientes
            end_date = datetime.now()
            start_date = last_date + timedelta(days=1)
            
            if start_date >= end_date:
                return result
            
            # Verificar cantidad de nuevos datos disponibles
            new_data = self.collector.get_stock_data(
                ticker, start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if new_data is not None and not new_data.empty:
                result['new_samples'] = len(new_data)
                result['last_data_date'] = new_data.index[-1].isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error verificando nuevos datos para {ticker}: {e}")
            return result
    def _check_performance_degradation(self, ticker: str) -> Dict[str, Any]:
        """Verificar degradación del rendimiento de manera optimizada."""
        result = {'degraded': False, 'degradation': 0.0}
        
        try:
            # Verificación temprana: obtener historial
            history = self.performance_history.get(ticker)
            if not history or len(history) < 2:
                return result
            
            # Optimización: usar solo últimos elementos para cálculo eficiente
            if len(history) < 5:
                return result
            
            # Obtener scores de manera eficiente
            recent_scores = [h['score'] for h in history[-3:]]  # Últimos 3
            baseline_scores = [h['score'] for h in history[-8:-3]]  # 5 anteriores
            
            if not baseline_scores:
                return result
            
            # Cálculo optimizado usando numpy
            recent_avg = np.mean(recent_scores)
            baseline_avg = np.mean(baseline_scores)
            
            if baseline_avg == 0:
                return result
            
            degradation = (baseline_avg - recent_avg) / baseline_avg
            
            result['degradation'] = float(degradation)  # Asegurar tipo Python
            result['degraded'] = bool(degradation > self.config.performance_degradation_threshold)
            
            return result
            
        except Exception as e:
            logger.error(f"Error verificando degradación para {ticker}: {e}")
            return result
    
    def _backup_current_model(self, ticker: str) -> Dict[str, Any]:
        """Respaldar modelo actual."""
        result = {'success': False, 'backup_path': None}
        
        try:
            model_path = self._get_model_path(ticker)
            if not model_path.exists():
                result['success'] = True  # No hay modelo que respaldar
                return result
            
            # Crear directorio de backups
            backup_dir = Path(self.config.models_directory) / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Generar nombre de backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"{ticker}_model_backup_{timestamp}.keras"
              # Copiar modelo
            shutil.copy2(model_path, backup_path)
            
            result['success'] = True
            result['backup_path'] = str(backup_path)
            
            logger.info(f"💾 Modelo respaldado en: {backup_path}")
            
            # Limpiar backups antiguos
            self._cleanup_old_backups(ticker)
            
            return result
            
        except Exception as e:
            logger.error(f"Error respaldando modelo para {ticker}: {e}")
            return result
    
    def _get_current_performance(self, ticker: str) -> Dict[str, float]:
        """Obtener rendimiento actual del modelo."""
        try:
            # Cargar información del modelo
            model_info = self._load_model_info(ticker)
            return model_info.get('last_performance', {})
        except Exception as e:
            logger.error(f"Error obteniendo rendimiento actual para {ticker}: {e}")
            return {}
    
    def _prepare_training_data(self, ticker: str, retrain_type: str) -> Dict[str, Any]:
        """Preparar datos para entrenamiento."""
        result = {'success': False, 'data': None, 'errors': []}
        
        try:
            # Determinar rango de fechas
            if retrain_type == 'incremental':
                # Para incremental, usar últimos datos + overlap
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)  # Último año
            else:
                # Para completo, usar todo el histórico disponible
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*3)  # 3 años
            
            # Recolectar datos
            data = self.collector.get_stock_data(
                ticker, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if data is None or data.empty:
                result['errors'].append('No se pudieron obtener datos')
                return result
            
            # Procesar datos
            processed_data = self.processor.process_stock_data(data)
            
            if processed_data is None or processed_data.empty:
                result['errors'].append('Error procesando datos')
                return result
            
            result['success'] = True
            result['data'] = processed_data
            
            logger.info(f"📊 Datos preparados para {ticker}: {len(processed_data)} muestras")
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparando datos para {ticker}: {e}")
            result['errors'].append(str(e))
            return result
    
    def _setup_trainer(self, ticker: str, retrain_type: str) -> LSTMTrainer:
        """Configurar entrenador según el tipo de reentrenamiento."""
        training_config = TrainingConfig()
        
        if retrain_type == 'incremental':
            # Configuración para entrenamiento incremental
            training_config.epochs = 50  # Menos épocas
            training_config.learning_rate = 0.0001  # Learning rate menor
            training_config.early_stopping_patience = 10
        else:
            # Configuración para entrenamiento completo
            training_config.epochs = 100
            training_config.learning_rate = 0.001
            training_config.early_stopping_patience = 15
        
        return LSTMTrainer(config=training_config)
    
    def _perform_incremental_training(self, trainer: LSTMTrainer, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Realizar entrenamiento incremental."""
        result = {'success': False, 'model': None, 'errors': []}
        
        try:
            # Cargar modelo existente
            model_path = self._get_model_path(ticker)
            if model_path.exists():
                existing_model = keras.models.load_model(model_path)
                logger.info(f"🔄 Modelo existente cargado para {ticker}")
            else:
                result['errors'].append('No se encontró modelo existente para entrenamiento incremental')
                return result
            
            # Preparar datos incrementales
            X, y = self.processor.create_sequences(data)
            
            if len(X) == 0:
                result['errors'].append('No se pudieron crear secuencias de datos')
                return result
            
            # Entrenar incrementalmente (transfer learning)
            # Congelar algunas capas iniciales para preservar conocimiento
            for i, layer in enumerate(existing_model.layers[:-2]):
                layer.trainable = False
            
            # Compilar con learning rate menor
            existing_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=trainer.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Entrenar
            history = trainer._train_model(existing_model, X, y)
            
            result['success'] = True
            result['model'] = existing_model
            result['history'] = history
            
            logger.info(f"✅ Entrenamiento incremental completado para {ticker}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en entrenamiento incremental para {ticker}: {e}")
            result['errors'].append(str(e))
            return result
    
    def _perform_complete_training(self, trainer: LSTMTrainer, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Realizar entrenamiento completo."""
        result = {'success': False, 'model': None, 'errors': []}
        
        try:
            # Entrenar modelo desde cero
            training_result = trainer.train_model(data, ticker)
            
            if training_result['success']:
                result['success'] = True
                result['model'] = training_result['model']
                result['history'] = training_result['history']
                
                logger.info(f"✅ Entrenamiento completo completado para {ticker}")
            else:
                result['errors'].extend(training_result.get('errors', []))
            
            return result
            
        except Exception as e:
            logger.error(f"Error en entrenamiento completo para {ticker}: {e}")
            result['errors'].append(str(e))
            return result
    
    def _evaluate_retrained_model(self, ticker: str, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluar modelo reentrenado."""
        result = {'success': False, 'metrics': {}}
        
        try:
            # Preparar datos de evaluación
            X, y = self.processor.create_sequences(data)
            
            if len(X) == 0:
                return result
            
            # Usar últimos datos para evaluación
            test_size = int(len(X) * 0.2)
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            
            # Evaluar modelo
            evaluation_result = self.evaluator.evaluate_model_comprehensive(
                model, X_test, y_test, ticker=ticker
            )
            
            result['success'] = True
            result['metrics'] = evaluation_result['overall_score']
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluando modelo reentrenado para {ticker}: {e}")
            return result
    
    def _calculate_improvement(self, old_performance: Dict, new_performance: Dict) -> Dict[str, float]:
        """Calcular mejora en el rendimiento."""
        improvement = {}
        
        for metric in ['overall_score', 'accuracy_score', 'directional_score']:
            if metric in old_performance and metric in new_performance:
                old_value = old_performance[metric]
                new_value = new_performance[metric]
                improvement[f'{metric}_improvement'] = new_value - old_value
                improvement[f'{metric}_improvement_pct'] = (new_value - old_value) / old_value if old_value != 0 else 0
        
        return improvement
    
    def _save_retrained_model(self, ticker: str, model, training_result: Dict) -> Dict[str, Any]:
        """Guardar modelo reentrenado con versionado."""
        result = {'success': False, 'version': None}
        
        try:
            # Generar versión
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Guardar modelo
            model_path = self._get_model_path(ticker)
            model.save(model_path)
            
            # Guardar información del modelo
            model_info = {
                'ticker': ticker,
                'version': version,
                'created_date': datetime.now().isoformat(),
                'last_training_date': datetime.now().isoformat(),
                'retrain_type': training_result['retrain_type'],
                'last_performance': training_result['new_performance'],
                'config': asdict(self.config)
            }
            
            info_path = Path(self.config.models_directory) / f"{ticker}_model_info.json"
            import json
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            result['success'] = True
            result['version'] = version
            
            logger.info(f"💾 Modelo guardado para {ticker} versión {version}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error guardando modelo para {ticker}: {e}")
            return result
    
    def _update_model_registry(self, ticker: str, training_result: Dict):
        """Actualizar registro de modelos."""
        self.model_registry[ticker] = {
            'last_update': datetime.now().isoformat(),
            'version': training_result.get('model_version'),
            'performance': training_result.get('new_performance', {}),
            'retrain_type': training_result.get('retrain_type')
        }
        
        # Actualizar historial de rendimiento
        if ticker not in self.performance_history:
            self.performance_history[ticker] = []
        
        self.performance_history[ticker].append({
            'date': datetime.now().isoformat(),
            'score': training_result.get('new_performance', {}).get('overall_score', 0),
            'type': training_result.get('retrain_type')
        })
        
        # Mantener solo últimas entradas
        self.performance_history[ticker] = self.performance_history[ticker][-20:]
    
    def get_model_status(self, ticker: str = None) -> Dict[str, Any]:
        """
        Obtener estado de modelos.
        
        Args:
            ticker: Ticker específico (None para todos)
            
        Returns:
            Estado de los modelos
        """
        if ticker:
            return self.model_registry.get(ticker, {})
        return self.model_registry.copy()
    
    def generate_incremental_report(self) -> str:
        """Generar reporte de entrenamiento incremental."""
        report = []
        report.append("📊 REPORTE DE ENTRENAMIENTO INCREMENTAL")
        report.append("=" * 60)
        
        # Estadísticas generales
        total_models = len(self.model_registry)
        report.append(f"Total de modelos: {total_models}")
        
        if total_models > 0:
            # Análisis por ticker
            report.append("\n🔍 ANÁLISIS POR TICKER:")
            for ticker, info in self.model_registry.items():
                report.append(f"\n📈 {ticker}:")
                report.append(f"   Última actualización: {info.get('last_update', 'N/A')}")
                report.append(f"   Versión: {info.get('version', 'N/A')}")
                report.append(f"   Tipo último entrenamiento: {info.get('retrain_type', 'N/A')}")
                
                performance = info.get('performance', {})
                if performance:
                    report.append(f"   Score general: {performance.get('overall_score', 0):.3f}")
                    report.append(f"   Grade: {performance.get('grade', 'N/A')}")
        
        # Recomendaciones
        report.append("\n💡 RECOMENDACIONES:")
        if total_models == 0:
            report.append("   - No hay modelos entrenados aún")
        else:
            # Verificar modelos que necesitan atención
            needs_attention = []
            for ticker in self.model_registry.keys():
                check = self.check_retrain_need(ticker)
                if check['needs_retrain']:
                    needs_attention.append(ticker)
            
            if needs_attention:
                report.append(f"   - {len(needs_attention)} modelos necesitan reentrenamiento: {', '.join(needs_attention)}")
            else:
                report.append("   - Todos los modelos están actualizados")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def _cleanup_old_backups(self, ticker: str, keep_count: int = 5):
        """Limpiar backups antiguos para mantener solo los más recientes."""
        try:
            backup_dir = Path(self.config.models_directory) / "backups"
            if not backup_dir.exists():
                return
            
            # Buscar backups del ticker específico
            backup_pattern = f"{ticker}_model_backup_*.keras"
            backup_files = list(backup_dir.glob(backup_pattern))
            
            # Ordenar por fecha de modificación (más reciente primero)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Eliminar archivos antiguos si exceden el límite
            if len(backup_files) > keep_count:
                for old_backup in backup_files[keep_count:]:
                    try:
                        old_backup.unlink()
                        logger.info(f"🗑️ Backup antiguo eliminado: {old_backup.name}")
                    except Exception as e:
                        logger.warning(f"No se pudo eliminar backup {old_backup}: {e}")
                        
        except Exception as e:
            logger.error(f"Error limpiando backups para {ticker}: {e}")

# Función de utilidad para uso directo
def create_incremental_trainer(config: IncrementalConfig = None) -> IncrementalTrainer:
    """
    Crear instancia de entrenador incremental.
    
    Args:
        config: Configuración personalizada
        
    Returns:
        Instancia de IncrementalTrainer
    """
    return IncrementalTrainer(config)


if __name__ == "__main__":
    # Ejemplo de uso
    trainer = create_incremental_trainer()
    
    # Verificar estado
    print(trainer.generate_incremental_report())
    
    # Ejemplo de verificación para un ticker
    # check_result = trainer.check_retrain_need("AAPL")
    # print(f"Resultado verificación: {check_result}")
