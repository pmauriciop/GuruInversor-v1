#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODEL-002 - Entrenador LSTM Básico - GuruInversor

Implementación del entrenador LSTM que usa la arquitectura diseñada en MODEL-001
para entrenar modelos con datos reales y guardarlo en la base de datos.

Funcionalidades:
- Entrenamiento de modelos LSTM con datos históricos
- Validación temporal (no aleatoria)
- Guardado y carga de modelos entrenados
- Integración con base de datos
- Métricas de evaluación completas
- Gestión de hiperparámetros
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import time

# Añadir el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from ml.model_architecture import LSTMConfig, LSTMArchitect, create_lstm_model, validate_model_architecture
from ml.data_integration import DataIntegrator
from ml.preprocessor import DataProcessor, ProcessingConfig
from ml.metrics import calculate_regression_metrics, calculate_directional_accuracy, evaluate_model_performance, print_evaluation_report
from database.connection import Database
from database.models import TrainedModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuración específica para entrenamiento."""
    
    # Datos de entrenamiento
    train_split: float = 0.7  # Porcentaje para entrenamiento
    validation_split: float = 0.2  # Porcentaje para validación
    test_split: float = 0.1  # Porcentaje para prueba
    
    # Hiperparámetros de entrenamiento
    batch_size: int = 32
    epochs: int = 100
    validation_freq: int = 1  # Frecuencia de validación
    
    # Parada temprana
    patience: int = 15
    min_delta: float = 0.001
    
    # Guardado de modelos
    save_best_only: bool = True
    save_weights_only: bool = False
    
    # Métricas de seguimiento
    monitor_metric: str = 'val_loss'
      # Configuración de datos
    min_data_points: int = 1000  # Mínimo de puntos para entrenamiento
    
    def __post_init__(self):
        """Validar configuración."""
        if abs(self.train_split + self.validation_split + self.test_split - 1.0) > 0.001:
            raise ValueError("Las proporciones de split deben sumar 1.0")
        
        # Validación más flexible para permitir pruebas
        if self.min_data_points < 50:
            raise ValueError("Mínimo 50 puntos de datos necesarios")
        
        # Advertencia para valores muy bajos en producción
        if self.min_data_points < 500:
            import warnings
            warnings.warn(f"Usando {self.min_data_points} puntos de datos. Recomendado mínimo 500 para producción.", 
                         UserWarning)


class LSTMTrainer:
    """
    Entrenador principal para modelos LSTM.
    
    Maneja el proceso completo de entrenamiento desde preparación de datos
    hasta guardado del modelo entrenado.
    """
    
    def __init__(self, 
                 lstm_config: LSTMConfig = None,
                 training_config: TrainingConfig = None,
                 processing_config: ProcessingConfig = None):
        """
        Inicializar entrenador LSTM.
        
        Args:
            lstm_config: Configuración del modelo LSTM
            training_config: Configuración de entrenamiento
            processing_config: Configuración de procesamiento de datos
        """
        self.lstm_config = lstm_config or LSTMConfig()
        self.training_config = training_config or TrainingConfig()
        self.processing_config = processing_config or ProcessingConfig()        # Componentes principales
        self.data_integrator = DataIntegrator()
        self.db = Database()
        
        # Estado del entrenamiento
        self.model = None
        self.history = None
        self.data_processor = None
        self.training_start_time = None
        self.training_end_time = None
        
        # Configurar GPU
        self._setup_tensorflow()
        
        logger.info("✅ LSTMTrainer inicializado correctamente")
    
    def _setup_tensorflow(self):
        """Configurar TensorFlow para entrenamiento."""
        try:
            # Configurar GPU si está disponible
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"🎮 GPU configurada: {len(gpus)} dispositivo(s)")
            else:
                logger.info("💻 Usando CPU para entrenamiento")
            
            # Configurar semillas para reproducibilidad
            tf.random.set_seed(42)
            np.random.seed(42)
            
        except Exception as e:
            logger.warning(f"⚠️  Error configurando TensorFlow: {e}")
    
    def prepare_data(self, ticker: str, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preparar datos para entrenamiento con división temporal.
        
        Args:
            ticker: Símbolo de la acción
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Tuple con X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"📊 Preparando datos para {ticker}")
        
        # Obtener datos procesados
        X_sequences, y_targets, processor = self.data_integrator.get_processed_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            config=self.processing_config
        )
        
        self.data_processor = processor
        
        if len(X_sequences) == 0:
            raise ValueError(f"No hay datos suficientes para {ticker}")
        
        if len(X_sequences) < self.training_config.min_data_points:
            raise ValueError(f"Datos insuficientes: {len(X_sequences)} < {self.training_config.min_data_points}")
        
        logger.info(f"✅ Datos obtenidos: {len(X_sequences)} secuencias")
        
        # División temporal (no aleatoria)
        total_samples = len(X_sequences)
        train_size = int(total_samples * self.training_config.train_split)
        val_size = int(total_samples * self.training_config.validation_split)
        
        # Dividir datos temporalmente
        X_train = X_sequences[:train_size]
        y_train = y_targets[:train_size]
        
        X_val = X_sequences[train_size:train_size + val_size]
        y_val = y_targets[train_size:train_size + val_size]
        
        X_test = X_sequences[train_size + val_size:]
        y_test = y_targets[train_size + val_size:]
        
        logger.info(f"📈 División temporal:")
        logger.info(f"   🟢 Entrenamiento: {len(X_train)} secuencias")
        logger.info(f"   🟡 Validación: {len(X_val)} secuencias")
        logger.info(f"   🔴 Prueba: {len(X_test)} secuencias")
        
        # Validar shapes
        expected_shape = (self.lstm_config.sequence_length, self.lstm_config.n_features)
        if X_train.shape[1:] != expected_shape:
            raise ValueError(f"Shape incorrecto: {X_train.shape[1:]} != {expected_shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, model_type: str = 'basic') -> tf.keras.Model:
        """
        Construir modelo LSTM usando la arquitectura de MODEL-001.
        
        Args:
            model_type: Tipo de modelo ('basic', 'advanced', 'ensemble')
            
        Returns:
            Modelo Keras compilado
        """
        logger.info(f"🏗️  Construyendo modelo LSTM tipo: {model_type}")
        
        # Crear arquitecto y construir modelo
        architect = LSTMArchitect(self.lstm_config)
        
        if model_type == 'basic':
            model = architect.build_basic_model()
        elif model_type == 'advanced':
            model = architect.build_advanced_model()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Validar arquitectura
        is_valid = validate_model_architecture(
            model, 
            (self.lstm_config.sequence_length, self.lstm_config.n_features)
        )
        
        if not is_valid:
            raise ValueError("Arquitectura del modelo inválida")
        
        self.model = model
        logger.info(f"✅ Modelo construido: {model.count_params():,} parámetros")
        
        return model
    
    def get_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """
        Obtener callbacks para entrenamiento.
        
        Args:
            model_name: Nombre del modelo para archivos
            
        Returns:
            Lista de callbacks configurados
        """
        callbacks = []
        
        # Crear directorios necesarios
        models_dir = Path(__file__).parent.parent.parent / 'models'
        logs_dir = Path(__file__).parent.parent.parent / 'logs'
        models_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=self.training_config.monitor_metric,
            patience=self.training_config.patience,
            min_delta=self.training_config.min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
          # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / f'{model_name}_best.keras'),
            monitor=self.training_config.monitor_metric,
            save_best_only=self.training_config.save_best_only,
            save_weights_only=self.training_config.save_weights_only,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.training_config.monitor_metric,
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(logs_dir / f'tensorboard_{model_name}'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        logger.info(f"📋 Configurados {len(callbacks)} callbacks")
        return callbacks
    
    def train(self, ticker: str, 
             model_type: str = 'basic',
             start_date: Optional[datetime] = None,
             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Entrenar modelo LSTM completo.
        
        Args:
            ticker: Símbolo de la acción
            model_type: Tipo de modelo a entrenar
            start_date: Fecha de inicio de datos
            end_date: Fecha de fin de datos
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        logger.info(f"🚀 Iniciando entrenamiento para {ticker}")
        self.training_start_time = time.time()
        
        try:
            # 1. Preparar datos
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
                ticker, start_date, end_date
            )
            
            # 2. Construir modelo
            model = self.build_model(model_type)
            
            # 3. Configurar callbacks
            model_name = f"{ticker}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            callbacks = self.get_callbacks(model_name)
            
            # 4. Entrenar modelo
            logger.info("🏃‍♂️ Iniciando entrenamiento...")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.training_config.batch_size,
                epochs=self.training_config.epochs,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # No mezclar para preservar orden temporal
            )
            
            self.history = history
            self.training_end_time = time.time()
            training_time = self.training_end_time - self.training_start_time
            
            logger.info(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
            
            # 5. Evaluar modelo
            evaluation_results = self.evaluate_model(X_test, y_test)
            
            # 6. Guardar modelo en base de datos
            model_metadata = self.save_model_to_database(
                ticker=ticker,
                model_name=model_name,
                model_type=model_type,
                evaluation_results=evaluation_results,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                training_time=training_time
            )
            
            # 7. Preparar resultados
            results = {
                'model_name': model_name,
                'model_metadata': model_metadata,
                'training_time': training_time,
                'history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'epochs_trained': len(history.history['loss'])
                },
                'evaluation': evaluation_results,
                'data_stats': {
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'features': X_train.shape[-1]
                }
            }
            
            logger.info("🎉 Entrenamiento MODEL-002 completado exitosamente")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            raise
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluar modelo con conjunto de prueba.
        
        Args:
            X_test: Datos de entrada de prueba
            y_test: Objetivos de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        logger.info("📊 Evaluando modelo...")
          # Hacer predicciones
        predictions = self.model.predict(X_test, verbose=0)
        
        # Calcular métricas usando funciones disponibles
        # Aplanar arrays si son multidimensionales
        y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
        predictions_flat = predictions.flatten() if predictions.ndim > 1 else predictions
        
        # Calcular métricas de regresión
        regression_metrics = calculate_regression_metrics(y_test_flat, predictions_flat)
        
        # Calcular precisión direccional
        directional_metrics = calculate_directional_accuracy(y_test_flat, predictions_flat)
        
        # Combinar métricas
        metrics_dict = {**regression_metrics, **directional_metrics}
        
        # Agregar métricas adicionales específicas
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        if isinstance(test_loss, list):
            metrics_dict['test_loss'] = test_loss[0]
            # Si hay métricas adicionales
            for i, metric_name in enumerate(self.model.metrics_names[1:], 1):
                metrics_dict[f'test_{metric_name}'] = test_loss[i]
        else:
            metrics_dict['test_loss'] = test_loss
        
        logger.info(f"✅ Evaluación completada:")
        for metric, value in metrics_dict.items():
            logger.info(f"   📈 {metric}: {value:.6f}")
        
        return metrics_dict
    
    def save_model_to_database(self, 
                              ticker: str,
                              model_name: str,
                              model_type: str,
                              evaluation_results: Dict[str, float],
                              training_samples: int,
                              validation_samples: int,
                              training_time: float) -> Dict[str, Any]:
        """
        Guardar metadatos del modelo en la base de datos.
        
        Args:
            ticker: Símbolo de la acción
            model_name: Nombre del modelo
            model_type: Tipo de modelo
            evaluation_results: Resultados de evaluación
            training_samples: Número de muestras de entrenamiento
            validation_samples: Número de muestras de validación
            training_time: Tiempo de entrenamiento en segundos
            
        Returns:
            Diccionario con metadatos guardados
        """
        logger.info("💾 Guardando modelo en base de datos...")
        
        try:
            # Obtener stock_id
            with self.db.get_session() as session:
                from database.models import Stock
                stock = session.query(Stock).filter(Stock.symbol == ticker).first()
                if not stock:
                    raise ValueError(f"Stock {ticker} no encontrado en base de datos")
                
                # Crear metadatos del modelo
                model_path = f"models/{model_name}_best.h5"
                hyperparameters = {
                    'lstm_config': asdict(self.lstm_config),
                    'training_config': asdict(self.training_config),
                    'processing_config': asdict(self.processing_config)
                }
                
                # Crear registro de modelo entrenado
                trained_model = TrainedModel(
                    stock_id=stock.id,
                    model_path=model_path,
                    version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type=model_type,
                    accuracy=evaluation_results.get('test_loss', 0.0),
                    loss=evaluation_results.get('test_loss', 0.0),
                    rmse=evaluation_results.get('rmse', 0.0),
                    mae=evaluation_results.get('mae', 0.0),
                    mape=evaluation_results.get('mape', 0.0),
                    training_samples=training_samples,
                    validation_samples=validation_samples,
                    epochs_trained=len(self.history.history['loss']) if self.history else 0,
                    training_time_seconds=training_time,
                    hyperparameters=json.dumps(hyperparameters),
                    is_active=True
                )
                
                # Desactivar modelos anteriores
                session.query(TrainedModel).filter(
                    TrainedModel.stock_id == stock.id,
                    TrainedModel.model_type == model_type
                ).update({'is_active': False})
                
                # Guardar nuevo modelo
                session.add(trained_model)
                session.commit()
                
                model_metadata = {
                    'id': trained_model.id,
                    'stock_id': stock.id,
                    'ticker': ticker,
                    'model_path': model_path,
                    'version': trained_model.version,
                    'model_type': model_type,
                    'training_date': trained_model.training_date,
                    'is_active': True
                }
                
                logger.info(f"✅ Modelo guardado con ID: {trained_model.id}")
                return model_metadata
                
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")
            raise
    
    def load_model(self, model_id: int) -> tf.keras.Model:
        """
        Cargar modelo entrenado desde base de datos.
        
        Args:
            model_id: ID del modelo en base de datos
            
        Returns:
            Modelo Keras cargado
        """
        logger.info(f"📥 Cargando modelo ID: {model_id}")
        
        try:
            with self.db.get_session() as session:
                trained_model = session.query(TrainedModel).filter(
                    TrainedModel.id == model_id
                ).first()
                
                if not trained_model:
                    raise ValueError(f"Modelo con ID {model_id} no encontrado")
                
                # Reconstruir rutas
                model_path = Path(__file__).parent.parent.parent / trained_model.model_path
                
                if not model_path.exists():
                    raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
                
                # Cargar modelo
                model = tf.keras.models.load_model(str(model_path))
                
                # Cargar configuraciones
                hyperparameters = json.loads(trained_model.hyperparameters)
                self.lstm_config = LSTMConfig(**hyperparameters['lstm_config'])
                self.training_config = TrainingConfig(**hyperparameters['training_config'])
                self.processing_config = ProcessingConfig(**hyperparameters['processing_config'])
                
                self.model = model
                logger.info(f"✅ Modelo cargado exitosamente")
                
                return model
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hacer predicciones con modelo entrenado.
        
        Args:
            X: Datos de entrada
            
        Returns:
            Predicciones del modelo
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado o cargado")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen del último entrenamiento.
        
        Returns:
            Diccionario con resumen de entrenamiento
        """
        if self.history is None:
            return {}
        
        history_dict = self.history.history
        
        summary = {
            'epochs_completed': len(history_dict['loss']),
            'final_loss': history_dict['loss'][-1],
            'final_val_loss': history_dict['val_loss'][-1],
            'best_loss': min(history_dict['loss']),
            'best_val_loss': min(history_dict['val_loss']),
            'training_time': getattr(self, 'training_end_time', 0) - getattr(self, 'training_start_time', 0)
        }
        
        return summary


def train_lstm_model(ticker: str,
                    model_type: str = 'basic',
                    lstm_config: LSTMConfig = None,
                    training_config: TrainingConfig = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para entrenar un modelo LSTM.
    
    Args:
        ticker: Símbolo de la acción
        model_type: Tipo de modelo ('basic', 'advanced')
        lstm_config: Configuración del modelo LSTM
        training_config: Configuración de entrenamiento
        start_date: Fecha de inicio de datos
        end_date: Fecha de fin de datos
        
    Returns:
        Diccionario con resultados del entrenamiento
    """
    trainer = LSTMTrainer(
        lstm_config=lstm_config,
        training_config=training_config
    )
    
    return trainer.train(
        ticker=ticker,
        model_type=model_type,
        start_date=start_date,
        end_date=end_date
    )


if __name__ == "__main__":
    # Ejemplo de uso básico
    print("🤖 MODEL-002 - Entrenador LSTM Básico - GuruInversor")
    print("=" * 60)
    
    # Configuraciones de ejemplo
    lstm_config = LSTMConfig(
        sequence_length=60,
        n_features=12,
        lstm_units=[50, 30],
        dense_units=[25, 10],
        dropout_rate=0.2
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        epochs=50,  # Reducido para ejemplo
        patience=10
    )
    
    # Ejemplo de entrenamiento
    try:
        logger.info("🔧 Configurando entrenador...")
        trainer = LSTMTrainer(lstm_config, training_config)
        
        logger.info("📋 Configuraciones cargadas:")
        logger.info(f"   📊 Secuencia: {lstm_config.sequence_length} días")
        logger.info(f"   🎯 Features: {lstm_config.n_features}")
        logger.info(f"   🏋️‍♂️ Batch size: {training_config.batch_size}")
        logger.info(f"   🔄 Épocas: {training_config.epochs}")
        
        print("\n✅ Entrenador MODEL-002 inicializado correctamente")
        print("🚀 Listo para entrenar modelos LSTM con datos reales")
        
    except Exception as e:
        logger.error(f"❌ Error en inicialización: {e}")
        print("❌ Error en MODEL-002 - revisar configuración")
    
    print("=" * 60)
