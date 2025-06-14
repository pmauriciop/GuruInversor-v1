#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODEL-003 - Pipeline de Entrenamiento - GuruInversor

Pipeline automatizado para entrenar modelos LSTM que orquesta todo el proceso
desde la recolección de datos hasta la evaluación y guardado de modelos.

Funcionalidades:
- Automatización completa del entrenamiento
- Manejo de múltiples tickers y configuraciones
- Validación y logging detallado
- Gestión de errores robusta
- Entrenamiento en lotes (batch training)
- Comparación de modelos
- Programación de entrenamientos
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import schedule

# Añadir el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from ml.trainer import LSTMTrainer, TrainingConfig, train_lstm_model
from ml.model_architecture import LSTMConfig
from ml.preprocessor import ProcessingConfig
from ml.metrics import evaluate_model_performance, print_evaluation_report
from data.collector import YahooFinanceCollector
from database.connection import Database
from database.models import Stock, TrainedModel, HistoricalData

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuración del pipeline de entrenamiento."""
    
    # Configuraciones de entrenamiento
    lstm_config: LSTMConfig = None
    training_config: TrainingConfig = None
    processing_config: ProcessingConfig = None
    
    # Configuración de pipeline
    max_workers: int = 4  # Máximo de entrenamientos paralelos
    update_data: bool = True  # Actualizar datos antes de entrenar
    validate_before_training: bool = True  # Validar datos antes de entrenar
    save_intermediate_results: bool = True  # Guardar resultados intermedios
    
    # Configuración de datos
    min_data_days: int = 365  # Mínimo de días de datos requeridos
    end_date: Optional[datetime] = None  # Fecha fin (default: hoy)
    lookback_days: int = 1095  # Días hacia atrás para obtener datos (3 años)
    
    # Configuración de modelos
    model_types: List[str] = None  # Tipos de modelo a entrenar ['basic', 'advanced']
    
    # Configuración de logging y guardado
    log_level: str = 'INFO'
    save_logs: bool = True
    results_dir: str = 'pipeline_results'
    
    def __post_init__(self):
        """Inicializar configuraciones por defecto."""
        if self.lstm_config is None:
            self.lstm_config = LSTMConfig()
        
        if self.training_config is None:
            self.training_config = TrainingConfig()
        
        if self.processing_config is None:
            self.processing_config = ProcessingConfig()
        
        if self.model_types is None:
            self.model_types = ['basic', 'advanced']
        
        if self.end_date is None:
            self.end_date = datetime.now()


@dataclass
class TrainingJob:
    """Trabajo de entrenamiento individual."""
    ticker: str
    model_type: str
    priority: int = 1  # 1=alta, 5=baja
    created_at: datetime = None
    status: str = 'pending'  # pending, running, completed, failed
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PipelineResults:
    """Resultados del pipeline de entrenamiento."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    start_time: datetime = None
    end_time: datetime = None
    results: List[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        """Calcular tasa de éxito."""
        if self.total_jobs == 0:
            return 0.0
        return self.completed_jobs / self.total_jobs
    
    @property
    def duration(self) -> float:
        """Calcular duración en segundos."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class TrainingPipeline:
    """
    Pipeline principal para automatizar el entrenamiento de modelos LSTM.
    
    Maneja la orquestación completa del proceso de entrenamiento desde
    la preparación de datos hasta la evaluación y comparación de modelos.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Inicializar pipeline de entrenamiento.
        
        Args:
            config: Configuración del pipeline
        """
        self.config = config or PipelineConfig()
        
        # Componentes principales
        self.db = Database()
        self.collector = YahooFinanceCollector()
        
        # Estado del pipeline
        self.jobs_queue: List[TrainingJob] = []
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: List[Dict[str, Any]] = []
        self.failed_jobs: List[Dict[str, Any]] = []
        
        # Configurar logging
        self._setup_logging()
        
        # Crear directorio de resultados
        self._setup_directories()
        
        logger.info("Pipeline de entrenamiento inicializado")
    
    def _setup_logging(self):
        """Configurar sistema de logging."""
        log_level = getattr(logging, self.config.log_level.upper())
        logger.setLevel(log_level)
        
        if self.config.save_logs:
            # Asegurar que el directorio de logs existe
            log_dir = Path(self.config.results_dir) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Crear handler para archivo
            log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
    
    def _setup_directories(self):
        """Crear directorios necesarios."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        
        # Subdirectorios para organizar resultados
        (results_dir / 'models').mkdir(exist_ok=True)
        (results_dir / 'reports').mkdir(exist_ok=True)
        (results_dir / 'logs').mkdir(exist_ok=True)
    
    def add_training_job(self, ticker: str, model_type: str = 'basic', priority: int = 1) -> str:
        """
        Añadir trabajo de entrenamiento a la cola.
        
        Args:
            ticker: Símbolo de la acción
            model_type: Tipo de modelo a entrenar
            priority: Prioridad del trabajo (1=alta, 5=baja)
            
        Returns:
            ID del trabajo
        """
        job = TrainingJob(
            ticker=ticker,
            model_type=model_type,
            priority=priority
        )
        
        self.jobs_queue.append(job)
        job_id = f"{ticker}_{model_type}_{job.created_at.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Trabajo añadido: {job_id} (prioridad: {priority})")
        return job_id
    
    def add_multiple_jobs(self, tickers: List[str], model_types: List[str] = None) -> List[str]:
        """
        Añadir múltiples trabajos de entrenamiento.
        
        Args:
            tickers: Lista de símbolos de acciones
            model_types: Lista de tipos de modelo (default: config.model_types)
            
        Returns:
            Lista de IDs de trabajos
        """
        if model_types is None:
            model_types = self.config.model_types
        
        job_ids = []
        for ticker in tickers:
            for model_type in model_types:
                job_id = self.add_training_job(ticker, model_type)
                job_ids.append(job_id)
        
        logger.info(f"{len(job_ids)} trabajos añadidos para {len(tickers)} tickers")
        return job_ids
    
    def validate_ticker_data(self, ticker: str) -> Tuple[bool, str]:
        """
        Validar que un ticker tiene datos suficientes para entrenamiento.
        
        Args:
            ticker: Símbolo de la acción
            
        Returns:
            Tuple[bool, str]: (es_válido, mensaje)
        """
        try:
            # Verificar si el ticker existe en la base de datos
            with self.db.get_session() as session:
                stock = session.query(Stock).filter(Stock.ticker == ticker).first()
                
                if not stock:
                    return False, f"Ticker {ticker} no encontrado en base de datos"
                
                # Verificar cantidad de datos históricos
                price_count = session.query(HistoricalData).filter(
                    HistoricalData.stock_id == stock.id
                ).count()
                
                if price_count < self.config.min_data_days:
                    return False, f"Datos insuficientes: {price_count} < {self.config.min_data_days}"
                
                # Verificar datos recientes
                latest_price = session.query(HistoricalData).filter(
                    HistoricalData.stock_id == stock.id
                ).order_by(HistoricalData.date.desc()).first()
                
                if latest_price:
                    days_old = (datetime.now().date() - latest_price.date).days
                    if days_old > 7:  # Más de una semana sin datos
                        return False, f"Datos desactualizados: {days_old} días"
                
                return True, f"Ticker {ticker} válido con {price_count} registros"
                
        except Exception as e:
            return False, f"Error validando {ticker}: {e}"
    
    def compare_models(self, ticker: str = None) -> pd.DataFrame:
        """
        Comparar modelos entrenados por métricas.
        
        Args:
            ticker: Filtrar por ticker específico (opcional)
            
        Returns:
            DataFrame con comparación de modelos
        """
        try:
            with self.db.get_session() as session:
                query = session.query(TrainedModel)
                
                if ticker:
                    stock = session.query(Stock).filter(Stock.ticker == ticker).first()
                    if stock:
                        query = query.filter(TrainedModel.stock_id == stock.id)
                
                models = query.filter(TrainedModel.is_active == True).all()
                
                # Crear DataFrame para comparación
                data = []
                for model in models:
                    stock = session.query(Stock).filter(Stock.id == model.stock_id).first()
                    data.append({
                        'ticker': stock.ticker if stock else 'Unknown',
                        'model_type': model.model_type,
                        'version': model.version,
                        'rmse': model.rmse,
                        'mae': model.mae,
                        'mape': model.mape,
                        'training_date': model.training_date,
                        'training_samples': model.training_samples,
                        'epochs_trained': model.epochs_trained,
                        'training_time': model.training_time_seconds
                    })
                
                df = pd.DataFrame(data)
                
                if not df.empty:
                    # Ordenar por RMSE (menor es mejor)
                    df = df.sort_values('rmse')
                
                return df
                
        except Exception as e:
            logger.error(f"Error comparando modelos: {e}")
            return pd.DataFrame()
    
    def generate_training_report(self, results: PipelineResults) -> str:
        """
        Generar reporte detallado del entrenamiento.
        
        Args:
            results: Resultados del pipeline
            
        Returns:
            Ruta del archivo de reporte generado
        """
        # Asegurar que el directorio de reportes existe
        reports_dir = Path(self.config.results_dir) / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Entrenamiento - Pipeline GuruInversor\n\n")
            f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumen ejecutivo
            f.write("## Resumen Ejecutivo\n\n")
            f.write(f"- **Total de trabajos:** {results.total_jobs}\n")
            f.write(f"- **Completados exitosamente:** {results.completed_jobs}\n")
            f.write(f"- **Fallidos:** {results.failed_jobs}\n")
            f.write(f"- **Tasa de éxito:** {results.success_rate:.1%}\n")
            f.write(f"- **Duración total:** {results.duration:.2f} segundos\n\n")
            
            # Configuración utilizada
            f.write("## Configuración del Pipeline\n\n")
            f.write("```json\n")
            f.write(json.dumps(asdict(self.config), indent=2, default=str))
            f.write("\n```\n")
        
        logger.info(f"Reporte generado: {report_path}")
        return str(report_path)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del pipeline.
        
        Returns:
            Estado del pipeline
        """
        return {
            'jobs_in_queue': len(self.jobs_queue),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'config': asdict(self.config)
        }


# Funciones de conveniencia
def create_training_pipeline(tickers: List[str], 
                           model_types: List[str] = None,
                           config: PipelineConfig = None) -> Tuple[TrainingPipeline, PipelineResults]:
    """
    Función de conveniencia para crear y ejecutar un pipeline de entrenamiento.
    
    Args:
        tickers: Lista de símbolos de acciones
        model_types: Tipos de modelo a entrenar
        config: Configuración del pipeline
        
    Returns:
        Tuple[TrainingPipeline, PipelineResults]: Pipeline y resultados
    """
    # Configurar pipeline
    if config is None:
        config = PipelineConfig(model_types=model_types or ['basic'])
    
    # Crear pipeline
    pipeline = TrainingPipeline(config)
    
    return pipeline, None


if __name__ == "__main__":
    # Ejemplo de uso del pipeline
    print("MODEL-003 - Pipeline de Entrenamiento - GuruInversor")
    print("=" * 60)
    
    # Configuración de ejemplo
    config = PipelineConfig(
        max_workers=2,
        model_types=['basic'],  # Solo básico para demo
        min_data_days=100,  # Reducido para demo
        update_data=False  # No actualizar datos en demo
    )
    
    try:
        logger.info("Configurando pipeline de entrenamiento...")
        pipeline = TrainingPipeline(config)
        
        print("\nPipeline MODEL-003 inicializado correctamente")
        print("Listo para automatizar entrenamiento de modelos LSTM")
        
    except Exception as e:
        logger.error(f"Error en pipeline: {e}")
        print("Error en MODEL-003 - revisar configuración")
    
    print("=" * 60)
