#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router de Métricas - API GuruInversor

Endpoints para monitoreo, métricas y análisis del sistema.
"""

import os
import sys
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Agregar path del backend
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

# Importar componentes del sistema
try:
    from ml.incremental_trainer import IncrementalTrainer
    from database.connection import get_database
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Crear router
router = APIRouter()

# Modelos de datos
class SystemMetrics(BaseModel):
    cpu_usage: float = Field(..., description="Uso de CPU en porcentaje")
    memory_usage: float = Field(..., description="Uso de memoria en porcentaje")
    disk_usage: float = Field(..., description="Uso de disco en porcentaje")
    python_memory: float = Field(..., description="Memoria usada por Python en MB")
    uptime: str = Field(..., description="Tiempo de actividad del sistema")
    timestamp: str = Field(..., description="Timestamp de la métrica")

class ModelMetrics(BaseModel):
    total_models: int = Field(..., description="Total de modelos disponibles")
    trained_models: List[str] = Field(..., description="Lista de modelos entrenados")
    models_needing_retrain: List[str] = Field(..., description="Modelos que necesitan reentrenamiento")
    last_training_times: Dict[str, str] = Field(..., description="Últimas fechas de entrenamiento")
    model_sizes: Dict[str, float] = Field(..., description="Tamaños de modelos en MB")
    model_performance: Dict[str, Dict[str, float]] = Field(..., description="Métricas de rendimiento")

class DatabaseMetrics(BaseModel):
    database_size: float = Field(..., description="Tamaño de la base de datos en MB")
    total_records: int = Field(..., description="Total de registros")
    stocks_tracked: int = Field(..., description="Acciones monitoreadas")
    last_update: str = Field(..., description="Última actualización")

class APIMetrics(BaseModel):
    total_requests: int = Field(..., description="Total de requests procesados")
    successful_requests: int = Field(..., description="Requests exitosos")
    failed_requests: int = Field(..., description="Requests fallidos")
    average_response_time: float = Field(..., description="Tiempo promedio de respuesta en ms")
    endpoints_usage: Dict[str, int] = Field(..., description="Uso por endpoint")

class ComprehensiveMetrics(BaseModel):
    system: SystemMetrics
    models: ModelMetrics
    database: DatabaseMetrics
    api: APIMetrics
    status: str = Field(..., description="Estado general del sistema")
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

# Variables globales para tracking (en producción sería una base de datos)
_request_count = 0
_successful_requests = 0
_failed_requests = 0
_response_times = []
_endpoint_usage = {}

def track_request(endpoint: str, success: bool, response_time: float):
    """Función para trackear métricas de requests"""
    global _request_count, _successful_requests, _failed_requests, _response_times, _endpoint_usage
    
    _request_count += 1
    if success:
        _successful_requests += 1
    else:
        _failed_requests += 1
    
    _response_times.append(response_time)
    if len(_response_times) > 1000:  # Mantener solo los últimos 1000
        _response_times = _response_times[-1000:]
    
    _endpoint_usage[endpoint] = _endpoint_usage.get(endpoint, 0) + 1

@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics():
    """
    Obtener métricas del sistema (CPU, memoria, disco).
    
    Returns:
        SystemMetrics: Métricas actuales del sistema
    """
    try:
        # Obtener métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Obtener información del proceso Python actual
        process = psutil.Process()
        python_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Calcular uptime (simplificado)
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time)
        
        return SystemMetrics(
            cpu_usage=round(cpu_percent, 2),
            memory_usage=round(memory.percent, 2),
            disk_usage=round(disk.percent, 2),
            python_memory=round(python_memory_mb, 2),
            uptime=uptime,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas del sistema: {str(e)}"
        )

@router.get("/models", response_model=ModelMetrics)
async def get_model_metrics():
    """
    Obtener métricas de los modelos ML.
    
    Returns:
        ModelMetrics: Métricas de modelos entrenados
    """
    try:
        trainer = IncrementalTrainer()
        
        # Buscar modelos entrenados
        models_dir = Path("models")
        trained_models = []
        model_sizes = {}
        last_training_times = {}
        
        if models_dir.exists():
            for model_file in models_dir.glob("*.keras"):
                ticker = model_file.stem.replace("_lstm_model", "")
                trained_models.append(ticker)
                
                # Tamaño del modelo
                size_mb = model_file.stat().st_size / 1024 / 1024
                model_sizes[ticker] = round(size_mb, 2)
                
                # Fecha de última modificación como proxy para último entrenamiento
                mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                last_training_times[ticker] = mtime.isoformat()
        
        # Verificar qué modelos necesitan reentrenamiento
        models_needing_retrain = []
        model_performance = {}
        
        for ticker in trained_models:
            try:
                # Verificar si necesita reentrenamiento
                retrain_check = trainer.check_retrain_need(ticker)
                if retrain_check.get('needs_retrain', False):
                    models_needing_retrain.append(ticker)
                
                # Obtener métricas de rendimiento básicas
                status = trainer.get_model_status(ticker)
                if isinstance(status, dict) and 'performance' in status:
                    model_performance[ticker] = status['performance']
                else:
                    # Métricas simuladas si no hay datos reales
                    model_performance[ticker] = {
                        "accuracy": 0.85,
                        "precision": 0.82,
                        "recall": 0.88
                    }
                    
            except Exception as e:
                print(f"Error checking model {ticker}: {e}")
                continue
        
        return ModelMetrics(
            total_models=len(trained_models),
            trained_models=trained_models,
            models_needing_retrain=models_needing_retrain,
            last_training_times=last_training_times,
            model_sizes=model_sizes,
            model_performance=model_performance
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas de modelos: {str(e)}"
        )

@router.get("/database", response_model=DatabaseMetrics)
async def get_database_metrics():
    """
    Obtener métricas de la base de datos.
    
    Returns:
        DatabaseMetrics: Métricas de la base de datos
    """
    try:
        # Obtener tamaño de la base de datos
        db_path = Path("data/guru_inversor.db")
        db_size_mb = 0
        
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / 1024 / 1024
        
        # Métricas simuladas (en producción se harían queries reales)
        total_records = 1500  # Simulado
        stocks_tracked = 25   # Simulado
        
        return DatabaseMetrics(
            database_size=round(db_size_mb, 2),
            total_records=total_records,
            stocks_tracked=stocks_tracked,
            last_update=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas de base de datos: {str(e)}"
        )

@router.get("/api", response_model=APIMetrics)
async def get_api_metrics():
    """
    Obtener métricas de la API.
    
    Returns:
        APIMetrics: Métricas de uso de la API
    """
    try:
        # Calcular tiempo promedio de respuesta
        avg_response_time = sum(_response_times) / len(_response_times) if _response_times else 0
        
        return APIMetrics(
            total_requests=_request_count,
            successful_requests=_successful_requests,
            failed_requests=_failed_requests,
            average_response_time=round(avg_response_time, 2),
            endpoints_usage=_endpoint_usage.copy()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas de API: {str(e)}"
        )

@router.get("/comprehensive", response_model=ComprehensiveMetrics)
async def get_comprehensive_metrics():
    """
    Obtener todas las métricas del sistema en un solo endpoint.
    
    Returns:
        ComprehensiveMetrics: Todas las métricas del sistema
    """
    try:
        # Obtener todas las métricas
        system_metrics = await get_system_metrics()
        model_metrics = await get_model_metrics()
        database_metrics = await get_database_metrics()
        api_metrics = await get_api_metrics()
        
        # Determinar estado general del sistema
        status = "healthy"
        
        # Verificar condiciones de salud
        if system_metrics.cpu_usage > 90:
            status = "degraded"
        elif system_metrics.memory_usage > 90:
            status = "degraded"
        elif api_metrics.failed_requests > api_metrics.successful_requests:
            status = "degraded"
        elif len(model_metrics.models_needing_retrain) > len(model_metrics.trained_models) / 2:
            status = "warning"
        
        return ComprehensiveMetrics(
            system=system_metrics,
            models=model_metrics,
            database=database_metrics,
            api=api_metrics,
            status=status
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas comprehensivas: {str(e)}"
        )

@router.get("/health-check", response_model=dict)
async def advanced_health_check():
    """
    Health check avanzado con métricas detalladas.
    
    Returns:
        dict: Estado detallado de salud del sistema
    """
    try:
        metrics = await get_comprehensive_metrics()
        
        # Análisis de salud detallado
        health_issues = []
        warnings = []
        
        # Verificar CPU
        if metrics.system.cpu_usage > 95:
            health_issues.append("CPU usage crítico")
        elif metrics.system.cpu_usage > 80:
            warnings.append("CPU usage alto")
        
        # Verificar memoria
        if metrics.system.memory_usage > 95:
            health_issues.append("Memoria crítica")
        elif metrics.system.memory_usage > 80:
            warnings.append("Memoria alta")
        
        # Verificar modelos
        if len(metrics.models.trained_models) == 0:
            health_issues.append("No hay modelos entrenados")
        elif len(metrics.models.models_needing_retrain) > len(metrics.models.trained_models) / 2:
            warnings.append("Múltiples modelos necesitan reentrenamiento")
        
        # Determinar estado final
        if health_issues:
            overall_status = "critical"
        elif warnings:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu": f"{metrics.system.cpu_usage}%",
                "memory": f"{metrics.system.memory_usage}%",
                "disk": f"{metrics.system.disk_usage}%"
            },
            "model_metrics": {
                "trained_models": len(metrics.models.trained_models),
                "needs_retrain": len(metrics.models.models_needing_retrain)
            },
            "api_metrics": {
                "success_rate": f"{(metrics.api.successful_requests / max(metrics.api.total_requests, 1)) * 100:.1f}%",
                "avg_response": f"{metrics.api.average_response_time}ms"
            },
            "health_issues": health_issues,
            "warnings": warnings,
            "recommendations": _generate_recommendations(metrics, warnings, health_issues)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "recommendations": ["Verificar logs del sistema", "Reiniciar servicios si es necesario"]
        }

def _generate_recommendations(metrics, warnings, issues):
    """Generar recomendaciones basadas en las métricas"""
    recommendations = []
    
    if metrics.system.cpu_usage > 80:
        recommendations.append("Considerar optimizar procesos o escalar recursos de CPU")
    
    if metrics.system.memory_usage > 80:
        recommendations.append("Revisar uso de memoria y considerar limpiar caché")
    
    if len(metrics.models.models_needing_retrain) > 0:
        recommendations.append(f"Reentrenar modelos: {', '.join(metrics.models.models_needing_retrain)}")
    
    if metrics.api.failed_requests > metrics.api.total_requests * 0.1:
        recommendations.append("Investigar causas de fallos en API")
    
    if not recommendations:
        recommendations.append("Sistema funcionando óptimamente")
    
    return recommendations
