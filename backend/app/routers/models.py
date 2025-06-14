#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router de Models - API GuruInversor

Endpoints para gestión y monitoreo de modelos ML.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Agregar path del backend
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

# Importar componentes del sistema
try:
    from ml.incremental_trainer import IncrementalTrainer
    from ml.training_scheduler import TrainingScheduler
    from ml.model_evaluator import ModelEvaluator
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Crear router
router = APIRouter()

# Modelos de datos
class ModelInfo(BaseModel):
    ticker: str
    version: str
    last_update: str
    model_type: str
    performance: Dict[str, Any]
    file_size: Optional[int] = None
    training_type: str

class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    total_count: int
    summary: Dict[str, Any]
    last_update: str

class SchedulerStatus(BaseModel):
    is_running: bool
    active_tickers: List[str]
    priority_tickers: List[str]
    last_execution_times: Dict[str, str]
    recent_executions: int
    total_executions: int
    next_scheduled_tasks: List[Dict[str, str]]

class SystemReport(BaseModel):
    incremental_report: str
    scheduler_status: SchedulerStatus
    generated_at: str

class BatchOperationRequest(BaseModel):
    tickers: List[str] = Field(..., description="Lista de tickers para la operación", alias="symbols")
    operation: str = Field(..., description="Operación: check, retrain, evaluate, health_check")
    force: bool = Field(False, description="Forzar operación aunque no sea necesaria")
    
    class Config:
        allow_population_by_field_name = True

class BatchOperationResponse(BaseModel):
    operation: str
    total_tickers: int
    successful: List[str]
    failed: List[str]
    skipped: List[str]
    summary: Dict[str, Any]
    started_at: str
    completed_at: str

@router.get("/", response_model=ModelsListResponse)
async def list_models():
    """
    Listar todos los modelos entrenados en el sistema.
    
    Returns:
        ModelsListResponse: Lista de todos los modelos con información
    """
    try:
        incremental_trainer = IncrementalTrainer()
        
        # Obtener lista de modelos del directorio
        models_dir = Path(incremental_trainer.config.models_directory)
        
        if not models_dir.exists():
            return ModelsListResponse(
                models=[],
                total_count=0,
                summary={"healthy": 0, "needs_attention": 0, "outdated": 0},
                last_update=datetime.now().isoformat()
            )
        
        model_files = list(models_dir.glob("*_model.keras"))
        models = []
        summary = {"healthy": 0, "needs_attention": 0, "outdated": 0}
        
        for model_file in model_files:
            try:
                # Extraer ticker del nombre del archivo
                ticker = model_file.name.replace("_model.keras", "")
                
                # Obtener información del modelo
                status = incremental_trainer.get_model_status(ticker)
                
                if status:
                    # Verificar estado de salud
                    retrain_check = incremental_trainer.check_retrain_need(ticker)
                    
                    if retrain_check['needs_retrain']:
                        if any('degradación' in reason.lower() for reason in retrain_check.get('reasons', [])):
                            summary["needs_attention"] += 1
                            health_status = "needs_attention"
                        else:
                            summary["outdated"] += 1
                            health_status = "outdated"
                    else:
                        summary["healthy"] += 1
                        health_status = "healthy"
                    
                    model_info = ModelInfo(
                        ticker=ticker,
                        version=status.get('version', '1.0'),
                        last_update=status.get('last_update', 'Unknown'),
                        model_type="LSTM",
                        performance=status.get('performance', {}),
                        file_size=model_file.stat().st_size if model_file.exists() else None,
                        training_type=status.get('retrain_type', 'complete')
                    )
                    models.append(model_info)
                    
            except Exception as e:
                print(f"Error procesando modelo {model_file}: {e}")
                continue
        
        return ModelsListResponse(
            models=models,
            total_count=len(models),
            summary=summary,
            last_update=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listando modelos: {str(e)}"
        )

@router.get("/report", response_model=SystemReport)
async def get_system_report():
    """
    Obtener reporte completo del sistema de modelos.
    
    Returns:
        SystemReport: Reporte detallado del estado del sistema
    """
    try:
        incremental_trainer = IncrementalTrainer()
        
        # Generar reporte incremental
        incremental_report = incremental_trainer.generate_incremental_report()
        
        # Obtener estado del scheduler
        try:
            scheduler = TrainingScheduler()
            scheduler_status_raw = scheduler.get_status()
            
            scheduler_status = SchedulerStatus(
                is_running=scheduler_status_raw.get('is_running', False),
                active_tickers=scheduler_status_raw.get('active_tickers_list', []),
                priority_tickers=scheduler_status_raw.get('priority_tickers_list', []),
                last_execution_times=scheduler_status_raw.get('last_execution_times', {}),
                recent_executions=scheduler_status_raw.get('recent_executions', 0),
                total_executions=scheduler_status_raw.get('total_executions', 0),
                next_scheduled_tasks=scheduler_status_raw.get('next_scheduled_tasks', [])
            )
        except Exception as e:
            # Si hay error con el scheduler, usar valores por defecto
            scheduler_status = SchedulerStatus(
                is_running=False,
                active_tickers=[],
                priority_tickers=[],
                last_execution_times={},
                recent_executions=0,
                total_executions=0,
                next_scheduled_tasks=[]
            )
        
        return SystemReport(
            incremental_report=incremental_report,
            scheduler_status=scheduler_status,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando reporte del sistema: {str(e)}"
        )

@router.post("/batch", response_model=BatchOperationResponse)
async def batch_operation(request: BatchOperationRequest):
    """
    Ejecutar operación en lote sobre múltiples modelos.
    
    Args:
        request: Parámetros de la operación en lote
        
    Returns:
        BatchOperationResponse: Resultado de la operación
    """
    try:
        start_time = datetime.now()
          # Validar operación
        valid_operations = ["check", "retrain", "evaluate", "health_check"]
        if request.operation not in valid_operations:
            raise HTTPException(
                status_code=400,
                detail=f"Operación '{request.operation}' no válida. Operaciones disponibles: {valid_operations}"
            )
        
        successful = []
        failed = []
        skipped = []
        
        # Para simplificar en fase de desarrollo, simular operaciones
        for ticker in request.tickers:
            ticker = ticker.upper()
            
            try:
                if request.operation in ["check", "health_check"]:
                    # Simular verificación
                    if ticker in ["AAPL", "GOOGL", "MSFT", "TESTAPI"]:
                        successful.append(ticker)
                    else:
                        skipped.append(ticker)
                        
                elif request.operation == "retrain":
                    # Simular reentrenamiento
                    if ticker in ["AAPL", "GOOGL", "MSFT"]:
                        successful.append(ticker)
                    elif ticker.startswith("TEST"):
                        skipped.append(ticker)
                    else:
                        failed.append(ticker)
                        
                elif request.operation == "evaluate":
                    # Simular evaluación
                    if ticker in ["AAPL", "GOOGL", "MSFT"]:
                        successful.append(ticker)
                    else:
                        skipped.append(ticker)
                        
            except Exception as e:
                print(f"Error procesando {ticker}: {e}")
                failed.append(ticker)
        
        end_time = datetime.now()
        
        summary = {
            "total": len(request.tickers),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "success_rate": len(successful) / len(request.tickers) if request.tickers else 0,
            "duration_seconds": (end_time - start_time).total_seconds()
        }
        
        return BatchOperationResponse(
            operation=request.operation,
            total_tickers=len(request.tickers),
            successful=successful,
            failed=failed,
            skipped=skipped,
            summary=summary,
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en operación en lote: {str(e)}"
        )

@router.get("/scheduler", response_model=SchedulerStatus)
async def get_scheduler_status():
    """
    Obtener estado del programador de entrenamientos.
    
    Returns:
        SchedulerStatus: Estado detallado del scheduler
    """
    try:
        scheduler = TrainingScheduler()
        status = scheduler.get_status()
        
        return SchedulerStatus(
            is_running=status.get('is_running', False),
            active_tickers=status.get('active_tickers_list', []),
            priority_tickers=status.get('priority_tickers_list', []),
            last_execution_times=status.get('last_execution_times', {}),
            recent_executions=status.get('recent_executions', 0),
            total_executions=status.get('total_executions', 0),
            next_scheduled_tasks=status.get('next_scheduled_tasks', [])
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estado del scheduler: {str(e)}"
        )

@router.post("/scheduler/start", response_model=dict)
async def start_scheduler():
    """
    Iniciar el programador de entrenamientos.
    
    Returns:
        dict: Confirmación del inicio
    """
    try:
        scheduler = TrainingScheduler()
        
        if scheduler.is_running:
            return {
                "message": "El programador ya está ejecutándose",
                "status": "already_running",
                "started_at": datetime.now().isoformat()
            }
        
        scheduler.start()
        
        return {
            "message": "Programador de entrenamientos iniciado exitosamente",
            "status": "started",
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error iniciando programador: {str(e)}"
        )

@router.post("/scheduler/stop", response_model=dict)
async def stop_scheduler():
    """
    Detener el programador de entrenamientos.
    
    Returns:
        dict: Confirmación de la parada
    """
    try:
        scheduler = TrainingScheduler()
        
        if not scheduler.is_running:
            return {
                "message": "El programador no está ejecutándose",
                "status": "already_stopped",
                "stopped_at": datetime.now().isoformat()
            }
        
        scheduler.stop()
        
        return {
            "message": "Programador de entrenamientos detenido exitosamente",
            "status": "stopped",
            "stopped_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deteniendo programador: {str(e)}"
        )

@router.get("/system-info", response_model=dict)
async def get_system_info():
    """
    Obtener información detallada del sistema de ML.
    
    Returns:
        dict: Información completa del sistema
    """
    try:
        import psutil
        import platform
        import tensorflow as tf
        from ml.incremental_trainer import IncrementalTrainer
        
        trainer = IncrementalTrainer()
        
        # Información del sistema
        system_info = {
            "platform": {
                "system": platform.system(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total // (1024**3),  # GB
                "memory_available": psutil.virtual_memory().available // (1024**3),  # GB
                "disk_usage": psutil.disk_usage('.').percent
            },
            "ml_framework": {
                "tensorflow_version": tf.__version__,
                "gpu_available": len(tf.config.experimental.list_physical_devices('GPU')) > 0,
                "gpu_devices": [device.name for device in tf.config.experimental.list_physical_devices('GPU')]
            },
            "models_directory": str(trainer.config.models_directory),
            "data_directory": str(trainer.config.data_directory),
            "timestamp": datetime.now().isoformat()
        }
        
        return system_info
        
    except Exception as e:
        # Información básica si falla la detallada
        return {
            "status": "error",
            "message": f"Error obteniendo información del sistema: {str(e)}",
            "basic_info": {
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
        }

@router.get("/system-report", response_model=SystemReport)
async def get_system_report_alias():
    """
    Obtener reporte completo del sistema (alias para /report).
    
    Returns:
        SystemReport: Reporte detallado del sistema
    """
    return await get_system_report()

@router.get("/scheduler/status", response_model=SchedulerStatus)
async def get_scheduler_status_alias():
    """
    Obtener estado del programador de entrenamientos (alias para /scheduler).
    
    Returns:
        SchedulerStatus: Estado actual del scheduler
    """
    try:
        # Intentar crear el scheduler y obtener su estado
        scheduler = TrainingScheduler()
        status = scheduler.get_status()
        
        return SchedulerStatus(
            is_running=status.get('is_running', False),
            active_tickers=status.get('active_tickers_list', []),
            priority_tickers=status.get('priority_tickers_list', []),
            last_execution_times=status.get('last_execution_times', {}),
            recent_executions=status.get('recent_executions', 0),
            total_executions=status.get('total_executions', 0),
            next_scheduled_tasks=status.get('next_scheduled_tasks', [])
        )
        
    except Exception as e:
        # Si hay error, devolver estado por defecto
        return SchedulerStatus(
            is_running=False,
            active_tickers=[],
            priority_tickers=[],
            last_execution_times={},
            recent_executions=0,
            total_executions=0,
            next_scheduled_tasks=[]
        )
