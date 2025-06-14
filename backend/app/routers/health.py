#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router de Health - API GuruInversor

Endpoints para verificar el estado y salud del sistema.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

# Agregar path del backend
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

# Importar componentes del sistema
try:
    from database.connection import get_database
    from data.collector import YahooFinanceCollector
    from ml.incremental_trainer import IncrementalTrainer
    from ml.training_scheduler import TrainingScheduler
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Crear router
router = APIRouter()

# Modelos de respuesta
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]
    summary: Dict[str, Any]

class SystemInfo(BaseModel):
    python_version: str
    platform: str
    api_version: str
    uptime: str
    version: str = "1.0.0"
    description: str = "GuruInversor API - Sistema de predicción de acciones con ML"

@router.get("/health", response_model=HealthResponse)
async def get_system_health():
    """
    Verificar el estado de salud de todos los componentes del sistema.
    
    Returns:
        HealthResponse: Estado detallado de cada componente
    """
    components = {}
    overall_status = "healthy"
    
    try:        # 1. Verificar base de datos
        try:
            database = get_database()
            with database.session_scope() as session:
                session.execute(text("SELECT 1"))
            components["database"] = {
                "status": "healthy",
                "message": "Base de datos conectada correctamente"
            }
        except Exception as e:
            components["database"] = {
                "status": "unhealthy", 
                "message": f"Error de base de datos: {str(e)}"
            }
            overall_status = "degraded"
        
        # 2. Verificar recolector de datos
        try:
            collector = YahooFinanceCollector()
            # Test básico con un ticker conocido
            test_data = collector.get_stock_info("AAPL")
            if test_data:
                components["data_collector"] = {
                    "status": "healthy",
                    "message": "Recolector de datos funcionando"
                }
            else:
                components["data_collector"] = {
                    "status": "degraded",
                    "message": "Recolector funciona pero sin datos de prueba"
                }
        except Exception as e:
            components["data_collector"] = {
                "status": "unhealthy",
                "message": f"Error en recolector: {str(e)}"
            }
            overall_status = "degraded"
        
        # 3. Verificar sistema de ML
        try:
            trainer = IncrementalTrainer()
            components["ml_system"] = {
                "status": "healthy",
                "message": "Sistema de ML inicializado",
                "models_directory": trainer.config.models_directory
            }
        except Exception as e:
            components["ml_system"] = {
                "status": "degraded",
                "message": f"Sistema ML con problemas: {str(e)}"
            }
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # 4. Verificar scheduler (opcional)
        try:
            scheduler = TrainingScheduler()
            components["scheduler"] = {
                "status": "healthy",
                "message": "Programador de entrenamientos disponible",
                "is_running": scheduler.is_running
            }
        except Exception as e:
            components["scheduler"] = {
                "status": "degraded",
                "message": f"Scheduler con problemas: {str(e)}"
            }
        
        # 5. Verificar directorios necesarios
        required_dirs = ["data", "models", "logs"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = backend_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            components["filesystem"] = {
                "status": "degraded",
                "message": f"Directorios faltantes: {', '.join(missing_dirs)}"
            }
            if overall_status == "healthy":
                overall_status = "degraded"
        else:
            components["filesystem"] = {
                "status": "healthy",
                "message": "Todos los directorios necesarios existen"
            }
        
        # Resumen
        healthy_count = sum(1 for comp in components.values() if comp["status"] == "healthy")
        total_count = len(components)
        
        summary = {
            "total_components": total_count,
            "healthy_components": healthy_count,
            "degraded_components": total_count - healthy_count,
            "health_percentage": round((healthy_count / total_count) * 100, 1) if total_count > 0 else 0
        }
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            components=components,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error verificando salud del sistema: {str(e)}"
        )

@router.get("/health/quick", response_model=dict)
async def quick_health_check():
    """
    Verificación rápida de salud (solo componentes esenciales).
    
    Returns:
        dict: Estado básico del sistema
    """
    try:
        # Solo verificar base de datos para respuesta rápida
        database = get_database()
        with database.session_scope() as session:
            session.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": "Sistema operacional"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Sistema no disponible: {str(e)}"
        )

@router.get("/info", response_model=SystemInfo)
async def get_system_info():
    """
    Obtener información general del sistema.
    
    Returns:
        SystemInfo: Información detallada del sistema
    """
    import platform
    import sys
    
    return SystemInfo(
        python_version=sys.version,
        platform=platform.platform(),
        api_version="1.0.0",
        uptime=str(datetime.now())  # En producción, usar tiempo real de uptime
    )
