#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Principal - GuruInversor

API REST básica construida con FastAPI para exponer funcionalidades
del sistema de predicción de acciones.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Agregar path del backend al PYTHONPATH
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

# Import database configuration
from database.config import get_database_url

# Importar routers y middleware
from app.routers import stocks, predictions, health, models, metrics, auth, management
from app.middleware import MetricsMiddleware

# Información de la aplicación
app_info = {
    "title": "GuruInversor API",
    "description": """
    API REST para el sistema de predicción de acciones GuruInversor.
      ## Características principales:
    
    * **Gestión de acciones**: Añadir, listar y eliminar acciones monitoreadas
    * **Predicciones avanzadas**: Predicciones con análisis de escenarios y riesgo
    * **Datos históricos**: Acceso a datos históricos de Yahoo Finance
    * **Modelos ML**: Gestión avanzada de modelos LSTM con métricas detalladas
    * **Entrenamiento**: Sistema incremental con entrenamiento en lote
    * **Monitoreo**: Métricas del sistema en tiempo real
    * **Autenticación**: Sistema JWT para acceso seguro
    * **Análisis avanzado**: Comparación de modelos y evaluación de riesgo
    
    ## Arquitectura:
    
    - **Base de datos**: SQLite para almacenamiento persistente
    - **Modelos ML**: LSTM con TensorFlow para predicciones
    - **Datos**: Yahoo Finance como fuente principal
    - **Entrenamiento**: Sistema incremental automático
    """,
    "version": "1.0.0",
    "contact": {
        "name": "GuruInversor API",
        "url": "https://github.com/tu-usuario/GuruInversor",
    },
    "license_info": {
        "name": "MIT",
    },
}

# Crear instancia de FastAPI
app = FastAPI(**app_info)

# Configurar middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Añadir middleware de métricas
app.add_middleware(MetricsMiddleware)

# Incluir routers
app.include_router(health.router, prefix="/api", tags=["System Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["System Metrics"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["Stocks"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(models.router, prefix="/api/models", tags=["ML Models"])
app.include_router(management.router, prefix="/api/management", tags=["Advanced Management"])

# Endpoint raíz
@app.get("/", response_model=dict)
async def root():
    """Endpoint raíz con información básica de la API."""
    return {
        "message": "Bienvenido a GuruInversor API",
        "version": app_info["version"],
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",        "api_info": {
            "authentication": "/api/auth",
            "metrics": "/api/metrics", 
            "stocks_endpoint": "/api/stocks",
            "predictions_endpoint": "/api/predictions", 
            "models_endpoint": "/api/models",
            "management": "/api/management",
            "health_endpoint": "/api/health"
        }
    }

# Manejador de errores global
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejador personalizado para excepciones HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejador para excepciones generales."""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Error interno del servidor",
            "detail": str(exc),
            "status_code": 500
        }
    )

# Función para ejecutar el servidor
def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Ejecutar el servidor de desarrollo."""
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    run_server()
