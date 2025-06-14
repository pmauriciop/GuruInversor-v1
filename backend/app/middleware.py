#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Middleware para tracking de métricas - API GuruInversor

Middleware para capturar métricas de requests automáticamente.
"""

import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.routers.metrics import track_request

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware para trackear métricas de requests automáticamente"""
    
    async def dispatch(self, request: Request, call_next):
        # Obtener información de la request
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        # Procesar request
        try:
            response = await call_next(request)
            success = response.status_code < 400
            
            # Calcular tiempo de respuesta
            response_time = (time.time() - start_time) * 1000  # en milisegundos
            
            # Trackear métricas
            track_request(endpoint, success, response_time)
            
            # Añadir headers de métricas
            response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
            
            return response
            
        except Exception as e:
            # En caso de error, también trackear
            response_time = (time.time() - start_time) * 1000
            track_request(endpoint, False, response_time)
            raise e
