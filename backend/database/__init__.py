# -*- coding: utf-8 -*-
"""
Backend Database Package - GuruInversor

Este paquete contiene toda la lógica de base de datos del proyecto.
Incluye modelos, configuración de conexión y operaciones CRUD.
"""

from .models import Stock, HistoricalData, Prediction, TrainedModel
from .connection import Database, get_database
from .crud import StockCRUD, HistoricalDataCRUD, PredictionCRUD, ModelCRUD

__all__ = [
    'Stock',
    'HistoricalData', 
    'Prediction',
    'TrainedModel',
    'Database',
    'get_database',
    'StockCRUD',
    'HistoricalDataCRUD',
    'PredictionCRUD',
    'ModelCRUD'
]

__version__ = "1.0.0"
__author__ = "GuruInversor Team"
