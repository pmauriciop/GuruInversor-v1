#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo ML - Machine Learning y Procesamiento de Datos

Este módulo contiene todas las utilidades para procesamiento de datos,
feature engineering, métricas de evaluación e integración de datos.

Componentes principales:
- preprocessor: Procesamiento y normalización de datos
- metrics: Evaluación de modelos y métricas financieras  
- data_integration: Integración con base de datos y recolector de datos
"""

from .preprocessor import (
    DataProcessor,
    ProcessingConfig,
    process_stock_data
)

from .metrics import (
    calculate_regression_metrics,
    calculate_directional_accuracy,
    calculate_financial_metrics,
    evaluate_model_performance,
    print_evaluation_report,
    validate_trading_strategy
)

# MODEL-004: Sistema de evaluación avanzado
from .model_evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    create_model_evaluator
)

# Integración de datos
from .data_integration import (
    DataIntegrator,
    initialize_sample_data
)

__all__ = [
    # Procesamiento de datos
    'DataProcessor',
    'ProcessingConfig', 
    'process_stock_data',
      # Métricas y evaluación
    'calculate_regression_metrics',
    'calculate_directional_accuracy',
    'calculate_financial_metrics',
    'evaluate_model_performance',
    'print_evaluation_report',
    'validate_trading_strategy',
    
    # MODEL-004: Evaluación avanzada
    'ModelEvaluator',
    'EvaluationConfig',
    'create_model_evaluator',
    
    # Integración de datos (comentado temporalmente)
    # 'DataIntegrator',
    # 'initialize_sample_data'
]
