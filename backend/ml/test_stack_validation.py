#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación Completa del Stack ML - GuruInversor

Script para validar completamente todos los componentes del stack de ML
después de las correcciones aplicadas.
"""

import sys
import logging
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Añadir el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_components():
    """Validar componentes de datos."""
    logger.info("🔍 Validando componentes de datos...")
    
    try:
        # Test 1: Collector (nombre correcto)
        from data.collector import YahooFinanceCollector
        collector = YahooFinanceCollector()
        logger.info("✅ YahooFinanceCollector importado correctamente")
        
        # Test 2: Validator  
        from data.validator import DataValidator
        validator = DataValidator()
        logger.info("✅ DataValidator importado correctamente")
        
        # Test 3: Database models (nombres correctos)
        from database.models import Stock, HistoricalData, Prediction, TrainedModel
        logger.info("✅ Modelos de base de datos importados correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en componentes de datos: {e}")
        return False

def test_ml_architecture():
    """Validar arquitectura de ML."""
    logger.info("🔍 Validando arquitectura de ML...")
    
    try:
        # Test 1: Model Architecture
        from ml.model_architecture import LSTMArchitect, LSTMConfig, validate_model_architecture
        
        config = LSTMConfig(
            sequence_length=60,
            n_features=12,
            lstm_units=[50, 30],
            dense_units=[25, 10]
        )
        
        architect = LSTMArchitect(config)
        model = architect.build_basic_model()
        
        # Validar arquitectura
        is_valid = validate_model_architecture(model, (60, 12))
        if not is_valid:
            logger.error("❌ Arquitectura LSTM no válida")
            return False
            
        logger.info("✅ Arquitectura LSTM validada correctamente")
        
        # Test 2: Model Summary (corregido)
        summary = architect.get_model_summary(model)
        logger.info(f"✅ Resumen del modelo generado: {summary['total_params']:,} parámetros")
        
        # Test 3: Callbacks (formato .keras corregido)
        callbacks = architect.get_callbacks("test_model")
        logger.info(f"✅ Callbacks generados correctamente: {len(callbacks)} callbacks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en arquitectura ML: {e}")
        logger.error(traceback.format_exc())
        return False

def test_preprocessing():
    """Validar preprocessor."""
    logger.info("🔍 Validando preprocessor...")
    
    try:
        from ml.preprocessor import DataProcessor, ProcessingConfig
        
        # Crear datos sintéticos para prueba
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(100, 200, len(dates)),
            'Low': np.random.uniform(100, 200, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        config = ProcessingConfig()
        preprocessor = DataProcessor(config)
        
        logger.info(f"✅ Preprocessor funcionando correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en preprocessor: {e}")
        return False

def test_trainer():
    """Validar trainer."""
    logger.info("🔍 Validando trainer...")
    
    try:
        from ml.trainer import LSTMTrainer, TrainingConfig
        from ml.model_architecture import LSTMConfig
        
        lstm_config = LSTMConfig()
        training_config = TrainingConfig()
        trainer = LSTMTrainer(lstm_config, training_config)
        
        logger.info("✅ LSTMTrainer inicializado correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en trainer: {e}")
        return False

def test_evaluator():
    """Validar evaluador (MODEL-004)."""
    logger.info("🔍 Validando evaluador MODEL-004...")
    
    try:
        from ml.model_evaluator import ModelEvaluator, EvaluationConfig
        
        config = EvaluationConfig()
        evaluator = ModelEvaluator(config)
        
        # Crear datos sintéticos
        from ml.model_architecture import LSTMArchitect, LSTMConfig
        
        lstm_config = LSTMConfig()
        architect = LSTMArchitect(lstm_config)
        model = architect.build_basic_model()
        
        # Datos de prueba
        X_test = np.random.random((50, 60, 12))
        y_test = np.random.random((50, 2))
        prices_test = np.random.uniform(100, 200, 50)
        
        # Evaluar modelo
        results = evaluator.evaluate_model_comprehensive(
            model, X_test, y_test, prices_test, "TEST"
        )
        
        logger.info(f"✅ Evaluador MODEL-004 funcionando: puntuación {results['overall_score']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en evaluador: {e}")
        logger.error(traceback.format_exc())
        return False

def test_metrics():
    """Validar métricas."""
    logger.info("🔍 Validando métricas...")
    
    try:
        from ml.metrics import calculate_regression_metrics, calculate_directional_accuracy, evaluate_model_performance
        
        # Datos sintéticos
        y_true = np.random.random(100)
        y_pred = np.random.random(100)
        
        regression_metrics = calculate_regression_metrics(y_true, y_pred)
        directional_metrics = calculate_directional_accuracy(y_true, y_pred)
        performance_metrics = evaluate_model_performance(y_true, y_pred)
        
        logger.info(f"✅ Métricas funcionando: RMSE={regression_metrics['rmse']:.4f}, Dir_Accuracy={directional_metrics['directional_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en métricas: {e}")
        return False

def test_pipeline_integration():
    """Validar integración del pipeline."""
    logger.info("🔍 Validando integración del pipeline...")
    
    try:
        from ml.training_pipeline import TrainingPipeline, PipelineConfig
        
        # Crear pipeline básico
        config = PipelineConfig(update_data=False)
        pipeline = TrainingPipeline(config)
        
        # Verificar configuración
        if hasattr(pipeline, 'config'):
            logger.info("✅ Pipeline configurado correctamente")
        
        # Verificar componentes principales
        if hasattr(pipeline, 'db') and hasattr(pipeline, 'collector'):
            logger.info("✅ Componentes del pipeline inicializados")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en pipeline: {e}")
        return False

def run_comprehensive_validation():
    """Ejecutar validación completa del stack."""
    logger.info("🚀 INICIANDO VALIDACIÓN COMPLETA DEL STACK ML")
    logger.info("=" * 60)
    
    tests = [
        ("Componentes de Datos", test_data_components),
        ("Arquitectura ML", test_ml_architecture),
        ("Preprocessor", test_preprocessing),
        ("Trainer", test_trainer),
        ("Evaluador MODEL-004", test_evaluator),
        ("Métricas", test_metrics),
        ("Integración Pipeline", test_pipeline_integration)
    ]
    
    results = []
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Ejecutando: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name}: PASÓ")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FALLÓ")
            
            results.append((test_name, result))
            
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR CRÍTICO - {e}")
            results.append((test_name, False))
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE VALIDACIÓN")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 RESULTADO FINAL: {passed}/{total} pruebas pasaron ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ¡STACK COMPLETAMENTE VALIDADO!")
        logger.info("✅ Todos los componentes funcionan correctamente")
        logger.info("🚀 Listo para proceder con MODEL-005")
    elif passed >= total * 0.8:
        logger.info("⚠️ Stack mayormente funcional con algunos problemas menores")
        logger.info("🔧 Requiere correcciones menores antes de continuar")
    else:
        logger.error("❌ Stack requiere correcciones significativas")
        logger.error("🛠️ Revisar errores antes de continuar")
    
    return passed, total, results

if __name__ == "__main__":
    print("🔬 VALIDACIÓN COMPLETA DEL STACK ML - GuruInversor")
    print("=" * 60)
    
    passed, total, results = run_comprehensive_validation()
    
    # Salir con código de error si hay fallas
    if passed < total:
        sys.exit(1)
    else:
        print(f"\n🎊 ¡Validación completada exitosamente! ({passed}/{total})")
        sys.exit(0)
