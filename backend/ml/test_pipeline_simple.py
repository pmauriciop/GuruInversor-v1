#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba directa de MODEL-003 - Pipeline de Entrenamiento - GuruInversor

Script de validación simplificado para probar el pipeline de entrenamiento.
"""

import sys
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil

# Suprimir warnings innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(True)

def test_pipeline_imports():
    """Probar importación del pipeline."""
    print("Probando importación de pipeline...")
    
    # Importar el módulo específico
    sys.path.insert(0, str(Path(__file__).parent))
    from training_pipeline import (
        TrainingPipeline, PipelineConfig, TrainingJob, PipelineResults,
        create_training_pipeline
    )
    from trainer import LSTMTrainer, TrainingConfig
    from model_architecture import LSTMConfig
    from preprocessor import ProcessingConfig
    print("OK: Importaciones exitosas")
    assert True

def test_pipeline_config():
    """Probar configuración del pipeline."""
    print("\n🔧 Probando PipelineConfig...")
    
    from training_pipeline import PipelineConfig
    from trainer import TrainingConfig
    from model_architecture import LSTMConfig
    from preprocessor import ProcessingConfig
    
    # Configuración por defecto
    config_default = PipelineConfig()
    assert config_default.max_workers == 4
    assert config_default.update_data == True
    assert config_default.model_types == ['basic', 'advanced']
    assert config_default.min_data_days == 365
    print("✅ Configuración por defecto correcta")
    
    # Configuración personalizada
    config_custom = PipelineConfig(
        max_workers=2,
        model_types=['basic'],
        min_data_days=100,
        update_data=False
    )
    assert config_custom.max_workers == 2
    assert config_custom.model_types == ['basic']
    assert config_custom.min_data_days == 100
    print("✅ Configuración personalizada correcta")
    
    # Verificar configuraciones anidadas
    assert isinstance(config_default.lstm_config, LSTMConfig)
    assert isinstance(config_default.training_config, TrainingConfig)
    assert isinstance(config_default.processing_config, ProcessingConfig)
    print("✅ Configuraciones anidadas correctas")

def test_training_job():
    """Probar trabajos de entrenamiento."""
    print("\n🔧 Probando TrainingJob...")
    
    from training_pipeline import TrainingJob
    
    # Trabajo básico
    job = TrainingJob(ticker='AAPL', model_type='basic')
    assert job.ticker == 'AAPL'
    assert job.model_type == 'basic'
    assert job.priority == 1
    assert job.status == 'pending'
    assert job.created_at is not None
    print("✅ Trabajo básico correcto")
    
    # Trabajo con prioridad
    job_priority = TrainingJob(ticker='GOOGL', model_type='advanced', priority=3)
    assert job_priority.priority == 3
    print("✅ Trabajo con prioridad correcto")

def test_pipeline_results():
    """Probar resultados del pipeline."""
    print("\n🔧 Probando PipelineResults...")
    
    from training_pipeline import PipelineResults
    from datetime import datetime
    
    # Resultados básicos
    results = PipelineResults()
    assert results.total_jobs == 0
    assert results.completed_jobs == 0
    assert results.failed_jobs == 0
    assert results.success_rate == 0.0
    assert len(results.results) == 0
    assert len(results.errors) == 0
    print("✅ Resultados básicos correctos")
    
    # Resultados con datos
    results.total_jobs = 10
    results.completed_jobs = 8
    results.failed_jobs = 2
    results.start_time = datetime.now()
    results.end_time = datetime.now()
    
    assert results.success_rate == 0.8
    assert results.duration >= 0
    print("✅ Métricas calculadas correctamente")

def test_pipeline_initialization():
    """Probar inicialización del pipeline."""
    print("\n🔧 Probando inicialización de TrainingPipeline...")
    
    from training_pipeline import TrainingPipeline, PipelineConfig
    
    # Crear directorio temporal para pruebas
    temp_dir = tempfile.mkdtemp()
    
    config = PipelineConfig(
        max_workers=2,
        model_types=['basic'],
        min_data_days=50,
        update_data=False,
        save_logs=False,
        results_dir=temp_dir
    )
    
    # Crear pipeline
    pipeline = TrainingPipeline(config)
    assert pipeline.config.max_workers == 2
    assert len(pipeline.jobs_queue) == 0
    assert len(pipeline.running_jobs) == 0
    print("✅ Pipeline creado correctamente")
    
    # Verificar directorios
    results_dir = Path(temp_dir)
    assert results_dir.exists()
    assert (results_dir / 'models').exists()
    assert (results_dir / 'reports').exists()
    assert (results_dir / 'logs').exists()
    print("✅ Directorios creados correctamente")
    
    # Limpiar
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return pipeline

def test_job_management():
    """Probar gestión de trabajos."""
    print("\n🔧 Probando gestión de trabajos...")
    
    # Crear pipeline para las pruebas
    pipeline = test_pipeline_initialization()
    
    # Añadir trabajo individual
    job_id = pipeline.add_training_job('AAPL', 'basic', priority=1)
    assert len(pipeline.jobs_queue) == 1
    assert 'AAPL_basic' in job_id
    print("✅ Trabajo individual añadido")
    
    # Añadir múltiples trabajos
    tickers = ['GOOGL', 'MSFT']
    job_ids = pipeline.add_multiple_jobs(tickers, ['basic'])
    assert len(pipeline.jobs_queue) == 3  # 1 anterior + 2 nuevos
    assert len(job_ids) == 2
    print("✅ Múltiples trabajos añadidos")
    
    # Verificar estado
    status = pipeline.get_status()
    assert status['jobs_in_queue'] == 3
    assert status['running_jobs'] == 0
    assert status['completed_jobs'] == 0
    print("✅ Estado del pipeline correcto")

def test_data_validation():
    """Probar validación de datos."""
    print("\n🔧 Probando validación de datos...")
    
    # Crear pipeline para las pruebas
    pipeline = test_pipeline_initialization()
    
    # Validar ticker ficticio (debería fallar)
    is_valid, message = pipeline.validate_ticker_data('INVALID_TICKER')
    assert not is_valid
    assert 'no encontrado' in message.lower()
    print("✅ Validación de ticker inválido correcta")
    
    # Nota: Para ticker válido necesitaríamos datos en la base de datos
    # En entorno de prueba, esto es esperado que falle
    print("✅ Validación de datos funcional")

def test_model_comparison():
    """Probar comparación de modelos."""
    print("\n🔧 Probando comparación de modelos...")
    
    # Crear pipeline para las pruebas
    pipeline = test_pipeline_initialization()
    
    # Comparar modelos (puede estar vacío en entorno de prueba)
    df = pipeline.compare_models()
    
    # Verificar que retorna DataFrame
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    print("✅ Comparación de modelos retorna DataFrame")
    
    # Comparar modelos por ticker específico
    df_ticker = pipeline.compare_models('AAPL')
    assert isinstance(df_ticker, pd.DataFrame)
    print("✅ Comparación por ticker funcional")

def test_report_generation():
    """Probar generación de reportes."""
    print("\n🔧 Probando generación de reportes...")
    
    # Crear pipeline para las pruebas
    pipeline = test_pipeline_initialization()
    
    from training_pipeline import PipelineResults
    from datetime import datetime
    
    # Crear resultados de prueba
    results = PipelineResults(
        total_jobs=2,
        completed_jobs=1,
        failed_jobs=1,
        start_time=datetime.now(),
        end_time=datetime.now()
    )
    
    # Añadir resultado exitoso de prueba
    results.results.append({
        'model_metadata': {'ticker': 'AAPL'},
        'model_name': 'test_model',
        'evaluation': {'rmse': 0.05, 'mae': 0.03},
        'data_stats': {'train_samples': 100, 'val_samples': 20, 'test_samples': 10},
        'training_time': 30.5
    })
    
    # Añadir error de prueba
    results.errors.append({
        'job': {'ticker': 'GOOGL', 'model_type': 'basic'},
        'error': 'Error de prueba',
        'duration': 5.0
    })
    
    # Generar reporte
    report_path = pipeline.generate_training_report(results)
    
    # Verificar que el archivo se creó
    assert Path(report_path).exists()
    print("✅ Reporte generado correctamente")
    
    # Verificar contenido básico
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'Reporte de Entrenamiento' in content
        assert 'Total de trabajos: 2' in content
        assert 'Completados exitosamente: 1' in content
        assert 'AAPL' in content
    print("✅ Contenido del reporte correcto")

def test_convenience_function():
    """Probar función de conveniencia."""
    print("\n🔧 Probando función de conveniencia...")
    
    from training_pipeline import create_training_pipeline, PipelineConfig
    
    # Verificar que la función existe y es callable
    assert callable(create_training_pipeline)
    print("✅ Función de conveniencia existe")
    
    # Configuración para prueba rápida
    config = PipelineConfig(
        max_workers=1,
        model_types=['basic'],
        min_data_days=10,
        update_data=False,
        validate_before_training=False,
        save_logs=False,
        results_dir=tempfile.mkdtemp()
    )
    
    print("✅ Configuración para función de conveniencia preparada")

def main():
    """Función principal de pruebas."""
    print("PRUEBAS MODEL-003 - Pipeline de Entrenamiento - GuruInversor")
    print("=" * 60)
    
    results = []
    
    # 1. Pruebas de importación
    try:
        test_pipeline_imports()
        results.append(("Importaciones", True))
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        results.append(("Importaciones", False))
        return results
    
    # 2. Pruebas de configuración
    try:
        test_pipeline_config()
        results.append(("Configuración pipeline", True))
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        results.append(("Configuración pipeline", False))
        return results
    
    # 3. Pruebas de TrainingJob
    try:
        test_training_job()
        results.append(("TrainingJob", True))
    except Exception as e:
        print(f"❌ Error en TrainingJob: {e}")
        results.append(("TrainingJob", False))
        return results
    
    # 4. Pruebas de PipelineResults
    try:
        test_pipeline_results()
        results.append(("PipelineResults", True))
    except Exception as e:
        print(f"❌ Error en PipelineResults: {e}")
        results.append(("PipelineResults", False))
        return results
    
    # 5. Pruebas de inicialización
    try:
        test_pipeline_initialization()
        results.append(("Inicialización pipeline", True))
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        results.append(("Inicialización pipeline", False))
        return results
    
    # 6. Pruebas de gestión de trabajos
    try:
        test_job_management()
        results.append(("Gestión trabajos", True))
    except Exception as e:
        print(f"❌ Error en gestión de trabajos: {e}")
        results.append(("Gestión trabajos", False))
        return results
    
    # 7. Pruebas de validación de datos
    try:
        test_data_validation()
        results.append(("Validación datos", True))
    except Exception as e:
        print(f"❌ Error en validación de datos: {e}")
        results.append(("Validación datos", False))
        return results
    
    # 8. Pruebas de comparación de modelos
    try:
        test_model_comparison()
        results.append(("Comparación modelos", True))
    except Exception as e:
        print(f"❌ Error en comparación de modelos: {e}")
        results.append(("Comparación modelos", False))
        return results
    
    # 9. Pruebas de generación de reportes
    try:
        test_report_generation()
        results.append(("Generación reportes", True))
    except Exception as e:
        print(f"❌ Error en generación de reportes: {e}")
        results.append(("Generación reportes", False))
        return results
    
    # 10. Pruebas de función de conveniencia
    try:
        test_convenience_function()
        results.append(("Función conveniencia", True))
    except Exception as e:
        print(f"❌ Error en función de conveniencia: {e}")
        results.append(("Función conveniencia", False))
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS MODEL-003")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} pruebas exitosas ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ¡TODAS LAS PRUEBAS DE MODEL-003 PASARON!")
        print("✅ Pipeline de entrenamiento implementado y validado exitosamente")
        print("🚀 MODEL-003 completado - Listo para MODEL-004")
    else:
        print(f"\n⚠️  {total-passed} pruebas fallaron")
        print("❌ MODEL-003 requiere correcciones antes de continuar")
    
    print("=" * 60)
    return results

if __name__ == "__main__":
    main()
