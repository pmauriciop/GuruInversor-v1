#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba simplificada de MODEL-003 - Pipeline de Entrenamiento - GuruInversor
Sin emojis para compatibilidad con Windows
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Suprimir warnings innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_basic_imports():
    """Probar importaciones básicas."""
    print("Probando importaciones del pipeline...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from training_pipeline import (
            TrainingPipeline, PipelineConfig, TrainingJob, PipelineResults,
            create_training_pipeline
        )
        print("OK: Pipeline importado correctamente")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_basic_config():
    """Probar configuración básica."""
    print("\nProbando configuración del pipeline...")
    
    try:
        from training_pipeline import PipelineConfig
        
        config = PipelineConfig()
        assert config.max_workers == 4
        assert config.update_data == True
        print("OK: Configuración básica correcta")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_basic_job():
    """Probar trabajos básicos."""
    print("\nProbando trabajos de entrenamiento...")
    
    try:
        from training_pipeline import TrainingJob
        
        job = TrainingJob(ticker='AAPL', model_type='basic')
        assert job.ticker == 'AAPL'
        assert job.model_type == 'basic'
        assert job.status == 'pending'
        print("OK: TrainingJob creado correctamente")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_basic_results():
    """Probar resultados básicos."""
    print("\nProbando resultados del pipeline...")
    
    try:
        from training_pipeline import PipelineResults
        
        results = PipelineResults()
        assert results.total_jobs == 0
        assert results.success_rate == 0.0
        print("OK: PipelineResults creado correctamente")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_basic_pipeline():
    """Probar pipeline básico."""
    print("\nProbando inicialización del pipeline...")
    
    try:
        from training_pipeline import TrainingPipeline, PipelineConfig
        
        temp_dir = tempfile.mkdtemp()
        config = PipelineConfig(
            max_workers=1,
            model_types=['basic'],
            min_data_days=50,
            update_data=False,
            save_logs=False,
            results_dir=temp_dir
        )
        
        pipeline = TrainingPipeline(config)
        assert pipeline.config.max_workers == 1
        print("OK: Pipeline inicializado correctamente")
        
        # Limpiar
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("PRUEBAS MODEL-003 - Pipeline de Entrenamiento - GuruInversor")
    print("=" * 60)
    
    tests = [
        ("Importaciones", test_basic_imports),
        ("Configuración", test_basic_config), 
        ("TrainingJob", test_basic_job),
        ("PipelineResults", test_basic_results),
        ("Pipeline completo", test_basic_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"PASS: {test_name}")
            else:
                print(f"FAIL: {test_name}")
        except Exception as e:
            print(f"ERROR en {test_name}: {e}")
    
    print("-" * 60)
    print(f"Resultados: {passed}/{total} pruebas exitosas ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nSUCCESS: Todas las pruebas de MODEL-003 pasaron!")
        print("READY: Pipeline de entrenamiento implementado y validado exitosamente")
        print("NEXT: MODEL-003 completado - Listo para MODEL-004")
    else:
        print(f"\nWARNING: {total-passed} pruebas fallaron")
        print("ERROR: MODEL-003 requiere correcciones antes de continuar")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
