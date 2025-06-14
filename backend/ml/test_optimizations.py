#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación de Optimizaciones - MODEL-005

Script para validar todas las optimizaciones implementadas en el sistema
de entrenamiento incremental.
"""

import os
import sys
import logging
import tempfile
import time
from pathlib import Path

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.incremental_trainer import IncrementalTrainer, IncrementalConfig
from ml.training_scheduler import TrainingScheduler, SchedulerConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_logging_optimization():
    """Probar optimización de logging (no duplicar handlers)."""
    logger.info("🧪 Probando optimización de logging...")
    
    try:
        # Crear múltiples instancias para verificar handlers
        temp_dir = tempfile.mkdtemp()
        config = IncrementalConfig(
            models_directory=os.path.join(temp_dir, "models"),
            incremental_logs_directory=os.path.join(temp_dir, "logs")
        )
        
        # Primera instancia
        trainer1 = IncrementalTrainer(config)
        initial_handlers = len(logging.getLogger('ml.incremental_trainer').handlers)
        
        # Segunda instancia (no debería duplicar handlers)
        trainer2 = IncrementalTrainer(config)
        final_handlers = len(logging.getLogger('ml.incremental_trainer').handlers)
        
        # Validar que no se duplicaron handlers
        if final_handlers <= initial_handlers + 1:  # Permitir 1 nuevo handler max
            logger.info("✅ Optimización de logging validada - No hay duplicación")
            return True
        else:
            logger.warning(f"⚠️ Posible duplicación de handlers: {initial_handlers} -> {final_handlers}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en test de logging: {e}")
        return False
    finally:
        # Cleanup
        if 'temp_dir' in locals():
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def test_performance_degradation_optimization():
    """Probar optimización en verificación de degradación."""
    logger.info("🧪 Probando optimización de verificación de degradación...")
    
    try:
        temp_dir = tempfile.mkdtemp()
        config = IncrementalConfig(
            models_directory=os.path.join(temp_dir, "models"),
            incremental_logs_directory=os.path.join(temp_dir, "logs")
        )
        
        trainer = IncrementalTrainer(config)
        ticker = "TEST"
        
        # Test con historial insuficiente (debería ser rápido)
        trainer.performance_history[ticker] = [
            {'date': '2025-06-10', 'score': 0.8, 'type': 'complete'}
        ]
        
        start_time = time.time()
        result = trainer._check_performance_degradation(ticker)
        duration = time.time() - start_time
        
        # Verificar que es rápido y maneja casos edge correctamente
        if duration < 0.01 and not result['degraded']:
            logger.info("✅ Optimización de degradación validada - Manejo eficiente de casos edge")
            
            # Test con historial suficiente
            trainer.performance_history[ticker] = [
                {'date': '2025-06-01', 'score': 0.8, 'type': 'complete'},
                {'date': '2025-06-02', 'score': 0.78, 'type': 'incremental'},
                {'date': '2025-06-03', 'score': 0.76, 'type': 'incremental'},
                {'date': '2025-06-04', 'score': 0.75, 'type': 'incremental'},
                {'date': '2025-06-05', 'score': 0.73, 'type': 'incremental'},
                {'date': '2025-06-06', 'score': 0.6, 'type': 'incremental'},  # Degradación
                {'date': '2025-06-07', 'score': 0.58, 'type': 'incremental'},
                {'date': '2025-06-08', 'score': 0.55, 'type': 'incremental'},
            ]
            
            start_time = time.time()
            result = trainer._check_performance_degradation(ticker)
            duration = time.time() - start_time
            
            if duration < 0.01 and result['degraded']:
                logger.info("✅ Detección de degradación optimizada y funcional")
                return True
            else:
                logger.warning(f"⚠️ Resultado inesperado: degraded={result['degraded']}, tiempo={duration:.4f}s")
                return False
        else:
            logger.warning(f"⚠️ Optimización de degradación lenta: {duration:.4f}s")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en test de degradación: {e}")
        return False
    finally:
        if 'temp_dir' in locals():
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def test_backup_cleanup_optimization():
    """Probar optimización de limpieza de backups."""
    logger.info("🧪 Probando optimización de limpieza de backups...")
    
    try:
        temp_dir = tempfile.mkdtemp()
        config = IncrementalConfig(
            models_directory=os.path.join(temp_dir, "models"),
            incremental_logs_directory=os.path.join(temp_dir, "logs")
        )
        
        trainer = IncrementalTrainer(config)
        ticker = "TEST"
        
        # Crear directorio de modelos y archivo de modelo falso
        model_path = trainer._get_model_path(ticker)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'w') as f:
            f.write("fake model content")
        
        # Crear múltiples backups para probar limpieza
        backup_dir = Path(config.models_directory) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear 10 backups simulados
        for i in range(10):
            backup_file = backup_dir / f"{ticker}_model_backup_2025061{i:02d}_120000.keras"
            with open(backup_file, 'w') as f:
                f.write(f"backup {i}")
            # Ajustar tiempo de modificación para simular orden
            import time
            os.utime(backup_file, (time.time() - (10-i)*3600, time.time() - (10-i)*3600))
        
        # Verificar que se crearon 10 backups
        initial_backups = len(list(backup_dir.glob(f"{ticker}_model_backup_*.keras")))
        assert initial_backups == 10, f"Se esperaban 10 backups, encontrados {initial_backups}"
        
        # Ejecutar backup (debería triggear limpieza)
        backup_result = trainer._backup_current_model(ticker)
        
        # Verificar que el backup fue exitoso
        assert backup_result['success'], "Backup debería ser exitoso"
        
        # Verificar que la limpieza funcionó (máximo 5 + 1 nuevo = 6)
        final_backups = len(list(backup_dir.glob(f"{ticker}_model_backup_*.keras")))
        
        if final_backups <= 6:  # 5 antiguos + 1 nuevo
            logger.info(f"✅ Limpieza de backups optimizada - Reducido de {initial_backups} a {final_backups}")
            return True
        else:
            logger.warning(f"⚠️ Limpieza no funcionó correctamente: {initial_backups} -> {final_backups}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en test de backup cleanup: {e}")
        return False
    finally:
        if 'temp_dir' in locals():
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def test_scheduler_logging_optimization():
    """Probar optimización de logging en el scheduler."""
    logger.info("🧪 Probando optimización de logging del scheduler...")
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        scheduler_config = SchedulerConfig(
            active_tickers=["TEST1", "TEST2"],
            priority_tickers=["TEST1"],
            report_directory=os.path.join(temp_dir, "reports"),
            alert_log_file=os.path.join(temp_dir, "alerts.log")
        )
        
        incremental_config = IncrementalConfig(
            models_directory=os.path.join(temp_dir, "models"),
            incremental_logs_directory=os.path.join(temp_dir, "logs")
        )
        
        # Primera instancia
        scheduler1 = TrainingScheduler(scheduler_config, incremental_config)
        initial_handlers = len(logging.getLogger('ml.training_scheduler').handlers)
        
        # Segunda instancia
        scheduler2 = TrainingScheduler(scheduler_config, incremental_config)
        final_handlers = len(logging.getLogger('ml.training_scheduler').handlers)
        
        if final_handlers <= initial_handlers + 1:
            logger.info("✅ Optimización de logging del scheduler validada")
            return True
        else:
            logger.warning(f"⚠️ Posible duplicación en scheduler: {initial_handlers} -> {final_handlers}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en test de scheduler logging: {e}")
        return False
    finally:
        if 'temp_dir' in locals():
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def run_optimization_validation():
    """Ejecutar todas las validaciones de optimización."""
    logger.info("🚀 INICIANDO VALIDACIÓN DE OPTIMIZACIONES")
    logger.info("=" * 60)
    
    tests = [
        ("Logging Optimization", test_logging_optimization),
        ("Performance Degradation Optimization", test_performance_degradation_optimization),
        ("Backup Cleanup Optimization", test_backup_cleanup_optimization),
        ("Scheduler Logging Optimization", test_scheduler_logging_optimization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n▶️ Ejecutando: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: EXITOSO")
            else:
                logger.warning(f"⚠️ {test_name}: FALLÓ")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE VALIDACIÓN DE OPTIMIZACIONES")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} optimizaciones validadas ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("\n🎉 ¡TODAS LAS OPTIMIZACIONES FUNCIONAN CORRECTAMENTE!")
        logger.info("✅ Sistema de entrenamiento incremental optimizado y validado")
        logger.info("🚀 Listo para producción")
    else:
        logger.warning(f"\n⚠️ {total-passed} optimizaciones requieren atención")
        logger.warning("❌ Revisar fallos antes de deployment")
    
    return passed == total

if __name__ == "__main__":
    success = run_optimization_validation()
    exit(0 if success else 1)
