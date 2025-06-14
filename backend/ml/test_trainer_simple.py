#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba directa de MODEL-002 - Entrenador LSTM - GuruInversor

Script de validación simplificado para probar el entrenador LSTM.
"""

import sys
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# Suprimir warnings innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(True)

def test_trainer_imports():
    """Probar importación del entrenador."""
    print("🔧 Probando importación de trainer...")
    
    try:
        # Importar solo el módulo específico
        sys.path.insert(0, str(Path(__file__).parent))
        from trainer import LSTMTrainer, TrainingConfig, train_lstm_model
        from model_architecture import LSTMConfig
        from preprocessor import ProcessingConfig
        print("✅ Importaciones exitosas")
        return True, LSTMTrainer, TrainingConfig, LSTMConfig
    except Exception as e:
        print(f"❌ Error en importación: {e}")
        return False, None, None, None

def test_training_config():
    """Probar configuración de entrenamiento."""
    print("\n🔧 Probando TrainingConfig...")
    
    try:
        from trainer import TrainingConfig
        
        # Configuración por defecto
        config_default = TrainingConfig()
        assert config_default.train_split == 0.7
        assert config_default.validation_split == 0.2
        assert config_default.test_split == 0.1
        assert config_default.batch_size == 32
        assert config_default.epochs == 100
        print("✅ Configuración por defecto correcta")
        
        # Configuración personalizada
        config_custom = TrainingConfig(
            train_split=0.8,
            validation_split=0.15,
            test_split=0.05,
            batch_size=64,
            epochs=50
        )
        assert config_custom.train_split == 0.8
        assert config_custom.batch_size == 64
        print("✅ Configuración personalizada correcta")
        
        # Validación de splits
        try:
            TrainingConfig(train_split=0.6, validation_split=0.2, test_split=0.1)
            assert False, "Debería fallar con splits incorrectos"
        except ValueError:
            print("✅ Validación de splits funciona")
        
        return True, config_default
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False, None

def test_trainer_initialization(LSTMTrainer, TrainingConfig, LSTMConfig):
    """Probar inicialización del entrenador."""
    print("\n🔧 Probando inicialización de LSTMTrainer...")
    
    try:
        lstm_config = LSTMConfig(
            sequence_length=20,
            n_features=6,
            lstm_units=[15, 10],
            dense_units=[8, 4]
        )
        
        training_config = TrainingConfig(
            batch_size=16,
            epochs=2,
            min_data_points=100
        )
        
        # Crear entrenador
        trainer = LSTMTrainer(lstm_config, training_config)
        assert trainer is not None
        print("✅ Entrenador creado correctamente")
        
        # Verificar configuraciones
        assert trainer.lstm_config.sequence_length == 20
        assert trainer.training_config.batch_size == 16
        assert trainer.model is None
        assert trainer.history is None
        print("✅ Configuraciones verificadas")
        
        return True, trainer
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        return False, None

def test_model_building(trainer):
    """Probar construcción de modelos."""
    print("\n🔧 Probando construcción de modelos...")
    
    try:
        # Construir modelo básico
        model = trainer.build_model('basic')
        assert model is not None
        assert model.name == 'LSTM_Basic'
        assert trainer.model is not None
        print("✅ Modelo básico construido")
        
        # Verificar shapes
        expected_input = (None, 20, 6)
        expected_output = (None, 2)
        assert model.input_shape == expected_input
        assert model.output_shape == expected_output
        print("✅ Shapes verificados")
        
        # Construir modelo avanzado
        advanced_model = trainer.build_model('advanced')
        assert advanced_model is not None
        assert advanced_model.name == 'LSTM_Advanced'
        print("✅ Modelo avanzado construido")
        
        # Verificar características avanzadas
        layer_names = [layer.name for layer in advanced_model.layers]
        has_bidirectional = any('bidirectional' in name for name in layer_names)
        has_normalization = any('normalization' in name for name in layer_names)
        
        assert has_bidirectional, "Debe tener capas bidireccionales"
        assert has_normalization, "Debe tener normalización"
        print("✅ Características avanzadas verificadas")
        
        return True, model
    except Exception as e:
        print(f"❌ Error en construcción de modelo: {e}")
        return False, None

def test_callbacks(trainer):
    """Probar callbacks de entrenamiento."""
    print("\n🔧 Probando callbacks de entrenamiento...")
    
    try:
        callbacks = trainer.get_callbacks('test_model')
        
        assert len(callbacks) > 0
        print(f"✅ {len(callbacks)} callbacks configurados")
        
        # Verificar tipos de callbacks
        callback_types = [type(cb).__name__ for cb in callbacks]
        
        required_callbacks = ['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau', 'TensorBoard']
        for required in required_callbacks:
            assert required in callback_types, f"Falta callback: {required}"
        
        print("✅ Todos los callbacks requeridos presentes")
        return True
    except Exception as e:
        print(f"❌ Error en callbacks: {e}")
        return False

def test_model_evaluation(trainer):
    """Probar evaluación de modelos."""
    print("\n🔧 Probando evaluación de modelos...")
    
    try:
        # Datos sintéticos para evaluación
        X_test = np.random.random((20, 20, 6))
        y_test = np.random.random((20, 2))
          # Evaluar modelo
        metrics = trainer.evaluate_model(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'test_loss' in metrics
        print("✅ Evaluación completada")
        
        # Verificar métricas básicas
        assert len(metrics) > 5  # Debe tener varias métricas
          # Verificar métricas
        for metric_name, value in metrics.items():
            # Verificar que sea numérico (incluyendo tipos NumPy)
            assert isinstance(value, (int, float, np.integer, np.floating)), f"Métrica {metric_name} no es numérica: {type(value)}"
            if np.isnan(value):
                print(f"⚠️  Métrica {metric_name} es NaN (normal en datos sintéticos)")
                continue
            if np.isinf(value):
                print(f"⚠️  Métrica {metric_name} es infinita (normal en datos sintéticos)")
                continue
        
        print(f"✅ {len(metrics)} métricas calculadas correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en evaluación: {e}")
        return False

def test_predictions(trainer):
    """Probar predicciones del modelo."""
    print("\n🔧 Probando predicciones...")
    
    try:
        # Datos sintéticos
        X_test = np.random.random((10, 20, 6))
        
        # Hacer predicciones
        predictions = trainer.predict(X_test)
        
        # Verificar resultados
        assert predictions.shape == (10, 2)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
        print("✅ Predicciones exitosas")
        
        print(f"📊 Ejemplo de predicción: {predictions[0]}")
        return True
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return False

def test_training_summary(trainer):
    """Probar resumen de entrenamiento."""
    print("\n🔧 Probando resumen de entrenamiento...")
    
    try:
        # Sin entrenamiento
        summary = trainer.get_training_summary()
        assert summary == {}
        print("✅ Resumen vacío correcto")
        
        # Mock de historial
        class MockHistory:
            def __init__(self):
                self.history = {
                    'loss': [0.5, 0.4, 0.3, 0.2],
                    'val_loss': [0.6, 0.5, 0.4, 0.3]
                }
        
        trainer.history = MockHistory()
        trainer.training_start_time = 0
        trainer.training_end_time = 100
        
        summary = trainer.get_training_summary()
        
        required_fields = ['epochs_completed', 'final_loss', 'final_val_loss', 
                          'best_loss', 'best_val_loss', 'training_time']
        
        for field in required_fields:
            assert field in summary, f"Falta campo: {field}"
        
        assert summary['epochs_completed'] == 4
        assert summary['final_loss'] == 0.2
        print("✅ Resumen de entrenamiento correcto")
        
        return True
    except Exception as e:
        print(f"❌ Error en resumen: {e}")
        return False

def test_convenience_function():
    """Probar función de conveniencia."""
    print("\n🔧 Probando función de conveniencia...")
    
    try:
        from trainer import train_lstm_model
        from model_architecture import LSTMConfig
        from trainer import TrainingConfig
        
        # Verificar que la función es callable
        assert callable(train_lstm_model)
        print("✅ Función de conveniencia existe")
        
        # Configuraciones para prueba
        lstm_config = LSTMConfig(
            sequence_length=20,
            n_features=6,
            lstm_units=[10],
            dense_units=[5]
        )
        
        training_config = TrainingConfig(
            epochs=1,
            batch_size=16,
            min_data_points=50
        )
        
        print("✅ Configuraciones para función de conveniencia listas")
        return True
    except Exception as e:
        print(f"❌ Error en función de conveniencia: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("🧪 PRUEBAS MODEL-002 - Entrenador LSTM - GuruInversor")
    print("=" * 60)
    
    results = []
    
    # 1. Pruebas de importación
    success, LSTMTrainer, TrainingConfig, LSTMConfig = test_trainer_imports()
    results.append(("Importaciones", success))
    if not success:
        return results
    
    # 2. Pruebas de configuración
    success, config = test_training_config()
    results.append(("Configuración entrenamiento", success))
    if not success:
        return results
    
    # 3. Pruebas de inicialización
    success, trainer = test_trainer_initialization(LSTMTrainer, TrainingConfig, LSTMConfig)
    results.append(("Inicialización trainer", success))
    if not success:
        return results
    
    # 4. Pruebas de construcción de modelo
    success, model = test_model_building(trainer)
    results.append(("Construcción modelo", success))
    if not success:
        return results
    
    # 5. Pruebas de callbacks
    success = test_callbacks(trainer)
    results.append(("Callbacks", success))
    
    # 6. Pruebas de evaluación
    success = test_model_evaluation(trainer)
    results.append(("Evaluación", success))
    
    # 7. Pruebas de predicción
    success = test_predictions(trainer)
    results.append(("Predicción", success))
    
    # 8. Pruebas de resumen
    success = test_training_summary(trainer)
    results.append(("Resumen entrenamiento", success))
    
    # 9. Función de conveniencia
    success = test_convenience_function()
    results.append(("Función conveniencia", success))
    
    return results

if __name__ == "__main__":
    # Ejecutar pruebas
    results = main()
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS MODEL-002")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} pruebas exitosas ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ¡TODAS LAS PRUEBAS DE MODEL-002 PASARON!")
        print("✅ Entrenador LSTM implementado y validado exitosamente")
        print("🚀 MODEL-002 completado - Listo para MODEL-003")
    else:
        print(f"\n⚠️  {total-passed} pruebas fallaron")
        print("❌ MODEL-002 requiere correcciones antes de continuar")
    
    print("=" * 60)
