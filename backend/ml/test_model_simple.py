#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba directa de MODEL-001 - Arquitectura LSTM - GuruInversor

Script de validación simplificado que no depende de imports externos.
"""

import sys
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# Suprimir warnings innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(True)

def test_basic_imports():
    """Probar importación básica de la arquitectura."""
    print("🔧 Probando importación de model_architecture...")
    
    try:
        # Importar solo el módulo específico
        sys.path.insert(0, str(Path(__file__).parent))
        from model_architecture import LSTMConfig, LSTMArchitect, create_lstm_model
        print("✅ Importaciones exitosas")
        return True, LSTMConfig, LSTMArchitect, create_lstm_model
    except Exception as e:
        print(f"❌ Error en importación: {e}")
        return False, None, None, None

def test_config_creation():
    """Probar creación de configuración."""
    print("\n🔧 Probando creación de LSTMConfig...")
    
    try:
        from model_architecture import LSTMConfig
        
        # Configuración por defecto
        config_default = LSTMConfig()
        assert config_default.sequence_length == 60
        assert config_default.n_features == 12
        assert config_default.n_outputs == 2
        assert config_default.lstm_units == [50, 50]
        assert config_default.dense_units == [25, 10]
        print("✅ Configuración por defecto correcta")
        
        # Configuración personalizada
        config_custom = LSTMConfig(
            sequence_length=30,
            n_features=8,
            lstm_units=[40, 30],
            dense_units=[20, 15]
        )
        assert config_custom.sequence_length == 30
        assert config_custom.n_features == 8
        assert config_custom.lstm_units == [40, 30]
        print("✅ Configuración personalizada correcta")
        
        return True, config_default
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False, None

def test_basic_model_creation(config):
    """Probar creación de modelo básico."""
    print("\n🔧 Probando creación de modelo LSTM básico...")
    
    try:
        from model_architecture import LSTMArchitect
        
        # Crear arquitecto
        architect = LSTMArchitect(config)
        assert architect is not None
        print("✅ Arquitecto creado correctamente")
        
        # Crear modelo básico
        model = architect.build_basic_model()
        assert model is not None
        assert model.name == 'LSTM_Basic'
        print("✅ Modelo básico creado")
        
        # Verificar shapes
        expected_input = (None, config.sequence_length, config.n_features)
        expected_output = (None, config.n_outputs)
        assert model.input_shape == expected_input
        assert model.output_shape == expected_output
        print("✅ Shapes de entrada y salida correctos")
        
        # Verificar compilación
        assert model.optimizer is not None
        assert model.loss is not None
        print("✅ Modelo compilado correctamente")
        
        # Verificar parámetros
        params = model.count_params()
        assert params > 0
        print(f"✅ Modelo tiene {params:,} parámetros")
        
        return True, model
    except Exception as e:
        print(f"❌ Error creando modelo básico: {e}")
        return False, None

def test_model_prediction(model, config):
    """Probar predicción del modelo."""
    print("\n🔧 Probando predicción del modelo...")
    
    try:
        # Crear datos de prueba
        test_input = np.random.random((5, config.sequence_length, config.n_features))
        print(f"✅ Datos de prueba creados: shape {test_input.shape}")
        
        # Hacer predicción
        predictions = model.predict(test_input, verbose=0)
        expected_shape = (5, config.n_outputs)
        assert predictions.shape == expected_shape
        print(f"✅ Predicción exitosa: shape {predictions.shape}")
        
        # Verificar que no hay NaN o Inf
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
        print("✅ Predicciones numéricamente válidas")
        
        # Mostrar ejemplo de predicción
        print(f"📊 Ejemplo de predicción: {predictions[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return False

def test_advanced_model():
    """Probar modelo avanzado."""
    print("\n🔧 Probando modelo LSTM avanzado...")
    
    try:
        from model_architecture import LSTMConfig, LSTMArchitect
        
        config = LSTMConfig(
            sequence_length=25,
            n_features=8,
            lstm_units=[20, 15],
            dense_units=[12, 6]
        )
        
        architect = LSTMArchitect(config)
        model = architect.build_advanced_model()
        
        assert model is not None
        assert model.name == 'LSTM_Advanced'
        print("✅ Modelo avanzado creado")
        
        # Verificar que tiene más complejidad
        layer_names = [layer.name for layer in model.layers]
        
        # Buscar características avanzadas
        has_bidirectional = any('bidirectional' in name for name in layer_names)
        has_normalization = any('normalization' in name for name in layer_names)
        has_attention = any('attention' in name for name in layer_names)
        
        assert has_bidirectional, "Debe tener capas bidireccionales"
        assert has_normalization, "Debe tener normalización"
        assert has_attention, "Debe tener mecanismo de atención"
        
        print("✅ Características avanzadas verificadas:")
        print(f"   - Bidireccional: {has_bidirectional}")
        print(f"   - Normalización: {has_normalization}")
        print(f"   - Atención: {has_attention}")
        
        return True
    except Exception as e:
        print(f"❌ Error en modelo avanzado: {e}")
        return False

def test_ensemble_model():
    """Probar ensemble de modelos."""
    print("\n🔧 Probando ensemble de modelos...")
    
    try:
        from model_architecture import LSTMConfig, LSTMArchitect
        
        config = LSTMConfig()
        architect = LSTMArchitect(config)
        ensemble = architect.build_ensemble_model()
        
        assert isinstance(ensemble, dict)
        expected_models = ['conservative', 'moderate', 'aggressive']
        assert set(ensemble.keys()) == set(expected_models)
        print("✅ Ensemble creado con modelos esperados")
        
        # Verificar diversidad
        param_counts = {name: model.count_params() for name, model in ensemble.items()}
        unique_counts = set(param_counts.values())
        assert len(unique_counts) == 3, "Los modelos deben tener diferentes tamaños"
        
        print("✅ Diversidad del ensemble verificada:")
        for name, count in param_counts.items():
            print(f"   - {name}: {count:,} parámetros")
        
        return True
    except Exception as e:
        print(f"❌ Error en ensemble: {e}")
        return False

def test_convenience_functions():
    """Probar funciones de conveniencia."""
    print("\n🔧 Probando funciones de conveniencia...")
    
    try:
        from model_architecture import LSTMConfig, create_lstm_model
        
        config = LSTMConfig(sequence_length=10, n_features=4)
        
        # Probar diferentes tipos
        basic_model = create_lstm_model(config, 'basic')
        assert basic_model.name == 'LSTM_Basic'
        print("✅ create_lstm_model('basic') funciona")
        
        advanced_model = create_lstm_model(config, 'advanced')
        assert advanced_model.name == 'LSTM_Advanced'
        print("✅ create_lstm_model('advanced') funciona")
        
        ensemble = create_lstm_model(config, 'ensemble')
        assert isinstance(ensemble, dict)
        assert len(ensemble) == 3
        print("✅ create_lstm_model('ensemble') funciona")
        
        # Probar tipo inválido
        try:
            create_lstm_model(config, 'invalid')
            assert False, "Debería fallar con tipo inválido"
        except ValueError:
            print("✅ Manejo de errores correcto")
        
        return True
    except Exception as e:
        print(f"❌ Error en funciones de conveniencia: {e}")
        return False

def test_model_validation():
    """Probar validación de modelos."""
    print("\n🔧 Probando validación de arquitectura...")
    
    try:
        from model_architecture import LSTMConfig, create_lstm_model, validate_model_architecture
        
        config = LSTMConfig(sequence_length=15, n_features=5)
        model = create_lstm_model(config, 'basic')
        
        # Validación correcta
        is_valid = validate_model_architecture(model, (15, 5))
        assert is_valid, "Validación debería ser exitosa"
        print("✅ Validación exitosa para shapes correctos")
        
        # Validación incorrecta
        is_valid_wrong = validate_model_architecture(model, (20, 5))
        assert not is_valid_wrong, "Validación debería fallar con shapes incorrectos"
        print("✅ Validación falla correctamente para shapes incorrectos")
        
        return True
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("🧪 PRUEBAS MODEL-001 - Arquitectura LSTM - GuruInversor")
    print("=" * 60)
    
    results = []
    
    # 1. Pruebas de importación
    success, LSTMConfig, LSTMArchitect, create_lstm_model = test_basic_imports()
    results.append(("Importaciones", success))
    if not success:
        return results
    
    # 2. Pruebas de configuración
    success, config = test_config_creation()
    results.append(("Configuración", success))
    if not success:
        return results
    
    # 3. Pruebas de modelo básico
    success, model = test_basic_model_creation(config)
    results.append(("Modelo básico", success))
    if not success:
        return results
    
    # 4. Pruebas de predicción
    success = test_model_prediction(model, config)
    results.append(("Predicción", success))
    
    # 5. Pruebas de modelo avanzado
    success = test_advanced_model()
    results.append(("Modelo avanzado", success))
    
    # 6. Pruebas de ensemble
    success = test_ensemble_model()
    results.append(("Ensemble", success))
    
    # 7. Pruebas de funciones de conveniencia
    success = test_convenience_functions()
    results.append(("Funciones conveniencia", success))
    
    # 8. Pruebas de validación
    success = test_model_validation()
    results.append(("Validación", success))
    
    return results

if __name__ == "__main__":
    # Ejecutar pruebas
    results = main()
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS MODEL-001")
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
        print("\n🎉 ¡TODAS LAS PRUEBAS DE MODEL-001 PASARON!")
        print("✅ Arquitectura LSTM diseñada y validada exitosamente")
        print("🚀 MODEL-001 completado - Listo para MODEL-002")
    else:
        print(f"\n⚠️  {total-passed} pruebas fallaron")
        print("❌ MODEL-001 requiere correcciones antes de continuar")
    
    print("=" * 60)
