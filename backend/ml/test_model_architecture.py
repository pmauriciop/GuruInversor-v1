#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas para MODEL-001 - Arquitectura LSTM - GuruInversor

Conjunto completo de pruebas para validar el diseño e implementación
de la arquitectura LSTM según los requerimientos de MODEL-001.
"""

import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import tempfile
import os

# Añadir el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from ml.model_architecture import (
    LSTMConfig,
    LSTMArchitect,
    create_lstm_model,
    validate_model_architecture
)


class TestLSTMConfig(unittest.TestCase):
    """Pruebas para la configuración de modelos LSTM."""
    
    def test_default_config(self):
        """Probar configuración por defecto."""
        config = LSTMConfig()
        
        # Verificar valores por defecto
        self.assertEqual(config.sequence_length, 60)
        self.assertEqual(config.n_features, 12)
        self.assertEqual(config.n_outputs, 2)
        self.assertEqual(config.lstm_layers, 2)
        self.assertEqual(config.dropout_rate, 0.2)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.optimizer, 'adam')
        self.assertTrue(config.early_stopping)
        
        # Verificar listas inicializadas
        self.assertEqual(config.lstm_units, [50, 50])
        self.assertEqual(config.dense_units, [25, 10])
        self.assertEqual(config.metrics, ['mae', 'mape'])
    
    def test_custom_config(self):
        """Probar configuración personalizada."""
        config = LSTMConfig(
            sequence_length=30,
            n_features=8,
            lstm_units=[40, 30],
            dense_units=[20, 15],
            dropout_rate=0.3,
            learning_rate=0.0005
        )
        
        self.assertEqual(config.sequence_length, 30)
        self.assertEqual(config.n_features, 8)
        self.assertEqual(config.lstm_units, [40, 30])
        self.assertEqual(config.dense_units, [20, 15])
        self.assertEqual(config.dropout_rate, 0.3)
        self.assertEqual(config.learning_rate, 0.0005)
    
    def test_config_validation(self):
        """Probar validación de configuración."""
        # Valores válidos
        config = LSTMConfig(sequence_length=30, n_features=10)
        self.assertGreater(config.sequence_length, 0)
        self.assertGreater(config.n_features, 0)
        self.assertGreaterEqual(config.dropout_rate, 0.0)
        self.assertLessEqual(config.dropout_rate, 1.0)


class TestLSTMArchitect(unittest.TestCase):
    """Pruebas para el arquitecto de modelos LSTM."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.config = LSTMConfig(
            sequence_length=30,
            n_features=8,
            lstm_units=[20, 15],
            dense_units=[10, 5]
        )
        self.architect = LSTMArchitect(self.config)
    
    def test_architect_initialization(self):
        """Probar inicialización del arquitecto."""
        self.assertIsNotNone(self.architect)
        self.assertEqual(self.architect.config.sequence_length, 30)
        self.assertEqual(self.architect.config.n_features, 8)
        self.assertIsNone(self.architect.model)
        self.assertIsNone(self.architect.history)
    
    def test_gpu_setup(self):
        """Probar configuración de GPU."""
        # Esta prueba verifica que el método no falle
        try:
            self.architect._setup_gpu()
            self.assertTrue(True)  # Si llega aquí, no hubo errores
        except Exception as e:
            self.fail(f"Error en configuración GPU: {e}")


class TestBasicModel(unittest.TestCase):
    """Pruebas para modelo LSTM básico."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.config = LSTMConfig(
            sequence_length=20,
            n_features=6,
            lstm_units=[15, 10],
            dense_units=[8, 4]
        )
        self.architect = LSTMArchitect(self.config)
    
    def test_build_basic_model(self):
        """Probar construcción de modelo básico."""
        model = self.architect.build_basic_model()
        
        # Verificar que el modelo se creó
        self.assertIsNotNone(model)
        self.assertEqual(model.name, 'LSTM_Basic')
        
        # Verificar input shape
        expected_input = (None, 20, 6)
        self.assertEqual(model.input_shape, expected_input)
        
        # Verificar output shape
        expected_output = (None, 2)
        self.assertEqual(model.output_shape, expected_output)
        
        # Verificar que está compilado
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        
        # Verificar número de parámetros
        self.assertGreater(model.count_params(), 0)
    
    def test_model_layers(self):
        """Probar capas del modelo."""
        model = self.architect.build_basic_model()
        
        # Verificar que tiene las capas esperadas
        layer_names = [layer.name for layer in model.layers]
        
        # Debe tener capas LSTM
        lstm_layers = [name for name in layer_names if 'lstm' in name]
        self.assertEqual(len(lstm_layers), 2)
        
        # Debe tener capas densas
        dense_layers = [name for name in layer_names if 'dense' in name]
        self.assertGreaterEqual(len(dense_layers), 2)
        
        # Debe tener capa de predicción
        self.assertIn('predictions', layer_names)
    
    def test_model_prediction(self):
        """Probar predicción del modelo."""
        model = self.architect.build_basic_model()
        
        # Crear datos de prueba
        test_input = np.random.random((5, 20, 6))
        
        # Hacer predicción
        predictions = model.predict(test_input, verbose=0)
        
        # Verificar shape de predicción
        self.assertEqual(predictions.shape, (5, 2))
        
        # Verificar que las predicciones son números válidos
        self.assertFalse(np.isnan(predictions).any())
        self.assertFalse(np.isinf(predictions).any())


class TestAdvancedModel(unittest.TestCase):
    """Pruebas para modelo LSTM avanzado."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.config = LSTMConfig(
            sequence_length=25,
            n_features=8,
            lstm_units=[20, 15],
            dense_units=[12, 6]
        )
        self.architect = LSTMArchitect(self.config)
    
    def test_build_advanced_model(self):
        """Probar construcción de modelo avanzado."""
        model = self.architect.build_advanced_model()
        
        # Verificar que el modelo se creó
        self.assertIsNotNone(model)
        self.assertEqual(model.name, 'LSTM_Advanced')
        
        # Verificar shapes
        self.assertEqual(model.input_shape, (None, 25, 8))
        self.assertEqual(model.output_shape, (None, 2))
        
        # Verificar que está compilado
        self.assertIsNotNone(model.optimizer)
        
        # El modelo avanzado debe tener más parámetros que el básico
        basic_model = self.architect.build_basic_model()
        self.assertGreater(model.count_params(), basic_model.count_params())
    
    def test_advanced_features(self):
        """Probar características avanzadas del modelo."""
        model = self.architect.build_advanced_model()
        layer_names = [layer.name for layer in model.layers]
        
        # Debe tener capas bidireccionales
        bidirectional_layers = [name for name in layer_names if 'bidirectional' in name]
        self.assertGreater(len(bidirectional_layers), 0)
        
        # Debe tener normalización
        norm_layers = [name for name in layer_names if 'normalization' in name]
        self.assertGreater(len(norm_layers), 0)
        
        # Debe tener mecanismo de atención
        attention_layers = [name for name in layer_names if 'attention' in name]
        self.assertGreater(len(attention_layers), 0)


class TestEnsembleModel(unittest.TestCase):
    """Pruebas para ensemble de modelos."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.config = LSTMConfig()
        self.architect = LSTMArchitect(self.config)
    
    def test_build_ensemble(self):
        """Probar construcción de ensemble."""
        ensemble = self.architect.build_ensemble_model()
        
        # Verificar que se crearon los modelos esperados
        expected_models = ['conservative', 'moderate', 'aggressive']
        self.assertEqual(set(ensemble.keys()), set(expected_models))
        
        # Verificar que todos son modelos válidos
        for name, model in ensemble.items():
            self.assertIsNotNone(model)
            self.assertTrue(name.title() in model.name)
            self.assertIsNotNone(model.optimizer)
    
    def test_ensemble_diversity(self):
        """Probar diversidad en el ensemble."""
        ensemble = self.architect.build_ensemble_model()
        
        # Obtener número de parámetros de cada modelo
        param_counts = {name: model.count_params() for name, model in ensemble.items()}
        
        # Los modelos deben tener diferentes números de parámetros
        unique_counts = set(param_counts.values())
        self.assertEqual(len(unique_counts), 3)  # Tres configuraciones diferentes
        
        # El modelo agresivo debe tener más parámetros
        self.assertGreater(param_counts['aggressive'], param_counts['moderate'])
        self.assertGreater(param_counts['moderate'], param_counts['conservative'])


class TestModelCallbacks(unittest.TestCase):
    """Pruebas para callbacks del modelo."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.config = LSTMConfig()
        self.architect = LSTMArchitect(self.config)
    
    def test_get_callbacks(self):
        """Probar obtención de callbacks."""
        callbacks = self.architect.get_callbacks('test_model')
        
        # Verificar que se crearon callbacks
        self.assertGreater(len(callbacks), 0)
        
        # Verificar tipos de callbacks
        callback_types = [type(cb).__name__ for cb in callbacks]
        
        if self.config.early_stopping:
            self.assertIn('EarlyStopping', callback_types)
        
        if self.config.reduce_lr:
            self.assertIn('ReduceLROnPlateau', callback_types)
        
        if self.config.model_checkpoint:
            self.assertIn('ModelCheckpoint', callback_types)
        
        self.assertIn('TensorBoard', callback_types)
    
    def test_custom_loss(self):
        """Probar función de pérdida personalizada."""
        loss_fn = self.architect.create_custom_loss()
        
        # Verificar que es callable
        self.assertTrue(callable(loss_fn))
        
        # Probar con datos sintéticos
        y_true = tf.constant([[100.0, 0.05], [95.0, -0.03]])
        y_pred = tf.constant([[101.0, 0.04], [94.0, -0.02]])
        
        loss_value = loss_fn(y_true, y_pred)
        
        # Verificar que la pérdida es un número válido
        self.assertFalse(tf.math.is_nan(loss_value))
        self.assertFalse(tf.math.is_inf(loss_value))
        self.assertGreater(loss_value, 0)


class TestModelValidation(unittest.TestCase):
    """Pruebas para validación de modelos."""
    
    def test_validate_model_architecture(self):
        """Probar validación de arquitectura."""
        config = LSTMConfig(sequence_length=15, n_features=5)
        model = create_lstm_model(config, 'basic')
        
        # Validación debe ser exitosa
        is_valid = validate_model_architecture(model, (15, 5))
        self.assertTrue(is_valid)
        
        # Validación con shape incorrecto debe fallar
        is_valid_wrong = validate_model_architecture(model, (20, 5))
        self.assertFalse(is_valid_wrong)
    
    def test_model_summary(self):
        """Probar resumen del modelo."""
        config = LSTMConfig()
        architect = LSTMArchitect(config)
        model = architect.build_basic_model()
        
        summary = architect.get_model_summary(model)
        
        # Verificar campos obligatorios
        required_fields = ['name', 'total_params', 'trainable_params', 
                          'layers', 'input_shape', 'output_shape', 'summary']
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Verificar valores
        self.assertEqual(summary['name'], 'LSTM_Basic')
        self.assertGreater(summary['total_params'], 0)
        self.assertGreater(summary['layers'], 0)
        self.assertIsInstance(summary['summary'], str)


class TestConvenienceFunctions(unittest.TestCase):
    """Pruebas para funciones de conveniencia."""
    
    def test_create_lstm_model(self):
        """Probar función de creación de modelos."""
        config = LSTMConfig(sequence_length=10, n_features=4)
        
        # Probar modelo básico
        basic_model = create_lstm_model(config, 'basic')
        self.assertEqual(basic_model.name, 'LSTM_Basic')
        
        # Probar modelo avanzado
        advanced_model = create_lstm_model(config, 'advanced')
        self.assertEqual(advanced_model.name, 'LSTM_Advanced')
        
        # Probar ensemble
        ensemble = create_lstm_model(config, 'ensemble')
        self.assertIsInstance(ensemble, dict)
        self.assertEqual(len(ensemble), 3)
    
    def test_invalid_model_type(self):
        """Probar tipo de modelo inválido."""
        config = LSTMConfig()
        
        with self.assertRaises(ValueError):
            create_lstm_model(config, 'invalid_type')


class TestIntegrationModel(unittest.TestCase):
    """Pruebas de integración completas."""
    
    def test_end_to_end_basic_workflow(self):
        """Probar flujo completo básico."""
        # 1. Crear configuración
        config = LSTMConfig(
            sequence_length=20,
            n_features=6,
            lstm_units=[15, 10],
            dense_units=[8, 4]
        )
        
        # 2. Crear arquitecto
        architect = LSTMArchitect(config)
        
        # 3. Construir modelo
        model = architect.build_basic_model()
        
        # 4. Validar arquitectura
        is_valid = validate_model_architecture(model, (20, 6))
        self.assertTrue(is_valid)
        
        # 5. Obtener callbacks
        callbacks = architect.get_callbacks('integration_test')
        self.assertGreater(len(callbacks), 0)
        
        # 6. Probar predicción
        test_data = np.random.random((10, 20, 6))
        predictions = model.predict(test_data, verbose=0)
        
        self.assertEqual(predictions.shape, (10, 2))
        self.assertFalse(np.isnan(predictions).any())
    
    def test_end_to_end_advanced_workflow(self):
        """Probar flujo completo avanzado."""
        # 1. Crear modelo avanzado
        config = LSTMConfig()
        model = create_lstm_model(config, 'advanced')
        
        # 2. Validar
        is_valid = validate_model_architecture(model, (60, 12))
        self.assertTrue(is_valid)
        
        # 3. Probar con batch más grande
        test_data = np.random.random((50, 60, 12))
        predictions = model.predict(test_data, verbose=0)
        
        self.assertEqual(predictions.shape, (50, 2))
        
        # 4. Verificar que las predicciones son consistentes
        # Misma entrada debe dar misma salida
        predictions_2 = model.predict(test_data[:1], verbose=0)
        predictions_3 = model.predict(test_data[:1], verbose=0)
        
        np.testing.assert_array_equal(predictions_2, predictions_3)


class TestModelRequirements(unittest.TestCase):
    """Pruebas específicas para requerimientos de MODEL-001."""
    
    def test_model_001_basic_requirements(self):
        """Verificar cumplimiento de requerimientos básicos MODEL-001."""
        config = LSTMConfig()
        architect = LSTMArchitect(config)
        
        # 1. Debe soportar secuencias de 60 días
        self.assertEqual(config.sequence_length, 60)
        
        # 2. Debe soportar 12 características de entrada
        self.assertEqual(config.n_features, 12)
        
        # 3. Debe predecir 2 valores (precio y cambio)
        self.assertEqual(config.n_outputs, 2)
        
        # 4. Debe tener capas LSTM configurables
        model = architect.build_basic_model()
        layer_names = [layer.name for layer in model.layers]
        lstm_layers = [name for name in layer_names if 'lstm' in name]
        self.assertGreaterEqual(len(lstm_layers), 1)
        
        # 5. Debe tener regularización (dropout)
        dropout_layers = [layer for layer in model.layers 
                         if hasattr(layer, 'rate') and layer.rate > 0]
        self.assertGreater(len(dropout_layers), 0)
    
    def test_model_001_advanced_requirements(self):
        """Verificar requerimientos avanzados MODEL-001."""
        config = LSTMConfig()
        architect = LSTMArchitect(config)
        
        # 1. Soporte para múltiples arquitecturas
        basic_model = architect.build_basic_model()
        advanced_model = architect.build_advanced_model()
        ensemble = architect.build_ensemble_model()
        
        self.assertIsNotNone(basic_model)
        self.assertIsNotNone(advanced_model)
        self.assertIsInstance(ensemble, dict)
        
        # 2. Configuración flexible
        custom_config = LSTMConfig(
            lstm_units=[80, 60, 40],
            dense_units=[30, 15, 8],
            dropout_rate=0.25
        )
        custom_architect = LSTMArchitect(custom_config)
        custom_model = custom_architect.build_basic_model()
        
        # Debe tener más parámetros por configuración más compleja
        self.assertGreater(custom_model.count_params(), basic_model.count_params())
        
        # 3. Callbacks para entrenamiento
        callbacks = architect.get_callbacks()
        callback_types = [type(cb).__name__ for cb in callbacks]
        
        self.assertIn('EarlyStopping', callback_types)
        self.assertIn('ReduceLROnPlateau', callback_types)
        self.assertIn('TensorBoard', callback_types)


if __name__ == '__main__':
    print("🧪 EJECUTANDO PRUEBAS MODEL-001 - Arquitectura LSTM")
    print("=" * 60)
    
    # Configurar TensorFlow para pruebas
    tf.config.run_functions_eagerly(True)
    
    # Ejecutar pruebas
    unittest.main(verbosity=2, buffer=True)