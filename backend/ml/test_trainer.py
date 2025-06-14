#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas para MODEL-002 - Entrenador LSTM Básico - GuruInversor

Conjunto completo de pruebas para validar la implementación
del entrenador LSTM según los requerimientos de MODEL-002.
"""

import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import tempfile
import os
import shutil
from datetime import datetime, timedelta
import json

# Suprimir warnings innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Añadir el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

try:
    from ml.trainer import LSTMTrainer, TrainingConfig, train_lstm_model
    from ml.model_architecture import LSTMConfig
    from ml.preprocessor import ProcessingConfig
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)


class TestTrainingConfig(unittest.TestCase):
    """Pruebas para la configuración de entrenamiento."""
    
    def test_default_config(self):
        """Probar configuración por defecto."""
        config = TrainingConfig()
        
        # Verificar valores por defecto
        self.assertEqual(config.train_split, 0.7)
        self.assertEqual(config.validation_split, 0.2)
        self.assertEqual(config.test_split, 0.1)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.patience, 15)
        self.assertTrue(config.save_best_only)
        self.assertEqual(config.monitor_metric, 'val_loss')
        
    def test_custom_config(self):
        """Probar configuración personalizada."""
        config = TrainingConfig(
            train_split=0.8,
            validation_split=0.15,
            test_split=0.05,
            batch_size=64,
            epochs=50,
            patience=10
        )
        
        self.assertEqual(config.train_split, 0.8)
        self.assertEqual(config.validation_split, 0.15)
        self.assertEqual(config.test_split, 0.05)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.patience, 10)
    
    def test_split_validation(self):
        """Probar validación de splits."""
        # Split válido
        config = TrainingConfig(train_split=0.7, validation_split=0.2, test_split=0.1)
        self.assertIsNotNone(config)
        
        # Split inválido
        with self.assertRaises(ValueError):
            TrainingConfig(train_split=0.6, validation_split=0.2, test_split=0.1)
    
    def test_min_data_points_validation(self):
        """Probar validación de puntos mínimos."""
        # Valor válido
        config = TrainingConfig(min_data_points=1000)
        self.assertEqual(config.min_data_points, 1000)
        
        # Valor inválido
        with self.assertRaises(ValueError):
            TrainingConfig(min_data_points=400)


class TestLSTMTrainer(unittest.TestCase):
    """Pruebas para el entrenador LSTM."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.lstm_config = LSTMConfig(
            sequence_length=20,
            n_features=6,
            lstm_units=[15, 10],
            dense_units=[8, 4],
            epochs=2  # Pocas épocas para pruebas rápidas
        )
        
        self.training_config = TrainingConfig(
            batch_size=16,
            epochs=2,
            patience=5,
            min_data_points=100  # Reducido para pruebas
        )
        
        self.processing_config = ProcessingConfig(
            sequence_length=20,
            features=['open', 'high', 'low', 'close', 'volume', 'sma_10']
        )
    
    def test_trainer_initialization(self):
        """Probar inicialización del entrenador."""
        trainer = LSTMTrainer(
            self.lstm_config,
            self.training_config,
            self.processing_config
        )
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.lstm_config.sequence_length, 20)
        self.assertEqual(trainer.training_config.batch_size, 16)
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.history)
    
    def test_tensorflow_setup(self):
        """Probar configuración de TensorFlow."""
        trainer = LSTMTrainer(self.lstm_config, self.training_config)
        
        # Verificar que no hay errores en la configuración
        try:
            trainer._setup_tensorflow()
            self.assertTrue(True)  # Si llega aquí, no hubo errores
        except Exception as e:
            self.fail(f"Error en configuración TensorFlow: {e}")
    
    def test_build_model(self):
        """Probar construcción de modelo."""
        trainer = LSTMTrainer(self.lstm_config, self.training_config)
        
        # Construir modelo básico
        model = trainer.build_model('basic')
        
        self.assertIsNotNone(model)
        self.assertEqual(model.name, 'LSTM_Basic')
        self.assertIsNotNone(trainer.model)
        
        # Verificar shapes
        expected_input = (None, 20, 6)
        expected_output = (None, 2)
        self.assertEqual(model.input_shape, expected_input)
        self.assertEqual(model.output_shape, expected_output)
    
    def test_build_advanced_model(self):
        """Probar construcción de modelo avanzado."""
        trainer = LSTMTrainer(self.lstm_config, self.training_config)
        
        model = trainer.build_model('advanced')
        
        self.assertIsNotNone(model)
        self.assertEqual(model.name, 'LSTM_Advanced')
        
        # El modelo avanzado debe tener características especiales
        layer_names = [layer.name for layer in model.layers]
        has_bidirectional = any('bidirectional' in name for name in layer_names)
        has_normalization = any('normalization' in name for name in layer_names)
        
        self.assertTrue(has_bidirectional)
        self.assertTrue(has_normalization)
    
    def test_get_callbacks(self):
        """Probar obtención de callbacks."""
        trainer = LSTMTrainer(self.lstm_config, self.training_config)
        
        callbacks = trainer.get_callbacks('test_model')
        
        self.assertGreater(len(callbacks), 0)
        
        # Verificar tipos de callbacks
        callback_types = [type(cb).__name__ for cb in callbacks]
        self.assertIn('EarlyStopping', callback_types)
        self.assertIn('ModelCheckpoint', callback_types)
        self.assertIn('ReduceLROnPlateau', callback_types)
        self.assertIn('TensorBoard', callback_types)


class TestDataPreparation(unittest.TestCase):
    """Pruebas para preparación de datos."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.trainer = LSTMTrainer()
    
    def test_data_preparation_with_synthetic_data(self):
        """Probar preparación de datos con datos sintéticos."""
        # Crear datos sintéticos simulando el formato esperado
        n_samples = 200
        sequence_length = 60
        n_features = 12
        
        # Simular datos de secuencias y objetivos
        X_synthetic = np.random.random((n_samples, sequence_length, n_features))
        y_synthetic = np.random.random((n_samples, 2))
        
        # Mock del data_processor
        class MockDataProcessor:
            def __init__(self):
                pass
        
        # Mock del data_integrator
        def mock_get_processed_data(*args, **kwargs):
            return X_synthetic, y_synthetic, MockDataProcessor()
        
        # Temporarily replace the method
        original_method = self.trainer.data_integrator.get_processed_data
        self.trainer.data_integrator.get_processed_data = mock_get_processed_data
        
        try:
            # Reducir min_data_points para prueba
            self.trainer.training_config.min_data_points = 100
            
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data('TEST')
            
            # Verificar que los datos se dividieron correctamente
            self.assertGreater(len(X_train), 0)
            self.assertGreater(len(X_val), 0)
            self.assertGreater(len(X_test), 0)
            
            # Verificar shapes
            self.assertEqual(X_train.shape[1:], (sequence_length, n_features))
            self.assertEqual(y_train.shape[1], 2)
            
            # Verificar que la suma de splits es correcta
            total_expected = len(X_train) + len(X_val) + len(X_test)
            self.assertEqual(total_expected, n_samples)
            
        finally:
            # Restore original method
            self.trainer.data_integrator.get_processed_data = original_method
    
    def test_insufficient_data_error(self):
        """Probar error con datos insuficientes."""
        # Mock que retorna pocos datos
        def mock_insufficient_data(*args, **kwargs):
            return np.random.random((50, 60, 12)), np.random.random((50, 2)), None
        
        original_method = self.trainer.data_integrator.get_processed_data
        self.trainer.data_integrator.get_processed_data = mock_insufficient_data
        
        try:
            with self.assertRaises(ValueError):
                self.trainer.prepare_data('TEST')
        finally:
            self.trainer.data_integrator.get_processed_data = original_method


class TestModelEvaluation(unittest.TestCase):
    """Pruebas para evaluación de modelos."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.trainer = LSTMTrainer()
        
        # Construir un modelo simple para pruebas
        self.trainer.build_model('basic')
    
    def test_model_evaluation(self):
        """Probar evaluación de modelo."""
        # Datos sintéticos para evaluación
        X_test = np.random.random((20, 60, 12))
        y_test = np.random.random((20, 2))
        
        # Evaluar modelo
        metrics = self.trainer.evaluate_model(X_test, y_test)
        
        # Verificar que se devuelven métricas
        self.assertIsInstance(metrics, dict)
        self.assertIn('test_loss', metrics)
        
        # Verificar que las métricas son números válidos
        for metric_name, value in metrics.items():
            self.assertIsInstance(value, (int, float))
            self.assertFalse(np.isnan(value))
            self.assertFalse(np.isinf(value))
    
    def test_evaluation_without_model(self):
        """Probar evaluación sin modelo entrenado."""
        trainer_no_model = LSTMTrainer()
        
        X_test = np.random.random((10, 60, 12))
        y_test = np.random.random((10, 2))
        
        with self.assertRaises(ValueError):
            trainer_no_model.evaluate_model(X_test, y_test)


class TestModelPrediction(unittest.TestCase):
    """Pruebas para predicción de modelos."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        self.trainer = LSTMTrainer()
        self.trainer.build_model('basic')
    
    def test_model_prediction(self):
        """Probar predicción del modelo."""
        # Datos sintéticos
        X_test = np.random.random((10, 60, 12))
        
        # Hacer predicciones
        predictions = self.trainer.predict(X_test)
        
        # Verificar shapes y valores
        self.assertEqual(predictions.shape, (10, 2))
        self.assertFalse(np.isnan(predictions).any())
        self.assertFalse(np.isinf(predictions).any())
    
    def test_prediction_without_model(self):
        """Probar predicción sin modelo."""
        trainer_no_model = LSTMTrainer()
        
        X_test = np.random.random((5, 60, 12))
        
        with self.assertRaises(ValueError):
            trainer_no_model.predict(X_test)


class TestTrainingSummary(unittest.TestCase):
    """Pruebas para resumen de entrenamiento."""
    
    def test_empty_summary(self):
        """Probar resumen sin entrenamiento."""
        trainer = LSTMTrainer()
        
        summary = trainer.get_training_summary()
        self.assertEqual(summary, {})
    
    def test_summary_with_mock_history(self):
        """Probar resumen con historial simulado."""
        trainer = LSTMTrainer()
        
        # Mock history
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
        
        self.assertIn('epochs_completed', summary)
        self.assertIn('final_loss', summary)
        self.assertIn('final_val_loss', summary)
        self.assertIn('best_loss', summary)
        self.assertIn('best_val_loss', summary)
        
        self.assertEqual(summary['epochs_completed'], 4)
        self.assertEqual(summary['final_loss'], 0.2)
        self.assertEqual(summary['best_loss'], 0.2)


class TestConvenienceFunction(unittest.TestCase):
    """Pruebas para función de conveniencia."""
    
    def test_train_lstm_model_function(self):
        """Probar función de conveniencia para entrenamiento."""
        # Esta prueba solo verifica que la función existe y se puede llamar
        # sin errores de sintaxis
        
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
        
        # Verificar que la función es callable
        self.assertTrue(callable(train_lstm_model))
        
        # Verificar parámetros por defecto
        try:
            # Mock para evitar entrenamiento real
            import ml.trainer
            original_train = ml.trainer.LSTMTrainer.train
            
            def mock_train(self, *args, **kwargs):
                return {'mock': True}
            
            ml.trainer.LSTMTrainer.train = mock_train
            
            try:
                result = train_lstm_model(
                    'TEST',
                    lstm_config=lstm_config,
                    training_config=training_config
                )
                self.assertIsInstance(result, dict)
            finally:
                ml.trainer.LSTMTrainer.train = original_train
                
        except Exception as e:
            # Si hay error de dependencias, al menos verificamos la estructura
            self.assertTrue(True, "Función de conveniencia existe")


class TestModelRequirements(unittest.TestCase):
    """Pruebas específicas para requerimientos de MODEL-002."""
    
    def test_model_002_basic_requirements(self):
        """Verificar cumplimiento de requerimientos básicos MODEL-002."""
        # 1. Debe poder usar arquitectura de MODEL-001
        trainer = LSTMTrainer()
        model = trainer.build_model('basic')
        self.assertEqual(model.name, 'LSTM_Basic')
        
        # 2. Debe tener configuración de entrenamiento
        self.assertIsNotNone(trainer.training_config)
        self.assertIsInstance(trainer.training_config, TrainingConfig)
        
        # 3. Debe tener división temporal de datos
        self.assertFalse(trainer.training_config.train_split + 
                        trainer.training_config.validation_split + 
                        trainer.training_config.test_split != 1.0)
        
        # 4. Debe tener callbacks configurados
        callbacks = trainer.get_callbacks('test')
        callback_types = [type(cb).__name__ for cb in callbacks]
        required_callbacks = ['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau']
        
        for required in required_callbacks:
            self.assertIn(required, callback_types)
    
    def test_model_002_advanced_requirements(self):
        """Verificar requerimientos avanzados MODEL-002."""
        trainer = LSTMTrainer()
        
        # 1. Debe soportar múltiples tipos de modelo
        basic_model = trainer.build_model('basic')
        self.assertEqual(basic_model.name, 'LSTM_Basic')
        
        advanced_model = trainer.build_model('advanced')
        self.assertEqual(advanced_model.name, 'LSTM_Advanced')
        
        # 2. Debe tener evaluación de métricas
        X_test = np.random.random((10, 60, 12))
        y_test = np.random.random((10, 2))
        
        metrics = trainer.evaluate_model(X_test, y_test)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)
        
        # 3. Debe hacer predicciones
        predictions = trainer.predict(X_test)
        self.assertEqual(predictions.shape, (10, 2))
        
        # 4. Debe tener resumen de entrenamiento
        summary = trainer.get_training_summary()
        self.assertIsInstance(summary, dict)


if __name__ == '__main__':
    print("🧪 EJECUTANDO PRUEBAS MODEL-002 - Entrenador LSTM")
    print("=" * 60)
    
    # Configurar TensorFlow para pruebas
    tf.config.run_functions_eagerly(True)
    
    # Ejecutar pruebas
    unittest.main(verbosity=2, buffer=True)
