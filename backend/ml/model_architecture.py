#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arquitectura LSTM - GuruInversor

Diseño e implementación de modelos LSTM para predicción de precios de acciones.
Incluye arquitecturas básicas, intermedias y avanzadas con diferentes configuraciones.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.regularizers import l1_l2

# Añadir el directorio backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuración para modelos LSTM."""
    
    # Arquitectura básica
    sequence_length: int = 60  # Días de historia
    n_features: int = 12  # Número de características de entrada
    n_outputs: int = 2  # Predicciones: precio_siguiente, cambio_porcentual
    
    # Capas LSTM
    lstm_units: List[int] = None  # Unidades por capa LSTM
    lstm_layers: int = 2  # Número de capas LSTM
    dropout_rate: float = 0.2  # Dropout para prevenir overfitting
    recurrent_dropout: float = 0.1  # Dropout en conexiones recurrentes
    
    # Capas densas
    dense_units: List[int] = None  # Unidades en capas densas
    dense_layers: int = 2  # Número de capas densas
    
    # Regularización
    l1_reg: float = 0.001  # Regularización L1
    l2_reg: float = 0.001  # Regularización L2
    batch_norm: bool = True  # Usar BatchNormalization
    
    # Entrenamiento
    learning_rate: float = 0.001  # Tasa de aprendizaje
    optimizer: str = 'adam'  # Optimizador
    loss_function: str = 'mse'  # Función de pérdida
    metrics: List[str] = None  # Métricas adicionales
    
    # Callbacks
    early_stopping: bool = True  # Parada temprana
    reduce_lr: bool = True  # Reducir learning rate
    model_checkpoint: bool = True  # Guardar mejor modelo
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.lstm_units is None:
            self.lstm_units = [50, 50]
        if self.dense_units is None:
            self.dense_units = [25, 10]
        if self.metrics is None:
            self.metrics = ['mae', 'mape']


class LSTMArchitect:
    """
    Arquitecto principal para diseño y construcción de modelos LSTM.
    """
    
    def __init__(self, config: LSTMConfig = None):
        """
        Inicializar arquitecto LSTM.
        
        Args:
            config: Configuración del modelo
        """
        self.config = config or LSTMConfig()
        self.model = None
        self.history = None
        
        # Configurar GPU si está disponible
        self._setup_gpu()
        
    def _setup_gpu(self):
        """Configurar GPU para entrenamiento si está disponible."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Permitir crecimiento de memoria GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configurada: {len(gpus)} dispositivo(s) disponible(s)")
            else:
                logger.info("Usando CPU para entrenamiento")
        except Exception as e:
            logger.warning(f"Error configurando GPU: {e}")
    
    def build_basic_model(self) -> keras.Model:
        """
        Construir modelo LSTM básico.
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Construyendo modelo LSTM básico...")
        
        # Input layer
        inputs = keras.Input(
            shape=(self.config.sequence_length, self.config.n_features),
            name='stock_sequence'
        )
        
        x = inputs
        
        # Capas LSTM
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'lstm_{i+1}'
            )(x)
            
            if self.config.batch_norm:
                x = layers.BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
        
        # Capas densas
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            
            if self.config.batch_norm:
                x = layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            
            x = layers.Dropout(self.config.dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        # Output layer
        outputs = layers.Dense(
            units=self.config.n_outputs,
            activation='linear',
            name='predictions'
        )(x)
        
        # Crear modelo
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Basic')
        
        # Compilar modelo
        self._compile_model(model)
        
        logger.info(f"Modelo básico construido: {model.count_params():,} parámetros")
        return model
    
    def build_advanced_model(self) -> keras.Model:
        """
        Construir modelo LSTM avanzado con características adicionales.
        
        Returns:
            Modelo Keras avanzado
        """
        logger.info("Construyendo modelo LSTM avanzado...")
        
        # Input layer
        inputs = keras.Input(
            shape=(self.config.sequence_length, self.config.n_features),
            name='stock_sequence'
        )
        
        x = inputs
        
        # Capa de normalización de entrada
        x = layers.LayerNormalization(name='input_normalization')(x)
        
        # Bloque LSTM bidireccional
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            
            # LSTM bidireccional
            lstm_layer = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg)
            )
            
            x = layers.Bidirectional(
                lstm_layer,
                name=f'bidirectional_lstm_{i+1}'
            )(x)
            
            # Normalización y dropout
            if self.config.batch_norm:
                x = layers.BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
            
            x = layers.Dropout(self.config.dropout_rate, name=f'dropout_lstm_{i+1}')(x)
        
        # Mecanismo de atención simple
        attention = layers.Dense(1, activation='softmax', name='attention_weights')(x)
        x = layers.Multiply(name='attention_applied')([x, attention])
        
        # Capas densas con conexiones residuales
        dense_input = x
        for i, units in enumerate(self.config.dense_units):
            x = layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l1_l2(self.config.l1_reg, self.config.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            
            if self.config.batch_norm:
                x = layers.BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            
            x = layers.Dropout(self.config.dropout_rate, name=f'dropout_dense_{i+1}')(x)
            
            # Conexión residual si las dimensiones coinciden
            if i == 0 and x.shape[-1] == dense_input.shape[-1]:
                x = layers.Add(name=f'residual_{i+1}')([x, dense_input])
        
        # Capa de salida con múltiples cabezas
        # Cabeza 1: Predicción de precio
        price_head = layers.Dense(
            units=1,
            activation='linear',
            name='price_prediction'
        )(x)
        
        # Cabeza 2: Predicción de dirección (cambio porcentual)
        direction_head = layers.Dense(
            units=1,
            activation='tanh',  # Para cambios porcentuales (-1 a 1)
            name='direction_prediction'
        )(x)
        
        # Combinar salidas
        outputs = layers.Concatenate(name='combined_output')([price_head, direction_head])
        
        # Crear modelo
        model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Advanced')
        
        # Compilar modelo
        self._compile_model(model)
        
        logger.info(f"Modelo avanzado construido: {model.count_params():,} parámetros")
        return model
    
    def build_ensemble_model(self) -> Dict[str, keras.Model]:
        """
        Construir ensemble de modelos LSTM con diferentes configuraciones.
        
        Returns:
            Diccionario con múltiples modelos
        """
        logger.info("Construyendo ensemble de modelos LSTM...")
        
        models_ensemble = {}
        
        # Configuraciones para diferentes modelos
        configs = {
            'conservative': LSTMConfig(
                lstm_units=[30, 20],
                dense_units=[15, 8],
                dropout_rate=0.3,
                learning_rate=0.0005
            ),
            'moderate': LSTMConfig(
                lstm_units=[50, 30],
                dense_units=[25, 12],
                dropout_rate=0.2,
                learning_rate=0.001
            ),
            'aggressive': LSTMConfig(
                lstm_units=[80, 50, 30],
                dense_units=[40, 20, 10],
                dropout_rate=0.15,
                learning_rate=0.002
            )
        }
        
        for name, config in configs.items():
            logger.info(f"Construyendo modelo {name}...")
            
            # Temporalmente cambiar configuración
            original_config = self.config
            self.config = config
            
            # Construir modelo básico con la nueva configuración
            model = self.build_basic_model()
            model._name = f'LSTM_{name.title()}'
            
            models_ensemble[name] = model
            
            # Restaurar configuración original
            self.config = original_config
        
        logger.info(f"Ensemble creado con {len(models_ensemble)} modelos")
        return models_ensemble
    
    def _compile_model(self, model: keras.Model):
        """
        Compilar modelo con optimizador y métricas.
        
        Args:
            model: Modelo a compilar
        """
        # Configurar optimizador
        if self.config.optimizer == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            optimizer = optimizers.SGD(learning_rate=self.config.learning_rate)
        else:
            optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        
        # Compilar modelo
        model.compile(
            optimizer=optimizer,
            loss=self.config.loss_function,
            metrics=self.config.metrics
        )
    
    def get_callbacks(self, model_name: str = 'lstm_model') -> List[callbacks.Callback]:
        """
        Obtener callbacks para entrenamiento.
        
        Args:
            model_name: Nombre base para archivos del modelo
            
        Returns:
            Lista de callbacks configurados
        """
        callback_list = []
        
        # Early stopping
        if self.config.early_stopping:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        # Reduce learning rate
        if self.config.reduce_lr:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,            verbose=1
            )
            callback_list.append(reduce_lr)
        
        # Model checkpoint
        if self.config.model_checkpoint:
            # Crear directorio para modelos si no existe
            models_dir = Path(__file__).parent.parent.parent / 'models'
            models_dir.mkdir(exist_ok=True)
            
            checkpoint = callbacks.ModelCheckpoint(
                filepath=str(models_dir / f'{model_name}_best.keras'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        # TensorBoard logging
        logs_dir = Path(__file__).parent.parent.parent / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        tensorboard = callbacks.TensorBoard(
            log_dir=str(logs_dir / f'tensorboard_{model_name}'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callback_list.append(tensorboard)
        
        return callback_list
    
    def create_custom_loss(self) -> callable:
        """
        Crear función de pérdida personalizada para trading.
        
        Returns:
            Función de pérdida personalizada
        """
        def trading_loss(y_true, y_pred):
            """
            Pérdida personalizada que penaliza más los errores direccionales.
            """
            # Pérdida MSE estándar
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            
            # Penalización por error direccional
            y_true_direction = tf.sign(y_true[:, 1])  # Dirección real
            y_pred_direction = tf.sign(y_pred[:, 1])  # Dirección predicha
            
            direction_error = tf.cast(
                tf.not_equal(y_true_direction, y_pred_direction),
                tf.float32
            )
            
            # Pérdida combinada
            total_loss = mse_loss + 0.5 * tf.reduce_mean(direction_error)
            
            return total_loss
        return trading_loss
    
    def get_model_summary(self, model: keras.Model) -> Dict[str, Any]:
        """
        Obtener resumen detallado del modelo.
        
        Args:
            model: Modelo a analizar
            
        Returns:
            Diccionario con información del modelo
        """
        # Capturar summary en string
        summary_string = []
        model.summary(print_fn=lambda x, line_break=None: summary_string.append(x))
        
        # Información básica
        summary_info = {
            'name': model.name,
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
            'layers': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'summary': '\n'.join(summary_string)
        }
        
        return summary_info


def create_lstm_model(config: LSTMConfig = None, model_type: str = 'basic') -> keras.Model:
    """
    Función de conveniencia para crear modelos LSTM.
    
    Args:
        config: Configuración del modelo
        model_type: Tipo de modelo ('basic', 'advanced', 'ensemble')
        
    Returns:
        Modelo LSTM configurado
    """
    architect = LSTMArchitect(config)
    
    if model_type == 'basic':
        return architect.build_basic_model()
    elif model_type == 'advanced':
        return architect.build_advanced_model()
    elif model_type == 'ensemble':
        return architect.build_ensemble_model()
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")


def validate_model_architecture(model: keras.Model, 
                               input_shape: Tuple[int, int]) -> bool:
    """
    Validar que la arquitectura del modelo sea correcta.
    
    Args:
        model: Modelo a validar
        input_shape: Forma esperada de entrada (sequence_length, n_features)
        
    Returns:
        True si la arquitectura es válida
    """
    try:
        # Verificar input shape
        expected_input = (None, input_shape[0], input_shape[1])
        actual_input = model.input_shape
        
        if actual_input != expected_input:
            logger.error(f"Input shape incorrecto: esperado {expected_input}, actual {actual_input}")
            return False
        
        # Verificar que el modelo esté compilado
        if not model.optimizer:
            logger.error("Modelo no compilado")
            return False
        
        # Verificar output shape
        if model.output_shape[-1] != 2:  # Debe predecir 2 valores
            logger.error(f"Output shape incorrecto: {model.output_shape}")
            return False
        
        # Prueba de forward pass
        test_input = np.random.random((1, input_shape[0], input_shape[1]))
        predictions = model.predict(test_input, verbose=0)
        
        if predictions.shape != (1, 2):
            logger.error(f"Predicción con shape incorrecto: {predictions.shape}")
            return False
        
        logger.info("✅ Arquitectura del modelo validada correctamente")
        return True
        
    except Exception as e:
        logger.error(f"Error validando arquitectura: {e}")
        return False


if __name__ == "__main__":
    # Ejemplo de uso
    print("🏗️ DISEÑO DE ARQUITECTURA LSTM - GuruInversor")
    print("=" * 60)
    
    # Configuración básica
    config = LSTMConfig(
        sequence_length=60,
        n_features=12,
        lstm_units=[50, 30],
        dense_units=[25, 10]
    )
    
    # Crear arquitecto
    architect = LSTMArchitect(config)
    
    # Construir modelo básico
    print("\n🔧 Construyendo modelo básico...")
    basic_model = architect.build_basic_model()
    
    # Mostrar resumen
    summary = architect.get_model_summary(basic_model)
    print(f"✅ Modelo creado: {summary['name']}")
    print(f"   📊 Parámetros totales: {summary['total_params']:,}")
    print(f"   🔧 Parámetros entrenables: {summary['trainable_params']:,}")
    print(f"   📏 Capas: {summary['layers']}")
    
    # Validar arquitectura
    print("\n🧪 Validando arquitectura...")
    is_valid = validate_model_architecture(basic_model, (60, 12))
    
    if is_valid:
        print("🎉 ¡Arquitectura LSTM diseñada y validada exitosamente!")
    else:
        print("❌ Error en validación de arquitectura")