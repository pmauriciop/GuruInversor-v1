#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades de Procesamiento de Datos - GuruInversor

Módulo principal con funciones para procesamiento, normalización y
preparación de datos para el modelo LSTM.

Funcionalidades:
- Normalización de precios OHLCV
- Feature engineering (indicadores técnicos)
- Preparación de secuencias temporales
- Validación y limpieza de datos
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuración para procesamiento de datos."""
    sequence_length: int = 60  # Días de historia para LSTM
    features: List[str] = None  # Características a usar
    normalize_method: str = 'minmax'  # 'minmax', 'zscore', 'robust'
    fill_method: str = 'forward'  # Método para llenar valores faltantes
    validation_strict: bool = True  # Validación estricta de datos
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['open', 'high', 'low', 'close', 'volume']


class DataProcessor:
    """
    Procesador principal de datos para preparación de entrenamiento de modelos.
    
    Maneja normalización, feature engineering y validación de datos históricos.
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.scalers = {}  # Almacenar escaladores para desnormalización
        self.stats = {}    # Estadísticas de los datos
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validar integridad y calidad de datos históricos.
        
        Args:
            df: DataFrame con datos históricos (columns: date, open, high, low, close, volume)
            
        Returns:
            Tuple[bool, List[str]]: (es_válido, lista_de_errores)
        """
        errors = []
        
        # Verificar columnas requeridas
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Columnas faltantes: {missing_columns}")
            return False, errors
        
        # Verificar que no haya valores nulos en columnas críticas
        null_counts = df[['open', 'high', 'low', 'close']].isnull().sum()
        if null_counts.any():
            errors.append(f"Valores nulos encontrados: {null_counts.to_dict()}")
        
        # Verificar precios válidos (positivos)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                errors.append(f"Precios no positivos en columna: {col}")
        
        # Verificar lógica OHLC
        if self.config.validation_strict:
            # High debe ser >= que Open, Low, Close
            if (df['high'] < df[['open', 'low', 'close']].max(axis=1)).any():
                errors.append("Valores 'high' inconsistentes con OHLC")
            
            # Low debe ser <= que Open, High, Close  
            if (df['low'] > df[['open', 'high', 'close']].min(axis=1)).any():
                errors.append("Valores 'low' inconsistentes con OHLC")
        
        # Verificar orden cronológico
        if not df['date'].is_monotonic_increasing:
            errors.append("Datos no están en orden cronológico")
        
        # Verificar gaps grandes en fechas
        df_sorted = df.sort_values('date')
        date_diffs = df_sorted['date'].diff().dropna()
        max_gap = date_diffs.max().days
        if max_gap > 7:  # Más de una semana
            logger.warning(f"Gap máximo entre fechas: {max_gap} días")
        
        is_valid = len(errors) == 0
        if not is_valid:
            logger.error(f"Validación de datos fallida: {errors}")
        else:
            logger.info("Validación de datos exitosa")
            
        return is_valid, errors
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar y preparar datos para procesamiento.
        
        Args:
            df: DataFrame con datos históricos
            
        Returns:
            DataFrame limpio y preparado
        """
        logger.info("Iniciando limpieza de datos...")
        
        # Crear copia para no modificar original
        df_clean = df.copy()
        
        # Asegurar que 'date' sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean['date']):
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Ordenar por fecha
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        # Eliminar duplicados por fecha
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['date'], keep='last')
        if len(df_clean) < initial_count:
            logger.warning(f"Eliminadas {initial_count - len(df_clean)} filas duplicadas")
          # Manejar valores faltantes
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if self.config.fill_method == 'forward':
            df_clean[numeric_columns] = df_clean[numeric_columns].ffill()
        elif self.config.fill_method == 'interpolate':
            df_clean[numeric_columns] = df_clean[numeric_columns].interpolate()
        elif self.config.fill_method == 'drop':
            df_clean = df_clean.dropna(subset=numeric_columns)
        
        # Eliminar outliers extremos (precios que cambian más del 50% en un día)
        price_changes = df_clean['close'].pct_change().abs()
        outlier_threshold = 0.5
        outliers = price_changes > outlier_threshold
        if outliers.any():
            logger.warning(f"Encontrados {outliers.sum()} outliers de precio (cambios >{outlier_threshold*100}%)")
            # No eliminar, solo registrar
        
        logger.info(f"Limpieza completada. Registros finales: {len(df_clean)}")
        return df_clean
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agregar indicadores técnicos básicos como features adicionales.
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con indicadores técnicos agregados
        """
        logger.info("Agregando indicadores técnicos...")
        
        df_features = df.copy()
        
        # Medias móviles
        df_features['sma_10'] = df_features['close'].rolling(window=10).mean()
        df_features['sma_20'] = df_features['close'].rolling(window=20).mean()
        df_features['sma_50'] = df_features['close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df_features['ema_12'] = df_features['close'].ewm(span=12).mean()
        df_features['ema_26'] = df_features['close'].ewm(span=26).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df_features['macd'] = df_features['ema_12'] - df_features['ema_26']
        df_features['macd_signal'] = df_features['macd'].ewm(span=9).mean()
        df_features['macd_histogram'] = df_features['macd'] - df_features['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = df_features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = df_features['close'].rolling(window=bb_period).std()
        df_features['bb_middle'] = df_features['close'].rolling(window=bb_period).mean()
        df_features['bb_upper'] = df_features['bb_middle'] + (bb_std * 2)
        df_features['bb_lower'] = df_features['bb_middle'] - (bb_std * 2)
        df_features['bb_width'] = df_features['bb_upper'] - df_features['bb_lower']
        df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / df_features['bb_width']
        
        # Volatilidad (desviación estándar de returns)
        df_features['returns'] = df_features['close'].pct_change()
        df_features['volatility'] = df_features['returns'].rolling(window=20).std()
        
        # Volume features
        df_features['volume_sma'] = df_features['volume'].rolling(window=20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
        
        # Price momentum features
        df_features['price_change_1d'] = df_features['close'].pct_change(1)
        df_features['price_change_5d'] = df_features['close'].pct_change(5)
        df_features['price_change_20d'] = df_features['close'].pct_change(20)
        
        # High/Low ratios
        df_features['hl_ratio'] = df_features['high'] / df_features['low']
        df_features['oc_ratio'] = df_features['open'] / df_features['close']
        
        logger.info(f"Indicadores técnicos agregados. Nuevas columnas: {len(df_features.columns) - len(df.columns)}")
        return df_features
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalizar datos numéricos para entrenamiento del modelo.
        
        Args:
            df: DataFrame con datos a normalizar
            fit: Si True, ajusta los escaladores. Si False, usa escaladores existentes.
            
        Returns:
            DataFrame normalizado
        """
        logger.info(f"Normalizando datos usando método: {self.config.normalize_method}")
        
        df_norm = df.copy()
        numeric_columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir columna de fecha si existe
        if 'date' in numeric_columns:
            numeric_columns.remove('date')
        
        for col in numeric_columns:
            if fit:
                if self.config.normalize_method == 'minmax':
                    # Min-Max scaling [0, 1]
                    col_min = df_norm[col].min()
                    col_max = df_norm[col].max()
                    self.scalers[col] = {'min': col_min, 'max': col_max, 'method': 'minmax'}
                    df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
                    
                elif self.config.normalize_method == 'zscore':
                    # Z-score normalization
                    col_mean = df_norm[col].mean()
                    col_std = df_norm[col].std()
                    self.scalers[col] = {'mean': col_mean, 'std': col_std, 'method': 'zscore'}
                    df_norm[col] = (df_norm[col] - col_mean) / col_std
                    
                elif self.config.normalize_method == 'robust':
                    # Robust scaling using median and IQR
                    col_median = df_norm[col].median()
                    col_q75 = df_norm[col].quantile(0.75)
                    col_q25 = df_norm[col].quantile(0.25)
                    col_iqr = col_q75 - col_q25
                    self.scalers[col] = {'median': col_median, 'iqr': col_iqr, 'method': 'robust'}
                    df_norm[col] = (df_norm[col] - col_median) / col_iqr
                    
            else:
                # Usar escaladores existentes
                if col in self.scalers:
                    scaler = self.scalers[col]
                    if scaler['method'] == 'minmax':
                        df_norm[col] = (df_norm[col] - scaler['min']) / (scaler['max'] - scaler['min'])
                    elif scaler['method'] == 'zscore':
                        df_norm[col] = (df_norm[col] - scaler['mean']) / scaler['std']
                    elif scaler['method'] == 'robust':
                        df_norm[col] = (df_norm[col] - scaler['median']) / scaler['iqr']
                else:
                    logger.warning(f"No se encontró escalador para columna: {col}")
        
        return df_norm
    
    def denormalize_predictions(self, predictions: np.ndarray, target_column: str = 'close') -> np.ndarray:
        """
        Desnormalizar predicciones para obtener valores reales.
        
        Args:
            predictions: Array con predicciones normalizadas
            target_column: Nombre de la columna objetivo (por defecto 'close')
            
        Returns:
            Array con predicciones desnormalizadas
        """
        if target_column not in self.scalers:
            logger.error(f"No se encontró escalador para columna: {target_column}")
            return predictions
        
        scaler = self.scalers[target_column]
        
        if scaler['method'] == 'minmax':
            return predictions * (scaler['max'] - scaler['min']) + scaler['min']
        elif scaler['method'] == 'zscore':
            return predictions * scaler['std'] + scaler['mean']
        elif scaler['method'] == 'robust':
            return predictions * scaler['iqr'] + scaler['median']
        
        return predictions
    
    def create_sequences(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear secuencias temporales para entrenamiento LSTM.
        
        Args:
            df: DataFrame normalizado con features
            target_column: Columna objetivo para predicción
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_sequences, y_targets)
        """
        logger.info(f"Creando secuencias temporales de longitud: {self.config.sequence_length}")
        
        # Seleccionar solo columnas de features especificadas
        feature_columns = [col for col in self.config.features if col in df.columns]
        if not feature_columns:
            logger.error("No se encontraron columnas de features válidas")
            return np.array([]), np.array([])
        
        # Agregar columnas adicionales si existen
        additional_features = ['sma_10', 'sma_20', 'rsi', 'macd', 'volatility', 'volume_ratio']
        for feat in additional_features:
            if feat in df.columns and feat not in feature_columns:
                feature_columns.append(feat)
        
        # Eliminar filas con NaN que podrían haber resultado de indicadores técnicos
        df_clean = df[feature_columns + [target_column]].dropna()
        
        if len(df_clean) < self.config.sequence_length + 1:
            logger.error(f"Datos insuficientes para crear secuencias. Necesarios: {self.config.sequence_length + 1}, disponibles: {len(df_clean)}")
            return np.array([]), np.array([])
        
        # Crear secuencias
        X_sequences = []
        y_targets = []
        
        for i in range(self.config.sequence_length, len(df_clean)):
            # Secuencia de features (ventana deslizante)
            sequence = df_clean[feature_columns].iloc[i - self.config.sequence_length:i].values
            X_sequences.append(sequence)
            
            # Target (precio de cierre del día siguiente)
            target = df_clean[target_column].iloc[i]
            y_targets.append(target)
        
        X_sequences = np.array(X_sequences)
        y_targets = np.array(y_targets)
        
        logger.info(f"Secuencias creadas - X: {X_sequences.shape}, y: {y_targets.shape}")
        return X_sequences, y_targets
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del procesamiento de datos.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'config': self.config.__dict__,
            'scalers': {col: {k: v for k, v in scaler.items() if k != 'method'} 
                       for col, scaler in self.scalers.items()},
            'feature_count': len(self.config.features),
            'sequence_length': self.config.sequence_length
        }


def process_stock_data(df: pd.DataFrame, 
                      config: ProcessingConfig = None,
                      add_features: bool = True) -> Tuple[np.ndarray, np.ndarray, DataProcessor]:
    """
    Función de conveniencia para procesar datos de acciones completamente.
    
    Args:
        df: DataFrame con datos históricos (date, open, high, low, close, volume)
        config: Configuración de procesamiento
        add_features: Si agregar indicadores técnicos
        
    Returns:
        Tuple[np.ndarray, np.ndarray, DataProcessor]: (X_sequences, y_targets, processor)
    """
    logger.info("Iniciando procesamiento completo de datos de acciones...")
    
    processor = DataProcessor(config)
    
    # Validar datos
    is_valid, errors = processor.validate_data(df)
    if not is_valid:
        logger.error(f"Datos inválidos: {errors}")
        return np.array([]), np.array([]), processor
    
    # Limpiar datos
    df_clean = processor.clean_data(df)
    
    # Agregar indicadores técnicos si se solicita
    if add_features:
        df_features = processor.add_technical_indicators(df_clean)
    else:
        df_features = df_clean
    
    # Normalizar datos
    df_normalized = processor.normalize_data(df_features, fit=True)
    
    # Crear secuencias
    X_sequences, y_targets = processor.create_sequences(df_normalized)
    
    logger.info("Procesamiento completo finalizado exitosamente")
    return X_sequences, y_targets, processor


if __name__ == "__main__":
    # Ejemplo de uso básico
    print("🚀 Utilidades de Procesamiento - GuruInversor")
    print("=" * 50)
    
    # Crear datos de ejemplo
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(150, 250, len(dates)),
        'low': np.random.uniform(50, 150, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Asegurar lógica OHLC
    sample_data['high'] = np.maximum.reduce([sample_data['open'], sample_data['high'], 
                                            sample_data['low'], sample_data['close']])
    sample_data['low'] = np.minimum.reduce([sample_data['open'], sample_data['high'], 
                                           sample_data['low'], sample_data['close']])
    
    print(f"📊 Datos de ejemplo creados: {len(sample_data)} registros")
    
    # Procesar datos
    X, y, processor = process_stock_data(sample_data)
    
    if len(X) > 0:
        print(f"✅ Procesamiento exitoso!")
        print(f"   - Secuencias X: {X.shape}")
        print(f"   - Objetivos y: {y.shape}")
        print(f"   - Features disponibles: {len(processor.config.features)}")
        
        stats = processor.get_processing_stats()
        print(f"📈 Estadísticas de procesamiento:")
        print(f"   - Método normalización: {stats['config']['normalize_method']}")
        print(f"   - Longitud secuencia: {stats['sequence_length']}")
        print(f"   - Escaladores creados: {len(stats['scalers'])}")
    else:
        print("❌ Error en el procesamiento")
