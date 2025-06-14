# -*- coding: utf-8 -*-
"""
Modelos de Base de Datos - GuruInversor

Define los modelos SQLAlchemy para todas las tablas de la base de datos.
Incluye validaciones y relaciones entre entidades.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
import re

Base = declarative_base()


class Stock(Base):
    """
    Modelo para la tabla de acciones.
    
    Almacena información básica de las acciones que se están monitoreando.
    """
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200))
    sector = Column(String(100))
    market = Column(String(50), default='US')  # Mercado (US, MX, etc.)
    currency = Column(String(3), default='USD')  # Moneda
    active = Column(Boolean, default=True)  # Si está activa para monitoreo
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    historical_data = relationship("HistoricalData", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")
    trained_models = relationship("TrainedModel", back_populates="stock", cascade="all, delete-orphan")
    
    @validates('ticker')
    def validate_ticker(self, key, value):
        """Valida que el ticker tenga formato correcto."""
        if not value:
            raise ValueError("Ticker no puede estar vacío")
        
        # Convertir a mayúsculas y limpiar espacios
        value = str(value).strip().upper()
        
        # Validar formato básico (letras, números, puntos, guiones)
        if not re.match(r'^[A-Z0-9.\-]{1,20}$', value):
            raise ValueError(f"Ticker '{value}' tiene formato inválido")
        
        return value
    
    def __repr__(self):
        return f"<Stock(ticker='{self.ticker}', name='{self.name}')>"
    
    def to_dict(self):
        """Convierte el objeto a diccionario para serialización."""
        return {
            'id': self.id,
            'ticker': self.ticker,
            'name': self.name,
            'sector': self.sector,
            'market': self.market,
            'currency': self.currency,
            'active': self.active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class HistoricalData(Base):
    """
    Modelo para la tabla de datos históricos.
    
    Almacena los datos OHLCV históricos de las acciones.
    """
    __tablename__ = 'historical_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adj_close = Column(Float)  # Precio ajustado por dividendos/splits
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relación
    stock = relationship("Stock", back_populates="historical_data")
    
    @validates('open', 'high', 'low', 'close', 'adj_close')
    def validate_prices(self, key, value):
        """Valida que los precios sean positivos."""
        if value is not None and value <= 0:
            raise ValueError(f"El precio {key} debe ser positivo, recibido: {value}")
        return value
    
    @validates('volume')
    def validate_volume(self, key, value):
        """Valida que el volumen sea no negativo."""
        if value is not None and value < 0:
            raise ValueError(f"El volumen debe ser no negativo, recibido: {value}")
        return value
    
    def __repr__(self):
        return f"<HistoricalData(stock_id={self.stock_id}, date='{self.date}', close={self.close})>"
    
    def to_dict(self):
        """Convierte el objeto a diccionario para serialización."""
        return {
            'id': self.id,
            'stock_id': self.stock_id,
            'date': self.date.isoformat() if self.date else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adj_close': self.adj_close,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Prediction(Base):
    """
    Modelo para la tabla de predicciones.
    
    Almacena las predicciones realizadas por los modelos de IA.
    """
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False, index=True)
    prediction_date = Column(Date, nullable=False, index=True)  # Fecha para la que se predice
    predicted_price = Column(Float, nullable=False)
    confidence = Column(Float)  # Nivel de confianza (0-1)
    actual_price = Column(Float)  # Precio real (se llena después)
    model_version = Column(String(50))  # Versión del modelo usado
    prediction_type = Column(String(20), default='close')  # Tipo de predicción (close, high, low)
    horizon_days = Column(Integer, default=1)  # Días hacia adelante de la predicción
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relación
    stock = relationship("Stock", back_populates="predictions")
    
    @validates('predicted_price', 'actual_price')
    def validate_prices(self, key, value):
        """Valida que los precios sean positivos."""
        if value is not None and value <= 0:
            raise ValueError(f"El precio {key} debe ser positivo, recibido: {value}")
        return value
    
    @validates('confidence')
    def validate_confidence(self, key, value):
        """Valida que la confianza esté entre 0 y 1."""
        if value is not None and (value < 0 or value > 1):
            raise ValueError(f"La confianza debe estar entre 0 y 1, recibido: {value}")
        return value
    
    def __repr__(self):
        return f"<Prediction(stock_id={self.stock_id}, date='{self.prediction_date}', price={self.predicted_price})>"
    
    def to_dict(self):
        """Convierte el objeto a diccionario para serialización."""
        return {
            'id': self.id,
            'stock_id': self.stock_id,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'actual_price': self.actual_price,
            'model_version': self.model_version,
            'prediction_type': self.prediction_type,
            'horizon_days': self.horizon_days,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TrainedModel(Base):
    """
    Modelo para la tabla de modelos entrenados.
    
    Almacena metadatos de los modelos de IA entrenados.
    """
    __tablename__ = 'trained_models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False, index=True)
    model_path = Column(Text, nullable=False)  # Ruta al archivo del modelo
    version = Column(String(50), nullable=False)  # Versión del modelo
    model_type = Column(String(50), default='LSTM')  # Tipo de modelo
    accuracy = Column(Float)  # Precisión en conjunto de prueba
    loss = Column(Float)  # Pérdida en conjunto de prueba
    rmse = Column(Float)  # Root Mean Square Error
    mae = Column(Float)  # Mean Absolute Error
    mape = Column(Float)  # Mean Absolute Percentage Error
    training_samples = Column(Integer)  # Número de muestras de entrenamiento
    validation_samples = Column(Integer)  # Número de muestras de validación
    epochs_trained = Column(Integer)  # Épocas entrenadas
    training_time_seconds = Column(Float)  # Tiempo de entrenamiento en segundos
    hyperparameters = Column(Text)  # Hiperparámetros en JSON
    is_active = Column(Boolean, default=True)  # Si es el modelo activo para esta acción
    training_date = Column(DateTime, default=datetime.utcnow)
    
    # Relación
    stock = relationship("Stock", back_populates="trained_models")
    
    @validates('accuracy', 'loss', 'rmse', 'mae', 'mape')
    def validate_metrics(self, key, value):
        """Valida que las métricas sean no negativas."""
        if value is not None and value < 0:
            raise ValueError(f"La métrica {key} debe ser no negativa, recibido: {value}")
        return value
    
    @validates('training_samples', 'validation_samples', 'epochs_trained')
    def validate_counts(self, key, value):
        """Valida que los contadores sean positivos."""
        if value is not None and value <= 0:
            raise ValueError(f"El contador {key} debe ser positivo, recibido: {value}")
        return value
    
    def __repr__(self):
        return f"<TrainedModel(stock_id={self.stock_id}, version='{self.version}', accuracy={self.accuracy})>"
    
    def to_dict(self):
        """Convierte el objeto a diccionario para serialización."""
        return {
            'id': self.id,
            'stock_id': self.stock_id,
            'model_path': self.model_path,
            'version': self.version,
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'epochs_trained': self.epochs_trained,
            'training_time_seconds': self.training_time_seconds,
            'hyperparameters': self.hyperparameters,
            'is_active': self.is_active,
            'training_date': self.training_date.isoformat() if self.training_date else None
        }
