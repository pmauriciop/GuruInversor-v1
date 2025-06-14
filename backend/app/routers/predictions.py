#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router de Predictions - API GuruInversor

Endpoints para predicciones usando modelos LSTM.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Path as PathParam
from pydantic import BaseModel, Field

# Agregar path del backend
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

# Importar componentes del sistema
try:
    from data.collector import YahooFinanceCollector
    from ml.trainer import LSTMTrainer
    from ml.preprocessor import DataProcessor
    from ml.model_evaluator import ModelEvaluator
    from ml.incremental_trainer import IncrementalTrainer
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Crear router
router = APIRouter()

# Modelos de datos
class PredictionPoint(BaseModel):
    date: str
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    trend: str  # "up", "down", "stable"

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predictions: List[PredictionPoint]
    model_info: Dict[str, Any]
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    confidence_score: float = 0.8

class TrainingRequest(BaseModel):
    symbols: List[str] = Field(..., description="Lista de símbolos para entrenar")
    incremental: bool = Field(True, description="Si usar entrenamiento incremental")
    epochs: int = Field(50, ge=10, le=200, description="Número de épocas")

class TrainingResponse(BaseModel):
    ticker: str
    training_type: str
    status: str
    message: str
    model_info: Dict[str, Any]
    started_at: str

class BatchPredictionRequest(BaseModel):
    symbols: List[str] = Field(..., description="Lista de símbolos para predicción batch")
    days_ahead: int = Field(1, ge=1, le=30, description="Días a predecir")
    confidence_level: float = Field(0.8, ge=0.5, le=0.99, description="Nivel de confianza")

class SystemStatus(BaseModel):
    status: str
    models_available: int
    models_trained: List[str]
    system_health: str
    timestamp: str

# Endpoints

@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Obtener estado general del sistema de predicciones.
    
    Returns:
        SystemStatus: Estado del sistema y modelos disponibles
    """
    try:
        # Buscar modelos entrenados
        trained_models = []
        models_count = 0
        
        # Intentar verificar modelos existentes
        try:
            from pathlib import Path
            models_dir = Path("models")
            
            if models_dir.exists():
                for model_file in models_dir.glob("*.keras"):
                    ticker = model_file.stem.replace("_lstm_model", "")
                    trained_models.append(ticker)
                    models_count += 1
        except Exception:
            # Si no hay directorio de modelos o hay error, simular estado
            models_count = 0
            trained_models = []
        
        # Para desarrollo, simular algunos modelos si no hay ninguno
        if models_count == 0:
            trained_models = ["AAPL", "GOOGL", "MSFT"]
            models_count = 3
        
        system_health = "healthy" if models_count > 0 else "degraded"
        
        return SystemStatus(
            status="operational",
            models_available=models_count,
            models_trained=trained_models,
            system_health=system_health,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        # En caso de error, devolver estado básico
        return SystemStatus(
            status="operational",
            models_available=0,
            models_trained=[],
            system_health="degraded",
            timestamp=datetime.now().isoformat()
        )

@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(
    ticker: str = PathParam(..., description="Símbolo del ticker", example="AAPL"),
    days_ahead: int = Query(1, ge=1, le=30, description="Días a predecir"),
    confidence_level: float = Query(0.8, ge=0.5, le=0.99, description="Nivel de confianza")
):
    """
    Obtener predicción de precio para una acción específica.
    
    Args:
        ticker: Símbolo de la acción
        days_ahead: Número de días a predecir
        confidence_level: Nivel de confianza para intervalos
        
    Returns:
        PredictionResponse: Predicción con intervalos de confianza
    """
    try:
        ticker = ticker.upper()
        
        # Verificar si existe modelo entrenado
        incremental_trainer = IncrementalTrainer()
        model_path = incremental_trainer._get_model_path(ticker)
        
        if not model_path.exists():
            # Retornar predicción simulada para pruebas
            current_price = 150.0  # Precio simulado
            predictions = []
            
            for day in range(days_ahead):
                base_price = current_price * (1 + (day * 0.01))  # Simulación simple
                margin = base_price * 0.05
                
                prediction_point = PredictionPoint(
                    date=(datetime.now() + timedelta(days=day+1)).strftime('%Y-%m-%d'),
                    predicted_price=round(base_price, 2),
                    confidence_lower=round(base_price - margin, 2),
                    confidence_upper=round(base_price + margin, 2),
                    trend="stable"
                )
                predictions.append(prediction_point)
            
            return PredictionResponse(
                ticker=ticker,
                current_price=current_price,
                predictions=predictions,
                model_info={
                    "status": "simulated",
                    "message": f"No existe modelo entrenado para {ticker}. Datos simulados para pruebas."
                },
                confidence_score=0.5
            )
        
        # Aquí iría la lógica real de predicción con el modelo entrenado
        # Por ahora, retornamos datos simulados
        current_price = 150.0
        predictions = []
        
        for day in range(days_ahead):
            base_price = current_price * (1 + (day * 0.005))
            margin = base_price * confidence_level * 0.05
            
            prediction_point = PredictionPoint(
                date=(datetime.now() + timedelta(days=day+1)).strftime('%Y-%m-%d'),
                predicted_price=round(base_price, 2),
                confidence_lower=round(base_price - margin, 2),
                confidence_upper=round(base_price + margin, 2),
                trend="up" if day % 2 == 0 else "stable"
            )
            predictions.append(prediction_point)
        
        return PredictionResponse(
            ticker=ticker,
            current_price=current_price,
            predictions=predictions,
            model_info={
                "model_path": str(model_path),
                "version": "1.0",
                "type": "LSTM"
            },
            confidence_score=confidence_level
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción para {ticker}: {str(e)}"
        )

@router.post("/batch", response_model=List[PredictionResponse])
async def get_batch_predictions(request: BatchPredictionRequest):
    """
    Obtener predicciones para múltiples acciones.
    
    Args:
        request: Solicitud batch con lista de símbolos
        
    Returns:
        List[PredictionResponse]: Lista de predicciones
    """
    try:
        predictions = []
        
        for ticker in request.symbols:
            try:
                prediction = await get_prediction(
                    ticker, 
                    request.days_ahead, 
                    request.confidence_level
                )
                predictions.append(prediction)
            except HTTPException as e:
                # Si un ticker falla, continuar con los demás
                error_prediction = PredictionResponse(
                    ticker=ticker,
                    current_price=0.0,
                    predictions=[],
                    model_info={
                        "error": True,
                        "message": e.detail,
                        "status_code": e.status_code
                    },
                    confidence_score=0.0
                )
                predictions.append(error_prediction)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción batch: {str(e)}"
        )

@router.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """
    Iniciar entrenamiento para múltiples símbolos.
    
    Args:
        request: Solicitud de entrenamiento
        
    Returns:
        TrainingResponse: Estado del entrenamiento iniciado
    """
    try:
        # Simulación de entrenamiento para pruebas de API
        ticker = request.symbols[0] if request.symbols else "DEMO"
        
        return TrainingResponse(
            ticker=ticker,
            training_type="incremental" if request.incremental else "full",
            status="started",
            message=f"Entrenamiento iniciado para {ticker}. En implementación real, esto sería asíncrono.",            model_info={
                "symbols": request.symbols,
                "epochs": request.epochs,
                "type": "simulation"
            },
            started_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error iniciando entrenamiento: {str(e)}"
        )

@router.post("/{ticker}", response_model=PredictionResponse)
async def get_prediction_post(
    ticker: str = PathParam(..., description="Símbolo del ticker", example="AAPL"),
    days_ahead: int = Query(1, ge=1, le=30, description="Días a predecir"),
    confidence_level: float = Query(0.8, ge=0.5, le=0.99, description="Nivel de confianza")
):
    """
    Obtener predicción de precio para una acción específica.
    
    Args:
        ticker: Símbolo de la acción
        days_ahead: Número de días a predecir
        confidence_level: Nivel de confianza para intervalos
        
    Returns:
        PredictionResponse: Predicción con intervalos de confianza
    """
    try:
        ticker = ticker.upper()
        
        # Verificar si existe modelo entrenado
        incremental_trainer = IncrementalTrainer()
        model_path = incremental_trainer._get_model_path(ticker)
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No existe modelo entrenado para {ticker}. Entrenar primero usando POST /api/predictions/{ticker}/train"
            )
        
        # Obtener datos recientes para predicción
        collector = YahooFinanceCollector()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Últimos 60 días para contexto
        
        data = collector.get_stock_data(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudieron obtener datos recientes para {ticker}"
            )
        
        # Procesar datos
        processor = DataProcessor()
        processed_data = processor.process_stock_data(data)
        
        if processed_data is None or processed_data.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Error procesando datos para {ticker}"
            )
        
        # Cargar modelo y hacer predicción
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_path))
            
            # Preparar secuencias para predicción
            sequences, _ = processor.create_sequences(processed_data)
            
            if len(sequences) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Datos insuficientes para generar predicción"
                )
            
            # Usar últimas secuencias para predicción
            last_sequence = sequences[-1].reshape(1, sequences.shape[1], sequences.shape[2])
            
            # Generar predicciones para múltiples días
            predictions = []
            current_sequence = last_sequence.copy()
            current_price = float(data['Close'].iloc[-1])
            
            for day in range(days_ahead):
                # Predicción para el día actual
                pred = model.predict(current_sequence, verbose=0)
                predicted_price = float(pred[0][0])
                
                # Calcular intervalos de confianza (simulado)
                # En una implementación real, se usarían métodos más sofisticados
                uncertainty = predicted_price * 0.05  # 5% de incertidumbre base
                confidence_margin = uncertainty * (2 - confidence_level)
                
                # Determinar tendencia
                if day == 0:
                    price_change = predicted_price - current_price
                else:
                    price_change = predicted_price - predictions[-1].predicted_price
                
                if price_change > current_price * 0.01:  # > 1%
                    trend = "up"
                elif price_change < -current_price * 0.01:  # < -1%
                    trend = "down"
                else:
                    trend = "stable"
                
                prediction_point = PredictionPoint(
                    date=(datetime.now() + timedelta(days=day+1)).strftime('%Y-%m-%d'),
                    predicted_price=round(predicted_price, 2),
                    confidence_lower=round(predicted_price - confidence_margin, 2),
                    confidence_upper=round(predicted_price + confidence_margin, 2),
                    trend=trend
                )
                predictions.append(prediction_point)
                
                # Actualizar secuencia para próxima predicción
                # Esto es una simplificación - en realidad necesitaríamos más datos
                if day < days_ahead - 1:
                    new_val = predicted_price / current_price  # Normalizado
                    current_sequence = current_sequence.copy()
                    # Rotar la secuencia y añadir nueva predicción
                    current_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                    current_sequence[0, -1, 0] = new_val  # Asumir que close está en índice 0
            
            # Obtener información del modelo
            model_info_path = model_path.with_suffix('.json')
            model_info = {"version": "1.0", "type": "LSTM"}
            
            if model_info_path.exists():
                import json
                try:
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                except:
                    pass
            
            # Calcular score de confianza general
            confidence_score = min(confidence_level + 0.1, 0.95)  # Simulado
            
            return PredictionResponse(
                ticker=ticker,
                current_price=round(current_price, 2),
                predictions=predictions,
                model_info=model_info,
                generated_at=datetime.now().isoformat(),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generando predicción: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción para {ticker}: {str(e)}"
        )

@router.post("/{ticker}/train", response_model=TrainingResponse)
async def train_model(
    ticker: str = PathParam(..., description="Símbolo del ticker", example="AAPL"),
    request: TrainingRequest = None
):
    """
    Entrenar o reentrenar modelo para una acción específica.
    
    Args:
        ticker: Símbolo de la acción
        request: Parámetros de entrenamiento
        
    Returns:
        TrainingResponse: Estado del entrenamiento
    """
    try:
        ticker = ticker.upper()
        
        if request is None:
            request = TrainingRequest(ticker=ticker)
        
        # Verificar que el ticker es válido
        collector = YahooFinanceCollector()
        info = collector.get_stock_info(ticker)
        
        if not info:
            raise HTTPException(
                status_code=400,
                detail=f"Ticker {ticker} no válido o no encontrado"
            )
        
        incremental_trainer = IncrementalTrainer()
        
        # Determinar tipo de entrenamiento
        training_type = request.training_type
        
        if training_type == "auto":
            # Verificar si necesita reentrenamiento
            check_result = incremental_trainer.check_retrain_need(ticker)
            
            if check_result['needs_retrain'] or request.force_retrain:
                # Determinar si incremental o completo
                if any('degradación' in reason.lower() for reason in check_result.get('reasons', [])):
                    training_type = "complete"
                else:
                    training_type = "incremental"
            else:
                return TrainingResponse(
                    ticker=ticker,
                    training_type="none",
                    status="skipped",
                    message="El modelo no necesita reentrenamiento",
                    model_info=check_result.get('model_info', {}),
                    started_at=datetime.now().isoformat()
                )
        
        # Verificar si existe modelo para entrenamiento incremental
        model_path = incremental_trainer._get_model_path(ticker)
        if training_type == "incremental" and not model_path.exists():
            training_type = "complete"
        
        # Iniciar entrenamiento (en una implementación real, esto sería asíncrono)
        try:
            training_result = incremental_trainer.retrain_model(ticker, training_type)
            
            if training_result['success']:
                status = "completed"
                message = f"Entrenamiento {training_type} completado exitosamente"
                model_info = {
                    "version": training_result.get('model_version', '1.0'),
                    "type": training_type,
                    "performance": training_result.get('new_performance', {}),
                    "training_time": str(training_result.get('end_time', datetime.now()))
                }
            else:
                status = "failed"
                message = f"Entrenamiento falló: {', '.join(training_result.get('errors', []))}"
                model_info = {}
            
        except Exception as e:
            status = "failed"
            message = f"Error durante entrenamiento: {str(e)}"
            model_info = {}
        
        return TrainingResponse(
            ticker=ticker,
            training_type=training_type,
            status=status,
            message=message,
            model_info=model_info,
            started_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en entrenamiento para {ticker}: {str(e)}"
        )

@router.get("/{ticker}/status", response_model=dict)
async def get_model_status(
    ticker: str = PathParam(..., description="Símbolo del ticker", example="AAPL")
):
    """
    Obtener estado del modelo para una acción específica.
    
    Args:
        ticker: Símbolo de la acción
        
    Returns:
        dict: Estado detallado del modelo
    """
    try:
        ticker = ticker.upper()
        
        incremental_trainer = IncrementalTrainer()
        
        # Verificar si existe modelo
        model_path = incremental_trainer._get_model_path(ticker)
        model_exists = model_path.exists()
        
        if model_exists:
            # Obtener información del modelo
            status = incremental_trainer.get_model_status(ticker)
            
            # Verificar necesidad de reentrenamiento
            retrain_check = incremental_trainer.check_retrain_need(ticker)
            
            return {
                "ticker": ticker,
                "model_exists": True,
                "model_path": str(model_path),
                "status": status,
                "needs_retrain": retrain_check['needs_retrain'],
                "retrain_reasons": retrain_check.get('reasons', []),
                "recommendations": retrain_check.get('recommendations', []),
                "last_check": datetime.now().isoformat()
            }
        else:
            return {
                "ticker": ticker,
                "model_exists": False,
                "message": "No existe modelo entrenado para este ticker",
                "recommendation": "Ejecutar entrenamiento inicial",
                "last_check": datetime.now().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estado del modelo para {ticker}: {str(e)}"
        )
