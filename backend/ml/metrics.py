#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades de Evaluación y Métricas - GuruInversor

Módulo con funciones para evaluar modelos de predicción de precios
y calcular métricas financieras relevantes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcular métricas de regresión estándar.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Diccionario con métricas calculadas
    """
    try:
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        
        # Añadir métricas adicionales
        metrics['mean_error'] = np.mean(y_pred - y_true)
        metrics['std_error'] = np.std(y_pred - y_true)
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculando métricas de regresión: {e}")
        return {}


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcular precisión direccional (si predice correctamente la dirección del cambio).
    
    Args:
        y_true: Precios reales
        y_pred: Precios predichos
        
    Returns:
        Diccionario con métricas direccionales
    """
    try:
        # Calcular cambios direccionales
        true_direction = np.diff(y_true) > 0  # True si sube, False si baja
        pred_direction = np.diff(y_pred) > 0
        
        # Precisión direccional
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        # Métricas por dirección
        up_mask = true_direction == True
        down_mask = true_direction == False
        
        up_accuracy = np.mean(pred_direction[up_mask] == True) if up_mask.any() else 0
        down_accuracy = np.mean(pred_direction[down_mask] == False) if down_mask.any() else 0
        
        return {
            'directional_accuracy': directional_accuracy,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'total_up_days': up_mask.sum(),
            'total_down_days': down_mask.sum()
        }
    except Exception as e:
        logger.error(f"Error calculando precisión direccional: {e}")
        return {}


def calculate_financial_metrics(prices: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Calcular métricas financieras específicas para trading.
    
    Args:
        prices: Serie de precios reales
        predictions: Serie de predicciones
        
    Returns:
        Diccionario con métricas financieras
    """
    try:
        # Calcular returns
        actual_returns = np.diff(prices) / prices[:-1]
        predicted_returns = np.diff(predictions) / predictions[:-1]
        
        # Volatilidad
        volatility_actual = np.std(actual_returns) * np.sqrt(252)  # Anualizada
        volatility_predicted = np.std(predicted_returns) * np.sqrt(252)
        
        # Sharpe ratio simulado (asumiendo rf = 0)
        mean_return_actual = np.mean(actual_returns) * 252  # Anualizado
        mean_return_predicted = np.mean(predicted_returns) * 252
        
        sharpe_actual = mean_return_actual / volatility_actual if volatility_actual > 0 else 0
        sharpe_predicted = mean_return_predicted / volatility_predicted if volatility_predicted > 0 else 0
        
        # Maximum Drawdown
        def max_drawdown(returns):
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            return np.max(drawdown)
        
        mdd_actual = max_drawdown(actual_returns)
        mdd_predicted = max_drawdown(predicted_returns)
        
        return {
            'volatility_actual': volatility_actual,
            'volatility_predicted': volatility_predicted,
            'sharpe_actual': sharpe_actual,
            'sharpe_predicted': sharpe_predicted,
            'max_drawdown_actual': mdd_actual,
            'max_drawdown_predicted': mdd_predicted,
            'return_correlation': np.corrcoef(actual_returns, predicted_returns)[0, 1]
        }
    except Exception as e:
        logger.error(f"Error calculando métricas financieras: {e}")
        return {}


def evaluate_model_performance(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             prices: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluación completa del rendimiento del modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        prices: Serie de precios originales (opcional, para métricas financieras)
        
    Returns:
        Diccionario completo con todas las métricas
    """
    logger.info("Calculando métricas de evaluación del modelo...")
    
    metrics = {}
    
    # Métricas de regresión
    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    metrics.update({f"regression_{k}": v for k, v in regression_metrics.items()})
    
    # Métricas direccionales
    directional_metrics = calculate_directional_accuracy(y_true, y_pred)
    metrics.update({f"directional_{k}": v for k, v in directional_metrics.items()})
    
    # Métricas financieras si se proporcionan precios
    if prices is not None:
        financial_metrics = calculate_financial_metrics(prices, y_pred)
        metrics.update({f"financial_{k}": v for k, v in financial_metrics.items()})
    
    # Métricas de error relativo
    relative_error = np.abs((y_true - y_pred) / y_true)
    metrics['mean_relative_error'] = np.mean(relative_error)
    metrics['median_relative_error'] = np.median(relative_error)
    
    logger.info(f"Evaluación completada. Métricas calculadas: {len(metrics)}")
    return metrics


def print_evaluation_report(metrics: Dict[str, float], model_name: str = "Modelo") -> None:
    """
    Imprimir un reporte formateado de las métricas de evaluación.
    
    Args:
        metrics: Diccionario con métricas calculadas
        model_name: Nombre del modelo para el reporte
    """
    print(f"\n📊 REPORTE DE EVALUACIÓN - {model_name}")
    print("=" * 60)
    
    # Métricas de regresión
    print("\n🎯 MÉTRICAS DE REGRESIÓN:")
    regression_keys = [k for k in metrics.keys() if k.startswith('regression_')]
    for key in regression_keys:
        metric_name = key.replace('regression_', '').upper()
        value = metrics[key]
        if 'mape' in key:
            print(f"   {metric_name:<15}: {value:.2f}%")
        else:
            print(f"   {metric_name:<15}: {value:.4f}")
    
    # Métricas direccionales
    print("\n📈 MÉTRICAS DIRECCIONALES:")
    directional_keys = [k for k in metrics.keys() if k.startswith('directional_')]
    for key in directional_keys:
        metric_name = key.replace('directional_', '').replace('_', ' ').title()
        value = metrics[key]
        if 'accuracy' in key:
            print(f"   {metric_name:<20}: {value:.2%}")
        else:
            print(f"   {metric_name:<20}: {value:.0f}")
    
    # Métricas financieras
    financial_keys = [k for k in metrics.keys() if k.startswith('financial_')]
    if financial_keys:
        print("\n💰 MÉTRICAS FINANCIERAS:")
        for key in financial_keys:
            metric_name = key.replace('financial_', '').replace('_', ' ').title()
            value = metrics[key]
            if 'volatility' in key or 'sharpe' in key:
                print(f"   {metric_name:<25}: {value:.4f}")
            elif 'drawdown' in key or 'correlation' in key:
                print(f"   {metric_name:<25}: {value:.4f}")
    
    # Métricas de error
    if 'mean_relative_error' in metrics:
        print(f"\n🎯 ERROR RELATIVO PROMEDIO: {metrics['mean_relative_error']:.2%}")
    if 'median_relative_error' in metrics:
        print(f"🎯 ERROR RELATIVO MEDIANO: {metrics['median_relative_error']:.2%}")
    
    print("=" * 60)


def validate_trading_strategy(predictions: np.ndarray, 
                            actual_prices: np.ndarray,
                            threshold: float = 0.01) -> Dict[str, float]:
    """
    Validar una estrategia de trading simple basada en predicciones.
    
    Args:
        predictions: Predicciones de precios
        actual_prices: Precios reales
        threshold: Umbral mínimo de cambio para generar señal
        
    Returns:
        Diccionario con métricas de la estrategia
    """
    try:
        # Calcular señales de trading
        price_changes = np.diff(predictions) / predictions[:-1]
        signals = np.where(price_changes > threshold, 1,  # Comprar
                          np.where(price_changes < -threshold, -1, 0))  # Vender/Mantener
        
        # Calcular returns de la estrategia
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]
        strategy_returns = signals * actual_returns
        
        # Métricas de la estrategia
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        # Estadísticas de trading
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        profitable_trades = np.sum(strategy_returns > 0)
        total_trades = np.sum(signals != 0)
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_return_per_trade': np.mean(strategy_returns[signals != 0]) if total_trades > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error validando estrategia de trading: {e}")
        return {}


if __name__ == "__main__":
    # Ejemplo de uso
    print("📊 Utilidades de Evaluación - GuruInversor")
    print("=" * 50)
    
    # Generar datos de ejemplo
    np.random.seed(42)
    n_samples = 100
    
    # Precios simulados con tendencia
    prices = 100 + np.cumsum(np.random.normal(0.1, 1, n_samples))
    predictions = prices + np.random.normal(0, 2, n_samples)  # Predicciones con ruido
    
    print(f"📈 Evaluando modelo con {n_samples} muestras...")
    
    # Evaluar modelo
    metrics = evaluate_model_performance(prices, predictions, prices)
    
    # Mostrar reporte
    print_evaluation_report(metrics, "Modelo de Ejemplo")
    
    # Validar estrategia de trading
    trading_metrics = validate_trading_strategy(predictions, prices)
    
    print(f"\n💼 ESTRATEGIA DE TRADING:")
    print(f"   Retorno Total: {trading_metrics.get('total_return', 0):.2%}")
    print(f"   Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Win Rate: {trading_metrics.get('win_rate', 0):.2%}")
    print(f"   Total Trades: {trading_metrics.get('total_trades', 0)}")
