#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador Avanzado de Modelos - GuruInversor
MODEL-004: Sistema completo de métricas de evaluación para modelos LSTM

Este módulo implementa un sistema comprehensivo de evaluación que incluye:
- Métricas básicas de regresión y clasificación
- Métricas financieras específicas para trading
- Evaluación temporal con backtesting
- Análisis de riesgo y performance
- Reportes detallados y visualizaciones
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Imports locales
from ml.metrics import (
    calculate_regression_metrics, 
    calculate_directional_accuracy,
    calculate_financial_metrics,
    evaluate_model_performance
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuración para evaluación de modelos."""
    
    # Configuración de backtesting
    rolling_window_days: int = 30
    min_test_samples: int = 100
    max_test_samples: int = 1000
    
    # Umbrales para métricas
    direction_threshold: float = 0.02  # 2% cambio mínimo para considerar dirección
    significant_move_threshold: float = 0.05  # 5% para movimientos significativos
    
    # Configuración de trading simulado
    transaction_cost: float = 0.001  # 0.1% costo por transacción
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% máximo por posición
    
    # Configuración de riesgo
    risk_free_rate: float = 0.02  # 2% anual
    confidence_level: float = 0.95  # Para VaR
    
    # Configuración de reportes
    save_detailed_results: bool = True
    generate_plots: bool = False  # Deshabilitado para compatibilidad
    export_format: str = 'json'  # 'json', 'csv', 'excel'


class ModelEvaluator:
    """
    Evaluador avanzado de modelos con métricas financieras y análisis temporal.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Inicializar evaluador.
        
        Args:
            config: Configuración de evaluación
        """
        self.config = config or EvaluationConfig()
        self.evaluation_results = {}
        self.backtest_results = {}
        
        logger.info("🎯 ModelEvaluator inicializado para MODEL-004")
    
    def evaluate_model_comprehensive(self, 
                                   model,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   prices_test: Optional[np.ndarray] = None,
                                   ticker: str = None) -> Dict[str, Any]:
        """
        Evaluación comprehensiva del modelo.
        
        Args:
            model: Modelo entrenado
            X_test: Datos de entrada de prueba
            y_test: Objetivos reales
            prices_test: Precios originales para contexto
            ticker: Símbolo de la acción
            
        Returns:
            Diccionario con evaluación completa
        """
        logger.info(f"🔍 Iniciando evaluación comprehensiva para {ticker or 'modelo'}")
        
        try:
            # 1. Predicciones del modelo
            predictions = model.predict(X_test, verbose=0)
            
            # 2. Métricas básicas
            basic_metrics = self._calculate_basic_metrics(y_test, predictions)
            
            # 3. Métricas financieras avanzadas
            financial_metrics = self._calculate_advanced_financial_metrics(
                y_test, predictions, prices_test
            )
            
            # 4. Análisis direccional avanzado
            directional_analysis = self._analyze_directional_performance(
                y_test, predictions
            )
            
            # 5. Análisis de riesgo
            risk_metrics = self._calculate_risk_metrics(y_test, predictions, prices_test)
            
            # 6. Backtesting temporal (si hay suficientes datos)
            if len(y_test) >= self.config.min_test_samples:
                backtest_results = self._perform_rolling_backtest(
                    model, X_test, y_test, prices_test
                )
            else:
                backtest_results = {}
            
            # 7. Análisis de estabilidad temporal
            stability_metrics = self._analyze_temporal_stability(
                y_test, predictions
            )
            
            # 8. Compilar resultados
            comprehensive_results = {
                'metadata': {
                    'ticker': ticker,
                    'evaluation_date': datetime.now().isoformat(),
                    'test_samples': len(y_test),
                    'model_type': getattr(model, 'name', 'Unknown'),
                    'config': asdict(self.config)
                },
                'basic_metrics': basic_metrics,
                'financial_metrics': financial_metrics,
                'directional_analysis': directional_analysis,
                'risk_metrics': risk_metrics,
                'backtest_results': backtest_results,
                'stability_metrics': stability_metrics,
                'overall_score': self._calculate_overall_score(
                    basic_metrics, financial_metrics, directional_analysis, risk_metrics
                )
            }
            
            # Guardar resultados
            self.evaluation_results[ticker or 'latest'] = comprehensive_results
            
            if self.config.save_detailed_results:
                self._save_evaluation_results(comprehensive_results, ticker)
            
            logger.info(f"✅ Evaluación comprehensiva completada para {ticker or 'modelo'}")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ Error en evaluación comprehensiva: {e}")
            raise
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcular métricas básicas usando el módulo existente."""
        logger.debug("Calculando métricas básicas...")
        
        # Usar funciones existentes
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        basic_metrics = evaluate_model_performance(y_true_flat, y_pred_flat)
        
        # Agregar métricas adicionales específicas
        basic_metrics.update({
            'prediction_range': float(np.max(y_pred_flat) - np.min(y_pred_flat)),
            'actual_range': float(np.max(y_true_flat) - np.min(y_true_flat)),
            'prediction_variance': float(np.var(y_pred_flat)),
            'actual_variance': float(np.var(y_true_flat)),
            'correlation_coefficient': float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
        })
        
        return basic_metrics
    
    def _calculate_advanced_financial_metrics(self, 
                                            y_true: np.ndarray,
                                            y_pred: np.ndarray,
                                            prices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calcular métricas financieras avanzadas."""
        logger.debug("Calculando métricas financieras avanzadas...")
        
        metrics = {}
        
        # Si tenemos precios, usar métricas existentes
        if prices is not None:
            y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
            y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
            
            financial_base = calculate_financial_metrics(prices, y_pred_flat)
            metrics.update(financial_base)
        
        # Métricas adicionales de trading
        if len(y_true) > 1:
            # Simular estrategia simple de trading
            trading_metrics = self._simulate_trading_strategy(y_true, y_pred, prices)
            metrics.update(trading_metrics)
            
            # Análisis de momentum
            momentum_metrics = self._analyze_momentum_performance(y_true, y_pred)
            metrics.update(momentum_metrics)
        
        return metrics
    
    def _analyze_directional_performance(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> Dict[str, float]:
        """Análisis detallado de performance direccional."""
        logger.debug("Analizando performance direccional...")
        
        # Usar función existente como base
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        directional_base = calculate_directional_accuracy(y_true_flat, y_pred_flat)
        
        # Análisis por magnitud de cambios
        analysis = dict(directional_base)
        
        if len(y_true_flat) > 1:
            # Cambios reales y predichos
            actual_changes = np.diff(y_true_flat)
            predicted_changes = np.diff(y_pred_flat)
            
            # Precisión direccional por umbrales
            for threshold in [0.01, 0.02, 0.05]:  # 1%, 2%, 5%
                significant_mask = np.abs(actual_changes) > threshold
                if np.any(significant_mask):
                    actual_directions = np.sign(actual_changes[significant_mask])
                    predicted_directions = np.sign(predicted_changes[significant_mask])
                    accuracy = np.mean(actual_directions == predicted_directions)
                    analysis[f'directional_accuracy_{int(threshold*100)}pct'] = float(accuracy)
            
            # Análisis de magnitud de errores direccionales
            direction_errors = predicted_changes - actual_changes
            analysis.update({
                'mean_direction_error': float(np.mean(direction_errors)),
                'std_direction_error': float(np.std(direction_errors)),
                'direction_error_skewness': float(self._calculate_skewness(direction_errors)),
                'large_error_ratio': float(np.mean(np.abs(direction_errors) > 0.05))  # >5% error
            })
        
        return analysis
    
    def _calculate_risk_metrics(self, 
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              prices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calcular métricas de riesgo."""
        logger.debug("Calculando métricas de riesgo...")
        
        metrics = {}
        
        if len(y_true) > 1:
            y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
            y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
            
            # Errores de predicción
            errors = y_pred_flat - y_true_flat
            relative_errors = errors / y_true_flat
            
            # Value at Risk (VaR) de errores
            var_95 = np.percentile(np.abs(errors), 95)
            var_99 = np.percentile(np.abs(errors), 99)
            
            # Expected Shortfall (CVaR)
            cvar_95 = np.mean(np.abs(errors)[np.abs(errors) >= var_95])
            
            # Downside deviation
            negative_errors = errors[errors < 0]
            downside_deviation = np.std(negative_errors) if len(negative_errors) > 0 else 0
            
            metrics.update({
                'value_at_risk_95': float(var_95),
                'value_at_risk_99': float(var_99),
                'conditional_var_95': float(cvar_95),
                'downside_deviation': float(downside_deviation),
                'error_skewness': float(self._calculate_skewness(errors)),
                'error_kurtosis': float(self._calculate_kurtosis(errors)),
                'max_absolute_error': float(np.max(np.abs(errors))),
                'relative_var_95': float(np.percentile(np.abs(relative_errors), 95))
            })
            
            # Si tenemos precios, calcular métricas adicionales
            if prices is not None:
                price_volatility = np.std(np.diff(prices) / prices[:-1])
                prediction_volatility = np.std(np.diff(y_pred_flat) / y_pred_flat[:-1])
                
                metrics.update({
                    'price_volatility': float(price_volatility),
                    'prediction_volatility': float(prediction_volatility),
                    'volatility_ratio': float(prediction_volatility / price_volatility) if price_volatility > 0 else 0
                })
        
        return metrics
    
    def _perform_rolling_backtest(self, 
                                model,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                prices_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Realizar backtesting con ventana rodante."""
        logger.debug("Realizando backtest con ventana rodante...")
        
        window_size = min(self.config.rolling_window_days, len(X_test) // 4)
        if window_size < 10:
            return {'error': 'Datos insuficientes para backtest rodante'}
        
        backtest_results = {
            'window_metrics': [],
            'stability_metrics': {},
            'performance_drift': []
        }
        
        # Realizar evaluaciones por ventanas
        for i in range(0, len(X_test) - window_size, window_size // 2):
            end_idx = min(i + window_size, len(X_test))
            
            X_window = X_test[i:end_idx]
            y_window = y_test[i:end_idx]
            prices_window = prices_test[i:end_idx] if prices_test is not None else None
            
            # Evaluar ventana
            predictions = model.predict(X_window, verbose=0)
            window_metrics = self._calculate_basic_metrics(y_window, predictions)
            
            window_result = {
                'window_start': int(i),
                'window_end': int(end_idx),
                'window_size': int(end_idx - i),
                'metrics': window_metrics
            }
            
            backtest_results['window_metrics'].append(window_result)
        
        # Calcular estabilidad temporal
        if len(backtest_results['window_metrics']) > 1:
            backtest_results['stability_metrics'] = self._calculate_stability_metrics(
                backtest_results['window_metrics']
            )
        
        return backtest_results
    
    def _analyze_temporal_stability(self, 
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """Analizar estabilidad temporal de las predicciones."""
        logger.debug("Analizando estabilidad temporal...")
        
        if len(y_true) < 20:  # Mínimo para análisis temporal
            return {}
        
        # Dividir en segmentos temporales
        n_segments = min(5, len(y_true) // 10)
        segment_size = len(y_true) // n_segments
        
        segment_accuracies = []
        segment_rmses = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(y_true)
            
            y_true_seg = y_true[start_idx:end_idx]
            y_pred_seg = y_pred[start_idx:end_idx]
            
            # Calcular RMSE para el segmento
            rmse_seg = np.sqrt(np.mean((y_true_seg - y_pred_seg) ** 2))
            segment_rmses.append(rmse_seg)
            
            # Calcular precisión direccional para el segmento
            if len(y_true_seg) > 1:
                actual_dirs = np.sign(np.diff(y_true_seg.flatten()))
                pred_dirs = np.sign(np.diff(y_pred_seg.flatten()))
                accuracy_seg = np.mean(actual_dirs == pred_dirs)
                segment_accuracies.append(accuracy_seg)
        
        # Métricas de estabilidad
        stability_metrics = {
            'rmse_stability': float(1.0 - (np.std(segment_rmses) / np.mean(segment_rmses))) if segment_rmses else 0,
            'accuracy_stability': float(1.0 - np.std(segment_accuracies)) if segment_accuracies else 0,
            'performance_trend': float(np.corrcoef(range(len(segment_rmses)), segment_rmses)[0, 1]) if len(segment_rmses) > 1 else 0,
            'consistency_score': float(1.0 - np.var(segment_rmses) / np.mean(segment_rmses)**2) if segment_rmses else 0
        }
        
        return stability_metrics
    
    def _simulate_trading_strategy(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 prices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Simular estrategia de trading basada en predicciones."""
        logger.debug("Simulando estrategia de trading...")
        
        if len(y_true) < 2:
            return {}
        
        # Preparar datos
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        # Calcular señales de trading
        predicted_returns = np.diff(y_pred_flat) / y_pred_flat[:-1]
        actual_returns = np.diff(y_true_flat) / y_true_flat[:-1]
        
        # Estrategia simple: comprar si se predice subida > umbral
        buy_threshold = self.config.direction_threshold
        sell_threshold = -self.config.direction_threshold
        
        signals = np.where(predicted_returns > buy_threshold, 1,
                          np.where(predicted_returns < sell_threshold, -1, 0))
        
        # Simular trading con costos
        portfolio_value = self.config.initial_capital
        portfolio_history = [portfolio_value]
        position = 0
        trades_count = 0
        profitable_trades = 0
        
        for i, signal in enumerate(signals):
            if signal != 0 and signal != position:
                # Realizar trade
                cost = portfolio_value * self.config.transaction_cost
                portfolio_value -= cost
                trades_count += 1
                
                # Calcular retorno del trade anterior
                if i > 0 and position != 0:
                    trade_return = position * actual_returns[i-1]
                    portfolio_value *= (1 + trade_return)
                    if trade_return > 0:
                        profitable_trades += 1
                
                position = signal
            
            portfolio_history.append(portfolio_value)
        
        # Métricas de trading
        total_return = (portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # Sharpe ratio anualizado
        if len(portfolio_history) > 1:
            portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
            excess_returns = portfolio_returns - self.config.risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Win rate
        win_rate = profitable_trades / trades_count if trades_count > 0 else 0
        
        return {
            'simulated_total_return': float(total_return),
            'simulated_sharpe_ratio': float(sharpe_ratio),
            'simulated_win_rate': float(win_rate),
            'simulated_total_trades': int(trades_count),
            'simulated_final_value': float(portfolio_value),
            'strategy_volatility': float(np.std(portfolio_returns)) if 'portfolio_returns' in locals() else 0
        }
    
    def _analyze_momentum_performance(self, 
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """Analizar performance en diferentes condiciones de momentum."""
        logger.debug("Analizando performance de momentum...")
        
        if len(y_true) < 5:
            return {}
        
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        # Identificar periodos de momentum
        returns = np.diff(y_true_flat) / y_true_flat[:-1]
        
        # Momentum alcista/bajista
        bullish_periods = returns > 0.01  # >1% ganancia
        bearish_periods = returns < -0.01  # >1% pérdida
        sideways_periods = ~(bullish_periods | bearish_periods)
        momentum_metrics = {}
        
        for period_name, mask in [('bullish', bullish_periods), 
                                 ('bearish', bearish_periods),
                                 ('sideways', sideways_periods)]:
            if np.any(mask):
                # Asegurar consistencia de tamaños para indexación
                max_len = min(len(y_true_flat) - 1, len(y_pred_flat) - 1, len(mask))
                
                # Ajustar máscaras y arrays al mismo tamaño
                mask_adjusted = mask[:max_len]
                y_true_adjusted = y_true_flat[1:max_len+1]
                y_pred_adjusted = y_pred_flat[1:max_len+1]
                
                # Aplicar máscara a arrays del mismo tamaño
                if np.any(mask_adjusted):
                    y_true_period = y_true_adjusted[mask_adjusted]
                    y_pred_period = y_pred_adjusted[mask_adjusted]
                    
                    if len(y_true_period) > 0:
                        rmse_period = np.sqrt(np.mean((y_true_period - y_pred_period) ** 2))
                        mae_period = np.mean(np.abs(y_true_period - y_pred_period))
                        
                        momentum_metrics.update({
                            f'{period_name}_rmse': float(rmse_period),
                            f'{period_name}_mae': float(mae_period),
                            f'{period_name}_samples': int(len(y_true_period))
                        })
        
        return momentum_metrics
    
    def _calculate_overall_score(self, 
                               basic_metrics: Dict[str, float],
                               financial_metrics: Dict[str, float],
                               directional_analysis: Dict[str, float],
                               risk_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calcular puntuación general del modelo."""
        logger.debug("Calculando puntuación general...")
        
        scores = {}
        
        # Puntuación de precisión (0-1, mayor es mejor)
        rmse = basic_metrics.get('regression_rmse', 1.0)
        mae = basic_metrics.get('regression_mae', 1.0)
        r2 = basic_metrics.get('regression_r2', 0.0)
        
        accuracy_score = max(0, min(1, (1 - rmse/100) * 0.4 + (1 - mae/100) * 0.3 + max(0, r2) * 0.3))
        scores['accuracy_score'] = float(accuracy_score)
        
        # Puntuación direccional (0-1, mayor es mejor)
        dir_accuracy = directional_analysis.get('directional_accuracy', 0.5)
        directional_score = max(0, min(1, dir_accuracy))
        scores['directional_score'] = float(directional_score)
        
        # Puntuación de riesgo (0-1, mayor es mejor = menor riesgo)
        max_error = risk_metrics.get('max_absolute_error', 100)
        var_95 = risk_metrics.get('value_at_risk_95', 50)
        risk_score = max(0, min(1, 1 - (max_error + var_95) / 200))
        scores['risk_score'] = float(risk_score)
        
        # Puntuación financiera (0-1, mayor es mejor)
        total_return = financial_metrics.get('simulated_total_return', 0)
        sharpe = financial_metrics.get('simulated_sharpe_ratio', 0)
        win_rate = financial_metrics.get('simulated_win_rate', 0.5)
        
        financial_score = max(0, min(1, (max(-1, min(1, total_return)) + 1) * 0.4 + 
                                       max(0, min(3, sharpe)) / 3 * 0.3 + 
                                       win_rate * 0.3))
        scores['financial_score'] = float(financial_score)
        
        # Puntuación general ponderada
        overall_score = (accuracy_score * 0.35 + 
                        directional_score * 0.25 + 
                        risk_score * 0.20 + 
                        financial_score * 0.20)
        
        scores['overall_score'] = float(overall_score)
        scores['grade'] = self._score_to_grade(overall_score)
        
        return scores
    
    def _score_to_grade(self, score: float) -> str:
        """Convertir puntuación numérica a calificación."""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C+'
        elif score >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def _calculate_stability_metrics(self, window_metrics: List[Dict]) -> Dict[str, float]:
        """Calcular métricas de estabilidad temporal."""
        if len(window_metrics) < 2:
            return {}
        
        # Extraer métricas por ventana
        rmses = [w['metrics'].get('regression_rmse', 0) for w in window_metrics]
        accuracies = [w['metrics'].get('directional_directional_accuracy', 0) for w in window_metrics]
        
        return {
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std': float(np.std(rmses)),
            'rmse_trend': float(np.corrcoef(range(len(rmses)), rmses)[0, 1]) if len(rmses) > 1 else 0,
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'stability_index': float(1.0 - np.std(rmses) / np.mean(rmses)) if np.mean(rmses) > 0 else 0
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calcular sesgo de la distribución."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calcular curtosis de la distribución."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _save_evaluation_results(self, results: Dict[str, Any], ticker: str = None):
        """Guardar resultados de evaluación."""
        try:
            # Crear directorio de resultados
            results_dir = Path('results/evaluations')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Nombre del archivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"evaluation_{ticker or 'model'}_{timestamp}.json"
            filepath = results_dir / filename
            
            # Guardar como JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📁 Resultados guardados en: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando resultados: {e}")
    
    def generate_evaluation_report(self, ticker: str = None) -> str:
        """Generar reporte de evaluación en texto."""
        if not self.evaluation_results:
            return "No hay resultados de evaluación disponibles."
        
        # Usar el ticker especificado o el último resultado
        key = ticker if ticker and ticker in self.evaluation_results else list(self.evaluation_results.keys())[-1]
        results = self.evaluation_results[key]
        
        report = []
        report.append(f"📊 REPORTE DE EVALUACIÓN MODEL-004")
        report.append("=" * 60)
        report.append(f"Ticker: {results['metadata']['ticker']}")
        report.append(f"Fecha: {results['metadata']['evaluation_date']}")
        report.append(f"Muestras de prueba: {results['metadata']['test_samples']}")
        report.append(f"Tipo de modelo: {results['metadata']['model_type']}")
        
        # Puntuación general
        overall = results['overall_score']
        report.append(f"\n🎯 PUNTUACIÓN GENERAL: {overall['overall_score']:.3f} ({overall['grade']})")
        report.append("-" * 40)
        report.append(f"   Precisión: {overall['accuracy_score']:.3f}")
        report.append(f"   Direccional: {overall['directional_score']:.3f}")
        report.append(f"   Riesgo: {overall['risk_score']:.3f}")
        report.append(f"   Financiera: {overall['financial_score']:.3f}")
        
        # Métricas básicas
        basic = results['basic_metrics']
        report.append(f"\n📈 MÉTRICAS BÁSICAS:")
        report.append(f"   RMSE: {basic.get('regression_rmse', 0):.4f}")
        report.append(f"   MAE: {basic.get('regression_mae', 0):.4f}")
        report.append(f"   R²: {basic.get('regression_r2', 0):.4f}")
        report.append(f"   Correlación: {basic.get('correlation_coefficient', 0):.4f}")
        
        # Métricas direccionales
        directional = results['directional_analysis']
        report.append(f"\n🎯 ANÁLISIS DIRECCIONAL:")
        report.append(f"   Precisión direccional: {directional.get('directional_accuracy', 0):.2%}")
        for thresh in [1, 2, 5]:
            key = f'directional_accuracy_{thresh}pct'
            if key in directional:
                report.append(f"   Precisión >{thresh}%: {directional[key]:.2%}")
        
        # Métricas financieras
        financial = results['financial_metrics']
        if financial:
            report.append(f"\n💰 MÉTRICAS FINANCIERAS:")
            if 'simulated_total_return' in financial:
                report.append(f"   Retorno total: {financial['simulated_total_return']:.2%}")
            if 'simulated_sharpe_ratio' in financial:
                report.append(f"   Sharpe Ratio: {financial['simulated_sharpe_ratio']:.2f}")
            if 'simulated_win_rate' in financial:
                report.append(f"   Win Rate: {financial['simulated_win_rate']:.2%}")
        
        # Métricas de riesgo
        risk = results['risk_metrics']
        if risk:
            report.append(f"\n⚠️  MÉTRICAS DE RIESGO:")
            if 'value_at_risk_95' in risk:
                report.append(f"   VaR 95%: {risk['value_at_risk_95']:.4f}")
            if 'max_absolute_error' in risk:
                report.append(f"   Error máximo: {risk['max_absolute_error']:.4f}")
            if 'downside_deviation' in risk:
                report.append(f"   Desviación bajista: {risk['downside_deviation']:.4f}")
        
        # Estabilidad temporal
        stability = results['stability_metrics']
        if stability:
            report.append(f"\n⏱️  ESTABILIDAD TEMPORAL:")
            if 'rmse_stability' in stability:
                report.append(f"   Estabilidad RMSE: {stability['rmse_stability']:.3f}")
            if 'consistency_score' in stability:
                report.append(f"   Puntuación consistencia: {stability['consistency_score']:.3f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def compare_models(self, evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
        """Comparar múltiples modelos evaluados."""
        if not evaluation_results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            row = {
                'model': model_name,
                'overall_score': results['overall_score']['overall_score'],
                'grade': results['overall_score']['grade'],
                'rmse': results['basic_metrics'].get('regression_rmse', np.nan),
                'mae': results['basic_metrics'].get('regression_mae', np.nan),
                'r2': results['basic_metrics'].get('regression_r2', np.nan),
                'directional_accuracy': results['directional_analysis'].get('directional_accuracy', np.nan),
                'simulated_return': results['financial_metrics'].get('simulated_total_return', np.nan),
                'sharpe_ratio': results['financial_metrics'].get('simulated_sharpe_ratio', np.nan),
                'var_95': results['risk_metrics'].get('value_at_risk_95', np.nan),
                'test_samples': results['metadata']['test_samples']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('overall_score', ascending=False)


def create_model_evaluator(config: EvaluationConfig = None) -> ModelEvaluator:
    """
    Función de conveniencia para crear evaluador de modelos.
    
    Args:
        config: Configuración de evaluación
        
    Returns:
        Instancia de ModelEvaluator
    """
    return ModelEvaluator(config)


if __name__ == "__main__":
    # Ejemplo de uso
    print("🎯 MODEL-004 - Sistema de Evaluación Avanzado")
    print("=" * 60)
    
    # Configuración
    config = EvaluationConfig(
        rolling_window_days=20,
        direction_threshold=0.02,
        save_detailed_results=True
    )
    
    # Crear evaluador
    evaluator = create_model_evaluator(config)
    
    print("✅ ModelEvaluator inicializado")
    print("🚀 MODEL-004 listo para evaluar modelos LSTM")
    print("=" * 60)
