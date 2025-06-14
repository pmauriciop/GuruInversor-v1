#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas para MODEL-004 - Sistema de Evaluación Avanzado
Validación del ModelEvaluator y métricas de evaluación comprehensiva.
"""

import sys
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import unittest
import logging
from unittest.mock import Mock, patch

# Configurar path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.insert(0, str(backend_dir))

# Imports del proyecto
from ml.model_evaluator import ModelEvaluator, EvaluationConfig, create_model_evaluator
from ml.model_architecture import LSTMConfig, LSTMArchitect

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelEvaluator(unittest.TestCase):
    """Pruebas para el evaluador de modelos MODEL-004."""
    
    def setUp(self):
        """Configurar entorno de pruebas."""
        # Configuración de evaluación
        self.config = EvaluationConfig(
            rolling_window_days=10,
            min_test_samples=20,
            direction_threshold=0.02,
            save_detailed_results=False  # No guardar durante pruebas
        )
        
        # Crear evaluador
        self.evaluator = ModelEvaluator(self.config)
        
        # Crear modelo mock simple para pruebas
        self.mock_model = self._create_mock_model()
        
        # Datos sintéticos para pruebas
        np.random.seed(42)
        self.n_samples = 100
        self.X_test = np.random.random((self.n_samples, 20, 6))
        
        # Precios simulados con tendencia
        self.prices_test = 100 + np.cumsum(np.random.normal(0.1, 1, self.n_samples))
        
        # y_test basado en precios (precio siguiente)
        self.y_test = self.prices_test[1:self.n_samples]
        self.X_test = self.X_test[:len(self.y_test)]  # Ajustar tamaño
        
        # Ajustar precios_test
        self.prices_test = self.prices_test[:len(self.y_test)]
    
    def _create_mock_model(self):
        """Crear modelo mock para pruebas."""
        mock_model = Mock()
        mock_model.name = 'LSTM_Test'
        
        # Mock para predict que retorna predicciones realistas
        def mock_predict(X, verbose=0):
            np.random.seed(42)  # Para reproducibilidad
            # Predicciones basadas en los datos de entrada con algo de ruido
            n_samples = X.shape[0]
            base_predictions = 100 + np.cumsum(np.random.normal(0.05, 0.8, n_samples))
            return base_predictions.reshape(-1, 1)
        
        mock_model.predict = mock_predict
        return mock_model
    
    def test_evaluator_initialization(self):
        """Probar inicialización del evaluador."""
        # Probar inicialización con configuración
        evaluator_with_config = ModelEvaluator(self.config)
        self.assertIsInstance(evaluator_with_config.config, EvaluationConfig)
        self.assertEqual(evaluator_with_config.config.rolling_window_days, 10)
          # Probar inicialización sin configuración
        evaluator_default = ModelEvaluator()
        self.assertIsInstance(evaluator_default.config, EvaluationConfig)
        self.assertEqual(evaluator_default.config.rolling_window_days, 30)  # Valor por defecto
        
        print("✅ Inicialización del evaluador")
    
    def test_basic_metrics_calculation(self):
        """Probar cálculo de métricas básicas."""
        # Datos sintéticos
        y_true = np.array([100, 101, 102, 100, 99])
        y_pred = np.array([100.5, 100.8, 101.5, 100.2, 99.1])
        
        basic_metrics = self.evaluator._calculate_basic_metrics(y_true, y_pred)
        
        # Verificar que se calcularon las métricas
        self.assertIsInstance(basic_metrics, dict)
        self.assertIn('regression_rmse', basic_metrics)
        self.assertIn('regression_mae', basic_metrics)
        self.assertIn('regression_r2', basic_metrics)
        self.assertIn('correlation_coefficient', basic_metrics)
        
        # Verificar que las métricas son números válidos (permitir NaN para casos específicos)
        for metric_name, value in basic_metrics.items():
            self.assertIsInstance(value, (int, float))
            # Para correlación, permitir NaN si hay datos constantes
            if metric_name == 'correlation_coefficient' and np.isnan(value):
                continue  # Es aceptable para datos con poca variabilidad
            # Para otros casos críticos, verificar que no sean NaN o infinito
            if metric_name in ['regression_rmse', 'regression_mae']:
                self.assertFalse(np.isnan(value), f"Métrica crítica {metric_name} es NaN")
                self.assertFalse(np.isinf(value), f"Métrica crítica {metric_name} es infinita")
        
        print(f"✅ Métricas básicas calculadas: {len(basic_metrics)} métricas")
    
    def test_directional_analysis(self):
        """Probar análisis direccional."""
        # Crear datos con patrones direccionales conocidos
        y_true = np.array([100, 102, 101, 105, 103, 108])
        y_pred = np.array([100, 101.5, 101.2, 104.5, 103.1, 107.5])
        
        directional_analysis = self.evaluator._analyze_directional_performance(y_true, y_pred)
        
        # Verificar métricas direccionales
        self.assertIsInstance(directional_analysis, dict)
        self.assertIn('directional_accuracy', directional_analysis)
        
        # Verificar análisis por umbrales
        threshold_keys = [k for k in directional_analysis.keys() if 'directional_accuracy_' in k and 'pct' in k]
        self.assertGreater(len(threshold_keys), 0)
        
        print(f"✅ Análisis direccional: {len(directional_analysis)} métricas")
    
    def test_risk_metrics_calculation(self):
        """Probar cálculo de métricas de riesgo."""
        # Datos sintéticos con algunos outliers
        np.random.seed(42)
        y_true = np.random.normal(100, 5, 50)
        y_pred = y_true + np.random.normal(0, 2, 50)
        
        # Agregar algunos outliers
        y_pred[10] = y_true[10] + 15  # Error grande
        y_pred[30] = y_true[30] - 12  # Error grande negativo
        
        risk_metrics = self.evaluator._calculate_risk_metrics(y_true, y_pred, y_true)
        
        # Verificar métricas de riesgo
        self.assertIsInstance(risk_metrics, dict)
        self.assertIn('value_at_risk_95', risk_metrics)
        self.assertIn('value_at_risk_99', risk_metrics)
        self.assertIn('conditional_var_95', risk_metrics)
        self.assertIn('max_absolute_error', risk_metrics)
        
        # Verificar que VaR 99% >= VaR 95%
        if 'value_at_risk_95' in risk_metrics and 'value_at_risk_99' in risk_metrics:
            self.assertGreaterEqual(risk_metrics['value_at_risk_99'], risk_metrics['value_at_risk_95'])
        
        print(f"✅ Métricas de riesgo calculadas: {len(risk_metrics)} métricas")
    
    def test_trading_simulation(self):
        """Probar simulación de trading."""
        # Crear datos de precio con tendencia
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0.1, 1, 50))
        y_true = prices
        y_pred = prices + np.random.normal(0, 0.5, 50)  # Predicciones con poco ruido
        
        trading_metrics = self.evaluator._simulate_trading_strategy(y_true, y_pred, prices)
        
        # Verificar métricas de trading
        self.assertIsInstance(trading_metrics, dict)
        self.assertIn('simulated_total_return', trading_metrics)
        self.assertIn('simulated_sharpe_ratio', trading_metrics)
        self.assertIn('simulated_win_rate', trading_metrics)
        self.assertIn('simulated_total_trades', trading_metrics)
        
        # Verificar valores razonables
        if 'simulated_win_rate' in trading_metrics:
            win_rate = trading_metrics['simulated_win_rate']
            self.assertGreaterEqual(win_rate, 0.0)
            self.assertLessEqual(win_rate, 1.0)
        
        print(f"✅ Simulación de trading: {len(trading_metrics)} métricas")
    
    def test_temporal_stability_analysis(self):
        """Probar análisis de estabilidad temporal."""
        # Crear datos con diferentes niveles de estabilidad
        np.random.seed(42)
        
        # Primera mitad: predicciones estables
        y_true_1 = np.random.normal(100, 2, 25)
        y_pred_1 = y_true_1 + np.random.normal(0, 1, 25)
        
        # Segunda mitad: predicciones menos estables
        y_true_2 = np.random.normal(100, 2, 25)
        y_pred_2 = y_true_2 + np.random.normal(0, 3, 25)
        
        y_true = np.concatenate([y_true_1, y_true_2])
        y_pred = np.concatenate([y_pred_1, y_pred_2])
        
        stability_metrics = self.evaluator._analyze_temporal_stability(y_true, y_pred)
        
        # Verificar métricas de estabilidad
        self.assertIsInstance(stability_metrics, dict)
        if stability_metrics:  # Si hay suficientes datos
            self.assertIn('rmse_stability', stability_metrics)
            self.assertIn('consistency_score', stability_metrics)
            
            # Verificar rangos válidos
            for metric_name, value in stability_metrics.items():
                self.assertFalse(np.isnan(value), f"Métrica {metric_name} es NaN")
        
        print(f"✅ Análisis de estabilidad temporal: {len(stability_metrics)} métricas")
    
    def test_comprehensive_evaluation(self):
        """Probar evaluación comprehensiva completa."""
        results = self.evaluator.evaluate_model_comprehensive(
            model=self.mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            prices_test=self.prices_test,
            ticker='TEST'
        )
        
        # Verificar estructura de resultados
        self.assertIsInstance(results, dict)
        
        # Verificar secciones principales
        required_sections = [
            'metadata',
            'basic_metrics',
            'financial_metrics',
            'directional_analysis',
            'risk_metrics',
            'stability_metrics',
            'overall_score'
        ]
        
        for section in required_sections:
            self.assertIn(section, results, f"Falta sección: {section}")
        
        # Verificar metadata
        metadata = results['metadata']
        self.assertEqual(metadata['ticker'], 'TEST')
        self.assertEqual(metadata['test_samples'], len(self.y_test))
        
        # Verificar puntuación general
        overall_score = results['overall_score']
        self.assertIn('overall_score', overall_score)
        self.assertIn('grade', overall_score)
        self.assertGreaterEqual(overall_score['overall_score'], 0)
        self.assertLessEqual(overall_score['overall_score'], 1)
        
        print(f"✅ Evaluación comprehensiva completada")
        print(f"    Puntuación general: {overall_score['overall_score']:.3f} ({overall_score['grade']})")
    
    def test_rolling_backtest(self):
        """Probar backtest con ventana rodante."""
        # Usar datos suficientes para backtest
        if len(self.X_test) >= self.config.min_test_samples:
            backtest_results = self.evaluator._perform_rolling_backtest(
                model=self.mock_model,
                X_test=self.X_test,
                y_test=self.y_test,
                prices_test=self.prices_test
            )
            
            # Verificar estructura de resultados
            self.assertIsInstance(backtest_results, dict)
            self.assertIn('window_metrics', backtest_results)
            
            # Verificar que se crearon ventanas
            window_metrics = backtest_results['window_metrics']
            if window_metrics:
                self.assertIsInstance(window_metrics, list)
                self.assertGreater(len(window_metrics), 0)
                
                # Verificar estructura de cada ventana
                for window in window_metrics:
                    self.assertIn('window_start', window)
                    self.assertIn('window_end', window)
                    self.assertIn('metrics', window)
                    self.assertIsInstance(window['metrics'], dict)
            
            print(f"✅ Backtest rodante: {len(window_metrics)} ventanas analizadas")
        else:
            print("⚠️  Datos insuficientes para backtest rodante")
    
    def test_overall_score_calculation(self):
        """Probar cálculo de puntuación general."""
        # Métricas de ejemplo
        basic_metrics = {
            'regression_rmse': 2.5,
            'regression_mae': 1.8,
            'regression_r2': 0.75
        }
        
        financial_metrics = {
            'simulated_total_return': 0.15,
            'simulated_sharpe_ratio': 1.2,
            'simulated_win_rate': 0.65
        }
        
        directional_analysis = {
            'directional_accuracy': 0.72
        }
        
        risk_metrics = {
            'max_absolute_error': 5.2,
            'value_at_risk_95': 3.1
        }
        
        overall_score = self.evaluator._calculate_overall_score(
            basic_metrics, financial_metrics, directional_analysis, risk_metrics
        )
        
        # Verificar estructura
        self.assertIsInstance(overall_score, dict)
        self.assertIn('overall_score', overall_score)
        self.assertIn('grade', overall_score)
        self.assertIn('accuracy_score', overall_score)
        self.assertIn('directional_score', overall_score)
        self.assertIn('risk_score', overall_score)
        self.assertIn('financial_score', overall_score)
        
        # Verificar rangos
        for score_name, score_value in overall_score.items():
            if score_name != 'grade':
                self.assertGreaterEqual(score_value, 0.0)
                self.assertLessEqual(score_value, 1.0)
        
        # Verificar calificación válida
        valid_grades = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D']
        self.assertIn(overall_score['grade'], valid_grades)
        
        print(f"✅ Puntuación general: {overall_score['overall_score']:.3f} ({overall_score['grade']})")
    
    def test_evaluation_report_generation(self):
        """Probar generación de reportes."""
        # Realizar evaluación primero
        results = self.evaluator.evaluate_model_comprehensive(
            model=self.mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            prices_test=self.prices_test,
            ticker='TEST_REPORT'
        )
        
        # Generar reporte
        report = self.evaluator.generate_evaluation_report('TEST_REPORT')
        
        # Verificar que el reporte es una cadena no vacía
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)
        
        # Verificar que contiene secciones esperadas
        expected_sections = [
            'REPORTE DE EVALUACIÓN MODEL-004',
            'PUNTUACIÓN GENERAL',
            'MÉTRICAS BÁSICAS',
            'ANÁLISIS DIRECCIONAL'
        ]
        
        for section in expected_sections:
            self.assertIn(section, report)
        
        print("✅ Reporte de evaluación generado correctamente")
    
    def test_convenience_function(self):
        """Probar función de conveniencia."""
        config = EvaluationConfig(rolling_window_days=15)
        evaluator = create_model_evaluator(config)
        
        self.assertIsInstance(evaluator, ModelEvaluator)
        self.assertEqual(evaluator.config.rolling_window_days, 15)
        
        print("✅ Función de conveniencia")
    
    def test_edge_cases(self):
        """Probar casos extremos."""
        # Datos mínimos
        y_true_min = np.array([100])
        y_pred_min = np.array([101])
        
        # Debe manejar datos mínimos sin errores
        basic_metrics = self.evaluator._calculate_basic_metrics(y_true_min, y_pred_min)
        self.assertIsInstance(basic_metrics, dict)
        
        # Datos idénticos
        y_identical = np.array([100, 100, 100])
        basic_metrics_identical = self.evaluator._calculate_basic_metrics(y_identical, y_identical)
        self.assertEqual(basic_metrics_identical.get('regression_rmse', -1), 0.0)
        
        # Datos con NaN (debe manejar graciosamente)
        y_with_nan = np.array([100, np.nan, 102])
        y_pred_normal = np.array([101, 101, 103])
        
        try:
            # Puede fallar o manejar NaN - ambos son aceptables
            basic_metrics_nan = self.evaluator._calculate_basic_metrics(y_with_nan, y_pred_normal)
        except:
            pass  # Es aceptable que falle con NaN
        
        print("✅ Casos extremos manejados")


def run_model_004_tests():
    """Ejecutar todas las pruebas de MODEL-004."""
    print("🧪 EJECUTANDO PRUEBAS MODEL-004 - Sistema de Evaluación Avanzado")
    print("=" * 70)
    
    # Configurar TensorFlow para pruebas
    tf.config.run_functions_eagerly(True)
    
    # Crear suite de pruebas
    test_suite = unittest.TestSuite()
    
    # Añadir pruebas específicas
    test_cases = [
        'test_evaluator_initialization',
        'test_basic_metrics_calculation',
        'test_directional_analysis',
        'test_risk_metrics_calculation',
        'test_trading_simulation',
        'test_temporal_stability_analysis',
        'test_comprehensive_evaluation',
        'test_rolling_backtest',
        'test_overall_score_calculation',
        'test_evaluation_report_generation',
        'test_convenience_function',
        'test_edge_cases'
    ]
    
    for test_case in test_cases:
        test_suite.addTest(TestModelEvaluator(test_case))
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(test_suite)
    
    # Resultados
    total_tests = result.testsRun
    passed_tests = total_tests - len(result.failures) - len(result.errors)
    
    print(f"\nResultados: {passed_tests}/{total_tests} pruebas exitosas ({passed_tests/total_tests*100:.1f}%)")
    
    # Mostrar fallos si los hay
    if result.failures:
        print(f"\n❌ Fallos ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('/')[-1] if '/' in traceback else traceback[:100]}")
    
    if result.errors:
        print(f"\n💥 Errores ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('/')[-1] if '/' in traceback else traceback[:100]}")
    
    if passed_tests == total_tests:
        print("\n🎉 ¡TODAS LAS PRUEBAS DE MODEL-004 PASARON!")
        print("✅ Sistema de evaluación avanzado implementado y validado")
        print("🚀 MODEL-004 completado - Listo para MODEL-005")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} pruebas fallaron")
        print("❌ MODEL-004 requiere correcciones antes de continuar")
    
    print("=" * 70)
    return passed_tests == total_tests


if __name__ == '__main__':
    success = run_model_004_tests()
    sys.exit(0 if success else 1)
