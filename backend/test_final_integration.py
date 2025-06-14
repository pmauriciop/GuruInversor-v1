#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas de Integración Final - GuruInversor

Suite de pruebas que valida la integración usando datos sintéticos
para evitar dependencias externas y problemas de conectividad.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Añadir el directorio backend al path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Importar módulos principales
from ml.preprocessor import DataProcessor, ProcessingConfig, process_stock_data
from ml.metrics import evaluate_model_performance, print_evaluation_report

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalIntegrationTests:
    """Suite de pruebas de integración final con datos sintéticos."""
    
    def __init__(self):
        self.test_results = {}
        
    def create_realistic_stock_data(self, symbol="AAPL", days=365, base_price=150.0):
        """Crear datos sintéticos realistas de acciones."""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Simular movimiento de precios realista
        np.random.seed(42)
        
        # Tendencia con volatilidad
        returns = np.random.normal(0.0005, 0.02, days)  # 0.05% tendencia diaria, 2% volatilidad
        
        prices = [base_price]
        for i in range(1, days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 1.0))  # Evitar precios negativos
        
        # Crear datos OHLCV realistas
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'volume': np.random.randint(50000000, 200000000, days)
        })
        
        # Calcular high y low basado en open/close
        for i in range(len(data)):
            open_price = data.loc[i, 'open']
            close_price = data.loc[i, 'close']
            
            # High es al menos el máximo de open/close + algo de variación
            high_extra = abs(np.random.normal(0, 0.01))
            data.loc[i, 'high'] = max(open_price, close_price) * (1 + high_extra)
            
            # Low es máximo el mínimo de open/close - algo de variación
            low_reduction = abs(np.random.normal(0, 0.01))
            data.loc[i, 'low'] = min(open_price, close_price) * (1 - low_reduction)
        
        return data
    
    def test_synthetic_data_processing(self):
        """Probar procesamiento completo con datos sintéticos."""
        logger.info("🧪 Probando procesamiento con datos sintéticos...")
        
        try:
            # Crear datos sintéticos de alta calidad
            stock_data = self.create_realistic_stock_data(days=300)
              # Configuraciones de procesamiento para probar
            configs = [
                ProcessingConfig(sequence_length=30, normalize_method='minmax'),
                ProcessingConfig(sequence_length=60, normalize_method='zscore'),
                ProcessingConfig(sequence_length=90, normalize_method='robust')
            ]
            
            results = []
            for i, config in enumerate(configs):
                # Siempre agregar indicadores técnicos usando el parámetro de la función
                X_sequences, y_targets, processor = process_stock_data(stock_data, config, add_features=True)
                
                # Validaciones
                assert len(X_sequences) > 0, f"Config {i}: No se generaron secuencias"
                assert X_sequences.shape[0] == y_targets.shape[0], f"Config {i}: Mismatch secuencias/targets"
                assert X_sequences.shape[1] == config.sequence_length, f"Config {i}: Longitud incorrecta"
                
                # Validar desnormalización
                sample_preds = y_targets[:5]
                denorm_preds = processor.denormalize_predictions(sample_preds)
                assert len(denorm_preds) == len(sample_preds), f"Config {i}: Error desnormalización"
                
                results.append({
                    'config': i+1,
                    'sequences': X_sequences.shape[0],
                    'features': X_sequences.shape[2],
                    'seq_length': config.sequence_length,
                    'normalization': config.normalize_method
                })
            
            for result in results:
                logger.info(f"   ✅ Config {result['config']}: {result['sequences']} secuencias, "
                          f"{result['features']} features, {result['seq_length']} longitud, "
                          f"{result['normalization']} normalización")
            
            self.test_results['synthetic_processing'] = True
            
        except Exception as e:
            logger.error(f"   ❌ Error en procesamiento sintético: {e}")
            self.test_results['synthetic_processing'] = False
    
    def test_comprehensive_metrics_evaluation(self):
        """Probar evaluación completa de métricas con múltiples escenarios."""
        logger.info("🧪 Probando evaluación completa de métricas...")
        
        try:
            # Crear datos para evaluación
            stock_data = self.create_realistic_stock_data(days=200)
            
            config = ProcessingConfig(sequence_length=60, normalize_method='minmax')
            X_sequences, y_targets, processor = process_stock_data(stock_data, config)
            
            # Simular diferentes escenarios de predicción
            scenarios = [
                {'name': 'Predicción Excelente', 'noise': 0.01, 'bias': 0.0},
                {'name': 'Predicción Buena', 'noise': 0.03, 'bias': 0.01},
                {'name': 'Predicción Regular', 'noise': 0.06, 'bias': 0.02},
                {'name': 'Predicción Pobre', 'noise': 0.12, 'bias': 0.05}
            ]
            
            scenario_results = []
            
            for scenario in scenarios:
                np.random.seed(42)
                
                # Simular predicciones con ruido y sesgo
                y_pred = y_targets + np.random.normal(scenario['bias'], scenario['noise'], y_targets.shape)
                
                # Desnormalizar para evaluación
                y_true_prices = processor.denormalize_predictions(y_targets[:, 1])  # Precio close
                y_pred_prices = processor.denormalize_predictions(y_pred[:, 1])
                
                # Calcular métricas
                metrics = evaluate_model_performance(y_true_prices, y_pred_prices, y_true_prices)
                
                # Validaciones detalladas
                assert 'regression_rmse' in metrics, f"{scenario['name']}: Falta RMSE"
                assert 'regression_mae' in metrics, f"{scenario['name']}: Falta MAE"
                assert 'regression_mape' in metrics, f"{scenario['name']}: Falta MAPE"
                assert 'regression_r2' in metrics, f"{scenario['name']}: Falta R²"
                assert 'directional_directional_accuracy' in metrics, f"{scenario['name']}: Falta precisión direccional"
                assert 'financial_sharpe_actual' in metrics, f"{scenario['name']}: Falta Sharpe actual"
                assert 'financial_max_drawdown_actual' in metrics, f"{scenario['name']}: Falta Max Drawdown"
                
                # Validar rangos de métricas
                assert metrics['regression_rmse'] > 0, f"{scenario['name']}: RMSE debe ser positivo"
                assert 0 <= metrics['directional_directional_accuracy'] <= 1, f"{scenario['name']}: Precisión direccional fuera de rango"
                assert -1 <= metrics['regression_r2'] <= 1, f"{scenario['name']}: R² fuera de rango válido"
                
                scenario_results.append({
                    'name': scenario['name'],
                    'rmse': metrics['regression_rmse'],
                    'mae': metrics['regression_mae'],
                    'mape': metrics['regression_mape'],
                    'r2': metrics['regression_r2'],
                    'dir_acc': metrics['directional_directional_accuracy'],
                    'sharpe': metrics['financial_sharpe_actual']
                })
            
            # Mostrar resultados
            for result in scenario_results:
                logger.info(f"   ✅ {result['name']}: RMSE={result['rmse']:.2f}, "
                          f"MAE={result['mae']:.2f}, R²={result['r2']:.3f}, "
                          f"Dir.Acc={result['dir_acc']:.2%}")
            
            # Validar que las métricas mejoran con menos ruido
            assert scenario_results[0]['rmse'] < scenario_results[-1]['rmse'], "RMSE debería mejorar con menos ruido"
            
            self.test_results['comprehensive_metrics'] = True
            
        except Exception as e:
            logger.error(f"   ❌ Error en métricas completas: {e}")
            self.test_results['comprehensive_metrics'] = False
    
    def test_multi_timeframe_analysis(self):
        """Probar análisis en múltiples marcos temporales."""
        logger.info("🧪 Probando análisis multi-timeframe...")
        
        try:
            # Crear datos para diferentes períodos
            timeframes = [
                {'name': 'Corto Plazo', 'days': 100, 'seq_len': 15},
                {'name': 'Medio Plazo', 'days': 250, 'seq_len': 30},
                {'name': 'Largo Plazo', 'days': 500, 'seq_len': 60}
            ]
            
            timeframe_results = []
            
            for timeframe in timeframes:                # Crear datos específicos para el timeframe
                data = self.create_realistic_stock_data(
                    days=timeframe['days'],
                    base_price=np.random.uniform(100, 200)
                )
                
                config = ProcessingConfig(
                    sequence_length=timeframe['seq_len'],
                    normalize_method='minmax'
                )
                
                X_seq, y_tar, proc = process_stock_data(data, config, add_features=True)
                
                # Validaciones por timeframe
                assert len(X_seq) > 0, f"{timeframe['name']}: Sin secuencias generadas"
                
                # Simular predicciones y evaluar
                np.random.seed(42)
                y_pred = y_tar + np.random.normal(0, 0.04, y_tar.shape)
                
                y_true_prices = proc.denormalize_predictions(y_tar[:, 1])
                y_pred_prices = proc.denormalize_predictions(y_pred[:, 1])
                
                metrics = evaluate_model_performance(y_true_prices, y_pred_prices, y_true_prices)
                
                timeframe_results.append({
                    'name': timeframe['name'],
                    'sequences': len(X_seq),
                    'features': X_seq.shape[2],
                    'rmse': metrics['regression_rmse'],
                    'dir_acc': metrics['directional_directional_accuracy']
                })
            
            for result in timeframe_results:
                logger.info(f"   ✅ {result['name']}: {result['sequences']} secuencias, "
                          f"{result['features']} features, RMSE={result['rmse']:.2f}, "
                          f"Acc={result['dir_acc']:.2%}")
            
            self.test_results['multi_timeframe'] = True
            
        except Exception as e:
            logger.error(f"   ❌ Error en multi-timeframe: {e}")
            self.test_results['multi_timeframe'] = False
    
    def test_feature_engineering_validation(self):
        """Probar y validar ingeniería de características."""
        logger.info("🧪 Probando ingeniería de características...")
        
        try:
            # Crear datos de prueba
            data = self.create_realistic_stock_data(days=200)
              # Configuraciones con diferentes niveles de features
            feature_configs = [
                {
                    'name': 'Solo OHLCV',
                    'config': ProcessingConfig(
                        sequence_length=30,
                        features=['open', 'high', 'low', 'close', 'volume']
                    ),
                    'add_features': False
                },
                {
                    'name': 'Con Indicadores Técnicos',
                    'config': ProcessingConfig(
                        sequence_length=30,
                        features=['open', 'high', 'low', 'close', 'volume']
                    ),
                    'add_features': True
                }
            ]
            
            feature_results = []
            
            for feature_setup in feature_configs:
                X_seq, y_tar, proc = process_stock_data(data, feature_setup['config'], add_features=feature_setup['add_features'])
                
                # Validaciones
                assert len(X_seq) > 0, f"{feature_setup['name']}: Sin secuencias"
                  # Verificar que se agregaron features técnicos cuando se solicitó
                if feature_setup['add_features']:
                    assert X_seq.shape[2] > 5, f"{feature_setup['name']}: Deberían haber más de 5 features"
                else:
                    assert X_seq.shape[2] <= 10, f"{feature_setup['name']}: Demasiados features para configuración básica"
                
                feature_results.append({
                    'name': feature_setup['name'],
                    'sequences': len(X_seq),
                    'features': X_seq.shape[2],
                    'data_shape': X_seq.shape
                })
            
            for result in feature_results:
                logger.info(f"   ✅ {result['name']}: {result['sequences']} secuencias, "
                          f"{result['features']} features, shape={result['data_shape']}")
            
            # Validar que la configuración con indicadores técnicos tiene más features
            basic_features = feature_results[0]['features']
            enhanced_features = feature_results[1]['features']
            assert enhanced_features > basic_features, "Indicadores técnicos deberían agregar más features"
            
            self.test_results['feature_engineering'] = True
            
        except Exception as e:
            logger.error(f"   ❌ Error en ingeniería de características: {e}")
            self.test_results['feature_engineering'] = False
    
    def test_system_stress_and_performance(self):
        """Probar rendimiento y resistencia del sistema."""
        logger.info("🧪 Probando rendimiento y resistencia del sistema...")
        
        try:
            # Test de volumen: procesar múltiples datasets
            datasets = []
            for i in range(5):
                data = self.create_realistic_stock_data(
                    days=150,
                    base_price=np.random.uniform(50, 300)
                )
                datasets.append(data)
            
            # Procesar todos los datasets
            total_sequences = 0
            processing_times = []
            
            for i, data in enumerate(datasets):
                start_time = datetime.now()
                
                config = ProcessingConfig(sequence_length=40, normalize_method='minmax')
                X_seq, y_tar, proc = process_stock_data(data, config)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                assert len(X_seq) > 0, f"Dataset {i+1}: Sin secuencias procesadas"
                
                total_sequences += len(X_seq)
                processing_times.append(processing_time)
            
            avg_processing_time = sum(processing_times) / len(processing_times)
            
            logger.info(f"   ✅ Procesados {len(datasets)} datasets exitosamente")
            logger.info(f"   ✅ Total secuencias generadas: {total_sequences}")
            logger.info(f"   ✅ Tiempo promedio de procesamiento: {avg_processing_time:.2f}s")
            
            # Validar rendimiento aceptable (debería ser < 5 segundos por dataset)
            assert avg_processing_time < 5.0, f"Rendimiento muy lento: {avg_processing_time:.2f}s"
            
            self.test_results['stress_performance'] = True
            
        except Exception as e:
            logger.error(f"   ❌ Error en pruebas de rendimiento: {e}")
            self.test_results['stress_performance'] = False
    
    def run_all_final_tests(self):
        """Ejecutar todas las pruebas finales de integración."""
        logger.info("🚀 Iniciando pruebas finales de integración del sistema")
        logger.info("=" * 80)
        
        # Ejecutar todas las pruebas
        self.test_synthetic_data_processing()
        self.test_comprehensive_metrics_evaluation()
        self.test_multi_timeframe_analysis()
        self.test_feature_engineering_validation()
        self.test_system_stress_and_performance()
        
        # Resumen final
        logger.info("=" * 80)
        logger.info("📊 RESUMEN FINAL DE PRUEBAS DE INTEGRACIÓN")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            logger.info(f"   {test_name.replace('_', ' ').title():<35}: {status}")
        
        logger.info("=" * 80)
        logger.info(f"RESULTADO FINAL: {passed_tests}/{total_tests} pruebas pasaron")
        
        success = passed_tests == total_tests
        
        if success:
            logger.info("🎉 ¡TODAS LAS PRUEBAS DE INTEGRACIÓN PASARON!")
            logger.info("✅ Sistema completamente validado y listo para producción")
            logger.info("🚀 Fundación del proyecto completada exitosamente")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} prueba(s) fallaron")
            logger.info("📝 Revisar logs para detalles de fallos")
        
        return success


def run_final_integration_tests():
    """Ejecutar las pruebas finales de integración."""
    print("🧪 PRUEBAS FINALES DE INTEGRACIÓN - GuruInversor")
    print("=" * 60)
    
    test_suite = FinalIntegrationTests()
    success = test_suite.run_all_final_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ✅ INTEGRACIÓN COMPLETA: TODAS LAS PRUEBAS PASARON")
        print("🚀 Sistema validado y listo para continuar con MODEL-001")
        print("📝 Fase 1 - Fundación: 100% COMPLETADA")
    else:
        print("⚠️ ❌ INTEGRACIÓN: Algunas pruebas fallaron")
        print("📝 Revisar logs para resolver problemas")
    
    return success


if __name__ == "__main__":
    run_final_integration_tests()
