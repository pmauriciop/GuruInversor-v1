#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas de Base de Datos - GuruInversor

Suite completa de pruebas para verificar el funcionamiento correcto
de la base de datos SQLite y las operaciones CRUD.
"""

import os
import sys
import tempfile
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

# Añadir el directorio backend al path para imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database.connection import Database
from database.models import Stock, HistoricalData, Prediction, TrainedModel
from database.crud import StockCRUD, HistoricalDataCRUD, PredictionCRUD, ModelCRUD

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseTestSuite:
    """Suite de pruebas para la base de datos."""
    
    def __init__(self):
        self.test_results = []
        self.database = None
        self.temp_db_path = None
    
    def setup(self):
        """Configura una base de datos temporal para las pruebas."""
        try:
            # Crear archivo temporal para la base de datos
            temp_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
            os.close(temp_fd)  # Cerrar el descriptor de archivo
            
            # Configurar base de datos temporal
            database_url = f"sqlite:///{self.temp_db_path}"
            self.database = Database(database_url=database_url, echo=False)
            self.database.init_db(drop_existing=True)
            
            logger.info(f"Base de datos de prueba configurada: {self.temp_db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error configurando base de datos de prueba: {e}")
            return False
    
    def teardown(self):
        """Limpia la base de datos temporal."""
        try:
            if self.database:
                self.database.close()
            
            if self.temp_db_path and os.path.exists(self.temp_db_path):
                os.unlink(self.temp_db_path)
                logger.info("Base de datos temporal eliminada")
            
        except Exception as e:
            logger.warning(f"Error limpiando base de datos temporal: {e}")
    
    def run_test(self, test_name: str, test_func):
        """Ejecuta una prueba individual."""
        try:
            logger.info(f"Ejecutando prueba: {test_name}")
            test_func()
            self.test_results.append((test_name, True, None))
            logger.info(f"✅ {test_name} - PASÓ")
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.test_results.append((test_name, False, error_msg))
            logger.error(f"❌ {test_name} - FALLÓ: {error_msg}")
            return False
    
    def test_database_connection(self):
        """Prueba la conexión a la base de datos."""
        # Verificar que la base de datos se inicializó correctamente
        assert self.database is not None, "Base de datos no inicializada"
        
        # Verificar información de la base de datos
        info = self.database.get_database_info()
        assert 'tables' in info, "No se pudo obtener información de tablas"
        assert info['total_tables'] >= 4, "No se crearon todas las tablas esperadas"
        
        # Verificar que todas las tablas esperadas existen
        expected_tables = {'stocks', 'historical_data', 'predictions', 'trained_models'}
        existing_tables = {table['name'] for table in info['tables']}
        missing_tables = expected_tables - existing_tables
        assert not missing_tables, f"Faltan tablas: {missing_tables}"
    
    def test_stock_crud_operations(self):
        """Prueba las operaciones CRUD para acciones."""
        stock_crud = StockCRUD()
        
        with self.database.session_scope() as session:
            # Test CREATE
            stock = stock_crud.create(
                session=session,
                ticker="AAPL",
                name="Apple Inc.",
                sector="Technology"
            )
            assert stock.id is not None, "Stock no tiene ID después de crear"
            assert stock.ticker == "AAPL", "Ticker no coincide"
            assert stock.name == "Apple Inc.", "Nombre no coincide"
            
            # Test READ by ticker
            found_stock = stock_crud.get_by_ticker(session, "AAPL")
            assert found_stock is not None, "No se pudo encontrar stock por ticker"
            assert found_stock.id == stock.id, "IDs no coinciden"
            
            # Test READ by ID
            found_by_id = stock_crud.get_by_id(session, stock.id)
            assert found_by_id is not None, "No se pudo encontrar stock por ID"
            assert found_by_id.ticker == "AAPL", "Ticker no coincide en búsqueda por ID"
            
            # Test UPDATE
            updated_stock = stock_crud.update(
                session=session,
                stock_id=stock.id,
                name="Apple Inc. Updated",
                sector="Technology & Innovation"
            )
            assert updated_stock is not None, "No se pudo actualizar stock"
            assert updated_stock.name == "Apple Inc. Updated", "Nombre no se actualizó"
            assert updated_stock.sector == "Technology & Innovation", "Sector no se actualizó"
            
            # Test GET ALL
            all_stocks = stock_crud.get_all(session)
            assert len(all_stocks) >= 1, "No se obtuvieron stocks en get_all"
            assert any(s.ticker == "AAPL" for s in all_stocks), "AAPL no está en la lista"
            
            # Test SOFT DELETE
            deleted = stock_crud.delete(session, stock.id)
            assert deleted is True, "No se pudo eliminar stock"
            
            # Verificar que está marcado como inactivo
            inactive_stock = stock_crud.get_by_id(session, stock.id)
            assert inactive_stock.active is False, "Stock no se marcó como inactivo"
    
    def test_historical_data_crud(self):
        """Prueba las operaciones CRUD para datos históricos."""
        stock_crud = StockCRUD()
        historical_crud = HistoricalDataCRUD()
        
        with self.database.session_scope() as session:
            # Crear acción de prueba
            stock = stock_crud.create(session, "GOOGL", "Alphabet Inc.")
            
            # Crear datos históricos de prueba
            test_data = []
            base_date = date.today() - timedelta(days=10)
            
            for i in range(5):
                test_data.append({
                    'date': base_date + timedelta(days=i),
                    'open': 100.0 + i,
                    'high': 105.0 + i,
                    'low': 95.0 + i,
                    'close': 102.0 + i,
                    'volume': 1000000 + i * 10000,
                    'adj_close': 102.0 + i
                })
            
            # Test CREATE BATCH
            created_count = historical_crud.create_batch(session, stock.id, test_data)
            assert created_count == 5, f"Se esperaban 5 registros, se crearon {created_count}"
            
            # Test GET BY STOCK
            historical_data = historical_crud.get_by_stock(session, stock.id)
            assert len(historical_data) == 5, "No se obtuvieron todos los registros históricos"
            
            # Test GET LATEST
            latest = historical_crud.get_latest(session, stock.id)
            assert latest is not None, "No se obtuvo el último registro"
            assert latest.date == base_date + timedelta(days=4), "La fecha del último registro no es correcta"
            
            # Test GET DATE RANGE
            date_range = historical_crud.get_date_range(session, stock.id)
            assert date_range is not None, "No se obtuvo rango de fechas"
            assert date_range[0] == base_date, "Fecha mínima incorrecta"
            assert date_range[1] == base_date + timedelta(days=4), "Fecha máxima incorrecta"
            
            # Test GET WITH DATE FILTER
            filtered_data = historical_crud.get_by_stock(
                session, stock.id,
                start_date=base_date + timedelta(days=2),
                end_date=base_date + timedelta(days=3)
            )
            assert len(filtered_data) == 2, "Filtro de fechas no funcionó correctamente"
    
    def test_prediction_crud(self):
        """Prueba las operaciones CRUD para predicciones."""
        stock_crud = StockCRUD()
        prediction_crud = PredictionCRUD()
        
        with self.database.session_scope() as session:
            # Crear acción de prueba
            stock = stock_crud.create(session, "TSLA", "Tesla Inc.")
            
            # Test CREATE PREDICTION
            prediction_date = date.today() + timedelta(days=1)
            prediction = prediction_crud.create(
                session=session,
                stock_id=stock.id,
                prediction_date=prediction_date,
                predicted_price=250.50,
                confidence=0.85,
                model_version="v1.0"
            )
            assert prediction.id is not None, "Predicción no tiene ID"
            assert prediction.predicted_price == 250.50, "Precio predicho no coincide"
            assert prediction.confidence == 0.85, "Confianza no coincide"
            
            # Test GET BY STOCK
            predictions = prediction_crud.get_by_stock(session, stock.id)
            assert len(predictions) == 1, "No se obtuvieron las predicciones"
            assert predictions[0].id == prediction.id, "ID de predicción no coincide"
            
            # Test UPDATE ACTUAL PRICE
            updated_prediction = prediction_crud.update_actual_price(
                session, prediction.id, 252.30
            )
            assert updated_prediction is not None, "No se pudo actualizar precio real"
            assert updated_prediction.actual_price == 252.30, "Precio real no se actualizó"
    
    def test_trained_model_crud(self):
        """Prueba las operaciones CRUD para modelos entrenados."""
        stock_crud = StockCRUD()
        model_crud = ModelCRUD()
        
        with self.database.session_scope() as session:
            # Crear acción de prueba
            stock = stock_crud.create(session, "NVDA", "NVIDIA Corporation")
            
            # Test CREATE MODEL
            model = model_crud.create(
                session=session,
                stock_id=stock.id,
                model_path="/models/nvda_v1.h5",
                version="v1.0",
                accuracy=0.92,
                loss=0.08,
                rmse=2.34,
                mae=1.87,
                training_samples=1000,
                epochs_trained=50
            )
            assert model.id is not None, "Modelo no tiene ID"
            assert model.is_active is True, "Modelo no está marcado como activo"
            assert model.accuracy == 0.92, "Accuracy no coincide"
            
            # Test GET ACTIVE MODEL
            active_model = model_crud.get_active_model(session, stock.id)
            assert active_model is not None, "No se encontró modelo activo"
            assert active_model.id == model.id, "ID de modelo activo no coincide"
            
            # Test CREATE SECOND MODEL (debería desactivar el primero)
            model2 = model_crud.create(
                session=session,
                stock_id=stock.id,
                model_path="/models/nvda_v2.h5",
                version="v2.0",
                accuracy=0.94,
                loss=0.06
            )
            
            # Verificar que el segundo modelo es ahora el activo
            new_active = model_crud.get_active_model(session, stock.id)
            assert new_active.id == model2.id, "El nuevo modelo no es el activo"
            
            # Verificar que el primer modelo ya no es activo
            session.refresh(model)  # Recargar desde BD
            assert model.is_active is False, "El primer modelo sigue activo"
            
            # Test GET ALL BY STOCK
            all_models = model_crud.get_all_by_stock(session, stock.id)
            assert len(all_models) == 2, "No se obtuvieron todos los modelos"
    
    def test_data_validation(self):
        """Prueba las validaciones de datos."""
        stock_crud = StockCRUD()
        
        with self.database.session_scope() as session:
            # Test validación de ticker
            try:
                stock_crud.create(session, "", "Empty Ticker")
                assert False, "Debería fallar con ticker vacío"
            except ValueError:
                pass  # Esperado
            
            try:
                stock_crud.create(session, "INVALID@TICKER", "Invalid Ticker")
                assert False, "Debería fallar con ticker inválido"
            except ValueError:
                pass  # Esperado
            
            # Test validación de precios negativos
            stock = stock_crud.create(session, "TEST", "Test Stock")
            
            historical_crud = HistoricalDataCRUD()
            try:
                historical_crud.create_batch(session, stock.id, [{
                    'date': date.today(),
                    'open': -10.0,  # Precio negativo
                    'high': 100.0,
                    'low': 90.0,
                    'close': 95.0,
                    'volume': 1000
                }])
                assert False, "Debería fallar con precio negativo"
            except ValueError:
                pass  # Esperado
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas."""
        logger.info("="*60)
        logger.info("INICIANDO SUITE DE PRUEBAS DE BASE DE DATOS")
        logger.info("="*60)
        
        if not self.setup():
            logger.error("No se pudo configurar la base de datos de prueba")
            return False
        
        try:
            # Ejecutar todas las pruebas
            tests = [
                ("Conexión a Base de Datos", self.test_database_connection),
                ("CRUD de Acciones", self.test_stock_crud_operations),
                ("CRUD de Datos Históricos", self.test_historical_data_crud),
                ("CRUD de Predicciones", self.test_prediction_crud),
                ("CRUD de Modelos Entrenados", self.test_trained_model_crud),
                ("Validación de Datos", self.test_data_validation)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                if self.run_test(test_name, test_func):
                    passed += 1
            
            # Mostrar resumen
            logger.info("="*60)
            logger.info("RESUMEN DE PRUEBAS")
            logger.info("="*60)
            
            for test_name, success, error in self.test_results:
                status = "✅ PASÓ" if success else "❌ FALLÓ"
                logger.info(f"{status} - {test_name}")
                if error:
                    logger.info(f"      Error: {error}")
            
            logger.info("="*60)
            logger.info(f"RESULTADO FINAL: {passed}/{total} pruebas pasaron")
            logger.info("="*60)
            
            return passed == total
            
        finally:
            self.teardown()


def main():
    """Función principal del script de pruebas."""
    test_suite = DatabaseTestSuite()
    
    try:
        success = test_suite.run_all_tests()
        
        if success:
            print("\n🎉 ¡Todas las pruebas de base de datos pasaron exitosamente!")
            sys.exit(0)
        else:
            print("\n💥 Algunas pruebas fallaron. Revisa los logs para más detalles.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error ejecutando pruebas: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
