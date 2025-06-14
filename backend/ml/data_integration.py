#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades de Integración de Datos - GuruInversor

Módulo para integrar el procesamiento de datos con la base de datos
y el recolector de datos Yahoo Finance.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Añadir el directorio backend al path para imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database.crud import StockCRUD, HistoricalDataCRUD, PredictionCRUD, ModelCRUD
from database.connection import Database
from data.collector import YahooFinanceCollector
from ml.preprocessor import DataProcessor, ProcessingConfig, process_stock_data

logger = logging.getLogger(__name__)


class DataIntegrator:
    """
    Integrador principal que combina recolección, almacenamiento y procesamiento de datos.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializar integrador de datos.
        
        Args:
            db_path: Ruta a la base de datos SQLite (opcional)
        """
        self.db = Database(db_path)
        self.stock_crud = StockCRUD()
        self.historical_crud = HistoricalDataCRUD()
        self.prediction_crud = PredictionCRUD()
        self.model_crud = ModelCRUD()
        self.collector = YahooFinanceCollector()
        self.processor = DataProcessor()
        
    def fetch_and_store_stock_data(self, 
                                  ticker: str, 
                                  period: str = "2y",
                                  update_existing: bool = True) -> bool:
        """
        Recolectar datos de Yahoo Finance y almacenar en base de datos.
        
        Args:
            ticker: Símbolo de la acción
            period: Período de datos ("1y", "2y", "5y", "max")
            update_existing: Si actualizar datos existentes
            
        Returns:
            True si fue exitoso, False en caso contrario
        """
        try:
            logger.info(f"Recolectando datos para {ticker} (período: {period})")
            
            # Verificar si el stock existe en la base de datos
            with self.db.get_session() as session:
                stock = self.stock_crud.get_by_ticker(session, ticker)
                if not stock:
                    # Obtener información del stock
                    stock_info = self.collector.get_stock_info(ticker)
                    if not stock_info:
                        logger.error(f"No se pudo obtener información para {ticker}")
                        return False
                    
                    # Crear nuevo stock en la base de datos
                    stock = self.stock_crud.create(
                        session=session,
                        ticker=ticker,
                        name=stock_info.get('shortName', ticker),
                        sector=stock_info.get('sector', 'Unknown')
                    )
                    logger.info(f"Stock creado en BD: {ticker}")
            
            # Recolectar datos históricos
            historical_data = self.collector.get_historical_data(ticker, period)
            if historical_data.empty:
                logger.error(f"No se pudieron obtener datos históricos para {ticker}")
                return False
            
            # Preparar datos para inserción
            records_data = []
            for _, row in historical_data.iterrows():
                record = {
                    'stock_id': stock.id,
                    'date': row.name.date(),  # row.name es el datetime index
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'adj_close': float(row['Adj Close']) if 'Adj Close' in row else float(row['Close'])
                }
                records_data.append(record)
            
            # Insertar datos históricos (batch)
            with self.db.get_session() as session:
                if update_existing:
                    # Eliminar datos existentes para actualizar
                    deleted_count = self.historical_crud.delete_by_stock(session, stock.id)
                    if deleted_count > 0:
                        logger.info(f"Eliminados {deleted_count} registros existentes para {ticker}")
                
                inserted_count = self.historical_crud.bulk_create(session, records_data)
                logger.info(f"Insertados {inserted_count} registros históricos para {ticker}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recolectando y almacenando datos para {ticker}: {e}")
            return False
    
    def get_processed_data(self, 
                          ticker: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          config: Optional[ProcessingConfig] = None) -> Tuple[np.ndarray, np.ndarray, DataProcessor]:
        """
        Obtener datos procesados para entrenamiento desde la base de datos.
        
        Args:
            ticker: Símbolo de la acción
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            config: Configuración de procesamiento
            
        Returns:
            Tuple[np.ndarray, np.ndarray, DataProcessor]: (X_sequences, y_targets, processor)
        """
        try:
            logger.info(f"Obteniendo datos procesados para {ticker}")
            
            # Obtener datos históricos de la base de datos
            with self.db.get_session() as session:
                stock = self.stock_crud.get_by_ticker(session, ticker)
                if not stock:
                    logger.error(f"Stock {ticker} no encontrado en la base de datos")
                    return np.array([]), np.array([]), DataProcessor()
                
                historical_data = self.historical_crud.get_by_stock(
                    session, stock.id, start_date, end_date
                )
            
            if not historical_data:
                logger.error(f"No se encontraron datos históricos para {ticker}")
                return np.array([]), np.array([]), DataProcessor()
            
            # Convertir a DataFrame
            df = pd.DataFrame([{
                'date': record.date,
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume,
                'adj_close': record.adj_close
            } for record in historical_data])
            
            # Procesar datos
            X_sequences, y_targets, processor = process_stock_data(df, config, add_features=True)
            
            logger.info(f"Datos procesados exitosamente para {ticker}")
            return X_sequences, y_targets, processor
            
        except Exception as e:
            logger.error(f"Error obteniendo datos procesados para {ticker}: {e}")
            return np.array([]), np.array([]), DataProcessor()
    
    def get_data_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Obtener resumen de datos disponibles para una acción.
        
        Args:
            ticker: Símbolo de la acción
            
        Returns:
            Diccionario con resumen de datos
        """
        try:
            with self.db.get_session() as session:
                stock = self.stock_crud.get_by_ticker(session, ticker)
                if not stock:
                    return {'error': f'Stock {ticker} no encontrado'}
                
                # Obtener estadísticas de datos históricos
                historical_count = len(self.historical_crud.get_by_stock(session, stock.id))
                latest_data = self.historical_crud.get_latest(session, stock.id)
                date_range = self.historical_crud.get_date_range(session, stock.id)
                
                summary = {
                    'ticker': ticker,
                    'name': stock.name,
                    'sector': stock.sector,
                    'is_active': stock.active,
                    'total_records': historical_count,
                    'latest_date': latest_data.date if latest_data else None,
                    'earliest_date': date_range[0] if date_range else None,
                    'latest_price': latest_data.close if latest_data else None,
                    'data_span_days': None
                }
                
                if date_range and len(date_range) == 2:
                    data_span = (date_range[1] - date_range[0]).days
                    summary['data_span_days'] = data_span
                
                return summary
                
        except Exception as e:
            logger.error(f"Error obteniendo resumen para {ticker}: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Cerrar conexiones."""
        try:
            self.db.close()
        except Exception as e:
            logger.error(f"Error cerrando conexiones: {e}")


def initialize_sample_data(integrator: DataIntegrator, 
                          sample_tickers: List[str] = None) -> Dict[str, bool]:
    """
    Inicializar datos de ejemplo para desarrollo y pruebas.
    
    Args:
        integrator: Instancia del integrador de datos
        sample_tickers: Lista de tickers de ejemplo
        
    Returns:
        Diccionario con resultados de inicialización
    """
    if sample_tickers is None:
        sample_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    logger.info(f"Inicializando datos de ejemplo para: {sample_tickers}")
    
    results = {}
    for ticker in sample_tickers:
        logger.info(f"Inicializando {ticker}...")
        success = integrator.fetch_and_store_stock_data(ticker, period="2y")
        results[ticker] = success
        
        if success:
            logger.info(f"✅ {ticker} inicializado exitosamente")
        else:
            logger.error(f"❌ Error inicializando {ticker}")
    
    return results


if __name__ == "__main__":
    # Ejemplo de uso
    print("🔗 Integrador de Datos - GuruInversor")
    print("=" * 50)
    
    # Crear integrador
    integrator = DataIntegrator()
    
    try:
        # Ejemplo: obtener datos para AAPL
        ticker = "AAPL"
        print(f"📊 Obteniendo resumen de datos para {ticker}...")
        
        summary = integrator.get_data_summary(ticker)
        if 'error' not in summary:
            print(f"   Registros totales: {summary['total_records']}")
            print(f"   Fecha más reciente: {summary['latest_date']}")
            print(f"   Precio más reciente: ${summary['latest_price']:.2f}")
        else:
            print(f"   {summary['error']}")
            
            # Si no hay datos, inicializar
            print(f"🔄 Inicializando datos para {ticker}...")
            success = integrator.fetch_and_store_stock_data(ticker, "1y")
            if success:
                print(f"✅ Datos inicializados para {ticker}")
                summary = integrator.get_data_summary(ticker)
                print(f"   Nuevos registros: {summary['total_records']}")
            else:
                print(f"❌ Error inicializando datos para {ticker}")
        
    finally:
        integrator.close()
