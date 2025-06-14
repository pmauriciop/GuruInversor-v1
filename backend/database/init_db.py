#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inicialización de Base de Datos - GuruInversor

Script para inicializar la base de datos SQLite con las tablas necesarias
y datos de ejemplo. Puede ser ejecutado independientemente o importado.
"""

import os
import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

# Añadir el directorio backend al path para imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database.connection import get_database, init_database
from database.crud import StockCRUD, HistoricalDataCRUD
from data.collector import YahooFinanceCollector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database(database_path: str = None, drop_existing: bool = False):
    """
    Configura la base de datos con las tablas necesarias.
    
    Args:
        database_path: Ruta a la base de datos (opcional)
        drop_existing: Si eliminar tablas existentes
    """
    try:
        logger.info("Iniciando configuración de base de datos...")
        
        # Configurar URL de base de datos
        if database_path:
            database_url = f"sqlite:///{database_path}"
        else:
            # Usar ruta por defecto
            data_dir = backend_dir.parent / "data"
            data_dir.mkdir(exist_ok=True)
            database_url = f"sqlite:///{data_dir}/guru_inversor.db"
        
        # Inicializar base de datos
        init_database(database_url=database_url, drop_existing=drop_existing)
        
        # Verificar inicialización
        database = get_database()
        info = database.get_database_info()
        
        logger.info(f"Base de datos configurada exitosamente:")
        logger.info(f"  - URL: {info['database_url']}")
        logger.info(f"  - Tablas: {info['total_tables']}")
        for table in info['tables']:
            logger.info(f"    • {table['name']}: {table['record_count']} registros")
        
        if info.get('database_size_bytes'):
            size_mb = info['database_size_bytes'] / (1024 * 1024)
            logger.info(f"  - Tamaño: {size_mb:.2f} MB")
        
        return database
        
    except Exception as e:
        logger.error(f"Error configurando base de datos: {e}")
        raise


def add_sample_stocks(database):
    """
    Añade acciones de ejemplo a la base de datos.
    
    Args:
        database: Instancia de base de datos
    """
    try:
        logger.info("Añadiendo acciones de ejemplo...")
        
        # Acciones de ejemplo
        sample_stocks = [
            {
                'ticker': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'market': 'US',
                'currency': 'USD'
            },
            {
                'ticker': 'GOOGL',
                'name': 'Alphabet Inc.',
                'sector': 'Technology',
                'market': 'US',
                'currency': 'USD'
            },
            {
                'ticker': 'MSFT',
                'name': 'Microsoft Corporation',
                'sector': 'Technology',
                'market': 'US',
                'currency': 'USD'
            },
            {
                'ticker': 'TSLA',
                'name': 'Tesla, Inc.',
                'sector': 'Automotive',
                'market': 'US',
                'currency': 'USD'
            },
            {
                'ticker': 'NVDA',
                'name': 'NVIDIA Corporation',
                'sector': 'Technology',
                'market': 'US',
                'currency': 'USD'
            }
        ]
        
        stock_crud = StockCRUD()
        
        with database.session_scope() as session:
            for stock_data in sample_stocks:
                # Verificar si ya existe
                existing = stock_crud.get_by_ticker(session, stock_data['ticker'])
                if existing:
                    logger.info(f"Acción {stock_data['ticker']} ya existe")
                else:
                    # Crear nueva acción
                    stock = stock_crud.create(session, **stock_data)
                    logger.info(f"Acción creada: {stock_data['ticker']} - {stock_data['name']}")
        
        logger.info("Configuración de acciones completada.")
        
    except Exception as e:
        logger.error(f"Error añadiendo acciones de ejemplo: {e}")
        raise


def download_sample_data(database, tickers: list = None, days: int = 30):
    """
    Descarga datos históricos de ejemplo.
    
    Args:
        database: Instancia de base de datos
        tickers: Lista de tickers (opcional, usa algunos por defecto)
        days: Número de días de historia a descargar
    """
    try:
        if tickers is None:
            tickers = ['AAPL', 'GOOGL']  # Solo algunos para la demostración
        
        logger.info(f"Descargando datos históricos para {len(tickers)} acciones...")
        
        collector = YahooFinanceCollector()
        stock_crud = StockCRUD()
        historical_crud = HistoricalDataCRUD()
        
        # Calcular fechas
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        with database.session_scope() as session:
            for ticker in tickers:
                try:
                    # Obtener stock de la base de datos
                    stock = stock_crud.get_by_ticker(session, ticker)
                    if not stock:
                        logger.warning(f"Acción {ticker} no encontrada en base de datos")
                        continue
                    
                    # Descargar datos históricos
                    logger.info(f"Descargando datos para {ticker}...")
                    df = collector.get_historical_data(ticker, start_date, end_date)
                    
                    if df is not None and not df.empty:
                        # Convertir DataFrame a lista de diccionarios
                        records = []
                        for idx, row in df.iterrows():
                            records.append({
                                'date': idx.date(),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'close': float(row['Close']),
                                'volume': int(row['Volume']),
                                'adj_close': float(row['Adj Close']) if 'Adj Close' in row else None
                            })
                        
                        # Guardar en base de datos
                        created_count = historical_crud.create_batch(session, stock.id, records)
                        logger.info(f"Guardados {created_count} registros para {ticker}")
                    
                    else:
                        logger.warning(f"No se pudieron obtener datos para {ticker}")
                
                except Exception as e:
                    logger.error(f"Error procesando {ticker}: {e}")
                    continue
        
        logger.info("Descarga de datos históricos completada")
        
    except Exception as e:
        logger.error(f"Error descargando datos de ejemplo: {e}")
        raise


def main():
    """Función principal del script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inicializar base de datos GuruInversor')
    parser.add_argument('--database-path', type=str, help='Ruta a la base de datos')
    parser.add_argument('--drop-existing', action='store_true', 
                       help='Eliminar tablas existentes')
    parser.add_argument('--skip-sample-data', action='store_true',
                       help='No descargar datos de ejemplo')
    parser.add_argument('--sample-days', type=int, default=60,
                       help='Días de datos históricos a descargar')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Logging detallado')
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Configurar base de datos
        database = setup_database(
            database_path=args.database_path,
            drop_existing=args.drop_existing
        )
        
        # Añadir acciones de ejemplo
        add_sample_stocks(database)
        
        # Descargar datos de ejemplo si se solicita
        if not args.skip_sample_data:
            # Usar solo los tickers como strings
            sample_tickers = ['AAPL', 'GOOGL']  # Solo primeras 2 para demo
            download_sample_data(database, sample_tickers, args.sample_days)
        
        logger.info("✅ Inicialización de base de datos completada exitosamente")
        
        # Mostrar resumen final
        info = database.get_database_info()
        print("\n" + "="*50)
        print("RESUMEN DE BASE DE DATOS")
        print("="*50)
        print(f"📁 Base de datos: {info['database_url']}")
        print(f"📊 Tablas creadas: {info['total_tables']}")
        for table in info['tables']:
            print(f"   • {table['name']}: {table['record_count']} registros")
        if info.get('database_size_bytes'):
            size_mb = info['database_size_bytes'] / (1024 * 1024)
            print(f"💾 Tamaño: {size_mb:.2f} MB")
        print("="*50)
        
    except Exception as e:
        logger.error(f"❌ Error en inicialización: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
