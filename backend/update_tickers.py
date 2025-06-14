#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para actualizar stocks con nuevos tickers y poblar datos históricos
"""

import logging
from database.connection import get_database
from database.models import Stock, HistoricalData
from datetime import datetime, timedelta
import yfinance as yf

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nuevos tickers a configurar
NEW_TICKERS = {
    'GGAL.BA': {
        'name': 'Grupo Financiero Galicia',
        'sector': 'Financial Services',
        'market': 'BYMA',
        'currency': 'ARS'
    },    'YPF': {
        'name': 'YPF S.A.',
        'sector': 'Energy',
        'market': 'NYSE',
        'currency': 'USD'
    },
    'JPM': {
        'name': 'JPMorgan Chase & Co.',
        'sector': 'Financial Services',
        'market': 'NYSE',
        'currency': 'USD'
    },
    'NKE': {
        'name': 'Nike Inc.',
        'sector': 'Consumer Discretionary',
        'market': 'NYSE',
        'currency': 'USD'
    },
    'KO': {
        'name': 'The Coca-Cola Company',
        'sector': 'Consumer Staples',
        'market': 'NYSE',
        'currency': 'USD'
    }
}

def clear_existing_data():
    """Limpiar datos existentes"""
    try:
        db = get_database()
        with db.session_scope() as session:
            # Eliminar datos históricos
            session.query(HistoricalData).delete()
            session.query(Stock).delete()
            logger.info("✅ Datos existentes eliminados")
            
    except Exception as e:
        logger.error(f"❌ Error limpiando datos: {e}")

def add_new_stocks():
    """Agregar nuevos stocks"""
    try:
        db = get_database()
        with db.session_scope() as session:
            for ticker, info in NEW_TICKERS.items():
                # Verificar si ya existe
                existing = session.query(Stock).filter(Stock.ticker == ticker).first()
                if existing:
                    logger.info(f"Stock {ticker} ya existe, actualizando...")
                    existing.name = info['name']
                    existing.sector = info['sector']
                    existing.market = info['market']
                    existing.currency = info['currency']
                else:
                    # Crear nuevo stock
                    new_stock = Stock(
                        ticker=ticker,
                        name=info['name'],
                        sector=info['sector'],
                        market=info['market'],
                        currency=info['currency'],
                        active=True
                    )
                    session.add(new_stock)
                    logger.info(f"✅ Stock {ticker} agregado")
                    
        logger.info("✅ Stocks configurados exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error configurando stocks: {e}")

def download_historical_data(days=90):
    """Descargar datos históricos para los nuevos tickers"""
    logger.info(f"📊 Descargando datos históricos para {days} días...")
    
    try:
        db = get_database()
        
        for ticker in NEW_TICKERS.keys():
            logger.info(f"📈 Descargando datos para {ticker}...")
            
            try:
                # Usar yfinance directamente para mayor control
                stock = yf.Ticker(ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                hist = stock.history(start=start_date, end=end_date)
                
                if hist.empty:
                    logger.warning(f"⚠️ No se encontraron datos para {ticker}")
                    continue
                
                # Procesar datos
                with db.session_scope() as session:
                    # Obtener el stock de la base de datos
                    stock_obj = session.query(Stock).filter(Stock.ticker == ticker).first()
                    if not stock_obj:
                        logger.error(f"❌ Stock {ticker} no encontrado en BD")
                        continue
                    
                    # Limpiar datos históricos existentes para este ticker
                    session.query(HistoricalData).filter(
                        HistoricalData.stock_id == stock_obj.id
                    ).delete()
                    
                    records_added = 0
                    for date, row in hist.iterrows():
                        try:
                            historical_data = HistoricalData(
                                stock_id=stock_obj.id,
                                date=date.date(),
                                open=float(row['Open']),
                                high=float(row['High']),
                                low=float(row['Low']),
                                close=float(row['Close']),
                                volume=int(row['Volume']),
                                adj_close=float(row['Close'])  # Usar close como adj_close
                            )
                            session.add(historical_data)
                            records_added += 1
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Error procesando fila para {ticker}: {e}")
                            continue
                    
                    logger.info(f"✅ {ticker}: {records_added} registros agregados")
                    
            except Exception as e:
                logger.error(f"❌ Error descargando {ticker}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"❌ Error en descarga de datos: {e}")

def verify_data():
    """Verificar los datos cargados"""
    try:
        db = get_database()
        with db.session_scope() as session:
            print("\n📊 RESUMEN DE DATOS CARGADOS:")
            print("=" * 50)
            
            for ticker in NEW_TICKERS.keys():
                stock = session.query(Stock).filter(Stock.ticker == ticker).first()
                if stock:
                    count = session.query(HistoricalData).filter(
                        HistoricalData.stock_id == stock.id
                    ).count()
                    
                    if count > 0:
                        latest = session.query(HistoricalData).filter(
                            HistoricalData.stock_id == stock.id
                        ).order_by(HistoricalData.date.desc()).first()
                        
                        print(f"📈 {ticker} ({stock.name})")
                        print(f"   Registros: {count}")
                        print(f"   Último precio: ${latest.close:.2f}")
                        print(f"   Última fecha: {latest.date}")
                        print()
                    else:
                        print(f"⚠️ {ticker}: Sin datos históricos")
                else:
                    print(f"❌ {ticker}: No encontrado en BD")
                    
    except Exception as e:
        logger.error(f"❌ Error verificando datos: {e}")

def main():
    """Función principal"""
    print("🚀 Iniciando actualización de tickers y datos históricos...")
    print("=" * 60)
    
    # Paso 1: Limpiar datos existentes
    print("\n1️⃣ Limpiando datos existentes...")
    clear_existing_data()
    
    # Paso 2: Agregar nuevos stocks
    print("\n2️⃣ Configurando nuevos stocks...")
    add_new_stocks()
    
    # Paso 3: Descargar datos históricos
    print("\n3️⃣ Descargando datos históricos...")
    download_historical_data(days=90)  # 3 meses de datos
    
    # Paso 4: Verificar datos
    print("\n4️⃣ Verificando datos cargados...")
    verify_data()
    
    print("\n🎉 ¡Proceso completado!")
    print("✅ Nuevos tickers configurados con datos históricos")
    print("📊 Base de datos lista para testing con Charts")

if __name__ == "__main__":
    main()
