#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Función corregida para obtener datos históricos desde la base de datos
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import HTTPException
from database.connection import get_database
from database.models import Stock, HistoricalData

def get_historical_data_from_db(ticker: str, days: int = 30, 
                               start_date: Optional[str] = None, 
                               end_date: Optional[str] = None):
    """
    Obtener datos históricos desde la base de datos
    
    Args:
        ticker: Símbolo del ticker
        days: Número de días hacia atrás (si no se especifican fechas)
        start_date: Fecha de inicio opcional (YYYY-MM-DD)
        end_date: Fecha de fin opcional (YYYY-MM-DD)
        
    Returns:
        List[dict]: Lista de datos históricos o None si no hay datos
    """
    try:
        db = get_database()
        ticker_upper = ticker.upper()
        
        with db.session_scope() as session:
            # Buscar el stock en la base de datos
            stock = session.query(Stock).filter(Stock.ticker == ticker_upper).first()
            
            if not stock:
                return None
            
            # Determinar fechas
            if start_date and end_date:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            else:
                end_date_obj = datetime.now().date()
                start_date_obj = end_date_obj - timedelta(days=days)
            
            # Obtener datos históricos desde la base de datos
            historical_records = session.query(HistoricalData).filter(
                HistoricalData.stock_id == stock.id,
                HistoricalData.date >= start_date_obj,
                HistoricalData.date <= end_date_obj
            ).order_by(HistoricalData.date.asc()).all()
            
            if not historical_records:
                return None
                
            # Convertir a formato de diccionario
            data_list = []
            for record in historical_records:
                data_list.append({
                    'ticker': ticker_upper,
                    'date': record.date.strftime('%Y-%m-%d'),
                    'open': float(record.open),
                    'high': float(record.high),
                    'low': float(record.low),
                    'close': float(record.close),
                    'volume': int(record.volume),
                    'adj_close': float(record.adj_close) if record.adj_close else float(record.close)
                })
            
            return data_list
            
    except Exception as e:
        print(f"Error obteniendo datos de BD para {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Test rápido
    test_ticker = "JPM"
    data = get_historical_data_from_db(test_ticker, days=30)
    if data:
        print(f"✅ Encontrados {len(data)} registros para {test_ticker}")
        print(f"   Primer registro: {data[0]['date']} - ${data[0]['close']}")
        print(f"   Último registro: {data[-1]['date']} - ${data[-1]['close']}")
    else:
        print(f"❌ No se encontraron datos para {test_ticker}")
