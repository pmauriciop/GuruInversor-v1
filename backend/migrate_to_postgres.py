#!/usr/bin/env python3
"""
Script de migración de datos de SQLite a PostgreSQL para producción
"""

import os
import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

def migrate_sqlite_to_postgresql():
    """Migra datos de SQLite local a PostgreSQL en producción"""
    
    # URLs de base de datos
    sqlite_path = "data/guru_inversor.db"
    postgres_url = os.getenv("DATABASE_URL")
    
    if not postgres_url:
        print("❌ ERROR: DATABASE_URL no encontrada en variables de entorno")
        return False
    
    if not os.path.exists(sqlite_path):
        print(f"❌ ERROR: No se encuentra la base de datos SQLite en {sqlite_path}")
        return False
    
    try:
        # Conectar a SQLite
        print("📱 Conectando a SQLite local...")
        sqlite_conn = sqlite3.connect(sqlite_path)
        
        # Conectar a PostgreSQL
        print("🐘 Conectando a PostgreSQL...")
        if postgres_url.startswith("postgres://"):
            postgres_url = postgres_url.replace("postgres://", "postgresql://", 1)
        pg_engine = create_engine(postgres_url)
        
        # Leer datos de SQLite
        print("📊 Leyendo datos de SQLite...")
        
        # Tabla de stocks
        stocks_df = pd.read_sql_query("SELECT * FROM stocks", sqlite_conn)
        print(f"   - Stocks: {len(stocks_df)} registros")
        
        # Tabla de historical_data
        historical_df = pd.read_sql_query("SELECT * FROM historical_data", sqlite_conn)
        print(f"   - Historical data: {len(historical_df)} registros")
        
        # Verificar datos
        if len(stocks_df) == 0 or len(historical_df) == 0:
            print("❌ ERROR: No hay datos para migrar")
            return False
        
        # Crear tablas en PostgreSQL
        print("🏗️  Creando esquema en PostgreSQL...")
        
        with pg_engine.connect() as conn:
            # Crear tabla stocks
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL UNIQUE,
                    name VARCHAR(255),
                    sector VARCHAR(255),
                    market_cap DECIMAL,
                    current_price DECIMAL,
                    last_update TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Crear tabla historical_data
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(15,2),
                    high DECIMAL(15,2),
                    low DECIMAL(15,2),
                    close DECIMAL(15,2),
                    volume BIGINT,
                    adj_close DECIMAL(15,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            """))
            
            conn.commit()
        
        # Migrar datos
        print("📤 Migrando datos a PostgreSQL...")
        
        # Migrar stocks
        stocks_df.to_sql('stocks', pg_engine, if_exists='replace', index=False, method='multi')
        print(f"   ✅ Stocks migrados: {len(stocks_df)}")
        
        # Migrar historical_data
        historical_df.to_sql('historical_data', pg_engine, if_exists='replace', index=False, method='multi')
        print(f"   ✅ Historical data migrado: {len(historical_df)}")
        
        # Verificar migración
        print("🔍 Verificando migración...")
        with pg_engine.connect() as conn:
            stocks_count = conn.execute(text("SELECT COUNT(*) FROM stocks")).scalar()
            historical_count = conn.execute(text("SELECT COUNT(*) FROM historical_data")).scalar()
            
            print(f"   - Stocks en PostgreSQL: {stocks_count}")
            print(f"   - Historical data en PostgreSQL: {historical_count}")
            
            if stocks_count == len(stocks_df) and historical_count == len(historical_df):
                print("✅ Migración completada exitosamente!")
                return True
            else:
                print("❌ ERROR: Los datos migrados no coinciden")
                return False
                
    except Exception as e:
        print(f"❌ ERROR durante la migración: {str(e)}")
        return False
    
    finally:
        if 'sqlite_conn' in locals():
            sqlite_conn.close()

def create_migration_summary():
    """Crea un resumen de la migración"""
    summary = {
        "migration_date": datetime.now().isoformat(),
        "source": "SQLite (local)",
        "destination": "PostgreSQL (Railway)",
        "status": "completed" if migrate_sqlite_to_postgresql() else "failed"
    }
    
    print(f"\n📋 Resumen de migración:")
    for key, value in summary.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    print("🚀 Iniciando migración GuruInversor SQLite → PostgreSQL")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        create_migration_summary()
    else:
        success = migrate_sqlite_to_postgresql()
        exit(0 if success else 1)
