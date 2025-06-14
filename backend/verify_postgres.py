#!/usr/bin/env python3
"""
Script para verificar el estado de PostgreSQL en producción
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect

def verify_postgresql():
    """Verifica el estado de PostgreSQL"""
    
    # URL de PostgreSQL de Railway
    postgres_url = "postgresql://postgres:GJGCLhJCfkTzUeKjuGEIKgCQFPLNwrxZ@autorack.proxy.rlwy.net:40879/railway"
    
    try:
        print("🐘 Conectando a PostgreSQL...")
        engine = create_engine(postgres_url)
        
        # Verificar conexión
        with engine.connect() as conn:
            print("✅ Conexión exitosa")
            
            # Verificar tablas existentes
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            print(f"📊 Tablas encontradas: {tables}")
            
            # Verificar si existe la tabla stocks
            if 'stocks' in tables:
                print("✅ Tabla 'stocks' existe")
                
                # Verificar estructura de la tabla stocks
                columns = inspector.get_columns('stocks')
                print("📋 Columnas de la tabla stocks:")
                for col in columns:
                    print(f"   - {col['name']}: {col['type']}")
                
                # Contar registros
                result = conn.execute(text("SELECT COUNT(*) FROM stocks"))
                count = result.scalar()
                print(f"📈 Registros en stocks: {count}")
                
                if count > 0:
                    # Mostrar algunos registros
                    result = conn.execute(text("SELECT ticker, name, market FROM stocks LIMIT 5"))
                    rows = result.fetchall()
                    print("📋 Primeros 5 registros:")
                    for row in rows:
                        print(f"   - {row}")
                        
            else:
                print("❌ Tabla 'stocks' NO existe")
                
            # Verificar tabla historical_data
            if 'historical_data' in tables:
                print("✅ Tabla 'historical_data' existe")
                result = conn.execute(text("SELECT COUNT(*) FROM historical_data"))
                count = result.scalar()
                print(f"📈 Registros en historical_data: {count}")
            else:
                print("❌ Tabla 'historical_data' NO existe")
                
    except Exception as e:
        print(f"❌ Error conectando a PostgreSQL: {e}")
        return False
    
    return True

if __name__ == "__main__":
    verify_postgresql()
