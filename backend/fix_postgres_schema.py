#!/usr/bin/env python3
"""
Script para corregir el esquema de PostgreSQL
Convierte el campo date de TEXT a DATE en la tabla historical_data
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from datetime import datetime

def fix_postgres_schema():
    """Corrige el esquema de PostgreSQL para el campo date"""
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ ERROR: DATABASE_URL no encontrada en variables de entorno")
        return False
    
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    try:
        print("🐘 Conectando a PostgreSQL...")
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Primero, inspeccionar la estructura actual
            print("🔍 Inspeccionando estructura actual...")
            inspector = inspect(engine)
            
            # Verificar si la tabla existe
            if 'historical_data' not in inspector.get_table_names():
                print("❌ ERROR: Tabla historical_data no existe")
                return False
            
            # Obtener información de las columnas
            columns = inspector.get_columns('historical_data')
            date_column = None
            for col in columns:
                if col['name'] == 'date':
                    date_column = col
                    break
            
            if not date_column:
                print("❌ ERROR: Columna 'date' no encontrada en historical_data")
                return False
            
            print(f"📊 Columna 'date' actual: {date_column['type']}")
            
            # Si la columna ya es DATE, no hacer nada
            if 'DATE' in str(date_column['type']).upper():
                print("✅ La columna 'date' ya es de tipo DATE")
                return True
            
            # Corregir el tipo de la columna
            print("🔧 Corrigiendo tipo de columna 'date'...")
            
            # Crear una tabla temporal con el esquema correcto
            conn.execute(text("""
                CREATE TABLE historical_data_temp AS 
                SELECT 
                    id,
                    stock_id,
                    CAST(date AS DATE) as date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    adj_close,
                    created_at
                FROM historical_data
                WHERE date IS NOT NULL AND date != ''
            """))
            
            # Eliminar la tabla original
            conn.execute(text("DROP TABLE historical_data"))
            
            # Renombrar la tabla temporal
            conn.execute(text("ALTER TABLE historical_data_temp RENAME TO historical_data"))
            
            # Recrear los índices
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_historical_data_stock_id 
                ON historical_data(stock_id)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_historical_data_date 
                ON historical_data(date)
            """))
            
            # Agregar la constraint de clave primaria
            conn.execute(text("""
                ALTER TABLE historical_data 
                ADD CONSTRAINT historical_data_pkey PRIMARY KEY (id)
            """))
            
            # Verificar que la corrección funcionó
            result = conn.execute(text("""
                SELECT data_type 
                FROM information_schema.columns 
                WHERE table_name = 'historical_data' AND column_name = 'date'
            """)).fetchone()
            
            if result and result[0] == 'date':
                print("✅ Esquema corregido exitosamente")
                
                # Verificar algunos datos
                count = conn.execute(text("SELECT COUNT(*) FROM historical_data")).scalar()
                print(f"📊 Registros en historical_data: {count}")
                
                return True
            else:
                print("❌ ERROR: No se pudo corregir el esquema")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Iniciando corrección de esquema PostgreSQL")
    print("=" * 50)
    
    success = fix_postgres_schema()
    
    if success:
        print("\n✅ Corrección completada exitosamente")
        print("🚀 El endpoint de históricos debería funcionar ahora")
    else:
        print("\n❌ Error en la corrección")
    
    exit(0 if success else 1)
