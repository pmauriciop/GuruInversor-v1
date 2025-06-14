#!/usr/bin/env python3
"""
Script para inicializar las tablas en PostgreSQL usando la variable de entorno actual
"""

import os
import sys
from pathlib import Path

# Agregar el path del backend
backend_path = Path(__file__).parent
sys.path.append(str(backend_path))

from database.config import engine, get_database_url
from database.models import Base

def init_postgresql_tables():
    """Inicializa las tablas en PostgreSQL"""
    
    try:
        print("🐘 Inicializando tablas en PostgreSQL...")
        print(f"📊 Usando configuración: {get_database_url()}")
        
        # Crear todas las tablas
        Base.metadata.create_all(bind=engine)
        print("✅ Tablas creadas exitosamente")
        
        # Verificar que las tablas se crearon
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"📋 Tablas disponibles: {tables}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando tablas: {e}")
        return False

if __name__ == "__main__":
    success = init_postgresql_tables()
    if not success:
        sys.exit(1)
