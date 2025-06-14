# -*- coding: utf-8 -*-
"""
Configuración de Conexión a Base de Datos - GuruInversor

Maneja la conexión a SQLite y proporciona sesiones de base de datos.
Incluye configuración para diferentes entornos y logging.
"""

import os
import logging
from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base
from .config import engine, SessionLocal

# Configurar logging
logger = logging.getLogger(__name__)


class Database:
    """
    Clase para manejar la conexión y configuración de la base de datos.
    
    Usa la configuración de config.py que soporta tanto PostgreSQL como SQLite.
    """
    
    def __init__(self):
        """
        Inicializa la conexión usando la configuración de config.py.
        """
        # Usar el engine y SessionLocal de config.py
        self.engine = engine
        self.SessionLocal = SessionLocal
        
        # Determinar tipo de base de datos
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            self.database_url = database_url
            self.db_type = "postgresql"
        else:
            self.database_url = "sqlite:///./data/guru_inversor.db"
            self.db_type = "sqlite"
            # Solo configurar eventos SQLite si estamos usando SQLite
            self._setup_sqlite_events()
        
        logger.info(f"Base de datos configurada: {self.db_type}")
    
    def _setup_sqlite_events(self):
        """Configura eventos específicos para SQLite."""
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Configura pragmas de SQLite para mejor rendimiento y seguridad."""
            cursor = dbapi_connection.cursor()
            
            # Habilitar foreign keys
            cursor.execute("PRAGMA foreign_keys=ON")
            
            # Configurar journal mode para mejor concurrencia
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # Configurar synchronous para balance entre velocidad y seguridad
            cursor.execute("PRAGMA synchronous=NORMAL")
            
            # Configurar cache size (en páginas de 4KB)
            cursor.execute("PRAGMA cache_size=10000")  # 40MB de cache
            
            # Configurar timeout para locks
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 segundos
            
            cursor.close()
    
    def init_db(self, drop_existing: bool = False):
        """
        Inicializa la base de datos creando todas las tablas.
        
        Args:
            drop_existing: Si eliminar las tablas existentes antes de crear nuevas
        """
        try:
            if drop_existing:
                logger.warning("Eliminando todas las tablas existentes")
                Base.metadata.drop_all(bind=self.engine)
            
            # Crear todas las tablas
            Base.metadata.create_all(bind=self.engine)
            logger.info("Base de datos inicializada correctamente")
            
            # Verificar que las tablas se crearon
            self._verify_tables()
            
        except Exception as e:
            logger.error(f"Error al inicializar base de datos: {e}")
            raise
    
    def _verify_tables(self):
        """Verifica que todas las tablas necesarias existan."""
        expected_tables = ['stocks', 'historical_data', 'predictions', 'trained_models']
        
        with self.get_session() as session:
            # Obtener nombres de tablas existentes
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            existing_tables = [row[0] for row in result.fetchall()]
            
            # Verificar que todas las tablas esperadas existan
            missing_tables = set(expected_tables) - set(existing_tables)
            if missing_tables:
                raise RuntimeError(f"Faltan tablas en la base de datos: {missing_tables}")
            
            logger.info(f"Verificación completada. Tablas encontradas: {existing_tables}")
    
    def get_session(self) -> Session:
        """
        Obtiene una nueva sesión de base de datos.
        
        Returns:
            Session: Nueva sesión de SQLAlchemy
        """
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager para manejar sesiones de base de datos con transacciones.
        
        Yields:
            Session: Sesión de base de datos que se commitea automáticamente
                    o hace rollback en caso de error
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error en transacción de base de datos: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Cierra la conexión a la base de datos."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Conexión a base de datos cerrada")
    
    def get_database_info(self) -> dict:
        """
        Obtiene información sobre la base de datos.
        
        Returns:
            dict: Información de la base de datos
        """
        try:
            with self.session_scope() as session:
                # Obtener información de tablas
                result = session.execute(text("""
                    SELECT 
                        name,
                        sql
                    FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """))
                
                tables = []
                for name, sql in result.fetchall():
                    # Contar registros en cada tabla
                    count_result = session.execute(text(f"SELECT COUNT(*) FROM {name}"))
                    record_count = count_result.scalar()
                    
                    tables.append({
                        'name': name,
                        'record_count': record_count,
                        'sql': sql
                    })
                
                # Obtener tamaño del archivo de base de datos
                db_size = None
                if self.database_url.startswith('sqlite:///'):
                    db_path = self.database_url.replace('sqlite:///', '')
                    if os.path.exists(db_path):
                        db_size = os.path.getsize(db_path)
                
                return {
                    'database_url': self.database_url,
                    'tables': tables,
                    'database_size_bytes': db_size,
                    'total_tables': len(tables)
                }
                
        except Exception as e:
            logger.error(f"Error al obtener información de base de datos: {e}")
            return {
                'database_url': self.database_url,
                'error': str(e)
            }


# Instancia global de base de datos
_database_instance: Optional[Database] = None


def get_database() -> Database:
    """
    Obtiene la instancia global de base de datos.
    
    Returns:
        Database: Instancia de base de datos
    """
    global _database_instance
    
    if _database_instance is None:
        _database_instance = Database()
    
    return _database_instance


def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency injection para FastAPI.
    
    Yields:
        Session: Sesión de base de datos
    """
    database = get_database()
    with database.session_scope() as session:
        yield session


def init_database(drop_existing: bool = False):
    """
    Función de conveniencia para inicializar la base de datos.
    
    Args:
        drop_existing: Si eliminar tablas existentes
    """
    database = get_database()
    database.init_db(drop_existing=drop_existing)
