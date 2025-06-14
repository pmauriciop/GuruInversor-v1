import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Environment detection
DATABASE_URL = os.getenv("DATABASE_URL")

# If no DATABASE_URL, use SQLite (development)
if not DATABASE_URL:
    SQLALCHEMY_DATABASE_URL = "sqlite:///./data/guru_inversor.db"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    # Production PostgreSQL
    # Railway provides DATABASE_URL in PostgreSQL format
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_database_url():
    """Get current database URL for logging"""
    if DATABASE_URL:
        # Don't log the full URL for security
        return "PostgreSQL (Production)"
    else:
        return "SQLite (Development)"
