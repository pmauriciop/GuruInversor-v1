# -*- coding: utf-8 -*-
"""
Operaciones CRUD para Base de Datos - GuruInversor

Define las operaciones Create, Read, Update, Delete para todos los modelos.
Proporciona una interfaz consistente para interactuar con la base de datos.
"""

import logging
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from .models import Stock, HistoricalData, Prediction, TrainedModel

logger = logging.getLogger(__name__)


class BaseCRUD:
    """Clase base para operaciones CRUD."""
    
    def __init__(self, model_class):
        self.model_class = model_class
    
    def _handle_db_error(self, operation: str, error: Exception):
        """Maneja errores de base de datos de forma consistente."""
        error_msg = f"Error en {operation}: {str(error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from error


class StockCRUD(BaseCRUD):
    """Operaciones CRUD para el modelo Stock."""
    
    def __init__(self):
        super().__init__(Stock)
    
    def create(self, session: Session, ticker: str, name: str = None, 
               sector: str = None, market: str = 'US', currency: str = 'USD') -> Stock:
        """
        Crea una nueva acción.
        
        Args:
            session: Sesión de base de datos
            ticker: Símbolo de la acción
            name: Nombre de la empresa
            sector: Sector de la empresa
            market: Mercado (US, MX, etc.)
            currency: Moneda
        
        Returns:
            Stock: Objeto de acción creado
        """
        try:
            stock = Stock(
                ticker=ticker,
                name=name,
                sector=sector,
                market=market,
                currency=currency
            )
            session.add(stock)
            session.flush()  # Para obtener el ID sin commit
            
            logger.info(f"Acción creada: {ticker}")
            return stock
            
        except IntegrityError as e:
            self._handle_db_error(f"crear acción {ticker}", e)
        except Exception as e:
            self._handle_db_error(f"crear acción {ticker}", e)
    
    def get_by_ticker(self, session: Session, ticker: str) -> Optional[Stock]:
        """
        Obtiene una acción por su ticker.
        
        Args:
            session: Sesión de base de datos
            ticker: Símbolo de la acción
        
        Returns:
            Stock: Objeto de acción o None si no existe
        """
        try:
            return session.query(Stock).filter(Stock.ticker == ticker.upper()).first()
        except Exception as e:
            self._handle_db_error(f"obtener acción {ticker}", e)
    
    def get_by_id(self, session: Session, stock_id: int) -> Optional[Stock]:
        """Obtiene una acción por su ID."""
        try:
            return session.query(Stock).filter(Stock.id == stock_id).first()
        except Exception as e:
            self._handle_db_error(f"obtener acción ID {stock_id}", e)
    
    def get_all(self, session: Session, active_only: bool = True) -> List[Stock]:
        """
        Obtiene todas las acciones.
        
        Args:
            session: Sesión de base de datos
            active_only: Si solo obtener acciones activas
        
        Returns:
            List[Stock]: Lista de acciones
        """
        try:
            query = session.query(Stock)
            if active_only:
                query = query.filter(Stock.active == True)
            return query.order_by(Stock.ticker).all()
        except Exception as e:
            self._handle_db_error("obtener todas las acciones", e)
    
    def update(self, session: Session, stock_id: int, **kwargs) -> Optional[Stock]:
        """
        Actualiza una acción.
        
        Args:
            session: Sesión de base de datos
            stock_id: ID de la acción
            **kwargs: Campos a actualizar
        
        Returns:
            Stock: Objeto actualizado o None si no existe
        """
        try:
            stock = self.get_by_id(session, stock_id)
            if not stock:
                return None
            
            # Actualizar campos permitidos
            allowed_fields = ['name', 'sector', 'market', 'currency', 'active']
            for field, value in kwargs.items():
                if field in allowed_fields and hasattr(stock, field):
                    setattr(stock, field, value)
            
            stock.updated_at = datetime.utcnow()
            session.flush()
            
            logger.info(f"Acción actualizada: {stock.ticker}")
            return stock
            
        except Exception as e:
            self._handle_db_error(f"actualizar acción ID {stock_id}", e)
    
    def delete(self, session: Session, stock_id: int) -> bool:
        """
        Elimina una acción (soft delete - marca como inactiva).
        
        Args:
            session: Sesión de base de datos
            stock_id: ID de la acción
        
        Returns:
            bool: True si se eliminó, False si no existía
        """
        try:
            stock = self.get_by_id(session, stock_id)
            if not stock:
                return False
            
            stock.active = False
            stock.updated_at = datetime.utcnow()
            session.flush()
            
            logger.info(f"Acción desactivada: {stock.ticker}")
            return True
            
        except Exception as e:
            self._handle_db_error(f"eliminar acción ID {stock_id}", e)


class HistoricalDataCRUD(BaseCRUD):
    """Operaciones CRUD para el modelo HistoricalData."""
    
    def __init__(self):
        super().__init__(HistoricalData)
    
    def create_batch(self, session: Session, stock_id: int, 
                    data_records: List[Dict[str, Any]]) -> int:
        """
        Crea múltiples registros de datos históricos.
        
        Args:
            session: Sesión de base de datos
            stock_id: ID de la acción
            data_records: Lista de diccionarios con datos OHLCV
        
        Returns:
            int: Número de registros creados
        """
        try:
            created_count = 0
            
            for record in data_records:
                # Verificar si ya existe el registro
                existing = session.query(HistoricalData).filter(
                    and_(
                        HistoricalData.stock_id == stock_id,
                        HistoricalData.date == record['date']
                    )
                ).first()
                
                if existing:
                    # Actualizar registro existente
                    for field in ['open', 'high', 'low', 'close', 'volume', 'adj_close']:
                        if field in record:
                            setattr(existing, field, record[field])
                else:
                    # Crear nuevo registro
                    historical_data = HistoricalData(
                        stock_id=stock_id,
                        **record
                    )
                    session.add(historical_data)
                    created_count += 1
            
            session.flush()
            logger.info(f"Creados/actualizados {created_count} registros históricos para stock_id {stock_id}")
            return created_count
            
        except Exception as e:
            self._handle_db_error(f"crear datos históricos para stock_id {stock_id}", e)
    
    def get_by_stock(self, session: Session, stock_id: int, 
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None,
                    limit: Optional[int] = None) -> List[HistoricalData]:
        """
        Obtiene datos históricos de una acción.
        
        Args:
            session: Sesión de base de datos
            stock_id: ID de la acción
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            limit: Límite de registros (opcional)
        
        Returns:
            List[HistoricalData]: Lista de datos históricos
        """
        try:
            query = session.query(HistoricalData).filter(HistoricalData.stock_id == stock_id)
            
            if start_date:
                query = query.filter(HistoricalData.date >= start_date)
            if end_date:
                query = query.filter(HistoricalData.date <= end_date)
            
            query = query.order_by(desc(HistoricalData.date))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
            
        except Exception as e:
            self._handle_db_error(f"obtener datos históricos para stock_id {stock_id}", e)
    
    def get_latest(self, session: Session, stock_id: int) -> Optional[HistoricalData]:
        """Obtiene el dato más reciente de una acción."""
        try:
            return session.query(HistoricalData).filter(
                HistoricalData.stock_id == stock_id
            ).order_by(desc(HistoricalData.date)).first()
        except Exception as e:
            self._handle_db_error(f"obtener último dato para stock_id {stock_id}", e)
    
    def get_date_range(self, session: Session, stock_id: int) -> Optional[Tuple[date, date]]:
        """
        Obtiene el rango de fechas disponibles para una acción.
        
        Returns:
            Tuple[date, date]: (fecha_min, fecha_max) o None si no hay datos
        """
        try:
            result = session.query(
                func.min(HistoricalData.date),
                func.max(HistoricalData.date)
            ).filter(HistoricalData.stock_id == stock_id).first()
            
            if result and result[0] and result[1]:
                return (result[0], result[1])
            return None
            
        except Exception as e:
            self._handle_db_error(f"obtener rango de fechas para stock_id {stock_id}", e)
    
    def delete_by_stock(self, session: Session, stock_id: int, 
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> int:
        """
        Elimina datos históricos de una acción.
        
        Args:
            session: Sesión de base de datos
            stock_id: ID de la acción
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
        
        Returns:
            int: Número de registros eliminados
        """
        try:
            query = session.query(HistoricalData).filter(HistoricalData.stock_id == stock_id)
            
            if start_date:
                query = query.filter(HistoricalData.date >= start_date)
            if end_date:
                query = query.filter(HistoricalData.date <= end_date)
            
            count = query.count()
            query.delete()
            session.flush()
            
            logger.info(f"Eliminados {count} registros históricos para stock_id {stock_id}")
            return count
            
        except Exception as e:
            self._handle_db_error(f"eliminar datos históricos para stock_id {stock_id}", e)


class PredictionCRUD(BaseCRUD):
    """Operaciones CRUD para el modelo Prediction."""
    
    def __init__(self):
        super().__init__(Prediction)
    
    def create(self, session: Session, stock_id: int, prediction_date: date,
               predicted_price: float, confidence: Optional[float] = None,
               model_version: Optional[str] = None, 
               prediction_type: str = 'close',
               horizon_days: int = 1) -> Prediction:
        """Crea una nueva predicción."""
        try:
            prediction = Prediction(
                stock_id=stock_id,
                prediction_date=prediction_date,
                predicted_price=predicted_price,
                confidence=confidence,
                model_version=model_version,
                prediction_type=prediction_type,
                horizon_days=horizon_days
            )
            session.add(prediction)
            session.flush()
            
            logger.info(f"Predicción creada para stock_id {stock_id}, fecha {prediction_date}")
            return prediction
            
        except Exception as e:
            self._handle_db_error(f"crear predicción para stock_id {stock_id}", e)
    
    def get_by_stock(self, session: Session, stock_id: int,
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None) -> List[Prediction]:
        """Obtiene predicciones de una acción."""
        try:
            query = session.query(Prediction).filter(Prediction.stock_id == stock_id)
            
            if start_date:
                query = query.filter(Prediction.prediction_date >= start_date)
            if end_date:
                query = query.filter(Prediction.prediction_date <= end_date)
            
            return query.order_by(desc(Prediction.prediction_date)).all()
            
        except Exception as e:
            self._handle_db_error(f"obtener predicciones para stock_id {stock_id}", e)
    
    def update_actual_price(self, session: Session, prediction_id: int, 
                           actual_price: float) -> Optional[Prediction]:
        """Actualiza el precio real de una predicción."""
        try:
            prediction = session.query(Prediction).filter(Prediction.id == prediction_id).first()
            if prediction:
                prediction.actual_price = actual_price
                session.flush()
                logger.info(f"Precio real actualizado para predicción {prediction_id}")
            return prediction
            
        except Exception as e:
            self._handle_db_error(f"actualizar precio real para predicción {prediction_id}", e)


class ModelCRUD(BaseCRUD):
    """Operaciones CRUD para el modelo TrainedModel."""
    
    def __init__(self):
        super().__init__(TrainedModel)
    
    def create(self, session: Session, stock_id: int, model_path: str,
               version: str, **metrics) -> TrainedModel:
        """Crea un nuevo modelo entrenado."""
        try:
            # Desactivar otros modelos para esta acción
            session.query(TrainedModel).filter(
                and_(
                    TrainedModel.stock_id == stock_id,
                    TrainedModel.is_active == True
                )
            ).update({'is_active': False})
            
            # Crear nuevo modelo
            model = TrainedModel(
                stock_id=stock_id,
                model_path=model_path,
                version=version,
                is_active=True,
                **metrics
            )
            session.add(model)
            session.flush()
            
            logger.info(f"Modelo creado para stock_id {stock_id}, versión {version}")
            return model
            
        except Exception as e:
            self._handle_db_error(f"crear modelo para stock_id {stock_id}", e)
    
    def get_active_model(self, session: Session, stock_id: int) -> Optional[TrainedModel]:
        """Obtiene el modelo activo para una acción."""
        try:
            return session.query(TrainedModel).filter(
                and_(
                    TrainedModel.stock_id == stock_id,
                    TrainedModel.is_active == True
                )
            ).first()
        except Exception as e:
            self._handle_db_error(f"obtener modelo activo para stock_id {stock_id}", e)
    
    def get_all_by_stock(self, session: Session, stock_id: int) -> List[TrainedModel]:
        """Obtiene todos los modelos de una acción."""
        try:
            return session.query(TrainedModel).filter(
                TrainedModel.stock_id == stock_id
            ).order_by(desc(TrainedModel.training_date)).all()
        except Exception as e:
            self._handle_db_error(f"obtener modelos para stock_id {stock_id}", e)
