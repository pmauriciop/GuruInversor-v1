"""
Recolector de datos de Yahoo Finance para GuruInversor.

Este módulo se encarga de descargar datos históricos de acciones
desde Yahoo Finance usando la biblioteca yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import time
import requests

# Configurar logging
logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """
    Recolector de datos históricos de Yahoo Finance.
    
    Maneja la descarga de datos OHLCV (Open, High, Low, Close, Volume)
    para cualquier ticker disponible en Yahoo Finance.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Inicializar el recolector.
        
        Args:
            cache_dir: Directorio para cache local (opcional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms entre requests
        
    def _wait_for_rate_limit(self):
        """Implementar rate limiting básico."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Obtener información básica de una acción.
        
        Args:
            ticker: Símbolo de la acción (ej: 'AAPL', 'MSFT')
            
        Returns:
            Diccionario con información de la acción
        """
        try:
            self._wait_for_rate_limit()
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extraer información relevante
            stock_info = {
                'ticker': ticker.upper(),
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'current_price': info.get('currentPrice'),
                'retrieved_at': datetime.now().isoformat()
            }
            
            logger.info(f"Información obtenida para {ticker}: {stock_info['name']}")
            return stock_info
            
        except Exception as e:
            logger.error(f"Error obteniendo información de {ticker}: {str(e)}")
            raise
    
    def get_historical_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Descargar datos históricos de una acción.
        
        Args:
            ticker: Símbolo de la acción
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            period: Período de datos ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Intervalo de datos ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame con datos históricos (Date, Open, High, Low, Close, Volume, Adj Close)
        """
        try:
            self._wait_for_rate_limit()
            
            logger.info(f"Descargando datos históricos para {ticker}")
            
            stock = yf.Ticker(ticker)
            
            # Descargar datos
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                raise ValueError(f"No se encontraron datos para {ticker}")
            
            # Limpiar y preparar datos
            hist = hist.reset_index()
            
            # Asegurar que las columnas estén en inglés y en el formato correcto
            hist.columns = [col.replace(' ', '_').lower() for col in hist.columns]
            
            # Renombrar columnas si es necesario
            column_mapping = {
                'adj_close': 'adj_close',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'open': 'open',
                'volume': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in hist.columns:
                    hist[new_col] = hist[old_col]
            
            # Asegurar que tenemos las columnas necesarias
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in hist.columns:
                    raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
            
            # Agregar metadatos
            hist['ticker'] = ticker.upper()
            hist['retrieved_at'] = datetime.now()
            
            # Ordenar por fecha
            hist = hist.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Descargados {len(hist)} registros para {ticker}")
            return hist
            
        except Exception as e:
            logger.error(f"Error descargando datos históricos de {ticker}: {str(e)}")
            raise
    
    def get_latest_price(self, ticker: str) -> Dict[str, Any]:
        """
        Obtener el precio más reciente de una acción.
        
        Args:
            ticker: Símbolo de la acción
            
        Returns:
            Diccionario con información de precio actual
        """
        try:
            self._wait_for_rate_limit()
            
            stock = yf.Ticker(ticker)
            
            # Obtener datos del día actual
            hist = stock.history(period="1d", interval="1m")
            
            if hist.empty:
                # Si no hay datos intraday, obtener el último día disponible
                hist = stock.history(period="5d", interval="1d")
                if hist.empty:
                    raise ValueError(f"No se encontraron datos de precio para {ticker}")
            
            latest = hist.iloc[-1]
            
            price_data = {
                'ticker': ticker.upper(),
                'price': float(latest['Close']),
                'volume': int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                'timestamp': latest.name.isoformat(),
                'retrieved_at': datetime.now().isoformat()
            }
            
            logger.info(f"Precio actual de {ticker}: ${price_data['price']:.2f}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error obteniendo precio actual de {ticker}: {str(e)}")
            raise
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validar si un ticker existe en Yahoo Finance.
        
        Args:
            ticker: Símbolo de la acción
            
        Returns:
            True si el ticker existe, False si no
        """
        try:
            self._wait_for_rate_limit()
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Verificar si tenemos información básica
            if 'regularMarketPrice' in info or 'currentPrice' in info or info.get('longName'):
                logger.info(f"Ticker {ticker} validado exitosamente")
                return True
            else:
                logger.warning(f"Ticker {ticker} no encontrado")
                return False
                
        except Exception as e:
            logger.error(f"Error validando ticker {ticker}: {str(e)}")
            return False
    
    def get_multiple_tickers(
        self, 
        tickers: List[str], 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Descargar datos históricos para múltiples tickers.
        
        Args:
            tickers: Lista de símbolos de acciones
            start_date: Fecha de inicio
            end_date: Fecha de fin
            period: Período de datos
            
        Returns:
            Diccionario con DataFrames por ticker
        """
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Procesando ticker {ticker}")
                data = self.get_historical_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
                results[ticker] = data
                
            except Exception as e:
                logger.error(f"Error procesando {ticker}: {str(e)}")
                results[ticker] = pd.DataFrame()  # DataFrame vacío en caso de error
        
        logger.info(f"Procesados {len(results)} tickers, {sum(1 for df in results.values() if not df.empty)} exitosos")
        return results


def test_collector():
    """Función de prueba básica del recolector."""
    collector = YahooFinanceCollector()
    
    # Probar con AAPL
    ticker = "AAPL"
    
    print(f"Probando recolector con {ticker}...")
    
    # Validar ticker
    if collector.validate_ticker(ticker):
        print(f"✓ Ticker {ticker} es válido")
    else:
        print(f"✗ Ticker {ticker} no es válido")
        return
    
    # Obtener información
    try:
        info = collector.get_stock_info(ticker)
        print(f"✓ Información: {info['name']} ({info['sector']})")
    except Exception as e:
        print(f"✗ Error obteniendo información: {e}")
    
    # Obtener datos históricos
    try:
        data = collector.get_historical_data(ticker, period="1mo")
        print(f"✓ Datos históricos: {len(data)} registros")
        print(f"  Rango de fechas: {data['date'].min()} a {data['date'].max()}")
    except Exception as e:
        print(f"✗ Error obteniendo datos históricos: {e}")
    
    # Obtener precio actual
    try:
        price = collector.get_latest_price(ticker)
        print(f"✓ Precio actual: ${price['price']:.2f}")
    except Exception as e:
        print(f"✗ Error obteniendo precio actual: {e}")


if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(level=logging.INFO)
    test_collector()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """
    Collector class for downloading stock data from Yahoo Finance.
    
    Features:
    - Download historical data for any stock ticker
    - Data validation and cleaning
    - Rate limiting to avoid API blocks
    - Error handling and retry logic
    """
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize the collector.
        
        Args:
            rate_limit_delay: Delay between API calls in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a stock.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            
        Returns:
            Dictionary with stock information or None if failed
        """
        try:
            self._apply_rate_limit()
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant information
            stock_info = {
                'ticker': ticker.upper(),
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'retrieved_at': datetime.now().isoformat()
            }
            
            logger.info(f"Retrieved info for {ticker}: {stock_info['name']}")
            return stock_info
            
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {str(e)}")
            return None
    
    def get_historical_data(
        self,
        ticker: str,
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Download historical stock data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            self._apply_rate_limit()
            
            stock = yf.Ticker(ticker)
            
            # Download data using either period or date range
            if start_date and end_date:
                data = stock.history(start=start_date, end=end_date, interval=interval)
                logger.info(f"Downloaded {ticker} data from {start_date} to {end_date}")
            else:
                data = stock.history(period=period, interval=interval)
                logger.info(f"Downloaded {ticker} data for period {period}")
            
            if data.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return None
            
            # Clean and validate data
            data = self._clean_data(data, ticker)
            
            if data is not None:
                logger.info(f"Successfully processed {len(data)} rows for {ticker}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None
    
    def _clean_data(self, data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Clean and validate the downloaded data.
        
        Args:
            data: Raw data from Yahoo Finance
            ticker: Stock ticker for logging
            
        Returns:
            Cleaned DataFrame or None if data is invalid
        """
        try:
            # Check if we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Missing columns for {ticker}: {missing_columns}")
                return None
            
            # Remove rows with NaN values in critical columns
            initial_rows = len(data)
            data = data.dropna(subset=required_columns)
            
            if len(data) < initial_rows:
                logger.info(f"Removed {initial_rows - len(data)} rows with NaN values for {ticker}")
            
            # Validate data integrity
            if not self._validate_ohlc_data(data, ticker):
                return None
            
            # Add ticker column
            data['Ticker'] = ticker.upper()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to standard format
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close',
                'Ticker': 'ticker'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Ensure date column is datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Sort by date
            data = data.sort_values('date')
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data for {ticker}: {str(e)}")
            return None
    
    def _validate_ohlc_data(self, data: pd.DataFrame, ticker: str) -> bool:
        """
        Validate OHLC data integrity.
        
        Args:
            data: DataFrame with OHLC data
            ticker: Stock ticker for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check for negative values
            if (data[['Open', 'High', 'Low', 'Close']] < 0).any().any():
                logger.error(f"Found negative prices in {ticker} data")
                return False
            
            # Check OHLC relationships (High >= Open,Close and Low <= Open,Close)
            high_check = (data['High'] >= data['Open']) & (data['High'] >= data['Close'])
            low_check = (data['Low'] <= data['Open']) & (data['Low'] <= data['Close'])
            
            if not high_check.all():
                invalid_count = (~high_check).sum()
                logger.warning(f"Found {invalid_count} rows with High < Open/Close in {ticker}")
            
            if not low_check.all():
                invalid_count = (~low_check).sum()
                logger.warning(f"Found {invalid_count} rows with Low > Open/Close in {ticker}")
            
            # Check for zero volume (might be valid for some assets)
            zero_volume = (data['Volume'] == 0).sum()
            if zero_volume > 0:
                logger.info(f"Found {zero_volume} rows with zero volume in {ticker}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {ticker}: {str(e)}")
            return False
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the latest price for a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Latest price or None if failed
        """
        try:
            self._apply_rate_limit()
            
            stock = yf.Ticker(ticker)
            
            # Get current data
            data = stock.history(period="1d", interval="1m")
            
            if data.empty:
                logger.warning(f"No current data for {ticker}")
                return None
            
            latest_price = data['Close'].iloc[-1]
            logger.info(f"Latest price for {ticker}: ${latest_price:.2f}")
            
            return float(latest_price)
            
        except Exception as e:
            logger.error(f"Error getting latest price for {ticker}: {str(e)}")
            return None
    
    def get_multiple_tickers_data(
        self,
        tickers: list,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Time period for each ticker
            interval: Data interval
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Processing ticker {ticker}")
            data = self.get_historical_data(ticker, period=period, interval=interval)
            
            if data is not None:
                results[ticker] = data
            else:
                logger.warning(f"Failed to get data for {ticker}")
            
            # Small delay between tickers
            time.sleep(0.5)
        
        logger.info(f"Successfully retrieved data for {len(results)}/{len(tickers)} tickers")
        return results


# Convenience functions
def get_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Simple function to get stock data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period
        
    Returns:
        DataFrame with stock data or None
    """
    collector = YahooFinanceCollector()
    return collector.get_historical_data(ticker, period=period)


def get_stock_info(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Simple function to get stock information.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock info or None
    """
    collector = YahooFinanceCollector()
    return collector.get_stock_info(ticker)


if __name__ == "__main__":
    # Test the collector
    collector = YahooFinanceCollector()
    
    # Test with a popular stock
    test_ticker = "AAPL"
    
    print(f"Testing with {test_ticker}")
    
    # Get stock info
    info = collector.get_stock_info(test_ticker)
    if info:
        print(f"Stock Info: {info['name']} ({info['ticker']})")
        print(f"Sector: {info['sector']}")
    
    # Get historical data
    data = collector.get_historical_data(test_ticker, period="1mo")
    if data is not None:
        print(f"\nHistorical Data Shape: {data.shape}")
        print(f"Date Range: {data['date'].min()} to {data['date'].max()}")
        print(f"Latest Close: ${data['close'].iloc[-1]:.2f}")
        print("\nFirst 5 rows:")
        print(data.head())
    
    # Get latest price
    latest_price = collector.get_latest_price(test_ticker)
    if latest_price:
        print(f"\nLatest Price: ${latest_price:.2f}")
