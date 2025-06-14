"""
Data validation utilities for stock market data.
Provides functions to validate, clean, and verify data integrity.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Class for validating stock market data integrity and quality.
    """
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame, ticker: str = "Unknown") -> Dict[str, any]:
        """
        Comprehensive validation of OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker for logging
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                results['errors'].append(f"Missing required columns: {missing_cols}")
                results['is_valid'] = False
                return results
            
            # Check for empty DataFrame
            if df.empty:
                results['errors'].append("DataFrame is empty")
                results['is_valid'] = False
                return results
            
            # Check for null values
            null_counts = df[required_cols].isnull().sum()
            if null_counts.any():
                for col, count in null_counts.items():
                    if count > 0:
                        results['warnings'].append(f"Column '{col}' has {count} null values")
            
            # Check for negative values
            negative_checks = {
                'open': (df['open'] < 0).sum(),
                'high': (df['high'] < 0).sum(),
                'low': (df['low'] < 0).sum(),
                'close': (df['close'] < 0).sum(),
                'volume': (df['volume'] < 0).sum()
            }
            
            for col, count in negative_checks.items():
                if count > 0:
                    results['errors'].append(f"Column '{col}' has {count} negative values")
                    results['is_valid'] = False
            
            # Validate OHLC relationships
            ohlc_errors = DataValidator._validate_ohlc_relationships(df)
            if ohlc_errors:
                results['warnings'].extend(ohlc_errors)
            
            # Check for data gaps
            if 'date' in df.columns:
                gap_info = DataValidator._check_data_gaps(df)
                if gap_info['gaps_found']:
                    results['warnings'].append(f"Found {gap_info['gap_count']} date gaps")
            
            # Calculate statistics
            results['stats'] = DataValidator._calculate_data_stats(df)
            
            logger.info(f"Validation completed for {ticker}: {'Valid' if results['is_valid'] else 'Invalid'}")
            
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
            logger.error(f"Error during validation of {ticker}: {str(e)}")
        
        return results
    
    @staticmethod
    def _validate_ohlc_relationships(df: pd.DataFrame) -> List[str]:
        """
        Validate OHLC price relationships.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of error messages
        """
        errors = []
        
        try:
            # High should be >= Open and Close
            high_vs_open = (df['high'] < df['open']).sum()
            high_vs_close = (df['high'] < df['close']).sum()
            
            if high_vs_open > 0:
                errors.append(f"{high_vs_open} rows where High < Open")
            
            if high_vs_close > 0:
                errors.append(f"{high_vs_close} rows where High < Close")
            
            # Low should be <= Open and Close
            low_vs_open = (df['low'] > df['open']).sum()
            low_vs_close = (df['low'] > df['close']).sum()
            
            if low_vs_open > 0:
                errors.append(f"{low_vs_open} rows where Low > Open")
            
            if low_vs_close > 0:
                errors.append(f"{low_vs_close} rows where Low > Close")
            
            # High should be >= Low
            high_vs_low = (df['high'] < df['low']).sum()
            if high_vs_low > 0:
                errors.append(f"{high_vs_low} rows where High < Low")
            
        except Exception as e:
            errors.append(f"Error validating OHLC relationships: {str(e)}")
        
        return errors
    
    @staticmethod
    def _check_data_gaps(df: pd.DataFrame) -> Dict[str, any]:
        """
        Check for gaps in date sequence.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            Dictionary with gap information
        """
        gap_info = {
            'gaps_found': False,
            'gap_count': 0,
            'largest_gap_days': 0,
            'gap_details': []
        }
        
        try:
            if 'date' not in df.columns:
                return gap_info
            
            # Sort by date
            df_sorted = df.sort_values('date')
            dates = pd.to_datetime(df_sorted['date'])
            
            # Calculate differences between consecutive dates
            date_diffs = dates.diff()
            
            # Find gaps larger than 1 day (excluding weekends)
            # For daily data, normal gap is 1-3 days (weekends)
            large_gaps = date_diffs > timedelta(days=7)  # More than a week
            
            if large_gaps.any():
                gap_info['gaps_found'] = True
                gap_info['gap_count'] = large_gaps.sum()
                gap_info['largest_gap_days'] = date_diffs.max().days
                
                # Get details of large gaps
                gap_indices = large_gaps[large_gaps].index
                for idx in gap_indices:
                    prev_date = dates.iloc[idx-1]
                    curr_date = dates.iloc[idx]
                    gap_days = (curr_date - prev_date).days
                    
                    gap_info['gap_details'].append({
                        'from_date': prev_date.strftime('%Y-%m-%d'),
                        'to_date': curr_date.strftime('%Y-%m-%d'),
                        'days': gap_days
                    })
            
        except Exception as e:
            logger.error(f"Error checking data gaps: {str(e)}")
        
        return gap_info
    
    @staticmethod
    def _calculate_data_stats(df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate basic statistics for the data.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        try:
            # Basic info
            stats['row_count'] = len(df)
            stats['column_count'] = len(df.columns)
            
            # Date range
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
                stats['date_range'] = {
                    'start': dates.min().strftime('%Y-%m-%d'),
                    'end': dates.max().strftime('%Y-%m-%d'),
                    'days': (dates.max() - dates.min()).days
                }
            
            # Price statistics
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                stats['price_stats'] = {
                    'min_price': float(df[['open', 'high', 'low', 'close']].min().min()),
                    'max_price': float(df[['open', 'high', 'low', 'close']].max().max()),
                    'avg_close': float(df['close'].mean()),
                    'latest_close': float(df['close'].iloc[-1]) if len(df) > 0 else None
                }
            
            # Volume statistics
            if 'volume' in df.columns:
                stats['volume_stats'] = {
                    'min_volume': int(df['volume'].min()),
                    'max_volume': int(df['volume'].max()),
                    'avg_volume': int(df['volume'].mean()),
                    'zero_volume_days': int((df['volume'] == 0).sum())
                }
            
            # Data quality metrics
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            stats['data_quality'] = {
                'completeness_pct': float(((total_cells - null_cells) / total_cells) * 100),
                'null_count': int(null_cells)
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            stats['error'] = str(e)
        
        return stats
    
    @staticmethod
    def clean_data(df: pd.DataFrame, ticker: str = "Unknown") -> pd.DataFrame:
        """
        Clean and fix common data issues.
        
        Args:
            df: DataFrame to clean
            ticker: Stock ticker for logging
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()
        
        try:
            # Remove rows where all price columns are null
            price_cols = ['open', 'high', 'low', 'close']
            available_price_cols = [col for col in price_cols if col in cleaned_df.columns]
            
            if available_price_cols:
                initial_count = len(cleaned_df)
                cleaned_df = cleaned_df.dropna(subset=available_price_cols, how='all')
                removed_count = initial_count - len(cleaned_df)
                
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} rows with all null prices for {ticker}")
            
            # Fill missing adj_close with close if available
            if 'close' in cleaned_df.columns and 'adj_close' in cleaned_df.columns:
                null_adj_close = cleaned_df['adj_close'].isnull().sum()
                if null_adj_close > 0:
                    cleaned_df['adj_close'] = cleaned_df['adj_close'].fillna(cleaned_df['close'])
                    logger.info(f"Filled {null_adj_close} missing adj_close values for {ticker}")
            
            # Sort by date if date column exists
            if 'date' in cleaned_df.columns:
                cleaned_df = cleaned_df.sort_values('date').reset_index(drop=True)
            
            # Remove duplicate rows
            initial_count = len(cleaned_df)
            if 'date' in cleaned_df.columns:
                cleaned_df = cleaned_df.drop_duplicates(subset=['date'])
            else:
                cleaned_df = cleaned_df.drop_duplicates()
            
            duplicate_count = initial_count - len(cleaned_df)
            if duplicate_count > 0:
                logger.info(f"Removed {duplicate_count} duplicate rows for {ticker}")
            
        except Exception as e:
            logger.error(f"Error cleaning data for {ticker}: {str(e)}")
        
        return cleaned_df


def validate_ticker_format(ticker: str) -> bool:
    """
    Validate ticker symbol format.
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if format is valid
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()
    
    # Basic validation: 1-10 characters
    if not (1 <= len(ticker) <= 10):
        return False
    
    # Must start with a letter
    if not ticker[0].isalpha():
        return False
    
    # Allow letters, numbers, and some special characters (., -, ^)
    allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^')
    if not set(ticker).issubset(allowed_chars):
        return False
    
    # Should not be all numbers
    if ticker.isdigit():
        return False
    
    return True


def format_ticker(ticker: str) -> str:
    """
    Format ticker symbol to standard format.
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Formatted ticker symbol
    """
    if not ticker:
        return ""
    
    return ticker.strip().upper()


if __name__ == "__main__":
    # Test the validator with sample data
    from collector import YahooFinanceCollector
    
    collector = YahooFinanceCollector()
    data = collector.get_historical_data("AAPL", period="1mo")
    
    if data is not None:
        validator = DataValidator()
        results = validator.validate_ohlcv_data(data, "AAPL")
        
        print("Validation Results:")
        print(f"Valid: {results['is_valid']}")
        
        if results['errors']:
            print(f"Errors: {results['errors']}")
        
        if results['warnings']:
            print(f"Warnings: {results['warnings']}")
        
        print(f"Statistics: {results['stats']}")
    else:
        print("Could not retrieve test data")
