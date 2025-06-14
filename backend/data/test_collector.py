"""
Test script for the Yahoo Finance data collector.
Tests various functions and validates the collected data.
"""
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import YahooFinanceCollector, get_stock_data, get_stock_info
from data.validator import DataValidator, validate_ticker_format, format_ticker
import pandas as pd
from datetime import datetime


def test_ticker_validation():
    """Test ticker format validation."""
    print("🧪 Testing ticker validation...")
    
    valid_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "BRK.B", "SPY"]
    invalid_tickers = ["", "TOOLONGTICKERRRR", "12345", "@#$%"]
    
    for ticker in valid_tickers:
        assert validate_ticker_format(ticker), f"Valid ticker {ticker} failed validation"
        print(f"✅ {ticker} - Valid")
    
    for ticker in invalid_tickers:
        assert not validate_ticker_format(ticker), f"Invalid ticker {ticker} passed validation"
        print(f"❌ {ticker} - Invalid (as expected)")
    
    print("✅ Ticker validation tests passed!\n")


def test_stock_info():
    """Test stock information retrieval."""
    print("🧪 Testing stock info retrieval...")
    
    collector = YahooFinanceCollector()
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        print(f"Getting info for {ticker}...")
        info = collector.get_stock_info(ticker)
        
        if info:
            print(f"✅ {ticker}: {info['name']} - {info['sector']}")
            assert info['ticker'] == ticker
            assert 'name' in info
            assert 'sector' in info
        else:
            print(f"❌ Failed to get info for {ticker}")
    
    print("✅ Stock info tests completed!\n")


def test_historical_data():
    """Test historical data collection."""
    print("🧪 Testing historical data collection...")
    
    collector = YahooFinanceCollector()
    test_ticker = "AAPL"
    
    # Test different periods
    periods = ["1mo", "3mo", "6mo"]
    
    for period in periods:
        print(f"Testing period: {period}")
        data = collector.get_historical_data(test_ticker, period=period)
        
        if data is not None:
            print(f"✅ {period}: {len(data)} rows, date range: {data['date'].min()} to {data['date'].max()}")
            
            # Validate required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                assert col in data.columns, f"Missing column: {col}"
            
            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(data['date']), "Date column should be datetime"
            assert pd.api.types.is_numeric_dtype(data['close']), "Close column should be numeric"
            
        else:
            print(f"❌ Failed to get data for period {period}")
    
    print("✅ Historical data tests completed!\n")


def test_data_validation():
    """Test data validation functionality."""
    print("🧪 Testing data validation...")
    
    # Get some real data to validate
    data = get_stock_data("AAPL", period="1mo")
    
    if data is not None:
        validator = DataValidator()
        results = validator.validate_ohlcv_data(data, "AAPL")
        
        print(f"Validation result: {'✅ Valid' if results['is_valid'] else '❌ Invalid'}")
        
        if results['errors']:
            print(f"Errors found: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print(f"Warnings found: {len(results['warnings'])}")
            for warning in results['warnings'][:3]:  # Show first 3 warnings
                print(f"  - {warning}")
        
        print(f"Data stats:")
        stats = results['stats']
        if 'row_count' in stats:
            print(f"  - Rows: {stats['row_count']}")
        if 'date_range' in stats:
            print(f"  - Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        if 'price_stats' in stats:
            print(f"  - Price range: ${stats['price_stats']['min_price']:.2f} - ${stats['price_stats']['max_price']:.2f}")
            print(f"  - Latest close: ${stats['price_stats']['latest_close']:.2f}")
        
        print("✅ Data validation tests completed!\n")
    else:
        print("❌ Could not get data for validation test\n")


def test_latest_price():
    """Test latest price retrieval."""
    print("🧪 Testing latest price retrieval...")
    
    collector = YahooFinanceCollector()
    test_tickers = ["AAPL", "MSFT"]
    
    for ticker in test_tickers:
        price = collector.get_latest_price(ticker)
        if price:
            print(f"✅ {ticker}: ${price:.2f}")
        else:
            print(f"❌ Failed to get latest price for {ticker}")
    
    print("✅ Latest price tests completed!\n")


def test_multiple_tickers():
    """Test downloading data for multiple tickers."""
    print("🧪 Testing multiple tickers download...")
    
    collector = YahooFinanceCollector()
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    results = collector.get_multiple_tickers_data(tickers, period="1mo")
    
    print(f"Successfully downloaded data for {len(results)}/{len(tickers)} tickers:")
    for ticker, data in results.items():
        print(f"  ✅ {ticker}: {len(data)} rows")
    
    failed_tickers = set(tickers) - set(results.keys())
    if failed_tickers:
        print(f"Failed tickers: {failed_tickers}")
    
    print("✅ Multiple tickers test completed!\n")


def test_convenience_functions():
    """Test convenience functions."""
    print("🧪 Testing convenience functions...")
    
    # Test get_stock_data function
    data = get_stock_data("AAPL", period="1mo")
    if data is not None:
        print(f"✅ get_stock_data: {len(data)} rows")
    else:
        print("❌ get_stock_data failed")
    
    # Test get_stock_info function
    info = get_stock_info("AAPL")
    if info:
        print(f"✅ get_stock_info: {info['name']}")
    else:
        print("❌ get_stock_info failed")
    
    print("✅ Convenience functions test completed!\n")


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Yahoo Finance Collector Tests")
    print("=" * 50)
    
    try:
        test_ticker_validation()
        test_stock_info()
        test_historical_data()
        test_data_validation()
        test_latest_price()
        test_multiple_tickers()
        test_convenience_functions()
        
        print("🎉 All tests completed successfully!")
        print("✅ Yahoo Finance collector is working properly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print("🔧 Please check the error and fix any issues")
        raise


if __name__ == "__main__":
    run_all_tests()
