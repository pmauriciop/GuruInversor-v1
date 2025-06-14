import React, { useState, useEffect } from 'react';
import ApiService from '../services/api';
import type { StockInfo, AddStockRequest } from '../types/api';
import './TickerSearchPage.css';

interface TickerSearchState {
  searchTicker: string;
  searchResults: StockInfo | null;
  monitoredStocks: StockInfo[];
  loading: boolean;
  searchLoading: boolean;
  error: string | null;
  successMessage: string | null;
}

const TickerSearchPage: React.FC = () => {
  const [state, setState] = useState<TickerSearchState>({
    searchTicker: '',
    searchResults: null,
    monitoredStocks: [],
    loading: true,
    searchLoading: false,
    error: null,
    successMessage: null
  });

  // Cargar stocks monitoreados al montar el componente
  useEffect(() => {
    loadMonitoredStocks();
  }, []);

  const loadMonitoredStocks = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      const response = await ApiService.getStocksList();
      
      if (response.data) {
        setState(prev => ({
          ...prev,
          monitoredStocks: response.data!.stocks,
          loading: false
        }));
      } else {
        setState(prev => ({
          ...prev,
          error: response.error || 'Error cargando stocks monitoreados',
          loading: false
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: 'Error de conexión',
        loading: false
      }));
    }
  };

  const searchTicker = async () => {
    if (!state.searchTicker.trim()) {
      setState(prev => ({ ...prev, error: 'Ingrese un ticker para buscar' }));
      return;
    }

    try {
      setState(prev => ({ 
        ...prev, 
        searchLoading: true, 
        error: null, 
        searchResults: null,
        successMessage: null
      }));

      const ticker = state.searchTicker.toUpperCase().trim();
      const response = await ApiService.getStockInfo(ticker);

      if (response.data) {
        setState(prev => ({
          ...prev,
          searchResults: response.data!,
          searchLoading: false
        }));
      } else {
        setState(prev => ({
          ...prev,
          error: response.error || 'Ticker no encontrado',
          searchLoading: false
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: 'Error buscando ticker',
        searchLoading: false
      }));
    }
  };

  const addToMonitored = async (stockInfo: StockInfo) => {
    try {
      setState(prev => ({ ...prev, error: null, successMessage: null }));

      const request: AddStockRequest = {
        ticker: stockInfo.ticker,
        name: stockInfo.name,
        auto_train: true
      };

      const response = await ApiService.addStock(request);

      if (response.data) {
        setState(prev => ({
          ...prev,
          successMessage: `${stockInfo.ticker} añadido exitosamente`,
          searchResults: null,
          searchTicker: ''
        }));

        // Recargar la lista de stocks monitoreados
        await loadMonitoredStocks();
      } else {
        setState(prev => ({
          ...prev,
          error: response.error || 'Error añadiendo ticker'
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: 'Error añadiendo ticker'
      }));
    }
  };

  const removeFromMonitored = async (ticker: string) => {
    if (!confirm(`¿Está seguro de eliminar ${ticker} del monitoreo?`)) {
      return;
    }

    try {
      setState(prev => ({ ...prev, error: null, successMessage: null }));

      const response = await ApiService.removeStock(ticker);

      if (response.data) {
        setState(prev => ({
          ...prev,
          successMessage: `${ticker} eliminado exitosamente`
        }));

        // Recargar la lista
        await loadMonitoredStocks();
      } else {
        setState(prev => ({
          ...prev,
          error: response.error || 'Error eliminando ticker'
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: 'Error eliminando ticker'
      }));
    }
  };

  const updateStockData = async (ticker: string) => {
    try {
      setState(prev => ({ ...prev, error: null, successMessage: null }));

      const response = await ApiService.updateStockData(ticker);

      if (response.data) {
        setState(prev => ({
          ...prev,
          successMessage: `Datos de ${ticker} actualizados`
        }));
      } else {
        setState(prev => ({
          ...prev,
          error: response.error || 'Error actualizando datos'
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: 'Error actualizando datos'
      }));
    }
  };

  const clearMessages = () => {
    setState(prev => ({ ...prev, error: null, successMessage: null }));
  };

  const formatPrice = (price: number | undefined) => {
    return price ? `$${price.toFixed(2)}` : 'N/A';
  };

  const formatMarketCap = (marketCap: number | undefined) => {
    if (!marketCap) return 'N/A';
    
    if (marketCap >= 1e12) {
      return `$${(marketCap / 1e12).toFixed(1)}T`;
    } else if (marketCap >= 1e9) {
      return `$${(marketCap / 1e9).toFixed(1)}B`;
    } else if (marketCap >= 1e6) {
      return `$${(marketCap / 1e6).toFixed(1)}M`;
    }
    return `$${marketCap.toLocaleString()}`;
  };

  return (
    <div className="ticker-search-page">
      <div className="container">
        <div className="page-header">
          <h1>Gestión de Tickers</h1>
          <p>Busca y añade tickers para monitoreo y análisis</p>
        </div>

        {/* Mensajes de estado */}
        {(state.error || state.successMessage) && (
          <div className="message-container">
            {state.error && (
              <div className="alert alert-error">
                <span>{state.error}</span>
                <button onClick={clearMessages} className="close-btn">&times;</button>
              </div>
            )}
            {state.successMessage && (
              <div className="alert alert-success">
                <span>{state.successMessage}</span>
                <button onClick={clearMessages} className="close-btn">&times;</button>
              </div>
            )}
          </div>
        )}

        {/* Sección de búsqueda */}
        <div className="search-section card">
          <h2>Buscar Ticker</h2>
          <div className="search-form">
            <div className="form-group">
              <label className="form-label">Símbolo del Ticker</label>
              <div className="search-input-group">
                <input
                  type="text"
                  className="form-input"
                  placeholder="Ej: AAPL, GOOGL, MSFT..."
                  value={state.searchTicker}
                  onChange={(e) => setState(prev => ({ 
                    ...prev, 
                    searchTicker: e.target.value.toUpperCase() 
                  }))}
                  onKeyPress={(e) => e.key === 'Enter' && searchTicker()}
                  disabled={state.searchLoading}
                />
                <button 
                  className="btn btn-primary search-btn"
                  onClick={searchTicker}
                  disabled={state.searchLoading || !state.searchTicker.trim()}
                >
                  {state.searchLoading ? (
                    <span className="loading-spinner"></span>
                  ) : (
                    'Buscar'
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Resultados de búsqueda */}
          {state.searchResults && (
            <div className="search-results">
              <h3>Resultado de Búsqueda</h3>
              <div className="stock-card">
                <div className="stock-header">
                  <div className="stock-main-info">
                    <h4>{state.searchResults.ticker}</h4>
                    <p className="stock-name">{state.searchResults.name}</p>
                  </div>
                  <div className="stock-price">
                    <span className="price">{formatPrice(state.searchResults.current_price)}</span>
                  </div>
                </div>
                
                <div className="stock-details">
                  <div className="detail-item">
                    <span className="label">Sector:</span>
                    <span className="value">{state.searchResults.sector || 'N/A'}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Cap. de Mercado:</span>
                    <span className="value">{formatMarketCap(state.searchResults.market_cap)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Última Actualización:</span>
                    <span className="value">
                      {new Date(state.searchResults.last_update).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                <div className="stock-actions">
                  <button
                    className="btn btn-success"
                    onClick={() => addToMonitored(state.searchResults!)}
                  >
                    ➕ Añadir al Monitoreo
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Stocks monitoreados */}
        <div className="monitored-section">
          <div className="section-header">
            <h2>Tickers Monitoreados</h2>
            <button className="btn btn-secondary" onClick={loadMonitoredStocks}>
              🔄 Actualizar Lista
            </button>
          </div>

          {state.loading ? (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <span>Cargando stocks monitoreados...</span>
            </div>
          ) : state.monitoredStocks.length === 0 ? (
            <div className="empty-state card">
              <h3>No hay tickers monitoreados</h3>
              <p>Usa la búsqueda arriba para añadir tickers al sistema de monitoreo.</p>
            </div>
          ) : (
            <div className="stocks-grid">
              {state.monitoredStocks.map((stock) => (
                <div key={stock.ticker} className="stock-card monitored">
                  <div className="stock-header">
                    <div className="stock-main-info">
                      <h4>{stock.ticker}</h4>
                      <p className="stock-name">{stock.name}</p>
                    </div>
                    <div className="stock-price">
                      <span className="price">{formatPrice(stock.current_price)}</span>
                    </div>
                  </div>

                  <div className="stock-details">
                    <div className="detail-item">
                      <span className="label">Sector:</span>
                      <span className="value">{stock.sector || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="label">Cap. de Mercado:</span>
                      <span className="value">{formatMarketCap(stock.market_cap)}</span>
                    </div>
                  </div>

                  <div className="stock-actions">
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={() => updateStockData(stock.ticker)}
                      title="Actualizar datos"
                    >
                      🔄
                    </button>
                    <button
                      className="btn btn-sm btn-primary"
                      onClick={() => {
                        // TODO: Navegar a página de predicciones
                        console.log('Ver predicciones para', stock.ticker);
                      }}
                      title="Ver predicciones"
                    >
                      📈
                    </button>
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => removeFromMonitored(stock.ticker)}
                      title="Eliminar del monitoreo"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Información adicional */}
        <div className="info-section card">
          <h3>ℹ️ Información</h3>
          <ul>
            <li><strong>Búsqueda:</strong> Ingresa el símbolo del ticker (ej: AAPL para Apple Inc.)</li>
            <li><strong>Auto-entrenamiento:</strong> Al añadir un ticker, se iniciará automáticamente el entrenamiento del modelo ML</li>
            <li><strong>Monitoreo:</strong> Los tickers monitoreados se actualizan automáticamente con nuevos datos de mercado</li>
            <li><strong>Predicciones:</strong> Solo están disponibles para tickers con modelos entrenados</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default TickerSearchPage;
