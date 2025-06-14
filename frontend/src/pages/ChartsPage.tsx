import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import ApiService from '../services/api';
import type { StockDataResponse, PredictionResponse, StockInfo } from '../types/api';
import './ChartsPage.css';

// Registrar componentes de Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface ChartsState {
  selectedTicker: string;
  stockData: StockDataResponse | null;
  predictions: PredictionResponse | null;
  availableStocks: StockInfo[];
  timeRange: string;
  loading: boolean;
  error: string | null;
  showPredictions: boolean;
  chartType: 'price' | 'candlestick';
}

const ChartsPage: React.FC = () => {
  const [state, setState] = useState<ChartsState>({
    selectedTicker: '',
    stockData: null,
    predictions: null,
    availableStocks: [],
    timeRange: '90', // días
    loading: true,
    error: null,
    showPredictions: true,
    chartType: 'price'
  });

  // Cargar lista de stocks disponibles al montar
  useEffect(() => {
    loadAvailableStocks();
  }, []);

  // Cargar datos cuando cambia el ticker o rango de tiempo
  useEffect(() => {
    if (state.selectedTicker) {
      loadStockData();
      if (state.showPredictions) {
        loadPredictions();
      }
    }
  }, [state.selectedTicker, state.timeRange]);

  const loadAvailableStocks = async () => {
    try {
      const response = await ApiService.getStocksList();
      if (response.data) {
        setState(prev => ({
          ...prev,
          availableStocks: response.data!.stocks,
          selectedTicker: response.data!.stocks.length > 0 ? response.data!.stocks[0].ticker : '',
          loading: false
        }));
      } else {
        setState(prev => ({
          ...prev,
          error: 'Error cargando lista de acciones',
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

  const loadStockData = async () => {
    if (!state.selectedTicker) return;

    try {
      setState(prev => ({ ...prev, loading: true, error: null }));

      const response = await ApiService.getStockHistory(
        state.selectedTicker,
        parseInt(state.timeRange)
      );

      if (response.data) {
        setState(prev => ({
          ...prev,
          stockData: response.data!,
          loading: false
        }));
      } else {
        setState(prev => ({
          ...prev,
          error: response.error || 'Error cargando datos históricos',
          loading: false
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: 'Error cargando datos históricos',
        loading: false
      }));
    }
  };

  const loadPredictions = async () => {
    if (!state.selectedTicker) return;

    try {
      const response = await ApiService.getPrediction(state.selectedTicker, 7, 0.8);
      
      if (response.data) {
        setState(prev => ({
          ...prev,
          predictions: response.data!
        }));
      }
      // No mostrar error si las predicciones fallan, son opcionales
    } catch (error) {
      console.warn('Predicciones no disponibles:', error);
    }
  };

  const handleTickerChange = (ticker: string) => {
    setState(prev => ({
      ...prev,
      selectedTicker: ticker,
      stockData: null,
      predictions: null
    }));
  };

  const handleTimeRangeChange = (range: string) => {
    setState(prev => ({
      ...prev,
      timeRange: range,
      stockData: null
    }));
  };

  const togglePredictions = () => {
    setState(prev => ({
      ...prev,
      showPredictions: !prev.showPredictions
    }));
    
    if (!state.showPredictions && state.selectedTicker) {
      loadPredictions();
    }
  };

  // Preparar datos para el gráfico
  const prepareChartData = () => {
    if (!state.stockData) return null;

    const data = state.stockData.data;
    const labels = data.map(item => {
      const date = new Date(item.date);
      return date.toLocaleDateString('es-ES', { 
        month: 'short', 
        day: 'numeric' 
      });
    });

    const prices = data.map(item => item.close);

    const datasets = [
      {
        label: `${state.selectedTicker} - Precio de Cierre`,
        data: prices,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointBackgroundColor: 'rgb(59, 130, 246)',
        pointBorderColor: 'rgb(255, 255, 255)',
        pointBorderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6
      }
    ];

    // Añadir predicciones si están disponibles
    if (state.showPredictions && state.predictions) {
      const predictionLabels = state.predictions.predictions.map(pred => {
        const date = new Date(pred.date);
        return date.toLocaleDateString('es-ES', { 
          month: 'short', 
          day: 'numeric' 
        });
      });

      const predictionPrices = state.predictions.predictions.map(pred => pred.predicted_price);
      const upperBounds = state.predictions.predictions.map(pred => pred.confidence_upper);
      const lowerBounds = state.predictions.predictions.map(pred => pred.confidence_lower);

      // Extender las etiquetas
      const extendedLabels = [...labels, ...predictionLabels];

      // Extender los datos históricos con valores null para las predicciones
      const extendedHistoricalData = [...prices, ...new Array(predictionPrices.length).fill(null)];

      // Crear datos de predicción con continuidad
      const lastHistoricalPrice = prices[prices.length - 1];
      const predictionData = [
        ...new Array(prices.length - 1).fill(null),
        lastHistoricalPrice,
        ...predictionPrices
      ];

      datasets[0].data = extendedHistoricalData;      datasets.push({
        label: 'Predicción',
        data: predictionData,
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5] as any,
        fill: false,
        tension: 0.1,
        pointBackgroundColor: 'rgb(16, 185, 129)',
        pointBorderColor: 'rgb(255, 255, 255)',
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 7
      } as any);

      // Banda de confianza
      const upperConfidence = [
        ...new Array(prices.length - 1).fill(null),
        lastHistoricalPrice,
        ...upperBounds
      ];
      const lowerConfidence = [
        ...new Array(prices.length - 1).fill(null),
        lastHistoricalPrice,
        ...lowerBounds
      ];      datasets.push({
        label: 'Límite Superior',
        data: upperConfidence,
        borderColor: 'rgba(16, 185, 129, 0.3)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 1,
        borderDash: [2, 2] as any,
        fill: 1 as any,
        pointRadius: 0,
        tension: 0.1
      } as any);

      datasets.push({
        label: 'Límite Inferior',
        data: lowerConfidence,
        borderColor: 'rgba(16, 185, 129, 0.3)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 1,
        borderDash: [2, 2] as any,
        fill: false,
        pointRadius: 0,
        tension: 0.1
      } as any);

      return {
        labels: extendedLabels,
        datasets
      };
    }

    return {
      labels,
      datasets
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          color: '#374151'
        }
      },
      title: {
        display: true,
        text: `${state.selectedTicker} - Análisis de Precios`,
        color: '#111827',        font: {
          size: 18,
          weight: 'bold' as const
        },
        padding: {
          top: 10,
          bottom: 30
        }
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        callbacks: {
          label: function(context: any) {
            const label = context.dataset.label;
            const value = context.parsed.y;
            if (value === null) return '';
            return `${label}: $${value.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Fecha',
          color: '#6B7280'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          color: '#6B7280',
          maxTicksLimit: 10
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Precio (USD)',
          color: '#6B7280'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          color: '#6B7280',
          callback: function(value: any) {
            return '$' + value.toFixed(2);
          }
        }
      }
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false
    },
    animation: {
      duration: 1000,
      easing: 'easeInOutQuart' as const
    }
  };

  const chartData = prepareChartData();

  if (state.loading) {
    return (
      <div className="charts-page">
        <div className="loading">
          <div className="spinner"></div>
          <p>Cargando datos...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="charts-page">
      <div className="charts-header">
        <h1>📈 Visualización de Gráficos</h1>
        <p>Analiza el comportamiento histórico y predicciones de tus acciones</p>
      </div>

      {/* Controles */}
      <div className="charts-controls">
        <div className="control-group">
          <label htmlFor="ticker-select">📊 Acción:</label>
          <select
            id="ticker-select"
            value={state.selectedTicker}
            onChange={(e) => handleTickerChange(e.target.value)}
            className="control-select"
          >
            <option value="">Seleccionar acción...</option>
            {state.availableStocks.map(stock => (
              <option key={stock.ticker} value={stock.ticker}>
                {stock.ticker} - {stock.name}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="time-range">⏱️ Periodo:</label>
          <select
            id="time-range"
            value={state.timeRange}
            onChange={(e) => handleTimeRangeChange(e.target.value)}
            className="control-select"
          >
            <option value="30">30 días</option>
            <option value="90">90 días</option>
            <option value="180">6 meses</option>
            <option value="365">1 año</option>
          </select>
        </div>

        <div className="control-group">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={state.showPredictions}
              onChange={togglePredictions}
              className="toggle-checkbox"
            />
            <span className="toggle-slider"></span>
            🔮 Mostrar Predicciones
          </label>
        </div>

        <button
          onClick={loadStockData}
          className="refresh-btn"
          disabled={!state.selectedTicker}
        >
          🔄 Actualizar
        </button>
      </div>

      {/* Error */}
      {state.error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {state.error}
        </div>
      )}

      {/* Información del Stock */}
      {state.stockData && (
        <div className="stock-info">
          <div className="info-cards">
            <div className="info-card">
              <div className="info-label">Precio Actual</div>
              <div className="info-value">
                ${state.stockData.data[state.stockData.data.length - 1]?.close.toFixed(2)}
              </div>
            </div>
            <div className="info-card">
              <div className="info-label">Registros</div>
              <div className="info-value">{state.stockData.total_records}</div>
            </div>
            <div className="info-card">
              <div className="info-label">Rango</div>
              <div className="info-value">
                {new Date(state.stockData.date_range.start).toLocaleDateString('es-ES')} - 
                {new Date(state.stockData.date_range.end).toLocaleDateString('es-ES')}
              </div>
            </div>
            {state.predictions && (
              <div className="info-card prediction-card">
                <div className="info-label">Predicción 7 días</div>
                <div className="info-value">
                  ${state.predictions.predictions[state.predictions.predictions.length - 1]?.predicted_price.toFixed(2)}
                  <span className="confidence">
                    (Confianza: {(state.predictions.confidence_score * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}      {/* Gráfico */}
      {chartData && (
        <div className="chart-container">
          <div className="chart-wrapper">
            <Line data={chartData} options={chartOptions as any} />
          </div>
        </div>
      )}

      {/* Análisis adicional */}
      {state.stockData && (
        <div className="analysis-section">
          <h3>📊 Análisis Técnico Básico</h3>
          <div className="analysis-grid">
            <div className="analysis-card">
              <h4>Precio Máximo</h4>
              <p>${Math.max(...state.stockData.data.map(d => d.high)).toFixed(2)}</p>
            </div>
            <div className="analysis-card">
              <h4>Precio Mínimo</h4>
              <p>${Math.min(...state.stockData.data.map(d => d.low)).toFixed(2)}</p>
            </div>
            <div className="analysis-card">
              <h4>Volumen Promedio</h4>
              <p>{(state.stockData.data.reduce((sum, d) => sum + d.volume, 0) / state.stockData.data.length).toLocaleString()}</p>
            </div>
            <div className="analysis-card">
              <h4>Variación del Periodo</h4>
              <p>
                {state.stockData.data.length > 1 ? (
                  ((state.stockData.data[state.stockData.data.length - 1].close - 
                    state.stockData.data[0].close) / 
                    state.stockData.data[0].close * 100).toFixed(2)
                ) : '0.00'}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Información sobre predicciones */}
      {state.showPredictions && state.predictions && (
        <div className="predictions-info">
          <h3>🔮 Información de Predicciones</h3>
          <div className="prediction-details">
            <p><strong>Modelo:</strong> {state.predictions.model_info.type || 'LSTM'}</p>
            <p><strong>Última actualización:</strong> {
              state.predictions.generated_at ? 
              new Date(state.predictions.generated_at).toLocaleString('es-ES') : 
              'N/A'
            }</p>
            <p><strong>Precio base:</strong> ${state.predictions.current_price.toFixed(2)}</p>
            <div className="prediction-list">
              <h4>Predicciones por día:</h4>
              {state.predictions.predictions.map((pred, index) => (
                <div key={index} className="prediction-item">
                  <span className="pred-date">{new Date(pred.date).toLocaleDateString('es-ES')}</span>
                  <span className="pred-price">${pred.predicted_price.toFixed(2)}</span>
                  <span className={`pred-trend trend-${pred.trend}`}>
                    {pred.trend === 'up' ? '📈' : pred.trend === 'down' ? '📉' : '➡️'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChartsPage;
