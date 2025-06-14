# Especificaciones Técnicas - GuruInversor

## Arquitectura del Sistema

### Diagrama de Componentes

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Data Layer    │
│   (React/Vue)   │◄───┤   (FastAPI)     │◄───┤   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AI Engine     │
                       │ (TensorFlow)    │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Yahoo Finance   │
                       │   Data Source   │
                       └─────────────────┘
```

## Requerimientos Técnicos

### Backend

**Lenguaje:** Python 3.9+

**Dependencias Principales:**
```python
fastapi>=0.104.0          # API REST framework
tensorflow>=2.13.0        # Machine Learning
yfinance>=0.2.20          # Yahoo Finance data
pandas>=2.1.0             # Data manipulation
numpy>=1.24.0             # Numerical computing
sqlite3                   # Database (built-in)
uvicorn>=0.23.0          # ASGI server
pydantic>=2.4.0          # Data validation
```

**Dependencias Adicionales:**
```python
scikit-learn>=1.3.0      # ML utilities
matplotlib>=3.7.0        # Plotting
seaborn>=0.12.0          # Statistical plotting
ta>=0.10.0               # Technical analysis
requests>=2.31.0         # HTTP requests
python-dotenv>=1.0.0     # Environment variables
```

### Frontend

**Framework:** React 18+ o Vue.js 3+

**Dependencias Principales:**
```json
{
  "react": "^18.2.0",
  "chart.js": "^4.4.0",
  "react-chartjs-2": "^5.2.0",
  "axios": "^1.5.0",
  "tailwindcss": "^3.3.0"
}
```

### Base de Datos

**Motor:** SQLite 3

**Esquema Inicial:**
```sql
-- Tabla de acciones
CREATE TABLE stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    name TEXT,
    sector TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de datos históricos
CREATE TABLE historical_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER,
    date DATE NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    adj_close REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks (id),
    UNIQUE(stock_id, date)
);

-- Tabla de predicciones
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER,
    prediction_date DATE NOT NULL,
    predicted_price REAL NOT NULL,
    confidence REAL,
    actual_price REAL,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks (id)
);

-- Tabla de modelos entrenados
CREATE TABLE trained_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER,
    model_path TEXT NOT NULL,
    version TEXT NOT NULL,
    accuracy REAL,
    loss REAL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (stock_id) REFERENCES stocks (id)
);
```

## Especificaciones del Modelo de IA

### Arquitectura LSTM

**Configuración Base:**
```python
model_config = {
    "sequence_length": 60,        # 60 días de historia
    "features": 5,                # OHLCV
    "lstm_units": [50, 50],       # 2 capas LSTM
    "dropout_rate": 0.2,
    "dense_units": 25,
    "output_units": 1,            # Precio predicho
    "activation": "relu",
    "optimizer": "adam",
    "loss": "mse",
    "batch_size": 32,
    "epochs": 100
}
```

**Características de Entrada:**
1. **Precio de Apertura** (normalizado)
2. **Precio Máximo** (normalizado)
3. **Precio Mínimo** (normalizado)
4. **Precio de Cierre** (normalizado)
5. **Volumen** (normalizado)

**Características Futuras:**
- Indicadores técnicos (RSI, MACD, Bollinger Bands)
- Medias móviles (SMA, EMA)
- Volatilidad histórica
- Datos de sentimiento del mercado

### Métricas de Evaluación

**Métricas Principales:**
```python
metrics = {
    "rmse": "Root Mean Square Error",
    "mae": "Mean Absolute Error",
    "mape": "Mean Absolute Percentage Error",
    "directional_accuracy": "% aciertos en dirección",
    "r2_score": "Coefficient of determination"
}
```

**Métricas de Negocio:**
- Rentabilidad simulada
- Sharpe Ratio
- Maximum Drawdown
- Win Rate

## API Endpoints

### Endpoints Principales

```python
# Gestión de acciones
GET    /api/stocks                    # Listar acciones monitoreadas
POST   /api/stocks                    # Añadir nueva acción
GET    /api/stocks/{ticker}           # Obtener datos de acción
DELETE /api/stocks/{ticker}           # Eliminar acción

# Datos históricos
GET    /api/stocks/{ticker}/history   # Obtener datos históricos
POST   /api/stocks/{ticker}/update    # Actualizar datos

# Predicciones
GET    /api/stocks/{ticker}/predict   # Obtener predicción
POST   /api/stocks/{ticker}/train     # Entrenar modelo

# Estrategias
GET    /api/stocks/{ticker}/strategy  # Obtener estrategia recomendada

# Sistema
GET    /api/health                    # Estado del sistema
GET    /api/models                    # Información de modelos
```

### Formato de Respuestas

**Predicción:**
```json
{
  "ticker": "AAPL",
  "current_price": 180.50,
  "predicted_price": 182.30,
  "confidence": 0.75,
  "prediction_date": "2025-06-12",
  "change_percent": 1.00,
  "direction": "up",
  "model_version": "v1.2.0"
}
```

**Estrategia:**
```json
{
  "ticker": "AAPL",
  "recommendation": "BUY",
  "confidence": 0.80,
  "target_price": 185.00,
  "stop_loss": 175.00,
  "risk_level": "MEDIUM",
  "reasoning": "Tendencia alcista confirmada con RSI favorable"
}
```

## Configuración del Entorno

### Variables de Entorno

```bash
# .env file
DATABASE_URL=sqlite:///./data/guruminversion.db
MODEL_PATH=./models/
DATA_PATH=./data/
LOG_LEVEL=INFO
API_HOST=localhost
API_PORT=8000

# Configuración del modelo
SEQUENCE_LENGTH=60
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.001
```

### Estructura de Directorios

```
GuruInversor/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app
│   │   ├── models/                 # Pydantic models
│   │   ├── routers/                # API routes
│   │   ├── services/               # Business logic
│   │   └── utils/                  # Utilities
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py            # Data collection
│   │   ├── processor.py            # Data processing
│   │   └── database.py             # Database operations
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py                # LSTM model
│   │   ├── trainer.py              # Training logic
│   │   ├── predictor.py            # Prediction logic
│   │   └── evaluator.py            # Model evaluation
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   ├── public/
│   └── package.json
├── data/                           # SQLite database
├── models/                         # Trained models
├── logs/                           # Application logs
└── tests/                          # Unit tests
```

## Consideraciones de Rendimiento

### Optimizaciones de Base de Datos
- Índices en campos de fecha y ticker
- Particionamiento por fecha si es necesario
- Cache de consultas frecuentes

### Optimizaciones de Modelo
- Carga lazy de modelos
- Cache de predicciones recientes
- Entrenamiento en background

### Optimizaciones de API
- Rate limiting
- Response caching
- Compresión de respuestas
- Paginación de resultados
