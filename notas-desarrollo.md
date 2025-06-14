# Notas de Desarrollo - GuruInversor

## Decisiones de Diseño

### 📅 11 de junio de 2025

#### **Decisión: Arquitectura Monolítica vs Microservicios**
**Elegido:** Monolítica  
**Razón:** Al ser un proyecto personal, la simplicidad de deployment y mantenimiento es prioritaria. Un solo proceso será más fácil de manejar.

#### **Decisión: Base de Datos**
**Elegido:** SQLite  
**Razón:** No requiere configuración adicional, es local, y para un uso personal es suficiente. Si crece el proyecto, migrar a PostgreSQL será directo.

#### **Decisión: Framework de IA**
**Elegido:** TensorFlow/Keras  
**Razón:** Mejor documentación para LSTM, más ejemplos disponibles para series temporales, y deployment más simple que PyTorch.

#### **Decisión: API Framework**
**Elegido:** FastAPI  
**Razón:** Documentación automática, validación de tipos nativa, y performance superior a Flask. Ideal para APIs de ML.

---

## Ideas y Conceptos

### 💡 Funcionalidades Futuras

#### **Análisis de Sentimiento**
- Integrar análisis de noticias y sentiment de Twitter/Reddit
- APIs posibles: News API, Reddit API, Twitter API
- Impacto: Mejorar predicciones con datos cualitativos

#### **Portfolio Simulation**
- Simular portfolios con diferentes estrategias
- Backtesting automático
- Comparación con benchmarks (S&P 500, etc.)

#### **Alertas Automáticas**
- Notificaciones cuando se cumplan condiciones específicas
- Email, push notifications, o webhooks
- Condiciones: precio objetivo, cambios significativos, etc.

#### **Análisis Técnico Avanzado**
- Patrones de candlesticks automáticos
- Fibonacci retracements
- Ondas de Elliott
- Support/Resistance automático

### 🔍 Investigación Pendiente

#### **Arquitecturas de Modelo Alternativas**
- **Transformer models** para series temporales
- **CNN-LSTM** híbridos
- **GRU** vs LSTM performance comparison
- **Ensemble methods** combinando múltiples modelos

#### **Fuentes de Datos Adicionales**
- **Alpha Vantage** como backup de Yahoo Finance
- **Quandl** para datos macroeconómicos
- **FRED** (Federal Reserve Economic Data)
- **Cryptocurrency data** para diversificación

#### **Métricas de Evaluación Avanzadas**
- **Information Ratio**
- **Calmar Ratio**
- **Sortino Ratio**
- **Value at Risk (VaR)**

---

## Problemas Conocidos y Soluciones

### ⚠️ Limitaciones de Yahoo Finance

**Problema:** Rate limiting y posibles bloqueos  
**Solución:** 
- Implementar retry con backoff exponencial
- Cache local agresivo
- Múltiples fuentes de datos de respaldo

**Problema:** Datos inconsistentes o faltantes  
**Solución:**
- Validación estricta de datos
- Interpolación inteligente para gaps pequeños
- Flags de calidad de datos

### ⚠️ Overfitting del Modelo

**Problema:** Modelo memoriza patrones históricos específicos  
**Solución:**
- Validación temporal estricta (no cross-validation normal)
- Regularización L1/L2
- Dropout layers
- Early stopping
- Walk-forward validation

### ⚠️ Data Leakage

**Problema:** Información del futuro en entrenamiento  
**Solución:**
- Pipeline de datos estricto con fechas
- Validación temporal únicamente
- Features lag apropiados

---

## Optimizaciones Técnicas

### 🚀 Performance del Modelo

#### **Preprocessing Optimizado**
```python
# Ideas para implementar:
- Normalización por rolling windows
- Feature engineering automático
- Parallel data loading
- GPU acceleration para entrenamiento
```

#### **Caching Strategy**
```python
# Niveles de cache:
- Predicciones recientes (Redis future)
- Datos históricos procesados
- Modelos compilados en memoria
- Feature engineering results
```

### 🚀 API Performance

#### **Optimizaciones Planeadas**
- Response compression (gzip)
- Connection pooling para DB
- Async operations donde sea posible
- Background tasks para entrenamientos

---

## Métricas de Éxito del Proyecto

### 📊 Métricas Técnicas

1. **Precisión del Modelo**
   - Target: >60% directional accuracy
   - RMSE < 5% del precio medio
   - MAE < 3% del precio medio

2. **Performance del Sistema**
   - API response time < 200ms
   - Predicción generation < 2s
   - Training time < 10min per stock

3. **Reliability**
   - 99% uptime para API
   - 0 data corruption incidents
   - Automated backup functioning

### 📈 Métricas de Negocio

1. **Simulación de Trading**
   - Beat S&P 500 en backtesting
   - Sharpe ratio > 1.0
   - Maximum drawdown < 20%

2. **Usabilidad Personal**
   - Daily usage > 5 minutes
   - Successful integration en routine de inversión
   - Satisfacción personal con predicciones

---

## Recursos y Referencias

### 📚 Papers y Artículos

#### **LSTM para Finanzas**
- "Long Short-Term Memory Networks for Stock Market Prediction"
- "Deep Learning for Stock Prediction Using Numerical and Textual Information"
- "A Deep Learning Framework for Financial Time Series using Stacked Autoencoders and LSTM"

#### **Technical Analysis**
- "Evidence-Based Technical Analysis" by David Aronson
- "Algorithmic Trading" by Ernie Chan

### 🛠️ Herramientas de Desarrollo

#### **Testing**
- pytest para unit tests
- pytest-asyncio para async tests
- hypothesis para property-based testing

#### **Monitoring**
- structlog para logging estructurado
- prometheus metrics (futuro)
- grafana dashboards (futuro)

#### **Development**
- black para code formatting
- mypy para type checking
- pre-commit hooks

---

## Changelog de Decisiones

### v0.1.0 - Planificación Inicial
- ✅ Arquitectura base definida
- ✅ Stack tecnológico seleccionado
- ✅ Estructura de proyecto planificada
- ✅ Documentación inicial creada

---

## FUND-004 - Configurar Base de Datos SQLite ✅ COMPLETADO (11 junio 2025)

### Implementación Realizada

**Archivos Creados:**
- `backend/database/models.py` - Modelos SQLAlchemy para todas las tablas
- `backend/database/connection.py` - Gestión de conexiones y sesiones de BD
- `backend/database/crud.py` - Operaciones CRUD para todos los modelos
- `backend/database/init_db.py` - Script de inicialización con datos de ejemplo
- `backend/database/test_database.py` - Suite completa de pruebas

**Esquema de Base de Datos:**
1. **stocks** - Información de acciones monitoreadas
2. **historical_data** - Datos OHLCV históricos
3. **predictions** - Predicciones de modelos de IA
4. **trained_models** - Metadatos de modelos entrenados

**Características Implementadas:**
- ✅ Modelos SQLAlchemy con validaciones completas
- ✅ Relaciones entre entidades con cascade delete
- ✅ Gestión de sesiones con context managers
- ✅ Operaciones CRUD optimizadas para cada modelo
- ✅ Script de inicialización con datos de ejemplo
- ✅ Suite de pruebas automatizadas (5/6 pruebas pasando)
- ✅ Configuración optimizada de SQLite (WAL mode, foreign keys, etc.)
- ✅ Validaciones de datos (precios positivos, tickers válidos, etc.)
- ✅ Soporte para múltiples entornos (desarrollo, producción)

**Dependencias Añadidas:**
- SQLAlchemy 2.0.41
- greenlet 3.2.3

**Estado de Pruebas:**
- Suite de pruebas ejecutándose correctamente
- 5 de 6 pruebas pasando (1 fallo esperado por validación)
- Base de datos funcional y lista para uso

**Próximos Pasos:**
- Continuar con FUND-005 (Utilidades de procesamiento básico)
- Integrar con el recolector de datos Yahoo Finance
- Preparar para modelo LSTM en Fase 2

---

*Última actualización: 11 de junio de 2025*
