# Configuración del Entorno de Desarrollo - GuruInversor

## Activación del Entorno Virtual

### Windows (PowerShell)
```bash
cd backend
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```bash
cd backend
venv\Scripts\activate.bat
```

### Linux/Mac
```bash
cd backend
source venv/bin/activate
```

## Verificación de la Instalación

Una vez activado el entorno virtual, puedes verificar que todo está instalado correctamente:

```bash
# Verificar TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Verificar yfinance
python -c "import yfinance as yf; print('yfinance: OK')"

# Verificar FastAPI
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

# Verificar todas las dependencias principales
pip list | grep -E "(tensorflow|fastapi|yfinance|pandas|numpy)"
```

## Dependencias Instaladas

- **FastAPI**: Framework para crear APIs REST
- **TensorFlow**: Biblioteca de machine learning
- **yfinance**: Descarga de datos de Yahoo Finance
- **pandas**: Manipulación de datos
- **numpy**: Computación numérica
- **scikit-learn**: Utilidades de ML
- **matplotlib/seaborn**: Visualización de datos
- **pytest**: Testing framework
- **black**: Formateador de código
- **mypy**: Type checking

## Variables de Entorno

El archivo `.env` contiene las variables de configuración del proyecto. Las principales son:

- `DATABASE_URL`: Ubicación de la base de datos SQLite
- `MODEL_PATH`: Directorio para modelos entrenados
- `DATA_PATH`: Directorio para datos
- `API_HOST` y `API_PORT`: Configuración del servidor API

## Próximos Pasos

Con el entorno configurado, puedes continuar con:
1. FUND-003: Implementar recolector de datos Yahoo Finance
2. FUND-004: Configurar base de datos SQLite
3. FUND-005: Crear utilidades de procesamiento básico
