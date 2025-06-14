# Sistema de Entrenamiento Incremental - GuruInversor

## Descripción General

El sistema de entrenamiento incremental (MODEL-005) permite mantener los modelos LSTM actualizados automáticamente conforme llegan nuevos datos de mercado, preservando el conocimiento previo mientras se adapta a nuevas tendencias.

## Componentes Principales

### 1. IncrementalTrainer (`incremental_trainer.py`)

**Propósito**: Gestor principal del entrenamiento incremental de modelos LSTM.

**Características principales**:
- Detección automática de necesidad de reentrenamiento
- Versionado automático de modelos
- Preservación del conocimiento previo
- Detección de degradación de rendimiento
- Sistema de backup automático

**Configuración (`IncrementalConfig`)**:
```python
@dataclass
class IncrementalConfig:
    models_directory: str = "models"  # Directorio para modelos
    incremental_logs_directory: str = "logs"  # Logs específicos
    retrain_threshold_days: int = 7  # Días antes de reentrenar
    min_new_samples: int = 20  # Mínimo de nuevas muestras
    max_model_age_days: int = 30  # Edad máxima del modelo
    performance_degradation_threshold: float = 0.15  # Umbral de degradación
```

**Métodos principales**:
- `check_retrain_need(ticker)`: Evalúa si un modelo necesita reentrenamiento
- `retrain_model(ticker, type)`: Ejecuta reentrenamiento incremental o completo
- `batch_retrain(tickers)`: Reentrenamiento masivo de múltiples tickers
- `generate_incremental_report()`: Genera reporte del estado del sistema

### 2. TrainingScheduler (`training_scheduler.py`)

**Propósito**: Automatización de tareas de entrenamiento y monitoreo.

**Características principales**:
- Programación automática de verificaciones de salud
- Reentrenamiento automático programado
- Generación de reportes periódicos
- Sistema de alertas por degradación
- Ejecución manual de tareas

**Configuración (`SchedulerConfig`)**:
```python
@dataclass
class SchedulerConfig:
    daily_check_time: str = "06:00"  # Verificación diaria
    weekly_retrain_day: str = "sunday"  # Reentrenamiento semanal
    weekly_retrain_time: str = "02:00"  # Hora de reentrenamiento
    health_check_interval_hours: int = 6  # Monitoreo periódico
    performance_alert_threshold: float = 0.20  # Umbral de alerta
    active_tickers: List[str] = None  # Tickers activos
    priority_tickers: List[str] = None  # Tickers prioritarios
```

**Programaciones automáticas**:
- **Verificación diaria**: 06:00 - Revisa la salud de todos los modelos
- **Verificación periódica**: Cada 6 horas - Monitoreo de tickers prioritarios
- **Reentrenamiento semanal**: Domingos 02:00 - Reentrenamiento programado
- **Reportes diarios**: 23:30 - Generación de reportes
- **Reportes semanales**: Lunes 08:00 - Resumen semanal

## Flujo de Funcionamiento

### 1. Verificación de Necesidad de Reentrenamiento

El sistema evalúa múltiples criterios:

1. **Existencia del modelo**: Si no existe, requiere entrenamiento inicial
2. **Edad del modelo**: Modelos superiores a `max_model_age_days` necesitan reentrenamiento
3. **Nuevos datos**: Si hay `min_new_samples` o más muestras nuevas
4. **Degradación de rendimiento**: Si el rendimiento cae por debajo del umbral

### 2. Tipos de Reentrenamiento

**Incremental**:
- Usa modelo existente como punto de partida
- Entrena solo con datos nuevos
- Preserva conocimiento previo
- Más rápido y eficiente

**Completo**:
- Entrena desde cero con todos los datos
- Usado cuando hay degradación significativa
- Garantiza mejor adaptación a cambios de mercado
- Más lento pero más robusto

### 3. Sistema de Versionado

Cada modelo mantiene:
- **Versión**: Incremento automático (v1.0, v1.1, v2.0)
- **Fecha de entrenamiento**: Timestamp de última actualización
- **Tipo de entrenamiento**: incremental/complete
- **Métricas de rendimiento**: Scores de evaluación
- **Backup automático**: Preservación de versión anterior

## Métricas y Evaluación

### Criterios de Degradación

El sistema monitorea:
- **Score general**: Métrica principal de rendimiento
- **Tendencia**: Degradación sostenida en últimas evaluaciones
- **Threshold**: Caída superior al 15% (configurable)

### Reportes Generados

**Reporte Incremental**:
- Estado de todos los modelos
- Análisis por ticker
- Recomendaciones de acción
- Estadísticas generales

**Reporte del Programador**:
- Historial de ejecuciones
- Tasa de éxito de tareas
- Estado de programaciones
- Alertas generadas

## Integración con Otros Componentes

### Base de Datos
- Acceso a datos históricos y nuevos
- Almacenamiento de métricas de rendimiento
- Registro de eventos de entrenamiento

### Recolector de Datos
- Obtención de datos actualizados de Yahoo Finance
- Detección de nueva información disponible

### Evaluador de Modelos
- Evaluación de rendimiento después del entrenamiento
- Generación de métricas comparativas
- Detección de mejoras/degradación

### Preprocesador
- Preparación de datos para entrenamiento incremental
- Normalización consistente entre entrenamientos

## Logging y Monitoreo

### Archivos de Log

**Incremental Logs** (`logs/incremental_YYYYMMDD.log`):
- Eventos de reentrenamiento
- Decisiones de tipo de entrenamiento
- Métricas de rendimiento
- Errores y warnings

**Alert Logs** (`logs/alerts.log`):
- Alertas de degradación crítica
- Fallos en programaciones
- Tickers que requieren atención inmediata

### Niveles de Logging

- **INFO**: Operaciones normales, inicio de entrenamientos
- **WARNING**: Degradación detectada, modelos que necesitan atención
- **ERROR**: Fallos en entrenamiento, problemas de datos
- **CRITICAL**: Fallos sistémicos, pérdida de modelos

## Archivos y Directorios

```
models/
├── AAPL_model.keras          # Modelo actual
├── AAPL_model_info.json      # Metadatos del modelo
├── AAPL_model_backup.keras   # Backup de versión anterior
├── GOOGL_model.keras
├── GOOGL_model_info.json
└── ...

logs/
├── incremental_20250611.log  # Logs diarios
├── alerts.log                # Log de alertas
└── ...

reports/
├── daily_report_20250611.txt   # Reportes diarios
├── weekly_report_week24.txt    # Reportes semanales
└── ...
```

## Uso del Sistema

### Inicialización Básica

```python
from ml.incremental_trainer import IncrementalTrainer, IncrementalConfig
from ml.training_scheduler import TrainingScheduler, SchedulerConfig

# Configurar entrenador incremental
config = IncrementalConfig(
    models_directory="models",
    retrain_threshold_days=7,
    min_new_samples=20
)

trainer = IncrementalTrainer(config)

# Verificar si un ticker necesita reentrenamiento
result = trainer.check_retrain_need("AAPL")
if result['needs_retrain']:
    print(f"AAPL necesita reentrenamiento: {result['reasons']}")
    
# Ejecutar reentrenamiento
retrain_result = trainer.retrain_model("AAPL", "incremental")
```

### Programación Automática

```python
# Configurar programador
scheduler_config = SchedulerConfig(
    active_tickers=["AAPL", "GOOGL", "MSFT"],
    priority_tickers=["AAPL", "MSFT"]
)

scheduler = TrainingScheduler(scheduler_config, config)

# Iniciar programación automática
scheduler.start()

# Obtener estado
status = scheduler.get_status()
print(f"Programador ejecutándose: {status['is_running']}")

# Ejecutar tarea manual
result = scheduler.execute_manual_task('health_check')
```

### Reentrenamiento Masivo

```python
# Reentrenar múltiples tickers
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
batch_result = trainer.batch_retrain(tickers, force=False)

print(f"Reentrenados: {batch_result['summary']['retrained_count']}")
print(f"Omitidos: {batch_result['summary']['skipped_count']}")
print(f"Fallidos: {batch_result['summary']['failed_count']}")
```

## Pruebas y Validación

El sistema incluye una suite completa de pruebas:

### TestIncrementalTrainer
- Inicialización y configuración
- Verificación de necesidad de reentrenamiento
- Registro y versionado de modelos
- Detección de degradación de rendimiento
- Funcionalidad de backup

### TestTrainingScheduler
- Inicialización del programador
- Ejecución manual de tareas
- Inicio y parada del programador
- Reporte de estado

### TestIntegrationIncremental
- Flujo completo de entrenamiento incremental
- Integración entre componentes
- Validación end-to-end

**Ejecutar pruebas**:
```bash
cd backend/ml
python test_model_005_incremental.py
```

## Rendimiento y Optimización

### Optimizaciones Implementadas

1. **Logging Eficiente**: Evita duplicación de handlers
2. **Backup Selectivo**: Solo cuando es necesario
3. **Verificación Inteligente**: Evaluación gradual de criterios
4. **Cache de Modelos**: Reutilización de modelos cargados
5. **Cleanup Automático**: Gestión de memoria y archivos temporales

### Recomendaciones de Rendimiento

- **Tickers Prioritarios**: Usar lista reducida para monitoreo frecuente
- **Intervalos Balanceados**: No hacer verificaciones muy frecuentes
- **Batch Processing**: Agrupar reentrenamientos cuando sea posible
- **Storage Management**: Limpieza periódica de backups antiguos

## Monitoreo y Alertas

### Alertas Automáticas

El sistema genera alertas cuando:
- Degradación de rendimiento > 20%
- Fallos repetidos en reentrenamiento
- Modelos con edad > límite configurado
- Errores críticos en programaciones

### Métricas de Salud

- **Tasa de éxito de reentrenamientos**
- **Tiempo promedio de entrenamiento**
- **Frecuencia de degradación detectada**
- **Disponibilidad del sistema**

## Escalabilidad

### Capacidad Actual
- **Tickers simultáneos**: Hasta 50 (recomendado)
- **Verificaciones por día**: Ilimitadas
- **Almacenamiento**: Escalable con storage disponible

### Limitaciones
- **Memoria**: Depende del tamaño de modelos cargados
- **CPU**: Entrenamiento secuencial (no paralelo)
- **Storage**: Crecimiento lineal con número de modelos

## Troubleshooting

### Problemas Comunes

**Error: "No existe modelo previo"**
- Solución: Ejecutar entrenamiento inicial completo

**Error: "Degradación del rendimiento"**
- Verificar calidad de datos nuevos
- Considerar entrenamiento completo en lugar de incremental

**Error: "Falló la programación"**
- Verificar recursos del sistema
- Revisar logs de alertas

**Error: "No se pudo configurar logging"**
- Verificar permisos de escritura en directorios
- Comprobar encoding UTF-8 del sistema

### Logs de Diagnóstico

```bash
# Ver logs de entrenamiento incremental
tail -f logs/incremental_$(date +%Y%m%d).log

# Ver alertas
tail -f logs/alerts.log

# Ver reporte más reciente
ls -la reports/ | head -5
```

---

**Documentación actualizada**: 11 de junio de 2025  
**Versión del sistema**: MODEL-005 v1.0  
**Estado**: Completamente funcional y validado
