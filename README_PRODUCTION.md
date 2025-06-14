# 🚀 GuruInversor v1.0 - Production Ready

[![Backend](https://img.shields.io/badge/Backend-Railway-purple)](https://railway.app)
[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black)](https://vercel.com)
[![Database](https://img.shields.io/badge/Database-PostgreSQL-blue)](https://postgresql.org)

Sistema de predicción de precios de acciones con IA, listo para producción.

## 🌐 URLs de Producción

- **Frontend**: https://guruzinversor-frontend.vercel.app
- **Backend API**: https://guruzinversor-backend.railway.app
- **Documentación**: https://guruzinversor-backend.railway.app/docs

## 📊 Estado del Sistema

- **Versión**: 1.0.0
- **Estado**: ✅ Producción
- **Tickers**: GGAL.BA, YPF, JPM, NKE, KO
- **Datos Históricos**: 307 registros (90 días por ticker)
- **Última Actualización**: 14 de junio de 2025

## 🛠️ Stack Tecnológico

### Backend
- **Framework**: FastAPI
- **Base de Datos**: PostgreSQL (Railway)
- **ML**: TensorFlow/Keras
- **Datos**: Yahoo Finance API
- **Deploy**: Railway

### Frontend
- **Framework**: React + TypeScript
- **Build Tool**: Vite
- **Charts**: Chart.js
- **Deploy**: Vercel

## 🚀 Deployment

Este repositorio está configurado para deployment automático:

### Backend (Railway)
1. Conectar Railway con este repositorio
2. Variables de entorno configuradas automáticamente
3. Deploy automático en cada push a `main`

### Frontend (Vercel)
1. Conectar Vercel con este repositorio
2. Deploy automático del directorio `frontend/`
3. Variables de entorno configuradas automáticamente

## 📁 Estructura del Proyecto

```
GuruInversor/
├── backend/                 # API FastAPI
│   ├── app/                # Aplicación principal
│   ├── data/               # Datos locales (dev)
│   ├── database/           # Configuración DB
│   ├── ml/                 # Modelos de ML
│   └── requirements.txt    # Dependencias Python
├── frontend/               # Aplicación React
│   ├── src/               # Código fuente
│   ├── public/            # Archivos públicos
│   └── package.json       # Dependencias Node
├── Dockerfile             # Configuración Docker
├── railway.json           # Configuración Railway
└── README.md             # Este archivo
```

## 🔧 Variables de Entorno

### Backend (Railway)
```env
DATABASE_URL=postgresql://...
ENVIRONMENT=production
PORT=8000
CORS_ORIGINS=https://guruzinversor-frontend.vercel.app
```

### Frontend (Vercel)
```env
VITE_API_URL=https://guruzinversor-backend.railway.app
```

## 📈 Funcionalidades

- ✅ **Datos en Tiempo Real**: Precios actualizados de Yahoo Finance
- ✅ **Predicciones ML**: Modelo entrenado para predicción de precios
- ✅ **Visualizaciones**: Gráficos interactivos de precios históricos
- ✅ **API REST**: Endpoints documentados con Swagger
- ✅ **Responsive**: Interfaz adaptable a dispositivos móviles

## 🔍 Health Checks

- **Backend**: `GET /api/health`
- **Frontend**: Disponibilidad general de la aplicación
- **Base de Datos**: Verificación automática de conexión

## 📞 Soporte

Para reportes de bugs o sugerencias, crear un issue en este repositorio.

---

**Desarrollado por**: Equipo GuruInversor  
**Licencia**: MIT  
**Versión**: 1.0.0
