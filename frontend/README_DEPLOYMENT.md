# GuruInversor Frontend

Frontend de la aplicación GuruInversor construido con React + TypeScript + Vite.

## 🚀 Deployment en Vercel

### Configuración Automática
El proyecto está configurado para deployment automático en Vercel desde GitHub.

### Variables de Entorno Requeridas
- `VITE_API_URL`: URL del backend (configurada automáticamente en vercel.json)

### Scripts Disponibles
```bash
npm run dev      # Desarrollo local
npm run build    # Build para producción
npm run preview  # Preview del build
npm run lint     # Verificar código
```

## 🔧 Configuración

### Backend URL
El frontend se conecta automáticamente al backend desplegado en Railway:
- **Producción**: `https://guruinversor-v1-production.up.railway.app`
- **Desarrollo**: `http://localhost:8000`

### Estructura del Proyecto
```
src/
├── components/     # Componentes React
├── pages/         # Páginas principales
├── services/      # Servicios de API
├── types/         # Tipos TypeScript
├── hooks/         # Custom hooks
└── utils/         # Utilidades
```

## 📱 Funcionalidades

- ✅ Dashboard de stocks
- ✅ Visualización de datos históricos
- ✅ Sistema de predicciones
- ✅ Gestión de portafolio
- ✅ Autenticación JWT
- ✅ Métricas del sistema

## 🌐 URLs de Producción

- **Frontend**: [Pendiente - se configurará en Vercel]
- **Backend**: https://guruinversor-v1-production.up.railway.app
- **API Docs**: https://guruinversor-v1-production.up.railway.app/docs
