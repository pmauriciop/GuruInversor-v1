# Deployment GuruInversor v1.0

Este documento describe el proceso de deployment de GuruInversor v1.0 a producción.

## 🎯 Objetivos del Deployment

1. **Preservar versión funcional** actual como baseline
2. **Ambiente de producción** accesible vía web
3. **Base de datos productiva** con datos reales
4. **Monitoreo básico** de la aplicación
5. **Rollback rápido** en caso de problemas

## 🛠️ Stack de Deployment

### Frontend
- **Plataforma**: Vercel
- **Dominio**: `guruzinversor-frontend.vercel.app`
- **Build**: React + Vite
- **Variables**: `VITE_API_URL`

### Backend  
- **Plataforma**: Railway
- **Dominio**: `guruzinversor-backend.railway.app`
- **Runtime**: Python 3.11
- **Base de Datos**: PostgreSQL (Railway)

### Monitoring
- **Health checks**: Endpoint `/api/health`
- **Logs**: Railway dashboard
- **Uptime**: UptimeRobot (opcional)

## 📋 Checklist Pre-Deployment

### Backend
- [x] Sistema funcionando localmente
- [x] Base de datos con datos reales (5 tickers, 307 registros)
- [x] Todos los endpoints operativos
- [x] Tests pasando
- [ ] Variables de entorno configuradas
- [ ] Dockerfile creado
- [ ] railway.json configurado

### Frontend
- [x] Aplicación funcionando localmente
- [x] Conexión con backend local OK
- [x] Build sin errores
- [ ] Variables de entorno para producción
- [ ] Build optimizado para producción

### Base de Datos
- [x] Esquema definido
- [x] Datos históricos reales
- [ ] Script de migración para PostgreSQL
- [ ] Backup de datos actual

## 🚀 Proceso de Deployment

### Fase 1: Preparación (30 min)
1. Crear archivos de configuración
2. Configurar variables de entorno
3. Adaptar código para PostgreSQL
4. Crear scripts de migración

### Fase 2: Backend Deploy (20 min)
1. Subir código a GitHub
2. Conectar Railway con repository
3. Configurar variables de entorno
4. Deploy automático
5. Migrar base de datos

### Fase 3: Frontend Deploy (15 min)
1. Configurar variables para producción
2. Deploy en Vercel
3. Conectar con backend productivo
4. Verificar funcionamiento

### Fase 4: Verificación (15 min)
1. Tests de endpoints
2. Verificación de datos
3. Tests de UI funcional
4. Documentación de URLs

## 📊 Métricas Post-Deployment

- **Uptime target**: 99%
- **Response time**: < 2s
- **Data consistency**: 100%
- **Feature parity**: 100% vs local

## 🔄 Proceso de Rollback

En caso de problemas:
1. Railway: Rollback a deployment anterior
2. Vercel: Rollback a build anterior  
3. Base de datos: Restore desde backup
4. DNS: Mantener URLs existentes

## 📝 URLs de Producción

- **Frontend**: https://guruzinversor-frontend.vercel.app
- **Backend API**: https://guruzinversor-backend.railway.app
- **Documentación**: https://guruzinversor-backend.railway.app/docs
- **Health Check**: https://guruzinversor-backend.railway.app/api/health

---

**Estado**: 🟡 En preparación  
**Responsable**: Equipo GuruInversor  
**Timeline**: ~1.5 horas
