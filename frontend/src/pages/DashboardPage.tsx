import React, { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth.tsx';
import ApiService from '../services/api';
import type { HealthCheck, SystemMetrics } from '../types/api';
import './DashboardPage.css';

const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const [healthData, setHealthData] = useState<HealthCheck | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadDashboardData();
    
    // Actualizar datos cada 30 segundos
    const interval = setInterval(loadDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      setError('');
      
      // Cargar health check y métricas del sistema en paralelo
      const [healthResponse, metricsResponse] = await Promise.all([
        ApiService.getHealthCheck(),
        ApiService.getSystemMetrics()
      ]);

      if (healthResponse.data) {
        setHealthData(healthResponse.data);
      }

      if (metricsResponse.data) {
        setSystemMetrics(metricsResponse.data);
      }

      if (healthResponse.error && metricsResponse.error) {
        setError('Error cargando datos del dashboard');
      }
    } catch (err) {
      setError('Error de conexión');
      console.error('Error loading dashboard:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy': return 'green';
      case 'warning': return 'orange';
      case 'critical': return 'red';
      default: return 'gray';
    }
  };

  const formatUptime = (uptime: string) => {
    if (!uptime) return 'N/A';
    
    // Si contiene 'day', extraer solo la parte relevante
    if (uptime.includes('day')) {
      const parts = uptime.split(',');
      return parts.length > 1 ? `${parts[0]}, ${parts[1].trim()}` : uptime;
    }
    
    return uptime;
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Cargando dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Dashboard</h1>
        <p>Bienvenido, <strong>{user?.username}</strong> ({user?.role})</p>
        <button onClick={loadDashboardData} className="refresh-btn">
          🔄 Actualizar
        </button>
      </div>

      {error && (
        <div className="error-banner">
          {error}
        </div>
      )}

      <div className="dashboard-grid">
        {/* Estado del Sistema */}
        <div className="dashboard-card system-status">
          <div className="card-header">
            <h2>🏥 Estado del Sistema</h2>
            {healthData && (
              <span 
                className={`status-badge ${getStatusColor(healthData.status)}`}
              >
                {healthData.status.toUpperCase()}
              </span>
            )}
          </div>
          
          {healthData ? (
            <div className="status-content">
              <div className="status-grid">
                <div className="status-item">
                  <span className="label">CPU:</span>
                  <span className="value">{healthData.system_metrics.cpu}</span>
                </div>
                <div className="status-item">
                  <span className="label">Memoria:</span>
                  <span className="value">{healthData.system_metrics.memory}</span>
                </div>
                <div className="status-item">
                  <span className="label">Disco:</span>
                  <span className="value">{healthData.system_metrics.disk}</span>
                </div>
                <div className="status-item">
                  <span className="label">Modelos:</span>
                  <span className="value">{healthData.model_metrics.trained_models}</span>
                </div>
              </div>

              {healthData.health_issues && healthData.health_issues.length > 0 && (
                <div className="health-issues">
                  <h4>⚠️ Problemas detectados:</h4>
                  <ul>
                    {healthData.health_issues.map((issue, index) => (
                      <li key={index}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="no-data">No se pudo cargar el estado del sistema</div>
          )}
        </div>

        {/* Métricas del Sistema */}
        <div className="dashboard-card system-metrics">
          <div className="card-header">
            <h2>📊 Métricas del Sistema</h2>
          </div>
          
          {systemMetrics ? (
            <div className="metrics-content">
              <div className="metric-item">
                <div className="metric-label">CPU Usage</div>
                <div className="metric-value">{systemMetrics.cpu_usage}%</div>
                <div className="metric-bar">
                  <div 
                    className="metric-fill cpu" 
                    style={{ width: `${systemMetrics.cpu_usage}%` }}
                  ></div>
                </div>
              </div>

              <div className="metric-item">
                <div className="metric-label">Memory Usage</div>
                <div className="metric-value">{systemMetrics.memory_usage}%</div>
                <div className="metric-bar">
                  <div 
                    className="metric-fill memory" 
                    style={{ width: `${systemMetrics.memory_usage}%` }}
                  ></div>
                </div>
              </div>

              <div className="metric-item">
                <div className="metric-label">Disk Usage</div>
                <div className="metric-value">{systemMetrics.disk_usage}%</div>
                <div className="metric-bar">
                  <div 
                    className="metric-fill disk" 
                    style={{ width: `${systemMetrics.disk_usage}%` }}
                  ></div>
                </div>
              </div>

              <div className="system-info">
                <div className="info-item">
                  <span>Python Memory:</span>
                  <span>{systemMetrics.python_memory} MB</span>
                </div>
                <div className="info-item">
                  <span>Uptime:</span>
                  <span>{formatUptime(systemMetrics.uptime)}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="no-data">No se pudieron cargar las métricas</div>
          )}
        </div>

        {/* Acciones Rápidas */}
        <div className="dashboard-card quick-actions">
          <div className="card-header">
            <h2>⚡ Acciones Rápidas</h2>
          </div>
          
          <div className="actions-grid">
            <button className="action-btn primary">
              📈 Nueva Predicción
            </button>
            <button className="action-btn secondary">
              🤖 Ver Modelos
            </button>
            <button className="action-btn secondary">
              📊 Análisis Avanzado
            </button>
            <button className="action-btn secondary">
              ⚙️ Configuración
            </button>
          </div>
        </div>

        {/* Información de API */}
        <div className="dashboard-card api-info">
          <div className="card-header">
            <h2>📡 API GuruInversor</h2>
          </div>
          
          <div className="api-content">
            <div className="api-item">
              <span>Estado:</span>
              <span className="status-badge green">ONLINE</span>
            </div>
            <div className="api-item">
              <span>Success Rate:</span>
              <span>{healthData?.api_metrics.success_rate || 'N/A'}</span>
            </div>
            <div className="api-item">
              <span>Tiempo Promedio:</span>
              <span>{healthData?.api_metrics.avg_response || 'N/A'}</span>
            </div>
            <div className="api-item">
              <span>Última Actualización:</span>
              <span>
                {systemMetrics ? 
                  new Date(systemMetrics.timestamp).toLocaleTimeString() : 
                  'N/A'
                }
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
