import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth.tsx';
import { Navigate } from 'react-router-dom';
import './LoginPage.css';

const LoginPage: React.FC = () => {
  const { login, isAuthenticated, isLoading } = useAuth();
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
  });
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Redirigir si ya está autenticado
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);

    try {
      const result = await login(credentials);
      
      if (!result.success) {
        setError(result.error || 'Error de autenticación');
      }
      // Si el login es exitoso, el redirect se maneja automáticamente
    } catch (err) {
      setError('Error de conexión');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCredentials({
      ...credentials,
      [e.target.name]: e.target.value,
    });
  };

  const fillDemoCredentials = (role: 'admin' | 'analyst' | 'user') => {
    const demoCredentials = {
      admin: { username: 'admin', password: 'admin123' },
      analyst: { username: 'analyst', password: 'analyst123' },
      user: { username: 'user', password: 'user123' },
    };

    setCredentials(demoCredentials[role]);
  };

  if (isLoading) {
    return (
      <div className="login-container">
        <div className="login-card">
          <div className="loading">Cargando...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1>
            <span className="logo-icon">📈</span>
            GuruInversor
          </h1>
          <p>Sistema de Predicción de Acciones con IA</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="username">Usuario</label>
            <input
              type="text"
              id="username"
              name="username"
              value={credentials.username}
              onChange={handleChange}
              required
              placeholder="Ingresa tu usuario"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Contraseña</label>
            <input
              type="password"
              id="password"
              name="password"
              value={credentials.password}
              onChange={handleChange}
              required
              placeholder="Ingresa tu contraseña"
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button
            type="submit"
            className="login-btn"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Iniciando sesión...' : 'Iniciar Sesión'}
          </button>
        </form>

        <div className="demo-section">
          <p className="demo-title">Usuarios de demostración:</p>
          <div className="demo-buttons">
            <button
              type="button"
              onClick={() => fillDemoCredentials('admin')}
              className="demo-btn admin"
            >
              Admin
            </button>
            <button
              type="button"
              onClick={() => fillDemoCredentials('analyst')}
              className="demo-btn analyst"
            >
              Analyst
            </button>
            <button
              type="button"
              onClick={() => fillDemoCredentials('user')}
              className="demo-btn user"
            >
              User
            </button>
          </div>
        </div>

        <div className="login-footer">
          <p>Powered by LSTM Neural Networks & FastAPI</p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
