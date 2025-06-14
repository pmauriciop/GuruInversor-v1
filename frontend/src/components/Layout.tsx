import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth.tsx';
import './Layout.css';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { user, logout, isAuthenticated } = useAuth();
  const location = useLocation();

  const handleLogout = async () => {
    await logout();
  };

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <Link to="/dashboard" className="logo">
            <span className="logo-icon">📈</span>
            GuruInversor
          </Link>
          
          {isAuthenticated && (
            <nav className="main-nav">
              <Link 
                to="/dashboard" 
                className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
              >
                📊 Dashboard
              </Link>              <Link 
                to="/tickers" 
                className={`nav-link ${isActive('/tickers') ? 'active' : ''}`}
              >
                🔍 Tickers
              </Link>
              <Link 
                to="/charts" 
                className={`nav-link ${isActive('/charts') ? 'active' : ''}`}
              >
                📈 Gráficos
              </Link>
            </nav>
          )}
          
          {isAuthenticated && user && (
            <div className="user-section">
              <div className="user-info">
                <span className="username">{user.username}</span>
                <span className="user-role">({user.role})</span>
              </div>
              <button 
                onClick={handleLogout}
                className="logout-btn"
              >
                Cerrar Sesión
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="main-content">
        {children}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p>&copy; 2025 GuruInversor - Sistema de Predicción de Acciones con IA</p>
          <p>Powered by LSTM Neural Networks</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
