import { useState, useEffect, useContext, createContext, type ReactNode } from 'react';
import type { User, LoginRequest } from '../types/api';
import ApiService from '../services/api';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginRequest) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!user;

  // Verificar token al cargar la aplicación
  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('authToken');
      if (token) {
        await refreshUser();
      } else {
        setIsLoading(false);
      }
    };

    initAuth();
  }, []);

  const refreshUser = async () => {
    try {
      setIsLoading(true);
      const response = await ApiService.getCurrentUser();
      
      if (response.data) {
        setUser(response.data);
      } else {
        // Token inválido o expirado
        localStorage.removeItem('authToken');
        setUser(null);
      }
    } catch (error) {
      console.error('Error al obtener usuario:', error);
      localStorage.removeItem('authToken');
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (credentials: LoginRequest) => {
    try {
      setIsLoading(true);
      const response = await ApiService.login(credentials);
      
      if (response.data) {
        // Login exitoso, obtener datos del usuario
        await refreshUser();
        return { success: true };
      } else {
        return { 
          success: false, 
          error: response.error || 'Error de autenticación' 
        };
      }
    } catch (error) {
      return { 
        success: false, 
        error: 'Error de conexión' 
      };
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      await ApiService.logout();
    } catch (error) {
      console.error('Error durante logout:', error);
    } finally {
      setUser(null);
    }
  };

  const value: AuthContextType = {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth debe ser usado dentro de un AuthProvider');
  }
  return context;
};

export default useAuth;
