import axios, { type AxiosResponse } from 'axios';
import type {
  ApiResponse,
  LoginRequest,
  LoginResponse,
  User,
  StockInfo,
  StockDataResponse,
  StockListResponse,
  AddStockRequest,
  PredictionResponse,
  SystemMetrics,
  ModelMetrics,
  HealthCheck
} from '../types/api';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Configuración base de Axios
const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor para añadir token de autenticación
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Interceptor para manejar respuestas
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expirado o inválido
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Funciones helper para manejar respuestas
const handleResponse = <T>(response: AxiosResponse<T>): ApiResponse<T> => {
  return {
    data: response.data,
    status: response.status,
  };
};

const handleError = (error: any): ApiResponse<any> => {
  return {
    error: error.response?.data?.detail || error.message || 'Error desconocido',
    status: error.response?.status || 500,
  };
};

// Servicios de API
export class ApiService {
  // Autenticación
  static async login(credentials: LoginRequest): Promise<ApiResponse<LoginResponse>> {
    try {
      const response = await api.post<LoginResponse>('/auth/login', credentials);
      if (response.data.access_token) {
        localStorage.setItem('authToken', response.data.access_token);
      }
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async getCurrentUser(): Promise<ApiResponse<User>> {
    try {
      const response = await api.get<User>('/auth/me');
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async logout(): Promise<void> {
    try {
      await api.post('/auth/logout');
    } catch (error) {
      console.error('Error durante logout:', error);
    } finally {
      localStorage.removeItem('authToken');
    }
  }

  // Health Check y métricas
  static async getHealthCheck(): Promise<ApiResponse<HealthCheck>> {
    try {
      const response = await api.get<HealthCheck>('/metrics/health-check');
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async getSystemMetrics(): Promise<ApiResponse<SystemMetrics>> {
    try {
      const response = await api.get<SystemMetrics>('/metrics/system');
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async getModelMetrics(): Promise<ApiResponse<ModelMetrics>> {
    try {
      const response = await api.get<ModelMetrics>('/metrics/models');
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }
  // Stocks y predicciones
  static async getStocksList(): Promise<ApiResponse<StockListResponse>> {
    try {
      const response = await api.get<StockListResponse>('/stocks');
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async getStockInfo(ticker: string): Promise<ApiResponse<StockInfo>> {
    try {
      const response = await api.get<StockInfo>(`/stocks/${ticker}`);
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async getStockHistory(
    ticker: string, 
    days?: number, 
    start_date?: string, 
    end_date?: string
  ): Promise<ApiResponse<StockDataResponse>> {
    try {
      const params = new URLSearchParams();
      if (days) params.append('days', days.toString());
      if (start_date) params.append('start_date', start_date);
      if (end_date) params.append('end_date', end_date);

      const url = `/stocks/${ticker}/history${params.toString() ? '?' + params.toString() : ''}`;
      const response = await api.get<StockDataResponse>(url);
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async addStock(request: AddStockRequest): Promise<ApiResponse<any>> {
    try {
      const response = await api.post('/stocks', request);
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async removeStock(ticker: string): Promise<ApiResponse<any>> {
    try {
      const response = await api.delete(`/stocks/${ticker}`);
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async updateStockData(ticker: string): Promise<ApiResponse<any>> {
    try {
      const response = await api.post(`/stocks/${ticker}/update`);
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async getPrediction(
    ticker: string, 
    days_ahead: number = 1, 
    confidence_level: number = 0.8
  ): Promise<ApiResponse<PredictionResponse>> {
    try {
      const params = new URLSearchParams({
        days_ahead: days_ahead.toString(),
        confidence_level: confidence_level.toString()
      });

      const response = await api.get<PredictionResponse>(`/predictions/${ticker}?${params.toString()}`);
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }

  static async trainModel(ticker: string): Promise<ApiResponse<any>> {
    try {
      const response = await api.post(`/predictions/${ticker}/train`);
      return handleResponse(response);
    } catch (error) {
      return handleError(error);
    }
  }
}

export default ApiService;
