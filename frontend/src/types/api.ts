// Tipos para la API GuruInversor
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}

// Tipos de autenticación
export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  username: string;
  role: string;
  permissions: string[];
  created_at: string;
}

// Tipos de stock y predicciones
export interface StockInfo {
  ticker: string;
  name: string;
  sector?: string;
  market_cap?: number;
  current_price?: number;
  last_update: string;
}

export interface StockData {
  ticker: string;
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adj_close?: number;
}

export interface StockDataResponse {
  ticker: string;
  data: StockData[];
  total_records: number;
  date_range: {
    start: string;
    end: string;
  };
}

export interface StockListResponse {
  stocks: StockInfo[];
  total_count: number;
  last_update: string;
}

export interface AddStockRequest {
  ticker: string;
  name?: string;
  auto_train?: boolean;
}

export interface PredictionPoint {
  date: string;
  predicted_price: number;
  confidence_lower: number;
  confidence_upper: number;
  trend: string;
}

export interface PredictionRequest {
  ticker: string;
  days?: number;
  confidence_level?: number;
}

export interface PredictionResponse {
  ticker: string;
  current_price: number;
  predictions: PredictionPoint[];
  model_info: Record<string, any>;
  generated_at?: string;
  confidence_score: number;
}

// Tipos de métricas del sistema
export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  python_memory: number;
  uptime: string;
  timestamp: string;
}

export interface ModelMetrics {
  total_models: number;
  trained_models: string[];
  models_needing_retrain: string[];
  model_sizes: Record<string, number>;
  model_performance: Record<string, any>;
}

export interface HealthCheck {
  status: string;
  timestamp: string;
  system_metrics: {
    cpu: string;
    memory: string;
    disk: string;
  };
  model_metrics: {
    trained_models: number;
    needs_retrain: number;
  };
  api_metrics: {
    success_rate: string;
    avg_response: string;
  };
  health_issues: string[];
}
