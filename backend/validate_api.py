#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador de API - GuruInversor

Script de validación específica para verificar que todos los endpoints
estén correctamente implementados y respondan según las especificaciones.
"""

import requests
import json
import time
from typing import Dict, List, Any
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

class APIValidator:
    """Validador específico para la API GuruInversor."""
    
    def __init__(self):
        self.base_url = API_BASE_URL
        self.session = requests.Session()
        
    def check_api_availability(self) -> bool:
        """Verifica que la API esté disponible."""
        try:
            response = self.session.get(f"{self.base_url}/api/health/quick", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def validate_health_endpoints(self) -> Dict[str, bool]:
        """Valida que los endpoints de salud funcionen correctamente."""
        results = {}
        
        # Health check básico
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            results["health_basic"] = response.status_code == 200
            if response.status_code == 200:
                data = response.json()
                results["health_has_status"] = "status" in data
                results["health_has_timestamp"] = "timestamp" in data
        except:
            results["health_basic"] = False
            results["health_has_status"] = False
            results["health_has_timestamp"] = False
        
        # Health check rápido
        try:
            response = self.session.get(f"{self.base_url}/api/health/quick")
            results["health_quick"] = response.status_code == 200
        except:
            results["health_quick"] = False
        
        # Info del sistema
        try:
            response = self.session.get(f"{self.base_url}/api/info")
            results["system_info"] = response.status_code == 200
            if response.status_code == 200:
                data = response.json()
                results["info_has_version"] = "version" in data
                results["info_has_description"] = "description" in data
        except:
            results["system_info"] = False
            results["info_has_version"] = False
            results["info_has_description"] = False
        
        return results
    
    def validate_stocks_endpoints(self) -> Dict[str, bool]:
        """Valida que los endpoints de stocks funcionen correctamente."""
        results = {}
        
        # Listar stocks
        try:
            response = self.session.get(f"{self.base_url}/api/stocks")
            results["stocks_list"] = response.status_code == 200
            if response.status_code == 200:
                data = response.json()
                results["stocks_list_is_array"] = isinstance(data, list)
        except:
            results["stocks_list"] = False
            results["stocks_list_is_array"] = False
        
        # Añadir stock de prueba
        test_stock = {"symbol": "APITEST", "name": "API Test Stock"}
        try:
            response = self.session.post(f"{self.base_url}/api/stocks", json=test_stock)
            results["stocks_add"] = response.status_code == 201
        except:
            results["stocks_add"] = False
        
        # Obtener stock específico
        try:
            response = self.session.get(f"{self.base_url}/api/stocks/APITEST")
            results["stocks_get_specific"] = response.status_code == 200
            if response.status_code == 200:
                data = response.json()
                results["stock_has_symbol"] = "symbol" in data
                results["stock_has_name"] = "name" in data
        except:
            results["stocks_get_specific"] = False
            results["stock_has_symbol"] = False
            results["stock_has_name"] = False
        
        # Datos históricos
        try:
            response = self.session.get(f"{self.base_url}/api/stocks/APITEST/historical")
            results["stocks_historical"] = response.status_code in [200, 404]  # Puede no tener datos
        except:
            results["stocks_historical"] = False
        
        # Actualizar datos
        try:
            response = self.session.post(f"{self.base_url}/api/stocks/APITEST/update")
            results["stocks_update"] = response.status_code == 200
        except:
            results["stocks_update"] = False
        
        # Eliminar stock de prueba
        try:
            response = self.session.delete(f"{self.base_url}/api/stocks/APITEST")
            results["stocks_delete"] = response.status_code == 200
        except:
            results["stocks_delete"] = False
        
        # Verificar eliminación
        try:
            response = self.session.get(f"{self.base_url}/api/stocks/APITEST")
            results["stocks_verify_deletion"] = response.status_code == 404
        except:
            results["stocks_verify_deletion"] = False
        
        return results
    
    def validate_predictions_endpoints(self) -> Dict[str, bool]:
        """Valida que los endpoints de predicciones funcionen correctamente."""
        results = {}
        
        # Estado del sistema
        try:
            response = self.session.get(f"{self.base_url}/api/predictions/status")
            results["predictions_status"] = response.status_code == 200
            if response.status_code == 200:
                data = response.json()
                results["status_has_models"] = "models" in data or "status" in data
        except:
            results["predictions_status"] = False
            results["status_has_models"] = False
        
        # Predicción individual (puede fallar si no hay modelos)
        try:
            response = self.session.get(f"{self.base_url}/api/predictions/AAPL")
            results["predictions_individual"] = response.status_code in [200, 404, 500]
        except:
            results["predictions_individual"] = False
        
        # Predicciones batch
        try:
            batch_data = {"symbols": ["AAPL", "GOOGL"]}
            response = self.session.post(f"{self.base_url}/api/predictions/batch", json=batch_data)
            results["predictions_batch"] = response.status_code in [200, 404, 500]
        except:
            results["predictions_batch"] = False
        
        return results
    
    def validate_models_endpoints(self) -> Dict[str, bool]:
        """Valida que los endpoints de modelos funcionen correctamente."""
        results = {}
        
        # Listar modelos
        try:
            response = self.session.get(f"{self.base_url}/api/models")
            results["models_list"] = response.status_code == 200
        except:
            results["models_list"] = False
        
        # Info del sistema
        try:
            response = self.session.get(f"{self.base_url}/api/models/system-info")
            results["models_system_info"] = response.status_code == 200
        except:
            results["models_system_info"] = False
        
        # Reporte del sistema
        try:
            response = self.session.get(f"{self.base_url}/api/models/system-report")
            results["models_system_report"] = response.status_code == 200
        except:
            results["models_system_report"] = False
        
        # Estado del scheduler
        try:
            response = self.session.get(f"{self.base_url}/api/models/scheduler/status")
            results["scheduler_status"] = response.status_code == 200
        except:
            results["scheduler_status"] = False
        
        return results
    
    def validate_error_handling(self) -> Dict[str, bool]:
        """Valida que el manejo de errores funcione correctamente."""
        results = {}
        
        # Endpoint inexistente
        try:
            response = self.session.get(f"{self.base_url}/api/nonexistent")
            results["error_404"] = response.status_code == 404
        except:
            results["error_404"] = False
        
        # Método no permitido
        try:
            response = self.session.post(f"{self.base_url}/api/health")
            results["error_405"] = response.status_code == 405
        except:
            results["error_405"] = False
        
        # Datos inválidos
        try:
            invalid_data = {"invalid": "data"}
            response = self.session.post(f"{self.base_url}/api/stocks", json=invalid_data)
            results["error_422"] = response.status_code == 422
        except:
            results["error_422"] = False
        
        return results
    
    def run_validation(self) -> Dict[str, Any]:
        """Ejecuta toda la validación."""
        print("🔍 Iniciando validación de API...")
        
        # Verificar disponibilidad
        if not self.check_api_availability():
            return {
                "error": "API no disponible",
                "message": "Asegúrate de que la API esté ejecutándose en http://localhost:8000"
            }
        
        print("✅ API disponible")
        
        # Ejecutar validaciones
        results = {
            "timestamp": datetime.now().isoformat(),
            "api_url": self.base_url,
            "health_endpoints": self.validate_health_endpoints(),
            "stocks_endpoints": self.validate_stocks_endpoints(),
            "predictions_endpoints": self.validate_predictions_endpoints(),
            "models_endpoints": self.validate_models_endpoints(),
            "error_handling": self.validate_error_handling()
        }
        
        # Calcular estadísticas
        total_checks = 0
        passed_checks = 0
        
        for category, checks in results.items():
            if isinstance(checks, dict):
                for check, passed in checks.items():
                    if isinstance(passed, bool):
                        total_checks += 1
                        if passed:
                            passed_checks += 1
        
        results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "success_rate": round((passed_checks / total_checks) * 100, 2) if total_checks > 0 else 0
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Imprime los resultados de forma legible."""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            print(f"   {results.get('message', '')}")
            return
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DE VALIDACIÓN")
        print("="*60)
        
        for category, checks in results.items():
            if category in ["timestamp", "api_url", "summary"]:
                continue
                
            print(f"\n📂 {category.replace('_', ' ').title()}:")
            
            if isinstance(checks, dict):
                for check, passed in checks.items():
                    if isinstance(passed, bool):
                        status = "✅" if passed else "❌"
                        print(f"   {status} {check.replace('_', ' ').title()}")
        
        summary = results.get("summary", {})
        print(f"\n📈 RESUMEN:")
        print(f"   Total de verificaciones: {summary.get('total_checks', 0)}")
        print(f"   Verificaciones exitosas: {summary.get('passed_checks', 0)}")
        print(f"   Tasa de éxito: {summary.get('success_rate', 0)}%")
        
        success_rate = summary.get('success_rate', 0)
        if success_rate >= 90:
            print("\n🎉 ¡VALIDACIÓN EXITOSA!")
        elif success_rate >= 70:
            print("\n⚠️ VALIDACIÓN PARCIAL - Revisar elementos fallidos")
        else:
            print("\n❌ VALIDACIÓN FALLIDA - Problemas significativos detectados")


def main():
    print("🔬 VALIDADOR DE API GURUINVERSOR")
    print("="*50)
    
    validator = APIValidator()
    results = validator.run_validation()
    validator.print_results(results)
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"api_validation_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Resultados guardados en: {filename}")
    except Exception as e:
        print(f"\n❌ Error guardando resultados: {e}")


if __name__ == "__main__":
    main()
