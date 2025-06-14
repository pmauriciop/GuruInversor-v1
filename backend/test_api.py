#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite para API GuruInversor - API-001

Suite completa de pruebas para validar todos los endpoints de la API.
Incluye pruebas de funcionalidad, manejo de errores y rendimiento.
"""

import os
import sys
import json
import time
import requests
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import concurrent.futures

# Configuración de la API
API_BASE_URL = "http://localhost:8000"
API_PREFIX = "/api"

class APITester:
    """Clase principal para realizar pruebas de la API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        self.start_time = None
        
    def log(self, message: str, level: str = "INFO"):
        """Registra mensajes con timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def test_endpoint(self, method: str, endpoint: str, expected_status: int = 200, 
                     data: dict = None, description: str = "") -> Dict[str, Any]:
        """Prueba un endpoint específico."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            elif method.upper() == "DELETE":
                response = self.session.delete(url)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
                
            response_time = time.time() - start_time
            
            # Verificar código de estado
            status_ok = response.status_code == expected_status
            
            # Intentar parsear JSON
            try:
                json_data = response.json()
            except:
                json_data = None
                
            result = {
                "success": status_ok,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": round(response_time, 3),
                "json_data": json_data,
                "description": description,
                "endpoint": endpoint,
                "method": method.upper()
            }
            
            # Log del resultado
            status_emoji = "✅" if status_ok else "❌"
            self.log(f"{status_emoji} {method.upper()} {endpoint} - {response.status_code} ({response_time:.3f}s)")
            
            if not status_ok:
                self.log(f"   Expected {expected_status}, got {response.status_code}", "WARNING")
                
            return result
            
        except Exception as e:
            self.log(f"❌ ERROR en {method.upper()} {endpoint}: {str(e)}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "endpoint": endpoint,
                "method": method.upper(),
                "description": description
            }
    
    def test_health_endpoints(self) -> Dict[str, Any]:
        """Prueba todos los endpoints de salud."""
        self.log("🔍 Probando endpoints de salud...")
        
        tests = [
            ("GET", "/api/health", 200, None, "Health check completo"),
            ("GET", "/api/health/quick", 200, None, "Health check rápido"),
            ("GET", "/api/info", 200, None, "Información del sistema")
        ]
        
        results = []
        for method, endpoint, status, data, desc in tests:
            result = self.test_endpoint(method, endpoint, status, data, desc)
            results.append(result)
            
        return {"category": "health", "tests": results}
    
    def test_stocks_endpoints(self) -> Dict[str, Any]:
        """Prueba todos los endpoints de acciones."""
        self.log("📈 Probando endpoints de acciones...")
        
        results = []
        
        # Test 1: Listar acciones
        result = self.test_endpoint("GET", "/api/stocks", 200, None, "Listar todas las acciones")
        results.append(result)
        
        # Test 2: Añadir una nueva acción
        test_stock = {"symbol": "TESTAPI", "name": "Test API Stock"}
        result = self.test_endpoint("POST", "/api/stocks", 201, test_stock, "Añadir nueva acción")
        results.append(result)
        
        # Test 3: Obtener información de la acción añadida
        result = self.test_endpoint("GET", "/api/stocks/TESTAPI", 200, None, "Obtener acción específica")
        results.append(result)
        
        # Test 4: Intentar añadir acción duplicada (debe fallar)
        result = self.test_endpoint("POST", "/api/stocks", 400, test_stock, "Intentar duplicar acción")
        results.append(result)
        
        # Test 5: Obtener datos históricos (puede no tener datos)
        result = self.test_endpoint("GET", "/api/stocks/TESTAPI/historical", 200, None, "Datos históricos")
        results.append(result)
        
        # Test 6: Actualizar datos de la acción
        result = self.test_endpoint("POST", "/api/stocks/TESTAPI/update", 200, None, "Actualizar datos")
        results.append(result)
        
        # Test 7: Eliminar la acción de prueba
        result = self.test_endpoint("DELETE", "/api/stocks/TESTAPI", 200, None, "Eliminar acción de prueba")
        results.append(result)
        
        # Test 8: Verificar que la acción fue eliminada
        result = self.test_endpoint("GET", "/api/stocks/TESTAPI", 404, None, "Verificar eliminación")
        results.append(result)
        
        return {"category": "stocks", "tests": results}
    
    def test_predictions_endpoints(self) -> Dict[str, Any]:
        """Prueba todos los endpoints de predicciones."""
        self.log("🤖 Probando endpoints de predicciones...")
        
        results = []
        
        # Test 1: Estado del sistema de predicciones
        result = self.test_endpoint("GET", "/api/predictions/status", 200, None, "Estado del sistema ML")
        results.append(result)
        
        # Test 2: Obtener predicción para AAPL (si existe)
        result = self.test_endpoint("GET", "/api/predictions/AAPL", 200, None, "Predicción para AAPL")
        results.append(result)
        
        # Test 3: Predicción batch para múltiples símbolos
        symbols_data = {"symbols": ["AAPL", "GOOGL", "MSFT"]}
        result = self.test_endpoint("POST", "/api/predictions/batch", 200, symbols_data, "Predicciones batch")
        results.append(result)
        
        # Test 4: Iniciar entrenamiento (puede tardar)
        train_data = {"symbols": ["AAPL"], "incremental": True}
        result = self.test_endpoint("POST", "/api/predictions/train", 200, train_data, "Iniciar entrenamiento")
        results.append(result)
        
        return {"category": "predictions", "tests": results}
    
    def test_models_endpoints(self) -> Dict[str, Any]:
        """Prueba todos los endpoints de modelos."""
        self.log("🧠 Probando endpoints de modelos...")
        
        results = []
        
        # Test 1: Listar modelos disponibles
        result = self.test_endpoint("GET", "/api/models", 200, None, "Listar modelos")
        results.append(result)
        
        # Test 2: Información del sistema
        result = self.test_endpoint("GET", "/api/models/system-info", 200, None, "Información del sistema")
        results.append(result)
        
        # Test 3: Reporte del sistema
        result = self.test_endpoint("GET", "/api/models/system-report", 200, None, "Reporte del sistema")
        results.append(result)
        
        # Test 4: Estado del scheduler
        result = self.test_endpoint("GET", "/api/models/scheduler/status", 200, None, "Estado del scheduler")
        results.append(result)
        
        # Test 5: Operaciones batch
        batch_data = {"operation": "health_check", "symbols": ["AAPL", "GOOGL"]}
        result = self.test_endpoint("POST", "/api/models/batch", 200, batch_data, "Operación batch")
        results.append(result)
        
        return {"category": "models", "tests": results}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Prueba el manejo de errores."""
        self.log("⚠️ Probando manejo de errores...")
        
        results = []
        
        # Test 1: Endpoint inexistente
        result = self.test_endpoint("GET", "/api/nonexistent", 404, None, "Endpoint inexistente")
        results.append(result)
        
        # Test 2: Método no permitido
        result = self.test_endpoint("POST", "/api/health", 405, None, "Método no permitido")
        results.append(result)
        
        # Test 3: Símbolo inexistente
        result = self.test_endpoint("GET", "/api/stocks/NONEXISTENT123", 404, None, "Símbolo inexistente")
        results.append(result)
        
        # Test 4: Datos inválidos en POST
        invalid_data = {"invalid": "data"}
        result = self.test_endpoint("POST", "/api/stocks", 422, invalid_data, "Datos inválidos")
        results.append(result)
        
        return {"category": "error_handling", "tests": results}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Ejecuta pruebas de rendimiento."""
        self.log("⚡ Ejecutando pruebas de rendimiento...")
        
        # Test de carga para endpoint de salud
        num_requests = 10
        endpoint = "/api/health/quick"
        
        start_time = time.time()
        
        # Ejecutar requests concurrentes
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.test_endpoint, "GET", endpoint, 200, None, f"Load test {i+1}")
                for i in range(num_requests)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        successful_requests = sum(1 for r in results if r.get("success", False))
        
        avg_response_time = sum(r.get("response_time", 0) for r in results) / len(results)
        
        performance_summary = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "success_rate": (successful_requests / num_requests) * 100,
            "total_time": round(total_time, 3),
            "avg_response_time": round(avg_response_time, 3),
            "requests_per_second": round(num_requests / total_time, 2)
        }
        
        self.log(f"   Requests exitosos: {successful_requests}/{num_requests}")
        self.log(f"   Tiempo promedio de respuesta: {avg_response_time:.3f}s")
        self.log(f"   Requests por segundo: {performance_summary['requests_per_second']}")
        
        return {"category": "performance", "summary": performance_summary, "tests": results}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Ejecuta todas las pruebas."""
        self.start_time = time.time()
        self.log("🚀 Iniciando suite completa de pruebas de API...")
        
        # Verificar que la API esté disponible
        try:
            response = self.session.get(f"{self.base_url}/api/health/quick", timeout=10)
            if response.status_code != 200:
                raise Exception(f"API no disponible (status: {response.status_code})")
        except Exception as e:
            self.log(f"❌ Error: No se puede conectar a la API: {e}", "ERROR")
            return {"error": "API no disponible", "details": str(e)}
        
        self.log("✅ API disponible, iniciando pruebas...")
        
        # Ejecutar todas las categorías de pruebas
        all_results = {}
        
        # Pruebas funcionales
        all_results["health"] = self.test_health_endpoints()
        all_results["stocks"] = self.test_stocks_endpoints()
        all_results["predictions"] = self.test_predictions_endpoints()
        all_results["models"] = self.test_models_endpoints()
        all_results["errors"] = self.test_error_handling()
        
        # Pruebas de rendimiento
        all_results["performance"] = self.run_performance_tests()
        
        # Generar resumen
        total_time = time.time() - self.start_time
        total_tests = sum(len(category.get("tests", [])) for category in all_results.values() 
                         if isinstance(category, dict) and "tests" in category)
        successful_tests = sum(
            sum(1 for test in category.get("tests", []) if test.get("success", False))
            for category in all_results.values() 
            if isinstance(category, dict) and "tests" in category
        )
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": round((successful_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            "total_execution_time": round(total_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
        all_results["summary"] = summary
        
        # Log del resumen final
        self.log("=" * 60)
        self.log("📊 RESUMEN DE PRUEBAS")
        self.log("=" * 60)
        self.log(f"Tests ejecutados: {total_tests}")
        self.log(f"Tests exitosos: {successful_tests}")
        self.log(f"Tasa de éxito: {summary['success_rate']}%")
        self.log(f"Tiempo total: {summary['total_execution_time']}s")
        self.log("=" * 60)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Guarda los resultados en un archivo JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_test_results_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.log(f"✅ Resultados guardados en: {filename}")
        except Exception as e:
            self.log(f"❌ Error guardando resultados: {e}", "ERROR")


def main():
    """Función principal para ejecutar las pruebas."""
    print("\n" + "="*80)
    print("🔬 SUITE DE PRUEBAS API GURUINVERSOR - API-001")
    print("="*80)
    
    # Verificar que la API esté ejecutándose
    print("\n🔍 Verificando disponibilidad de la API...")
    print(f"URL base: {API_BASE_URL}")
    print("\n⚠️  NOTA: Asegúrate de que la API esté ejecutándose con:")
    print("   python run_api.py")
    print("\nPresiona Enter para continuar o Ctrl+C para cancelar...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Pruebas canceladas por el usuario")
        return
    
    # Crear y ejecutar el tester
    tester = APITester()
    results = tester.run_all_tests()
    
    # Guardar resultados
    tester.save_results(results)
    
    # Mostrar estado final
    if "error" in results:
        print(f"\n❌ Error en las pruebas: {results['error']}")
        return 1
    
    summary = results.get("summary", {})
    success_rate = summary.get("success_rate", 0)
    
    if success_rate >= 90:
        print("\n🎉 ¡PRUEBAS COMPLETADAS CON ÉXITO!")
        print("✅ La API está funcionando correctamente")
    elif success_rate >= 70:
        print("\n⚠️ PRUEBAS COMPLETADAS CON ADVERTENCIAS")
        print("🔧 Algunas funcionalidades pueden necesitar atención")
    else:
        print("\n❌ PRUEBAS FALLARON")
        print("🚨 La API tiene problemas significativos")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
