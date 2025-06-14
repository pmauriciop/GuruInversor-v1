#!/usr/bin/env python3
"""
Script de preparación para deployment de GuruInversor v1.0

Este script prepara el proyecto para ser desplegado en producción.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, cwd=None):
    """Ejecuta un comando y retorna True si es exitoso"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {command}")
            return True
        else:
            print(f"   ❌ {command} - Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ {command} - Exception: {str(e)}")
        return False

def check_prerequisites():
    """Verifica que todos los prerequisitos estén cumplidos"""
    print("🔍 Verificando prerequisitos...")
    
    checks = []
    
    # Verificar que el backend funcione
    print("   Verificando backend local...")
    backend_running = run_command("curl -s http://localhost:8000/api/health")
    checks.append(("Backend funcionando", backend_running))
    
    # Verificar que el frontend funcione
    print("   Verificando frontend local...")
    frontend_running = run_command("curl -s http://localhost:3001")
    checks.append(("Frontend funcionando", frontend_running))
    
    # Verificar base de datos con datos
    db_path = Path("backend/data/guru_inversor.db")
    db_exists = db_path.exists()
    checks.append(("Base de datos existe", db_exists))
    
    # Verificar archivos de configuración
    config_files = [
        "Dockerfile",
        "railway.json", 
        "frontend/vercel.json",
        "backend/database/config.py",
        ".env.production"
    ]
    
    for file_path in config_files:
        exists = Path(file_path).exists()
        checks.append((f"Archivo {file_path}", exists))
    
    # Mostrar resultados
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def prepare_backend():
    """Prepara el backend para deployment"""
    print("\n🐍 Preparando backend...")
    
    # Instalar dependencias
    print("   Instalando dependencias...")
    success = run_command("pip install -r requirements.txt", cwd="backend")
    
    # Ejecutar tests
    print("   Ejecutando tests...")
    test_success = run_command("python -m pytest tests/ -v", cwd="backend")
    
    # Verificar que la configuración de DB funcione
    print("   Verificando configuración de base de datos...")
    db_success = run_command("python -c 'from database.config import get_database_url; print(get_database_url())'", cwd="backend")
    
    return success and test_success and db_success

def prepare_frontend():
    """Prepara el frontend para deployment"""
    print("\n⚛️  Preparando frontend...")
    
    # Instalar dependencias
    print("   Instalando dependencias...")
    install_success = run_command("npm install", cwd="frontend")
    
    # Ejecutar build de prueba
    print("   Ejecutando build de prueba...")
    build_success = run_command("npm run build", cwd="frontend")
    
    # Limpiar build
    if build_success:
        print("   Limpiando build temporal...")
        run_command("rm -rf dist", cwd="frontend")
    
    return install_success and build_success

def create_deployment_checklist():
    """Crea un checklist de deployment"""
    checklist = {
        "pre_deployment": {
            "codigo_funcionando_local": True,
            "tests_pasando": True,
            "configuracion_creada": True,
            "variables_entorno_definidas": True
        },
        "deployment_backend": {
            "repository_en_github": False,
            "railway_app_creada": False,
            "variables_entorno_configuradas": False,
            "deployment_exitoso": False,
            "base_datos_migrada": False
        },
        "deployment_frontend": {
            "vercel_app_creada": False,
            "variables_entorno_configuradas": False,
            "deployment_exitoso": False,
            "conexion_backend_ok": False
        },
        "post_deployment": {
            "health_check_ok": False,
            "endpoints_funcionando": False,
            "frontend_cargando": False,
            "datos_sincronizados": False
        }
    }
    
    with open("deployment_checklist.json", "w") as f:
        json.dump(checklist, f, indent=2)
    
    print("\n📋 Checklist de deployment creado: deployment_checklist.json")

def main():
    """Función principal"""
    print("🚀 Preparación para Deployment GuruInversor v1.0")
    print("=" * 60)
    
    # Verificar prerequisitos
    if not check_prerequisites():
        print("\n❌ Los prerequisitos no están completos. Por favor corrige los errores antes de continuar.")
        return False
    
    # Preparar backend
    if not prepare_backend():
        print("\n❌ Error preparando backend")
        return False
    
    # Preparar frontend
    if not prepare_frontend():
        print("\n❌ Error preparando frontend")
        return False
    
    # Crear checklist
    create_deployment_checklist()
    
    print("\n✅ Preparación completada exitosamente!")
    print("\n🎯 Próximos pasos:")
    print("   1. Subir código a GitHub")
    print("   2. Crear app en Railway")
    print("   3. Configurar variables de entorno")
    print("   4. Hacer deploy del backend")
    print("   5. Crear app en Vercel")
    print("   6. Hacer deploy del frontend")
    print("   7. Ejecutar migración de datos")
    print("   8. Verificar funcionamiento completo")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
