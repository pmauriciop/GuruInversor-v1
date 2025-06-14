#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de inicio para API GuruInversor

Script para ejecutar fácilmente el servidor de desarrollo de la API.
"""

import os
import sys
from pathlib import Path

# Agregar path del backend
backend_path = Path(__file__).parent
sys.path.append(str(backend_path))

# Configurar variables de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow

def main():
    """Función principal para iniciar el servidor."""
    import uvicorn
    
    print("🚀 Iniciando GuruInversor API...")
    print("📊 Sistema de predicción de acciones con ML")
    print("=" * 50)
    
    # Configuración del servidor
    config = {
        "app": "app.main:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True
    }
    
    print(f"🌐 Servidor: http://{config['host']}:{config['port']}")
    print(f"📚 Documentación: http://{config['host']}:{config['port']}/docs")
    print(f"📖 ReDoc: http://{config['host']}:{config['port']}/redoc")
    print("=" * 50)
    print("✅ Para detener el servidor: Ctrl+C")
    print()
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando servidor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
