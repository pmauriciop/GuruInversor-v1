#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router de Autenticación - API GuruInversor

Endpoints para autenticación y autorización básica.
"""

import os
import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Crear router
router = APIRouter()

# Configuración de seguridad
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "desarrollo-secret-key-no-usar-en-produccion")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Security scheme
security = HTTPBearer()

# Usuarios demo (en producción sería una base de datos)
DEMO_USERS = {
    "admin": {
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "permissions": ["read", "write", "admin"]
    },
    "user": {
        "username": "user",
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user",
        "permissions": ["read"]
    },
    "analyst": {
        "username": "analyst",
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "role": "analyst",
        "permissions": ["read", "write"]
    }
}

# Modelos de datos
class LoginRequest(BaseModel):
    username: str = Field(..., description="Nombre de usuario")
    password: str = Field(..., description="Contraseña")

class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Token de acceso")
    token_type: str = Field(default="bearer", description="Tipo de token")
    expires_in: int = Field(..., description="Tiempo de expiración en segundos")
    user_info: Dict[str, Any] = Field(..., description="Información del usuario")

class UserInfo(BaseModel):
    username: str
    role: str
    permissions: list
    created_at: str

class APIKeyRequest(BaseModel):
    name: str = Field(..., description="Nombre descriptivo para la API key")
    permissions: list = Field(default=["read"], description="Permisos para la API key")
    expires_days: int = Field(default=30, description="Días hasta expiración")

class APIKeyResponse(BaseModel):
    api_key: str = Field(..., description="API key generada")
    name: str
    permissions: list
    expires_at: str
    created_at: str

# Funciones de utilidad
def hash_password(password: str) -> str:
    """Hash de password usando SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verificar password contra hash"""
    return hash_password(password) == password_hash

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crear JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificar JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(token_data: dict = Depends(verify_token)):
    """Obtener usuario actual desde token"""
    username = token_data.get("sub")
    if username not in DEMO_USERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado"
        )
    return DEMO_USERS[username]

def require_permission(permission: str):
    """Decorator para requerir permisos específicos"""
    def permission_checker(current_user: dict = Depends(get_current_user)):
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permiso requerido: {permission}"
            )
        return current_user
    return permission_checker

# Endpoints
@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Autenticar usuario y obtener token de acceso.
    
    Args:
        request: Credenciales de login
        
    Returns:
        TokenResponse: Token de acceso y información del usuario
    """
    try:
        # Verificar credenciales
        user = DEMO_USERS.get(request.username)
        if not user or not verify_password(request.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales incorrectas",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Crear token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info={
                "username": user["username"],
                "role": user["role"],
                "permissions": user["permissions"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en autenticación: {str(e)}"
        )

@router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Obtener información del usuario actual.
    
    Returns:
        UserInfo: Información del usuario autenticado
    """
    return UserInfo(
        username=current_user["username"],
        role=current_user["role"],
        permissions=current_user["permissions"],
        created_at=datetime.now().isoformat()
    )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """
    Renovar token de acceso.
    
    Returns:
        TokenResponse: Nuevo token de acceso
    """
    try:
        # Crear nuevo token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": current_user["username"], "role": current_user["role"]},
            expires_delta=access_token_expires
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info={
                "username": current_user["username"],
                "role": current_user["role"],
                "permissions": current_user["permissions"]
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error renovando token: {str(e)}"
        )

@router.post("/api-key", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    current_user: dict = Depends(require_permission("admin"))
):
    """
    Crear una nueva API key (solo administradores).
    
    Args:
        request: Configuración de la API key
        
    Returns:
        APIKeyResponse: Información de la API key creada
    """
    try:
        # Generar API key única
        timestamp = str(int(datetime.now().timestamp()))
        api_key_data = f"{request.name}:{current_user['username']}:{timestamp}"
        api_key = f"giv_{hashlib.sha256(api_key_data.encode()).hexdigest()[:32]}"
        
        # Calcular fecha de expiración
        expires_at = datetime.now() + timedelta(days=request.expires_days)
        
        # En producción, esto se guardaría en base de datos
        # Por ahora solo retornamos la información
        
        return APIKeyResponse(
            api_key=api_key,
            name=request.name,
            permissions=request.permissions,
            expires_at=expires_at.isoformat(),
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando API key: {str(e)}"
        )

@router.get("/validate", response_model=dict)
async def validate_token(current_user: dict = Depends(get_current_user)):
    """
    Validar token actual.
    
    Returns:
        dict: Estado de validación del token
    """
    return {
        "valid": True,
        "user": current_user["username"],
        "role": current_user["role"],
        "permissions": current_user["permissions"],
        "validated_at": datetime.now().isoformat()
    }

@router.post("/logout", response_model=dict)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Cerrar sesión (invalidar token).
    
    Note: En esta implementación básica, los tokens JWT no se pueden invalidar
    hasta su expiración natural. En producción se usaría una blacklist.
    
    Returns:
        dict: Confirmación de logout
    """
    return {
        "message": "Sesión cerrada exitosamente",
        "user": current_user["username"],
        "logged_out_at": datetime.now().isoformat(),
        "note": "Token expirará naturalmente en el tiempo configurado"
    }

@router.get("/demo-users", response_model=dict)
async def get_demo_users():
    """
    Obtener lista de usuarios demo disponibles (solo para desarrollo).
    
    Returns:
        dict: Lista de usuarios demo y sus credenciales
    """
    return {
        "demo_users": [
            {
                "username": "admin",
                "password": "admin123",
                "role": "admin",
                "description": "Administrador con todos los permisos"
            },
            {
                "username": "analyst",
                "password": "analyst123",
                "role": "analyst",
                "description": "Analista con permisos de lectura y escritura"
            },
            {
                "username": "user",
                "password": "user123",
                "role": "user",
                "description": "Usuario básico con permisos de solo lectura"
            }
        ],
        "note": "Estas credenciales son solo para desarrollo y testing",
        "jwt_info": {
            "expire_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
            "algorithm": ALGORITHM
        }
    }
