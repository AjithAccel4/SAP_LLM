"""
Authentication and Authorization for SAP_LLM API

Implements API key-based authentication with role-based access control.
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)

# Security settings
API_KEY_HEADER = "X-API-Key"
# Load from environment variable or generate a new one (not recommended for production)
SECRET_KEY = os.getenv("API_SECRET_KEY")
if not SECRET_KEY:
    logger.warning("API_SECRET_KEY not set in environment. Using generated key (not suitable for production!)")
    SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    role: str = "user"  # user, admin, service


class TokenData(BaseModel):
    """Token payload."""
    username: Optional[str] = None
    role: Optional[str] = None


# Mock API key database (TODO: Replace with real database)
API_KEYS = {
    "dev_key_12345": User(
        username="dev_user",
        email="dev@example.com",
        role="admin",
    ),
    "prod_key_67890": User(
        username="prod_user",
        email="prod@example.com",
        role="user",
    ),
}


# Mock user database (TODO: Replace with real database)
USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
    },
    "user": {
        "username": "user",
        "email": "user@example.com",
        "hashed_password": pwd_context.hash("user123"),
        "role": "user",
    },
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Token payload
        expires_delta: Token expiration time

    Returns:
        JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify JWT token.

    Args:
        token: JWT token string

    Returns:
        Token data

    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")

        if username is None:
            raise credentials_exception

        return TokenData(username=username, role=role)

    except JWTError:
        raise credentials_exception


def verify_api_key(api_key: str) -> Optional[User]:
    """
    Verify API key.

    Args:
        api_key: API key string

    Returns:
        User object if valid, None otherwise
    """
    # Hash API key for comparison
    # In production, store hashed API keys in database
    user = API_KEYS.get(api_key)

    if user is None:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        return None

    if user.disabled:
        logger.warning(f"Disabled user attempted access: {user.username}")
        return None

    return user


async def get_current_user_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> User:
    """
    Get current user from API key.

    Args:
        api_key: API key from header

    Returns:
        User object

    Raises:
        HTTPException: If API key is invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    user = verify_api_key(api_key)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return user


async def get_current_user_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> User:
    """
    Get current user from Bearer token.

    Args:
        credentials: Bearer token credentials

    Returns:
        User object

    Raises:
        HTTPException: If token is invalid
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = verify_token(credentials.credentials)

    # Get user from database
    user_dict = USERS_DB.get(token_data.username)

    if user_dict is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return User(**user_dict)


async def get_current_user(
    api_key_user: Optional[User] = Depends(get_current_user_api_key),
    bearer_user: Optional[User] = Depends(get_current_user_bearer),
) -> User:
    """
    Get current user from either API key or Bearer token.

    Args:
        api_key_user: User from API key
        bearer_user: User from Bearer token

    Returns:
        User object

    Raises:
        HTTPException: If no valid authentication provided
    """
    # Prefer API key over Bearer token
    if api_key_user is not None:
        return api_key_user

    if bearer_user is not None:
        return bearer_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.

    Args:
        current_user: Current user

    Returns:
        User object

    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return current_user


async def require_admin(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Require admin role.

    Args:
        current_user: Current user

    Returns:
        User object

    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return current_user


def generate_api_key() -> str:
    """
    Generate a new API key.

    Returns:
        API key string
    """
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage.

    Args:
        api_key: API key string

    Returns:
        Hashed API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()
