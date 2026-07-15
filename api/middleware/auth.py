"""
认证中间件模块

提供JWT和API Key认证支持。

主要组件:
    - AuthMiddleware: 主认证中间件
    - JWTAuthBackend: JWT认证后端
    - APIKeyAuthBackend: API Key认证后端

使用示例:
    >>> from agi_unified_framework.api.middleware import AuthMiddleware
    >>> app.add_middleware(AuthMiddleware, jwt_secret="secret", api_key_header="X-API-Key")
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class User:
    """用户数据类"""
    id: str
    username: str
    email: Optional[str] = None
    permissions: List[str] = None
    is_active: bool = True
    is_superuser: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}
    
    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        if self.is_superuser:
            return True
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "metadata": self.metadata,
        }


class AuthBackend(ABC):
    """认证后端抽象基类"""
    
    @abstractmethod
    async def authenticate(self, request: Request) -> Optional[User]:
        """
        认证请求
        
        Args:
            request: FastAPI请求对象
            
        Returns:
            认证成功返回User对象，失败返回None
        """
        pass
    
    @abstractmethod
    def get_scheme(self) -> str:
        """获取认证方案名称"""
        pass


class JWTAuthBackend(AuthBackend):
    """
    JWT认证后端
    
    使用JWT Token进行认证。
    
    Attributes:
        secret_key: JWT密钥
        algorithm: 加密算法
        token_header: Token请求头名称
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        token_header: str = "Authorization",
        token_prefix: str = "Bearer",
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_header = token_header
        self.token_prefix = token_prefix
        
        # 尝试导入PyJWT
        try:
            import jwt
            self._jwt = jwt
        except ImportError:
            logger.warning("PyJWT not installed, JWT authentication will not work")
            self._jwt = None
    
    def get_scheme(self) -> str:
        return "bearer"
    
    async def authenticate(self, request: Request) -> Optional[User]:
        """JWT认证"""
        if self._jwt is None:
            return None
        
        # 获取Authorization头
        auth_header = request.headers.get(self.token_header)
        if not auth_header:
            return None
        
        # 解析Token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != self.token_prefix.lower():
            return None
        
        token = parts[1]
        
        try:
            # 验证Token
            payload = self._jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
            
            # 检查过期时间
            exp = payload.get("exp")
            if exp and exp < time.time():
                logger.debug("JWT token expired")
                return None
            
            # 创建用户对象
            return User(
                id=payload.get("sub", ""),
                username=payload.get("username", ""),
                email=payload.get("email"),
                permissions=payload.get("permissions", []),
                is_active=payload.get("is_active", True),
                is_superuser=payload.get("is_superuser", False),
                metadata=payload.get("metadata", {}),
            )
        
        except self._jwt.ExpiredSignatureError:
            logger.debug("JWT token expired")
            return None
        except self._jwt.InvalidTokenError as e:
            logger.debug(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT authentication error: {e}")
            return None
    
    def create_token(
        self,
        user_id: str,
        username: str,
        expires_in: int = 3600,
        **claims: Any,
    ) -> str:
        """
        创建JWT Token
        
        Args:
            user_id: 用户ID
            username: 用户名
            expires_in: 过期时间（秒）
            **claims: 额外声明
            
        Returns:
            JWT Token字符串
        """
        if self._jwt is None:
            raise RuntimeError("PyJWT not installed")
        
        now = time.time()
        payload = {
            "sub": user_id,
            "username": username,
            "iat": now,
            "exp": now + expires_in,
            **claims,
        }
        
        return self._jwt.encode(payload, self.secret_key, algorithm=self.algorithm)


class APIKeyAuthBackend(AuthBackend):
    """
    API Key认证后端
    
    使用API Key进行认证。
    
    Attributes:
        api_keys: 有效的API Key字典 {key: user}
        header_name: API Key请求头名称
        query_param: API Key查询参数名称
    """
    
    def __init__(
        self,
        api_keys: Optional[Dict[str, User]] = None,
        header_name: str = "X-API-Key",
        query_param: str = "api_key",
    ):
        self.api_keys = api_keys or {}
        self.header_name = header_name
        self.query_param = query_param
    
    def get_scheme(self) -> str:
        return "apiKey"
    
    async def authenticate(self, request: Request) -> Optional[User]:
        """API Key认证"""
        # 从Header获取
        api_key = request.headers.get(self.header_name)
        
        # 从查询参数获取
        if not api_key:
            api_key = request.query_params.get(self.query_param)
        
        if not api_key:
            return None
        
        # 验证API Key
        user = self.api_keys.get(api_key)
        if user and user.is_active:
            return user
        
        return None
    
    def add_api_key(self, api_key: str, user: User) -> None:
        """添加API Key"""
        self.api_keys[api_key] = user
    
    def remove_api_key(self, api_key: str) -> bool:
        """移除API Key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False
    
    def generate_api_key(self, prefix: str = "agi") -> str:
        """生成新的API Key"""
        import secrets
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"


class AuthMiddleware:
    """
    认证中间件
    
    提供统一的认证处理，支持多种认证后端。
    
    Attributes:
        jwt_backend: JWT认证后端
        api_key_backend: API Key认证后端
        exclude_paths: 排除认证的路径列表
        public_paths: 公开访问的路径列表
    
    Example:
        >>> jwt_backend = JWTAuthBackend(secret_key="secret")
        >>> api_key_backend = APIKeyAuthBackend()
        >>> middleware = AuthMiddleware(jwt_backend, api_key_backend)
    """
    
    def __init__(
        self,
        app=None,  # FastAPI 自动传入
        jwt_backend: Optional[JWTAuthBackend] = None,
        api_key_backend: Optional[APIKeyAuthBackend] = None,
        exclude_paths: Optional[List[str]] = None,
        public_paths: Optional[List[str]] = None,
        auto_error: bool = False,
    ):
        self.app = app  # 存储ASGI应用
        self.jwt_backend = jwt_backend
        self.api_key_backend = api_key_backend
        self.exclude_paths = set(exclude_paths or ["/docs", "/redoc", "/openapi.json", "/health", "/api/v1/health"])
        self.public_paths = set(public_paths or ["/", "/api"])
        self.auto_error = auto_error
        self._backends: List[AuthBackend] = []
        
        if jwt_backend:
            self._backends.append(jwt_backend)
        if api_key_backend:
            self._backends.append(api_key_backend)
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        中间件调用
        
        Args:
            request: FastAPI请求
            call_next: 下一个中间件/处理器
            
        Returns:
            Response对象
        """
        path = request.url.path
        
        # 检查是否需要跳过认证
        if self._should_skip_auth(path):
            return await call_next(request)
        
        # 尝试认证
        user = await self._authenticate(request)
        
        # 将用户信息附加到请求状态
        request.state.user = user
        request.state.is_authenticated = user is not None
        
        # 继续处理请求
        response = await call_next(request)
        
        return response
    
    def _should_skip_auth(self, path: str) -> bool:
        """检查是否应该跳过认证"""
        # 检查排除路径
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return True
        
        # 检查公开路径
        if path in self.public_paths:
            return True
        
        return False
    
    async def _authenticate(self, request: Request) -> Optional[User]:
        """执行认证"""
        for backend in self._backends:
            try:
                user = await backend.authenticate(request)
                if user:
                    logger.debug(f"Authenticated via {backend.get_scheme()}: {user.username}")
                    return user
            except Exception as e:
                logger.error(f"Authentication error with {backend.get_scheme()}: {e}")
        
        return None


def create_demo_users() -> Dict[str, User]:
    """创建演示用户"""
    return {
        "admin": User(
            id="user-001",
            username="admin",
            email="admin@example.com",
            permissions=["*"],  # 所有权限
            is_active=True,
            is_superuser=True,
        ),
        "user": User(
            id="user-002",
            username="user",
            email="user@example.com",
            permissions=[
                "personality:read",
                "channel:read",
                "message:read",
                "message:create",
            ],
            is_active=True,
            is_superuser=False,
        ),
    }


# 全局用户存储（实际应用应使用数据库）
_users_db: Dict[str, User] = create_demo_users()
_api_keys_db: Dict[str, str] = {
    "agi_demo_key_12345": "admin",
    "agi_demo_key_67890": "user",
}


def get_user_by_username(username: str) -> Optional[User]:
    """通过用户名获取用户"""
    return _users_db.get(username)


def get_user_by_api_key(api_key: str) -> Optional[User]:
    """通过API Key获取用户"""
    username = _api_keys_db.get(api_key)
    if username:
        return _users_db.get(username)
    return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码（简化实现）"""
    # 实际应用中应使用bcrypt等安全哈希
    expected_hash = hashlib.sha256(plain_password.encode()).hexdigest()
    return hmac.compare_digest(expected_hash, hashed_password)


def hash_password(password: str) -> str:
    """哈希密码（简化实现）"""
    return hashlib.sha256(password.encode()).hexdigest()


# 导出
__all__ = [
    "User",
    "AuthBackend",
    "JWTAuthBackend",
    "APIKeyAuthBackend",
    "AuthMiddleware",
    "get_user_by_username",
    "get_user_by_api_key",
    "verify_password",
    "hash_password",
]
