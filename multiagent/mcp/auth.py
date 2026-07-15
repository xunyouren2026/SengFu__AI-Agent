"""MCP认证模块。

本模块实现了MCP协议的认证机制，支持API Key和OAuth2两种认证方式。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod


class AuthType(Enum):
    """认证类型枚举。"""
    API_KEY = auto()
    OAUTH2 = auto()
    BEARER = auto()
    BASIC = auto()
    NONE = auto()


class AuthError(Exception):
    """认证错误基类。"""
    pass


class InvalidCredentialsError(AuthError):
    """无效凭证错误。"""
    pass


class ExpiredTokenError(AuthError):
    """令牌过期错误。"""
    pass


class InsufficientScopeError(AuthError):
    """权限不足错误。"""
    pass


@dataclass
class AuthCredentials:
    """认证凭证基类。
    
    Attributes:
        auth_type: 认证类型
    """
    auth_type: AuthType = AuthType.NONE
    
    def to_headers(self) -> Dict[str, str]:
        """转换为HTTP头。"""
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {"auth_type": self.auth_type.name}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuthCredentials:
        """从字典创建凭证。"""
        auth_type_name = data.get("auth_type", "NONE")
        auth_type = AuthType[auth_type_name]
        return cls(auth_type=auth_type)


@dataclass
class APIKeyCredentials(AuthCredentials):
    """API Key凭证。
    
    Attributes:
        api_key: API密钥
        header_name: HTTP头名称
        prefix: 前缀
    """
    api_key: str = ""
    header_name: str = "X-API-Key"
    prefix: str = ""
    
    def __post_init__(self) -> None:
        self.auth_type = AuthType.API_KEY
    
    def to_headers(self) -> Dict[str, str]:
        """转换为HTTP头。"""
        value = f"{self.prefix}{self.api_key}" if self.prefix else self.api_key
        return {self.header_name: value}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "auth_type": self.auth_type.name,
            "api_key": self.api_key,
            "header_name": self.header_name,
            "prefix": self.prefix
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> APIKeyCredentials:
        """从字典创建凭证。"""
        return cls(
            api_key=data.get("api_key", ""),
            header_name=data.get("header_name", "X-API-Key"),
            prefix=data.get("prefix", "")
        )


@dataclass
class BearerCredentials(AuthCredentials):
    """Bearer Token凭证。
    
    Attributes:
        token: 访问令牌
    """
    token: str = ""
    
    def __post_init__(self) -> None:
        self.auth_type = AuthType.BEARER
    
    def to_headers(self) -> Dict[str, str]:
        """转换为HTTP头。"""
        return {"Authorization": f"Bearer {self.token}"}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "auth_type": self.auth_type.name,
            "token": self.token
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BearerCredentials:
        """从字典创建凭证。"""
        return cls(token=data.get("token", ""))


@dataclass
class BasicCredentials(AuthCredentials):
    """Basic认证凭证。
    
    Attributes:
        username: 用户名
        password: 密码
    """
    username: str = ""
    password: str = ""
    
    def __post_init__(self) -> None:
        self.auth_type = AuthType.BASIC
    
    def to_headers(self) -> Dict[str, str]:
        """转换为HTTP头。"""
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "auth_type": self.auth_type.name,
            "username": self.username,
            "password": self.password
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BasicCredentials:
        """从字典创建凭证。"""
        return cls(
            username=data.get("username", ""),
            password=data.get("password", "")
        )


@dataclass
class OAuth2Token:
    """OAuth2令牌。
    
    Attributes:
        access_token: 访问令牌
        token_type: 令牌类型
        expires_in: 过期时间（秒）
        refresh_token: 刷新令牌
        scope: 权限范围
        issued_at: 签发时间戳
    """
    access_token: str = ""
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: str = ""
    issued_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查令牌是否过期。"""
        if self.expires_in <= 0:
            return False  # 永不过期
        return time.time() > self.issued_at + self.expires_in
    
    def is_valid(self) -> bool:
        """检查令牌是否有效。"""
        return bool(self.access_token) and not self.is_expired()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
            "issued_at": self.issued_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OAuth2Token:
        """从字典创建令牌。"""
        return cls(
            access_token=data.get("access_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", 3600),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope", ""),
            issued_at=data.get("issued_at", time.time())
        )


@dataclass
class OAuth2Credentials(AuthCredentials):
    """OAuth2凭证。
    
    Attributes:
        client_id: 客户端ID
        client_secret: 客户端密钥
        token: OAuth2令牌
        token_endpoint: 令牌端点
        authorization_endpoint: 授权端点
        redirect_uri: 重定向URI
        scope: 权限范围
    """
    client_id: str = ""
    client_secret: str = ""
    token: Optional[OAuth2Token] = None
    token_endpoint: str = ""
    authorization_endpoint: str = ""
    redirect_uri: str = ""
    scope: str = ""
    
    def __post_init__(self) -> None:
        self.auth_type = AuthType.OAUTH2
    
    def to_headers(self) -> Dict[str, str]:
        """转换为HTTP头。"""
        if self.token and self.token.is_valid():
            return {"Authorization": f"{self.token.token_type} {self.token.access_token}"}
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "auth_type": self.auth_type.name,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "token": self.token.to_dict() if self.token else None,
            "token_endpoint": self.token_endpoint,
            "authorization_endpoint": self.authorization_endpoint,
            "redirect_uri": self.redirect_uri,
            "scope": self.scope
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OAuth2Credentials:
        """从字典创建凭证。"""
        token_data = data.get("token")
        token = OAuth2Token.from_dict(token_data) if token_data else None
        return cls(
            client_id=data.get("client_id", ""),
            client_secret=data.get("client_secret", ""),
            token=token,
            token_endpoint=data.get("token_endpoint", ""),
            authorization_endpoint=data.get("authorization_endpoint", ""),
            redirect_uri=data.get("redirect_uri", ""),
            scope=data.get("scope", "")
        )


class AuthProvider(ABC):
    """认证提供者抽象基类。"""
    
    @abstractmethod
    def authenticate(self, credentials: AuthCredentials) -> bool:
        """验证凭证。
        
        Args:
            credentials: 认证凭证
            
        Returns:
            认证是否成功
        """
        pass
    
    @abstractmethod
    def validate_request(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """验证请求头中的认证信息。
        
        Args:
            headers: HTTP请求头
            
        Returns:
            认证成功返回用户信息，失败返回None
        """
        pass


class APIKeyAuthProvider(AuthProvider):
    """API Key认证提供者。
    
    Attributes:
        valid_keys: 有效的API Key集合
        key_to_user: Key到用户信息的映射
    """
    
    def __init__(
        self,
        valid_keys: Optional[List[str]] = None,
        header_name: str = "X-API-Key",
        prefix: str = ""
    ):
        """初始化API Key认证提供者。
        
        Args:
            valid_keys: 有效的API Key列表
            header_name: HTTP头名称
            prefix: Key前缀
        """
        self.valid_keys: set[str] = set(valid_keys or [])
        self.header_name = header_name
        self.prefix = prefix
        self.key_to_user: Dict[str, Dict[str, Any]] = {}
        
        # 为每个key生成默认用户信息
        for key in self.valid_keys:
            self.key_to_user[key] = {"id": key[:8], "api_key": key}
    
    def add_key(self, key: str, user_info: Optional[Dict[str, Any]] = None) -> None:
        """添加有效的API Key。
        
        Args:
            key: API Key
            user_info: 用户信息
        """
        self.valid_keys.add(key)
        self.key_to_user[key] = user_info or {"id": key[:8], "api_key": key}
    
    def remove_key(self, key: str) -> None:
        """移除API Key。
        
        Args:
            key: API Key
        """
        self.valid_keys.discard(key)
        self.key_to_user.pop(key, None)
    
    def generate_key(self, user_info: Optional[Dict[str, Any]] = None) -> str:
        """生成新的API Key。
        
        Args:
            user_info: 用户信息
            
        Returns:
            生成的API Key
        """
        key = secrets.token_urlsafe(32)
        self.valid_keys.add(key)
        self.key_to_user[key] = user_info or {"id": key[:8], "api_key": key}
        return key
    
    def authenticate(self, credentials: AuthCredentials) -> bool:
        """验证凭证。"""
        if not isinstance(credentials, APIKeyCredentials):
            return False
        
        key = credentials.api_key
        if self.prefix and key.startswith(self.prefix):
            key = key[len(self.prefix):]
        
        return key in self.valid_keys
    
    def validate_request(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """验证请求头。"""
        auth_header = headers.get(self.header_name)
        if not auth_header:
            return None
        
        key = auth_header
        if self.prefix and key.startswith(self.prefix):
            key = key[len(self.prefix):]
        
        if key in self.valid_keys:
            return self.key_to_user.get(key)
        
        return None


class BearerAuthProvider(AuthProvider):
    """Bearer Token认证提供者。
    
    Attributes:
        valid_tokens: 有效令牌集合
        token_to_user: 令牌到用户信息的映射
    """
    
    def __init__(
        self,
        token_expiry: int = 3600,
        issuer: str = "mcp-server"
    ):
        """初始化Bearer Token认证提供者。
        
        Args:
            token_expiry: 令牌过期时间（秒）
            issuer: 签发者标识
        """
        self.token_expiry = token_expiry
        self.issuer = issuer
        self.valid_tokens: Dict[str, OAuth2Token] = {}
        self.token_to_user: Dict[str, Dict[str, Any]] = {}
        self._secret = secrets.token_bytes(32)
    
    def generate_token(
        self,
        user_id: str,
        scope: str = "",
        user_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """为用户生成令牌。
        
        Args:
            user_id: 用户ID
            scope: 权限范围
            user_info: 用户信息
            
        Returns:
            访问令牌
        """
        # 生成令牌
        timestamp = str(int(time.time()))
        payload = f"{user_id}:{timestamp}:{scope}"
        signature = hmac.new(
            self._secret,
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        token_str = f"{payload}:{signature}"
        access_token = base64.urlsafe_b64encode(token_str.encode("utf-8")).decode("ascii")
        
        # 创建令牌对象
        token = OAuth2Token(
            access_token=access_token,
            token_type="Bearer",
            expires_in=self.token_expiry,
            scope=scope,
            issued_at=time.time()
        )
        
        self.valid_tokens[access_token] = token
        self.token_to_user[access_token] = user_info or {"id": user_id, "scope": scope}
        
        return access_token
    
    def revoke_token(self, token: str) -> None:
        """撤销令牌。
        
        Args:
            token: 访问令牌
        """
        self.valid_tokens.pop(token, None)
        self.token_to_user.pop(token, None)
    
    def authenticate(self, credentials: AuthCredentials) -> bool:
        """验证凭证。"""
        if isinstance(credentials, BearerCredentials):
            token = credentials.token
        elif isinstance(credentials, OAuth2Credentials) and credentials.token:
            token = credentials.token.access_token
        else:
            return False
        
        return self._validate_token(token) is not None
    
    def _validate_token(self, token: str) -> Optional[OAuth2Token]:
        """验证令牌。"""
        token_obj = self.valid_tokens.get(token)
        if not token_obj:
            return None
        
        if token_obj.is_expired():
            self.revoke_token(token)
            return None
        
        return token_obj
    
    def validate_request(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """验证请求头。"""
        auth_header = headers.get("Authorization")
        if not auth_header:
            return None
        
        if not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header[7:]
        token_obj = self._validate_token(token)
        
        if token_obj:
            return self.token_to_user.get(token)
        
        return None


class OAuth2AuthProvider(AuthProvider):
    """OAuth2认证提供者。
    
    支持授权码流程和客户端凭证流程。
    """
    
    def __init__(
        self,
        issuer: str = "mcp-server",
        token_expiry: int = 3600,
        refresh_token_expiry: int = 86400 * 30
    ):
        """初始化OAuth2认证提供者。
        
        Args:
            issuer: 签发者标识
            token_expiry: 访问令牌过期时间（秒）
            refresh_token_expiry: 刷新令牌过期时间（秒）
        """
        self.issuer = issuer
        self.token_expiry = token_expiry
        self.refresh_token_expiry = refresh_token_expiry
        
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.auth_codes: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, OAuth2Token] = {}
        self.token_to_user: Dict[str, Dict[str, Any]] = {}
        
        self._secret = secrets.token_bytes(32)
    
    def register_client(
        self,
        client_id: str,
        client_secret: str,
        redirect_uris: List[str],
        scope: str = ""
    ) -> None:
        """注册OAuth2客户端。
        
        Args:
            client_id: 客户端ID
            client_secret: 客户端密钥
            redirect_uris: 允许的重定向URI列表
            scope: 允许的权限范围
        """
        self.clients[client_id] = {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": redirect_uris,
            "scope": scope
        }
    
    def generate_authorization_code(
        self,
        client_id: str,
        redirect_uri: str,
        user_id: str,
        scope: str = ""
    ) -> str:
        """生成授权码。
        
        Args:
            client_id: 客户端ID
            redirect_uri: 重定向URI
            user_id: 用户ID
            scope: 权限范围
            
        Returns:
            授权码
        """
        code = secrets.token_urlsafe(32)
        self.auth_codes[code] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "user_id": user_id,
            "scope": scope,
            "expires_at": time.time() + 300  # 5分钟过期
        }
        return code
    
    def exchange_code_for_token(
        self,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str
    ) -> Optional[OAuth2Token]:
        """用授权码换取令牌。
        
        Args:
            code: 授权码
            client_id: 客户端ID
            client_secret: 客户端密钥
            redirect_uri: 重定向URI
            
        Returns:
            OAuth2令牌，失败返回None
        """
        # 验证授权码
        code_data = self.auth_codes.get(code)
        if not code_data:
            return None
        
        # 检查过期
        if time.time() > code_data["expires_at"]:
            self.auth_codes.pop(code, None)
            return None
        
        # 验证客户端
        if code_data["client_id"] != client_id:
            return None
        
        client = self.clients.get(client_id)
        if not client or client["client_secret"] != client_secret:
            return None
        
        if redirect_uri != code_data["redirect_uri"]:
            return None
        
        # 删除授权码（一次性使用）
        self.auth_codes.pop(code, None)
        
        # 生成令牌
        return self._generate_tokens(
            user_id=code_data["user_id"],
            scope=code_data["scope"],
            client_id=client_id
        )
    
    def client_credentials_grant(
        self,
        client_id: str,
        client_secret: str,
        scope: str = ""
    ) -> Optional[OAuth2Token]:
        """客户端凭证授权。
        
        Args:
            client_id: 客户端ID
            client_secret: 客户端密钥
            scope: 权限范围
            
        Returns:
            OAuth2令牌
        """
        client = self.clients.get(client_id)
        if not client or client["client_secret"] != client_secret:
            return None
        
        return self._generate_tokens(
            user_id=client_id,
            scope=scope or client.get("scope", ""),
            client_id=client_id
        )
    
    def refresh_token(
        self,
        refresh_token: str,
        client_id: str,
        client_secret: str
    ) -> Optional[OAuth2Token]:
        """刷新令牌。
        
        Args:
            refresh_token: 刷新令牌
            client_id: 客户端ID
            client_secret: 客户端密钥
            
        Returns:
            新的OAuth2令牌
        """
        # 查找对应的访问令牌
        old_token = None
        for token in self.tokens.values():
            if token.refresh_token == refresh_token:
                old_token = token
                break
        
        if not old_token:
            return None
        
        # 验证客户端
        client = self.clients.get(client_id)
        if not client or client["client_secret"] != client_secret:
            return None
        
        # 获取用户信息
        user_info = self.token_to_user.get(old_token.access_token, {})
        
        # 撤销旧令牌
        self.tokens.pop(old_token.access_token, None)
        self.token_to_user.pop(old_token.access_token, None)
        
        # 生成新令牌
        return self._generate_tokens(
            user_id=user_info.get("id", client_id),
            scope=old_token.scope,
            client_id=client_id
        )
    
    def _generate_tokens(
        self,
        user_id: str,
        scope: str,
        client_id: str
    ) -> OAuth2Token:
        """生成访问令牌和刷新令牌。"""
        timestamp = str(int(time.time()))
        
        # 访问令牌
        access_payload = f"{user_id}:{timestamp}:access:{scope}"
        access_signature = hmac.new(
            self._secret,
            access_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        access_token = base64.urlsafe_b64encode(
            f"{access_payload}:{access_signature}".encode("utf-8")
        ).decode("ascii")
        
        # 刷新令牌
        refresh_payload = f"{user_id}:{timestamp}:refresh:{scope}"
        refresh_signature = hmac.new(
            self._secret,
            refresh_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        refresh_token = base64.urlsafe_b64encode(
            f"{refresh_payload}:{refresh_signature}".encode("utf-8")
        ).decode("ascii")
        
        # 创建令牌对象
        token = OAuth2Token(
            access_token=access_token,
            token_type="Bearer",
            expires_in=self.token_expiry,
            refresh_token=refresh_token,
            scope=scope,
            issued_at=time.time()
        )
        
        self.tokens[access_token] = token
        self.token_to_user[access_token] = {
            "id": user_id,
            "scope": scope,
            "client_id": client_id
        }
        
        return token
    
    def authenticate(self, credentials: AuthCredentials) -> bool:
        """验证凭证。"""
        if not isinstance(credentials, OAuth2Credentials):
            return False
        
        if credentials.token and credentials.token.is_valid():
            return credentials.token.access_token in self.tokens
        
        return False
    
    def validate_request(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """验证请求头。"""
        auth_header = headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token_str = auth_header[7:]
        token = self.tokens.get(token_str)
        
        if not token or token.is_expired():
            return None
        
        return self.token_to_user.get(token_str)


class AuthMiddleware:
    """认证中间件。
    
    用于在MCP服务器中验证请求的认证信息。
    """
    
    def __init__(
        self,
        provider: AuthProvider,
        required: bool = True,
        required_scope: Optional[str] = None
    ):
        """初始化认证中间件。
        
        Args:
            provider: 认证提供者
            required: 是否必须认证
            required_scope: 必需的权限范围
        """
        self.provider = provider
        self.required = required
        self.required_scope = required_scope
    
    def process(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """处理请求认证。
        
        Args:
            headers: HTTP请求头
            
        Returns:
            用户信息，认证失败返回None
            
        Raises:
            AuthError: 认证错误
        """
        user_info = self.provider.validate_request(headers)
        
        if user_info is None:
            if self.required:
                raise InvalidCredentialsError("Authentication required")
            return None
        
        # 检查权限范围
        if self.required_scope:
            user_scope = user_info.get("scope", "")
            if not self._check_scope(user_scope, self.required_scope):
                raise InsufficientScopeError(
                    f"Required scope: {self.required_scope}, got: {user_scope}"
                )
        
        return user_info
    
    def _check_scope(self, user_scope: str, required_scope: str) -> bool:
        """检查权限范围。"""
        if not user_scope:
            return False
        
        user_scopes = set(user_scope.split())
        required_scopes = set(required_scope.split())
        
        return required_scopes.issubset(user_scopes)


class AuthManager:
    """认证管理器。
    
    统一管理多种认证方式。
    """
    
    def __init__(self) -> None:
        """初始化认证管理器。"""
        self.providers: Dict[AuthType, AuthProvider] = {}
        self.credentials: Dict[str, AuthCredentials] = {}
    
    def register_provider(self, auth_type: AuthType, provider: AuthProvider) -> None:
        """注册认证提供者。
        
        Args:
            auth_type: 认证类型
            provider: 认证提供者
        """
        self.providers[auth_type] = provider
    
    def add_credentials(self, name: str, credentials: AuthCredentials) -> None:
        """添加凭证。
        
        Args:
            name: 凭证名称
            credentials: 凭证对象
        """
        self.credentials[name] = credentials
    
    def get_credentials(self, name: str) -> Optional[AuthCredentials]:
        """获取凭证。
        
        Args:
            name: 凭证名称
            
        Returns:
            凭证对象
        """
        return self.credentials.get(name)
    
    def authenticate(
        self,
        credentials: AuthCredentials,
        auth_type: Optional[AuthType] = None
    ) -> bool:
        """验证凭证。
        
        Args:
            credentials: 凭证对象
            auth_type: 指定认证类型
            
        Returns:
            认证是否成功
        """
        provider_type = auth_type or credentials.auth_type
        provider = self.providers.get(provider_type)
        
        if not provider:
            return False
        
        return provider.authenticate(credentials)
    
    def validate_request(
        self,
        headers: Dict[str, str],
        auth_type: AuthType = AuthType.BEARER
    ) -> Optional[Dict[str, Any]]:
        """验证请求。
        
        Args:
            headers: HTTP请求头
            auth_type: 认证类型
            
        Returns:
            用户信息
        """
        provider = self.providers.get(auth_type)
        if not provider:
            return None
        
        return provider.validate_request(headers)


__all__ = [
    "AuthType",
    "AuthError",
    "InvalidCredentialsError",
    "ExpiredTokenError",
    "InsufficientScopeError",
    "AuthCredentials",
    "APIKeyCredentials",
    "BearerCredentials",
    "BasicCredentials",
    "OAuth2Token",
    "OAuth2Credentials",
    "AuthProvider",
    "APIKeyAuthProvider",
    "BearerAuthProvider",
    "OAuth2AuthProvider",
    "AuthMiddleware",
    "AuthManager",
]
