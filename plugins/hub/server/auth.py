"""
认证授权模块

提供JWT认证、OAuth集成、权限控制和API密钥管理功能。
"""

import hashlib
import hmac
import json
import secrets
import time
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class Permission(Enum):
    """权限枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    PUBLISH = "publish"
    REVIEW = "review"


@dataclass
class User:
    """用户"""
    user_id: str
    username: str
    email: str = ""
    password_hash: str = ""
    salt: str = ""
    permissions: Set[str] = field(default_factory=lambda: {Permission.READ.value})
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    is_active: bool = True
    is_verified: bool = False
    oauth_provider: Optional[str] = None
    oauth_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'permissions': list(self.permissions),
            'created_at': self.created_at,
            'last_login': self.last_login,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
        }
    
    def has_permission(self, permission: str) -> bool:
        """检查权限"""
        return permission in self.permissions or Permission.ADMIN.value in self.permissions


@dataclass
class APIKey:
    """API密钥"""
    key_id: str
    key_hash: str
    user_id: str
    name: str = ""
    permissions: Set[str] = field(default_factory=lambda: {Permission.READ.value})
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    usage_count: int = 0
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key_id': self.key_id,
            'user_id': self.user_id,
            'name': self.name,
            'permissions': list(self.permissions),
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'last_used': self.last_used,
            'usage_count': self.usage_count,
            'is_active': self.is_active,
        }
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class JWTToken:
    """JWT令牌"""
    token: str
    user_id: str
    expires_at: float
    issued_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() > self.expires_at


# ---------------------------------------------------------------------------
# JWT认证
# ---------------------------------------------------------------------------

class JWTAuth:
    """JWT认证处理器"""
    
    def __init__(self, secret_key: Optional[str] = None,
                 algorithm: str = "HS256",
                 access_token_expire: int = 3600,
                 refresh_token_expire: int = 86400 * 7):
        """
        Args:
            secret_key: JWT密钥
            algorithm: 算法
            access_token_expire: 访问令牌过期时间（秒）
            refresh_token_expire: 刷新令牌过期时间（秒）
        """
        self._secret = secret_key or secrets.token_hex(32)
        self._algorithm = algorithm
        self._access_expire = access_token_expire
        self._refresh_expire = refresh_token_expire
        self._refresh_tokens: Dict[str, str] = {}  # token -> user_id
        self._lock = threading.Lock()
    
    def create_access_token(self, user_id: str,
                            additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """创建访问令牌
        
        Args:
            user_id: 用户ID
            additional_claims: 额外声明
            
        Returns:
            JWT令牌
        """
        now = time.time()
        payload = {
            'sub': user_id,
            'iat': now,
            'exp': now + self._access_expire,
            'type': 'access',
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return self._encode(payload)
    
    def create_refresh_token(self, user_id: str) -> str:
        """创建刷新令牌"""
        now = time.time()
        payload = {
            'sub': user_id,
            'iat': now,
            'exp': now + self._refresh_expire,
            'type': 'refresh',
            'jti': secrets.token_hex(16),
        }
        
        token = self._encode(payload)
        
        with self._lock:
            self._refresh_tokens[payload['jti']] = user_id
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            解码后的payload，验证失败返回None
        """
        try:
            payload = self._decode(token)
            
            # 检查过期
            if payload.get('exp', 0) < time.time():
                return None
            
            # 检查类型
            if payload.get('type') != 'access':
                return None
            
            return payload
        except Exception:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """刷新访问令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            新的访问令牌，失败返回None
        """
        try:
            payload = self._decode(refresh_token)
            
            # 检查类型
            if payload.get('type') != 'refresh':
                return None
            
            # 检查过期
            if payload.get('exp', 0) < time.time():
                return None
            
            jti = payload.get('jti')
            user_id = payload.get('sub')
            
            with self._lock:
                if jti not in self._refresh_tokens:
                    return None
                
                if self._refresh_tokens[jti] != user_id:
                    return None
            
            return self.create_access_token(user_id)
        except Exception:
            return None
    
    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """撤销刷新令牌"""
        try:
            payload = self._decode(refresh_token)
            jti = payload.get('jti')
            
            with self._lock:
                if jti in self._refresh_tokens:
                    del self._refresh_tokens[jti]
                    return True
            
            return False
        except Exception:
            return False
    
    def _encode(self, payload: Dict[str, Any]) -> str:
        """编码JWT"""
        # 简化实现，实际应使用PyJWT库
        header = base64.urlsafe_b64encode(
            json.dumps({'alg': self._algorithm, 'typ': 'JWT'}).encode()
        ).decode().rstrip('=')
        
        payload_encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip('=')
        
        signature = hmac.new(
            self._secret.encode(),
            f"{header}.{payload_encoded}".encode(),
            hashlib.sha256
        ).digest()
        signature_encoded = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f"{header}.{payload_encoded}.{signature_encoded}"
    
    def _decode(self, token: str) -> Dict[str, Any]:
        """解码JWT"""
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        
        # 验证签名
        signature = hmac.new(
            self._secret.encode(),
            f"{parts[0]}.{parts[1]}".encode(),
            hashlib.sha256
        ).digest()
        expected_sig = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        if not hmac.compare_digest(parts[2], expected_sig):
            raise ValueError("Invalid signature")
        
        # 解码payload
        payload_json = base64.urlsafe_b64decode(parts[1] + '==').decode()
        return json.loads(payload_json)


# ---------------------------------------------------------------------------
# OAuth处理器
# ---------------------------------------------------------------------------

class OAuthHandler:
    """OAuth处理器"""
    
    def __init__(self):
        self._providers: Dict[str, Dict[str, str]] = {}
        self._state_store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_provider(self, name: str, client_id: str,
                          client_secret: str,
                          authorize_url: str,
                          token_url: str,
                          userinfo_url: str) -> None:
        """注册OAuth提供商
        
        Args:
            name: 提供商名称
            client_id: 客户端ID
            client_secret: 客户端密钥
            authorize_url: 授权URL
            token_url: 令牌URL
            userinfo_url: 用户信息URL
        """
        self._providers[name] = {
            'client_id': client_id,
            'client_secret': client_secret,
            'authorize_url': authorize_url,
            'token_url': token_url,
            'userinfo_url': userinfo_url,
        }
    
    def get_authorization_url(self, provider: str,
                              redirect_uri: str,
                              scope: str = "read") -> str:
        """获取授权URL
        
        Args:
            provider: 提供商名称
            redirect_uri: 重定向URI
            scope: 权限范围
            
        Returns:
            授权URL
        """
        if provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        state = secrets.token_urlsafe(32)
        
        with self._lock:
            self._state_store[state] = {
                'provider': provider,
                'redirect_uri': redirect_uri,
                'created_at': time.time(),
            }
        
        config = self._providers[provider]
        
        return (
            f"{config['authorize_url']}"
            f"?client_id={config['client_id']}"
            f"&redirect_uri={redirect_uri}"
            f"&scope={scope}"
            f"&state={state}"
            f"&response_type=code"
        )
    
    def handle_callback(self, provider: str, code: str,
                        state: str) -> Optional[Dict[str, Any]]:
        """处理OAuth回调
        
        Args:
            provider: 提供商名称
            code: 授权码
            state: 状态码
            
        Returns:
            用户信息，失败返回None
        """
        # 验证state
        with self._lock:
            if state not in self._state_store:
                return None
            
            state_data = self._state_store[state]
            del self._state_store[state]
        
        if state_data['provider'] != provider:
            return None
        
        # 这里应该实现实际的OAuth令牌交换
        # 简化实现，返回模拟数据
        return {
            'provider': provider,
            'oauth_id': secrets.token_hex(16),
            'email': f"user@{provider}.com",
            'username': f"user_{secrets.token_hex(8)}",
        }
    
    def cleanup_expired_states(self, max_age: int = 600) -> int:
        """清理过期的state
        
        Args:
            max_age: 最大存活时间（秒）
            
        Returns:
            清理数量
        """
        cutoff = time.time() - max_age
        removed = 0
        
        with self._lock:
            expired = [
                state for state, data in self._state_store.items()
                if data['created_at'] < cutoff
            ]
            for state in expired:
                del self._state_store[state]
                removed += 1
        
        return removed


# ---------------------------------------------------------------------------
# 权限管理器
# ---------------------------------------------------------------------------

class PermissionManager:
    """权限管理器"""
    
    def __init__(self):
        self._role_permissions: Dict[str, Set[str]] = {
            'guest': {Permission.READ.value},
            'user': {Permission.READ.value, Permission.WRITE.value},
            'developer': {
                Permission.READ.value,
                Permission.WRITE.value,
                Permission.PUBLISH.value,
            },
            'reviewer': {
                Permission.READ.value,
                Permission.REVIEW.value,
            },
            'admin': {
                Permission.READ.value,
                Permission.WRITE.value,
                Permission.DELETE.value,
                Permission.ADMIN.value,
                Permission.PUBLISH.value,
                Permission.REVIEW.value,
            },
        }
        
        self._resource_permissions: Dict[str, Dict[str, Set[str]]] = {}
        self._lock = threading.Lock()
    
    def check_permission(self, user: User, permission: str,
                         resource: Optional[str] = None) -> bool:
        """检查用户权限
        
        Args:
            user: 用户
            permission: 权限
            resource: 资源（可选）
            
        Returns:
            是否有权限
        """
        # 检查用户权限
        if user.has_permission(permission):
            return True
        
        # 检查资源权限
        if resource:
            with self._lock:
                resource_perms = self._resource_permissions.get(resource, {})
                if permission in resource_perms.get(user.user_id, set()):
                    return True
        
        return False
    
    def grant_permission(self, user_id: str, permission: str,
                         resource: Optional[str] = None) -> None:
        """授予权限"""
        with self._lock:
            if resource:
                if resource not in self._resource_permissions:
                    self._resource_permissions[resource] = {}
                if user_id not in self._resource_permissions[resource]:
                    self._resource_permissions[resource][user_id] = set()
                self._resource_permissions[resource][user_id].add(permission)
    
    def revoke_permission(self, user_id: str, permission: str,
                          resource: Optional[str] = None) -> None:
        """撤销权限"""
        with self._lock:
            if resource:
                if resource in self._resource_permissions:
                    if user_id in self._resource_permissions[resource]:
                        self._resource_permissions[resource][user_id].discard(permission)
    
    def get_role_permissions(self, role: str) -> Set[str]:
        """获取角色权限"""
        return self._role_permissions.get(role, set()).copy()


# ---------------------------------------------------------------------------
# API密钥管理器
# ---------------------------------------------------------------------------

class APIKeyManager:
    """API密钥管理器"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path
        self._keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self._key_index: Dict[str, str] = {}  # key_hash -> key_id
        self._lock = threading.Lock()
    
    def create_key(self, user_id: str, name: str = "",
                   permissions: Optional[Set[str]] = None,
                   expires_in: Optional[int] = None) -> Tuple[str, APIKey]:
        """创建API密钥
        
        Args:
            user_id: 用户ID
            name: 密钥名称
            permissions: 权限集合
            expires_in: 过期时间（秒）
            
        Returns:
            (原始密钥, APIKey对象)
        """
        key_id = secrets.token_hex(16)
        raw_key = f"ch_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions or {Permission.READ.value},
            expires_at=expires_at,
        )
        
        with self._lock:
            self._keys[key_id] = api_key
            self._key_index[key_hash] = key_id
        
        return raw_key, api_key
    
    def verify_key(self, raw_key: str) -> Optional[APIKey]:
        """验证API密钥
        
        Args:
            raw_key: 原始密钥
            
        Returns:
            APIKey对象，无效返回None
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        with self._lock:
            key_id = self._key_index.get(key_hash)
            if not key_id:
                return None
            
            api_key = self._keys.get(key_id)
            if not api_key:
                return None
            
            if not api_key.is_active or api_key.is_expired():
                return None
            
            # 更新使用统计
            api_key.last_used = time.time()
            api_key.usage_count += 1
            
            return api_key
    
    def revoke_key(self, key_id: str) -> bool:
        """撤销API密钥"""
        with self._lock:
            if key_id not in self._keys:
                return False
            
            api_key = self._keys[key_id]
            api_key.is_active = False
            
            # 从索引中移除
            if api_key.key_hash in self._key_index:
                del self._key_index[api_key.key_hash]
            
            return True
    
    def delete_key(self, key_id: str) -> bool:
        """删除API密钥"""
        with self._lock:
            if key_id not in self._keys:
                return False
            
            api_key = self._keys[key_id]
            
            # 从索引中移除
            if api_key.key_hash in self._key_index:
                del self._key_index[api_key.key_hash]
            
            # 删除密钥
            del self._keys[key_id]
            
            return True
    
    def list_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """列出API密钥"""
        with self._lock:
            keys = list(self._keys.values())
            
            if user_id:
                keys = [k for k in keys if k.user_id == user_id]
            
            return keys
    
    def get_key(self, key_id: str) -> Optional[APIKey]:
        """获取API密钥"""
        with self._lock:
            return self._keys.get(key_id)


# ---------------------------------------------------------------------------
# 认证管理器
# ---------------------------------------------------------------------------

class AuthManager:
    """认证管理器
    
    整合所有认证功能的主类。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path
        
        self._jwt = JWTAuth()
        self._oauth = OAuthHandler()
        self._permissions = PermissionManager()
        self._api_keys = APIKeyManager()
        
        self._users: Dict[str, User] = {}
        self._username_index: Dict[str, str] = {}  # username -> user_id
        self._lock = threading.Lock()
    
    def register_user(self, username: str, password: str,
                      email: str = "") -> Optional[User]:
        """注册用户
        
        Args:
            username: 用户名
            password: 密码
            email: 邮箱
            
        Returns:
            创建的用户，失败返回None
        """
        with self._lock:
            if username in self._username_index:
                return None
            
            user_id = secrets.token_hex(16)
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
            )
            
            self._users[user_id] = user
            self._username_index[username] = user_id
            
            return user
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """用户认证
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            认证成功的用户，失败返回None
        """
        with self._lock:
            user_id = self._username_index.get(username)
            if not user_id:
                return None
            
            user = self._users.get(user_id)
            if not user:
                return None
            
            if not user.is_active:
                return None
            
            password_hash = self._hash_password(password, user.salt)
            if not hmac.compare_digest(password_hash, user.password_hash):
                return None
            
            user.last_login = time.time()
            
            return user
    
    def authenticate_oauth(self, provider: str,
                           oauth_data: Dict[str, Any]) -> Optional[User]:
        """OAuth认证
        
        Args:
            provider: 提供商
            oauth_data: OAuth数据
            
        Returns:
            认证成功的用户，失败返回None
        """
        oauth_id = oauth_data.get('oauth_id')
        email = oauth_data.get('email', '')
        username = oauth_data.get('username', '')
        
        with self._lock:
            # 查找现有用户
            for user in self._users.values():
                if user.oauth_provider == provider and user.oauth_id == oauth_id:
                    user.last_login = time.time()
                    return user
            
            # 创建新用户
            user_id = secrets.token_hex(16)
            user = User(
                user_id=user_id,
                username=username or f"{provider}_{oauth_id[:8]}",
                email=email,
                oauth_provider=provider,
                oauth_id=oauth_id,
                is_verified=True,
            )
            
            self._users[user_id] = user
            self._username_index[user.username] = user_id
            
            return user
    
    def change_password(self, user_id: str, old_password: str,
                        new_password: str) -> bool:
        """修改密码"""
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return False
            
            # 验证旧密码
            old_hash = self._hash_password(old_password, user.salt)
            if not hmac.compare_digest(old_hash, user.password_hash):
                return False
            
            # 更新密码
            new_salt = secrets.token_hex(16)
            user.password_hash = self._hash_password(new_password, new_salt)
            user.salt = new_salt
            
            return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        with self._lock:
            return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """通过用户名获取用户"""
        with self._lock:
            user_id = self._username_index.get(username)
            if user_id:
                return self._users.get(user_id)
            return None
    
    def update_user(self, user: User) -> bool:
        """更新用户信息"""
        with self._lock:
            if user.user_id not in self._users:
                return False
            
            self._users[user.user_id] = user
            return True
    
    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return False
            
            del self._users[user_id]
            del self._username_index[user.username]
            
            return True
    
    def _hash_password(self, password: str, salt: str) -> str:
        """哈希密码"""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
    
    @property
    def jwt(self) -> JWTAuth:
        """获取JWT处理器"""
        return self._jwt
    
    @property
    def oauth(self) -> OAuthHandler:
        """获取OAuth处理器"""
        return self._oauth
    
    @property
    def permissions(self) -> PermissionManager:
        """获取权限管理器"""
        return self._permissions
    
    @property
    def api_keys(self) -> APIKeyManager:
        """获取API密钥管理器"""
        return self._api_keys
