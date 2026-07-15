"""
认证处理器 (Auth Handler)

提供多种认证方式的统一处理，支持主流国产/国际模型的认证机制。

支持的认证方式:
- bearer: Bearer Token (OpenAI/大部分国产模型)
- api_key_query: URL参数API Key (部分国产模型)
- api_key_header: 自定义Header API Key
- oauth2: OAuth2.0 (百度文心等)
- hmac_sha256: HMAC-SHA256签名 (讯飞星火等)
- aws_sigv4: AWS签名 (部分云服务)
- custom: 自定义认证 (通过配置灵活扩展)

Author: AGI Team
Version: 1.0.0
"""

import time
import json
import hashlib
import hmac as hmac_module
import base64
import logging
import os
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Union
)
from urllib.parse import urlparse, urlencode, quote, urlunparse
from collections import OrderedDict

import aiohttp

logger = logging.getLogger(__name__)


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class AuthConfig:
    """
    认证配置

    Attributes:
        auth_type: 认证类型
        api_key: API密钥
        secret_key: 密钥 (用于签名)
        env_var: 环境变量名 (从环境变量读取密钥)
        header_name: 自定义Header名称
        query_param: URL查询参数名
        token_url: OAuth2 Token端点
        client_id: OAuth2 客户端ID
        client_secret: OAuth2 客户端密钥
        scope: OAuth2 权限范围
        grant_type: OAuth2 授权类型
        extra: 额外配置
    """
    auth_type: str = "bearer"
    api_key: str = ""
    secret_key: str = ""
    env_var: str = ""
    header_name: str = "X-API-Key"
    query_param: str = "api_key"

    # OAuth2 配置
    token_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    scope: str = ""
    grant_type: str = "client_credentials"

    # AWS 配置
    aws_region: str = ""
    aws_service: str = ""
    aws_access_key: str = ""
    aws_secret_key: str = ""

    # HMAC 配置
    hmac_algorithm: str = "sha256"
    hmac_header_prefix: str = "X-"
    hmac_timestamp_header: str = "X-Timestamp"
    hmac_nonce_header: str = "X-Nonce"
    hmac_signature_header: str = "X-Signature"

    # 自定义配置
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthConfig":
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class AuthResult:
    """
    认证结果

    Attributes:
        success: 是否成功
        headers: 需要添加的HTTP头
        query_params: 需要添加的URL查询参数
        url: 修改后的URL (可选)
        error: 错误信息
        token: 获取到的token (如果有)
        expires_at: token过期时间 (如果有)
    """
    success: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    error: Optional[str] = None
    token: Optional[str] = None
    expires_at: Optional[datetime] = None


# ============================================================
# Token缓存
# ============================================================

class TokenCache:
    """
    Token缓存管理器

    用于缓存OAuth2等需要动态获取的认证token，避免频繁请求。

    Attributes:
        max_size: 最大缓存数量
        default_ttl: 默认过期时间(秒)
    """

    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        """
        初始化Token缓存。

        Args:
            max_size: 最大缓存数量
            default_ttl: 默认过期时间(秒)
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[str, datetime]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        """
        获取缓存的token

        Args:
            key: 缓存键

        Returns:
            token字符串，如果不存在或已过期返回None
        """
        async with self._lock:
            if key not in self._cache:
                return None

            token, expires_at = self._cache[key]
            if datetime.now() >= expires_at:
                del self._cache[key]
                return None

            # 移到末尾 (LRU)
            self._cache.move_to_end(key)
            return token

    async def set(
        self,
        key: str,
        token: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        设置缓存token

        Args:
            key: 缓存键
            token: token值
            ttl: 过期时间(秒)，None使用默认值
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

            expires_at = datetime.now() + timedelta(
                seconds=ttl or self._default_ttl
            )
            self._cache[key] = (token, expires_at)

            # 超出容量时移除最旧的
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    async def remove(self, key: str) -> None:
        """
        移除缓存token

        Args:
            key: 缓存键
        """
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """清空所有缓存"""
        async with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """当前缓存数量"""
        return len(self._cache)


# ============================================================
# 认证处理器基类
# ============================================================

class BaseAuthHandler(ABC):
    """
    认证处理器基类

    所有认证处理器都应继承此类并实现抽象方法。
    """

    def __init__(self, config: AuthConfig):
        """
        初始化认证处理器。

        Args:
            config: 认证配置
        """
        self._config = config
        self._token_cache = TokenCache()

    @abstractmethod
    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行认证

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头 (可修改)
            body: 请求体

        Returns:
            认证结果
        """
        pass

    def _resolve_api_key(self) -> str:
        """
        解析API密钥

        优先级: 直接配置 > 环境变量 > extra字段

        Returns:
            API密钥字符串
        """
        if self._config.api_key:
            return self._config.api_key

        if self._config.env_var:
            value = os.environ.get(self._config.env_var, "")
            if value:
                return value

        return self._config.extra.get("api_key", "")

    def _resolve_secret_key(self) -> str:
        """
        解析密钥

        Returns:
            密钥字符串
        """
        if self._config.secret_key:
            return self._config.secret_key

        env_var = self._config.extra.get("secret_env_var", "")
        if env_var:
            return os.environ.get(env_var, "")

        return self._config.extra.get("secret_key", "")


# ============================================================
# Bearer Token 认证
# ============================================================

class BearerAuthHandler(BaseAuthHandler):
    """
    Bearer Token 认证处理器

    最常用的认证方式，适用于:
    - OpenAI (GPT系列)
    - 智谱AI (GLM系列)
    - 通义千问 (Qwen系列)
    - DeepSeek
    - Kimi (月之暗面)
    - MiniMax
    - 零一万物 (Yi系列)
    - 百川智能 (Baichuan系列)
    - 以及大部分OpenAI兼容的国产模型
    """

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行Bearer Token认证

        在Authorization头中添加Bearer Token。

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            认证结果
        """
        api_key = self._resolve_api_key()
        if not api_key:
            return AuthResult(
                success=False,
                error="API密钥未配置",
            )

        headers["Authorization"] = f"Bearer {api_key}"
        return AuthResult(success=True, token=api_key)


# ============================================================
# API Key Query 认证
# ============================================================

class ApiKeyQueryAuthHandler(BaseAuthHandler):
    """
    URL参数API Key认证处理器

    将API Key作为URL查询参数传递，适用于:
    - 部分国产模型的旧版API
    - 某些需要URL参数认证的内部服务

    Example:
        GET https://api.example.com/v1/chat?api_key=your_key
    """

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行URL参数API Key认证

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            认证结果，包含修改后的URL
        """
        api_key = self._resolve_api_key()
        if not api_key:
            return AuthResult(
                success=False,
                error="API密钥未配置",
            )

        param_name = self._config.query_param
        parsed = urlparse(url)

        # 合并现有查询参数
        query_params: Dict[str, str] = {}
        if parsed.query:
            for pair in parsed.query.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    query_params[k] = v

        query_params[param_name] = api_key

        new_query = urlencode(query_params)
        new_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        ))

        return AuthResult(
            success=True,
            url=new_url,
            query_params={param_name: api_key},
            token=api_key,
        )


# ============================================================
# API Key Header 认证
# ============================================================

class ApiKeyHeaderAuthHandler(BaseAuthHandler):
    """
    自定义Header API Key认证处理器

    将API Key放在自定义的HTTP头中，适用于:
    - 部分使用非标准Header名称的国产模型
    - 内部API服务

    Example:
        X-API-Key: your_key
        X-App-Key: your_key
    """

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行自定义Header API Key认证

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            认证结果
        """
        api_key = self._resolve_api_key()
        if not api_key:
            return AuthResult(
                success=False,
                error="API密钥未配置",
            )

        header_name = self._config.header_name
        headers[header_name] = api_key

        return AuthResult(success=True, token=api_key)


# ============================================================
# OAuth2.0 认证
# ============================================================

class OAuth2AuthHandler(BaseAuthHandler):
    """
    OAuth2.0 认证处理器

    适用于需要OAuth2授权流程的模型提供商:
    - 百度文心一言 (ERNIE系列)
    - 部分企业级API服务

    支持的授权类型:
    - client_credentials: 客户端凭证模式
    - authorization_code: 授权码模式 (需配合回调)
    - refresh_token: 刷新令牌模式

    Token会自动缓存并在过期前刷新。
    """

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行OAuth2.0认证

        自动获取或刷新access_token。

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            认证结果
        """
        cache_key = f"oauth2_{self._config.client_id}"

        # 尝试从缓存获取
        cached_token = await self._token_cache.get(cache_key)
        if cached_token:
            headers["Authorization"] = f"Bearer {cached_token}"
            return AuthResult(success=True, token=cached_token)

        # 获取新token
        token_result = await self._fetch_token()
        if not token_result.success:
            return token_result

        # 缓存token
        ttl = None
        if token_result.expires_at:
            delta = token_result.expires_at - datetime.now()
            ttl = max(int(delta.total_seconds()) - 300, 60)  # 提前5分钟刷新

        await self._token_cache.set(cache_key, token_result.token, ttl)

        headers["Authorization"] = f"Bearer {token_result.token}"
        return AuthResult(
            success=True,
            token=token_result.token,
            expires_at=token_result.expires_at,
        )

    async def _fetch_token(self) -> AuthResult:
        """
        获取OAuth2 access token

        Returns:
            包含token的认证结果
        """
        if not self._config.token_url:
            return AuthResult(
                success=False,
                error="OAuth2 token_url未配置",
            )

        token_data: Dict[str, str] = {
            "grant_type": self._config.grant_type,
        }

        if self._config.grant_type == "client_credentials":
            token_data["client_id"] = self._config.client_id
            token_data["client_secret"] = self._config.client_secret
            if self._config.scope:
                token_data["scope"] = self._config.scope

        elif self._config.grant_type == "refresh_token":
            token_data["refresh_token"] = self._config.extra.get(
                "refresh_token", ""
            )
            token_data["client_id"] = self._config.client_id
            token_data["client_secret"] = self._config.client_secret

        elif self._config.grant_type == "authorization_code":
            token_data["code"] = self._config.extra.get("auth_code", "")
            token_data["redirect_uri"] = self._config.extra.get(
                "redirect_uri", ""
            )
            token_data["client_id"] = self._config.client_id
            token_data["client_secret"] = self._config.client_secret

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.token_url,
                    data=token_data,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    body = await resp.json()

                    if resp.status >= 400:
                        error_desc = body.get("error_description", body.get("error", "未知错误"))
                        return AuthResult(
                            success=False,
                            error=f"OAuth2 token获取失败: {error_desc}",
                        )

                    access_token = body.get("access_token", "")
                    expires_in = body.get("expires_in", 3600)

                    if not access_token:
                        return AuthResult(
                            success=False,
                            error="OAuth2响应中缺少access_token",
                        )

                    expires_at = datetime.now() + timedelta(
                        seconds=expires_in
                    )

                    return AuthResult(
                        success=True,
                        token=access_token,
                        expires_at=expires_at,
                    )

        except aiohttp.ClientError as e:
            return AuthResult(
                success=False,
                error=f"OAuth2请求失败: {e}",
            )
        except json.JSONDecodeError as e:
            return AuthResult(
                success=False,
                error=f"OAuth2响应解析失败: {e}",
            )


# ============================================================
# HMAC-SHA256 签名认证
# ============================================================

class HmacSha256AuthHandler(BaseAuthHandler):
    """
    HMAC-SHA256 签名认证处理器

    适用于需要请求签名的模型提供商:
    - 讯飞星火 (Spark系列)
    - 部分安全要求较高的企业API

    签名流程:
    1. 构建签名字符串 (method + path + query + timestamp + nonce + body_hash)
    2. 使用密钥对签名字符串进行HMAC-SHA256计算
    3. 将签名结果添加到请求头中
    """

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行HMAC-SHA256签名认证

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            认证结果，包含签名头
        """
        secret_key = self._resolve_secret_key()
        if not secret_key:
            return AuthResult(
                success=False,
                error="签名密钥未配置",
            )

        parsed = urlparse(url)
        path = parsed.path or "/"
        query = parsed.query

        timestamp = str(int(time.time()))
        nonce = hashlib.md5(
            f"{timestamp}{self._config.client_id or 'default'}".encode()
        ).hexdigest()

        # 构建签名字符串
        sign_parts = [method.upper(), path, query, timestamp, nonce]

        if body:
            body_hash = hashlib.sha256(body).hexdigest()
            sign_parts.append(body_hash)

        sign_str = "\n".join(sign_parts)

        # 计算签名
        algorithm = self._config.hmac_algorithm.lower()
        if algorithm == "sha256":
            hash_func = hashlib.sha256
        elif algorithm == "sha512":
            hash_func = hashlib.sha512
        elif algorithm == "sha1":
            hash_func = hashlib.sha1
        else:
            hash_func = hashlib.sha256

        signature = hmac_module.new(
            secret_key.encode("utf-8"),
            sign_str.encode("utf-8"),
            hash_func
        ).hexdigest()

        # 添加签名头
        prefix = self._config.hmac_header_prefix
        headers[f"{prefix}Timestamp"] = timestamp
        headers[f"{prefix}Nonce"] = nonce
        headers[f"{prefix}Signature"] = signature

        # 也支持自定义头名称
        if self._config.hmac_timestamp_header:
            headers[self._config.hmac_timestamp_header] = timestamp
        if self._config.hmac_nonce_header:
            headers[self._config.hmac_nonce_header] = nonce
        if self._config.hmac_signature_header:
            headers[self._config.hmac_signature_header] = signature

        # 如果配置了API Key，也添加到头中
        api_key = self._resolve_api_key()
        if api_key:
            headers[f"{prefix}Api-Key"] = api_key

        return AuthResult(
            success=True,
            headers=headers.copy(),
            token=api_key,
        )


# ============================================================
# AWS Signature V4 认证
# ============================================================

class AwsSigV4AuthHandler(BaseAuthHandler):
    """
    AWS Signature V4 认证处理器

    适用于使用AWS签名认证的云服务:
    - 阿里云DashScope (部分接口)
    - 华为云ModelArts
    - AWS Bedrock
    - 其他兼容AWS SigV4的服务

    实现AWS Signature Version 4签名流程。
    """

    # AWS需要的头信息
    REQUIRED_HEADERS = {"host", "x-amz-date", "x-amz-content-sha256"}

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行AWS Signature V4认证

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            认证结果
        """
        access_key = self._config.aws_access_key or self._resolve_api_key()
        secret_key = self._config.aws_secret_key or self._resolve_secret_key()

        if not access_key or not secret_key:
            return AuthResult(
                success=False,
                error="AWS Access Key或Secret Key未配置",
            )

        region = self._config.aws_region or "us-east-1"
        service = self._config.aws_service or "execute-api"

        now = datetime.utcnow()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        parsed = urlparse(url)
        host = parsed.netloc
        canonical_uri = parsed.path or "/"

        # 规范查询字符串
        canonical_querystring = self._canonical_query_string(parsed.query)

        # 请求体哈希
        if body:
            payload_hash = hashlib.sha256(body).hexdigest()
        else:
            payload_hash = hashlib.sha256(b"").hexdigest()

        # 添加必要的头
        headers["host"] = host
        headers["x-amz-date"] = amz_date
        headers["x-amz-content-sha256"] = payload_hash

        # 规范头
        signed_header_keys = sorted(
            k.lower() for k in headers.keys()
        )
        canonical_headers = ""
        for key in signed_header_keys:
            canonical_headers += f"{key}:{headers.get(key, '').strip()}\n"
        signed_headers = ";".join(signed_header_keys)

        # 规范请求
        canonical_request = "\n".join([
            method.upper(),
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash,
        ])

        # 待签字符串
        credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
        string_to_sign = "\n".join([
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ])

        # 签名密钥
        signing_key = self._get_signature_key(
            secret_key, date_stamp, region, service
        )

        # 计算签名
        signature = hmac_module.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        # 添加Authorization头
        authorization_header = (
            f"AWS4-HMAC-SHA256 "
            f"Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )
        headers["Authorization"] = authorization_header

        return AuthResult(
            success=True,
            headers=headers.copy(),
            token=access_key,
        )

    @staticmethod
    def _canonical_query_string(query: str) -> str:
        """
        构建规范查询字符串

        Args:
            query: 原始查询字符串

        Returns:
            规范化的查询字符串
        """
        if not query:
            return ""

        params: List[Tuple[str, str]] = []
        for pair in query.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params.append((quote(k, safe=""), quote(v, safe="")))
            else:
                params.append((quote(pair, safe=""), ""))

        params.sort()
        return "&".join(f"{k}={v}" for k, v in params)

    @staticmethod
    def _get_signature_key(
        key: str,
        date_stamp: str,
        region: str,
        service: str
    ) -> bytes:
        """
        派生签名密钥

        Args:
            key: Secret Key
            date_stamp: 日期戳
            region: 区域
            service: 服务名

        Returns:
            签名密钥
        """
        k_date = hmac_module.new(
            f"AWS4{key}".encode("utf-8"),
            date_stamp.encode("utf-8"),
            hashlib.sha256
        ).digest()
        k_region = hmac_module.new(
            k_date, region.encode("utf-8"), hashlib.sha256
        ).digest()
        k_service = hmac_module.new(
            k_region, service.encode("utf-8"), hashlib.sha256
        ).digest()
        k_signing = hmac_module.new(
            k_service, b"aws4_request", hashlib.sha256
        ).digest()
        return k_signing


# ============================================================
# 自定义认证处理器
# ============================================================

class CustomAuthHandler(BaseAuthHandler):
    """
    自定义认证处理器

    通过配置灵活定义认证方式，适用于:
    - 非标准认证流程的模型
    - 内部私有API
    - 任何无法用标准方式覆盖的认证场景

    配置示例:
        {
            "auth_type": "custom",
            "extra": {
                "headers": {
                    "X-Custom-Auth": "Bearer {api_key}",
                    "X-App-Id": "my-app"
                },
                "template_vars": {
                    "api_key": "env:MY_API_KEY"
                },
                "token_url": "https://auth.example.com/token",
                "token_path": "data.token",
                "token_header": "X-Access-Token",
                "token_cache_ttl": 1800
            }
        }
    """

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行自定义认证

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体

        Returns:
            认证结果
        """
        extra = self._config.extra

        # 1. 处理自定义头模板
        custom_headers = extra.get("headers", {})
        template_vars = self._resolve_template_vars(
            extra.get("template_vars", {})
        )

        for header_name, header_value in custom_headers.items():
            resolved_value = self._apply_template(header_value, template_vars)
            headers[header_name] = resolved_value

        # 2. 处理自定义token获取
        token_url = extra.get("token_url")
        if token_url:
            token_result = await self._fetch_custom_token(
                token_url, extra, template_vars
            )
            if token_result.success and token_result.token:
                token_header = extra.get("token_header", "Authorization")
                token_prefix = extra.get("token_prefix", "Bearer ")
                headers[token_header] = f"{token_prefix}{token_result.token}"
                return AuthResult(
                    success=True,
                    token=token_result.token,
                    expires_at=token_result.expires_at,
                )
            elif not token_result.success:
                return token_result

        return AuthResult(success=True)

    def _resolve_template_vars(
        self,
        vars_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        解析模板变量

        支持的变量来源:
        - "env:VAR_NAME" - 从环境变量读取
        - "config:field_name" - 从AuthConfig读取
        - 直接值 - 使用原值

        Args:
            vars_config: 变量配置

        Returns:
            解析后的变量字典
        """
        result: Dict[str, str] = {}

        for var_name, var_source in vars_config.items():
            if isinstance(var_source, str):
                if var_source.startswith("env:"):
                    env_name = var_source[4:]
                    result[var_name] = os.environ.get(env_name, "")
                elif var_source.startswith("config:"):
                    field_name = var_source[7:]
                    result[var_name] = getattr(self._config, field_name, "")
                else:
                    result[var_name] = var_source
            else:
                result[var_name] = str(var_source)

        return result

    @staticmethod
    def _apply_template(
        template: str,
        variables: Dict[str, str]
    ) -> str:
        """
        应用模板变量替换

        Args:
            template: 模板字符串，使用 {var_name} 格式
            variables: 变量字典

        Returns:
            替换后的字符串
        """
        try:
            return template.format(**variables)
        except KeyError:
            return template

    async def _fetch_custom_token(
        self,
        token_url: str,
        extra: Dict[str, Any],
        template_vars: Dict[str, str]
    ) -> AuthResult:
        """
        获取自定义token

        Args:
            token_url: Token端点URL
            extra: 额外配置
            template_vars: 模板变量

        Returns:
            包含token的认证结果
        """
        cache_key = f"custom_{token_url}"
        cached = await self._token_cache.get(cache_key)
        if cached:
            return AuthResult(success=True, token=cached)

        try:
            # 构建token请求
            request_config = extra.get("token_request", {})
            req_method = request_config.get("method", "POST").upper()
            req_headers = request_config.get("headers", {})
            req_body = request_config.get("body", {})

            # 应用模板变量
            resolved_url = self._apply_template(token_url, template_vars)
            resolved_body = {
                k: self._apply_template(str(v), template_vars)
                for k, v in req_body.items()
            }

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    req_method,
                    resolved_url,
                    headers=req_headers,
                    json=resolved_body if req_body else None,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    body = await resp.json()

                    if resp.status >= 400:
                        return AuthResult(
                            success=False,
                            error=f"自定义token获取失败: HTTP {resp.status}",
                        )

                    # 从响应中提取token
                    token_path = extra.get("token_path", "access_token")
                    token = self._get_nested_value(body, token_path)

                    if not token:
                        return AuthResult(
                            success=False,
                            error=f"响应中未找到token (路径: {token_path})",
                        )

                    # 缓存token
                    ttl = extra.get("token_cache_ttl", 1800)
                    await self._token_cache.set(cache_key, token, ttl)

                    return AuthResult(success=True, token=token)

        except Exception as e:
            return AuthResult(
                success=False,
                error=f"自定义token请求异常: {e}",
            )

    @staticmethod
    def _get_nested_value(data: Any, path: str) -> Any:
        """
        从嵌套结构中获取值

        Args:
            data: 数据
            path: 点号分隔的路径

        Returns:
            找到的值
        """
        current = data
        for key in path.split("."):
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(key)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if current is None:
                return None
        return current


# ============================================================
# 认证处理器工厂
# ============================================================

class AuthHandler:
    """
    认证处理器 (统一入口)

    根据认证类型自动选择合适的认证处理器，提供统一的认证接口。

    Example:
        ```python
        # 创建认证处理器
        auth = AuthHandler(AuthConfig(
            auth_type="bearer",
            api_key="your-api-key",
        ))

        # 执行认证
        headers = {"Content-Type": "application/json"}
        result = await auth.authenticate("POST", "https://api.example.com/v1/chat", headers)
        if result.success:
            print("认证成功，请求头:", result.headers)
        ```
    """

    # 认证类型到处理器类的映射
    HANDLER_MAP: Dict[str, type] = {
        "bearer": BearerAuthHandler,
        "api_key_query": ApiKeyQueryAuthHandler,
        "api_key_header": ApiKeyHeaderAuthHandler,
        "oauth2": OAuth2AuthHandler,
        "hmac_sha256": HmacSha256AuthHandler,
        "aws_sigv4": AwsSigV4AuthHandler,
        "custom": CustomAuthHandler,
    }

    def __init__(self, config: AuthConfig):
        """
        初始化认证处理器。

        Args:
            config: 认证配置

        Raises:
            ValueError: 不支持的认证类型
        """
        self._config = config
        auth_type = config.auth_type.lower()

        handler_class = self.HANDLER_MAP.get(auth_type)
        if not handler_class:
            raise ValueError(
                f"不支持的认证类型: {auth_type}。"
                f"支持的类型: {list(self.HANDLER_MAP.keys())}"
            )

        self._handler: BaseAuthHandler = handler_class(config)
        logger.debug(f"已初始化认证处理器: {auth_type}")

    async def authenticate(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> AuthResult:
        """
        执行认证

        根据配置的认证类型，自动选择对应的认证方式处理请求。

        Args:
            method: HTTP方法 (GET/POST等)
            url: 请求URL
            headers: 请求头字典 (会被修改)
            body: 请求体字节

        Returns:
            AuthResult 认证结果，包含:
            - success: 是否成功
            - headers: 需要添加/修改的头
            - query_params: 需要添加的URL参数
            - url: 修改后的URL (如有)
            - error: 错误信息 (如失败)
        """
        return await self._handler.authenticate(method, url, headers, body)

    @property
    def auth_type(self) -> str:
        """获取当前认证类型"""
        return self._config.auth_type

    @property
    def config(self) -> AuthConfig:
        """获取认证配置"""
        return self._config

    @classmethod
    def create(
        cls,
        auth_type: str,
        api_key: str = "",
        **kwargs: Any
    ) -> "AuthHandler":
        """
        快速创建认证处理器

        Args:
            auth_type: 认证类型
            api_key: API密钥
            **kwargs: 其他配置参数

        Returns:
            AuthHandler实例

        Example:
            ```python
            auth = AuthHandler.create("bearer", api_key="sk-xxx")
            auth = AuthHandler.create("oauth2", token_url="...", client_id="...")
            ```
        """
        config = AuthConfig(auth_type=auth_type, api_key=api_key, **kwargs)
        return cls(config)

    @classmethod
    def supported_types(cls) -> List[str]:
        """
        获取所有支持的认证类型

        Returns:
            认证类型列表
        """
        return list(cls.HANDLER_MAP.keys())

    async def clear_token_cache(self) -> None:
        """清空token缓存"""
        await self._handler._token_cache.clear()

    async def validate(self) -> Tuple[bool, str]:
        """
        验证认证配置是否有效

        Returns:
            (是否有效, 错误信息)
        """
        auth_type = self._config.auth_type.lower()

        if auth_type == "bearer":
            if not self._handler._resolve_api_key():
                return False, "API密钥未配置"

        elif auth_type == "api_key_query":
            if not self._handler._resolve_api_key():
                return False, "API密钥未配置"

        elif auth_type == "api_key_header":
            if not self._handler._resolve_api_key():
                return False, "API密钥未配置"

        elif auth_type == "oauth2":
            if not self._config.token_url:
                return False, "OAuth2 token_url未配置"
            if not self._config.client_id:
                return False, "OAuth2 client_id未配置"

        elif auth_type == "hmac_sha256":
            if not self._handler._resolve_secret_key():
                return False, "HMAC签名密钥未配置"

        elif auth_type == "aws_sigv4":
            if not self._config.aws_access_key:
                return False, "AWS Access Key未配置"
            if not self._config.aws_secret_key:
                return False, "AWS Secret Key未配置"

        return True, ""
