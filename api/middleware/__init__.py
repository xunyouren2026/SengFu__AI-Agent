"""
API中间件模块

提供各种中间件组件用于请求处理流程。

中间件列表:
    - auth: 认证中间件（JWT和API Key）
    - rate_limit: 限流中间件
    - logging: 日志中间件
    - cors: CORS中间件

使用示例:
    >>> from agi_unified_framework.api.middleware import AuthMiddleware
    >>> app.add_middleware(AuthMiddleware)
"""

from __future__ import annotations

from .auth import AuthMiddleware, JWTAuthBackend, APIKeyAuthBackend
from .rate_limit import RateLimitMiddleware
from .logging import LoggingMiddleware
from .cors import CORSMiddleware

__all__ = [
    # 认证中间件
    "AuthMiddleware",
    "JWTAuthBackend",
    "APIKeyAuthBackend",
    # 限流中间件
    "RateLimitMiddleware",
    # 日志中间件
    "LoggingMiddleware",
    # CORS中间件
    "CORSMiddleware",
]
