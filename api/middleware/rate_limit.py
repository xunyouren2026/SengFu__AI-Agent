"""
限流中间件模块

提供请求限流功能，支持滑动窗口算法。

主要组件:
    - RateLimitMiddleware: 限流中间件
    - SlidingWindowCounter: 滑动窗口计数器
    - RateLimitConfig: 限流配置

使用示例:
    >>> from agi_unified_framework.api.middleware import RateLimitMiddleware
    >>> app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """
    限流配置
    
    Attributes:
        requests_per_minute: 每分钟请求数限制
        requests_per_hour: 每小时请求数限制
        burst_size: 突发请求数
        key_prefix: 键前缀
        excluded_paths: 排除路径
        excluded_ips: 排除IP
    """
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    key_prefix: str = "ratelimit"
    excluded_paths: List[str] = field(default_factory=list)
    excluded_ips: List[str] = field(default_factory=list)


class SlidingWindowCounter:
    """
    滑动窗口计数器
    
    使用滑动窗口算法实现精确的请求限流。
    
    Attributes:
        window_size: 窗口大小（秒）
        max_requests: 最大请求数
    """
    
    def __init__(self, window_size: int = 60, max_requests: int = 60):
        self.window_size = window_size
        self.max_requests = max_requests
        self._requests: deque = deque()
        self._lock = False
    
    def is_allowed(self) -> Tuple[bool, int]:
        """
        检查是否允许请求
        
        Returns:
            (是否允许, 剩余请求数)
        """
        now = time.time()
        
        # 清理过期请求
        while self._requests and self._requests[0] < now - self.window_size:
            self._requests.popleft()
        
        # 检查是否超过限制
        if len(self._requests) >= self.max_requests:
            remaining = 0
            reset_time = int(self._requests[0] + self.window_size - now)
            return False, reset_time
        
        # 记录请求
        self._requests.append(now)
        remaining = self.max_requests - len(self._requests)
        return True, remaining
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        now = time.time()
        # 清理过期请求
        while self._requests and self._requests[0] < now - self.window_size:
            self._requests.popleft()
        
        return {
            "current_requests": len(self._requests),
            "max_requests": self.max_requests,
            "window_size": self.window_size,
            "remaining": self.max_requests - len(self._requests),
        }


class RateLimitMiddleware:
    """
    限流中间件
    
    基于滑动窗口算法实现请求限流。
    
    Attributes:
        config: 限流配置
        counters: 计数器字典 {key: counter}
    
    Example:
        >>> config = RateLimitConfig(requests_per_minute=100)
        >>> middleware = RateLimitMiddleware(config)
        >>> app.add_middleware(RateLimitMiddleware, config=config)
    """
    
    def __init__(
        self,
        app=None,  # FastAPI 自动传入
        config: Optional[RateLimitConfig] = None,
        key_func: Optional[Callable[[Request], str]] = None,
    ):
        self.app = app  # 存储ASGI应用
        self.config = config or RateLimitConfig()
        self.key_func = key_func or self._default_key_func
        self._counters: Dict[str, SlidingWindowCounter] = {}
        self._hourly_counters: Dict[str, SlidingWindowCounter] = {}
    
    def _default_key_func(self, request: Request) -> str:
        """默认键生成函数（基于客户端IP）"""
        # 获取客户端IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"{self.config.key_prefix}:{client_ip}"
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        中间件调用
        
        Args:
            request: FastAPI请求
            call_next: 下一个处理器
            
        Returns:
            Response对象
        """
        # 检查是否排除
        if self._is_excluded(request):
            return await call_next(request)
        
        # 生成限流键
        key = self.key_func(request)
        
        # 检查限流
        allowed, remaining = self._check_rate_limit(key)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for key: {key}")
            return self._create_rate_limit_response(remaining)
        
        # 处理请求
        response = await call_next(request)
        
        # 添加限流响应头
        response = self._add_headers(response, key, remaining)
        
        return response
    
    def _is_excluded(self, request: Request) -> bool:
        """检查是否应该排除限流"""
        path = request.url.path
        
        # 检查排除路径
        for excluded_path in self.config.excluded_paths:
            if path.startswith(excluded_path):
                return True
        
        # 获取客户端IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else ""
        
        # 检查排除IP
        if client_ip in self.config.excluded_ips:
            return True
        
        return False
    
    def _check_rate_limit(self, key: str) -> Tuple[bool, int]:
        """检查限流"""
        # 获取或创建计数器
        if key not in self._counters:
            self._counters[key] = SlidingWindowCounter(
                window_size=60,
                max_requests=self.config.requests_per_minute
            )
        
        if key not in self._hourly_counters:
            self._hourly_counters[key] = SlidingWindowCounter(
                window_size=3600,
                max_requests=self.config.requests_per_hour
            )
        
        # 检查分钟限制
        minute_allowed, minute_remaining = self._counters[key].is_allowed()
        if not minute_allowed:
            return False, minute_remaining
        
        # 检查小时限制
        hour_allowed, hour_remaining = self._hourly_counters[key].is_allowed()
        if not hour_allowed:
            return False, hour_remaining
        
        return True, min(minute_remaining, hour_remaining)
    
    def _create_rate_limit_response(self, retry_after: int) -> JSONResponse:
        """创建限流响应"""
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "message": "Rate limit exceeded. Please try again later.",
                "retry_after": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(self.config.requests_per_minute),
                "X-RateLimit-Remaining": "0",
            },
        )
    
    def _add_headers(self, response: Response, key: str, remaining: int) -> Response:
        """添加限流响应头"""
        response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        
        # 获取计数器统计
        if key in self._counters:
            stats = self._counters[key].get_stats()
            reset_time = int(time.time()) + 60
            response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
    
    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        获取限流统计
        
        Args:
            key: 限流键，如果为None则返回所有统计
            
        Returns:
            统计信息字典
        """
        if key:
            counter = self._counters.get(key)
            if counter:
                return {key: counter.get_stats()}
            return {}
        
        return {k: v.get_stats() for k, v in self._counters.items()}
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        重置计数器
        
        Args:
            key: 限流键，如果为None则重置所有
        """
        if key:
            if key in self._counters:
                del self._counters[key]
            if key in self._hourly_counters:
                del self._hourly_counters[key]
        else:
            self._counters.clear()
            self._hourly_counters.clear()


# 导出
__all__ = [
    "RateLimitConfig",
    "SlidingWindowCounter",
    "RateLimitMiddleware",
]
