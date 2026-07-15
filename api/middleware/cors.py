"""
CORS中间件模块

提供跨域资源共享支持。

主要组件:
    - CORSMiddleware: CORS中间件

使用示例:
    >>> from agi_unified_framework.api.middleware import CORSMiddleware
    >>> app.add_middleware(CORSMiddleware, allow_origins=["*"])
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional, Pattern, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class CORSMiddleware:
    """
    CORS中间件
    
    处理跨域资源共享请求。
    
    Attributes:
        allow_origins: 允许的源列表
        allow_methods: 允许的方法列表
        allow_headers: 允许的请求头列表
        allow_credentials: 是否允许凭证
        expose_headers: 暴露的响应头列表
        max_age: 预检缓存时间
    
    Example:
        >>> middleware = CORSMiddleware(
        ...     allow_origins=["https://example.com"],
        ...     allow_methods=["GET", "POST"],
        ...     allow_headers=["*"],
        ... )
        >>> app.add_middleware(CORSMiddleware, allow_origins=["*"])
    """
    
    def __init__(
        self,
        allow_origins: Union[List[str], str] = "*",
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = False,
        expose_headers: List[str] = None,
        max_age: int = 600,
    ):
        # 处理允许的来源
        if allow_origins == "*":
            self.allow_origins = ["*"]
            self.allow_origins_regex = None
        else:
            self.allow_origins = list(allow_origins) if isinstance(allow_origins, (list, tuple)) else [allow_origins]
            self.allow_origins_regex = None
            # 检查是否有正则表达式模式
            for origin in self.allow_origins:
                if "*" in origin:
                    # 转换为正则表达式
                    pattern = origin.replace(".", r"\.")
                    pattern = pattern.replace("*", ".*")
                    self.allow_origins_regex = re.compile(pattern)
                    break
        
        # 默认方法
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        
        # 默认请求头
        self.allow_headers = [h.lower() for h in (allow_headers or ["*"])]
        
        self.allow_credentials = allow_credentials
        self.expose_headers = expose_headers or []
        self.max_age = max_age
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        中间件调用
        
        Args:
            request: FastAPI请求
            call_next: 下一个处理器
            
        Returns:
            Response对象
        """
        origin = request.headers.get("origin")
        
        # 检查是否是预检请求
        if request.method == "OPTIONS":
            return self._handle_preflight(request, origin)
        
        # 处理实际请求
        response = await call_next(request)
        
        # 添加CORS头
        return self._add_cors_headers(response, origin)
    
    def _handle_preflight(self, request: Request, origin: Optional[str]) -> Response:
        """处理预检请求"""
        response = JSONResponse(content={}, status_code=200)
        
        # 检查来源
        if not self._is_origin_allowed(origin):
            return response
        
        # 添加CORS头
        response = self._add_cors_headers(response, origin)
        
        # 预检特定头
        requested_method = request.headers.get("access-control-request-method")
        if requested_method:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        
        requested_headers = request.headers.get("access-control-request-headers")
        if requested_headers:
            if "*" in self.allow_headers:
                response.headers["Access-Control-Allow-Headers"] = requested_headers
            else:
                response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: Optional[str]) -> Response:
        """添加CORS响应头"""
        if not origin:
            return response
        
        # 检查来源是否允许
        if not self._is_origin_allowed(origin):
            return response
        
        # 设置允许的来源
        if "*" in self.allow_origins and not self.allow_credentials:
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        
        # 设置凭证
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # 设置暴露的头
        if self.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        
        return response
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """检查来源是否允许"""
        if not origin:
            return False
        
        if "*" in self.allow_origins:
            return True
        
        if origin in self.allow_origins:
            return True
        
        if self.allow_origins_regex:
            return bool(self.allow_origins_regex.match(origin))
        
        return False


# 导出
__all__ = [
    "CORSMiddleware",
]
