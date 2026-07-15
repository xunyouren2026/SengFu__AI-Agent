"""
日志中间件模块

提供请求日志记录和响应时间统计。

主要组件:
    - LoggingMiddleware: 日志中间件
    - RequestLog: 请求日志数据类

使用示例:
    >>> from agi_unified_framework.api.middleware import LoggingMiddleware
    >>> app.add_middleware(LoggingMiddleware)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response

logger = logging.getLogger(__name__)


@dataclass
class RequestLog:
    """
    请求日志数据类
    
    Attributes:
        request_id: 请求ID
        method: HTTP方法
        path: 请求路径
        query_params: 查询参数
        client_ip: 客户端IP
        user_agent: 用户代理
        start_time: 开始时间
        end_time: 结束时间
        duration_ms: 处理时间（毫秒）
        status_code: 响应状态码
        response_size: 响应大小
        error: 错误信息
    """
    request_id: str
    method: str
    path: str
    query_params: str = ""
    client_ip: str = ""
    user_agent: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status_code: int = 0
    response_size: int = 0
    error: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "query_params": self.query_params,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "status_code": self.status_code,
            "response_size": self.response_size,
            "error": self.error,
            "user_id": self.user_id,
        }


class LoggingMiddleware:
    """
    日志中间件
    
    记录请求信息和响应时间。
    
    Attributes:
        app: ASGI应用（由FastAPI自动传入）
        logger: 日志记录器
        log_level: 日志级别
        exclude_paths: 排除路径
    
    Example:
        >>> middleware = LoggingMiddleware(app)
        >>> app.add_middleware(LoggingMiddleware)
    """
    
    def __init__(
        self,
        app=None,  # FastAPI 自动传入
        logger_name: str = "api.request",
        log_level: int = logging.INFO,
        exclude_paths: Optional[list] = None,
    ):
        self.app = app  # 存储ASGI应用
        self.logger = logging.getLogger(logger_name) if isinstance(logger_name, str) else logging.getLogger("api.request")
        self.log_level = log_level if isinstance(log_level, int) else logging.INFO
        self.exclude_paths = set(exclude_paths or ["/health", "/docs", "/redoc", "/openapi.json"])
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        中间件调用
        
        Args:
            request: FastAPI请求
            call_next: 下一个处理器
            
        Returns:
            Response对象
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # 创建日志对象
        log = RequestLog(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            start_time=time.time(),
        )
        
        # 获取用户信息
        if hasattr(request.state, "user") and request.state.user:
            log.user_id = request.state.user.id
        
        # 记录请求开始
        if not self._should_exclude(request.url.path):
            self.logger.info(f"[{request_id}] {request.method} {request.url.path} - Started")
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 更新日志信息
            log.end_time = time.time()
            log.duration_ms = (log.end_time - log.start_time) * 1000
            log.status_code = response.status_code
            log.response_size = int(response.headers.get("content-length", 0))
            
            # 添加请求ID到响应头
            response.headers["X-Request-ID"] = request_id
            
            # 记录日志
            self._log_response(log)
            
            return response
        
        except Exception as e:
            # 记录错误
            log.end_time = time.time()
            log.duration_ms = (log.end_time - log.start_time) * 1000
            log.error = str(e)
            
            self.logger.error(f"[{request_id}] Error processing request: {e}")
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _should_exclude(self, path: str) -> bool:
        """检查是否应该排除日志"""
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return True
        return False
    
    def _log_response(self, log: RequestLog) -> None:
        """记录响应日志"""
        if self._should_exclude(log.path):
            return
        
        # 根据状态码选择日志级别
        if log.status_code >= 500:
            level = logging.ERROR
        elif log.status_code >= 400:
            level = logging.WARNING
        else:
            level = self.log_level
        
        # 构建日志消息
        message = (
            f"[{log.request_id}] {log.method} {log.path} - "
            f"{log.status_code} ({log.duration_ms:.2f}ms)"
        )
        
        if log.error:
            message += f" - Error: {log.error}"
        
        self.logger.log(level, message)
        
        # 记录详细日志（DEBUG级别）
        self.logger.debug(f"Request details: {log.to_dict()}")


# 导出
__all__ = [
    "RequestLog",
    "LoggingMiddleware",
]
