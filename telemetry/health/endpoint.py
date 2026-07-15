"""
Health Endpoint Module

健康检查端点实现，提供/health接口和依赖检查功能。
"""

from __future__ import annotations

import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..config import HealthConfig

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    
    @property
    def is_healthy(self) -> bool:
        """是否健康"""
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    @property
    def http_status_code(self) -> int:
        """HTTP状态码"""
        codes = {
            HealthStatus.HEALTHY: 200,
            HealthStatus.DEGRADED: 200,
            HealthStatus.UNHEALTHY: 503,
            HealthStatus.UNKNOWN: 503
        }
        return codes[self]


@dataclass
class HealthResult:
    """
    健康检查结果
    
    Attributes:
        name: 检查名称
        status: 状态
        message: 消息
        response_time_ms: 响应时间（毫秒）
        timestamp: 时间戳
        metadata: 元数据
    """
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": round(self.response_time_ms, 2),
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class HealthCheck(ABC):
    """健康检查抽象基类"""
    
    def __init__(self, name: str, timeout_ms: float = 5000.0):
        self.name = name
        self.timeout_ms = timeout_ms
    
    @abstractmethod
    def check(self) -> HealthResult:
        """执行检查"""
        pass


class DependencyCheck(HealthCheck):
    """
    依赖检查
    
    检查外部依赖（数据库、缓存等）的健康状态。
    """
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout_ms: float = 5000.0
    ):
        super().__init__(name, timeout_ms)
        self._check_func = check_func
    
    def check(self) -> HealthResult:
        """执行依赖检查"""
        start = time.time()
        
        try:
            healthy = self._check_func()
            elapsed = (time.time() - start) * 1000
            
            if healthy:
                return HealthResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"{self.name} is healthy",
                    response_time_ms=elapsed
                )
            else:
                return HealthResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"{self.name} check failed",
                    response_time_ms=elapsed
                )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return HealthResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.name} check error: {str(e)}",
                response_time_ms=elapsed
            )


class HealthEndpoint:
    """
    健康检查端点
    
    提供/health接口和依赖检查功能。
    
    Example:
        >>> health = HealthEndpoint(config)
        >>> 
        >>> # Add checks
        >>> health.add_check(DependencyCheck("database", check_db))
        >>> health.add_check(DependencyCheck("cache", check_cache))
        >>> 
        >>> # Check health
        >>> result = health.check()
        >>> print(result.status)
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        """
        初始化健康检查端点
        
        Args:
            config: 健康检查配置
        """
        self._config = config or HealthConfig()
        self._checks: List[HealthCheck] = []
        self._lock = threading.Lock()
        self._last_results: Dict[str, HealthResult] = {}
        self._running = False
    
    def start(self) -> None:
        """启动健康检查服务"""
        self._running = True
        logger.info("HealthEndpoint started")
    
    def stop(self) -> None:
        """停止健康检查服务"""
        self._running = False
        logger.info("HealthEndpoint stopped")
    
    def add_check(self, check: HealthCheck) -> None:
        """
        添加健康检查
        
        Args:
            check: 健康检查实例
        """
        with self._lock:
            self._checks.append(check)
    
    def remove_check(self, name: str) -> bool:
        """
        移除健康检查
        
        Args:
            name: 检查名称
            
        Returns:
            是否成功移除
        """
        with self._lock:
            for i, check in enumerate(self._checks):
                if check.name == name:
                    self._checks.pop(i)
                    return True
            return False
    
    def check(self) -> HealthResult:
        """
        执行所有健康检查
        
        Returns:
            整体健康结果
        """
        with self._lock:
            checks = self._checks.copy()
        
        if not checks:
            return HealthResult(
                name="health",
                status=HealthStatus.HEALTHY,
                message="No checks configured"
            )
        
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for check in checks:
            try:
                result = check.check()
                results.append(result)
                self._last_results[check.name] = result
                
                # Update overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                logger.error(f"Health check {check.name} failed: {e}")
                error_result = HealthResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}"
                )
                results.append(error_result)
                self._last_results[check.name] = error_result
                overall_status = HealthStatus.UNHEALTHY
        
        # Determine message
        unhealthy_count = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
        if unhealthy_count == 0:
            message = "All checks passed"
        elif unhealthy_count == len(results):
            message = "All checks failed"
        else:
            message = f"{unhealthy_count}/{len(results)} checks failed"
        
        return HealthResult(
            name="health",
            status=overall_status,
            message=message,
            metadata={
                "checks": [r.to_dict() for r in results],
                "check_count": len(results),
                "unhealthy_count": unhealthy_count
            }
        )
    
    def get_last_results(self) -> Dict[str, HealthResult]:
        """获取上次检查结果"""
        return self._last_results.copy()
    
    def to_http_response(self) -> tuple:
        """
        转换为HTTP响应
        
        Returns:
            (status_code, body_dict)
        """
        result = self.check()
        return result.status.http_status_code, result.to_dict()


from typing import Optional
