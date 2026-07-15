"""
模型熔断器 (Model Circuit Breaker)

该模块实现熔断器模式，防止故障模型影响整体服务质量。功能包括：
- 失败率监控
- 半开状态探测
- 恢复自动熔合
- 阈值配置

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque

# 配置日志
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """
    熔断器状态
    """
    CLOSED = auto()    # 关闭状态 - 正常请求
    OPEN = auto()      # 打开状态 - 拒绝请求
    HALF_OPEN = auto() # 半开状态 - 探测恢复


@dataclass
class CircuitConfig:
    """
    熔断器配置
    
    Attributes:
        failure_threshold: 失败率阈值 (0-1)
        success_threshold: 成功阈值 (恢复所需)
        timeout_seconds: 超时时间 (秒)
        min_requests: 最小请求数 (用于计算失败率)
        window_seconds: 统计窗口 (秒)
    """
    failure_threshold: float = 0.5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    min_requests: int = 10
    window_seconds: float = 60.0


@dataclass
class CircuitMetrics:
    """
    熔断器指标
    
    Attributes:
        total_requests: 总请求数
        total_failures: 总失败数
        failure_rate: 当前失败率
        consecutive_failures: 连续失败数
        consecutive_successes: 连续成功数
        state: 当前状态
        last_failure_time: 最后失败时间
        last_success_time: 最后成功时间
        opened_at: 打开时间
    """
    total_requests: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "failure_rate": self.failure_rate,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "state": self.state.name,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


class CircuitBreaker:
    """
    模型熔断器
    
    Features:
        - 三态熔断机制 (关闭/打开/半开)
        - 失败率监控
        - 自动恢复探测
        - 可配置阈值
        - 状态持久化支持
    
    Example:
        ```python
        # 创建熔断器
        breaker = CircuitBreaker(
            model_id="gpt-4",
            config=CircuitConfig(
                failure_threshold=0.5,
                timeout_seconds=60
            )
        )
        
        # 检查是否可以请求
        if breaker.can_request():
            result = call_model("gpt-4", prompt)
            breaker.record_success() if result else breaker.record_failure()
        else:
            print("Circuit is open, using fallback")
        ```
    """
    
    def __init__(
        self,
        model_id: str,
        config: Optional[CircuitConfig] = None,
        on_state_change: Optional[Callable[[str, CircuitState], None]] = None
    ):
        """
        初始化熔断器。
        
        Args:
            model_id: 模型ID
            config: 熔断器配置
            on_state_change: 状态变化回调
        """
        self._model_id = model_id
        self._config = config or CircuitConfig()
        self._on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        
        self._request_history: deque = deque(maxlen=1000)
        self._state_lock = threading.RLock()
        
        self._last_state_change = datetime.now()
    
    @property
    def model_id(self) -> str:
        """获取模型ID"""
        return self._model_id
    
    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        with self._state_lock:
            self._check_state_transition()
            return self._state
    
    @property
    def metrics(self) -> CircuitMetrics:
        """获取指标"""
        with self._state_lock:
            self._update_metrics()
            return self._metrics
    
    def can_request(self) -> bool:
        """
        检查是否可以发起请求。
        
        Returns:
            是否可以请求
        """
        with self._state_lock:
            self._check_state_transition()
            return self._state != CircuitState.OPEN
    
    def record_success(self) -> None:
        """记录成功请求"""
        with self._state_lock:
            now = datetime.now()
            
            # 清理过期记录
            self._cleanup_old_records(now)
            
            # 记录成功
            self._request_history.append({
                "success": True,
                "timestamp": now
            })
            
            # 更新指标
            self._metrics.total_requests += 1
            self._metrics.consecutive_failures = 0
            self._metrics.consecutive_successes += 1
            self._metrics.last_success_time = now
            
            self._update_metrics()
            
            # 检查状态转换
            self._handle_success()
    
    def record_failure(self, error: Optional[str] = None) -> None:
        """
        记录失败请求。
        
        Args:
            error: 错误信息
        """
        with self._state_lock:
            now = datetime.now()
            
            # 清理过期记录
            self._cleanup_old_records(now)
            
            # 记录失败
            self._request_history.append({
                "success": False,
                "timestamp": now,
                "error": error
            })
            
            # 更新指标
            self._metrics.total_requests += 1
            self._metrics.total_failures += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = now
            
            self._update_metrics()
            
            # 检查状态转换
            self._handle_failure()
    
    def _cleanup_old_records(self, now: datetime) -> None:
        """清理过期记录"""
        cutoff = now - timedelta(seconds=self._config.window_seconds)
        while self._request_history and self._request_history[0]["timestamp"] < cutoff:
            old_record = self._request_history.popleft()
            # 调整计数器
            if old_record["success"]:
                self._metrics.total_requests -= 1
                self._metrics.total_requests += 1  # 保持一致性
            else:
                self._metrics.total_requests -= 1
                self._metrics.total_failures -= 1
    
    def _update_metrics(self) -> None:
        """更新指标"""
        # 计算窗口内的失败率
        window_cutoff = datetime.now() - timedelta(seconds=self._config.window_seconds)
        window_requests = [
            r for r in self._request_history
            if r["timestamp"] >= window_cutoff
        ]
        
        if len(window_requests) >= self._config.min_requests:
            failures = sum(1 for r in window_requests if not r["success"])
            self._metrics.failure_rate = failures / len(window_requests)
        else:
            # 数据不足时不更新失败率
            pass
    
    def _check_state_transition(self) -> None:
        """检查状态转换"""
        now = datetime.now()
        
        if self._state == CircuitState.OPEN:
            # 检查是否应该进入半开状态
            time_since_open = (now - self._metrics.opened_at).total_seconds()
            if time_since_open >= self._config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
        
        elif self._state == CircuitState.HALF_OPEN:
            # 在半开状态下，如果连续成功，达到阈值后关闭
            if self._metrics.consecutive_successes >= self._config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
            # 如果发生失败，立即打开
            elif self._metrics.consecutive_failures >= 1:
                self._transition_to(CircuitState.OPEN)
    
    def _handle_success(self) -> None:
        """处理成功"""
        if self._state == CircuitState.HALF_OPEN:
            if self._metrics.consecutive_successes >= self._config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
    
    def _handle_failure(self) -> None:
        """处理失败"""
        if self._state == CircuitState.CLOSED:
            # 检查是否应该打开
            if (self._metrics.failure_rate >= self._config.failure_threshold and
                len(self._request_history) >= self._config.min_requests):
                self._transition_to(CircuitState.OPEN)
        
        elif self._state == CircuitState.HALF_OPEN:
            # 在半开状态下失败，立即打开
            self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换"""
        if self._state == new_state:
            return
        
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()
        
        logger.warning(
            f"Circuit breaker for {self._model_id}: "
            f"{old_state.name} -> {new_state.name}"
        )
        
        # 更新指标
        if new_state == CircuitState.OPEN:
            self._metrics.opened_at = datetime.now()
        elif new_state == CircuitState.CLOSED:
            # 重置统计
            self._metrics.consecutive_failures = 0
            self._metrics.consecutive_successes = 0
        
        self._metrics.state = new_state
        
        # 触发回调
        if self._on_state_change:
            try:
                self._on_state_change(self._model_id, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def reset(self) -> None:
        """重置熔断器"""
        with self._state_lock:
            self._request_history.clear()
            self._metrics = CircuitMetrics()
            self._transition_to(CircuitState.CLOSED)
            logger.info(f"Circuit breaker for {self._model_id} has been reset")
    
    def force_open(self) -> None:
        """强制打开熔断器"""
        with self._state_lock:
            self._transition_to(CircuitState.OPEN)
    
    def force_close(self) -> None:
        """强制关闭熔断器"""
        with self._state_lock:
            self._transition_to(CircuitState.CLOSED)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._state_lock:
            self._update_metrics()
            return {
                "model_id": self._model_id,
                "state": self._state.name,
                "metrics": self._metrics.to_dict(),
                "config": {
                    "failure_threshold": self._config.failure_threshold,
                    "success_threshold": self._config.success_threshold,
                    "timeout_seconds": self._config.timeout_seconds,
                    "min_requests": self._config.min_requests,
                    "window_seconds": self._config.window_seconds,
                },
                "history_size": len(self._request_history),
                "last_state_change": self._last_state_change.isoformat(),
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.get_stats()


class CircuitBreakerManager:
    """
    熔断器管理器
    
    统一管理多个模型的熔断器。
    """
    
    def __init__(self):
        """初始化熔断器管理器"""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitConfig()
        self._lock = threading.RLock()
    
    def get_breaker(
        self,
        model_id: str,
        config: Optional[CircuitConfig] = None
    ) -> CircuitBreaker:
        """
        获取或创建熔断器。
        
        Args:
            model_id: 模型ID
            config: 熔断器配置
            
        Returns:
            熔断器实例
        """
        with self._lock:
            if model_id not in self._breakers:
                self._breakers[model_id] = CircuitBreaker(
                    model_id=model_id,
                    config=config or self._default_config
                )
            return self._breakers[model_id]
    
    def remove_breaker(self, model_id: str) -> bool:
        """移除熔断器"""
        with self._lock:
            if model_id in self._breakers:
                del self._breakers[model_id]
                return True
            return False
    
    def can_request(self, model_id: str) -> bool:
        """检查模型是否可以请求"""
        breaker = self._breakers.get(model_id)
        if breaker:
            return breaker.can_request()
        return True  # 未知的模型默认允许请求
    
    def record_success(self, model_id: str) -> None:
        """记录成功"""
        breaker = self._breakers.get(model_id)
        if breaker:
            breaker.record_success()
    
    def record_failure(self, model_id: str, error: Optional[str] = None) -> None:
        """记录失败"""
        breaker = self._breakers.get(model_id)
        if breaker:
            breaker.record_failure(error)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有熔断器统计"""
        with self._lock:
            return {
                model_id: breaker.get_stats()
                for model_id, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """重置所有熔断器"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def set_default_config(self, config: CircuitConfig) -> None:
        """设置默认配置"""
        self._default_config = config
    
    def get_open_circuits(self) -> List[str]:
        """获取所有打开的熔断器"""
        with self._lock:
            return [
                model_id for model_id, breaker in self._breakers.items()
                if breaker.state == CircuitState.OPEN
            ]
