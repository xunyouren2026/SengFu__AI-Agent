"""
AGI Unified Framework - Rate Limiter
基于令牌桶算法的多维度速率限制器
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    rpm: int = 60  # 每分钟请求数
    tpm: int = 100000  # 每分钟Token数
    daily_limit: int = 0  # 每日限制（0表示不限制）
    rps: int = 10  # 每秒请求数
    burst_size: int = 5  # 突发容量

    def to_dict(self) -> Dict[str, int]:
        return {
            "rpm": self.rpm,
            "tpm": self.tpm,
            "daily_limit": self.daily_limit,
            "rps": self.rps,
            "burst_size": self.burst_size,
        }


@dataclass
class RateLimitStats:
    """速率限制统计信息"""
    total_requests: int = 0
    total_tokens: int = 0
    rejected_requests: int = 0
    rejected_tokens: int = 0
    wait_time_total: float = 0.0
    current_rpm: float = 0.0
    current_tpm: float = 0.0
    daily_usage: int = 0

    @property
    def rejection_rate(self) -> float:
        return self.rejected_requests / max(self.total_requests, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "rejected_requests": self.rejected_requests,
            "rejected_tokens": self.rejected_tokens,
            "wait_time_total": round(self.wait_time_total, 3),
            "current_rpm": round(self.current_rpm, 2),
            "current_tpm": round(self.current_tpm, 2),
            "daily_usage": self.daily_usage,
            "rejection_rate": round(self.rejection_rate, 4),
        }


class TokenBucketAlgorithm:
    """
    令牌桶算法实现

    原理：
    - 桶中以固定速率生成令牌
    - 每次请求消耗令牌
    - 桶满时不再生成令牌
    - 桶空时请求需要等待或被拒绝
    """

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 令牌生成速率（每秒）
            capacity: 桶容量
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float = 30.0) -> bool:
        """
        获取令牌

        Args:
            tokens: 需要的令牌数
            blocking: 是否阻塞等待
            timeout: 最大等待时间（秒）

        Returns:
            bool: 是否成功获取令牌
        """
        deadline = time.time() + timeout if blocking else 0

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                if not blocking:
                    return False

                # 计算需要等待的时间
                deficit = tokens - self._tokens
                wait_time = deficit / self._rate

            if deadline > 0 and time.time() + wait_time > deadline:
                return False

            time.sleep(min(wait_time, 0.1))

    def try_acquire(self, tokens: int = 1) -> bool:
        """非阻塞获取令牌"""
        return self.acquire(tokens, blocking=False)

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数"""
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> int:
        return self._capacity


class SlidingWindowCounter:
    """
    滑动窗口计数器

    用于统计时间窗口内的请求/Token数量
    """

    def __init__(self, window_seconds: float = 60.0):
        self._window = window_seconds
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def add(self, count: int = 1):
        """添加计数"""
        now = time.time()
        with self._lock:
            self._timestamps.extend([now] * count)
            self._cleanup(now)

    def _cleanup(self, now: float):
        """清理过期的时间戳"""
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    @property
    def count(self) -> int:
        """当前窗口内的计数"""
        now = time.time()
        with self._lock:
            self._cleanup(now)
            return len(self._timestamps)

    @property
    def rate_per_second(self) -> float:
        """每秒速率"""
        return self.count / self._window


class RateLimiter:
    """
    多维度速率限制器

    支持：
    - 令牌桶算法（RPS限制）
    - 滑动窗口计数器（RPM/TPM限制）
    - 每日限额
    - 分维度限制（按模型、按用户等）
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self._config = config or RateLimitConfig()
        self._stats = RateLimitStats()

        # 令牌桶（RPS限制）
        self._rps_bucket = TokenBucketAlgorithm(
            rate=self._config.rps,
            capacity=self._config.burst_size,
        )

        # 滑动窗口（RPM限制）
        self._rpm_window = SlidingWindowCounter(window_seconds=60.0)

        # 滑动窗口（TPM限制）
        self._tpm_window = SlidingWindowCounter(window_seconds=60.0)

        # 每日使用量
        self._daily_usage = 0
        self._daily_reset_time = self._get_next_reset_time()

        # 分维度限制
        self._dimension_limiters: Dict[str, Dict[str, Any]] = {}
        self._dimension_lock = threading.Lock()

    def _get_next_reset_time(self) -> float:
        """获取下一次每日重置时间（UTC午夜）"""
        import datetime
        now = datetime.datetime.utcnow()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow += datetime.timedelta(days=1)
        return tomorrow.timestamp()

    def _check_daily_reset(self):
        """检查是否需要重置每日限额"""
        if time.time() >= self._daily_reset_time:
            self._daily_usage = 0
            self._daily_reset_time = self._get_next_reset_time()

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float = 30.0) -> bool:
        """
        获取令牌（综合所有限制维度）

        Args:
            tokens: 请求的Token数
            blocking: 是否阻塞等待
            timeout: 最大等待时间

        Returns:
            bool: 是否成功获取
        """
        self._stats.total_requests += 1
        self._stats.total_tokens += tokens

        # 检查每日限额
        self._check_daily_reset()
        if self._config.daily_limit > 0:
            if self._daily_usage + tokens > self._config.daily_limit:
                self._stats.rejected_requests += 1
                self._stats.rejected_tokens += tokens
                return False

        # 检查RPM限制
        if self._rpm_window.count + 1 > self._config.rpm:
            if not blocking:
                self._stats.rejected_requests += 1
                self._stats.rejected_tokens += tokens
                return False
            # 等待直到RPM窗口有空间
            wait_start = time.time()
            while self._rpm_window.count + 1 > self._config.rpm:
                if time.time() - wait_start > timeout:
                    self._stats.rejected_requests += 1
                    self._stats.rejected_tokens += tokens
                    return False
                time.sleep(0.1)
            self._stats.wait_time_total += time.time() - wait_start

        # 检查TPM限制
        if self._tpm_window.count + tokens > self._config.tpm:
            if not blocking:
                self._stats.rejected_requests += 1
                self._stats.rejected_tokens += tokens
                return False
            wait_start = time.time()
            while self._tpm_window.count + tokens > self._config.tpm:
                if time.time() - wait_start > timeout:
                    self._stats.rejected_requests += 1
                    self._stats.rejected_tokens += tokens
                    return False
                time.sleep(0.1)
            self._stats.wait_time_total += time.time() - wait_start

        # 获取RPS令牌
        if not self._rps_bucket.acquire(1, blocking=blocking, timeout=timeout):
            self._stats.rejected_requests += 1
            self._stats.rejected_tokens += tokens
            return False

        # 记录使用量
        self._rpm_window.add(1)
        self._tpm_window.add(tokens)
        self._daily_usage += tokens
        self._stats.daily_usage = self._daily_usage

        return True

    def wait(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """
        等待可用令牌

        Args:
            tokens: 需要的Token数
            timeout: 最大等待时间

        Returns:
            bool: 是否成功获取
        """
        return self.acquire(tokens, blocking=True, timeout=timeout)

    def try_acquire(self, tokens: int = 1) -> bool:
        """非阻塞获取令牌"""
        return self.acquire(tokens, blocking=False, timeout=0)

    def add_dimension(
        self,
        dimension: str,
        key: str,
        rpm: int = 60,
        tpm: int = 100000,
    ) -> None:
        """
        添加分维度限制

        Args:
            dimension: 维度名称（如"model", "user"）
            key: 维度值（如"gpt-4", "user_123"）
            rpm: 每分钟请求限制
            tpm: 每分钟Token限制
        """
        with self._dimension_lock:
            dim_key = f"{dimension}:{key}"
            self._dimension_limiters[dim_key] = {
                "rpm_window": SlidingWindowCounter(window_seconds=60.0),
                "tpm_window": SlidingWindowCounter(window_seconds=60.0),
                "rpm_limit": rpm,
                "tpm_limit": tpm,
            }

    def acquire_dimension(
        self,
        dimension: str,
        key: str,
        tokens: int = 1,
        blocking: bool = True,
        timeout: float = 30.0,
    ) -> bool:
        """
        分维度获取令牌

        Args:
            dimension: 维度名称
            key: 维度值
            tokens: Token数
            blocking: 是否阻塞
            timeout: 超时时间

        Returns:
            bool: 是否成功
        """
        dim_key = f"{dimension}:{key}"

        with self._dimension_lock:
            limiter = self._dimension_limiters.get(dim_key)
            if limiter is None:
                return True  # 无限制

        # 检查RPM
        if limiter["rpm_window"].count + 1 > limiter["rpm_limit"]:
            if not blocking:
                return False
            wait_start = time.time()
            while limiter["rpm_window"].count + 1 > limiter["rpm_limit"]:
                if time.time() - wait_start > timeout:
                    return False
                time.sleep(0.1)

        # 检查TPM
        if limiter["tpm_window"].count + tokens > limiter["tpm_limit"]:
            if not blocking:
                return False
            wait_start = time.time()
            while limiter["tpm_window"].count + tokens > limiter["tpm_limit"]:
                if time.time() - wait_start > timeout:
                    return False
                time.sleep(0.1)

        limiter["rpm_window"].add(1)
        limiter["tpm_window"].add(tokens)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self._stats.current_rpm = self._rpm_window.rate_per_second * 60
        self._stats.current_tpm = self._tpm_window.rate_per_second * 60
        return {
            **self._stats.to_dict(),
            "rps_available": round(self._rps_bucket.available_tokens, 2),
            "rpm_current": self._rpm_window.count,
            "tpm_current": self._tpm_window.count,
            "daily_limit": self._config.daily_limit,
            "daily_remaining": max(0, self._config.daily_limit - self._daily_usage) if self._config.daily_limit > 0 else -1,
            "config": self._config.to_dict(),
        }

    def reset(self) -> None:
        """重置所有限制器"""
        self._rps_bucket = TokenBucketAlgorithm(
            rate=self._config.rps,
            capacity=self._config.burst_size,
        )
        self._rpm_window = SlidingWindowCounter(window_seconds=60.0)
        self._tpm_window = SlidingWindowCounter(window_seconds=60.0)
        self._daily_usage = 0
        self._daily_reset_time = self._get_next_reset_time()
        self._stats = RateLimitStats()

    def update_config(self, config: RateLimitConfig) -> None:
        """更新限制配置"""
        self._config = config
        self._rps_bucket = TokenBucketAlgorithm(
            rate=config.rps,
            capacity=config.burst_size,
        )
