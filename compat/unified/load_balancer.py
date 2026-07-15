"""
负载均衡器 (Load Balancer)

在多个推理后端实例之间分配请求，支持轮询、加权轮询、最少连接、
随机、一致性哈希等负载均衡策略。

模块路径: compat/unified/load_balancer.py
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """负载均衡策略类型。"""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    PRIORITY = "priority"


@dataclass
class BackendEndpoint:
    """后端端点配置。

    Attributes:
        name: 端点名称。
        base_url: 服务地址。
        api_key: 可选的 API 密钥。
        backend_type: 后端类型（ollama/vllm/tgi/localai/llama_cpp）。
        weight: 权重（用于加权轮询）。
        max_connections: 最大并发连接数。
        priority: 优先级（数字越小越优先）。
        enabled: 是否启用。
        metadata: 额外元数据。
    """

    name: str
    base_url: str
    api_key: Optional[str] = None
    backend_type: str = "unknown"
    weight: int = 100
    max_connections: int = 100
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """判断端点是否健康。"""
        return self.enabled


@dataclass
class EndpointStats:
    """端点运行时统计。

    Attributes:
        active_connections: 当前活跃连接数。
        total_requests: 总请求数。
        total_successes: 总成功数。
        total_failures: 总失败数。
        total_latency_ms: 总延迟（毫秒）。
        last_request_time: 上次请求时间。
        last_error: 上次错误信息。
    """

    active_connections: int = 0
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_latency_ms: float = 0.0
    last_request_time: float = 0.0
    last_error: Optional[str] = None

    @property
    def avg_latency_ms(self) -> float:
        """计算平均延迟（毫秒）。"""
        if self.total_successes == 0:
            return 0.0
        return self.total_latency_ms / self.total_successes

    @property
    def success_rate(self) -> float:
        """计算成功率。"""
        if self.total_requests == 0:
            return 1.0
        return self.total_successes / self.total_requests


class LoadBalancer:
    """负载均衡器。

    在多个后端端点之间分配推理请求，支持多种负载均衡策略。

    Args:
        strategy: 负载均衡策略。
        endpoints: 后端端点列表。
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        endpoints: Optional[List[BackendEndpoint]] = None,
    ) -> None:
        self._strategy = strategy
        self._endpoints: List[BackendEndpoint] = endpoints or []
        self._stats: Dict[str, EndpointStats] = {}
        self._round_robin_index: int = 0
        self._weighted_index: int = 0
        self._weighted_current_weight: int = 0
        self._lock = threading.Lock()

        for ep in self._endpoints:
            self._stats[ep.name] = EndpointStats()

    @property
    def strategy(self) -> LoadBalancingStrategy:
        """获取当前负载均衡策略。"""
        return self._strategy

    @strategy.setter
    def strategy(self, value: LoadBalancingStrategy) -> None:
        """设置负载均衡策略。"""
        self._strategy = value
        with self._lock:
            self._round_robin_index = 0
            self._weighted_index = 0
            self._weighted_current_weight = 0

    @property
    def endpoints(self) -> List[BackendEndpoint]:
        """获取所有端点。"""
        return list(self._endpoints)

    def add_endpoint(self, endpoint: BackendEndpoint) -> None:
        """添加后端端点。

        Args:
            endpoint: 端点配置。
        """
        self._endpoints.append(endpoint)
        self._stats[endpoint.name] = EndpointStats()
        logger.info("添加端点: %s (%s)", endpoint.name, endpoint.base_url)

    def remove_endpoint(self, name: str) -> bool:
        """移除后端端点。

        Args:
            name: 端点名称。

        Returns:
            是否成功移除。
        """
        original_len = len(self._endpoints)
        self._endpoints = [ep for ep in self._endpoints if ep.name != name]
        self._stats.pop(name, None)
        removed = len(self._endpoints) < original_len
        if removed:
            logger.info("移除端点: %s", name)
        return removed

    def enable_endpoint(self, name: str) -> None:
        """启用端点。

        Args:
            name: 端点名称。
        """
        for ep in self._endpoints:
            if ep.name == name:
                ep.enabled = True
                logger.info("启用端点: %s", name)
                return

    def disable_endpoint(self, name: str) -> None:
        """禁用端点。

        Args:
            name: 端点名称。
        """
        for ep in self._endpoints:
            if ep.name == name:
                ep.enabled = False
                logger.info("禁用端点: %s", name)
                return

    def get_healthy_endpoints(self) -> List[BackendEndpoint]:
        """获取所有健康的端点。

        Returns:
            健康端点列表。
        """
        return [ep for ep in self._endpoints if ep.is_healthy]

    def _select_round_robin(self, endpoints: List[BackendEndpoint]) -> Optional[BackendEndpoint]:
        """轮询策略选择端点。"""
        if not endpoints:
            return None
        with self._lock:
            idx = self._round_robin_index % len(endpoints)
            self._round_robin_index += 1
            return endpoints[idx]

    def _select_weighted_round_robin(self, endpoints: List[BackendEndpoint]) -> Optional[BackendEndpoint]:
        """加权轮询策略选择端点。"""
        if not endpoints:
            return None
        with self._lock:
            total_weight = sum(ep.weight for ep in endpoints)
            if total_weight == 0:
                return endpoints[0]

            if self._weighted_current_weight == 0:
                self._weighted_current_weight = total_weight

            for ep in endpoints:
                self._weighted_current_weight -= ep.weight
                if self._weighted_current_weight <= 0:
                    self._weighted_current_weight = 0
                    return ep

            self._weighted_current_weight = 0
            return endpoints[0]

    def _select_least_connections(self, endpoints: List[BackendEndpoint]) -> Optional[BackendEndpoint]:
        """最少连接策略选择端点。"""
        if not endpoints:
            return None
        def sort_key(ep: BackendEndpoint) -> tuple:
            stats = self._stats.get(ep.name)
            conn = stats.active_connections if stats else 0
            return (conn, ep.priority)
        return min(endpoints, key=sort_key)

    def _select_random(self, endpoints: List[BackendEndpoint]) -> Optional[BackendEndpoint]:
        """随机策略选择端点。"""
        if not endpoints:
            return None
        return random.choice(endpoints)

    def _select_consistent_hash(self, endpoints: List[BackendEndpoint], key: str = "") -> Optional[BackendEndpoint]:
        """一致性哈希策略选择端点。

        Args:
            endpoints: 可用端点列表。
            key: 哈希键（默认使用当前时间戳的秒数）。
        """
        if not endpoints:
            return None
        if not key:
            key = str(int(time.time()))
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return endpoints[hash_val % len(endpoints)]

    def _select_priority(self, endpoints: List[BackendEndpoint]) -> Optional[BackendEndpoint]:
        """优先级策略选择端点。"""
        if not endpoints:
            return None
        return min(endpoints, key=lambda ep: ep.priority)

    def select_endpoint(self, key: str = "") -> Optional[BackendEndpoint]:
        """根据当前策略选择一个端点。

        Args:
            key: 用于一致性哈希的键。

        Returns:
            选中的端点，或 None（无可用端点）。
        """
        healthy = self.get_healthy_endpoints()
        if not healthy:
            logger.warning("没有可用的健康端点")
            return None

        strategy_map = {
            LoadBalancingStrategy.ROUND_ROBIN: lambda: self._select_round_robin(healthy),
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: lambda: self._select_weighted_round_robin(healthy),
            LoadBalancingStrategy.LEAST_CONNECTIONS: lambda: self._select_least_connections(healthy),
            LoadBalancingStrategy.RANDOM: lambda: self._select_random(healthy),
            LoadBalancingStrategy.CONSISTENT_HASH: lambda: self._select_consistent_hash(healthy, key),
            LoadBalancingStrategy.PRIORITY: lambda: self._select_priority(healthy),
        }

        selector = strategy_map.get(self._strategy)
        if selector is None:
            logger.warning("未知策略: %s，使用轮询", self._strategy)
            return self._select_round_robin(healthy)

        return selector()

    def record_request_start(self, endpoint_name: str) -> None:
        """记录请求开始。

        Args:
            endpoint_name: 端点名称。
        """
        stats = self._stats.get(endpoint_name)
        if stats:
            stats.active_connections += 1
            stats.total_requests += 1
            stats.last_request_time = time.time()

    def record_request_success(self, endpoint_name: str, latency_ms: float) -> None:
        """记录请求成功。

        Args:
            endpoint_name: 端点名称。
            latency_ms: 请求延迟（毫秒）。
        """
        stats = self._stats.get(endpoint_name)
        if stats:
            stats.active_connections = max(0, stats.active_connections - 1)
            stats.total_successes += 1
            stats.total_latency_ms += latency_ms
            stats.last_error = None

    def record_request_failure(self, endpoint_name: str, error: str) -> None:
        """记录请求失败。

        Args:
            endpoint_name: 端点名称。
            error: 错误信息。
        """
        stats = self._stats.get(endpoint_name)
        if stats:
            stats.active_connections = max(0, stats.active_connections - 1)
            stats.total_failures += 1
            stats.last_error = error

    def get_endpoint_stats(self, endpoint_name: str) -> Optional[EndpointStats]:
        """获取端点统计信息。

        Args:
            endpoint_name: 端点名称。

        Returns:
            端点统计，或 None。
        """
        return self._stats.get(endpoint_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有端点的统计信息。

        Returns:
            端点名称到统计信息的映射。
        """
        result: Dict[str, Dict[str, Any]] = {}
        for ep in self._endpoints:
            stats = self._stats.get(ep.name)
            if stats:
                result[ep.name] = {
                    "active_connections": stats.active_connections,
                    "total_requests": stats.total_requests,
                    "total_successes": stats.total_successes,
                    "total_failures": stats.total_failures,
                    "avg_latency_ms": round(stats.avg_latency_ms, 2),
                    "success_rate": round(stats.success_rate, 4),
                    "last_error": stats.last_error,
                    "enabled": ep.enabled,
                    "weight": ep.weight,
                }
        return result

    def reset_stats(self) -> None:
        """重置所有端点统计。"""
        for stats in self._stats.values():
            stats.active_connections = 0
            stats.total_requests = 0
            stats.total_successes = 0
            stats.total_failures = 0
            stats.total_latency_ms = 0.0
            stats.last_error = None
        logger.info("所有端点统计已重置")
