"""
客户端负载均衡
从注册中心获取多个实例随机或轮询调用
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .schema import AgentMetadata, AgentStatus


@dataclass
class EndpointStats:
    """端点统计信息"""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    consecutive_failures: int = 0
    last_request_time: Optional[float] = None
    last_failure_time: Optional[float] = None

    @property
    def average_latency_ms(self) -> float:
        """平均延迟"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def is_healthy(self) -> bool:
        """是否健康"""
        return self.consecutive_failures < 3

    def record_success(self, latency_ms: float) -> None:
        """记录成功请求"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0
        self.last_request_time = time.time()

    def record_failure(self) -> None:
        """记录失败请求"""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LATENCY = "least_latency"
    CONSISTENT_HASH = "consistent_hash"
    HEALTH_PRIORITY = "health_priority"


class LoadBalancer(ABC):
    """负载均衡器基类"""

    def __init__(self, endpoints: Optional[List[AgentMetadata]] = None):
        self._endpoints: Dict[str, AgentMetadata] = {}
        self._stats: Dict[str, EndpointStats] = {}
        self._lock = threading.RLock()
        
        if endpoints:
            for endpoint in endpoints:
                self.add_endpoint(endpoint)

    def add_endpoint(self, endpoint: AgentMetadata) -> None:
        """添加端点"""
        with self._lock:
            self._endpoints[endpoint.agent_id] = endpoint
            if endpoint.agent_id not in self._stats:
                self._stats[endpoint.agent_id] = EndpointStats(agent_id=endpoint.agent_id)

    def remove_endpoint(self, agent_id: str) -> bool:
        """移除端点"""
        with self._lock:
            if agent_id in self._endpoints:
                del self._endpoints[agent_id]
                return True
            return False

    def update_endpoints(self, endpoints: List[AgentMetadata]) -> None:
        """更新端点列表"""
        with self._lock:
            current_ids = set(self._endpoints.keys())
            new_ids = {e.agent_id for e in endpoints}
            
            for removed_id in current_ids - new_ids:
                del self._endpoints[removed_id]
            
            for endpoint in endpoints:
                self._endpoints[endpoint.agent_id] = endpoint
                if endpoint.agent_id not in self._stats:
                    self._stats[endpoint.agent_id] = EndpointStats(agent_id=endpoint.agent_id)

    def get_healthy_endpoints(self) -> List[AgentMetadata]:
        """获取健康的端点"""
        with self._lock:
            healthy = []
            for agent_id, endpoint in self._endpoints.items():
                stats = self._stats.get(agent_id)
                if endpoint.status == AgentStatus.HEALTHY and (not stats or stats.is_healthy):
                    healthy.append(endpoint)
            return healthy

    def record_request_result(self, agent_id: str, success: bool, latency_ms: float = 0.0) -> None:
        """记录请求结果"""
        with self._lock:
            stats = self._stats.get(agent_id)
            if stats:
                if success:
                    stats.record_success(latency_ms)
                else:
                    stats.record_failure()

    def get_stats(self) -> Dict[str, EndpointStats]:
        """获取统计信息"""
        with self._lock:
            return dict(self._stats)

    @abstractmethod
    def select(self, key: Optional[str] = None) -> Optional[AgentMetadata]:
        """选择一个端点"""
        pass


class RandomLoadBalancer(LoadBalancer):
    """随机负载均衡器"""

    def select(self, key: Optional[str] = None) -> Optional[AgentMetadata]:
        with self._lock:
            endpoints = self.get_healthy_endpoints()
            if not endpoints:
                return None
            return random.choice(endpoints)


class RoundRobinLoadBalancer(LoadBalancer):
    """轮询负载均衡器"""

    def __init__(self, endpoints: Optional[List[AgentMetadata]] = None):
        super().__init__(endpoints)
        self._counter = 0

    def select(self, key: Optional[str] = None) -> Optional[AgentMetadata]:
        with self._lock:
            endpoints = self.get_healthy_endpoints()
            if not endpoints:
                return None
            selected = endpoints[self._counter % len(endpoints)]
            self._counter += 1
            return selected


class WeightedRandomLoadBalancer(LoadBalancer):
    """加权随机负载均衡器"""

    def select(self, key: Optional[str] = None) -> Optional[AgentMetadata]:
        with self._lock:
            endpoints = self.get_healthy_endpoints()
            if not endpoints:
                return None
            
            weights = []
            for endpoint in endpoints:
                stats = self._stats.get(endpoint.agent_id)
                if stats:
                    weight = max(10, stats.success_rate * 100)
                else:
                    weight = 100
                weights.append(weight)
            
            total = sum(weights)
            r = random.uniform(0, total)
            cumulative = 0
            
            for endpoint, weight in zip(endpoints, weights):
                cumulative += weight
                if r <= cumulative:
                    return endpoint
            
            return endpoints[-1]


class LeastLatencyLoadBalancer(LoadBalancer):
    """最低延迟负载均衡器"""

    def select(self, key: Optional[str] = None) -> Optional[AgentMetadata]:
        with self._lock:
            endpoints = self.get_healthy_endpoints()
            if not endpoints:
                return None
            
            if len(endpoints) == 1:
                return endpoints[0]
            
            def get_latency(endpoint: AgentMetadata) -> float:
                stats = self._stats.get(endpoint.agent_id)
                return stats.average_latency_ms if stats else float('inf')
            
            return min(endpoints, key=get_latency)


class ConsistentHashLoadBalancer(LoadBalancer):
    """一致性哈希负载均衡器"""

    def __init__(self, endpoints: Optional[List[AgentMetadata]] = None, virtual_nodes: int = 150):
        super().__init__(endpoints)
        self._virtual_nodes = virtual_nodes
        self._hash_ring: Dict[int, str] = {}
        self._rebuild_ring()

    def _rebuild_ring(self) -> None:
        """重建哈希环"""
        self._hash_ring = {}
        for agent_id in self._endpoints:
            for i in range(self._virtual_nodes):
                hash_key = self._hash(f"{agent_id}:{i}")
                self._hash_ring[hash_key] = agent_id

    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_endpoint(self, endpoint: AgentMetadata) -> None:
        super().add_endpoint(endpoint)
        self._rebuild_ring()

    def remove_endpoint(self, agent_id: str) -> bool:
        result = super().remove_endpoint(agent_id)
        if result:
            self._rebuild_ring()
        return result

    def update_endpoints(self, endpoints: List[AgentMetadata]) -> None:
        super().update_endpoints(endpoints)
        self._rebuild_ring()

    def select(self, key: Optional[str] = None) -> Optional[AgentMetadata]:
        with self._lock:
            if not self._hash_ring:
                return None
            
            if key is None:
                key = str(random.randint(0, 1000000))
            
            hash_key = self._hash(key)
            
            # 找到顺时针方向的第一个节点
            sorted_hashes = sorted(self._hash_ring.keys())
            for h in sorted_hashes:
                if h >= hash_key:
                    agent_id = self._hash_ring[h]
                    return self._endpoints.get(agent_id)
            
            # 回环到第一个节点
            agent_id = self._hash_ring[sorted_hashes[0]]
            return self._endpoints.get(agent_id)


class LoadBalancerFactory:
    """负载均衡器工厂"""

    _strategies: Dict[LoadBalanceStrategy, type] = {
        LoadBalanceStrategy.RANDOM: RandomLoadBalancer,
        LoadBalanceStrategy.ROUND_ROBIN: RoundRobinLoadBalancer,
        LoadBalanceStrategy.WEIGHTED_RANDOM: WeightedRandomLoadBalancer,
        LoadBalanceStrategy.LEAST_LATENCY: LeastLatencyLoadBalancer,
        LoadBalanceStrategy.CONSISTENT_HASH: ConsistentHashLoadBalancer,
    }

    @classmethod
    def create(
        cls,
        strategy: LoadBalanceStrategy,
        endpoints: Optional[List[AgentMetadata]] = None
    ) -> LoadBalancer:
        """创建负载均衡器"""
        balancer_class = cls._strategies.get(strategy, RandomLoadBalancer)
        return balancer_class(endpoints)

    @classmethod
    def register_strategy(
        cls,
        strategy: LoadBalanceStrategy,
        balancer_class: type
    ) -> None:
        """注册自定义策略"""
        cls._strategies[strategy] = balancer_class


class ServiceLoadBalancer:
    """
    服务级负载均衡器
    
    为每个服务维护独立的负载均衡器
    """

    def __init__(self, default_strategy: LoadBalanceStrategy = LoadBalanceStrategy.RANDOM):
        self._default_strategy = default_strategy
        self._balancers: Dict[str, LoadBalancer] = {}
        self._lock = threading.RLock()

    def get_balancer(self, service_name: str) -> LoadBalancer:
        """获取服务的负载均衡器"""
        with self._lock:
            if service_name not in self._balancers:
                self._balancers[service_name] = LoadBalancerFactory.create(
                    self._default_strategy
                )
            return self._balancers[service_name]

    def update_service_endpoints(
        self,
        service_name: str,
        endpoints: List[AgentMetadata]
    ) -> None:
        """更新服务端点"""
        balancer = self.get_balancer(service_name)
        balancer.update_endpoints(endpoints)

    def select_endpoint(
        self,
        service_name: str,
        key: Optional[str] = None
    ) -> Optional[AgentMetadata]:
        """为服务选择端点"""
        balancer = self.get_balancer(service_name)
        return balancer.select(key)

    def remove_service(self, service_name: str) -> bool:
        """移除服务"""
        with self._lock:
            if service_name in self._balancers:
                del self._balancers[service_name]
                return True
            return False

    def get_all_services(self) -> List[str]:
        """获取所有服务名"""
        with self._lock:
            return list(self._balancers.keys())
