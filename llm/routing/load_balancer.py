"""
模型负载均衡器 (Model Load Balancer)

该模块提供多种负载均衡策略，支持：
- 轮询/加权/最少连接策略
- 模型可用性检测
- 动态权重调整
- 健康检查集成

核心功能：
1. 多种负载均衡算法
2. 模型健康监控
3. 动态权重调整
4. 连接池管理
5. 故障转移

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, TypeVar
)
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import heapq

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class BalanceStrategy(Enum):
    """
    负载均衡策略枚举
    """
    # 基础策略
    ROUND_ROBIN = auto()          # 轮询
    WEIGHTED_ROUND_ROBIN = auto() # 加权轮询
    RANDOM = auto()              # 随机
    WEIGHTED_RANDOM = auto()      # 加权随机
    
    # 连接导向策略
    LEAST_CONNECTIONS = auto()    # 最少连接
    LEAST_RESPONSE_TIME = auto()  # 最短响应时间
    LEAST_LOAD = auto()          # 最低负载
    
    # 智能策略
    ADAPTIVE = auto()            # 自适应
    CONSISTENT_HASH = auto()      # 一致性哈希
    PRIORITY = auto()            # 优先级
    
    # 成本导向策略
    COST_AWARE = auto()          # 成本感知
    QUALITY_AWARE = auto()        # 质量感知


class HealthStatus(Enum):
    """
    模型健康状态
    """
    HEALTHY = auto()             # 健康
    DEGRADED = auto()            # 降级
    UNHEALTHY = auto()           # 不健康
    MAINTENANCE = auto()         # 维护中
    UNKNOWN = auto()            # 未知


@dataclass
class ModelInstance:
    """
    模型实例
    
    代表一个可用的模型实例或部署点。
    
    Attributes:
        instance_id: 实例ID
        model_id: 模型ID
        endpoint: 端点地址
        region: 区域
        weight: 权重 (用于加权策略)
        max_connections: 最大连接数
        current_connections: 当前连接数
        health_status: 健康状态
        avg_response_time_ms: 平均响应时间
        total_requests: 总请求数
        failed_requests: 失败请求数
        last_health_check: 最后健康检查时间
        metadata: 其他元数据
    """
    instance_id: str
    model_id: str
    endpoint: Optional[str] = None
    region: str = "default"
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    health_status: HealthStatus = HealthStatus.HEALTHY
    avg_response_time_ms: float = 1000.0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_available(self) -> bool:
        """是否可用"""
        return (
            self.health_status == HealthStatus.HEALTHY or
            self.health_status == HealthStatus.DEGRADED
        ) and self.current_connections < self.max_connections
    
    @property
    def failure_rate(self) -> float:
        """失败率"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return 1.0 - self.failure_rate
    
    def record_request_start(self) -> None:
        """记录请求开始"""
        self.current_connections += 1
        self.total_requests += 1
    
    def record_request_end(
        self,
        success: bool,
        response_time_ms: Optional[float] = None
    ) -> None:
        """记录请求结束"""
        if self.current_connections > 0:
            self.current_connections -= 1
        
        if not success:
            self.failed_requests += 1
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # 更新平均响应时间
        if response_time_ms is not None:
            alpha = 0.2  # EMA系数
            self.avg_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.avg_response_time_ms
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "instance_id": self.instance_id,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "region": self.region,
            "weight": self.weight,
            "max_connections": self.max_connections,
            "current_connections": self.current_connections,
            "health_status": self.health_status.name,
            "avg_response_time_ms": self.avg_response_time_ms,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "failure_rate": self.failure_rate,
            "success_rate": self.success_rate,
            "is_available": self.is_available,
        }


@dataclass
class LoadBalanceResult:
    """
    负载均衡结果
    
    Attributes:
        selected_instance: 选中的实例
        strategy_used: 使用的策略
        reason: 选择原因
        alternative_instances: 备选实例列表
    """
    selected_instance: Optional[ModelInstance]
    strategy_used: BalanceStrategy
    reason: str
    alternative_instances: List[ModelInstance] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "selected_instance": self.selected_instance.to_dict() if self.selected_instance else None,
            "strategy_used": self.strategy_used.name,
            "reason": self.reason,
            "alternative_count": len(self.alternative_instances),
        }


@dataclass
class HealthCheckConfig:
    """
    健康检查配置
    """
    enabled: bool = True
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    failure_threshold: int = 3        # 连续失败次数阈值
    recovery_threshold: int = 2       # 恢复需要的成功次数
    degraded_threshold: float = 0.3   # 降级失败率阈值


class HealthChecker:
    """
    健康检查器
    
    定期检查模型实例的健康状态。
    """
    
    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
        check_func: Optional[Callable[[ModelInstance], bool]] = None
    ):
        """
        初始化健康检查器。
        
        Args:
            config: 健康检查配置
            check_func: 自定义检查函数
        """
        self._config = config or HealthCheckConfig()
        self._check_func = check_func
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self, instances: Dict[str, ModelInstance]) -> None:
        """启动健康检查"""
        if self._running:
            return
        
        self._running = True
        self._instances = instances
        self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._thread.start()
        logger.info("Health checker started")
    
    def stop(self) -> None:
        """停止健康检查"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Health checker stopped")
    
    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(self._config.interval_seconds)
    
    def _perform_health_checks(self) -> None:
        """执行健康检查"""
        with self._lock:
            instances = list(self._instances.values())
        
        for instance in instances:
            try:
                is_healthy = self._check_instance(instance)
                
                with self._lock:
                    if is_healthy:
                        self._handle_healthy(instance)
                    else:
                        self._handle_unhealthy(instance)
                        
            except Exception as e:
                logger.error(f"Failed to check instance {instance.instance_id}: {e}")
                with self._lock:
                    instance.consecutive_failures += 1
                    self._handle_unhealthy(instance)
    
    def _check_instance(self, instance: ModelInstance) -> bool:
        """检查单个实例"""
        # 使用自定义检查函数
        if self._check_func:
            return self._check_func(instance)
        
        # 默认检查：基于失败率
        if instance.failure_rate > self._config.degraded_threshold:
            return False
        
        if instance.consecutive_failures >= self._config.failure_threshold:
            return False
        
        return True
    
    def _handle_healthy(self, instance: ModelInstance) -> None:
        """处理健康实例"""
        if instance.health_status == HealthStatus.UNHEALTHY:
            # 尝试恢复
            if instance.consecutive_failures < self._config.recovery_threshold:
                instance.health_status = HealthStatus.DEGRADED
                logger.info(f"Instance {instance.instance_id} transitioning to DEGRADED")
        elif instance.health_status == HealthStatus.DEGRADED:
            if instance.consecutive_failures == 0 and instance.failure_rate < 0.1:
                instance.health_status = HealthStatus.HEALTHY
                logger.info(f"Instance {instance.instance_id} recovered to HEALTHY")
    
    def _handle_unhealthy(self, instance: ModelInstance) -> None:
        """处理不健康实例"""
        if instance.consecutive_failures >= self._config.failure_threshold:
            if instance.health_status != HealthStatus.UNHEALTHY:
                instance.health_status = HealthStatus.UNHEALTHY
                logger.warning(
                    f"Instance {instance.instance_id} marked as UNHEALTHY "
                    f"({instance.consecutive_failures} consecutive failures)"
                )
        elif instance.health_status == HealthStatus.HEALTHY:
            if instance.failure_rate > self._config.degraded_threshold:
                instance.health_status = HealthStatus.DEGRADED
                logger.warning(
                    f"Instance {instance.instance_id} degraded due to high failure rate"
                )


class LoadBalancerStrategy(ABC):
    """负载均衡策略基类"""
    
    @abstractmethod
    def select(
        self,
        instances: List[ModelInstance]
    ) -> Optional[ModelInstance]:
        """
        选择一个实例
        
        Args:
            instances: 可用实例列表
            
        Returns:
            选中的实例
        """
        pass
    
    @abstractmethod
    def name(self) -> BalanceStrategy:
        """获取策略名称"""
        pass


class RoundRobinStrategy(LoadBalancerStrategy):
    """轮询策略"""
    
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        available = [i for i in instances if i.is_available]
        if not available:
            return None
        
        with self._lock:
            index = self._counter % len(available)
            self._counter += 1
            return available[index]
    
    def name(self) -> BalanceStrategy:
        return BalanceStrategy.ROUND_ROBIN


class WeightedRoundRobinStrategy(LoadBalancerStrategy):
    """加权轮询策略"""
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._global_lock = threading.Lock()
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        available = [i for i in instances if i.is_available]
        if not available:
            return None
        
        # 按权重分组
        weighted_instances = []
        for i in available:
            # 权重越高，被选中的概率越高
            for _ in range(int(i.weight * 10)):
                weighted_instances.append(i)
        
        if not weighted_instances:
            return available[0]
        
        with self._global_lock:
            index = sum(self._counters.values()) % len(weighted_instances)
            for iid, lock in self._locks.items():
                with lock:
                    self._counters[iid] = (self._counters[iid] + 1) % 10
            
            return weighted_instances[index]
    
    def name(self) -> BalanceStrategy:
        return BalanceStrategy.WEIGHTED_ROUND_ROBIN


class LeastConnectionsStrategy(LoadBalancerStrategy):
    """最少连接策略"""
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        available = [i for i in instances if i.is_available]
        if not available:
            return None
        
        # 选择连接数最少的实例
        return min(available, key=lambda i: i.current_connections / max(i.max_connections, 1))
    
    def name(self) -> BalanceStrategy:
        return BalanceStrategy.LEAST_CONNECTIONS


class LeastResponseTimeStrategy(LoadBalancerStrategy):
    """最短响应时间策略"""
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        available = [i for i in instances if i.is_available]
        if not available:
            return None
        
        # 计算综合得分：响应时间 + 连接压力
        def score(i: ModelInstance) -> float:
            # 归一化响应时间 (越低越好)
            response_score = i.avg_response_time_ms / 5000.0
            # 连接压力 (越高越不好)
            connection_penalty = i.current_connections / max(i.max_connections, 1)
            return response_score + connection_penalty
        
        return min(available, key=score)
    
    def name(self) -> BalanceStrategy:
        return BalanceStrategy.LEAST_RESPONSE_TIME


class AdaptiveStrategy(LoadBalancerStrategy):
    """自适应策略"""
    
    def __init__(self):
        self._scores: Dict[str, float] = defaultdict(lambda: 1.0)
        self._lock = threading.Lock()
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        available = [i for i in instances if i.is_available]
        if not available:
            return None
        
        # 计算自适应得分
        for i in available:
            with self._lock:
                # 综合考虑成功率、响应时间、连接数
                success_weight = 0.5
                latency_weight = 0.3
                connection_weight = 0.2
                
                success_score = i.success_rate
                latency_score = max(0, 1 - i.avg_response_time_ms / 5000.0)
                connection_score = max(0, 1 - i.current_connections / max(i.max_connections, 1))
                
                self._scores[i.instance_id] = (
                    success_weight * success_score +
                    latency_weight * latency_score +
                    connection_weight * connection_score
                )
        
        # 选择得分最高的
        with self._lock:
            if not self._scores:
                return available[0]
            
            best_instance = max(available, key=lambda i: self._scores.get(i.instance_id, 0))
            return best_instance
    
    def name(self) -> BalanceStrategy:
        return BalanceStrategy.ADAPTIVE
    
    def update_score(self, instance_id: str, success: bool, response_time_ms: float) -> None:
        """更新实例得分"""
        with self._lock:
            current = self._scores.get(instance_id, 1.0)
            # 简单移动平均
            if success:
                self._scores[instance_id] = current * 0.9 + 0.1
            else:
                self._scores[instance_id] = current * 0.7


class CostAwareStrategy(LoadBalancerStrategy):
    """成本感知策略"""
    
    def __init__(self, model_costs: Optional[Dict[str, float]] = None):
        self._model_costs = model_costs or {}
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        available = [i for i in instances if i.is_available]
        if not available:
            return None
        
        # 计算成本效益得分
        def cost_efficiency(i: ModelInstance) -> float:
            cost = self._model_costs.get(i.model_id, 0.01)
            # 质量/成本比率
            quality = i.success_rate * max(0.5, 1 - i.avg_response_time_ms / 10000)
            return quality / cost
        
        return max(available, key=cost_efficiency)
    
    def name(self) -> BalanceStrategy:
        return BalanceStrategy.COST_AWARE


class LoadBalancer:
    """
    模型负载均衡器
    
    Features:
        - 多种负载均衡策略
        - 模型健康监控
        - 动态权重调整
        - 连接池管理
        - 故障转移
        - 统计和监控
    
    Example:
        ```python
        # 创建负载均衡器
        lb = LoadBalancer(strategy=BalanceStrategy.LEAST_CONNECTIONS)
        
        # 注册模型实例
        lb.register_instance(ModelInstance(
            instance_id="gpt4-us-east",
            model_id="gpt-4",
            endpoint="https://api.openai.com",
            region="us-east",
            weight=1.0
        ))
        
        # 选择实例
        result = lb.select_instance("gpt-4")
        print(f"Selected: {result.selected_instance.instance_id}")
        
        # 记录请求结果
        lb.record_result(
            result.selected_instance.instance_id,
            success=True,
            response_time_ms=1500
        )
        ```
    """
    
    def __init__(
        self,
        strategy: BalanceStrategy = BalanceStrategy.ROUND_ROBIN,
        model_costs: Optional[Dict[str, float]] = None,
        health_check_config: Optional[HealthCheckConfig] = None
    ):
        """
        初始化负载均衡器。
        
        Args:
            strategy: 负载均衡策略
            model_costs: 模型成本映射
            health_check_config: 健康检查配置
        """
        self._instances: Dict[str, ModelInstance] = {}
        self._model_instances: Dict[str, List[str]] = defaultdict(list)
        self._strategy = strategy
        self._model_costs = model_costs or {}
        self._health_checker = HealthChecker(health_check_config)
        
        # 策略实例
        self._strategies: Dict[BalanceStrategy, LoadBalancerStrategy] = {
            BalanceStrategy.ROUND_ROBIN: RoundRobinStrategy(),
            BalanceStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy(),
            BalanceStrategy.LEAST_CONNECTIONS: LeastConnectionsStrategy(),
            BalanceStrategy.LEAST_RESPONSE_TIME: LeastResponseTimeStrategy(),
            BalanceStrategy.ADAPTIVE: AdaptiveStrategy(),
            BalanceStrategy.COST_AWARE: CostAwareStrategy(self._model_costs),
        }
        
        self._current_strategy = self._get_strategy_instance(strategy)
        self._lock = threading.RLock()
        self._selection_stats: Dict[str, int] = defaultdict(int)
        
        # 启动健康检查
        self._health_checker.start(self._instances)
    
    def _get_strategy_instance(self, strategy: BalanceStrategy) -> LoadBalancerStrategy:
        """获取策略实例"""
        if strategy in self._strategies:
            return self._strategies[strategy]
        
        # 默认使用轮询
        return self._strategies[BalanceStrategy.ROUND_ROBIN]
    
    def register_instance(self, instance: ModelInstance) -> bool:
        """
        注册模型实例。
        
        Args:
            instance: 模型实例
            
        Returns:
            是否注册成功
        """
        with self._lock:
            if instance.instance_id in self._instances:
                logger.warning(f"Instance {instance.instance_id} already registered")
                return False
            
            self._instances[instance.instance_id] = instance
            self._model_instances[instance.model_id].append(instance.instance_id)
            
            logger.info(
                f"Registered instance {instance.instance_id} "
                f"for model {instance.model_id}"
            )
            return True
    
    def unregister_instance(self, instance_id: str) -> bool:
        """
        注销模型实例。
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否注销成功
        """
        with self._lock:
            if instance_id not in self._instances:
                return False
            
            instance = self._instances[instance_id]
            model_id = instance.model_id
            
            del self._instances[instance_id]
            self._model_instances[model_id].remove(instance_id)
            
            logger.info(f"Unregistered instance {instance_id}")
            return True
    
    def update_instance(self, instance_id: str, **kwargs) -> bool:
        """
        更新实例配置。
        
        Args:
            instance_id: 实例ID
            **kwargs: 要更新的字段
            
        Returns:
            是否更新成功
        """
        with self._lock:
            if instance_id not in self._instances:
                return False
            
            instance = self._instances[instance_id]
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            return True
    
    def get_instance(self, instance_id: str) -> Optional[ModelInstance]:
        """获取实例"""
        return self._instances.get(instance_id)
    
    def get_instances_for_model(self, model_id: str) -> List[ModelInstance]:
        """获取模型的所有实例"""
        with self._lock:
            instance_ids = self._model_instances.get(model_id, [])
            return [
                self._instances[iid] 
                for iid in instance_ids 
                if iid in self._instances
            ]
    
    def select_instance(
        self,
        model_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        strategy: Optional[BalanceStrategy] = None
    ) -> LoadBalanceResult:
        """
        选择一个实例。
        
        Args:
            model_id: 模型ID (如果为None则选择任意模型)
            instance_id: 特定实例ID (强制选择)
            strategy: 覆盖默认策略
            
        Returns:
            负载均衡结果
        """
        with self._lock:
            # 如果指定了特定实例
            if instance_id:
                instance = self._instances.get(instance_id)
                if instance and instance.is_available:
                    self._selection_stats[instance_id] += 1
                    return LoadBalanceResult(
                        selected_instance=instance,
                        strategy_used=self._strategy,
                        reason=f"Forced selection: {instance_id}"
                    )
            
            # 获取可用实例列表
            if model_id:
                instances = self.get_instances_for_model(model_id)
            else:
                instances = list(self._instances.values())
            
            available = [i for i in instances if i.is_available]
            
            if not available:
                return LoadBalanceResult(
                    selected_instance=None,
                    strategy_used=self._strategy,
                    reason="No available instances"
                )
            
            # 使用策略选择
            strat = self._get_strategy_instance(strategy) if strategy else self._current_strategy
            selected = strat.select(available)
            
            if selected:
                self._selection_stats[selected.instance_id] += 1
            
            # 生成备选列表
            alternatives = [i for i in available if i != selected][:3]
            
            return LoadBalanceResult(
                selected_instance=selected,
                strategy_used=strat.name(),
                reason=f"Selected by {strat.name().name} strategy",
                alternative_instances=alternatives
            )
    
    def record_result(
        self,
        instance_id: str,
        success: bool,
        response_time_ms: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        记录请求结果。
        
        Args:
            instance_id: 实例ID
            success: 是否成功
            response_time_ms: 响应时间
            error: 错误信息
        """
        with self._lock:
            instance = self._instances.get(instance_id)
            if not instance:
                return
            
            instance.record_request_end(success, response_time_ms)
            
            # 如果是自适应策略，更新得分
            if isinstance(self._current_strategy, AdaptiveStrategy):
                self._current_strategy.update_score(
                    instance_id, success, response_time_ms or 1000
                )
            
            if success:
                logger.debug(
                    f"Request to {instance_id} completed successfully "
                    f"in {response_time_ms}ms"
                )
            else:
                logger.warning(f"Request to {instance_id} failed: {error}")
    
    def set_strategy(self, strategy: BalanceStrategy) -> None:
        """
        设置负载均衡策略。
        
        Args:
            strategy: 策略类型
        """
        with self._lock:
            self._strategy = strategy
            self._current_strategy = self._get_strategy_instance(strategy)
            logger.info(f"Load balancer strategy changed to {strategy.name}")
    
    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        动态更新实例权重。
        
        Args:
            weights: 实例ID到权重的映射
        """
        with self._lock:
            for instance_id, weight in weights.items():
                if instance_id in self._instances:
                    self._instances[instance_id].weight = weight
            logger.info(f"Updated weights for {len(weights)} instances")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_requests = sum(i.total_requests for i in self._instances.values())
            total_failures = sum(i.failed_requests for i in self._instances.values())
            
            return {
                "total_instances": len(self._instances),
                "total_requests": total_requests,
                "total_failures": total_failures,
                "overall_failure_rate": total_failures / max(total_requests, 1),
                "current_strategy": self._strategy.name,
                "instance_stats": {
                    iid: {
                        "total_requests": inst.total_requests,
                        "failed_requests": inst.failed_requests,
                        "failure_rate": inst.failure_rate,
                        "avg_response_time_ms": inst.avg_response_time_ms,
                        "current_connections": inst.current_connections,
                        "health_status": inst.health_status.name,
                        "selection_count": self._selection_stats.get(iid, 0),
                    }
                    for iid, inst in self._instances.items()
                },
                "model_distribution": {
                    model_id: len(instances)
                    for model_id, instances in self._model_instances.items()
                },
            }
    
    def get_healthy_instances(
        self,
        model_id: Optional[str] = None,
        status: Optional[HealthStatus] = None
    ) -> List[ModelInstance]:
        """获取健康实例"""
        with self._lock:
            instances = self.get_instances_for_model(model_id) if model_id else list(self._instances.values())
            
            if status:
                instances = [i for i in instances if i.health_status == status]
            else:
                instances = [i for i in instances if i.is_available]
            
            return instances
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        with self._lock:
            return {
                "strategy": self._strategy.name,
                "model_costs": self._model_costs,
                "instances": {
                    iid: inst.to_dict()
                    for iid, inst in self._instances.items()
                },
            }
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """导入配置"""
        with self._lock:
            # 导入策略
            if "strategy" in config:
                self.set_strategy(BalanceStrategy[config["strategy"]])
            
            # 导入成本
            if "model_costs" in config:
                self._model_costs.update(config["model_costs"])
            
            # 导入实例
            for iid, inst_data in config.get("instances", {}).items():
                instance = ModelInstance(
                    instance_id=inst_data["instance_id"],
                    model_id=inst_data["model_id"],
                    endpoint=inst_data.get("endpoint"),
                    region=inst_data.get("region", "default"),
                    weight=inst_data.get("weight", 1.0),
                    max_connections=inst_data.get("max_connections", 100),
                )
                self.register_instance(instance)
    
    def __del__(self):
        """析构时停止健康检查"""
        try:
            self._health_checker.stop()
        except Exception:
            pass
