"""
资源管理器 - 模拟算力、带宽稀缺时的竞争行为
"""
from __future__ import annotations
import random
import heapq
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

from .world import Agent, Position, World


class ResourceType(Enum):
    """资源类型"""
    COMPUTE = auto()       # 算力
    BANDWIDTH = auto()     # 带宽
    ENERGY = auto()        # 能量
    MEMORY = auto()        # 内存
    STORAGE = auto()       # 存储


class AllocationStrategy(Enum):
    """资源分配策略"""
    FIFO = auto()          # 先进先出
    PRIORITY = auto()      # 优先级
    AUCTION = auto()       # 拍卖
    LOTTERY = auto()       # 抽签
    PROPORTIONAL = auto()  # 比例分配


@dataclass
class ResourceRequest:
    """资源请求"""
    agent_id: str
    resource_type: ResourceType
    amount: float
    priority: float = 1.0
    timestamp: float = field(default_factory=lambda: random.random())
    deadline: Optional[float] = None
    callback: Optional[Callable[[bool, float], None]] = None


@dataclass
class ResourceNode:
    """资源节点"""
    node_id: str
    position: Position
    resource_type: ResourceType
    capacity: float
    current_load: float = 0.0
    allocated_to: Dict[str, float] = field(default_factory=dict)
    queue: List[ResourceRequest] = field(default_factory=list)

    @property
    def available(self) -> float:
        return max(0, self.capacity - self.current_load)

    @property
    def utilization_rate(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 0


@dataclass
class ResourcePacket:
    """资源包"""
    resource_type: ResourceType
    amount: float
    quality: float = 1.0  # 资源质量因子
    expiration: Optional[float] = None


class ResourceManager:
    """
    资源管理器
    管理多种稀缺资源，支持竞争、排队、分配策略
    """

    def __init__(self, world: World):
        self.world = world
        self.nodes: Dict[str, ResourceNode] = {}
        self.global_resources: Dict[ResourceType, float] = {}
        self.global_capacity: Dict[ResourceType, float] = {}
        self.agent_allocations: Dict[str, Dict[ResourceType, float]] = {}
        self.request_history: deque = deque(maxlen=1000)
        self.allocation_strategy: AllocationStrategy = AllocationStrategy.PRIORITY
        self.competition_factor: float = 1.0  # 竞争强度

    def create_resource_node(
        self,
        node_id: str,
        position: Position,
        resource_type: ResourceType,
        capacity: float
    ) -> ResourceNode:
        """创建资源节点"""
        node = ResourceNode(
            node_id=node_id,
            position=position,
            resource_type=resource_type,
            capacity=capacity
        )
        self.nodes[node_id] = node

        # 更新全局资源统计
        if resource_type not in self.global_capacity:
            self.global_capacity[resource_type] = 0
            self.global_resources[resource_type] = 0
        self.global_capacity[resource_type] += capacity
        self.global_resources[resource_type] += capacity

        return node

    def request_resource(
        self,
        agent_id: str,
        resource_type: ResourceType,
        amount: float,
        priority: float = 1.0,
        deadline: Optional[float] = None
    ) -> bool:
        """
        请求资源
        返回: 是否立即获得资源
        """
        request = ResourceRequest(
            agent_id=agent_id,
            resource_type=resource_type,
            amount=amount,
            priority=priority,
            deadline=deadline,
            timestamp=self.world.current_time
        )

        self.request_history.append({
            "time": self.world.current_time,
            "agent_id": agent_id,
            "type": resource_type,
            "amount": amount,
            "priority": priority
        })

        # 查找最佳资源节点
        node = self._find_best_node(agent_id, resource_type)
        if not node:
            return False

        # 尝试立即分配
        if node.available >= amount:
            self._allocate(node, agent_id, amount)
            return True
        else:
            # 加入队列
            node.queue.append(request)
            self._sort_queue(node)
            return False

    def release_resource(
        self,
        agent_id: str,
        node_id: str,
        amount: float
    ) -> None:
        """释放资源"""
        node = self.nodes.get(node_id)
        if not node or agent_id not in node.allocated_to:
            return

        allocated = node.allocated_to[agent_id]
        release_amount = min(amount, allocated)

        node.allocated_to[agent_id] -= release_amount
        if node.allocated_to[agent_id] <= 0:
            del node.allocated_to[agent_id]

        node.current_load -= release_amount
        self.global_resources[node.resource_type] += release_amount

        # 处理等待队列
        self._process_queue(node)

    def update(self, dt: float) -> None:
        """更新资源管理器状态"""
        # 处理各节点的队列
        for node in self.nodes.values():
            self._process_queue(node)

            # 资源自然恢复（如能量再生）
            if node.resource_type == ResourceType.ENERGY:
                recovery = node.capacity * 0.01 * dt
                node.current_load = max(0, node.current_load - recovery)

        # 检查竞争情况
        self._update_competition_metrics()

    def _find_best_node(
        self,
        agent_id: str,
        resource_type: ResourceType
    ) -> Optional[ResourceNode]:
        """查找最佳资源节点"""
        agent = self.world.get_agent(agent_id)
        if not agent:
            return None

        candidates = [
            node for node in self.nodes.values()
            if node.resource_type == resource_type
        ]

        if not candidates:
            return None

        # 按距离和可用性评分
        def score_node(node: ResourceNode) -> float:
            distance = agent.position.distance_to(node.position)
            availability = node.available / node.capacity if node.capacity > 0 else 0
            queue_penalty = len(node.queue) * 0.1
            return availability * 100 - distance - queue_penalty

        return max(candidates, key=score_node)

    def _allocate(
        self,
        node: ResourceNode,
        agent_id: str,
        amount: float
    ) -> None:
        """执行资源分配"""
        actual_amount = min(amount, node.available)

        if agent_id not in node.allocated_to:
            node.allocated_to[agent_id] = 0
        node.allocated_to[agent_id] += actual_amount
        node.current_load += actual_amount
        self.global_resources[node.resource_type] -= actual_amount

        # 更新Agent的资源记录
        if agent_id not in self.agent_allocations:
            self.agent_allocations[agent_id] = {}
        if node.resource_type not in self.agent_allocations[agent_id]:
            self.agent_allocations[agent_id][node.resource_type] = 0
        self.agent_allocations[agent_id][node.resource_type] += actual_amount

    def _sort_queue(self, node: ResourceNode) -> None:
        """根据分配策略排序队列"""
        if self.allocation_strategy == AllocationStrategy.PRIORITY:
            node.queue.sort(key=lambda r: (-r.priority, r.timestamp))
        elif self.allocation_strategy == AllocationStrategy.FIFO:
            node.queue.sort(key=lambda r: r.timestamp)
        elif self.allocation_strategy == AllocationStrategy.LOTTERY:
            random.shuffle(node.queue)

    def _process_queue(self, node: ResourceNode) -> None:
        """处理资源节点的等待队列"""
        current_time = self.world.current_time

        # 移除过期请求
        node.queue = [
            req for req in node.queue
            if req.deadline is None or req.deadline > current_time
        ]

        # 尝试满足队列中的请求
        fulfilled = []
        for request in node.queue[:]:
            if node.available >= request.amount:
                self._allocate(node, request.agent_id, request.amount)
                fulfilled.append(request)
                node.queue.remove(request)
            else:
                # 如果是拍卖策略，可能需要部分分配
                if self.allocation_strategy == AllocationStrategy.AUCTION:
                    partial = node.available * 0.5
                    if partial > 0:
                        self._allocate(node, request.agent_id, partial)

    def _update_competition_metrics(self) -> None:
        """更新竞争指标"""
        total_requests = len(self.request_history)
        if total_requests == 0:
            return

        # 计算资源紧张程度
        scarcity = {}
        for res_type in ResourceType:
            capacity = self.global_capacity.get(res_type, 1)
            available = self.global_resources.get(res_type, 0)
            scarcity[res_type] = 1 - (available / capacity) if capacity > 0 else 0

        # 更新竞争因子
        avg_scarcity = sum(scarcity.values()) / len(scarcity) if scarcity else 0
        self.competition_factor = 1 + avg_scarcity * 2

    def get_agent_resource(
        self,
        agent_id: str,
        resource_type: ResourceType
    ) -> float:
        """获取Agent拥有的资源量"""
        return self.agent_allocations.get(agent_id, {}).get(resource_type, 0)

    def get_node_utilization(self, node_id: str) -> float:
        """获取节点利用率"""
        node = self.nodes.get(node_id)
        return node.utilization_rate if node else 0

    def get_global_utilization(self, resource_type: ResourceType) -> float:
        """获取全局资源利用率"""
        capacity = self.global_capacity.get(resource_type, 0)
        available = self.global_resources.get(resource_type, 0)
        if capacity > 0:
            return (capacity - available) / capacity
        return 0

    def get_queue_length(self, node_id: str) -> int:
        """获取节点队列长度"""
        node = self.nodes.get(node_id)
        return len(node.queue) if node else 0

    def get_competition_stats(self) -> Dict[str, Any]:
        """获取竞争统计信息"""
        stats = {
            "competition_factor": self.competition_factor,
            "total_nodes": len(self.nodes),
            "resource_utilization": {},
            "queue_lengths": {},
            "total_requests": len(self.request_history)
        }

        for res_type in ResourceType:
            stats["resource_utilization"][res_type.name] = self.get_global_utilization(res_type)

        for node_id, node in self.nodes.items():
            stats["queue_lengths"][node_id] = len(node.queue)

        return stats

    def set_allocation_strategy(self, strategy: AllocationStrategy) -> None:
        """设置分配策略"""
        self.allocation_strategy = strategy

    def simulate_contention(
        self,
        resource_type: ResourceType,
        contenders: List[str],
        total_amount: float
    ) -> Dict[str, float]:
        """
        模拟多Agent资源竞争
        返回: {agent_id: 分配到的资源量}
        """
        if not contenders:
            return {}

        allocations = {}

        if self.allocation_strategy == AllocationStrategy.PROPORTIONAL:
            # 按比例分配
            per_agent = total_amount / len(contenders)
            for agent_id in contenders:
                allocations[agent_id] = per_agent

        elif self.allocation_strategy == AllocationStrategy.LOTTERY:
            # 抽签分配
            winner = random.choice(contenders)
            allocations[winner] = total_amount

        elif self.allocation_strategy == AllocationStrategy.AUCTION:
            # 简化的拍卖：随机出价
            bids = {agent_id: random.random() for agent_id in contenders}
            total_bid = sum(bids.values())
            if total_bid > 0:
                for agent_id, bid in bids.items():
                    allocations[agent_id] = total_amount * (bid / total_bid)

        else:
            # 默认均分
            per_agent = total_amount / len(contenders)
            for agent_id in contenders:
                allocations[agent_id] = per_agent

        return allocations


class ComputeResourceManager(ResourceManager):
    """专门的算力资源管理器"""

    def __init__(self, world: World):
        super().__init__(world)
        self.task_queue: List[Dict[str, Any]] = []
        self.compute_nodes: List[str] = []

    def submit_task(
        self,
        agent_id: str,
        compute_demand: float,
        priority: float = 1.0,
        deadline: Optional[float] = None
    ) -> str:
        """提交计算任务"""
        task_id = f"task_{agent_id}_{self.world.current_time}"
        task = {
            "task_id": task_id,
            "agent_id": agent_id,
            "demand": compute_demand,
            "priority": priority,
            "deadline": deadline,
            "submitted_at": self.world.current_time,
            "status": "queued"
        }
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: (-t["priority"], t["submitted_at"]))
        return task_id

    def allocate_compute(self, dt: float) -> None:
        """分配算力资源"""
        # 获取所有算力节点
        compute_nodes = [
            node for node in self.nodes.values()
            if node.resource_type == ResourceType.COMPUTE
        ]

        total_compute = sum(node.available for node in compute_nodes)

        # 处理任务队列
        for task in self.task_queue[:]:
            if task["status"] != "queued":
                continue

            demand = task["demand"]
            if total_compute >= demand:
                # 分配算力
                for node in compute_nodes:
                    if node.available > 0:
                        alloc = min(demand, node.available)
                        self._allocate(node, task["agent_id"], alloc)
                        demand -= alloc
                        if demand <= 0:
                            break

                task["status"] = "running"
                self.task_queue.remove(task)
            else:
                # 检查截止时间
                if task["deadline"] and self.world.current_time > task["deadline"]:
                    task["status"] = "failed"
                    self.task_queue.remove(task)


class BandwidthManager(ResourceManager):
    """专门的带宽资源管理器"""

    def __init__(self, world: World):
        super().__init__(world)
        self.transmissions: List[Dict[str, Any]] = []
        self.network_congestion: float = 0.0

    def request_transmission(
        self,
        agent_id: str,
        data_size: float,
        destination: Position,
        priority: float = 1.0
    ) -> bool:
        """请求数据传输"""
        agent = self.world.get_agent(agent_id)
        if not agent:
            return False

        # 计算距离和所需带宽
        distance = agent.position.distance_to(destination)
        bandwidth_needed = data_size / max(1, distance)

        # 考虑网络拥塞
        effective_need = bandwidth_needed * (1 + self.network_congestion)

        return self.request_resource(
            agent_id,
            ResourceType.BANDWIDTH,
            effective_need,
            priority
        )

    def update_network_state(self) -> None:
        """更新网络状态"""
        # 计算整体网络拥塞
        total_capacity = sum(
            node.capacity for node in self.nodes.values()
            if node.resource_type == ResourceType.BANDWIDTH
        )
        total_used = sum(
            node.current_load for node in self.nodes.values()
            if node.resource_type == ResourceType.BANDWIDTH
        )

        if total_capacity > 0:
            self.network_congestion = total_used / total_capacity
        else:
            self.network_congestion = 0

    def get_transmission_latency(
        self,
        source: Position,
        destination: Position
    ) -> float:
        """估算传输延迟"""
        distance = source.distance_to(destination)
        base_latency = distance * 0.1
        congestion_penalty = base_latency * self.network_congestion * 2
        return base_latency + congestion_penalty
