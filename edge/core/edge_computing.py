"""
Edge Computing Module - Comprehensive Edge Computing Framework
边缘计算模块 - 综合边缘计算框架

This module provides a complete implementation of edge computing concepts including:
- Edge device and server simulation
- Task offloading with multiple algorithms
- Resource allocation strategies
- Mobility management
- Caching strategies
- Federated learning on edge

Author: AGI Unified Framework
Version: 1.0.0
"""

from __future__ import annotations

import random
import math
import time
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Union
from collections import deque, defaultdict
from enum import Enum, auto
import copy


# =============================================================================
# EdgeConfig - Configuration
# =============================================================================

@dataclass
class EdgeConfig:
    """边缘计算配置类"""
    # Device settings
    default_compute_capacity: float = 1e9  # 1 GHz (cycles/sec)
    default_memory: int = 4 * 1024 * 1024 * 1024  # 4 GB
    default_battery: float = 4000  # mAh
    default_bandwidth: float = 10e6  # 10 Mbps

    # Server settings
    server_compute_capacity: float = 10e9  # 10 GHz
    server_memory: int = 64 * 1024 * 1024 * 1024  # 64 GB

    # Energy model parameters
    idle_power: float = 0.5  # Watts
    compute_power_coeff: float = 1e-9  # Energy per cycle
    transmission_power: float = 0.1  # Watts

    # Network settings
    cloud_latency: float = 100e-3  # 100ms
    edge_latency: float = 5e-3  # 5ms

    # Q-learning parameters
    ql_learning_rate: float = 0.1
    ql_discount_factor: float = 0.9
    ql_epsilon: float = 0.1

    # Lyapunov parameters
    lyapunov_v: float = 1.0

    # Caching parameters
    cache_size: int = 1000

    def __post_init__(self):
        self.validate()

    def validate(self):
        """验证配置参数"""
        assert self.default_compute_capacity > 0
        assert self.default_memory > 0
        assert self.default_battery > 0
        assert self.ql_learning_rate > 0
        assert self.ql_discount_factor > 0


# =============================================================================
# Task - Computation Task
# =============================================================================

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    OFFLOADED = auto()


@dataclass
class Task:
    """
    计算任务类

    Attributes:
        task_id: 任务唯一标识
        data_size: 数据大小 (bits)
        compute_requirement: 计算需求 (CPU cycles)
        deadline: 截止时间 (seconds)
        priority: 优先级 (1-10, 10为最高)
        dependencies: 依赖任务列表
        arrival_time: 到达时间
        status: 任务状态
    """
    task_id: str
    data_size: float  # bits
    compute_requirement: float  # CPU cycles
    deadline: float  # seconds
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING

    # Runtime metrics
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    offloading_decision: Optional[str] = None  # 'local', 'edge', 'cloud'

    def __post_init__(self):
        assert self.data_size > 0
        assert self.compute_requirement > 0
        assert self.deadline > 0
        assert 1 <= self.priority <= 10

    def get_remaining_time(self, current_time: float) -> float:
        """获取剩余时间"""
        return max(0, self.deadline - (current_time - self.arrival_time))

    def is_deadline_met(self, current_time: float) -> bool:
        """检查是否满足截止时间"""
        if self.completion_time is None:
            return False
        return (self.completion_time - self.arrival_time) <= self.deadline

    def get_latency(self) -> Optional[float]:
        """获取任务延迟"""
        if self.completion_time is None or self.start_time is None:
            return None
        return self.completion_time - self.arrival_time

    def __lt__(self, other: Task) -> bool:
        """用于优先级队列比较"""
        return self.priority > other.priority  # Higher priority first

    def __hash__(self) -> int:
        return hash(self.task_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return False
        return self.task_id == other.task_id


# =============================================================================
# EdgeDevice - Edge Device Simulation
# =============================================================================

class EdgeDevice:
    """
    边缘设备模拟类

    模拟具有计算能力、内存、电池和网络带宽的边缘设备。
    包含任务队列、能耗模型和在线状态管理。
    """

    def __init__(
        self,
        device_id: str,
        config: EdgeConfig = None,
        compute_capacity: float = None,
        memory: int = None,
        battery: float = None,
        bandwidth: float = None,
        location: Tuple[float, float] = (0.0, 0.0)
    ):
        self.device_id = device_id
        self.config = config or EdgeConfig()

        # Hardware specs
        self.compute_capacity = compute_capacity or self.config.default_compute_capacity
        self.memory = memory or self.config.default_memory
        self.initial_battery = battery or self.config.default_battery
        self.battery = self.initial_battery
        self.bandwidth = bandwidth or self.config.default_bandwidth
        self.location = location

        # Task management
        self.task_queue: deque[Task] = deque()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []

        # Status
        self.is_online = True
        self.current_cpu_usage = 0.0  # 0-1
        self.current_memory_usage = 0.0  # 0-1

        # Energy tracking
        self.total_energy_consumed = 0.0  # Joules
        self.energy_history: List[Tuple[float, float]] = []  # (time, energy)

        # Performance metrics
        self.total_tasks_processed = 0
        self.total_latency = 0.0
        self.deadline_violations = 0

    def get_compute_time(self, task: Task) -> float:
        """计算本地执行时间"""
        return task.compute_requirement / self.compute_capacity

    def get_compute_energy(self, task: Task) -> float:
        """计算本地执行能耗 (E = P * t)"""
        compute_time = self.get_compute_time(task)
        power = self.config.idle_power + self.config.compute_power_coeff * self.compute_capacity
        return power * compute_time

    def get_transmission_energy(self, data_size: float, distance: float = 100) -> float:
        """
        计算传输能耗
        使用自由空间路径损耗模型
        """
        # Simplified energy model: E_tx = P_tx * T_tx
        transmission_time = data_size / self.bandwidth
        return self.config.transmission_power * transmission_time

    def submit_task(self, task: Task) -> bool:
        """提交任务到设备"""
        if not self.is_online:
            return False

        # Check dependencies
        pending_deps = [d for d in task.dependencies if d not in [t.task_id for t in self.completed_tasks]]
        if pending_deps:
            task.status = TaskStatus.PENDING
        else:
            task.status = TaskStatus.QUEUED

        self.task_queue.append(task)
        return True

    def process_tasks(self, duration: float, current_time: float) -> List[Task]:
        """
        处理任务队列

        Args:
            duration: 处理时间窗口
            current_time: 当前时间

        Returns:
            完成的任务列表
        """
        completed = []
        remaining_time = duration

        while remaining_time > 0 and self.task_queue:
            # Get highest priority task
            task = self._get_next_task()
            if task is None:
                break

            compute_time = self.get_compute_time(task)
            energy = self.get_compute_energy(task)

            if compute_time <= remaining_time:
                # Task can be completed
                task.start_time = current_time + (duration - remaining_time)
                task.completion_time = task.start_time + compute_time
                task.status = TaskStatus.COMPLETED
                task.offloading_decision = 'local'

                self.battery -= energy / 3600 * 1000  # Convert to mAh (simplified)
                self.total_energy_consumed += energy
                self.total_tasks_processed += 1

                latency = task.get_latency()
                if latency:
                    self.total_latency += latency
                    if not task.is_deadline_met(task.completion_time):
                        self.deadline_violations += 1

                self.completed_tasks.append(task)
                completed.append(task)
                remaining_time -= compute_time
            else:
                # Task partially processed, put back
                task.compute_requirement -= remaining_time * self.compute_capacity
                self.task_queue.appendleft(task)
                break

        self.energy_history.append((current_time, self.total_energy_consumed))
        return completed

    def _get_next_task(self) -> Optional[Task]:
        """获取下一个要执行的任务（按优先级）"""
        if not self.task_queue:
            return None

        # Find highest priority ready task
        ready_tasks = [
            t for t in self.task_queue
            if t.status == TaskStatus.QUEUED or
               (t.status == TaskStatus.PENDING and all(
                   d in [ct.task_id for ct in self.completed_tasks]
                   for d in t.dependencies
               ))
        ]

        if not ready_tasks:
            # Check for pending tasks that can now run
            for task in list(self.task_queue):
                if task.status == TaskStatus.PENDING:
                    if all(d in [ct.task_id for ct in self.completed_tasks] for d in task.dependencies):
                        task.status = TaskStatus.QUEUED
                        ready_tasks.append(task)

        if ready_tasks:
            ready_tasks.sort(key=lambda t: (-t.priority, t.arrival_time))
            task = ready_tasks[0]
            self.task_queue.remove(task)
            return task

        return None

    def get_battery_percentage(self) -> float:
        """获取电池百分比"""
        return (self.battery / self.initial_battery) * 100

    def set_online_status(self, online: bool):
        """设置在线状态"""
        self.is_online = online

    def update_location(self, new_location: Tuple[float, float]):
        """更新设备位置"""
        self.location = new_location

    def get_average_latency(self) -> float:
        """获取平均延迟"""
        if self.total_tasks_processed == 0:
            return 0.0
        return self.total_latency / self.total_tasks_processed

    def get_metrics(self) -> Dict[str, Any]:
        """获取设备性能指标"""
        return {
            'device_id': self.device_id,
            'battery_percentage': self.get_battery_percentage(),
            'total_energy_consumed': self.total_energy_consumed,
            'tasks_processed': self.total_tasks_processed,
            'average_latency': self.get_average_latency(),
            'deadline_violations': self.deadline_violations,
            'queue_length': len(self.task_queue),
            'is_online': self.is_online,
            'location': self.location
        }


# =============================================================================
# EdgeServer - Edge Server/Cloudlet
# =============================================================================

class EdgeServer:
    """
    边缘服务器/云let类

    管理多个边缘设备，提供负载均衡和缓存服务。
    """

    def __init__(
        self,
        server_id: str,
        config: EdgeConfig = None,
        compute_capacity: float = None,
        memory: int = None,
        location: Tuple[float, float] = (0.0, 0.0),
        coverage_radius: float = 500.0  # meters
    ):
        self.server_id = server_id
        self.config = config or EdgeConfig()

        # Hardware specs
        self.compute_capacity = compute_capacity or self.config.server_compute_capacity
        self.memory = memory or self.config.server_memory
        self.location = location
        self.coverage_radius = coverage_radius

        # Device management
        self.connected_devices: Dict[str, EdgeDevice] = {}
        self.device_distances: Dict[str, float] = {}

        # Task management
        self.task_queue: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []

        # Resource allocation
        self.cpu_shares: Dict[str, float] = {}  # Device ID -> CPU share
        self.memory_allocations: Dict[str, int] = {}

        # Load tracking
        self.current_load = 0.0  # 0-1
        self.load_history: List[Tuple[float, float]] = []

        # Performance metrics
        self.total_tasks_processed = 0
        self.total_latency = 0.0
        self.deadline_violations = 0

    def connect_device(self, device: EdgeDevice) -> bool:
        """连接设备到服务器"""
        distance = self._calculate_distance(device.location)
        if distance <= self.coverage_radius:
            self.connected_devices[device.device_id] = device
            self.device_distances[device.device_id] = distance
            self._update_cpu_shares()
            return True
        return False

    def disconnect_device(self, device_id: str):
        """断开设备连接"""
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            del self.device_distances[device_id]
            self._update_cpu_shares()

    def _calculate_distance(self, location: Tuple[float, float]) -> float:
        """计算欧几里得距离"""
        return math.sqrt(
            (location[0] - self.location[0]) ** 2 +
            (location[1] - self.location[1]) ** 2
        )

    def _update_cpu_shares(self):
        """更新CPU份额（比例分配）"""
        if not self.connected_devices:
            return

        num_devices = len(self.connected_devices)
        share = 1.0 / num_devices
        for device_id in self.connected_devices:
            self.cpu_shares[device_id] = share

    def get_compute_time(self, task: Task, device_id: str = None) -> float:
        """计算在边缘服务器的执行时间"""
        if device_id and device_id in self.cpu_shares:
            effective_capacity = self.compute_capacity * self.cpu_shares[device_id]
        else:
            effective_capacity = self.compute_capacity / max(1, len(self.connected_devices))

        return task.compute_requirement / effective_capacity

    def get_transmission_time(self, task: Task, device: EdgeDevice) -> float:
        """计算传输时间"""
        distance = self.device_distances.get(device.device_id, 100)
        # Path loss model: higher distance = lower effective bandwidth
        effective_bandwidth = device.bandwidth * (1 / (1 + 0.01 * distance))
        return task.data_size / effective_bandwidth

    def submit_task(self, task: Task, device_id: str) -> bool:
        """从设备提交任务到服务器"""
        if device_id not in self.connected_devices:
            return False

        task.status = TaskStatus.QUEUED
        heapq.heappush(self.task_queue, task)
        return True

    def process_tasks(self, duration: float, current_time: float) -> List[Task]:
        """处理任务队列"""
        completed = []
        remaining_time = duration

        # Sort by priority
        while self.task_queue and remaining_time > 0:
            task = heapq.heappop(self.task_queue)

            # Find which device submitted this task
            device_id = None
            for did, dev in self.connected_devices.items():
                if any(t.task_id == task.task_id for t in dev.task_queue):
                    device_id = did
                    break

            compute_time = self.get_compute_time(task, device_id)

            if compute_time <= remaining_time:
                task.start_time = current_time + (duration - remaining_time)
                task.completion_time = task.start_time + compute_time
                task.status = TaskStatus.COMPLETED
                task.offloading_decision = 'edge'

                self.total_tasks_processed += 1
                latency = task.get_latency()
                if latency:
                    self.total_latency += latency
                    if not task.is_deadline_met(task.completion_time):
                        self.deadline_violations += 1

                self.completed_tasks.append(task)
                completed.append(task)
                remaining_time -= compute_time
            else:
                # Put back with updated compute requirement
                task.compute_requirement -= remaining_time * self.compute_capacity
                heapq.heappush(self.task_queue, task)
                break

        # Update load
        self.current_load = len(self.task_queue) / max(1, self.compute_capacity / 1e9)
        self.load_history.append((current_time, self.current_load))

        return completed

    def get_load_balance_score(self) -> float:
        """
        获取负载均衡分数
        使用变异系数 (CV) 的倒数
        """
        if len(self.connected_devices) <= 1:
            return 1.0

        loads = [self.cpu_shares.get(did, 0) for did in self.connected_devices]
        mean_load = sum(loads) / len(loads)
        if mean_load == 0:
            return 1.0

        variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
        cv = math.sqrt(variance) / mean_load
        return 1.0 / (1.0 + cv)

    def get_metrics(self) -> Dict[str, Any]:
        """获取服务器性能指标"""
        return {
            'server_id': self.server_id,
            'connected_devices': len(self.connected_devices),
            'current_load': self.current_load,
            'tasks_processed': self.total_tasks_processed,
            'average_latency': self.total_latency / max(1, self.total_tasks_processed),
            'deadline_violations': self.deadline_violations,
            'load_balance_score': self.get_load_balance_score(),
            'queue_length': len(self.task_queue)
        }


# =============================================================================
# TaskOffloading - Offloading Decision Algorithms
# =============================================================================

class TaskOffloading:
    """
    任务卸载决策类

    实现多种卸载决策算法：
    1. 本地执行成本计算
    2. 边缘卸载成本计算
    3. 云端卸载成本计算
    4. Lyapunov优化
    5. Q-learning
    6. 博弈论卸载
    """

    def __init__(self, config: EdgeConfig = None):
        self.config = config or EdgeConfig()

        # Q-learning state
        self.ql_q_table: Dict[Tuple, Dict[str, float]] = {}
        self.ql_state_history: List[Tuple] = []

        # Game theory state
        self.gt_strategies: Dict[str, np.ndarray] = {}
        self.gt_payoffs: Dict[str, float] = {}

    def calculate_local_cost(
        self,
        task: Task,
        device: EdgeDevice,
        current_time: float
    ) -> Dict[str, float]:
        """
        计算本地执行成本

        Returns:
            {'latency': float, 'energy': float, 'total_cost': float}
        """
        latency = device.get_compute_time(task)
        energy = device.get_compute_energy(task)

        # Check deadline violation penalty
        deadline_violation = max(0, latency - task.get_remaining_time(current_time))
        penalty = 1000 * deadline_violation  # High penalty for violation

        total_cost = latency + 0.5 * energy + penalty

        return {
            'latency': latency,
            'energy': energy,
            'deadline_violation': deadline_violation,
            'total_cost': total_cost
        }

    def calculate_edge_cost(
        self,
        task: Task,
        device: EdgeDevice,
        server: EdgeServer,
        current_time: float
    ) -> Dict[str, float]:
        """
        计算边缘卸载成本

        Cost = communication_time + compute_time + energy
        """
        # Transmission time (uplink)
        transmission_time = server.get_transmission_time(task, device)

        # Compute time at edge
        compute_time = server.get_compute_time(task, device.device_id)

        # Total latency
        latency = transmission_time + compute_time

        # Energy (transmission + idle waiting)
        transmission_energy = device.get_transmission_energy(task.data_size)
        idle_energy = self.config.idle_power * latency
        energy = transmission_energy + idle_energy

        # Deadline violation
        deadline_violation = max(0, latency - task.get_remaining_time(current_time))
        penalty = 1000 * deadline_violation

        total_cost = latency + 0.5 * energy + penalty

        return {
            'latency': latency,
            'energy': energy,
            'transmission_time': transmission_time,
            'compute_time': compute_time,
            'deadline_violation': deadline_violation,
            'total_cost': total_cost
        }

    def calculate_cloud_cost(
        self,
        task: Task,
        device: EdgeDevice,
        current_time: float
    ) -> Dict[str, float]:
        """
        计算云端卸载成本

        Cost = 2 * transmission_time + cloud_latency + energy
        """
        # Assume cloud has infinite compute capacity
        cloud_compute_time = task.compute_requirement / (100e9)  # 100 GHz

        # Transmission to cloud (via cellular/WiFi)
        transmission_time = task.data_size / (device.bandwidth * 0.5)  # Lower bandwidth to cloud

        # Total latency (uplink + cloud + downlink)
        latency = transmission_time + self.config.cloud_latency + cloud_compute_time

        # Energy
        transmission_energy = device.get_transmission_energy(task.data_size * 2)  # Up + down
        idle_energy = self.config.idle_power * latency
        energy = transmission_energy + idle_energy

        deadline_violation = max(0, latency - task.get_remaining_time(current_time))
        penalty = 1000 * deadline_violation

        total_cost = latency + 0.5 * energy + penalty

        return {
            'latency': latency,
            'energy': energy,
            'cloud_latency': self.config.cloud_latency,
            'deadline_violation': deadline_violation,
            'total_cost': total_cost
        }

    def lyapunov_offloading(
        self,
        task: Task,
        device: EdgeDevice,
        server: Optional[EdgeServer],
        current_time: float,
        queue_backlog: float
    ) -> str:
        """
        Lyapunov优化动态卸载

        基于队列稳定性和能耗最小化的联合优化。
        使用漂移加惩罚算法。
        """
        V = self.config.lyapunov_v

        # Calculate costs
        local_cost = self.calculate_local_cost(task, device, current_time)
        local_drift = queue_backlog * local_cost['latency'] + V * local_cost['energy']

        if server and server.connect_device(device):
            edge_cost = self.calculate_edge_cost(task, device, server, current_time)
            edge_drift = queue_backlog * edge_cost['latency'] + V * edge_cost['energy']
            server.disconnect_device(device.device_id)
        else:
            edge_drift = float('inf')

        cloud_cost = self.calculate_cloud_cost(task, device, current_time)
        cloud_drift = queue_backlog * cloud_cost['latency'] + V * cloud_cost['energy']

        # Choose minimum drift
        costs = {
            'local': local_drift,
            'edge': edge_drift,
            'cloud': cloud_drift
        }

        return min(costs, key=costs.get)

    def q_learning_offloading(
        self,
        task: Task,
        device: EdgeDevice,
        server: Optional[EdgeServer],
        current_time: float,
        state: Tuple
    ) -> str:
        """
        基于Q-learning的卸载决策

        State: (battery_level, queue_length, task_size, deadline)
        Action: {'local', 'edge', 'cloud'}
        """
        actions = ['local', 'edge', 'cloud']

        # Initialize Q-table for state if needed
        if state not in self.ql_q_table:
            self.ql_q_table[state] = {a: 0.0 for a in actions}

        # Epsilon-greedy action selection
        if random.random() < self.config.ql_epsilon:
            action = random.choice(actions)
        else:
            q_values = self.ql_q_table[state]
            action = max(q_values, key=q_values.get)

        # Calculate reward (negative cost)
        if action == 'local':
            cost = self.calculate_local_cost(task, device, current_time)
        elif action == 'edge' and server:
            cost = self.calculate_edge_cost(task, device, server, current_time)
        else:
            cost = self.calculate_cloud_cost(task, device, current_time)

        reward = -cost['total_cost']

        # Update Q-value (simplified, would need next_state in practice)
        alpha = self.config.ql_learning_rate
        gamma = self.config.ql_discount_factor

        current_q = self.ql_q_table[state][action]
        self.ql_q_table[state][action] = (1 - alpha) * current_q + alpha * (reward + gamma * 0)

        return action

    def game_theoretic_offloading(
        self,
        task: Task,
        device: EdgeDevice,
        other_devices: List[EdgeDevice],
        server: EdgeServer,
        current_time: float
    ) -> str:
        """
        博弈论卸载决策（纳什均衡）

        将卸载决策建模为势博弈，寻找纳什均衡。
        """
        actions = ['local', 'edge']

        # Build payoff matrix
        # Simplified: consider congestion at edge server
        n_edge = sum(1 for d in other_devices if d.task_queue)
        congestion_factor = 1 + 0.1 * n_edge

        local_cost = self.calculate_local_cost(task, device, current_time)
        edge_cost = self.calculate_edge_cost(task, device, server, current_time)

        # Adjust edge cost for congestion
        edge_cost['total_cost'] *= congestion_factor

        # Find best response (Nash equilibrium for single player)
        payoffs = {
            'local': -local_cost['total_cost'],
            'edge': -edge_cost['total_cost']
        }

        return max(payoffs, key=payoffs.get)

    def make_offloading_decision(
        self,
        task: Task,
        device: EdgeDevice,
        server: Optional[EdgeServer],
        current_time: float,
        algorithm: str = 'lyapunov',
        **kwargs
    ) -> str:
        """
        统一的卸载决策接口

        Args:
            algorithm: 'local', 'edge', 'cloud', 'lyapunov', 'q_learning', 'game_theory'
        """
        if algorithm == 'local':
            return 'local'
        elif algorithm == 'edge':
            return 'edge' if server else 'local'
        elif algorithm == 'cloud':
            return 'cloud'
        elif algorithm == 'lyapunov':
            queue_backlog = kwargs.get('queue_backlog', len(device.task_queue))
            return self.lyapunov_offloading(task, device, server, current_time, queue_backlog)
        elif algorithm == 'q_learning':
            state = kwargs.get('state', (
                int(device.get_battery_percentage() / 10),
                len(device.task_queue),
                int(task.data_size / 1e6),
                int(task.deadline / 0.1)
            ))
            return self.q_learning_offloading(task, device, server, current_time, state)
        elif algorithm == 'game_theory':
            other_devices = kwargs.get('other_devices', [])
            return self.game_theoretic_offloading(task, device, other_devices, server, current_time)
        else:
            # Default: choose minimum cost
            costs = {'local': self.calculate_local_cost(task, device, current_time)['total_cost']}
            if server:
                costs['edge'] = self.calculate_edge_cost(task, device, server, current_time)['total_cost']
            costs['cloud'] = self.calculate_cloud_cost(task, device, current_time)['total_cost']
            return min(costs, key=costs.get)


# =============================================================================
# ResourceAllocation - Resource Management
# =============================================================================

class ResourceAllocation:
    """
    资源分配类

    实现多种资源管理算法：
    1. CPU分配：比例共享、最大-最小公平
    2. 内存管理：分页、缓存
    3. 带宽分配：令牌桶、漏桶
    4. 能耗感知调度
    """

    def __init__(self, config: EdgeConfig = None):
        self.config = config or EdgeConfig()

        # Token bucket state
        self.token_buckets: Dict[str, Dict] = {}

        # Leaky bucket state
        self.leaky_buckets: Dict[str, Dict] = {}

        # Memory management
        self.page_table: Dict[str, List[int]] = {}
        self.cache: Dict[str, Any] = {}

    def proportional_share_cpu(
        self,
        devices: List[EdgeDevice],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        比例共享CPU分配

        Args:
            devices: 设备列表
            weights: 权重字典 {device_id: weight}

        Returns:
            CPU份额分配 {device_id: share}
        """
        if not devices:
            return {}

        if weights is None:
            weights = {d.device_id: 1.0 for d in devices}

        total_weight = sum(weights.get(d.device_id, 1.0) for d in devices)

        shares = {}
        for device in devices:
            weight = weights.get(device.device_id, 1.0)
            shares[device.device_id] = weight / total_weight

        return shares

    def max_min_fairness(
        self,
        devices: List[EdgeDevice],
        demands: Dict[str, float],
        total_resource: float
    ) -> Dict[str, float]:
        """
        最大-最小公平分配

        保证最小需求的同时，公平分配剩余资源。
        """
        if not devices or total_resource <= 0:
            return {}

        n = len(devices)
        allocation = {d.device_id: 0.0 for d in devices}
        remaining = total_resource

        # Sort by demand
        sorted_devices = sorted(devices, key=lambda d: demands.get(d.device_id, 0))

        for i, device in enumerate(sorted_devices):
            demand = demands.get(device.device_id, 0)
            fair_share = remaining / (n - i)

            if demand <= fair_share:
                allocation[device.device_id] = demand
                remaining -= demand
            else:
                allocation[device.device_id] = fair_share
                remaining -= fair_share

        return allocation

    def token_bucket_allocate(
        self,
        flow_id: str,
        packet_size: float,
        rate: float,  # tokens per second
        bucket_size: float,  # max tokens
        current_time: float
    ) -> bool:
        """
        令牌桶带宽分配

        Returns:
            True if packet can be transmitted
        """
        if flow_id not in self.token_buckets:
            self.token_buckets[flow_id] = {
                'tokens': bucket_size,
                'last_update': current_time
            }

        bucket = self.token_buckets[flow_id]

        # Add tokens based on time elapsed
        elapsed = current_time - bucket['last_update']
        bucket['tokens'] = min(bucket_size, bucket['tokens'] + rate * elapsed)
        bucket['last_update'] = current_time

        # Check if enough tokens
        if bucket['tokens'] >= packet_size:
            bucket['tokens'] -= packet_size
            return True
        return False

    def leaky_bucket_shape(
        self,
        flow_id: str,
        packet_size: float,
        rate: float,  # leak rate
        bucket_size: float,
        current_time: float
    ) -> float:
        """
        漏桶流量整形

        Returns:
            Delay before packet can be sent
        """
        if flow_id not in self.leaky_buckets:
            self.leaky_buckets[flow_id] = {
                'volume': 0,
                'last_update': current_time
            }

        bucket = self.leaky_buckets[flow_id]

        # Leak out tokens
        elapsed = current_time - bucket['last_update']
        bucket['volume'] = max(0, bucket['volume'] - rate * elapsed)
        bucket['last_update'] = current_time

        # Calculate delay
        if bucket['volume'] + packet_size <= bucket_size:
            delay = 0
        else:
            delay = (bucket['volume'] + packet_size - bucket_size) / rate

        bucket['volume'] += packet_size
        return delay

    def allocate_memory_pages(
        self,
        process_id: str,
        num_pages: int,
        available_frames: List[int]
    ) -> List[int]:
        """
        内存分页分配

        Returns:
            分配的页框列表
        """
        if len(available_frames) < num_pages:
            return []  # Not enough memory

        allocated = available_frames[:num_pages]
        self.page_table[process_id] = allocated
        return allocated

    def lru_cache_access(
        self,
        key: str,
        value: Any,
        cache_size: int,
        access_history: deque
    ) -> Tuple[bool, Optional[str]]:
        """
        LRU缓存访问

        Returns:
            (hit: bool, evicted_key: Optional[str])
        """
        if key in self.cache:
            # Hit - update access order
            access_history.remove(key)
            access_history.append(key)
            return True, None

        # Miss
        evicted = None
        if len(self.cache) >= cache_size:
            evicted = access_history.popleft()
            del self.cache[evicted]

        self.cache[key] = value
        access_history.append(key)
        return False, evicted

    def energy_aware_schedule(
        self,
        tasks: List[Task],
        device: EdgeDevice,
        current_time: float
    ) -> List[Task]:
        """
        能耗感知调度

        根据电池状态和任务特性进行调度。
        """
        battery_pct = device.get_battery_percentage()

        if battery_pct > 50:
            # High battery: prioritize latency (EDF)
            return sorted(tasks, key=lambda t: t.deadline)
        elif battery_pct > 20:
            # Medium battery: balance latency and energy
            return sorted(tasks, key=lambda t: t.deadline + 0.5 * t.compute_requirement)
        else:
            # Low battery: prioritize energy efficiency
            return sorted(tasks, key=lambda t: t.compute_requirement / max(1, t.priority))

    def dynamic_voltage_frequency_scaling(
        self,
        workload: float,
        deadline: float,
        max_frequency: float
    ) -> float:
        """
        动态电压频率调整 (DVFS)

        根据工作负载和截止时间调整频率以节省能量。
        """
        required_frequency = workload / deadline
        frequency = min(max_frequency, max(required_frequency, max_frequency * 0.2))
        return frequency


# =============================================================================
# MobilityManagement - Handle Device Mobility
# =============================================================================

class MobilityManagement:
    """
    移动性管理类

    处理设备移动、切换和服务迁移。
    """

    def __init__(self, config: EdgeConfig = None):
        self.config = config or EdgeConfig()

        # Handover tracking
        self.handover_history: List[Dict] = []
        self.active_handovers: Dict[str, Dict] = {}

        # Migration state
        self.migration_queue: List[Dict] = []
        self.service_locations: Dict[str, str] = {}  # service_id -> server_id

        # Trajectory prediction
        self.trajectory_history: Dict[str, List[Tuple[float, Tuple[float, float]]]] = {}

        # MDP state
        self.mdp_states: Dict[str, Dict] = {}

    def calculate_distance(
        self,
        loc1: Tuple[float, float],
        loc2: Tuple[float, float]
    ) -> float:
        """计算两点间距离"""
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    def check_handover_needed(
        self,
        device: EdgeDevice,
        current_server: EdgeServer,
        servers: List[EdgeServer]
    ) -> Optional[EdgeServer]:
        """
        检查是否需要切换

        Returns:
            目标服务器或None
        """
        current_distance = self.calculate_distance(device.location, current_server.location)

        # Check if still in coverage
        if current_distance <= current_server.coverage_radius:
            return None

        # Find best target server
        best_server = None
        best_distance = float('inf')

        for server in servers:
            if server.server_id == current_server.server_id:
                continue

            distance = self.calculate_distance(device.location, server.location)
            if distance <= server.coverage_radius and distance < best_distance:
                best_distance = distance
                best_server = server

        return best_server

    def perform_handover(
        self,
        device: EdgeDevice,
        source_server: EdgeServer,
        target_server: EdgeServer,
        current_time: float
    ) -> bool:
        """执行切换"""
        # Disconnect from source
        source_server.disconnect_device(device.device_id)

        # Connect to target
        success = target_server.connect_device(device)

        if success:
            handover_record = {
                'device_id': device.device_id,
                'source': source_server.server_id,
                'target': target_server.server_id,
                'time': current_time,
                'latency': self.config.edge_latency
            }
            self.handover_history.append(handover_record)

        return success

    def predict_trajectory(
        self,
        device_id: str,
        current_location: Tuple[float, float],
        history_length: int = 5
    ) -> Tuple[float, float]:
        """
        基于历史轨迹预测未来位置

        使用线性外推法。
        """
        if device_id not in self.trajectory_history:
            self.trajectory_history[device_id] = []

        history = self.trajectory_history[device_id]
        history.append((time.time(), current_location))

        # Keep only recent history
        if len(history) > history_length:
            history.pop(0)

        if len(history) < 2:
            return current_location

        # Calculate velocity vector
        dt = history[-1][0] - history[0][0]
        if dt == 0:
            return current_location

        dx = history[-1][1][0] - history[0][1][0]
        dy = history[-1][1][1] - history[0][1][1]

        vx = dx / dt
        vy = dy / dt

        # Predict next position (1 second ahead)
        predicted_x = current_location[0] + vx
        predicted_y = current_location[1] + vy

        return (predicted_x, predicted_y)

    def predictive_migration(
        self,
        device: EdgeDevice,
        current_server: EdgeServer,
        servers: List[EdgeServer],
        service_id: str
    ) -> Optional[EdgeServer]:
        """
        基于轨迹预测的主动迁移

        在设备离开覆盖范围前预先迁移服务。
        """
        predicted_location = self.predict_trajectory(device.device_id, device.location)

        # Check if predicted location is out of current coverage
        current_distance = self.calculate_distance(predicted_location, current_server.location)

        if current_distance > current_server.coverage_radius * 0.8:  # 80% threshold
            # Find target server for predicted location
            for server in servers:
                if server.server_id == current_server.server_id:
                    continue

                distance = self.calculate_distance(predicted_location, server.location)
                if distance <= server.coverage_radius:
                    return server

        return None

    def mdp_migration_decision(
        self,
        device: EdgeDevice,
        current_server: EdgeServer,
        candidate_servers: List[EdgeServer],
        service_load: float
    ) -> Tuple[bool, Optional[EdgeServer]]:
        """
        基于马尔可夫决策过程的服务迁移决策

        State: (signal_strength, server_load, device_velocity)
        Action: {stay, migrate_to_server_i}
        """
        # Simplified MDP: use value iteration concept

        # Calculate current signal strength (inverse of distance)
        current_distance = self.calculate_distance(device.location, current_server.location)
        signal_strength = max(0, 1 - current_distance / current_server.coverage_radius)

        # Calculate values for each action
        values = {}

        # Stay action
        stay_cost = (1 - signal_strength) * 10 + service_load * 5
        values['stay'] = -stay_cost

        # Migration actions
        for server in candidate_servers:
            if server.server_id == current_server.server_id:
                continue

            distance = self.calculate_distance(device.location, server.location)
            future_signal = max(0, 1 - distance / server.coverage_radius)

            migration_cost = 5  # Migration overhead
            future_benefit = future_signal * 10 - server.current_load * 3

            values[server.server_id] = future_benefit - migration_cost

        # Choose best action
        best_action = max(values, key=values.get)

        if best_action == 'stay':
            return False, None
        else:
            target = next(s for s in candidate_servers if s.server_id == best_action)
            return True, target

    def migrate_service(
        self,
        service_id: str,
        source_server: EdgeServer,
        target_server: EdgeServer,
        service_state: Dict
    ) -> bool:
        """执行服务迁移"""
        # Update service location
        self.service_locations[service_id] = target_server.server_id

        migration_record = {
            'service_id': service_id,
            'source': source_server.server_id,
            'target': target_server.server_id,
            'state_size': len(str(service_state)),
            'time': time.time()
        }
        self.migration_queue.append(migration_record)

        return True


# =============================================================================
# CachingStrategy - Edge Caching
# =============================================================================

class CachingStrategy:
    """
    边缘缓存策略类

    实现多种缓存算法：
    1. LRU, LFU, FIFO
    2. 基于流行度的缓存
    3. 协作缓存
    4. 强化学习缓存
    """

    def __init__(self, cache_size: int = 100, config: EdgeConfig = None):
        self.config = config or EdgeConfig()
        self.cache_size = cache_size

        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict] = {}

        # Access tracking
        self.access_history: Dict[str, deque] = {}
        self.access_frequency: Dict[str, int] = {}

        # RL state
        self.rl_q_table: Dict[Tuple, Dict[str, float]] = {}

        # Cooperative caching
        self.neighbor_caches: Dict[str, 'CachingStrategy'] = {}

    def lru_access(self, key: str, value: Any = None) -> Tuple[bool, Any]:
        """
        LRU缓存访问

        Returns:
            (hit, value)
        """
        if key in self.cache:
            # Hit - update access time
            self.cache_metadata[key]['last_access'] = time.time()
            return True, self.cache[key]

        # Miss
        if value is None:
            return False, None

        # Evict if necessary
        if len(self.cache) >= self.cache_size:
            self._lru_evict()

        # Insert
        self.cache[key] = value
        self.cache_metadata[key] = {
            'insert_time': time.time(),
            'last_access': time.time(),
            'access_count': 1
        }

        return False, value

    def _lru_evict(self):
        """LRU淘汰"""
        if not self.cache:
            return

        lru_key = min(self.cache, key=lambda k: self.cache_metadata[k]['last_access'])
        del self.cache[lru_key]
        del self.cache_metadata[lru_key]

    def lfu_access(self, key: str, value: Any = None) -> Tuple[bool, Any]:
        """
        LFU缓存访问
        """
        self.access_frequency[key] = self.access_frequency.get(key, 0) + 1

        if key in self.cache:
            return True, self.cache[key]

        if value is None:
            return False, None

        if len(self.cache) >= self.cache_size:
            self._lfu_evict()

        self.cache[key] = value
        return False, value

    def _lfu_evict(self):
        """LFU淘汰"""
        if not self.cache:
            return

        lfu_key = min(self.cache, key=lambda k: self.access_frequency.get(k, 0))
        del self.cache[lfu_key]

    def fifo_access(self, key: str, value: Any = None) -> Tuple[bool, Any]:
        """
        FIFO缓存访问
        """
        if key in self.cache:
            return True, self.cache[key]

        if value is None:
            return False, None

        if len(self.cache) >= self.cache_size:
            self._fifo_evict()

        self.cache[key] = value
        self.cache_metadata[key] = {'insert_time': time.time()}
        return False, value

    def _fifo_evict(self):
        """FIFO淘汰"""
        if not self.cache:
            return

        fifo_key = min(self.cache, key=lambda k: self.cache_metadata[k]['insert_time'])
        del self.cache[fifo_key]
        del self.cache_metadata[fifo_key]

    def popularity_based_cache(
        self,
        content_catalog: Dict[str, float],  # content_id -> popularity
        zipf_alpha: float = 1.0
    ) -> Set[str]:
        """
        基于流行度的缓存 (Zipf分布)

        缓存最流行的内容。
        """
        # Sort by popularity
        sorted_content = sorted(
            content_catalog.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Cache top-k
        cached = set()
        for content_id, _ in sorted_content[:self.cache_size]:
            cached.add(content_id)

        return cached

    def cooperative_cache_lookup(
        self,
        key: str,
        neighbors: List['CachingStrategy']
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        协作缓存查找

        Returns:
            (found, value, source_cache_id)
        """
        # Check local cache
        if key in self.cache:
            return True, self.cache[key], 'local'

        # Check neighbor caches
        for neighbor in neighbors:
            if key in neighbor.cache:
                return True, neighbor.cache[key], neighbor

        return False, None, None

    def q_learning_cache(
        self,
        content_id: str,
        content_features: Tuple,  # (size, popularity, recency)
        reward: float
    ) -> bool:
        """
        基于Q-learning的缓存决策

        State: content features
        Action: {cache, not_cache}
        """
        actions = ['cache', 'not_cache']

        if content_features not in self.rl_q_table:
            self.rl_q_table[content_features] = {a: 0.0 for a in actions}

        # Choose action
        q_values = self.rl_q_table[content_features]
        action = max(q_values, key=q_values.get)

        # Update Q-value
        alpha = 0.1
        current_q = q_values[action]
        self.rl_q_table[content_features][action] = (1 - alpha) * current_q + alpha * reward

        return action == 'cache'

    def get_cache_hit_rate(self, total_requests: int) -> float:
        """获取缓存命中率"""
        hits = sum(1 for meta in self.cache_metadata.values() if meta.get('access_count', 0) > 1)
        if total_requests == 0:
            return 0.0
        return hits / total_requests


# =============================================================================
# FederatedEdge - Federated Learning on Edge
# =============================================================================

class FederatedEdge:
    """
    边缘联邦学习类

    实现分层联邦学习、异步联邦学习和聚类联邦学习。
    """

    def __init__(self, config: EdgeConfig = None):
        self.config = config or EdgeConfig()

        # Model state
        self.global_model: Optional[Dict] = None
        self.edge_models: Dict[str, Dict] = {}  # server_id -> model
        self.local_models: Dict[str, Dict] = {}  # device_id -> model

        # Aggregation weights
        self.aggregation_weights: Dict[str, float] = {}

        # Async FL state
        self.staleness_tolerance = 5
        self.pending_updates: List[Dict] = []

        # Clustered FL state
        self.clusters: Dict[str, List[str]] = {}  # cluster_id -> device_ids
        self.cluster_models: Dict[str, Dict] = {}

    def initialize_model(self, model_params: Dict):
        """初始化全局模型"""
        self.global_model = copy.deepcopy(model_params)

    def hierarchical_fl_round(
        self,
        edge_servers: List[EdgeServer],
        devices: List[EdgeDevice],
        local_epochs: int = 5
    ) -> Dict:
        """
        分层联邦学习 (Device-Edge-Cloud)

        1. Devices train locally
        2. Edge servers aggregate from devices
        3. Cloud aggregates from edge servers
        """
        # Step 1: Local training at devices
        device_updates = {}
        for device in devices:
            update = self._local_train(device, local_epochs)
            device_updates[device.device_id] = update

        # Step 2: Edge aggregation
        for server in edge_servers:
            connected_devices = [
                d for d in devices
                if d.device_id in server.connected_devices
            ]

            if connected_devices:
                updates = [device_updates[d.device_id] for d in connected_devices]
                weights = [1.0 / len(updates)] * len(updates)
                self.edge_models[server.server_id] = self._aggregate(updates, weights)

        # Step 3: Cloud aggregation
        if self.edge_models:
            updates = list(self.edge_models.values())
            weights = [1.0 / len(updates)] * len(updates)
            self.global_model = self._aggregate(updates, weights)

        return self.global_model

    def _local_train(self, device: EdgeDevice, epochs: int) -> Dict:
        """模拟本地训练"""
        # Simplified: return perturbed model
        if device.device_id not in self.local_models:
            self.local_models[device.device_id] = copy.deepcopy(self.global_model)

        # Simulate training by adding small noise
        update = {}
        for key, value in self.local_models[device.device_id].items():
            if isinstance(value, (int, float)):
                update[key] = value + random.gauss(0, 0.01)
            else:
                update[key] = value

        return update

    def _aggregate(self, updates: List[Dict], weights: List[float]) -> Dict:
        """加权聚合模型更新"""
        if not updates:
            return {}

        aggregated = {}
        for key in updates[0].keys():
            weighted_sum = sum(
                u[key] * w for u, w in zip(updates, weights)
                if isinstance(u[key], (int, float))
            )
            aggregated[key] = weighted_sum

        return aggregated

    def async_federated_edge(
        self,
        device: EdgeDevice,
        update: Dict,
        timestamp: float
    ) -> Optional[Dict]:
        """
        异步联邦边缘学习

        处理延迟到达的更新。
        """
        # Add to pending updates
        self.pending_updates.append({
            'device_id': device.device_id,
            'update': update,
            'timestamp': timestamp,
            'staleness': 0
        })

        # Check staleness
        current_time = time.time()
        valid_updates = []

        for pu in self.pending_updates:
            pu['staleness'] = current_time - pu['timestamp']
            if pu['staleness'] <= self.staleness_tolerance:
                valid_updates.append(pu['update'])

        # Aggregate if enough valid updates
        if len(valid_updates) >= 3:
            weights = [1.0 / len(valid_updates)] * len(valid_updates)
            self.global_model = self._aggregate(valid_updates, weights)
            self.pending_updates = []
            return self.global_model

        return None

    def clustered_federated_learning(
        self,
        devices: List[EdgeDevice],
        num_clusters: int
    ) -> Dict[str, Dict]:
        """
        聚类联邦学习

        根据数据分布将设备分组。
        """
        # Simplified clustering based on location
        self.clusters = {}
        cluster_size = len(devices) // num_clusters

        for i in range(num_clusters):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < num_clusters - 1 else len(devices)
            cluster_devices = devices[start_idx:end_idx]
            self.clusters[f'cluster_{i}'] = [d.device_id for d in cluster_devices]

        # Train cluster models
        for cluster_id, device_ids in self.clusters.items():
            cluster_updates = [
                self._local_train(next(d for d in devices if d.device_id == did), 5)
                for did in device_ids
            ]
            weights = [1.0 / len(cluster_updates)] * len(cluster_updates)
            self.cluster_models[cluster_id] = self._aggregate(cluster_updates, weights)

        return self.cluster_models

    def get_federated_metrics(self) -> Dict[str, Any]:
        """获取联邦学习指标"""
        return {
            'global_model_size': len(str(self.global_model)) if self.global_model else 0,
            'edge_models': len(self.edge_models),
            'local_models': len(self.local_models),
            'clusters': len(self.clusters),
            'pending_updates': len(self.pending_updates)
        }


# =============================================================================
# EdgeSimulator - Full Edge Computing Simulation
# =============================================================================

class EdgeSimulator:
    """
    边缘计算完整模拟器

    模拟网络拓扑、工作负载生成和性能评估。
    """

    def __init__(self, config: EdgeConfig = None):
        self.config = config or EdgeConfig()

        # Network components
        self.devices: Dict[str, EdgeDevice] = {}
        self.servers: Dict[str, EdgeServer] = {}
        self.cloud: Optional[EdgeServer] = None

        # Algorithms
        self.offloader = TaskOffloading(config)
        self.resource_allocator = ResourceAllocation(config)
        self.mobility_manager = MobilityManagement(config)
        self.caching = CachingStrategy(config.cache_size, config)
        self.federated = FederatedEdge(config)

        # Simulation state
        self.current_time = 0.0
        self.time_step = 1.0
        self.is_running = False

        # Metrics
        self.metrics_history: List[Dict] = []
        self.task_completion_log: List[Dict] = []

    def create_topology(
        self,
        num_devices: int = 10,
        num_servers: int = 3,
        area_size: float = 1000.0
    ):
        """创建网络拓扑"""
        # Create cloud
        self.cloud = EdgeServer(
            'cloud',
            self.config,
            compute_capacity=100e9,
            location=(area_size/2, area_size/2),
            coverage_radius=area_size * 2
        )

        # Create edge servers
        for i in range(num_servers):
            angle = 2 * math.pi * i / num_servers
            radius = area_size / 3
            x = area_size/2 + radius * math.cos(angle)
            y = area_size/2 + radius * math.sin(angle)

            server = EdgeServer(
                f'server_{i}',
                self.config,
                location=(x, y),
                coverage_radius=area_size / 2
            )
            self.servers[server.server_id] = server

        # Create devices
        for i in range(num_devices):
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)

            device = EdgeDevice(
                f'device_{i}',
                self.config,
                location=(x, y)
            )
            self.devices[device.device_id] = device

            # Connect to nearest server
            nearest_server = self._find_nearest_server(device)
            if nearest_server:
                nearest_server.connect_device(device)

    def _find_nearest_server(self, device: EdgeDevice) -> Optional[EdgeServer]:
        """找到最近的边缘服务器"""
        nearest = None
        min_distance = float('inf')

        for server in self.servers.values():
            distance = math.sqrt(
                (device.location[0] - server.location[0])**2 +
                (device.location[1] - server.location[1])**2
            )
            if distance < min_distance and distance <= server.coverage_radius:
                min_distance = distance
                nearest = server

        return nearest

    def generate_workload(
        self,
        task_rate: float = 1.0,  # tasks per second per device
        data_size_range: Tuple[float, float] = (1e6, 10e6),  # 1-10 Mb
        compute_range: Tuple[float, float] = (1e8, 1e9),  # 0.1-1 Gcycles
        deadline_range: Tuple[float, float] = (0.5, 2.0)  # 0.5-2 seconds
    ) -> List[Task]:
        """生成工作负载"""
        tasks = []

        for device in self.devices.values():
            if not device.is_online:
                continue

            # Poisson arrival (using numpy-style poisson via simple approximation)
            # For small lambda, use direct method
            L = math.exp(-task_rate)
            p = 1.0
            num_tasks = 0
            while p > L:
                num_tasks += 1
                p *= random.random()
            num_tasks -= 1

            for _ in range(num_tasks):
                task = Task(
                    task_id=f'task_{self.current_time}_{random.randint(0, 10000)}',
                    data_size=random.uniform(*data_size_range),
                    compute_requirement=random.uniform(*compute_range),
                    deadline=random.uniform(*deadline_range),
                    priority=random.randint(1, 10),
                    arrival_time=self.current_time
                )
                tasks.append((device, task))

        return tasks

    def run_simulation(
        self,
        duration: float = 100.0,
        algorithm: str = 'lyapunov'
    ) -> Dict[str, Any]:
        """运行模拟"""
        self.is_running = True
        num_steps = int(duration / self.time_step)

        for step in range(num_steps):
            self.current_time = step * self.time_step

            # 1. Generate workload
            new_tasks = self.generate_workload()

            # 2. Make offloading decisions
            for device, task in new_tasks:
                server = self._find_nearest_server(device)

                decision = self.offloader.make_offloading_decision(
                    task, device, server, self.current_time, algorithm
                )

                if decision == 'local':
                    device.submit_task(task)
                elif decision == 'edge' and server:
                    server.submit_task(task, device.device_id)
                else:
                    # Cloud offloading (treat cloud as special server)
                    if self.cloud:
                        self.cloud.submit_task(task, device.device_id)

            # 3. Process tasks at devices
            for device in self.devices.values():
                completed = device.process_tasks(self.time_step, self.current_time)
                for task in completed:
                    self.task_completion_log.append({
                        'task_id': task.task_id,
                        'device_id': device.device_id,
                        'decision': task.offloading_decision,
                        'latency': task.get_latency(),
                        'deadline_met': task.is_deadline_met(task.completion_time)
                    })

            # 4. Process tasks at servers
            for server in self.servers.values():
                completed = server.process_tasks(self.time_step, self.current_time)
                for task in completed:
                    self.task_completion_log.append({
                        'task_id': task.task_id,
                        'server_id': server.server_id,
                        'decision': 'edge',
                        'latency': task.get_latency(),
                        'deadline_met': task.is_deadline_met(task.completion_time)
                    })

            # 5. Handle mobility
            self._update_mobility()

            # 6. Collect metrics
            if step % 10 == 0:
                self._collect_metrics()

        self.is_running = False
        return self._get_final_metrics()

    def _update_mobility(self):
        """更新设备移动性"""
        for device in self.devices.values():
            # Random walk mobility model
            speed = random.uniform(0, 5)  # m/s
            direction = random.uniform(0, 2 * math.pi)

            dx = speed * self.time_step * math.cos(direction)
            dy = speed * self.time_step * math.sin(direction)

            new_x = device.location[0] + dx
            new_y = device.location[1] + dy

            device.update_location((new_x, new_y))

            # Check handover
            current_server = None
            for server in self.servers.values():
                if device.device_id in server.connected_devices:
                    current_server = server
                    break

            if current_server:
                target = self.mobility_manager.check_handover_needed(
                    device, current_server, list(self.servers.values())
                )
                if target:
                    self.mobility_manager.perform_handover(
                        device, current_server, target, self.current_time
                    )

    def _collect_metrics(self):
        """收集性能指标"""
        metrics = {
            'time': self.current_time,
            'device_metrics': [d.get_metrics() for d in self.devices.values()],
            'server_metrics': [s.get_metrics() for s in self.servers.values()],
            'handover_count': len(self.mobility_manager.handover_history)
        }
        self.metrics_history.append(metrics)

    def _get_final_metrics(self) -> Dict[str, Any]:
        """获取最终性能指标"""
        total_tasks = len(self.task_completion_log)
        if total_tasks == 0:
            return {}

        latencies = [t['latency'] for t in self.task_completion_log if t['latency']]
        deadline_met = sum(1 for t in self.task_completion_log if t.get('deadline_met'))

        # Offloading distribution
        offloading_counts = defaultdict(int)
        for t in self.task_completion_log:
            offloading_counts[t['decision']] += 1

        return {
            'total_tasks': total_tasks,
            'average_latency': sum(latencies) / len(latencies) if latencies else 0,
            'max_latency': max(latencies) if latencies else 0,
            'min_latency': min(latencies) if latencies else 0,
            'deadline_meet_rate': deadline_met / total_tasks,
            'offloading_distribution': dict(offloading_counts),
            'handover_count': len(self.mobility_manager.handover_history),
            'total_energy_consumed': sum(
                d.total_energy_consumed for d in self.devices.values()
            )
        }

    def get_summary(self) -> str:
        """获取模拟摘要"""
        lines = [
            "=" * 60,
            "Edge Computing Simulation Summary",
            "=" * 60,
            f"Devices: {len(self.devices)}",
            f"Edge Servers: {len(self.servers)}",
            f"Simulation Time: {self.current_time:.2f}s",
            "",
            "Components:",
            "  - TaskOffloading: Lyapunov, Q-Learning, Game Theory",
            "  - ResourceAllocation: Proportional Share, Max-Min Fairness",
            "  - MobilityManagement: Handover, Predictive Migration, MDP",
            "  - CachingStrategy: LRU, LFU, FIFO, RL-based",
            "  - FederatedEdge: Hierarchical, Async, Clustered FL",
            "=" * 60
        ]
        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_simulation() -> EdgeSimulator:
    """创建默认模拟配置"""
    config = EdgeConfig()
    simulator = EdgeSimulator(config)
    simulator.create_topology(num_devices=20, num_servers=4)
    return simulator


def run_comparison_experiment() -> Dict[str, Dict]:
    """运行不同卸载算法的对比实验"""
    algorithms = ['local', 'edge', 'cloud', 'lyapunov', 'q_learning']
    results = {}

    for algo in algorithms:
        simulator = create_default_simulation()
        metrics = simulator.run_simulation(duration=50.0, algorithm=algo)
        results[algo] = metrics

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Initializing Edge Computing Simulation...")

    # Create simulator
    simulator = create_default_simulation()
    print(simulator.get_summary())

    # Run simulation
    print("\nRunning simulation with Lyapunov optimization...")
    metrics = simulator.run_simulation(duration=30.0, algorithm='lyapunov')

    # Print results
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Run comparison
    print("\n\nRunning comparison experiment...")
    comparison = run_comparison_experiment()

    print("\nComparison Results:")
    print(f"{'Algorithm':<15} {'Avg Latency':<15} {'Deadline Met':<15}")
    print("-" * 45)
    for algo, result in comparison.items():
        avg_lat = result.get('average_latency', 0)
        deadline_met = result.get('deadline_meet_rate', 0)
        print(f"{algo:<15} {avg_lat:<15.4f} {deadline_met:<15.2%}")
