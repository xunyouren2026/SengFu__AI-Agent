"""
优先级经验回放模块 (Prioritized Experience Replay Module)

该模块实现了基于优先级的经验存储和采样机制，用于持续学习和强化学习。
使用SumTree数据结构实现高效的O(log n)采样，支持重要性采样权重计算
以修正优先级采样带来的偏差。

核心功能:
- PrioritizedReplayBuffer类：基于优先级的经验存储和采样
- SumTree数据结构：高效O(log n)优先级采样
- 重要性采样权重：修正优先级偏置
- 动态优先级更新：基于TD误差
- 与StateBus集成，支持胜复学架构

关键算法:
- 优先级计算: p_i = |δ_i| + ε，δ_i是TD误差
- 采样概率: P(i) = p_i^α / Σ p_j^α
- 重要性权重: w_i = (N * P(i))^(-β)
- SumTree：完全二叉树，父节点=子节点和

作者: AGI Universal Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
import random
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Iterator
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import warnings

# 尝试导入state_bus用于集成
try:
    from ..core.state_bus import StateBus, EventType, get_global_state_bus, StateCoefficient
    _HAS_STATE_BUS = True
except ImportError:
    _HAS_STATE_BUS = False


# ============================================================================
# 类型定义
# ============================================================================

class BufferType(Enum):
    """回放缓冲区类型枚举"""
    PRIORITIZED = "prioritized"    # 优先级回放
    UNIFORM = "uniform"            # 均匀回放
    MULTI_STEP = "multi_step"      # 多步回放
    RESERVOIR = "reservoir"        # 水库采样


class SamplingStrategy(Enum):
    """采样策略枚举"""
    PROPORTIONAL = "proportional"  # 比例采样
    RANK_BASED = "rank_based"      # 基于排名的采样


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Experience:
    """
    经验数据类
    
    存储单次交互的经验，包括状态、动作、奖励、下一状态等。
    支持任意形状的状态和动作。
    
    Attributes:
        state: 当前状态 (任意形状)
        action: 执行的动作 (标量或数组)
        reward: 获得的奖励 (标量)
        next_state: 下一个状态 (与state相同形状)
        done: 是否结束 (布尔值)
        info: 额外信息字典
        priority: 经验优先级
        timestamp: 经验时间戳
        episode_id: 所属episode标识
        step_id: 步骤标识
    
    示例:
        >>> exp = Experience(
        ...     state=np.array([1.0, 2.0, 3.0]),
        ...     action=0,
        ...     reward=1.0,
        ...     next_state=np.array([1.1, 2.1, 3.1]),
        ...     done=False
        ... )
    """
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)
    episode_id: Optional[int] = None
    step_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含所有字段的字典
        """
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'info': self.info.copy(),
            'priority': self.priority,
            'timestamp': self.timestamp,
            'episode_id': self.episode_id,
            'step_id': self.step_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """
        从字典创建Experience实例
        
        Args:
            data: 包含经验数据的字典
            
        Returns:
            Experience实例
        """
        return cls(
            state=np.array(data['state']),
            action=data['action'],
            reward=data['reward'],
            next_state=np.array(data['next_state']),
            done=data['done'],
            info=data.get('info', {}),
            priority=data.get('priority', 1.0),
            timestamp=data.get('timestamp', time.time()),
            episode_id=data.get('episode_id'),
            step_id=data.get('step_id')
        )
    
    def compute_td_error(
        self,
        q_values: np.ndarray,
        next_q_values: np.ndarray,
        gamma: float = 0.99
    ) -> float:
        """
        计算TD误差
        
        Args:
            q_values: 当前状态Q值 [num_actions]
            next_q_values: 下一状态Q值 [num_actions]
            gamma: 折扣因子
            
        Returns:
            TD误差绝对值
        """
        if isinstance(self.action, int):
            current_q = q_values[self.action]
        else:
            current_q = np.sum(q_values * self.action)
        
        next_q = np.max(next_q_values)
        target = self.reward + gamma * next_q * (1 - float(self.done))
        td_error = abs(target - current_q)
        
        return td_error


@dataclass
class BatchSample:
    """
    批次采样结果数据类
    
    Attributes:
        states: 状态批次
        actions: 动作批次
        rewards: 奖励批次
        next_states: 下一状态批次
        dones: 结束标志批次
        infos: 信息列表
        weights: 重要性采样权重
        indices: 数据索引
        priorities: 优先级
    """
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]]
    weights: np.ndarray
    indices: np.ndarray
    priorities: Optional[np.ndarray] = None
    
    def to_tensor(self, device: str = 'cpu') -> Dict[str, Any]:
        """
        转换为PyTorch张量
        
        Args:
            device: 目标设备
            
        Returns:
            包含张量的字典
        """
        import torch
        
        return {
            'states': torch.from_numpy(self.states).float().to(device),
            'actions': torch.from_numpy(self.actions).long().to(device),
            'rewards': torch.from_numpy(self.rewards).float().to(device),
            'next_states': torch.from_numpy(self.next_states).float().to(device),
            'dones': torch.from_numpy(self.dones).float().to(device),
            'weights': torch.from_numpy(self.weights).float().to(device),
            'indices': torch.from_numpy(self.indices).long().to(device),
            'infos': self.infos
        }


# ============================================================================
# SumTree数据结构
# ============================================================================

class SumTree:
    """
    SumTree数据结构
    
    用于高效实现优先经验回放的采样操作。SumTree是一种完全二叉树，
    其中父节点的值等于子节点值的和。支持O(log n)时间的采样和更新操作。
    
    树结构:
    - 叶子节点存储实际数据的优先级
    - 内部节点存储子树的优先级和
    - 根节点存储所有优先级的总和
    
    Attributes:
        capacity: 树的容量（叶子节点数）
        tree: 树结构数组 [2*capacity - 1]
        data_pointer: 当前数据写入位置
    
    示例:
        >>> tree = SumTree(capacity=1000)
        >>> tree.add(priority=1.0, data_idx=0)
        >>> tree.add(priority=2.0, data_idx=1)
        >>> tree_idx, priority, data_idx = tree.get(1.5)
        >>> print(f"采样到数据索引: {data_idx}")
    """
    
    def __init__(self, capacity: int):
        """
        初始化SumTree
        
        Args:
            capacity: 树的容量（必须是正整数）
            
        Raises:
            ValueError: 当capacity不是正整数时
        """
        if capacity <= 0:
            raise ValueError(f"capacity必须是正整数，当前值: {capacity}")
        
        self.capacity = capacity
        # 树的大小为2*capacity - 1（完全二叉树）
        # 索引0是根节点，索引capacity-1到2*capacity-2是叶子节点
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self._lock = threading.Lock()
    
    def _propagate(self, idx: int, change: float) -> None:
        """
        向上传播优先级变化
        
        从叶子节点向上更新到根节点。
        
        Args:
            idx: 叶子节点索引（在tree数组中的位置）
            change: 优先级变化值（新值 - 旧值）
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        根据优先级值检索叶子节点
        
        从根节点开始，根据累积优先级值找到对应的叶子节点。
        
        Args:
            idx: 当前节点索引
            s: 优先级采样值
            
        Returns:
            叶子节点索引
        """
        left = 2 * idx + 1
        right = left + 1
        
        # 到达叶子节点层
        if left >= len(self.tree):
            return idx
        
        # 根据累积值选择左子树或右子树
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """
        获取总优先级
        
        Returns:
            根节点的值，即所有优先级之和
        """
        with self._lock:
            return self.tree[0]
    
    def add(self, priority: float, data_idx: int) -> None:
        """
        添加或更新优先级
        
        Args:
            priority: 优先级值（必须为正数）
            data_idx: 数据索引（0到capacity-1）
            
        Raises:
            ValueError: 当priority不是正数或data_idx越界时
        """
        if priority <= 0:
            raise ValueError(f"priority必须是正数，当前值: {priority}")
        if data_idx < 0 or data_idx >= self.capacity:
            raise ValueError(f"data_idx必须在[0, {self.capacity})范围内，当前值: {data_idx}")
        
        with self._lock:
            # 计算叶子节点索引
            tree_idx = data_idx + self.capacity - 1
            
            # 计算变化值
            change = priority - self.tree[tree_idx]
            
            # 更新叶子节点
            self.tree[tree_idx] = priority
            
            # 向上传播变化
            self._propagate(tree_idx, change)
    
    def get(self, s: float) -> Tuple[int, float, int]:
        """
        根据优先级值获取数据索引
        
        Args:
            s: 优先级采样值（必须在[0, total()]范围内）
            
        Returns:
            (树索引, 优先级值, 数据索引)元组
            
        Raises:
            ValueError: 当s不在有效范围内时
        """
        with self._lock:
            total = self.tree[0]
            if total == 0:
                raise ValueError("SumTree为空，无法采样")
            if s < 0 or s > total:
                raise ValueError(f"采样值s={s}必须在[0, {total}]范围内")
            
            tree_idx = self._retrieve(0, s)
            data_idx = tree_idx - self.capacity + 1
            
            return tree_idx, self.tree[tree_idx], data_idx
    
    def update(self, tree_idx: int, priority: float) -> None:
        """
        更新指定树索引的优先级
        
        Args:
            tree_idx: 树索引（在tree数组中的位置）
            priority: 新优先级值
            
        Raises:
            ValueError: 当priority不是正数时
        """
        if priority <= 0:
            raise ValueError(f"priority必须是正数，当前值: {priority}")
        
        with self._lock:
            change = priority - self.tree[tree_idx]
            self.tree[tree_idx] = priority
            self._propagate(tree_idx, change)
    
    def get_priority(self, data_idx: int) -> float:
        """
        获取数据索引对应的优先级
        
        Args:
            data_idx: 数据索引
            
        Returns:
            优先级值
        """
        with self._lock:
            tree_idx = data_idx + self.capacity - 1
            return self.tree[tree_idx]
    
    def get_max_priority(self) -> float:
        """
        获取最大优先级
        
        Returns:
            所有叶子节点中的最大优先级
        """
        with self._lock:
            leaf_nodes = self.tree[self.capacity - 1:]
            return float(np.max(leaf_nodes)) if len(leaf_nodes) > 0 else 1.0
    
    def get_min_priority(self) -> float:
        """
        获取最小非零优先级
        
        Returns:
            所有非零叶子节点中的最小优先级
        """
        with self._lock:
            leaf_nodes = self.tree[self.capacity - 1:]
            non_zero = leaf_nodes[leaf_nodes > 0]
            return float(np.min(non_zero)) if len(non_zero) > 0 else 1.0
    
    def get_mean_priority(self) -> float:
        """
        获取平均优先级
        
        Returns:
            所有非零叶子节点的平均优先级
        """
        with self._lock:
            leaf_nodes = self.tree[self.capacity - 1:]
            non_zero = leaf_nodes[leaf_nodes > 0]
            return float(np.mean(non_zero)) if len(non_zero) > 0 else 1.0
    
    def __len__(self) -> int:
        """
        获取非零优先级数量
        
        Returns:
            具有非零优先级的叶子节点数量
        """
        with self._lock:
            leaf_nodes = self.tree[self.capacity - 1:]
            return int(np.sum(leaf_nodes > 0))


# ============================================================================
# 优先级经验回放缓冲区
# ============================================================================

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    实现基于优先级的经验回放机制，使用SumTree实现高效采样。
    支持重要性采样权重计算，用于纠正优先级采样带来的偏差。
    
    优先级计算:
    - p_i = |δ_i| + ε，其中δ_i是TD误差
    - 采样概率 P(i) = p_i^α / Σ p_j^α
    - 重要性权重 w_i = (N * P(i))^(-β)
    
    Attributes:
        capacity: 缓冲区容量
        alpha: 优先级指数（0=均匀采样，1=完全优先级）
        beta: 重要性采样指数
        beta_increment: beta增量（用于退火）
        epsilon: 小常数，避免零优先级
        sum_tree: SumTree实例
    
    示例:
        >>> buffer = PrioritizedReplayBuffer(
        ...     capacity=10000,
        ...     alpha=0.6,
        ...     beta=0.4,
        ...     state_shape=(4,)
        ... )
        >>> 
        >>> # 添加经验
        >>> for _ in range(100):
        ...     exp = Experience(
        ...         state=np.random.randn(4),
        ...         action=np.random.randint(0, 2),
        ...         reward=np.random.randn(),
        ...         next_state=np.random.randn(4),
        ...         done=False
        ...     )
        ...     buffer.add(exp)
        >>> 
        >>> # 采样
        >>> batch, weights, indices = buffer.sample(batch_size=32)
        >>> 
        >>> # 更新优先级（基于TD误差）
        >>> td_errors = np.random.randn(32)
        >>> buffer.update_priorities(indices, td_errors)
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        state_shape: Optional[Tuple[int, ...]] = None,
        action_shape: Optional[Tuple[int, ...]] = None,
        state_bus: Optional[Any] = None
    ):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量（正整数）
            alpha: 优先级指数（0=均匀采样，1=完全优先级）
            beta: 重要性采样指数（初始值）
            beta_increment: beta增量，用于退火到1.0
            epsilon: 小常数，避免零优先级
            state_shape: 状态形状（可选，用于预分配内存）
            action_shape: 动作形状（可选，用于预分配内存）
            state_bus: 状态总线实例，用于与胜复学架构集成
            
        Raises:
            ValueError: 当参数无效时
        """
        if capacity <= 0:
            raise ValueError(f"capacity必须是正整数，当前值: {capacity}")
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha必须在[0, 1]范围内，当前值: {alpha}")
        if not (0 <= beta <= 1):
            raise ValueError(f"beta必须在[0, 1]范围内，当前值: {beta}")
        
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.state_shape = state_shape
        self.action_shape = action_shape
        self._state_bus = state_bus
        
        # 初始化SumTree
        self.sum_tree = SumTree(capacity)
        
        # 数据存储（使用动态列表，支持任意形状）
        self._states: List[np.ndarray] = [None] * capacity
        self._actions: List[Union[int, np.ndarray]] = [None] * capacity
        self._rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._next_states: List[np.ndarray] = [None] * capacity
        self._dones: np.ndarray = np.zeros(capacity, dtype=np.bool_)
        self._infos: List[Dict[str, Any]] = [None] * capacity
        self._timestamps: np.ndarray = np.zeros(capacity, dtype=np.float64)
        
        # 当前大小和位置
        self._size = 0
        self._position = 0
        
        # 统计信息
        self._total_added = 0
        self._total_sampled = 0
        self._total_updated = 0
        
        # 线程锁
        self._lock = threading.RLock()
    
    def add(
        self,
        experience: Experience,
        priority: Optional[float] = None
    ) -> int:
        """
        添加经验到缓冲区
        
        如果没有指定优先级，使用当前最大优先级，确保新经验被充分采样。
        
        Args:
            experience: 经验对象
            priority: 优先级（如果为None，使用最大优先级）
            
        Returns:
            数据索引
            
        Raises:
            ValueError: 当experience无效时
        """
        if not isinstance(experience, Experience):
            raise ValueError(f"experience必须是Experience类型，当前类型: {type(experience)}")
        
        with self._lock:
            # 如果没有指定优先级，使用最大优先级
            if priority is None:
                priority = self.sum_tree.get_max_priority()
            
            # 确保优先级为正
            priority = max(priority, self.epsilon)
            
            # 存储数据
            idx = self._position
            self._states[idx] = experience.state.copy()
            self._actions[idx] = experience.action
            self._rewards[idx] = experience.reward
            self._next_states[idx] = experience.next_state.copy()
            self._dones[idx] = experience.done
            self._infos[idx] = experience.info.copy() if experience.info else {}
            self._timestamps[idx] = experience.timestamp
            
            # 更新SumTree（使用alpha指数）
            priority_alpha = priority ** self.alpha
            self.sum_tree.add(priority_alpha, idx)
            
            # 更新位置和大小
            self._position = (self._position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            self._total_added += 1
            
            # 发布到状态总线
            self._publish_to_state_bus("add", {"size": self._size})
            
            return idx
    
    def sample(
        self,
        batch_size: int,
        device: Optional[str] = None
    ) -> BatchSample:
        """
        采样一批经验
        
        使用SumTree进行基于优先级的采样，并计算重要性采样权重。
        
        Args:
            batch_size: 批次大小
            device: 设备（预留参数，当前未使用）
            
        Returns:
            BatchSample对象，包含经验数据和重要性权重
            
        Raises:
            ValueError: 当batch_size无效或缓冲区为空时
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size必须是正整数，当前值: {batch_size}")
        
        with self._lock:
            if self._size == 0:
                raise ValueError("缓冲区为空，无法采样")
            
            if self._size < batch_size:
                batch_size = self._size
                warnings.warn(f"缓冲区大小({self._size})小于batch_size，调整为{batch_size}")
            
            # 存储采样结果
            indices = np.zeros(batch_size, dtype=np.int32)
            priorities = np.zeros(batch_size, dtype=np.float64)
            
            # 将总优先级分成batch_size个区间
            total_priority = self.sum_tree.total()
            segment_size = total_priority / batch_size
            
            # 从每个区间采样（分层采样）
            for i in range(batch_size):
                low = segment_size * i
                high = segment_size * (i + 1)
                value = random.uniform(low, high)
                
                try:
                    tree_idx, priority, data_idx = self.sum_tree.get(value)
                    indices[i] = data_idx
                    priorities[i] = priority
                except ValueError as e:
                    # 如果采样失败，随机选择一个有效索引
                    data_idx = random.randint(0, self._size - 1)
                    indices[i] = data_idx
                    priorities[i] = self.sum_tree.get_priority(data_idx)
            
            # 计算重要性采样权重
            # w_i = (N * P(i))^(-beta) / max_w
            sampling_probs = priorities / total_priority
            
            # 最小优先级用于归一化
            min_priority = self.sum_tree.get_min_priority() ** self.alpha
            min_prob = min_priority / total_priority
            max_weight = (self._size * min_prob) ** (-self.beta)
            
            # 计算权重
            weights = (self._size * sampling_probs) ** (-self.beta)
            weights /= max_weight  # 归一化到[0, 1]
            
            # 准备经验数据
            states = np.stack([self._states[i] for i in indices])
            actions = np.array([self._actions[i] for i in indices])
            rewards = self._rewards[indices]
            next_states = np.stack([self._next_states[i] for i in indices])
            dones = self._dones[indices]
            infos = [self._infos[i] for i in indices]
            
            # 更新beta（退火）
            self.beta = min(1.0, self.beta + self.beta_increment)
            self._total_sampled += batch_size
            
            # 发布到状态总线
            self._publish_to_state_bus("sample", {"batch_size": batch_size})
            
            return BatchSample(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                infos=infos,
                weights=weights.astype(np.float32),
                indices=indices,
                priorities=priorities.astype(np.float32)
            )
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray
    ) -> None:
        """
        更新经验的优先级
        
        通常基于TD误差更新优先级：priority = |td_error| + epsilon
        
        Args:
            indices: 数据索引数组
            priorities: 新的优先级数组
            
        Raises:
            ValueError: 当输入无效时
        """
        if len(indices) != len(priorities):
            raise ValueError(f"indices和priorities长度必须相同: {len(indices)} vs {len(priorities)}")
        
        with self._lock:
            for idx, priority in zip(indices, priorities):
                # 确保索引有效
                if idx < 0 or idx >= self._size:
                    continue
                
                # 确保优先级为正
                priority = max(abs(priority) + self.epsilon, self.epsilon)
                priority_alpha = priority ** self.alpha
                
                # 更新SumTree
                tree_idx = idx + self.capacity - 1
                self.sum_tree.update(tree_idx, priority_alpha)
                
                self._total_updated += 1
    
    def update_priority_from_td_errors(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        基于TD误差更新优先级
        
        便捷方法，自动计算优先级：priority = |td_error| + epsilon
        
        Args:
            indices: 数据索引数组
            td_errors: TD误差数组
        """
        priorities = np.abs(td_errors) + self.epsilon
        self.update_priorities(indices, priorities)
    
    def __len__(self) -> int:
        """
        获取当前缓冲区大小
        
        Returns:
            当前存储的经验数量
        """
        with self._lock:
            return self._size
    
    def is_full(self) -> bool:
        """
        检查缓冲区是否已满
        
        Returns:
            是否达到容量上限
        """
        with self._lock:
            return self._size >= self.capacity
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """
        检查缓冲区是否准备好采样
        
        Args:
            min_size: 最小所需大小
            
        Returns:
            是否准备好
        """
        with self._lock:
            return self._size >= min_size
    
    def clear(self) -> None:
        """
        清空缓冲区
        
        重置所有数据和统计信息。
        """
        with self._lock:
            self._size = 0
            self._position = 0
            self.sum_tree = SumTree(self.capacity)
            self._states = [None] * self.capacity
            self._actions = [None] * self.capacity
            self._rewards = np.zeros(self.capacity, dtype=np.float32)
            self._next_states = [None] * self.capacity
            self._dones = np.zeros(self.capacity, dtype=np.bool_)
            self._infos = [None] * self.capacity
            self._timestamps = np.zeros(self.capacity, dtype=np.float64)
            self._total_added = 0
            self._total_sampled = 0
            self._total_updated = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓冲区统计信息
        
        Returns:
            包含各种统计指标的字典
        """
        with self._lock:
            stats = {
                'capacity': self.capacity,
                'size': self._size,
                'fill_ratio': self._size / self.capacity,
                'total_added': self._total_added,
                'total_sampled': self._total_sampled,
                'total_updated': self._total_updated,
                'alpha': self.alpha,
                'beta': self.beta,
                'max_priority': float(self.sum_tree.get_max_priority()),
                'min_priority': float(self.sum_tree.get_min_priority()),
                'mean_priority': float(self.sum_tree.get_mean_priority()),
                'total_priority': float(self.sum_tree.total())
            }
            
            if self._size > 0:
                # 计算奖励统计
                valid_rewards = self._rewards[:self._size]
                stats['reward_stats'] = {
                    'mean': float(np.mean(valid_rewards)),
                    'std': float(np.std(valid_rewards)),
                    'min': float(np.min(valid_rewards)),
                    'max': float(np.max(valid_rewards))
                }
                
                # 计算结束率
                stats['done_ratio'] = float(np.mean(self._dones[:self._size]))
            
            return stats
    
    def get_experience(self, idx: int) -> Optional[Experience]:
        """
        获取指定索引的经验
        
        Args:
            idx: 索引
            
        Returns:
            经验对象，如果索引无效则返回None
        """
        with self._lock:
            if idx < 0 or idx >= self._size:
                return None
            
            return Experience(
                state=self._states[idx].copy(),
                action=self._actions[idx],
                reward=self._rewards[idx],
                next_state=self._next_states[idx].copy(),
                done=self._dones[idx],
                info=self._infos[idx].copy() if self._infos[idx] else {},
                priority=self.sum_tree.get_priority(idx),
                timestamp=self._timestamps[idx]
            )
    
    def get_recent_experiences(self, n: int) -> List[Experience]:
        """
        获取最近的n条经验
        
        Args:
            n: 经验数量
            
        Returns:
            经验列表（按时间倒序）
        """
        with self._lock:
            n = min(n, self._size)
            experiences = []
            
            for i in range(n):
                idx = (self._position - 1 - i) % self.capacity
                if idx < self._size:
                    exp = self.get_experience(idx)
                    if exp:
                        experiences.append(exp)
            
            return experiences
    
    def set_beta(self, beta: float) -> None:
        """
        设置beta值
        
        Args:
            beta: 新的beta值（必须在[0, 1]范围内）
        """
        with self._lock:
            self.beta = max(0.0, min(1.0, beta))
    
    def set_alpha(self, alpha: float) -> None:
        """
        设置alpha值
        
        Args:
            alpha: 新的alpha值（必须在[0, 1]范围内）
        """
        with self._lock:
            self.alpha = max(0.0, min(1.0, alpha))
    
    def _publish_to_state_bus(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        将事件发布到状态总线
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if self._state_bus is None or not _HAS_STATE_BUS:
            return
        
        try:
            if event_type == "add":
                # 根据缓冲区填充度更新复值
                fill_ratio = self._size / self.capacity
                self._state_bus.update_coefficient("balance", fill_ratio)
            elif event_type == "sample":
                # 采样时更新发值
                self._state_bus.update_coefficient("release", 0.7)
        except Exception:
            # 忽略状态总线错误
            pass


# ============================================================================
# 均匀经验回放缓冲区
# ============================================================================

class UniformReplayBuffer:
    """
    均匀经验回放缓冲区
    
    简单的均匀采样回放缓冲区，用于对比和基线测试。
    所有经验具有相同的采样概率。
    
    Attributes:
        capacity: 缓冲区容量
        buffer: 经验存储队列
    
    示例:
        >>> buffer = UniformReplayBuffer(capacity=1000)
        >>> buffer.add(experience)
        >>> batch, weights, indices = buffer.sample(32)
    """
    
    def __init__(
        self,
        capacity: int,
        state_bus: Optional[Any] = None
    ):
        """
        初始化均匀回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_bus: 状态总线实例
        """
        self.capacity = capacity
        self._state_bus = state_bus
        
        # 数据存储
        self._buffer: deque = deque(maxlen=capacity)
        
        # 统计
        self._total_added = 0
        self._total_sampled = 0
        
        # 线程锁
        self._lock = threading.RLock()
    
    def add(self, experience: Experience) -> int:
        """
        添加经验
        
        Args:
            experience: 经验对象
            
        Returns:
            当前缓冲区大小
        """
        with self._lock:
            self._buffer.append(experience)
            self._total_added += 1
            return len(self._buffer)
    
    def sample(
        self,
        batch_size: int
    ) -> BatchSample:
        """
        均匀采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            BatchSample对象
        """
        with self._lock:
            if len(self._buffer) == 0:
                raise ValueError("缓冲区为空，无法采样")
            
            if len(self._buffer) < batch_size:
                batch_size = len(self._buffer)
            
            # 均匀采样
            experiences = random.sample(list(self._buffer), batch_size)
            
            # 准备数据
            states = np.stack([e.state for e in experiences])
            actions = np.array([e.action for e in experiences])
            rewards = np.array([e.reward for e in experiences])
            next_states = np.stack([e.next_state for e in experiences])
            dones = np.array([e.done for e in experiences])
            infos = [e.info for e in experiences]
            
            # 均匀权重（全部为1）
            weights = np.ones(batch_size, dtype=np.float32)
            indices = np.arange(batch_size)
            
            self._total_sampled += batch_size
            
            return BatchSample(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                infos=infos,
                weights=weights,
                indices=indices
            )
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray
    ) -> None:
        """
        更新优先级（均匀缓冲区无操作）
        
        Args:
            indices: 索引
            priorities: 优先级
        """
        pass  # 均匀缓冲区不需要更新优先级
    
    def __len__(self) -> int:
        """获取缓冲区大小"""
        with self._lock:
            return len(self._buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
            self._total_added = 0
            self._total_sampled = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'capacity': self.capacity,
                'size': len(self._buffer),
                'fill_ratio': len(self._buffer) / self.capacity,
                'total_added': self._total_added,
                'total_sampled': self._total_sampled,
                'type': 'uniform'
            }


# ============================================================================
# 多步回放缓冲区
# ============================================================================

class MultiStepReplayBuffer(PrioritizedReplayBuffer):
    """
    多步回放缓冲区
    
    扩展优先回放缓冲区以支持n步回报计算。
    将连续的n步经验合并为一条经验，使用n步折扣回报。
    
    n步回报公式:
    R_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ... + γ^(n-1)*r_{t+n-1} + γ^n*max(Q(s_{t+n}))
    
    Attributes:
        n_step: n步数
        gamma: 折扣因子
        _n_step_buffer: n步经验缓冲区
    
    示例:
        >>> buffer = MultiStepReplayBuffer(
        ...     capacity=10000,
        ...     n_step=3,
        ...     gamma=0.99
        ... )
    """
    
    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        **kwargs
    ):
        """
        初始化多步回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            n_step: n步数
            gamma: 折扣因子
            **kwargs: 其他参数传递给父类
        """
        super().__init__(capacity, **kwargs)
        self.n_step = n_step
        self.gamma = gamma
        
        # n步缓冲区
        self._n_step_buffer: deque = deque(maxlen=n_step)
    
    def add(
        self,
        experience: Experience,
        priority: Optional[float] = None
    ) -> Optional[int]:
        """
        添加经验并计算n步回报
        
        当n步缓冲区满时，计算n步回报并添加到主缓冲区。
        
        Args:
            experience: 经验对象
            priority: 优先级
            
        Returns:
            数据索引（如果添加了n步经验）或None
        """
        self._n_step_buffer.append(experience)
        
        if len(self._n_step_buffer) < self.n_step:
            return None
        
        # 计算n步回报
        n_step_experience = self._compute_n_step_return()
        return super().add(n_step_experience, priority)
    
    def _compute_n_step_return(self) -> Experience:
        """
        计算n步回报
        
        Returns:
            n步经验对象
        """
        # 获取n步经验
        experiences = list(self._n_step_buffer)
        
        # 初始状态
        state = experiences[0].state
        action = experiences[0].action
        
        # 计算n步折扣回报
        reward = 0.0
        gamma_power = 1.0
        done = False
        
        for exp in experiences:
            reward += gamma_power * exp.reward
            gamma_power *= self.gamma
            
            if exp.done:
                done = True
                break
        
        # 最终状态
        last_exp = experiences[-1]
        next_state = last_exp.next_state
        info = last_exp.info
        
        return Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info
        )
    
    def on_episode_end(self) -> None:
        """
        Episode结束时处理剩余的经验
        
        将n步缓冲区中剩余的经验全部添加到主缓冲区。
        """
        while len(self._n_step_buffer) > 0:
            self._n_step_buffer.popleft()
            
            if len(self._n_step_buffer) > 0:
                n_step_experience = self._compute_n_step_return()
                super().add(n_step_experience)


# ============================================================================
# 水库采样缓冲区（用于持续学习）
# ============================================================================

class ReservoirReplayBuffer(UniformReplayBuffer):
    """
    水库采样回放缓冲区
    
    使用水库采样算法，适用于数据流场景和持续学习。
    保证每个经验被采样的概率相等，无论数据流长度如何。
    
    水库采样算法:
    - 前capacity个经验直接存储
    - 对于第i个经验（i > capacity），以capacity/i的概率替换已有经验
    
    示例:
        >>> buffer = ReservoirReplayBuffer(capacity=1000)
        >>> for i in range(10000):
        ...     buffer.add(experience)
    """
    
    def __init__(
        self,
        capacity: int,
        state_bus: Optional[Any] = None
    ):
        """
        初始化水库采样缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_bus: 状态总线实例
        """
        super().__init__(capacity, state_bus)
        self._total_seen = 0
    
    def add(self, experience: Experience) -> int:
        """
        添加经验（使用水库采样）
        
        Args:
            experience: 经验对象
            
        Returns:
            当前缓冲区大小
        """
        with self._lock:
            self._total_seen += 1
            
            if len(self._buffer) < self.capacity:
                # 缓冲区未满，直接添加
                self._buffer.append(experience)
            else:
                # 水库采样：以capacity/total_seen的概率替换
                idx = random.randint(0, self._total_seen - 1)
                if idx < self.capacity:
                    # 替换索引idx处的经验
                    self._buffer[idx] = experience
            
            self._total_added += 1
            return len(self._buffer)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = super().get_statistics()
            stats['total_seen'] = self._total_seen
            stats['type'] = 'reservoir'
            return stats


# ============================================================================
# 便捷函数
# ============================================================================

def create_replay_buffer(
    buffer_type: str = "prioritized",
    capacity: int = 100000,
    **kwargs
) -> Union[PrioritizedReplayBuffer, UniformReplayBuffer, MultiStepReplayBuffer, ReservoirReplayBuffer]:
    """
    创建回放缓冲区的便捷函数
    
    Args:
        buffer_type: 缓冲区类型 ("prioritized", "uniform", "multistep", "reservoir")
        capacity: 容量
        **kwargs: 其他参数
        
    Returns:
        回放缓冲区实例
        
    Raises:
        ValueError: 当buffer_type无效时
        
    示例:
        >>> buffer = create_replay_buffer(
        ...     buffer_type="prioritized",
        ...     capacity=10000,
        ...     alpha=0.6,
        ...     beta=0.4
        ... )
    """
    buffer_type = buffer_type.lower()
    
    if buffer_type == "prioritized":
        return PrioritizedReplayBuffer(capacity, **kwargs)
    elif buffer_type == "uniform":
        return UniformReplayBuffer(capacity, **kwargs)
    elif buffer_type in ["multistep", "multi_step"]:
        return MultiStepReplayBuffer(capacity, **kwargs)
    elif buffer_type == "reservoir":
        return ReservoirReplayBuffer(capacity, **kwargs)
    else:
        raise ValueError(f"未知的缓冲区类型: {buffer_type}")


def compute_td_error(
    q_value: float,
    reward: float,
    next_max_q: float,
    done: bool,
    gamma: float = 0.99
) -> float:
    """
    计算TD误差
    
    Args:
        q_value: 当前Q值
        reward: 奖励
        next_max_q: 下一状态最大Q值
        done: 是否结束
        gamma: 折扣因子
        
    Returns:
        TD误差
    """
    target = reward + gamma * next_max_q * (1 - float(done))
    return target - q_value


def create_per_buffer_with_state_bus(
    capacity: int = 100000,
    state_bus: Optional[Any] = None,
    **kwargs
) -> PrioritizedReplayBuffer:
    """
    创建与StateBus集成的PER缓冲区
    
    Args:
        capacity: 缓冲区容量
        state_bus: 状态总线实例（None则使用全局实例）
        **kwargs: 其他参数
        
    Returns:
        配置好的PrioritizedReplayBuffer实例
    """
    if state_bus is None and _HAS_STATE_BUS:
        state_bus = get_global_state_bus()
    
    return PrioritizedReplayBuffer(capacity, state_bus=state_bus, **kwargs)


# 导出公共接口
__all__ = [
    'Experience',
    'BatchSample',
    'SumTree',
    'PrioritizedReplayBuffer',
    'UniformReplayBuffer',
    'MultiStepReplayBuffer',
    'ReservoirReplayBuffer',
    'BufferType',
    'SamplingStrategy',
    'create_replay_buffer',
    'compute_td_error',
    'create_per_buffer_with_state_bus'
]
