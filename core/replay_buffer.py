"""
PER经验回放缓冲区模块 (Prioritized Experience Replay Buffer)

该模块实现了优先经验回放机制，支持TD误差优先级计算、SumTree高效采样、
重要性采样权重计算等功能。适用于强化学习中的样本高效利用。

核心功能:
- 优先经验回放
- TD误差优先级计算
- SumTree高效采样
- 重要性采样权重
- 动态缓冲区管理

作者: AGI Universal Framework Team
版本: 1.0.0
"""

import numpy as np
import random
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class Experience:
    """
    经验数据类
    
    存储单次交互的经验，包括状态、动作、奖励、下一状态等。
    
    Attributes:
        state: 当前状态
        action: 执行的动作
        reward: 获得的奖励
        next_state: 下一个状态
        done: 是否结束
        info: 额外信息
        priority: 经验优先级
        timestamp: 经验时间戳
    """
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'info': self.info,
            'priority': self.priority,
            'timestamp': self.timestamp
        }


class SumTree:
    """
    SumTree数据结构
    
    用于高效实现优先经验回放的采样操作。
    SumTree是一种二叉树，其中父节点的值等于子节点值的和。
    支持O(log n)时间的采样和更新操作。
    
    Attributes:
        capacity: 树的容量（叶子节点数）
        tree: 树结构数组
        data_pointer: 当前数据指针
    """
    
    def __init__(self, capacity: int):
        """
        初始化SumTree
        
        Args:
            capacity: 树的容量
        """
        self.capacity = capacity
        # 树的大小为2*capacity - 1（完全二叉树）
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self._lock = threading.Lock()
    
    def _propagate(self, idx: int, change: float) -> None:
        """
        向上传播变化
        
        Args:
            idx: 叶子节点索引
            change: 变化值
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        根据优先级值检索叶子节点
        
        Args:
            idx: 当前节点索引
            s: 优先级值
            
        Returns:
            叶子节点索引
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """获取总优先级"""
        with self._lock:
            return self.tree[0]
    
    def add(self, priority: float, data_idx: int) -> None:
        """
        添加优先级
        
        Args:
            priority: 优先级值
            data_idx: 数据索引
        """
        with self._lock:
            # 计算叶子节点索引
            tree_idx = data_idx + self.capacity - 1
            
            # 更新叶子节点
            change = priority - self.tree[tree_idx]
            self.tree[tree_idx] = priority
            
            # 向上传播
            self._propagate(tree_idx, change)
    
    def get(self, s: float) -> Tuple[int, float, int]:
        """
        根据优先级值获取数据索引
        
        Args:
            s: 优先级值
            
        Returns:
            (树索引, 优先级, 数据索引)元组
        """
        with self._lock:
            tree_idx = self._retrieve(0, s)
            data_idx = tree_idx - self.capacity + 1
            
            return tree_idx, self.tree[tree_idx], data_idx
    
    def update(self, tree_idx: int, priority: float) -> None:
        """
        更新优先级
        
        Args:
            tree_idx: 树索引
            priority: 新优先级值
        """
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
        """获取最大优先级"""
        with self._lock:
            leaf_nodes = self.tree[self.capacity - 1:]
            return np.max(leaf_nodes) if len(leaf_nodes) > 0 else 1.0
    
    def get_min_priority(self) -> float:
        """获取最小优先级"""
        with self._lock:
            leaf_nodes = self.tree[self.capacity - 1:]
            non_zero = leaf_nodes[leaf_nodes > 0]
            return np.min(non_zero) if len(non_zero) > 0 else 1.0


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    实现基于优先级的经验回放机制，使用SumTree实现高效采样。
    支持重要性采样权重计算，用于纠正优先级采样带来的偏差。
    
    Attributes:
        capacity: 缓冲区容量
        alpha: 优先级指数（0=均匀采样，1=完全优先级）
        beta: 重要性采样指数
        beta_increment: beta增量（用于退火）
        epsilon: 小常数，避免零优先级
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        state_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数（0=均匀采样，1=完全优先级）
            beta: 重要性采样指数
            beta_increment: beta增量，用于退火
            epsilon: 小常数，避免零优先级
            state_shape: 状态形状（可选）
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.state_shape = state_shape
        
        # 初始化SumTree
        self.sum_tree = SumTree(capacity)
        
        # 数据存储
        self.states = np.zeros((capacity,) + (state_shape or ()), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity,) + (state_shape or ()), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.infos: List[Dict[str, Any]] = [None] * capacity
        
        # 当前大小和位置
        self.size = 0
        self.position = 0
        
        # 统计信息
        self._total_added = 0
        self._total_sampled = 0
        
        # 线程锁
        self._lock = threading.RLock()
    
    def add(
        self,
        experience: Experience,
        priority: Optional[float] = None
    ) -> None:
        """
        添加经验到缓冲区
        
        Args:
            experience: 经验对象
            priority: 优先级（如果为None，使用最大优先级）
        """
        with self._lock:
            # 如果没有指定优先级，使用最大优先级
            if priority is None:
                priority = self.sum_tree.get_max_priority()
            
            # 确保优先级为正
            priority = max(priority, self.epsilon)
            
            # 存储数据
            idx = self.position
            self.states[idx] = experience.state
            self.actions[idx] = experience.action
            self.rewards[idx] = experience.reward
            self.next_states[idx] = experience.next_state
            self.dones[idx] = experience.done
            self.infos[idx] = experience.info
            
            # 更新SumTree
            priority_alpha = priority ** self.alpha
            self.sum_tree.add(priority_alpha, idx)
            
            # 更新位置和大小
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self._total_added += 1
    
    def sample(
        self,
        batch_size: int,
        device: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        采样一批经验
        
        Args:
            batch_size: 批次大小
            device: 设备（预留参数）
            
        Returns:
            (经验字典, 重要性采样权重, 数据索引)元组
        """
        with self._lock:
            if self.size < batch_size:
                batch_size = self.size
            
            # 存储采样结果
            indices = np.zeros(batch_size, dtype=np.int32)
            priorities = np.zeros(batch_size, dtype=np.float64)
            
            # 将总优先级分成batch_size个区间
            total_priority = self.sum_tree.total()
            segment_size = total_priority / batch_size
            
            # 从每个区间采样
            for i in range(batch_size):
                low = segment_size * i
                high = segment_size * (i + 1)
                value = random.uniform(low, high)
                
                tree_idx, priority, data_idx = self.sum_tree.get(value)
                
                indices[i] = data_idx
                priorities[i] = priority
            
            # 计算重要性采样权重
            # w_i = (N * P(i))^(-beta) / max_w
            sampling_probs = priorities / total_priority
            
            # 最小优先级用于归一化
            min_priority = self.sum_tree.get_min_priority() ** self.alpha
            min_prob = min_priority / total_priority
            max_weight = (self.size * min_prob) ** (-self.beta)
            
            # 计算权重
            weights = (self.size * sampling_probs) ** (-self.beta)
            weights /= max_weight  # 归一化
            
            # 准备经验数据
            batch = {
                'states': self.states[indices],
                'actions': self.actions[indices],
                'rewards': self.rewards[indices],
                'next_states': self.next_states[indices],
                'dones': self.dones[indices],
                'infos': [self.infos[i] for i in indices]
            }
            
            # 更新beta（退火）
            self.beta = min(1.0, self.beta + self.beta_increment)
            self._total_sampled += batch_size
            
            return batch, weights.astype(np.float32), indices
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray
    ) -> None:
        """
        更新经验的优先级
        
        Args:
            indices: 数据索引数组
            priorities: 新的优先级数组（通常是TD误差）
        """
        with self._lock:
            for idx, priority in zip(indices, priorities):
                # 确保优先级为正
                priority = max(abs(priority) + self.epsilon, self.epsilon)
                priority_alpha = priority ** self.alpha
                
                tree_idx = idx + self.capacity - 1
                self.sum_tree.update(tree_idx, priority_alpha)
    
    def update_priority_from_td_errors(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        基于TD误差更新优先级
        
        Args:
            indices: 数据索引数组
            td_errors: TD误差数组
        """
        priorities = np.abs(td_errors) + self.epsilon
        self.update_priorities(indices, priorities)
    
    def __len__(self) -> int:
        """获取当前缓冲区大小"""
        with self._lock:
            return self.size
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        with self._lock:
            return self.size >= self.capacity
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """
        检查缓冲区是否准备好采样
        
        Args:
            min_size: 最小所需大小
            
        Returns:
            是否准备好
        """
        with self._lock:
            return self.size >= min_size
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self.size = 0
            self.position = 0
            self.sum_tree = SumTree(self.capacity)
            self.states = np.zeros_like(self.states)
            self.actions = np.zeros_like(self.actions)
            self.rewards = np.zeros_like(self.rewards)
            self.next_states = np.zeros_like(self.next_states)
            self.dones = np.zeros_like(self.dones)
            self.infos = [None] * self.capacity
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓冲区统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'capacity': self.capacity,
                'size': self.size,
                'fill_ratio': self.size / self.capacity,
                'total_added': self._total_added,
                'total_sampled': self._total_sampled,
                'alpha': self.alpha,
                'beta': self.beta,
                'max_priority': float(self.sum_tree.get_max_priority()),
                'min_priority': float(self.sum_tree.get_min_priority()),
                'total_priority': float(self.sum_tree.total())
            }
    
    def get_experience(self, idx: int) -> Optional[Experience]:
        """
        获取指定索引的经验
        
        Args:
            idx: 索引
            
        Returns:
            经验对象，如果索引无效则返回None
        """
        with self._lock:
            if idx < 0 or idx >= self.size:
                return None
            
            return Experience(
                state=self.states[idx].copy(),
                action=self.actions[idx],
                reward=self.rewards[idx],
                next_state=self.next_states[idx].copy(),
                done=self.dones[idx],
                info=self.infos[idx].copy() if self.infos[idx] else {},
                priority=self.sum_tree.get_priority(idx)
            )
    
    def get_recent_experiences(self, n: int) -> List[Experience]:
        """
        获取最近的n条经验
        
        Args:
            n: 经验数量
            
        Returns:
            经验列表
        """
        with self._lock:
            n = min(n, self.size)
            experiences = []
            
            for i in range(n):
                idx = (self.position - 1 - i) % self.capacity
                if idx < self.size:
                    exp = self.get_experience(idx)
                    if exp:
                        experiences.append(exp)
            
            return experiences
    
    def set_beta(self, beta: float) -> None:
        """
        设置beta值
        
        Args:
            beta: 新的beta值
        """
        with self._lock:
            self.beta = max(0.0, min(1.0, beta))
    
    def set_alpha(self, alpha: float) -> None:
        """
        设置alpha值
        
        Args:
            alpha: 新的alpha值
        """
        with self._lock:
            self.alpha = max(0.0, min(1.0, alpha))


class UniformReplayBuffer:
    """
    均匀经验回放缓冲区
    
    简单的均匀采样回放缓冲区，用于对比和基线测试。
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        初始化均匀回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_shape: 状态形状
        """
        self.capacity = capacity
        self.state_shape = state_shape
        
        # 数据存储
        self.buffer: deque = deque(maxlen=capacity)
        
        # 统计
        self._total_added = 0
        self._total_sampled = 0
        
        # 线程锁
        self._lock = threading.RLock()
    
    def add(self, experience: Experience) -> None:
        """
        添加经验
        
        Args:
            experience: 经验对象
        """
        with self._lock:
            self.buffer.append(experience)
            self._total_added += 1
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        均匀采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (经验字典, 均匀权重, 索引)元组
        """
        with self._lock:
            if len(self.buffer) < batch_size:
                batch_size = len(self.buffer)
            
            # 均匀采样
            experiences = random.sample(list(self.buffer), batch_size)
            
            # 准备数据
            states = np.array([e.state for e in experiences])
            actions = np.array([e.action for e in experiences])
            rewards = np.array([e.reward for e in experiences])
            next_states = np.array([e.next_state for e in experiences])
            dones = np.array([e.done for e in experiences])
            
            batch = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones,
                'infos': [e.info for e in experiences]
            }
            
            # 均匀权重
            weights = np.ones(batch_size, dtype=np.float32)
            indices = np.arange(batch_size)
            
            self._total_sampled += batch_size
            
            return batch, weights, indices
    
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
            return len(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self.buffer.clear()
            self._total_added = 0
            self._total_sampled = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'capacity': self.capacity,
                'size': len(self.buffer),
                'fill_ratio': len(self.buffer) / self.capacity,
                'total_added': self._total_added,
                'total_sampled': self._total_sampled,
                'type': 'uniform'
            }


class MultiStepReplayBuffer(PrioritizedReplayBuffer):
    """
    多步回放缓冲区
    
    扩展优先回放缓冲区以支持n步回报计算。
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
    ) -> None:
        """
        添加经验并计算n步回报
        
        Args:
            experience: 经验对象
            priority: 优先级
        """
        self._n_step_buffer.append(experience)
        
        if len(self._n_step_buffer) < self.n_step:
            return
        
        # 计算n步回报
        n_step_experience = self._compute_n_step_return()
        super().add(n_step_experience, priority)
    
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
        
        for exp in experiences:
            reward += gamma_power * exp.reward
            gamma_power *= self.gamma
            
            if exp.done:
                break
        
        # 最终状态
        last_exp = experiences[-1]
        next_state = last_exp.next_state
        done = last_exp.done
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
         episode结束时处理剩余的经验
        """
        while len(self._n_step_buffer) > 0:
            self._n_step_buffer.popleft()
            
            if len(self._n_step_buffer) > 0:
                n_step_experience = self._compute_n_step_return()
                super().add(n_step_experience)


# 便捷函数
def create_replay_buffer(
    buffer_type: str = "prioritized",
    capacity: int = 100000,
    **kwargs
) -> Union[PrioritizedReplayBuffer, UniformReplayBuffer]:
    """
    创建回放缓冲区的便捷函数
    
    Args:
        buffer_type: 缓冲区类型 ("prioritized", "uniform", "multistep")
        capacity: 容量
        **kwargs: 其他参数
        
    Returns:
        回放缓冲区实例
    """
    if buffer_type == "prioritized":
        return PrioritizedReplayBuffer(capacity, **kwargs)
    elif buffer_type == "uniform":
        return UniformReplayBuffer(capacity, **kwargs)
    elif buffer_type == "multistep":
        return MultiStepReplayBuffer(capacity, **kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")
