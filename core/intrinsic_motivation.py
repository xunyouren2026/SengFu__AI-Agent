"""
内在动机驱动模块 (Intrinsic Motivation Engine)

该模块实现了多种内在奖励计算机制,用于驱动智能体的自主探索和学习。
包括计数型新奇奖励、信息增益奖励、能力进步奖励等,支持多动机融合。

主要特性:
- 计数型内在奖励 (novelty bonus)
- 信息增益奖励
- 能力进步奖励 (learning progress)
- 目标达成奖励
- 多动机融合与权重自适应

作者: AGI Unified Framework Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import hashlib
import json


@dataclass
class IntrinsicMotivationConfig:
    """内在动机配置类
    
    Attributes:
        motivation_types: 启用的动机类型列表
        weights: 各动机的初始权重
        novelty_decay: 新奇奖励衰减系数
        info_gain_threshold: 信息增益阈值
        progress_window: 学习进度计算窗口
        goal_threshold: 目标达成阈值
        exploration_bonus: 探索奖励系数
        adaptive_weights: 是否启用自适应权重
        state_hash_precision: 状态哈希精度
        max_state_memory: 最大状态记忆数
    """
    motivation_types: List[str] = field(
        default_factory=lambda: ['novelty', 'information_gain', 'learning_progress']
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            'novelty': 1.0,
            'information_gain': 0.5,
            'learning_progress': 0.8,
            'goal_achievement': 0.6,
            'exploration': 0.4
        }
    )
    novelty_decay: float = 0.99
    info_gain_threshold: float = 0.01
    progress_window: int = 100
    goal_threshold: float = 0.95
    exploration_bonus: float = 0.1
    adaptive_weights: bool = True
    state_hash_precision: int = 4
    max_state_memory: int = 10000


class StateHasher:
    """状态哈希器
    
    将连续状态空间离散化为可计数的哈希值,
    用于访问计数和状态识别。
    """
    
    def __init__(self, precision: int = 4):
        """
        Args:
            precision: 哈希精度(小数位数)
        """
        self.precision = precision
    
    def hash(self, state: Union[np.ndarray, torch.Tensor, Dict]) -> str:
        """
        计算状态的哈希值
        
        Args:
            state: 输入状态
            
        Returns:
            哈希字符串
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        
        if isinstance(state, np.ndarray):
            # 离散化并哈希
            state_rounded = np.round(state, self.precision)
            state_bytes = state_rounded.tobytes()
            return hashlib.md5(state_bytes).hexdigest()
        
        elif isinstance(state, dict):
            # 对字典状态进行排序后哈希
            sorted_items = sorted(state.items())
            state_str = json.dumps(sorted_items, sort_keys=True, default=str)
            return hashlib.md5(state_str.encode()).hexdigest()
        
        else:
            # 其他类型直接字符串哈希
            return hashlib.md5(str(state).encode()).hexdigest()
    
    def hash_batch(
        self,
        states: Union[List, torch.Tensor, np.ndarray]
    ) -> List[str]:
        """
        批量计算状态哈希
        
        Args:
            states: 状态批次
            
        Returns:
            哈希字符串列表
        """
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        
        if isinstance(states, np.ndarray) and len(states.shape) > 1:
            return [self.hash(s) for s in states]
        elif isinstance(states, list):
            return [self.hash(s) for s in states]
        else:
            return [self.hash(states)]


class BaseMotivation(ABC):
    """内在动机基类"""
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Args:
            name: 动机名称
            weight: 动机权重
        """
        self.name = name
        self.weight = weight
        self.stats = {
            'total_reward': 0.0,
            'count': 0,
            'mean_reward': 0.0
        }
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """计算内在奖励"""
        pass
    
    def update_stats(self, reward: float):
        """更新统计信息"""
        self.stats['total_reward'] += reward
        self.stats['count'] += 1
        self.stats['mean_reward'] = (
            self.stats['total_reward'] / self.stats['count']
        )
    
    def reset(self):
        """重置动机状态"""
        self.stats = {
            'total_reward': 0.0,
            'count': 0,
            'mean_reward': 0.0
        }


class NoveltyMotivation(BaseMotivation):
    """计数型新奇动机
    
    基于状态访问计数的新奇奖励,访问越少的状态奖励越高。
    使用指数衰减防止奖励消失。
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        decay: float = 0.99,
        max_memory: int = 10000,
        hash_precision: int = 4
    ):
        """
        Args:
            weight: 动机权重
            decay: 访问计数衰减系数
            max_memory: 最大状态记忆数
            hash_precision: 状态哈希精度
        """
        super().__init__('novelty', weight)
        self.decay = decay
        self.max_memory = max_memory
        self.state_hasher = StateHasher(hash_precision)
        self.state_counts: Dict[str, int] = defaultdict(int)
        self.state_visits: Dict[str, float] = defaultdict(float)
    
    def compute(self, state: Union[np.ndarray, torch.Tensor, Dict]) -> float:
        """
        计算新奇奖励
        
        Args:
            state: 当前状态
            
        Returns:
            新奇奖励值
        """
        state_hash = self.state_hasher.hash(state)
        
        # 计算基于访问次数的奖励
        visit_count = self.state_counts[state_hash]
        
        # 指数衰减的新奇奖励
        novelty_reward = np.exp(-self.decay * visit_count)
        
        # 更新访问计数
        self.state_counts[state_hash] += 1
        self.state_visits[state_hash] += 1.0
        
        # 内存管理:清理最少访问的状态
        if len(self.state_counts) > self.max_memory:
            self._cleanup_memory()
        
        self.update_stats(novelty_reward)
        return novelty_reward * self.weight
    
    def compute_batch(
        self,
        states: Union[List, torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        批量计算新奇奖励
        
        Args:
            states: 状态批次
            
        Returns:
            奖励数组
        """
        state_hashes = self.state_hasher.hash_batch(states)
        rewards = []
        
        for state_hash in state_hashes:
            visit_count = self.state_counts[state_hash]
            novelty_reward = np.exp(-self.decay * visit_count)
            self.state_counts[state_hash] += 1
            rewards.append(novelty_reward * self.weight)
        
        return np.array(rewards)
    
    def _cleanup_memory(self):
        """清理内存,保留最常访问的状态"""
        # 按访问次数排序,保留前80%
        sorted_states = sorted(
            self.state_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        keep_count = int(self.max_memory * 0.8)
        
        new_counts = defaultdict(int)
        for state_hash, count in sorted_states[:keep_count]:
            new_counts[state_hash] = count
        
        self.state_counts = new_counts
    
    def get_novelty_score(self, state: Union[np.ndarray, torch.Tensor, Dict]) -> float:
        """
        获取状态的新奇度分数(不更新计数)
        
        Args:
            state: 输入状态
            
        Returns:
            新奇度分数 [0, 1]
        """
        state_hash = self.state_hasher.hash(state)
        visit_count = self.state_counts[state_hash]
        return np.exp(-self.decay * visit_count)
    
    def reset(self):
        """重置动机状态"""
        super().reset()
        self.state_counts.clear()
        self.state_visits.clear()


class InformationGainMotivation(BaseMotivation):
    """信息增益动机
    
    基于预测不确定性和误差的信息增益奖励,
    鼓励探索信息丰富的状态。
    """
    
    def __init__(
        self,
        weight: float = 0.5,
        threshold: float = 0.01,
        window_size: int = 50
    ):
        """
        Args:
            weight: 动机权重
            threshold: 信息增益阈值
            window_size: 历史窗口大小
        """
        super().__init__('information_gain', weight)
        self.threshold = threshold
        self.window_size = window_size
        self.prediction_history: deque = deque(maxlen=window_size)
        self.error_history: deque = deque(maxlen=window_size)
        self.uncertainty_history: deque = deque(maxlen=window_size)
    
    def compute(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        uncertainty: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> float:
        """
        计算信息增益奖励
        
        Args:
            prediction: 模型预测
            target: 真实目标
            uncertainty: 预测不确定性,可选
            
        Returns:
            信息增益奖励
        """
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        if uncertainty is not None and isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.detach().cpu().numpy()
        
        # 计算预测误差
        error = np.mean((prediction - target) ** 2)
        
        # 基础信息增益:误差变化
        info_gain = 0.0
        if len(self.error_history) > 0:
            prev_error = np.mean(list(self.error_history)[-10:])
            error_change = prev_error - error
            info_gain += max(0, error_change)
        
        # 不确定性奖励
        if uncertainty is not None:
            uncertainty_value = np.mean(uncertainty)
            self.uncertainty_history.append(uncertainty_value)
            
            # 高不确定性但低误差 = 学到了新知识
            if len(self.uncertainty_history) > 10:
                prev_uncertainty = np.mean(list(self.uncertainty_history)[-10:])
                uncertainty_drop = prev_uncertainty - uncertainty_value
                if error < prev_error:  # 误差也降低了
                    info_gain += max(0, uncertainty_drop)
        
        # 存储历史
        self.prediction_history.append(prediction)
        self.error_history.append(error)
        
        # 阈值过滤
        if info_gain < self.threshold:
            info_gain = 0.0
        
        self.update_stats(info_gain)
        return info_gain * self.weight
    
    def compute_from_model(
        self,
        model: nn.Module,
        state: torch.Tensor,
        target: torch.Tensor,
        num_samples: int = 10
    ) -> float:
        """
        使用模型不确定性计算信息增益
        
        Args:
            model: 神经网络模型
            state: 输入状态
            target: 真实目标
            num_samples: MC Dropout采样次数
            
        Returns:
            信息增益奖励
        """
        model.train()  # 启用dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = model(state)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean().item()
        
        return self.compute(mean_pred, target, uncertainty)
    
    def reset(self):
        """重置动机状态"""
        super().reset()
        self.prediction_history.clear()
        self.error_history.clear()
        self.uncertainty_history.clear()


class LearningProgressMotivation(BaseMotivation):
    """能力进步动机
    
    基于学习曲线斜率的进步奖励,
    鼓励智能体学习能快速提升能力的技能。
    """
    
    def __init__(
        self,
        weight: float = 0.8,
        window_size: int = 100,
        min_improvement: float = 0.001
    ):
        """
        Args:
            weight: 动机权重
            window_size: 学习进度窗口大小
            min_improvement: 最小改善阈值
        """
        super().__init__('learning_progress', weight)
        self.window_size = window_size
        self.min_improvement = min_improvement
        self.performance_history: deque = deque(maxlen=window_size)
        self.skill_histories: Dict[str, deque] = {}
    
    def compute(self, performance: float, skill_id: Optional[str] = None) -> float:
        """
        计算学习进度奖励
        
        Args:
            performance: 当前性能指标
            skill_id: 技能标识符,可选
            
        Returns:
            学习进度奖励
        """
        # 存储性能历史
        self.performance_history.append(performance)
        
        if skill_id is not None:
            if skill_id not in self.skill_histories:
                self.skill_histories[skill_id] = deque(maxlen=self.window_size)
            self.skill_histories[skill_id].append(performance)
            history = self.skill_histories[skill_id]
        else:
            history = self.performance_history
        
        # 计算学习进度
        if len(history) < 10:
            return 0.0
        
        # 使用线性回归计算学习曲线斜率
        y = np.array(list(history))
        x = np.arange(len(y))
        
        # 分段计算斜率
        mid = len(y) // 2
        recent_mean = y[mid:].mean()
        past_mean = y[:mid].mean()
        
        # 学习进度 = 近期表现 - 过去表现
        progress = recent_mean - past_mean
        
        # 归一化到合理范围
        progress_reward = np.tanh(progress * 10)
        
        # 最小改善阈值
        if abs(progress) < self.min_improvement:
            progress_reward = 0.0
        
        self.update_stats(max(0, progress_reward))
        return max(0, progress_reward) * self.weight
    
    def compute_from_history(self, history: List[float]) -> float:
        """
        从历史数据计算学习进度
        
        Args:
            history: 性能历史列表
            
        Returns:
            学习进度奖励
        """
        if len(history) < 10:
            return 0.0
        
        y = np.array(history[-self.window_size:])
        mid = len(y) // 2
        
        recent_mean = y[mid:].mean()
        past_mean = y[:mid].mean()
        
        progress = recent_mean - past_mean
        progress_reward = np.tanh(progress * 10)
        
        return max(0, progress_reward) * self.weight
    
    def get_learning_curve(self, skill_id: Optional[str] = None) -> List[float]:
        """
        获取学习曲线
        
        Args:
            skill_id: 技能标识符
            
        Returns:
            性能历史列表
        """
        if skill_id is not None and skill_id in self.skill_histories:
            return list(self.skill_histories[skill_id])
        return list(self.performance_history)
    
    def reset(self):
        """重置动机状态"""
        super().reset()
        self.performance_history.clear()
        self.skill_histories.clear()


class GoalAchievementMotivation(BaseMotivation):
    """目标达成动机
    
    基于目标完成度的奖励,
    支持多目标、子目标和层次目标结构。
    """
    
    def __init__(
        self,
        weight: float = 0.6,
        threshold: float = 0.95,
        subgoal_bonus: float = 0.3
    ):
        """
        Args:
            weight: 动机权重
            threshold: 目标达成阈值
            subgoal_bonus: 子目标奖励系数
        """
        super().__init__('goal_achievement', weight)
        self.threshold = threshold
        self.subgoal_bonus = subgoal_bonus
        self.active_goals: Dict[str, Dict] = {}
        self.completed_goals: set = set()
        self.goal_hierarchy: Dict[str, List[str]] = {}
    
    def add_goal(
        self,
        goal_id: str,
        target_state: Union[np.ndarray, Dict],
        parent_goal: Optional[str] = None,
        priority: float = 1.0
    ):
        """
        添加目标
        
        Args:
            goal_id: 目标标识符
            target_state: 目标状态
            parent_goal: 父目标ID
            priority: 目标优先级
        """
        self.active_goals[goal_id] = {
            'target': target_state,
            'priority': priority,
            'progress': 0.0,
            'parent': parent_goal
        }
        
        if parent_goal is not None:
            if parent_goal not in self.goal_hierarchy:
                self.goal_hierarchy[parent_goal] = []
            self.goal_hierarchy[parent_goal].append(goal_id)
    
    def compute(
        self,
        current_state: Union[np.ndarray, torch.Tensor, Dict],
        goal_id: Optional[str] = None
    ) -> float:
        """
        计算目标达成奖励
        
        Args:
            current_state: 当前状态
            goal_id: 目标标识符,None则评估所有目标
            
        Returns:
            目标达成奖励
        """
        total_reward = 0.0
        
        if goal_id is not None:
            # 评估特定目标
            if goal_id in self.active_goals:
                reward = self._evaluate_goal(current_state, goal_id)
                total_reward += reward
        else:
            # 评估所有活跃目标
            for gid in list(self.active_goals.keys()):
                reward = self._evaluate_goal(current_state, gid)
                total_reward += reward
        
        self.update_stats(total_reward)
        return total_reward * self.weight
    
    def _evaluate_goal(
        self,
        current_state: Union[np.ndarray, torch.Tensor, Dict],
        goal_id: str
    ) -> float:
        """
        评估单个目标
        
        Args:
            current_state: 当前状态
            goal_id: 目标标识符
            
        Returns:
            目标奖励
        """
        goal = self.active_goals[goal_id]
        target = goal['target']
        
        # 计算进度
        if isinstance(current_state, torch.Tensor):
            current_state = current_state.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        if isinstance(current_state, np.ndarray) and isinstance(target, np.ndarray):
            distance = np.linalg.norm(current_state - target)
            progress = np.exp(-distance)
        elif isinstance(current_state, dict) and isinstance(target, dict):
            # 字典状态比较
            matches = sum(
                1 for k in target if k in current_state and current_state[k] == target[k]
            )
            progress = matches / len(target) if len(target) > 0 else 0.0
        else:
            progress = 1.0 if current_state == target else 0.0
        
        # 更新进度
        prev_progress = goal['progress']
        goal['progress'] = progress
        
        reward = 0.0
        
        # 进度改善奖励
        if progress > prev_progress:
            reward += (progress - prev_progress) * goal['priority']
        
        # 目标达成奖励
        if progress >= self.threshold and goal_id not in self.completed_goals:
            reward += goal['priority']  # 基础达成奖励
            
            # 子目标奖励
            if goal_id in self.goal_hierarchy:
                completed_subgoals = sum(
                    1 for sub in self.goal_hierarchy[goal_id]
                    if sub in self.completed_goals
                )
                reward += completed_subgoals * self.subgoal_bonus
            
            self.completed_goals.add(goal_id)
            
            # 移除已完成的目标
            del self.active_goals[goal_id]
        
        return reward
    
    def get_goal_progress(self, goal_id: str) -> float:
        """
        获取目标进度
        
        Args:
            goal_id: 目标标识符
            
        Returns:
            目标进度 [0, 1]
        """
        if goal_id in self.active_goals:
            return self.active_goals[goal_id]['progress']
        elif goal_id in self.completed_goals:
            return 1.0
        return 0.0
    
    def reset(self):
        """重置动机状态"""
        super().reset()
        self.active_goals.clear()
        self.completed_goals.clear()
        self.goal_hierarchy.clear()


class ExplorationMotivation(BaseMotivation):
    """探索动机
    
    鼓励智能体探索未访问区域,
    基于状态空间覆盖率和动作多样性。
    """
    
    def __init__(
        self,
        weight: float = 0.4,
        state_dim: Optional[int] = None,
        n_bins: int = 10
    ):
        """
        Args:
            weight: 动机权重
            state_dim: 状态维度
            n_bins: 离散化分箱数
        """
        super().__init__('exploration', weight)
        self.state_dim = state_dim
        self.n_bins = n_bins
        self.visited_bins: set = set()
        self.action_history: deque = deque(maxlen=1000)
        self.state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def compute(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Optional[Union[int, np.ndarray]] = None
    ) -> float:
        """
        计算探索奖励
        
        Args:
            state: 当前状态
            action: 执行的动作
            
        Returns:
            探索奖励
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        
        reward = 0.0
        
        # 状态空间覆盖奖励
        state_bin = self._discretize_state(state)
        if state_bin not in self.visited_bins:
            reward += 1.0  # 新区域奖励
            self.visited_bins.add(state_bin)
        else:
            # 重访惩罚(轻微)
            reward -= 0.1
        
        # 动作多样性奖励
        if action is not None:
            self.action_history.append(action)
            if len(self.action_history) >= 10:
                recent_actions = list(self.action_history)[-10:]
                diversity = self._compute_action_diversity(recent_actions)
                reward += diversity * 0.5
        
        self.update_stats(reward)
        return reward * self.weight
    
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """离散化状态"""
        if self.state_bounds is None:
            # 初始化边界
            self.state_bounds = (
                np.zeros_like(state),
                np.ones_like(state)
            )
        
        # 归一化并分箱
        state_norm = (state - self.state_bounds[0]) / (
            self.state_bounds[1] - self.state_bounds[0] + 1e-8
        )
        bins = np.digitize(state_norm, np.linspace(0, 1, self.n_bins))
        return tuple(bins)
    
    def _compute_action_diversity(self, actions: List) -> float:
        """计算动作多样性"""
        if len(actions) == 0:
            return 0.0
        
        unique_actions = len(set(map(str, actions)))
        total_actions = len(actions)
        
        return unique_actions / total_actions if total_actions > 0 else 0.0
    
    def get_coverage(self) -> float:
        """
        获取状态空间覆盖率
        
        Returns:
            覆盖率估计 [0, 1]
        """
        # 简化的覆盖率估计
        total_possible_bins = self.n_bins ** (self.state_dim or 1)
        return len(self.visited_bins) / total_possible_bins if total_possible_bins > 0 else 0.0
    
    def update_state_bounds(self, min_bounds: np.ndarray, max_bounds: np.ndarray):
        """
        更新状态边界
        
        Args:
            min_bounds: 最小边界
            max_bounds: 最大边界
        """
        self.state_bounds = (min_bounds, max_bounds)
    
    def reset(self):
        """重置动机状态"""
        super().reset()
        self.visited_bins.clear()
        self.action_history.clear()


class IntrinsicMotivationEngine:
    """内在动机引擎主类
    
    整合多种内在动机,支持自适应权重调整和动机融合。
    """
    
    def __init__(
        self,
        motivation_types: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[IntrinsicMotivationConfig] = None
    ):
        """
        Args:
            motivation_types: 动机类型列表
            weights: 动机权重字典
            config: 动机配置
        """
        self.config = config or IntrinsicMotivationConfig()
        self.motivation_types = motivation_types or self.config.motivation_types
        self.weights = weights or self.config.weights.copy()
        
        # 初始化动机模块
        self.motivations: Dict[str, BaseMotivation] = {}
        self._init_motivations()
        
        # 自适应权重
        self.adaptive_weights = self.config.adaptive_weights
        self.weight_history: deque = deque(maxlen=100)
        self.reward_history: Dict[str, deque] = {
            name: deque(maxlen=50) for name in self.motivation_types
        }
    
    def _init_motivations(self):
        """初始化动机模块"""
        for mot_type in self.motivation_types:
            weight = self.weights.get(mot_type, 1.0)
            
            if mot_type == 'novelty':
                self.motivations[mot_type] = NoveltyMotivation(
                    weight=weight,
                    decay=self.config.novelty_decay,
                    max_memory=self.config.max_state_memory,
                    hash_precision=self.config.state_hash_precision
                )
            elif mot_type == 'information_gain':
                self.motivations[mot_type] = InformationGainMotivation(
                    weight=weight,
                    threshold=self.config.info_gain_threshold
                )
            elif mot_type == 'learning_progress':
                self.motivations[mot_type] = LearningProgressMotivation(
                    weight=weight,
                    window_size=self.config.progress_window
                )
            elif mot_type == 'goal_achievement':
                self.motivations[mot_type] = GoalAchievementMotivation(
                    weight=weight,
                    threshold=self.config.goal_threshold
                )
            elif mot_type == 'exploration':
                self.motivations[mot_type] = ExplorationMotivation(
                    weight=weight
                )
    
    def compute_novelty_bonus(
        self,
        state: Union[np.ndarray, torch.Tensor, Dict]
    ) -> float:
        """
        计算计数型新奇奖励
        
        Args:
            state: 当前状态
            
        Returns:
            新奇奖励值
        """
        if 'novelty' in self.motivations:
            return self.motivations['novelty'].compute(state)
        return 0.0
    
    def compute_information_gain(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        uncertainty: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> float:
        """
        计算信息增益奖励
        
        Args:
            pred: 预测值
            target: 目标值
            uncertainty: 不确定性
            
        Returns:
            信息增益奖励
        """
        if 'information_gain' in self.motivations:
            return self.motivations['information_gain'].compute(pred, target, uncertainty)
        return 0.0
    
    def compute_learning_progress(self, history: List[float]) -> float:
        """
        计算能力进步奖励
        
        Args:
            history: 性能历史
            
        Returns:
            学习进度奖励
        """
        if 'learning_progress' in self.motivations:
            return self.motivations['learning_progress'].compute_from_history(history)
        return 0.0
    
    def compute_total_intrinsic_reward(
        self,
        state: Union[np.ndarray, torch.Tensor, Dict],
        action: Optional[Any] = None,
        info: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        计算总内在奖励
        
        Args:
            state: 当前状态
            action: 执行的动作
            info: 额外信息字典,包含预测、目标等
            
        Returns:
            包含各动机奖励和总奖励的字典
        """
        info = info or {}
        rewards = {}
        
        # 计算各动机奖励
        for name, motivation in self.motivations.items():
            if name == 'novelty':
                reward = motivation.compute(state)
            elif name == 'information_gain':
                pred = info.get('prediction')
                target = info.get('target')
                uncertainty = info.get('uncertainty')
                if pred is not None and target is not None:
                    reward = motivation.compute(pred, target, uncertainty)
                else:
                    reward = 0.0
            elif name == 'learning_progress':
                performance = info.get('performance', 0.0)
                skill_id = info.get('skill_id')
                reward = motivation.compute(performance, skill_id)
            elif name == 'goal_achievement':
                goal_id = info.get('goal_id')
                reward = motivation.compute(state, goal_id)
            elif name == 'exploration':
                reward = motivation.compute(state, action)
            else:
                reward = 0.0
            
            rewards[name] = reward
            self.reward_history[name].append(reward)
        
        # 计算总奖励
        total_reward = sum(rewards.values())
        rewards['total'] = total_reward
        
        # 自适应权重更新
        if self.adaptive_weights:
            self._update_weights()
        
        return rewards
    
    def _update_weights(self):
        """自适应更新权重"""
        # 基于奖励方差调整权重
        # 方差小的动机可能需要更高权重(稳定但不足)
        # 方差大的动机可能需要更低权重(不稳定)
        
        for name, history in self.reward_history.items():
            if len(history) < 10:
                continue
            
            hist_array = np.array(list(history))
            variance = np.var(hist_array)
            mean_reward = np.mean(hist_array)
            
            # 简单自适应:均值低但方差适中的动机增加权重
            if mean_reward < 0.1 and 0.01 < variance < 1.0:
                self.weights[name] = min(2.0, self.weights[name] * 1.05)
            # 方差过高的动机降低权重
            elif variance > 2.0:
                self.weights[name] = max(0.1, self.weights[name] * 0.95)
            
            # 更新动机权重
            if name in self.motivations:
                self.motivations[name].weight = self.weights[name]
    
    def get_motivation_stats(self) -> Dict[str, Dict]:
        """
        获取动机统计信息
        
        Returns:
            各动机的统计信息
        """
        stats = {}
        for name, motivation in self.motivations.items():
            stats[name] = {
                'weight': motivation.weight,
                'mean_reward': motivation.stats['mean_reward'],
                'total_reward': motivation.stats['total_reward'],
                'count': motivation.stats['count']
            }
        return stats
    
    def add_goal(
        self,
        goal_id: str,
        target_state: Union[np.ndarray, Dict],
        parent_goal: Optional[str] = None,
        priority: float = 1.0
    ):
        """
        添加目标(用于目标达成动机)
        
        Args:
            goal_id: 目标标识符
            target_state: 目标状态
            parent_goal: 父目标ID
            priority: 目标优先级
        """
        if 'goal_achievement' in self.motivations:
            self.motivations['goal_achievement'].add_goal(
                goal_id, target_state, parent_goal, priority
            )
    
    def reset(self):
        """重置引擎状态"""
        for motivation in self.motivations.values():
            motivation.reset()
        self.weight_history.clear()
        for history in self.reward_history.values():
            history.clear()


# 辅助函数
def create_intrinsic_motivation_engine(
    config_dict: Optional[Dict] = None
) -> IntrinsicMotivationEngine:
    """
    从配置字典创建内在动机引擎
    
    Args:
        config_dict: 配置字典
        
    Returns:
        IntrinsicMotivationEngine实例
    """
    if config_dict:
        config = IntrinsicMotivationConfig(**config_dict)
        return IntrinsicMotivationEngine(config=config)
    return IntrinsicMotivationEngine()
