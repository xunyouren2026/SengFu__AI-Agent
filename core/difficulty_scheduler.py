"""
课程学习难度调度器 (Curriculum Learning Difficulty Scheduler)

该模块实现了自适应课程学习框架,根据智能体的学习进度动态调整任务难度,
    实现从简单到复杂任务的平滑过渡,优化学习效率。

主要特性:
- 动态难度评估
- 自适应课程生成
- 难度阈值调整
- 学习进度跟踪
- 难度-性能平衡

作者: AGI Unified Framework Team
版本: 1.0.0
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
import random


class DifficultyLevel(Enum):
    """难度等级枚举"""
    VERY_EASY = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4
    EXPERT = 5


@dataclass
class Task:
    """任务数据类
    
    Attributes:
        task_id: 任务唯一标识
        difficulty: 任务难度值 [0, 1]
        data: 任务数据
        metadata: 任务元信息
        success_rate: 历史成功率
        attempt_count: 尝试次数
    """
    task_id: str
    difficulty: float
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    attempt_count: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'difficulty': self.difficulty,
            'metadata': self.metadata,
            'success_rate': self.success_rate,
            'attempt_count': self.attempt_count
        }


@dataclass
class DifficultySchedulerConfig:
    """难度调度器配置类
    
    Attributes:
        initial_difficulty: 初始难度
        min_difficulty: 最小难度
        max_difficulty: 最大难度
        performance_window: 性能评估窗口大小
        success_threshold: 成功率阈值(用于升级)
        failure_threshold: 失败率阈值(用于降级)
        difficulty_step: 难度调整步长
        smoothing_factor: 平滑因子
        adaptive_step: 是否使用自适应步长
        curriculum_strategy: 课程策略 ('linear', 'exponential', 'adaptive')
        min_tasks_per_level: 每级最小任务数
        max_tasks_per_level: 每级最大任务数
    """
    initial_difficulty: float = 0.1
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    performance_window: int = 50
    success_threshold: float = 0.8
    failure_threshold: float = 0.3
    difficulty_step: float = 0.1
    smoothing_factor: float = 0.9
    adaptive_step: bool = True
    curriculum_strategy: str = 'adaptive'
    min_tasks_per_level: int = 10
    max_tasks_per_level: int = 100


class PerformanceTracker:
    """性能跟踪器
    
    跟踪和评估智能体在任务上的表现,
    提供平滑的性能指标。
    """
    
    def __init__(
        self,
        window_size: int = 50,
        smoothing_factor: float = 0.9
    ):
        """
        Args:
            window_size: 滑动窗口大小
            smoothing_factor: 指数平滑因子
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.rewards: deque = deque(maxlen=window_size)
        self.successes: deque = deque(maxlen=window_size)
        self.smoothed_performance: float = 0.0
        self.difficulty_history: deque = deque(maxlen=window_size)
    
    def update(
        self,
        reward: float,
        success: bool,
        difficulty: float
    ):
        """
        更新性能记录
        
        Args:
            reward: 获得的奖励
            success: 是否成功
            difficulty: 任务难度
        """
        self.rewards.append(reward)
        self.successes.append(1.0 if success else 0.0)
        self.difficulty_history.append(difficulty)
        
        # 指数平滑
        current_performance = reward if success else reward * 0.5
        self.smoothed_performance = (
            self.smoothing_factor * self.smoothed_performance +
            (1 - self.smoothing_factor) * current_performance
        )
    
    def get_success_rate(self) -> float:
        """
        获取成功率
        
        Returns:
            最近窗口内的成功率
        """
        if len(self.successes) == 0:
            return 0.0
        return np.mean(self.successes)
    
    def get_average_reward(self) -> float:
        """
        获取平均奖励
        
        Returns:
            最近窗口内的平均奖励
        """
        if len(self.rewards) == 0:
            return 0.0
        return np.mean(self.rewards)
    
    def get_performance_trend(self) -> float:
        """
        获取性能趋势
        
        Returns:
            性能变化趋势(正数表示提升)
        """
        if len(self.rewards) < 10:
            return 0.0
        
        recent = np.mean(list(self.rewards)[-10:])
        past = np.mean(list(self.rewards)[:-10]) if len(self.rewards) > 10 else recent
        
        return recent - past
    
    def get_difficulty_correlation(self) -> float:
        """
        获取难度-性能相关性
        
        Returns:
            难度与性能的相关性 [-1, 1]
        """
        if len(self.rewards) < 10 or len(self.difficulty_history) < 10:
            return 0.0
        
        rewards_array = np.array(self.rewards)
        difficulty_array = np.array(self.difficulty_history)
        
        # 计算Pearson相关系数
        if np.std(rewards_array) == 0 or np.std(difficulty_array) == 0:
            return 0.0
        
        correlation = np.corrcoef(rewards_array, difficulty_array)[0, 1]
        return float(correlation)
    
    def reset(self):
        """重置跟踪器"""
        self.rewards.clear()
        self.successes.clear()
        self.difficulty_history.clear()
        self.smoothed_performance = 0.0


class DifficultyAssessor:
    """难度评估器
    
    评估任务的难度级别,支持多种难度指标。
    """
    
    def __init__(self):
        """初始化难度评估器"""
        self.difficulty_metrics: Dict[str, Callable] = {}
        self.register_default_metrics()
    
    def register_default_metrics(self):
        """注册默认难度指标"""
        self.difficulty_metrics['complexity'] = self._complexity_metric
        self.difficulty_metrics['uncertainty'] = self._uncertainty_metric
        self.difficulty_metrics['steps'] = self._steps_metric
    
    def _complexity_metric(self, task: Task) -> float:
        """基于任务复杂度的难度"""
        return task.metadata.get('complexity', task.difficulty)
    
    def _uncertainty_metric(self, task: Task) -> float:
        """基于不确定性的难度"""
        return 1.0 - task.success_rate if task.attempt_count > 0 else 0.5
    
    def _steps_metric(self, task: Task) -> float:
        """基于所需步骤数的难度"""
        optimal_steps = task.metadata.get('optimal_steps', 10)
        max_steps = task.metadata.get('max_steps', 100)
        return min(1.0, optimal_steps / max_steps)
    
    def assess_difficulty(
        self,
        task: Task,
        metric_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        综合评估任务难度
        
        Args:
            task: 任务对象
            metric_weights: 指标权重
            
        Returns:
            综合难度值 [0, 1]
        """
        if metric_weights is None:
            metric_weights = {name: 1.0 for name in self.difficulty_metrics}
        
        total_weight = sum(metric_weights.values())
        weighted_difficulty = 0.0
        
        for metric_name, metric_fn in self.difficulty_metrics.items():
            weight = metric_weights.get(metric_name, 1.0)
            difficulty = metric_fn(task)
            weighted_difficulty += weight * difficulty
        
        return weighted_difficulty / total_weight if total_weight > 0 else 0.5
    
    def register_metric(self, name: str, metric_fn: Callable[[Task], float]):
        """
        注册自定义难度指标
        
        Args:
            name: 指标名称
            metric_fn: 指标计算函数
        """
        self.difficulty_metrics[name] = metric_fn


class CurriculumGenerator:
    """课程生成器
    
    根据当前难度级别生成合适的课程任务序列。
    """
    
    def __init__(
        self,
        strategy: str = 'adaptive',
        min_tasks: int = 10,
        max_tasks: int = 100
    ):
        """
        Args:
            strategy: 生成策略
            min_tasks: 每级最小任务数
            max_tasks: 每级最大任务数
        """
        self.strategy = strategy
        self.min_tasks = min_tasks
        self.max_tasks = max_tasks
        self.task_pool: List[Task] = []
        self.generated_curricula: Dict[float, List[Task]] = {}
    
    def set_task_pool(self, tasks: List[Task]):
        """
        设置任务池
        
        Args:
            tasks: 可用任务列表
        """
        self.task_pool = tasks
        self.generated_curricula.clear()
    
    def generate_curriculum(
        self,
        current_difficulty: float,
        performance: float,
        num_tasks: Optional[int] = None
    ) -> List[Task]:
        """
        生成课程
        
        Args:
            current_difficulty: 当前难度
            performance: 当前性能
            num_tasks: 任务数量
            
        Returns:
            课程任务列表
        """
        if num_tasks is None:
            num_tasks = self._compute_num_tasks(performance)
        
        # 检查缓存
        cache_key = round(current_difficulty, 2)
        if cache_key in self.generated_curricula:
            cached = self.generated_curricula[cache_key]
            if len(cached) >= num_tasks:
                return cached[:num_tasks]
        
        # 根据策略选择任务
        if self.strategy == 'linear':
            tasks = self._linear_selection(current_difficulty, num_tasks)
        elif self.strategy == 'exponential':
            tasks = self._exponential_selection(current_difficulty, num_tasks)
        elif self.strategy == 'adaptive':
            tasks = self._adaptive_selection(current_difficulty, performance, num_tasks)
        elif self.strategy == 'mastery':
            tasks = self._mastery_selection(current_difficulty, performance, num_tasks)
        else:
            tasks = self._linear_selection(current_difficulty, num_tasks)
        
        # 缓存结果
        self.generated_curricula[cache_key] = tasks
        
        return tasks
    
    def _compute_num_tasks(self, performance: float) -> int:
        """根据性能计算任务数量"""
        # 性能越好,任务越多(挑战更多)
        num = int(self.min_tasks + (self.max_tasks - self.min_tasks) * performance)
        return max(self.min_tasks, min(self.max_tasks, num))
    
    def _linear_selection(
        self,
        difficulty: float,
        num_tasks: int
    ) -> List[Task]:
        """线性难度选择"""
        # 选择难度接近current_difficulty的任务
        sorted_tasks = sorted(
            self.task_pool,
            key=lambda t: abs(t.difficulty - difficulty)
        )
        return sorted_tasks[:num_tasks]
    
    def _exponential_selection(
        self,
        difficulty: float,
        num_tasks: int
    ) -> List[Task]:
        """指数难度选择"""
        # 大部分任务在当前难度,少量更难
        tasks = []
        easy_count = int(num_tasks * 0.3)
        current_count = int(num_tasks * 0.5)
        hard_count = num_tasks - easy_count - current_count
        
        # 简单任务
        easy_tasks = [t for t in self.task_pool if t.difficulty < difficulty * 0.8]
        tasks.extend(random.sample(easy_tasks, min(easy_count, len(easy_tasks))))
        
        # 当前难度任务
        current_tasks = [
            t for t in self.task_pool
            if difficulty * 0.8 <= t.difficulty <= difficulty * 1.2
        ]
        tasks.extend(random.sample(current_tasks, min(current_count, len(current_tasks))))
        
        # 困难任务
        hard_tasks = [t for t in self.task_pool if t.difficulty > difficulty * 1.2]
        tasks.extend(random.sample(hard_tasks, min(hard_count, len(hard_tasks))))
        
        return tasks
    
    def _adaptive_selection(
        self,
        difficulty: float,
        performance: float,
        num_tasks: int
    ) -> List[Task]:
        """自适应选择"""
        # 根据性能调整难度分布
        if performance > 0.8:
            # 表现好,增加难度
            target_difficulty = min(1.0, difficulty * 1.1)
        elif performance < 0.4:
            # 表现差,降低难度
            target_difficulty = max(0.0, difficulty * 0.9)
        else:
            target_difficulty = difficulty
        
        # 在目标难度周围选择任务
        difficulty_range = 0.2 * (1 - performance)  # 性能越好,范围越窄
        
        candidates = [
            t for t in self.task_pool
            if abs(t.difficulty - target_difficulty) <= difficulty_range
        ]
        
        if len(candidates) < num_tasks:
            # 扩展搜索范围
            candidates = self._linear_selection(target_difficulty, num_tasks * 2)
        
        return random.sample(candidates, min(num_tasks, len(candidates)))
    
    def _mastery_selection(
        self,
        difficulty: float,
        performance: float,
        num_tasks: int
    ) -> List[Task]:
        """掌握度-based选择"""
        # 优先选择成功率适中的任务(0.3-0.7)
        candidates = [
            t for t in self.task_pool
            if 0.3 <= t.success_rate <= 0.7 or t.attempt_count < 5
        ]
        
        if len(candidates) < num_tasks:
            candidates = self.task_pool
        
        # 按掌握度排序(掌握度低的优先)
        candidates.sort(key=lambda t: (t.success_rate, -t.attempt_count))
        
        return candidates[:num_tasks]


class DifficultyScheduler:
    """课程学习难度调度器主类
    
    整合性能跟踪、难度评估和课程生成,
    实现完整的自适应课程学习流程。
    """
    
    def __init__(self, config: Optional[DifficultySchedulerConfig] = None):
        """
        Args:
            config: 调度器配置
        """
        self.config = config or DifficultySchedulerConfig()
        
        # 初始化组件
        self.performance_tracker = PerformanceTracker(
            window_size=self.config.performance_window,
            smoothing_factor=self.config.smoothing_factor
        )
        self.difficulty_assessor = DifficultyAssessor()
        self.curriculum_generator = CurriculumGenerator(
            strategy=self.config.curriculum_strategy,
            min_tasks=self.config.min_tasks_per_level,
            max_tasks=self.config.max_tasks_per_level
        )
        
        # 状态变量
        self.current_difficulty: float = self.config.initial_difficulty
        self.current_level: int = 0
        self.task_history: deque = deque(maxlen=1000)
        self.difficulty_history: List[float] = []
        self.performance_history: List[float] = []
        
        # 当前课程
        self.current_curriculum: List[Task] = []
        self.curriculum_index: int = 0
        
        # 统计信息
        self.total_tasks_completed: int = 0
        self.total_successes: int = 0
    
    def assess_performance(self, recent_rewards: Optional[List[float]] = None) -> float:
        """
        评估当前性能
        
        Args:
            recent_rewards: 最近奖励列表,None则使用跟踪器数据
            
        Returns:
            性能评分 [0, 1]
        """
        if recent_rewards is not None:
            # 使用提供的奖励计算
            success_rate = np.mean([r > 0 for r in recent_rewards])
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        else:
            # 使用跟踪器数据
            success_rate = self.performance_tracker.get_success_rate()
            avg_reward = self.performance_tracker.get_average_reward()
        
        # 综合性能分数
        performance = 0.6 * success_rate + 0.4 * np.tanh(avg_reward)
        
        # 考虑趋势
        trend = self.performance_tracker.get_performance_trend()
        performance += 0.1 * np.tanh(trend)
        
        return max(0.0, min(1.0, performance))
    
    def adjust_difficulty(self, performance: Optional[float] = None) -> float:
        """
        根据性能调整难度
        
        Args:
            performance: 性能评分,None则自动评估
            
        Returns:
            调整后的难度值
        """
        if performance is None:
            performance = self.assess_performance()
        
        # 计算难度调整步长
        if self.config.adaptive_step:
            # 自适应步长:性能越好步长越大
            step = self.config.difficulty_step * (0.5 + performance)
        else:
            step = self.config.difficulty_step
        
        # 根据性能调整难度
        if performance >= self.config.success_threshold:
            # 表现好,增加难度
            self.current_difficulty = min(
                self.config.max_difficulty,
                self.current_difficulty + step
            )
            self.current_level += 1
        elif performance <= self.config.failure_threshold:
            # 表现差,降低难度
            self.current_difficulty = max(
                self.config.min_difficulty,
                self.current_difficulty - step * 1.5  # 降级更快
            )
            self.current_level = max(0, self.current_level - 1)
        
        # 记录历史
        self.difficulty_history.append(self.current_difficulty)
        self.performance_history.append(performance)
        
        return self.current_difficulty
    
    def generate_curriculum(
        self,
        current_level: Optional[int] = None,
        num_tasks: Optional[int] = None
    ) -> List[Task]:
        """
        生成课程
        
        Args:
            current_level: 当前级别,None则使用内部状态
            num_tasks: 任务数量
            
        Returns:
            课程任务列表
        """
        if current_level is not None:
            self.current_level = current_level
            # 根据级别计算难度
            self.current_difficulty = min(
                1.0,
                self.config.initial_difficulty + current_level * self.config.difficulty_step
            )
        
        performance = self.assess_performance()
        
        self.current_curriculum = self.curriculum_generator.generate_curriculum(
            self.current_difficulty,
            performance,
            num_tasks
        )
        
        self.curriculum_index = 0
        return self.current_curriculum
    
    def get_next_task(self) -> Optional[Task]:
        """
        获取下一个任务
        
        Returns:
            下一个任务,课程结束则返回None
        """
        if self.curriculum_index >= len(self.current_curriculum):
            # 课程结束,生成新课程
            self.generate_curriculum()
        
        if self.curriculum_index < len(self.current_curriculum):
            task = self.current_curriculum[self.curriculum_index]
            self.curriculum_index += 1
            return task
        
        return None
    
    def report_task_result(
        self,
        task: Task,
        reward: float,
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """
        报告任务结果
        
        Args:
            task: 完成的任务
            reward: 获得的奖励
            success: 是否成功
            metadata: 额外元信息
        """
        # 更新任务统计
        task.attempt_count += 1
        if success:
            task.success_rate = (
                (task.success_rate * (task.attempt_count - 1) + 1) / task.attempt_count
            )
            self.total_successes += 1
        else:
            task.success_rate = (
                task.success_rate * (task.attempt_count - 1) / task.attempt_count
            )
        
        # 更新性能跟踪
        self.performance_tracker.update(reward, success, task.difficulty)
        
        # 记录历史
        self.task_history.append({
            'task_id': task.task_id,
            'difficulty': task.difficulty,
            'reward': reward,
            'success': success,
            'metadata': metadata or {}
        })
        
        self.total_tasks_completed += 1
        
        # 检查是否需要调整难度
        if self.total_tasks_completed % self.config.min_tasks_per_level == 0:
            self.adjust_difficulty()
    
    def get_current_difficulty(self) -> float:
        """获取当前难度"""
        return self.current_difficulty
    
    def get_current_level(self) -> int:
        """获取当前级别"""
        return self.current_level
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """
        获取学习进度统计
        
        Returns:
            统计信息字典
        """
        return {
            'current_difficulty': self.current_difficulty,
            'current_level': self.current_level,
            'total_tasks_completed': self.total_tasks_completed,
            'total_successes': self.total_successes,
            'success_rate': self.total_successes / max(1, self.total_tasks_completed),
            'current_performance': self.assess_performance(),
            'success_rate_recent': self.performance_tracker.get_success_rate(),
            'avg_reward_recent': self.performance_tracker.get_average_reward(),
            'performance_trend': self.performance_tracker.get_performance_trend(),
            'difficulty_correlation': self.performance_tracker.get_difficulty_correlation(),
            'curriculum_progress': self.curriculum_index / max(1, len(self.current_curriculum)),
            'difficulty_history': list(self.difficulty_history),
            'performance_history': list(self.performance_history)
        }
    
    def set_task_pool(self, tasks: List[Task]):
        """
        设置任务池
        
        Args:
            tasks: 可用任务列表
        """
        self.curriculum_generator.set_task_pool(tasks)
    
    def reset(self):
        """重置调度器状态"""
        self.current_difficulty = self.config.initial_difficulty
        self.current_level = 0
        self.curriculum_index = 0
        self.current_curriculum = []
        self.total_tasks_completed = 0
        self.total_successes = 0
        
        self.performance_tracker.reset()
        self.task_history.clear()
        self.difficulty_history.clear()
        self.performance_history.clear()
    
    def should_advance_level(self) -> bool:
        """
        判断是否应进入下一级别
        
        Returns:
            是否应升级
        """
        success_rate = self.performance_tracker.get_success_rate()
        return (
            success_rate >= self.config.success_threshold and
            len(self.performance_tracker.successes) >= self.config.min_tasks_per_level
        )
    
    def should_regress_level(self) -> bool:
        """
        判断是否应回退级别
        
        Returns:
            是否应降级
        """
        success_rate = self.performance_tracker.get_success_rate()
        return (
            success_rate <= self.config.failure_threshold and
            len(self.performance_tracker.successes) >= self.config.min_tasks_per_level and
            self.current_level > 0
        )


# 辅助函数
def create_difficulty_scheduler(
    config_dict: Optional[Dict] = None
) -> DifficultyScheduler:
    """
    从配置字典创建难度调度器
    
    Args:
        config_dict: 配置字典
        
    Returns:
        DifficultyScheduler实例
    """
    if config_dict:
        config = DifficultySchedulerConfig(**config_dict)
        return DifficultyScheduler(config)
    return DifficultyScheduler()


def create_tasks_from_data(
    data_list: List[Dict],
    difficulty_key: str = 'difficulty',
    id_key: str = 'id'
) -> List[Task]:
    """
    从数据列表创建任务对象
    
    Args:
        data_list: 数据字典列表
        difficulty_key: 难度字段名
        id_key: ID字段名
        
    Returns:
        任务对象列表
    """
    tasks = []
    for data in data_list:
        task = Task(
            task_id=str(data.get(id_key, len(tasks))),
            difficulty=float(data.get(difficulty_key, 0.5)),
            data=data,
            metadata={k: v for k, v in data.items() if k not in [difficulty_key, id_key]}
        )
        tasks.append(task)
    return tasks


def linear_curriculum_schedule(
    epoch: int,
    total_epochs: int,
    min_difficulty: float = 0.0,
    max_difficulty: float = 1.0
) -> float:
    """
    线性课程调度
    
    Args:
        epoch: 当前轮数
        total_epochs: 总轮数
        min_difficulty: 最小难度
        max_difficulty: 最大难度
        
    Returns:
        当前难度
    """
    progress = epoch / max(1, total_epochs - 1)
    return min_difficulty + (max_difficulty - min_difficulty) * progress


def exponential_curriculum_schedule(
    epoch: int,
    total_epochs: int,
    min_difficulty: float = 0.0,
    max_difficulty: float = 1.0,
    growth_rate: float = 3.0
) -> float:
    """
    指数课程调度
    
    Args:
        epoch: 当前轮数
        total_epochs: 总轮数
        min_difficulty: 最小难度
        max_difficulty: 最大难度
        growth_rate: 增长率
        
    Returns:
        当前难度
    """
    progress = epoch / max(1, total_epochs - 1)
    exp_progress = (np.exp(growth_rate * progress) - 1) / (np.exp(growth_rate) - 1)
    return min_difficulty + (max_difficulty - min_difficulty) * exp_progress
