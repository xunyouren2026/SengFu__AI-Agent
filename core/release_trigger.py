"""
策略层 - 爆发动作触发器 (Release Trigger Layer)

胜复学架构的第五层，负责在系统检测到持续异常（郁值持续高位）时，
触发爆发性调节动作以打破不良状态。使用多臂老虎机算法选择最优爆发动作。

核心功能:
- 多臂老虎机选择爆发动作 (复用multi_armed_bandit)
- 爆发动作库管理
- 爆发动作执行
- 延迟奖励计算 (爆发后100步评估郁值下降)
- 动作效果反馈

爆发动作:
1. 重置学习率 (reset_lr)
2. 添加参数噪声 (add_noise)
3. 切换优化器 (switch_optimizer)
4. 重置网络层 (reset_layer)
5. 增大dropout (increase_dropout)
6. 触发遗传编程 (genetic_programming)
7. 启动自我博弈 (self_play)
"""

import math
import random
import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    NamedTuple, Protocol
)
from enum import Enum, auto
from collections import deque

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD, AdamW

# 导入多臂老虎机
from agi_unified_framework.core.genesis.bandit.multi_armed_bandit import (
    MultiArmedBandit, BanditConfig, UCB1
)


# ============================================================
# 1. 类型定义与数据结构
# ============================================================

class EruptionActionType(Enum):
    """爆发动作类型"""
    RESET_LR = "reset_lr"                          # 重置学习率
    ADD_NOISE = "add_noise"                        # 添加参数噪声
    SWITCH_OPTIMIZER = "switch_optimizer"          # 切换优化器
    RESET_LAYER = "reset_layer"                    # 重置网络层
    INCREASE_DROPOUT = "increase_dropout"          # 增大dropout
    GENETIC_PROGRAMMING = "genetic_programming"    # 触发遗传编程
    SELF_PLAY = "self_play"                        # 启动自我博弈


@dataclass
class EruptionAction:
    """爆发动作定义"""
    action_type: EruptionActionType
    name: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    cooldown_steps: int = 100  # 冷却步数
    last_executed: float = field(default_factory=lambda: 0.0)

    def is_on_cooldown(self, current_step: int, cooldown_duration: int = 100) -> bool:
        """检查是否在冷却期"""
        return (current_step - self.last_executed) < cooldown_duration


@dataclass
class ActionEffect:
    """动作效果记录"""
    action_type: EruptionActionType
    timestamp: float
    pre_halt_value: float
    post_halt_value: float
    reward: float
    steps_to_evaluate: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReleaseConfig:
    """释放触发器配置"""
    # 爆发检测配置
    eruption_threshold: float = 0.8           # 爆发阈值
    eruption_duration_threshold: int = 50     # 持续步数阈值
    min_action_interval: int = 100            # 最小动作间隔

    # 多臂老虎机配置
    bandit_algorithm: str = "ucb1"            # 老虎机算法
    bandit_exploration_factor: float = 1.0    # 探索因子

    # 延迟奖励配置
    reward_evaluation_delay: int = 100        # 奖励评估延迟步数
    reward_discount_factor: float = 0.95      # 奖励折扣因子

    # 动作特定配置
    reset_lr_value: float = 1e-3              # 重置学习率值
    noise_std: float = 0.01                   # 噪声标准差
    dropout_increase_factor: float = 1.5      # dropout增加倍数
    max_dropout_rate: float = 0.5             # 最大dropout率

    # 遗传编程配置
    gp_population_size: int = 50              # GP种群大小
    gp_generations: int = 10                  # GP迭代次数

    # 自我博弈配置
    self_play_rounds: int = 10                # 自我博弈轮数


# ============================================================
# 2. 爆发动作执行器基类
# ============================================================

class ActionExecutor(ABC):
    """爆发动作执行器抽象基类"""

    @abstractmethod
    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行爆发动作

        Args:
            model: 神经网络模型
            optimizer: 优化器
            config: 配置
            **kwargs: 额外参数

        Returns:
            执行结果
        """
        pass

    @abstractmethod
    def get_action_type(self) -> EruptionActionType:
        """返回动作类型"""
        pass


class ResetLRExecutor(ActionExecutor):
    """重置学习率执行器"""

    def get_action_type(self) -> EruptionActionType:
        return EruptionActionType.RESET_LR

    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """重置学习率到基础值"""
        old_lrs = []
        for param_group in optimizer.param_groups:
            old_lrs.append(param_group['lr'])
            param_group['lr'] = config.reset_lr_value

        return {
            'action': 'reset_lr',
            'old_lrs': old_lrs,
            'new_lr': config.reset_lr_value,
            'message': f'学习率已重置为 {config.reset_lr_value}'
        }


class AddNoiseExecutor(ActionExecutor):
    """添加参数噪声执行器"""

    def get_action_type(self) -> EruptionActionType:
        return EruptionActionType.ADD_NOISE

    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        noise_scale: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """向模型参数添加高斯噪声"""
        std = noise_scale or config.noise_std
        affected_params = 0

        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * std
                param.data.add_(noise)
                affected_params += param.numel()

        return {
            'action': 'add_noise',
            'noise_std': std,
            'affected_params': affected_params,
            'message': f'已向 {affected_params} 个参数添加噪声 (std={std})'
        }


class SwitchOptimizerExecutor(ActionExecutor):
    """切换优化器执行器"""

    OPTIMIZER_MAP = {
        'adam': Adam,
        'sgd': SGD,
        'adamw': AdamW
    }

    def get_action_type(self) -> EruptionActionType:
        return EruptionActionType.SWITCH_OPTIMIZER

    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        target_optimizer: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """切换到不同的优化器"""
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 确定目标优化器类型
        current_type = type(optimizer).__name__.lower()
        if target_optimizer is None:
            # 自动选择不同的优化器
            alternatives = [k for k in self.OPTIMIZER_MAP.keys() if k != current_type]
            target_optimizer = random.choice(alternatives) if alternatives else 'adam'

        # 创建新优化器
        optimizer_class = self.OPTIMIZER_MAP.get(target_optimizer.lower(), Adam)
        new_optimizer = optimizer_class(model.parameters(), lr=current_lr)

        return {
            'action': 'switch_optimizer',
            'old_optimizer': current_type,
            'new_optimizer': target_optimizer,
            'new_optimizer_instance': new_optimizer,
            'message': f'优化器已从 {current_type} 切换为 {target_optimizer}'
        }


class ResetLayerExecutor(ActionExecutor):
    """重置网络层执行器"""

    def get_action_type(self) -> EruptionActionType:
        return EruptionActionType.RESET_LAYER

    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        layer_names: Optional[List[str]] = None,
        reset_probability: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """重置指定或随机选择的网络层"""
        reset_layers = []

        if layer_names:
            # 重置指定层
            for name, module in model.named_modules():
                if name in layer_names and hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                    reset_layers.append(name)
        else:
            # 随机选择层重置
            for name, module in model.named_modules():
                if hasattr(module, 'reset_parameters') and random.random() < reset_probability:
                    # 保存层类型信息
                    layer_info = type(module).__name__
                    module.reset_parameters()
                    reset_layers.append(f"{name}({layer_info})")

        return {
            'action': 'reset_layer',
            'reset_layers': reset_layers,
            'num_layers_reset': len(reset_layers),
            'message': f'已重置 {len(reset_layers)} 个网络层'
        }


class IncreaseDropoutExecutor(ActionExecutor):
    """增大Dropout执行器"""

    def get_action_type(self) -> EruptionActionType:
        return EruptionActionType.INCREASE_DROPOUT

    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        increase_factor: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """增大模型中的dropout率"""
        factor = increase_factor or config.dropout_increase_factor
        modified_layers = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                old_p = module.p
                new_p = min(old_p * factor, config.max_dropout_rate)
                module.p = new_p
                modified_layers.append({
                    'name': name,
                    'old_p': old_p,
                    'new_p': new_p
                })

        return {
            'action': 'increase_dropout',
            'modified_layers': modified_layers,
            'num_layers_modified': len(modified_layers),
            'message': f'已增大 {len(modified_layers)} 个dropout层的比率'
        }


class GeneticProgrammingExecutor(ActionExecutor):
    """遗传编程执行器"""

    def get_action_type(self) -> EruptionActionType:
        return EruptionActionType.GENETIC_PROGRAMMING

    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        fitness_fn: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        触发简化的遗传编程优化

        对模型结构进行小规模进化搜索。
        """
        # 简化的GP实现：对参数进行进化式扰动
        population_size = config.gp_population_size
        generations = config.gp_generations

        best_fitness = float('-inf')
        best_params = None

        # 保存原始参数
        original_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        # 简化的进化过程
        for gen in range(generations):
            # 生成变异
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 添加自适应噪声
                    noise_scale = 0.01 * (1 - gen / generations)
                    noise = torch.randn_like(param) * noise_scale
                    param.data.add_(noise)

            # 评估（简化：使用参数范数作为适应度代理）
            if fitness_fn:
                fitness = fitness_fn(model)
            else:
                fitness = -sum(p.norm().item() for p in model.parameters())

            if fitness > best_fitness:
                best_fitness = fitness
                best_params = {
                    name: param.data.clone()
                    for name, param in model.named_parameters()
                }

        # 恢复最佳参数
        if best_params:
            for name, param in model.named_parameters():
                if name in best_params:
                    param.data.copy_(best_params[name])

        return {
            'action': 'genetic_programming',
            'population_size': population_size,
            'generations': generations,
            'best_fitness': best_fitness,
            'message': f'遗传编程完成，最佳适应度: {best_fitness:.4f}'
        }


class SelfPlayExecutor(ActionExecutor):
    """自我博弈执行器"""

    def get_action_type(self) -> EruptionActionType:
        return EruptionActionType.SELF_PLAY

    def execute(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: ReleaseConfig,
        environment: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        启动自我博弈

        模型与自身的历史版本进行对抗/协作。
        """
        rounds = config.self_play_rounds

        # 保存当前模型状态
        current_state = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        # 简化的自我博弈：参数插值
        # 实际实现中应该使用更复杂的博弈逻辑
        improvements = 0

        for round_idx in range(rounds):
            # 创建历史版本（参数平滑）
            alpha = 0.5 + 0.3 * math.sin(round_idx * math.pi / rounds)

            for name, param in model.named_parameters():
                if name in current_state:
                    # 插值更新
                    historical = current_state[name] * (1 - alpha)
                    current = param.data * alpha
                    param.data.copy_(historical + current)

            improvements += 1

        return {
            'action': 'self_play',
            'rounds': rounds,
            'improvements': improvements,
            'message': f'自我博弈完成，执行了 {rounds} 轮优化'
        }


# ============================================================
# 3. 爆发动作注册表
# ============================================================

class ActionRegistry:
    """爆发动作注册表"""

    def __init__(self):
        self._executors: Dict[EruptionActionType, ActionExecutor] = {}
        self._actions: Dict[EruptionActionType, EruptionAction] = {}
        self._register_default_actions()

    def _register_default_actions(self):
        """注册默认爆发动作"""
        default_actions = [
            (EruptionActionType.RESET_LR, ResetLRExecutor(),
             "重置学习率到基础值，打破局部最优"),
            (EruptionActionType.ADD_NOISE, AddNoiseExecutor(),
             "向参数添加高斯噪声，增加探索"),
            (EruptionActionType.SWITCH_OPTIMIZER, SwitchOptimizerExecutor(),
             "切换优化器类型，改变优化动态"),
            (EruptionActionType.RESET_LAYER, ResetLayerExecutor(),
             "重置部分网络层，重新学习特征"),
            (EruptionActionType.INCREASE_DROPOUT, IncreaseDropoutExecutor(),
             "增大dropout率，增强正则化"),
            (EruptionActionType.GENETIC_PROGRAMMING, GeneticProgrammingExecutor(),
             "触发遗传编程，进化搜索最优结构"),
            (EruptionActionType.SELF_PLAY, SelfPlayExecutor(),
             "启动自我博弈，通过与自身对抗提升"),
        ]

        for action_type, executor, description in default_actions:
            self.register(action_type, executor, description)

    def register(
        self,
        action_type: EruptionActionType,
        executor: ActionExecutor,
        description: str = "",
        params: Optional[Dict[str, Any]] = None
    ):
        """注册爆发动作"""
        self._executors[action_type] = executor
        self._actions[action_type] = EruptionAction(
            action_type=action_type,
            name=action_type.value,
            description=description,
            params=params or {}
        )

    def get_executor(self, action_type: EruptionActionType) -> Optional[ActionExecutor]:
        """获取执行器"""
        return self._executors.get(action_type)

    def get_action(self, action_type: EruptionActionType) -> Optional[EruptionAction]:
        """获取动作定义"""
        return self._actions.get(action_type)

    def list_actions(self) -> List[EruptionAction]:
        """列出所有可用动作"""
        return list(self._actions.values())

    def is_action_available(
        self,
        action_type: EruptionActionType,
        current_step: int
    ) -> bool:
        """检查动作是否可用（不在冷却期）"""
        action = self._actions.get(action_type)
        if action is None:
            return False
        return not action.is_on_cooldown(current_step, action.cooldown_steps)

    def mark_executed(self, action_type: EruptionActionType, step: int):
        """标记动作已执行"""
        if action_type in self._actions:
            self._actions[action_type].last_executed = step


# ============================================================
# 4. 延迟奖励计算器
# ============================================================

class DelayedRewardCalculator:
    """
    延迟奖励计算器

    在爆发动作执行后，经过一定步数评估其效果。
    奖励 = 郁值下降程度 + 训练稳定性提升
    """

    def __init__(self, evaluation_delay: int = 100):
        self.evaluation_delay = evaluation_delay
        self.pending_evaluations: deque = deque()
        self.completed_evaluations: List[ActionEffect] = []

    def record_action(
        self,
        action_type: EruptionActionType,
        pre_halt_value: float,
        current_step: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """记录动作执行，等待后续评估"""
        self.pending_evaluations.append({
            'action_type': action_type,
            'pre_halt_value': pre_halt_value,
            'execution_step': current_step,
            'evaluation_step': current_step + self.evaluation_delay,
            'metadata': metadata or {}
        })

    def evaluate(
        self,
        current_halt_value: float,
        current_step: int
    ) -> List[ActionEffect]:
        """
        评估待处理的爆发动作

        Args:
            current_halt_value: 当前郁值
            current_step: 当前步数

        Returns:
            完成的评估列表
        """
        completed = []
        remaining = []

        for pending in self.pending_evaluations:
            if current_step >= pending['evaluation_step']:
                # 计算奖励
                halt_improvement = pending['pre_halt_value'] - current_halt_value

                # 奖励设计：
                # 1. 郁值下降为正奖励
                # 2. 郁值上升为负奖励
                # 3. 添加稳定性奖励（郁值保持在低位）
                base_reward = halt_improvement * 10  # 缩放因子

                # 如果郁值保持在低位，给予额外奖励
                stability_bonus = 0.0
                if current_halt_value < 0.3:
                    stability_bonus = 0.5

                reward = base_reward + stability_bonus

                effect = ActionEffect(
                    action_type=pending['action_type'],
                    timestamp=time.time(),
                    pre_halt_value=pending['pre_halt_value'],
                    post_halt_value=current_halt_value,
                    reward=reward,
                    steps_to_evaluate=self.evaluation_delay,
                    metadata=pending['metadata']
                )

                completed.append(effect)
                self.completed_evaluations.append(effect)
            else:
                remaining.append(pending)

        self.pending_evaluations = deque(remaining)
        return completed

    def get_average_reward(self, action_type: EruptionActionType, n_recent: int = 10) -> float:
        """获取某动作类型的平均奖励"""
        relevant = [
            e for e in self.completed_evaluations
            if e.action_type == action_type
        ][-n_recent:]

        if not relevant:
            return 0.0

        return sum(e.reward for e in relevant) / len(relevant)

    def get_action_statistics(self) -> Dict[EruptionActionType, Dict[str, float]]:
        """获取各动作的统计信息"""
        stats = {}

        for action_type in EruptionActionType:
            effects = [e for e in self.completed_evaluations if e.action_type == action_type]
            if effects:
                rewards = [e.reward for e in effects]
                stats[action_type] = {
                    'count': len(effects),
                    'mean_reward': sum(rewards) / len(rewards),
                    'min_reward': min(rewards),
                    'max_reward': max(rewards),
                    'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
                }
            else:
                stats[action_type] = {
                    'count': 0,
                    'mean_reward': 0.0,
                    'min_reward': 0.0,
                    'max_reward': 0.0,
                    'success_rate': 0.0
                }

        return stats


# ============================================================
# 5. 主释放触发器类
# ============================================================

class ReleaseTrigger:
    """
    释放触发器

    胜复学架构的策略层核心类，负责检测爆发条件、选择并执行爆发动作。

    Attributes:
        config: 配置
        action_registry: 动作注册表
        reward_calculator: 延迟奖励计算器
        bandit: 多臂老虎机
        halt_history: 郁值历史
        action_history: 动作执行历史
    """

    def __init__(self, actions: Optional[List[EruptionActionType]] = None, config: Optional[ReleaseConfig] = None):
        """
        初始化释放触发器

        Args:
            actions: 可用的爆发动作列表，默认为全部
            config: 配置
        """
        self.config = config or ReleaseConfig()
        self.action_registry = ActionRegistry()
        self.reward_calculator = DelayedRewardCalculator(
            self.config.reward_evaluation_delay
        )

        # 初始化多臂老虎机
        available_actions = actions or list(EruptionActionType)
        self.num_actions = len(available_actions)
        self.action_types = available_actions
        self.action_to_index = {a: i for i, a in enumerate(available_actions)}

        # 使用UCB1作为默认策略
        self.bandit = UCB1(
            num_arms=self.num_actions,
            c=self.config.bandit_exploration_factor
        )

        # 状态
        self.halt_history: deque = deque(maxlen=200)
        self.action_history: List[Dict[str, Any]] = []
        self.current_step: int = 0
        self.last_action_step: int = -self.config.min_action_interval
        self.eruption_count: int = 0

        # 郁值持续高位计数
        self.consecutive_high_halt: int = 0

    def check_eruption(self, halt_value: float, duration: Optional[int] = None) -> bool:
        """
        检查是否应该触发爆发

        Args:
            halt_value: 当前郁值
            duration: 持续步数（可选，使用历史记录计算）

        Returns:
            是否应该爆发
        """
        self.halt_history.append(halt_value)
        self.current_step += 1

        # 检查最小动作间隔
        if self.current_step - self.last_action_step < self.config.min_action_interval:
            return False

        # 检查郁值是否超过阈值
        if halt_value < self.config.eruption_threshold:
            self.consecutive_high_halt = 0
            return False

        # 累计持续高位
        self.consecutive_high_halt += 1

        # 检查持续步数
        duration_threshold = duration or self.config.eruption_duration_threshold
        if self.consecutive_high_halt >= duration_threshold:
            return True

        # 或者检查历史平均值
        if len(self.halt_history) >= duration_threshold:
            recent_avg = sum(list(self.halt_history)[-duration_threshold:]) / duration_threshold
            if recent_avg > self.config.eruption_threshold:
                return True

        return False

    def select_action(self) -> EruptionActionType:
        """
        使用多臂老虎机选择爆发动作

        Returns:
            选择的动作类型
        """
        # 获取可用动作（不在冷却期）
        available_indices = []
        for i, action_type in enumerate(self.action_types):
            if self.action_registry.is_action_available(action_type, self.current_step):
                available_indices.append(i)

        if not available_indices:
            # 所有动作都在冷却期，强制选择第一个
            return self.action_types[0]

        # 使用老虎机选择
        selected_index = self.bandit.select_arm()

        # 如果选择的动作不可用，从可用动作中随机选择
        if selected_index not in available_indices:
            selected_index = random.choice(available_indices)

        return self.action_types[selected_index]

    def execute_action(
        self,
        action: EruptionActionType,
        model: nn.Module,
        optimizer: Optimizer,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行爆发动作

        Args:
            action: 动作类型
            model: 模型
            optimizer: 优化器
            **kwargs: 额外参数

        Returns:
            执行结果
        """
        executor = self.action_registry.get_executor(action)
        if executor is None:
            return {
                'success': False,
                'error': f'未找到动作 {action} 的执行器'
            }

        # 执行动作
        try:
            result = executor.execute(model, optimizer, self.config, **kwargs)
            result['success'] = True
            result['action_type'] = action.value
            result['timestamp'] = time.time()
            result['step'] = self.current_step
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'action_type': action.value
            }

        # 更新历史
        self.action_history.append(result)
        self.last_action_step = self.current_step
        self.eruption_count += 1

        # 标记动作已执行（进入冷却期）
        self.action_registry.mark_executed(action, self.current_step)

        # 记录待评估
        current_halt = self.halt_history[-1] if self.halt_history else 0.5
        self.reward_calculator.record_action(
            action_type=action,
            pre_halt_value=current_halt,
            current_step=self.current_step,
            metadata=result
        )

        return result

    def compute_delayed_reward(self, current_halt_value: float) -> List[Dict[str, Any]]:
        """
        计算延迟奖励

        Args:
            current_halt_value: 当前郁值

        Returns:
            完成的评估列表
        """
        completed = self.reward_calculator.evaluate(
            current_halt_value,
            self.current_step
        )

        # 更新老虎机
        for effect in completed:
            action_index = self.action_to_index.get(effect.action_type)
            if action_index is not None:
                # 归一化奖励到[0, 1]范围用于老虎机
                normalized_reward = (effect.reward + 5) / 10  # 假设奖励范围[-5, 5]
                normalized_reward = max(0.0, min(1.0, normalized_reward))
                self.bandit.update(action_index, normalized_reward)

        return [self._effect_to_dict(e) for e in completed]

    def _effect_to_dict(self, effect: ActionEffect) -> Dict[str, Any]:
        """转换ActionEffect为字典"""
        return {
            'action_type': effect.action_type.value,
            'timestamp': effect.timestamp,
            'pre_halt_value': effect.pre_halt_value,
            'post_halt_value': effect.post_halt_value,
            'reward': effect.reward,
            'metadata': effect.metadata
        }

    def update_bandit(self, action: EruptionActionType, reward: float):
        """
        手动更新老虎机

        Args:
            action: 动作类型
            reward: 奖励值（应归一化到[0, 1]）
        """
        action_index = self.action_to_index.get(action)
        if action_index is not None:
            self.bandit.update(action_index, reward)

    def trigger_eruption(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        halt_value: float,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        触发爆发（完整流程）

        Args:
            model: 模型
            optimizer: 优化器
            halt_value: 当前郁值
            **kwargs: 传递给动作执行器的参数

        Returns:
            爆发结果，如果未触发则返回None
        """
        # 检查是否应该爆发
        if not self.check_eruption(halt_value):
            # 仍然检查延迟奖励
            self.compute_delayed_reward(halt_value)
            return None

        # 选择动作
        action = self.select_action()

        # 执行动作
        result = self.execute_action(action, model, optimizer, **kwargs)

        # 添加额外信息
        result['eruption_triggered'] = True
        result['halt_value'] = halt_value
        result['consecutive_high_halt'] = self.consecutive_high_halt

        # 重置持续计数
        self.consecutive_high_halt = 0

        return result

    def get_action_recommendations(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        获取动作推荐（基于老虎机估值）

        Args:
            n: 推荐数量

        Returns:
            推荐列表
        """
        # 获取每个动作的估值
        action_values = []
        for i, action_type in enumerate(self.action_types):
            value = self.bandit.values[i] if hasattr(self.bandit, 'values') else 0.0
            count = self.bandit.counts[i] if hasattr(self.bandit, 'counts') else 0
            action_values.append({
                'action': action_type.value,
                'estimated_value': value,
                'execution_count': count,
                'available': self.action_registry.is_action_available(
                    action_type, self.current_step
                )
            })

        # 按估值排序
        action_values.sort(key=lambda x: x['estimated_value'], reverse=True)

        return action_values[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """获取触发器统计信息"""
        action_stats = self.reward_calculator.get_action_statistics()

        return {
            'eruption_count': self.eruption_count,
            'current_step': self.current_step,
            'halt_history_size': len(self.halt_history),
            'average_halt': sum(self.halt_history) / len(self.halt_history) if self.halt_history else 0.0,
            'action_history_size': len(self.action_history),
            'consecutive_high_halt': self.consecutive_high_halt,
            'action_statistics': {
                k.value: v for k, v in action_stats.items()
            },
            'recommendations': self.get_action_recommendations()
        }

    def reset(self):
        """重置触发器状态"""
        self.halt_history.clear()
        self.action_history.clear()
        self.current_step = 0
        self.last_action_step = -self.config.min_action_interval
        self.eruption_count = 0
        self.consecutive_high_halt = 0

        # 重置老虎机
        self.bandit = UCB1(
            num_arms=self.num_actions,
            c=self.config.bandit_exploration_factor
        )


# ============================================================
# 6. 便捷函数
# ============================================================

def create_release_trigger(
    available_actions: Optional[List[str]] = None,
    eruption_threshold: float = 0.8,
    evaluation_delay: int = 100
) -> ReleaseTrigger:
    """
    创建释放触发器的便捷函数

    Args:
        available_actions: 可用动作名称列表，默认为全部
        eruption_threshold: 爆发阈值
        evaluation_delay: 奖励评估延迟

    Returns:
        配置好的ReleaseTrigger实例
    """
    config = ReleaseConfig(
        eruption_threshold=eruption_threshold,
        reward_evaluation_delay=evaluation_delay
    )

    if available_actions:
        action_types = [
            EruptionActionType(a) for a in available_actions
            if a in [e.value for e in EruptionActionType]
        ]
    else:
        action_types = None

    return ReleaseTrigger(actions=action_types, config=config)


def quick_eruption_check(
    trigger: ReleaseTrigger,
    halt_value: float,
    model: nn.Module,
    optimizer: Optimizer
) -> Optional[Dict[str, Any]]:
    """
    快速爆发检查函数

    Args:
        trigger: 释放触发器
        halt_value: 当前郁值
        model: 模型
        optimizer: 优化器

    Returns:
        爆发结果，如果未触发则返回None
    """
    return trigger.trigger_eruption(model, optimizer, halt_value)
