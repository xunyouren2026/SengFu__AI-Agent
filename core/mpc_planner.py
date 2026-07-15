"""
MPC模型预测控制规划器 (Model Predictive Control Planner)

该模块实现了基于模型预测控制的规划器,用于动作序列优化和决策制定。
通过世界模型rollout、CEM采样优化和约束处理,实现多步预测和最优控制。

主要特性:
- 动作序列优化
- 世界模型rollout
- CEM采样优化
- 约束处理
- 多步预测

作者: AGI Unified Framework Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
import math


@dataclass
class MPCConfig:
    """MPC配置类
    
    Attributes:
        horizon: 预测时域
        num_samples: CEM采样数量
        num_elites: 精英样本数量
        num_iterations: CEM迭代次数
        action_dim: 动作维度
        state_dim: 状态维度
        action_min: 动作最小值
        action_max: 动作最大值
        gamma: 折扣因子
        temperature: 采样温度
        use_constraints: 是否使用约束
        constraint_penalty: 约束违反惩罚
        replan_interval: 重规划间隔
        keep_previous_solution: 是否保留上次解作为初始化
    """
    horizon: int = 10
    num_samples: int = 1000
    num_elites: int = 100
    num_iterations: int = 5
    action_dim: int = 4
    state_dim: int = 10
    action_min: float = -1.0
    action_max: float = 1.0
    gamma: float = 0.99
    temperature: float = 1.0
    use_constraints: bool = True
    constraint_penalty: float = 100.0
    replan_interval: int = 1
    keep_previous_solution: bool = True


class WorldModel(ABC):
    """世界模型基类
    
    用于预测状态转移的抽象基类。
    """
    
    @abstractmethod
    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        预测下一状态
        
        Args:
            state: 当前状态
            action: 执行动作
            
        Returns:
            (下一状态, 额外信息)
        """
        pass
    
    @abstractmethod
    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行多步rollout
        
        Args:
            initial_state: 初始状态
            actions: 动作序列 [horizon, action_dim] 或 [batch, horizon, action_dim]
            
        Returns:
            (状态序列, 奖励序列)
        """
        pass


class NeuralWorldModel(WorldModel, nn.Module):
    """神经网络世界模型
    
    使用神经网络学习状态转移动力学。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        deterministic: bool = False
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            deterministic: 是否确定性模型
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.deterministic = deterministic
        
        # 构建网络
        layers = []
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # 状态预测头
        self.state_head = nn.Linear(hidden_dim, state_dim)
        
        # 奖励预测头
        self.reward_head = nn.Linear(hidden_dim, 1)
        
        # 不确定性估计(仅用于随机模型)
        if not deterministic:
            self.uncertainty_head = nn.Linear(hidden_dim, state_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 当前状态
            action: 执行动作
            
        Returns:
            包含预测结果的字典
        """
        x = torch.cat([state, action], dim=-1)
        features = self.network(x)
        
        # 预测状态变化
        state_delta = self.state_head(features)
        next_state = state + state_delta
        
        # 预测奖励
        reward = self.reward_head(features).squeeze(-1)
        
        result = {
            'next_state': next_state,
            'reward': reward,
            'state_delta': state_delta
        }
        
        # 预测不确定性
        if not self.deterministic:
            uncertainty = torch.exp(self.uncertainty_head(features))
            result['uncertainty'] = uncertainty
        
        return result
    
    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        预测下一状态
        
        Args:
            state: 当前状态
            action: 执行动作
            
        Returns:
            (下一状态, 额外信息)
        """
        with torch.no_grad():
            result = self.forward(state, action)
        
        next_state = result['next_state']
        info = {
            'reward': result['reward'],
            'state_delta': result['state_delta']
        }
        
        if 'uncertainty' in result:
            info['uncertainty'] = result['uncertainty']
        
        return next_state, info
    
    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行多步rollout
        
        Args:
            initial_state: 初始状态 [batch, state_dim] 或 [state_dim]
            actions: 动作序列 [horizon, action_dim] 或 [batch, horizon, action_dim]
            
        Returns:
            (状态序列, 奖励序列)
        """
        # 处理输入维度
        single_batch = False
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
            single_batch = True
        
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        
        batch_size, horizon, _ = actions.shape
        device = initial_state.device
        
        # 初始化
        states = torch.zeros(batch_size, horizon + 1, self.state_dim, device=device)
        rewards = torch.zeros(batch_size, horizon, device=device)
        
        states[:, 0] = initial_state
        current_state = initial_state
        
        # 逐步预测
        for t in range(horizon):
            action = actions[:, t]
            next_state, info = self.predict(current_state, action)
            
            states[:, t + 1] = next_state
            rewards[:, t] = info['reward']
            
            current_state = next_state
        
        if single_batch:
            states = states.squeeze(0)
            rewards = rewards.squeeze(0)
        
        return states, rewards


class CEMOptimizer:
    """交叉熵方法优化器
    
    使用CEM进行动作序列优化。
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_elites: int = 100,
        num_iterations: int = 5,
        action_dim: int = 4,
        horizon: int = 10,
        action_min: float = -1.0,
        action_max: float = 1.0,
        temperature: float = 1.0
    ):
        """
        Args:
            num_samples: 采样数量
            num_elites: 精英样本数量
            num_iterations: 迭代次数
            action_dim: 动作维度
            horizon: 预测时域
            action_min: 动作最小值
            action_max: 动作最大值
            temperature: 采样温度
        """
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_min = action_min
        self.action_max = action_max
        self.temperature = temperature
        
        # 初始化分布参数
        self.mean = torch.zeros(horizon, action_dim)
        self.std = torch.ones(horizon, action_dim)
    
    def optimize(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        initial_mean: Optional[torch.Tensor] = None,
        initial_std: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行CEM优化
        
        Args:
            objective_fn: 目标函数,接收动作序列返回分数
            initial_mean: 初始均值
            initial_std: 初始标准差
            
        Returns:
            (最优动作序列, 分数)
        """
        # 初始化分布
        if initial_mean is not None:
            mean = initial_mean.clone()
        else:
            mean = self.mean.clone()
        
        if initial_std is not None:
            std = initial_std.clone()
        else:
            std = self.std.clone()
        
        best_action = None
        best_score = float('-inf')
        
        for iteration in range(self.num_iterations):
            # 采样动作序列
            actions = self._sample_actions(mean, std)
            
            # 评估
            scores = objective_fn(actions)
            
            # 选择精英
            elite_indices = torch.argsort(scores, descending=True)[:self.num_elites]
            elite_actions = actions[elite_indices]
            elite_scores = scores[elite_indices]
            
            # 更新分布
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6
            
            # 更新最优解
            if elite_scores[0] > best_score:
                best_score = elite_scores[0]
                best_action = elite_actions[0]
        
        # 保存最终分布
        self.mean = mean
        self.std = std
        
        return best_action, best_score
    
    def _sample_actions(
        self,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> torch.Tensor:
        """
        采样动作序列
        
        Args:
            mean: 均值
            std: 标准差
            
        Returns:
            动作序列样本 [num_samples, horizon, action_dim]
        """
        # 重参数化采样
        eps = torch.randn(self.num_samples, self.horizon, self.action_dim)
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        
        # 裁剪到动作范围
        actions = torch.clamp(actions, self.action_min, self.action_max)
        
        return actions
    
    def reset(self):
        """重置优化器"""
        self.mean = torch.zeros(self.horizon, self.action_dim)
        self.std = torch.ones(self.horizon, self.action_dim)


class ConstraintHandler:
    """约束处理器
    
    处理状态和动作约束。
    """
    
    def __init__(
        self,
        state_constraints: Optional[List[Callable]] = None,
        action_constraints: Optional[List[Callable]] = None,
        penalty_weight: float = 100.0
    ):
        """
        Args:
            state_constraints: 状态约束函数列表
            action_constraints: 动作约束函数列表
            penalty_weight: 惩罚权重
        """
        self.state_constraints = state_constraints or []
        self.action_constraints = action_constraints or []
        self.penalty_weight = penalty_weight
    
    def check_state_constraints(self, state: torch.Tensor) -> torch.Tensor:
        """
        检查状态约束
        
        Args:
            state: 状态张量
            
        Returns:
            约束违反惩罚
        """
        penalty = torch.zeros(state.shape[0], device=state.device)
        
        for constraint_fn in self.state_constraints:
            violation = constraint_fn(state)
            penalty += torch.clamp(violation, min=0)
        
        return penalty * self.penalty_weight
    
    def check_action_constraints(self, action: torch.Tensor) -> torch.Tensor:
        """
        检查动作约束
        
        Args:
            action: 动作张量
            
        Returns:
            约束违反惩罚
        """
        penalty = torch.zeros(action.shape[0], device=action.device)
        
        for constraint_fn in self.action_constraints:
            violation = constraint_fn(action)
            penalty += torch.clamp(violation, min=0)
        
        return penalty * self.penalty_weight
    
    def compute_total_penalty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        计算总约束惩罚
        
        Args:
            states: 状态序列
            actions: 动作序列
            
        Returns:
            总惩罚
        """
        state_penalty = self.check_state_constraints(states.reshape(-1, states.shape[-1]))
        action_penalty = self.check_action_constraints(actions.reshape(-1, actions.shape[-1]))
        
        total_penalty = state_penalty.sum() + action_penalty.sum()
        
        return total_penalty


class MPCPlanner:
    """MPC规划器主类
    
    实现完整的模型预测控制流程,包括动作优化、
    世界模型rollout和约束处理。
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        horizon: int = 10,
        config: Optional[MPCConfig] = None
    ):
        """
        Args:
            world_model: 世界模型
            horizon: 预测时域
            config: 配置
        """
        self.world_model = world_model
        self.config = config or MPCConfig()
        
        # 更新配置
        self.config.horizon = horizon
        
        # 初始化CEM优化器
        self.cem_optimizer = CEMOptimizer(
            num_samples=self.config.num_samples,
            num_elites=self.config.num_elites,
            num_iterations=self.config.num_iterations,
            action_dim=self.config.action_dim,
            horizon=horizon,
            action_min=self.config.action_min,
            action_max=self.config.action_max,
            temperature=self.config.temperature
        )
        
        # 初始化约束处理器
        self.constraint_handler = ConstraintHandler(
            penalty_weight=self.config.constraint_penalty
        ) if self.config.use_constraints else None
        
        # 当前动作序列(用于warm start)
        self.current_action_plan: Optional[torch.Tensor] = None
        self.plan_step = 0
    
    def plan(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        cost_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        规划动作序列
        
        Args:
            state: 当前状态
            goal: 目标状态,可选
            cost_fn: 代价函数,可选
            
        Returns:
            最优动作序列
        """
        # 定义目标函数
        def objective_fn(actions: torch.Tensor) -> torch.Tensor:
            return self._evaluate_action_sequences(state, actions, goal, cost_fn)
        
        # 准备初始分布
        initial_mean = None
        initial_std = None
        
        if self.config.keep_previous_solution and self.current_action_plan is not None:
            # 使用时间平移的先前解作为初始化
            previous_plan = self.current_action_plan
            shifted_plan = torch.cat([
                previous_plan[1:],
                torch.zeros(1, self.config.action_dim)
            ], dim=0)
            initial_mean = shifted_plan
            initial_std = torch.ones_like(shifted_plan) * 0.5
        
        # CEM优化
        optimal_actions, score = self.cem_optimizer.optimize(
            objective_fn, initial_mean, initial_std
        )
        
        # 保存规划结果
        self.current_action_plan = optimal_actions
        self.plan_step = 0
        
        return optimal_actions
    
    def rollout_action_sequence(
        self,
        state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行动作序列rollout
        
        Args:
            state: 初始状态
            actions: 动作序列
            
        Returns:
            (预测状态序列, 预测奖励序列)
        """
        return self.world_model.rollout(state, actions)
    
    def cem_optimize(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        cost_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        使用CEM优化动作
        
        Args:
            state: 当前状态
            goal: 目标状态
            cost_fn: 代价函数
            
        Returns:
            最优动作
        """
        optimal_actions = self.plan(state, goal, cost_fn)
        return optimal_actions
    
    def execute_plan(self, state: torch.Tensor) -> torch.Tensor:
        """
        执行规划,返回当前动作
        
        Args:
            state: 当前状态
            
        Returns:
            当前执行的动作
        """
        # 检查是否需要重新规划
        if (
            self.current_action_plan is None or
            self.plan_step >= self.config.replan_interval or
            self.plan_step >= len(self.current_action_plan)
        ):
            self.plan(state)
        
        # 获取当前动作
        action = self.current_action_plan[self.plan_step]
        self.plan_step += 1
        
        return action
    
    def _evaluate_action_sequences(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        goal: Optional[torch.Tensor],
        cost_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    ) -> torch.Tensor:
        """
        评估动作序列
        
        Args:
            state: 初始状态
            actions: 动作序列批次 [num_samples, horizon, action_dim]
            goal: 目标状态
            cost_fn: 代价函数
            
        Returns:
            评估分数 [num_samples]
        """
        num_samples = actions.shape[0]
        device = actions.device
        
        # 扩展状态以匹配批次
        if state.dim() == 1:
            state = state.unsqueeze(0).expand(num_samples, -1)
        
        # rollout
        total_rewards = torch.zeros(num_samples, device=device)
        
        current_state = state
        for t in range(self.config.horizon):
            action = actions[:, t]
            
            # 预测
            next_state, info = self.world_model.predict(current_state, action)
            reward = info['reward']
            
            # 折扣累积
            discount = self.config.gamma ** t
            total_rewards += discount * reward
            
            # 约束惩罚
            if self.constraint_handler is not None:
                state_penalty = self.constraint_handler.check_state_constraints(next_state)
                action_penalty = self.constraint_handler.check_action_constraints(action)
                total_rewards -= discount * (state_penalty + action_penalty)
            
            # 目标代价
            if goal is not None:
                goal_reward = self._compute_goal_reward(next_state, goal)
                total_rewards += discount * goal_reward
            
            # 自定义代价
            if cost_fn is not None:
                custom_cost = cost_fn(next_state, action)
                total_rewards -= discount * custom_cost
            
            current_state = next_state
        
        return total_rewards
    
    def _compute_goal_reward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor
    ) -> torch.Tensor:
        """
        计算目标奖励
        
        Args:
            state: 当前状态
            goal: 目标状态
            
        Returns:
            目标奖励
        """
        # 负距离作为奖励
        distance = torch.norm(state - goal, dim=-1)
        reward = -distance
        return reward
    
    def reset(self):
        """重置规划器"""
        self.current_action_plan = None
        self.plan_step = 0
        self.cem_optimizer.reset()
    
    def set_constraints(
        self,
        state_constraints: Optional[List[Callable]] = None,
        action_constraints: Optional[List[Callable]] = None
    ):
        """
        设置约束
        
        Args:
            state_constraints: 状态约束
            action_constraints: 动作约束
        """
        if self.constraint_handler is not None:
            if state_constraints is not None:
                self.constraint_handler.state_constraints = state_constraints
            if action_constraints is not None:
                self.constraint_handler.action_constraints = action_constraints
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """
        获取规划统计
        
        Returns:
            统计信息字典
        """
        return {
            'horizon': self.config.horizon,
            'plan_step': self.plan_step,
            'has_active_plan': self.current_action_plan is not None,
            'cem_mean': self.cem_optimizer.mean.numpy(),
            'cem_std': self.cem_optimizer.std.numpy()
        }


# 辅助函数
def create_mpc_planner(
    world_model: WorldModel,
    config_dict: Optional[Dict] = None
) -> MPCPlanner:
    """
    从配置创建MPC规划器
    
    Args:
        world_model: 世界模型
        config_dict: 配置字典
        
    Returns:
        MPCPlanner实例
    """
    if config_dict:
        config = MPCConfig(**config_dict)
    else:
        config = None
    
    return MPCPlanner(world_model, config=config)


def create_neural_world_model(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    deterministic: bool = False
) -> NeuralWorldModel:
    """
    创建神经网络世界模型
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        deterministic: 是否确定性
        
    Returns:
        NeuralWorldModel实例
    """
    return NeuralWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        deterministic=deterministic
    )


def box_constraint(
    state: torch.Tensor,
    min_bounds: torch.Tensor,
    max_bounds: torch.Tensor
) -> torch.Tensor:
    """
    盒约束
    
    检查状态是否在边界内。
    
    Args:
        state: 状态
        min_bounds: 最小边界
        max_bounds: 最大边界
        
    Returns:
        约束违反量
    """
    lower_violation = torch.clamp(min_bounds - state, min=0)
    upper_violation = torch.clamp(state - max_bounds, min=0)
    
    total_violation = lower_violation.sum(dim=-1) + upper_violation.sum(dim=-1)
    return total_violation


def collision_constraint(
    state: torch.Tensor,
    obstacles: List[torch.Tensor],
    min_distance: float = 1.0
) -> torch.Tensor:
    """
    碰撞约束
    
    检查与障碍物的距离。
    
    Args:
        state: 状态(位置)
        obstacles: 障碍物位置列表
        min_distance: 最小距离
        
    Returns:
        约束违反量
    """
    violations = torch.zeros(state.shape[0], device=state.device)
    
    for obstacle in obstacles:
        distance = torch.norm(state - obstacle, dim=-1)
        violation = torch.clamp(min_distance - distance, min=0)
        violations += violation
    
    return violations


def smoothness_cost(
    actions: torch.Tensor,
    weight: float = 0.1
) -> torch.Tensor:
    """
    平滑度代价
    
    惩罚动作变化过大。
    
    Args:
        actions: 动作序列 [horizon, action_dim]
        weight: 权重
        
    Returns:
        平滑度代价
    """
    if len(actions) < 2:
        return torch.tensor(0.0)
    
    action_diff = actions[1:] - actions[:-1]
    cost = weight * torch.sum(action_diff ** 2)
    
    return cost


def energy_cost(
    actions: torch.Tensor,
    weight: float = 0.01
) -> torch.Tensor:
    """
    能量代价
    
    惩罚大动作。
    
    Args:
        actions: 动作序列
        weight: 权重
        
    Returns:
        能量代价
    """
    return weight * torch.sum(actions ** 2)
