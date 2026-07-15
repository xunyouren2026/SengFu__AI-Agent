"""
多臂老虎机模块 - Multi-Armed Bandit

实现多种老虎机算法：
- UCB (Upper Confidence Bound)
- Thompson Sampling
- ε-Greedy
- Contextual Bandits
- EXP3 (对抗性老虎机)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
import random


@dataclass
class BanditConfig:
    """老虎机配置"""
    num_arms: int = 10
    context_dim: int = 0  # 上下文维度（0表示非上下文）
    seed: int = 42


class BanditBase(ABC):
    """老虎机基类"""
    
    def __init__(self, config: BanditConfig):
        self.config = config
        self.num_arms = config.num_arms
        
        # 统计
        self.counts = np.zeros(num_arms)  # 每个臂的拉动次数
        self.values = np.zeros(num_arms)   # 每个臂的平均奖励
        self.total_reward = 0.0
        self.total_pulls = 0
        
        # 随机数生成器
        self.rng = np.random.RandomState(config.seed)
    
    @abstractmethod
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """选择臂"""
        pass
    
    def update(self, arm: int, reward: float) -> None:
        """更新统计"""
        self.counts[arm] += 1
        n = self.counts[arm]
        
        # 增量更新均值
        self.values[arm] = self.values[arm] + (reward - self.values[arm]) / n
        
        self.total_reward += reward
        self.total_pulls += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'total_reward': self.total_reward,
            'total_pulls': self.total_pulls,
            'average_reward': self.total_reward / max(1, self.total_pulls)
        }


class UCB(BanditBase):
    """
    Upper Confidence Bound
    
    选择使上置信界最大的臂：
    arm = argmax_i [μ_i + c * sqrt(ln(t) / n_i)]
    
    其中c是探索参数，通常取sqrt(2)。
    """
    
    def __init__(self, config: BanditConfig, c: float = 1.0):
        super().__init__(config)
        self.c = c  # 探索参数
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        # 初始化：每个臂至少拉一次
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
        
        # UCB计算
        t = self.total_pulls
        ucb_values = self.values + self.c * np.sqrt(
            np.log(t) / self.counts
        )
        
        return int(np.argmax(ucb_values))


class ThompsonSampling(BanditBase):
    """
    Thompson Sampling
    
    从后验分布采样，选择采样值最大的臂。
    
    对于Bernoulli奖励：使用Beta分布
    对于高斯奖励：使用正态分布
    """
    
    def __init__(
        self,
        config: BanditConfig,
        reward_type: str = "bernoulli"
    ):
        super().__init__(config)
        self.reward_type = reward_type
        
        if reward_type == "bernoulli":
            # Beta分布参数
            self.alpha = np.ones(self.num_arms)
            self.beta = np.ones(self.num_arms)
        else:
            # 正态分布参数
            self.mu = np.zeros(self.num_arms)
            self.sigma_sq = np.ones(self.num_arms)
            self.prior_mu = 0.0
            self.prior_sigma_sq = 1.0
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        if self.reward_type == "bernoulli":
            # 从Beta分布采样
            samples = self.rng.beta(self.alpha, self.beta)
        else:
            # 从正态分布采样
            samples = self.rng.normal(self.mu, np.sqrt(self.sigma_sq))
        
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        
        if self.reward_type == "bernoulli":
            # 更新Beta参数
            self.alpha[arm] += reward
            self.beta[arm] += 1 - reward
        else:
            # 更新正态分布参数（共轭先验）
            n = self.counts[arm]
            prior_precision = 1 / self.prior_sigma_sq
            
            # 后验精度
            posterior_precision = prior_precision + n
            
            # 后验均值
            posterior_mu = (
                prior_precision * self.prior_mu + n * self.values[arm]
            ) / posterior_precision
            
            self.mu[arm] = posterior_mu
            self.sigma_sq[arm] = 1 / posterior_precision


class EpsilonGreedy(BanditBase):
    """
    ε-Greedy
    
    以ε概率随机探索，以1-ε概率利用当前最优。
    """
    
    def __init__(
        self,
        config: BanditConfig,
        epsilon: float = 0.1,
        decay: float = 0.999,
        min_epsilon: float = 0.01
    ):
        super().__init__(config)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        if self.rng.random() < self.epsilon:
            # 探索：随机选择
            return self.rng.randint(0, self.num_arms)
        else:
            # 利用：选择最优
            return int(np.argmax(self.values))
    
    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        
        # 衰减ε
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay
        )


class ContextualBandit(BanditBase):
    """
    上下文老虎机
    
    基于上下文选择臂，使用线性模型：
    reward_i = context · θ_i + noise
    
    使用LinUCB算法。
    """
    
    def __init__(
        self,
        config: BanditConfig,
        alpha: float = 1.0
    ):
        super().__init__(config)
        self.context_dim = config.context_dim
        self.alpha = alpha  # 探索参数
        
        # 每个臂的参数
        d = self.context_dim
        
        # A_a = I_d (d×d单位矩阵)
        self.A = [np.eye(d) for _ in range(self.num_arms)]
        
        # b_a = 0_d (d维零向量)
        self.b = [np.zeros(d) for _ in range(self.num_arms)]
        
        # θ_a = A_a^{-1} b_a
        self.theta = [np.zeros(d) for _ in range(self.num_arms)]
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        if context is None:
            raise ValueError("Context required for contextual bandit")
        
        context = np.array(context)
        
        ucb_values = np.zeros(self.num_arms)
        
        for arm in range(self.num_arms):
            # 计算θ
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            self.theta[arm] = theta
            
            # UCB
            ucb = theta @ context + self.alpha * np.sqrt(
                context @ A_inv @ context
            )
            ucb_values[arm] = ucb
        
        return int(np.argmax(ucb_values))
    
    def update(self, arm: int, reward: float, context: np.ndarray = None) -> None:
        if context is None:
            raise ValueError("Context required for contextual bandit")
        
        context = np.array(context)
        
        # 更新A和b
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # 调用父类更新
        super().update(arm, reward)


class EXP3(BanditBase):
    """
    EXP3算法
    
    适用于对抗性老虎机（非随机奖励）。
    
    使用指数权重更新：
    w_i(t+1) = w_i(t) * exp(γ * r_i(t) / p_i(t))
    """
    
    def __init__(
        self,
        config: BanditConfig,
        gamma: Optional[float] = None
    ):
        super().__init__(config)
        
        # 探索参数
        if gamma is None:
            self.gamma = min(1, np.sqrt(
                np.log(self.num_arms) / self.num_arms
            ))
        else:
            self.gamma = gamma
        
        # 权重
        self.weights = np.ones(self.num_arms)
        
        # 累积奖励估计
        self.estimated_rewards = np.zeros(self.num_arms)
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        # 计算概率分布
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * self.weights / total_weight + self.gamma / self.num_arms
        
        # 按概率选择
        return int(self.rng.choice(self.num_arms, p=probs))
    
    def update(self, arm: int, reward: float) -> None:
        # 计算选择概率
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma) * self.weights / total_weight + self.gamma / self.num_arms
        
        # 无偏奖励估计
        estimated_reward = reward / probs[arm]
        
        # 更新权重
        self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.num_arms)
        
        # 更新统计
        self.estimated_rewards[arm] += estimated_reward
        super().update(arm, reward)


class BanditEnsemble:
    """
    老虎机集成
    
    组合多种算法，动态选择最佳策略。
    """
    
    def __init__(
        self,
        config: BanditConfig,
        algorithms: List[str] = None
    ):
        self.config = config
        algorithms = algorithms or ['ucb', 'thompson', 'epsilon_greedy']
        
        # 创建算法实例
        self.algorithms: Dict[str, BanditBase] = {}
        
        for algo in algorithms:
            if algo == 'ucb':
                self.algorithms['ucb'] = UCB(config)
            elif algo == 'thompson':
                self.algorithms['thompson'] = ThompsonSampling(config)
            elif algo == 'epsilon_greedy':
                self.algorithms['epsilon_greedy'] = EpsilonGreedy(config)
            elif algo == 'exp3':
                self.algorithms['exp3'] = EXP3(config)
        
        # 元老虎机（选择哪个算法）
        self.meta_bandit = UCB(BanditConfig(num_arms=len(algorithms)))
        self.algo_names = list(self.algorithms.keys())
        
        # 历史记录
        self.history: List[Tuple[str, int, float]] = []
    
    def select_arm(self, context: Optional[np.ndarray] = None) -> Tuple[int, str]:
        """
        选择臂
        
        Returns:
            (臂索引, 使用的算法名)
        """
        # 选择算法
        algo_idx = self.meta_bandit.select_arm(context)
        algo_name = self.algo_names[algo_idx]
        
        # 使用选定算法选择臂
        arm = self.algorithms[algo_name].select_arm(context)
        
        return arm, algo_name
    
    def update(
        self,
        arm: int,
        reward: float,
        algo_name: str,
        context: Optional[np.ndarray] = None
    ) -> None:
        """更新"""
        # 更新具体算法
        self.algorithms[algo_name].update(arm, reward)
        
        # 更新元老虎机
        algo_idx = self.algo_names.index(algo_name)
        self.meta_bandit.update(algo_idx, reward)
        
        # 记录历史
        self.history.append((algo_name, arm, reward))


def compute_regret(
    rewards: List[float],
    optimal_reward: float
) -> List[float]:
    """
    计算累积遗憾
    
    Regret(t) = t * μ* - Σ_{i=1}^{t} r_i
    
    其中μ*是最优臂的期望奖励。
    """
    cumulative_rewards = np.cumsum(rewards)
    t = np.arange(1, len(rewards) + 1)
    regret = t * optimal_reward - cumulative_rewards
    return regret.tolist()
