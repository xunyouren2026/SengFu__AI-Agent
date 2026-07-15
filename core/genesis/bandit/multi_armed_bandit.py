"""
多臂老虎机综合模块 - Comprehensive Multi-Armed Bandit Algorithms

实现多种老虎机算法，包含真实数学推导，纯Python实现：
- BanditArm: 臂表示（Bernoulli, Gaussian, Beta）
- BanditEnvironment: 环境模拟（随机/上下文/非平稳）
- 探索策略: EpsilonGreedy, UCB1, UCB2, ThompsonSampling, EXP3, EXP3.S, EXP4,
            BayesUCB, GaussianTS, LinUCB, LinearThompsonSampling
- RegretAnalysis: 遗憾分析与理论界
- MultiArmedBandit: 主编排器（实验/比较/统计检验）
- AdvancedBandit: 级联/组合/层次/休眠/对抗性老虎机
- BanditOptimizer: 超参数优化（连续消除/竞速算法）
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Any, Callable, Union, Sequence
)


# ============================================================
# 1. BanditArm - 臂表示与奖励分布
# ============================================================

class BanditArm(ABC):
    """老虎机臂的抽象基类"""

    @abstractmethod
    def pull(self) -> float:
        """拉动臂，返回奖励"""
        pass

    @abstractmethod
    def expected_reward(self) -> float:
        """期望奖励"""
        pass


class BernoulliArm(BanditArm):
    """伯努利臂：奖励为0或1"""

    def __init__(self, probability: float):
        assert 0.0 <= probability <= 1.0, "概率必须在[0,1]范围内"
        self.probability = probability

    def pull(self) -> float:
        return 1.0 if random.random() < self.probability else 0.0

    def expected_reward(self) -> float:
        return self.probability


class GaussianArm(BanditArm):
    """高斯臂：奖励服从正态分布"""

    def __init__(self, mean: float, std: float = 1.0):
        self.mean = mean
        self.std = max(std, 1e-9)

    def pull(self) -> float:
        return random.gauss(self.mean, self.std)

    def expected_reward(self) -> float:
        return self.mean


class BetaArm(BanditArm):
    """Beta分布臂：奖励从Beta分布中采样"""

    def __init__(self, alpha: float, beta: float):
        assert alpha > 0 and beta > 0, "Beta参数必须为正"
        self.alpha = alpha
        self.beta = beta

    def pull(self) -> float:
        # 使用Python标准库实现Beta采样
        x = random.gammavariate(self.alpha, 1.0)
        y = random.gammavariate(self.beta, 1.0)
        return x / (x + y) if (x + y) > 0 else 0.5

    def expected_reward(self) -> float:
        return self.alpha / (self.alpha + self.beta)


# ============================================================
# 2. BanditEnvironment - 环境模拟
# ============================================================

class BanditEnvironment:
    """多臂老虎机环境"""

    def __init__(self, arms: List[BanditArm], seed: int = 42):
        self.arms = arms
        self.num_arms = len(arms)
        self.rng = random.Random(seed)
        self.optimal_arm = max(range(self.num_arms),
                               key=lambda i: arms[i].expected_reward())
        self.optimal_reward = arms[self.optimal_arm].expected_reward()

    def pull(self, arm: int) -> float:
        return self.arms[arm].pull()

    def get_regret(self, arm: int) -> float:
        return self.optimal_reward - self.arms[arm].expected_reward()

    def get_gap(self, arm: int) -> float:
        """获取臂arm与最优臂之间的差距"""
        return self.get_regret(arm)

    def summary(self) -> Dict[str, Any]:
        return {
            "num_arms": self.num_arms,
            "optimal_arm": self.optimal_arm,
            "optimal_reward": self.optimal_reward,
            "arm_means": [a.expected_reward() for a in self.arms],
        }


class ContextualBanditEnvironment:
    """上下文老虎机环境：奖励 = context . theta_arm + noise"""

    def __init__(self, num_arms: int, context_dim: int, seed: int = 42):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.rng = random.Random(seed)
        # 每个臂的真实参数向量
        self.true_theta = [
            [self.rng.gauss(0, 1) for _ in range(context_dim)]
            for _ in range(num_arms)
        ]
        self.noise_std = 0.1

    def get_context(self) -> List[float]:
        return [self.rng.gauss(0, 1) for _ in range(self.context_dim)]

    def pull(self, arm: int, context: List[float]) -> float:
        mean = sum(c * t for c, t in zip(context, self.true_theta[arm]))
        return mean + self.rng.gauss(0, self.noise_std)

    def optimal_arm(self, context: List[float]) -> int:
        values = [
            sum(c * t for c, t in zip(context, self.true_theta[i]))
            for i in range(self.num_arms)
        ]
        return max(range(self.num_arms), key=lambda i: values[i])

    def optimal_reward(self, context: List[float]) -> float:
        arm = self.optimal_arm(context)
        return sum(c * t for c, t in zip(context, self.true_theta[arm]))


class NonStationaryEnvironment:
    """非平稳奖励环境：奖励分布随时间变化"""

    def __init__(self, num_arms: int, seed: int = 42,
                 drift_rate: float = 0.01):
        self.num_arms = num_arms
        self.rng = random.Random(seed)
        self.drift_rate = drift_rate
        # 初始奖励概率
        self.probs = [self.rng.random() for _ in range(num_arms)]
        self.time = 0

    def pull(self, arm: int) -> float:
        self._drift()
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0

    def _drift(self):
        """奖励概率随机游走"""
        for i in range(self.num_arms):
            self.probs[i] += self.rng.gauss(0, self.drift_rate)
            self.probs[i] = max(0.01, min(0.99, self.probs[i]))
        self.time += 1

    def optimal_reward(self) -> float:
        return max(self.probs)


# ============================================================
# 3. 探索策略
# ============================================================

class ExplorationStrategy(ABC):
    """探索策略抽象基类"""

    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    @abstractmethod
    def select_arm(self) -> int:
        pass

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        pass


class EpsilonGreedy(ExplorationStrategy):
    """
    ε-greedy策略：以ε概率随机探索，1-ε概率利用最优臂
    支持衰减: ε_t = max(ε_min, ε_0 * decay^t)
    """

    def __init__(self, num_arms: int, epsilon: float = 0.1,
                 decay: float = 0.999, min_epsilon: float = 0.01):
        super().__init__(num_arms)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.counts = [0] * num_arms
        self.values = [0.0] * num_arms

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.num_arms - 1)
        return max(range(self.num_arms), key=lambda i: self.values[i])

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # 增量均值更新
        self.values[arm] += (reward - self.values[arm]) / n
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


class UCB1(ExplorationStrategy):
    """
    UCB1: Upper Confidence Bound 1
    选择 arm = argmax_i [μ_i + sqrt(ln(t) / n_i)]
    理论遗憾界: O(sqrt(K * T * ln(T)))
    """

    def __init__(self, num_arms: int, c: float = 1.0):
        super().__init__(num_arms)
        self.c = c
        self.counts = [0] * num_arms
        self.values = [0.0] * num_arms
        self.total_pulls = 0

    def select_arm(self) -> int:
        # 初始化：每个臂至少拉一次
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
        t = self.total_pulls
        ucb_values = []
        for i in range(self.num_arms):
            exploration = self.c * math.sqrt(math.log(t) / self.counts[i])
            ucb_values.append(self.values[i] + exploration)
        return max(range(self.num_arms), key=lambda i: ucb_values[i])

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.total_pulls += 1


class UCB2(ExplorationStrategy):
    """
    UCB2: 使用padding参数α
    每个臂有虚拟拉动次数: τ_i = ceil((1+α)^{r_i})，r_i为该臂的轮次
    选择 arm = argmax_i [μ_i + sqrt((1+α) * ln(e*T/τ_i) / (2*τ_i))]
    """

    def __init__(self, num_arms: int, alpha: float = 0.1):
        super().__init__(num_arms)
        self.alpha = alpha
        self.counts = [0] * num_arms
        self.values = [0.0] * num_arms
        self.total_pulls = 0
        self.r_i = [0] * num_arms  # 每个臂的轮次计数

    def _tau(self, r: int) -> int:
        """虚拟拉动次数"""
        return math.ceil((1 + self.alpha) ** r)

    def select_arm(self) -> int:
        # 初始化
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
        t = max(self.total_pulls, 1)
        best_arm = 0
        best_val = -float('inf')
        for i in range(self.num_arms):
            tau_i = max(self._tau(self.r_i[i]), 1)
            bonus = math.sqrt(
                (1 + self.alpha) * math.log(math.e * t / tau_i) / (2 * tau_i)
            )
            val = self.values[i] + bonus
            if val > best_val:
                best_val = val
                best_arm = i
        return best_arm

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.total_pulls += 1
        # 当虚拟拉动次数用完时，增加轮次
        if self.counts[arm] >= self._tau(self.r_i[arm] + 1):
            self.r_i[arm] += 1


class ThompsonSampling(ExplorationStrategy):
    """
    Thompson Sampling: Beta-Bernoulli后验采样
    先验: Beta(1,1)，观测到reward后:
      α += reward, β += (1 - reward)
    从Beta(α, β)采样，选择采样值最大的臂
    """

    def __init__(self, num_arms: int):
        super().__init__(num_arms)
        self.alpha = [1.0] * num_arms
        self.beta = [1.0] * num_arms

    def select_arm(self) -> int:
        samples = []
        for i in range(self.num_arms):
            x = random.gammavariate(self.alpha[i], 1.0)
            y = random.gammavariate(self.beta[i], 1.0)
            samples.append(x / (x + y) if (x + y) > 0 else 0.5)
        return max(range(self.num_arms), key=lambda i: samples[i])

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += (1.0 - reward)


class EXP3(ExplorationStrategy):
    """
    EXP3: Exponential-weight algorithm for exploration
    适用于对抗性设置
    概率: p_i(t) = (1-γ) * w_i(t) / W(t) + γ/K
    权重更新: w_i(t+1) = w_i(t) * exp(γ * x_i(t) / (K * p_i(t)))
    其中 x_i(t) 是重要性加权奖励估计
    遗憾界: O(sqrt(K*T*ln(K)))
    """

    def __init__(self, num_arms: int, gamma: Optional[float] = None):
        super().__init__(num_arms)
        if gamma is None:
            self.gamma = min(1.0, math.sqrt(math.log(num_arms) / num_arms))
        else:
            self.gamma = gamma
        self.weights = [1.0] * num_arms
        self.probs = [1.0 / num_arms] * num_arms

    def select_arm(self) -> int:
        total_w = sum(self.weights)
        self.probs = [
            (1 - self.gamma) * w / total_w + self.gamma / self.num_arms
            for w in self.weights
        ]
        # 按概率采样
        r = random.random()
        cum = 0.0
        for i, p in enumerate(self.probs):
            cum += p
            if r <= cum:
                return i
        return self.num_arms - 1

    def update(self, arm: int, reward: float) -> None:
        # 重要性加权奖励估计
        x = reward / self.probs[arm] if self.probs[arm] > 0 else 0
        # 权重更新
        self.weights[arm] *= math.exp(
            self.gamma * x / self.num_arms
        )
        # 防止数值溢出
        max_w = max(self.weights)
        if max_w > 1e100:
            self.weights = [w / max_w for w in self.weights]


class EXP3S(ExplorationStrategy):
    """
    EXP3.S: EXP3 with implicit exploration
    使用混合估计器，添加隐式探索项
    概率: p_i(t) = (1-γ) * w_i(t) / W(t) + γ/K
    估计器: x_i(t) = (I{a_t=i} * r_t / p_i(t)) + β/K
    """

    def __init__(self, num_arms: int, gamma: Optional[float] = None,
                 beta: float = 0.0):
        super().__init__(num_arms)
        if gamma is None:
            self.gamma = min(1.0, math.sqrt(math.log(num_arms) / num_arms))
        else:
            self.gamma = gamma
        self.beta = beta
        self.weights = [1.0] * num_arms
        self.probs = [1.0 / num_arms] * num_arms

    def select_arm(self) -> int:
        total_w = sum(self.weights)
        self.probs = [
            (1 - self.gamma) * w / total_w + self.gamma / self.num_arms
            for w in self.weights
        ]
        r = random.random()
        cum = 0.0
        for i, p in enumerate(self.probs):
            cum += p
            if r <= cum:
                return i
        return self.num_arms - 1

    def update(self, arm: int, reward: float) -> None:
        # 混合估计器：重要性加权 + 隐式探索
        x = reward / self.probs[arm] if self.probs[arm] > 0 else 0
        for i in range(self.num_arms):
            estimate = (x if i == arm else 0) + self.beta / self.num_arms
            self.weights[i] *= math.exp(
                self.gamma * estimate / self.num_arms
            )
        max_w = max(self.weights)
        if max_w > 1e100:
            self.weights = [w / max_w for w in self.weights]


class EXP4(ExplorationStrategy):
    """
    EXP4: Mixing expert advice
    组合多个专家的建议，适用于对抗性设置
    概率: p_i(t) = (1-γ) * Σ_j q_j(t) * ξ_{j,i}(t) + γ/K
    其中 q_j 是专家权重, ξ_{j,i} 是专家j对臂i的建议
    """

    def __init__(self, num_arms: int, num_experts: int,
                 gamma: Optional[float] = None):
        super().__init__(num_arms)
        self.num_experts = num_experts
        if gamma is None:
            self.gamma = min(1.0, math.sqrt(
                math.log(num_experts) * num_arms / num_experts
            ))
        else:
            self.gamma = gamma
        self.expert_weights = [1.0] * num_experts

    def select_arm(self, expert_advice: List[List[float]]) -> int:
        """
        expert_advice[j][i] = 专家j对臂i的概率建议
        """
        # 计算混合概率
        total_w = sum(self.expert_weights)
        mixed = [0.0] * self.num_arms
        for i in range(self.num_arms):
            for j in range(self.num_experts):
                mixed[i] += (self.expert_weights[j] / total_w) * expert_advice[j][i]
            mixed[i] = (1 - self.gamma) * mixed[i] + self.gamma / self.num_arms

        # 归一化
        total = sum(mixed)
        if total > 0:
            mixed = [p / total for p in mixed]

        self._current_probs = mixed
        r = random.random()
        cum = 0.0
        for i, p in enumerate(mixed):
            cum += p
            if r <= cum:
                return i
        return self.num_arms - 1

    def update(self, arm: int, reward: float,
               expert_advice: List[List[float]]) -> None:
        probs = getattr(self, '_current_probs', [1.0 / self.num_arms] * self.num_arms)
        importance_weight = reward / probs[arm] if probs[arm] > 0 else 0
        for j in range(self.num_experts):
            self.expert_weights[j] *= math.exp(
                self.gamma * importance_weight * expert_advice[j][arm] / self.num_arms
            )
        max_w = max(self.expert_weights)
        if max_w > 1e100:
            self.expert_weights = [w / max_w for w in self.expert_weights]


class BayesUCB(ExplorationStrategy):
    """
    BayesUCB: 贝叶斯UCB，使用Gaussian后验
    先验: N(0, 1)，观测后:
      μ_post = (n * x̄) / (n + 1)
      σ²_post = 1 / (n + 1)
    选择 arm = argmax_i [μ_post_i + q_{1-1/t} * σ_post_i]
    其中 q_{1-1/t} 是标准正态的 (1-1/t) 分位数
    """

    def __init__(self, num_arms: int):
        super().__init__(num_arms)
        self.counts = [0] * num_arms
        self.sum_rewards = [0.0] * num_arms
        self.total_pulls = 0

    def _normal_quantile(self, p: float) -> float:
        """近似标准正态分布分位数（Rational近似）"""
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0
        if p < 0.5:
            return -self._normal_quantile(1 - p)
        # Abramowitz and Stegun近似
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    def select_arm(self) -> int:
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
        t = self.total_pulls
        q = self._normal_quantile(1.0 - 1.0 / t)
        best_arm = 0
        best_val = -float('inf')
        for i in range(self.num_arms):
            n = self.counts[i]
            mu_post = self.sum_rewards[i] / (n + 1)
            sigma_post = 1.0 / math.sqrt(n + 1)
            val = mu_post + q * sigma_post
            if val > best_val:
                best_val = val
                best_arm = i
        return best_arm

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_pulls += 1


class GaussianTS(ExplorationStrategy):
    """
    Gaussian Thompson Sampling: 高斯奖励的Thompson采样
    使用共轭先验: μ ~ N(μ_0, σ²_0), σ² ~ InvGamma(α, β)
    简化: 固定方差，仅更新均值
    后验: μ_post = (μ_0/σ²_0 + n*x̄/σ²) / (1/σ²_0 + n/σ²)
    """

    def __init__(self, num_arms: int, prior_mu: float = 0.0,
                 prior_sigma_sq: float = 1.0, reward_var: float = 1.0):
        super().__init__(num_arms)
        self.prior_mu = prior_mu
        self.prior_sigma_sq = prior_sigma_sq
        self.reward_var = reward_var
        self.counts = [0] * num_arms
        self.sum_rewards = [0.0] * num_arms
        self.post_mu = [prior_mu] * num_arms
        self.post_var = [prior_sigma_sq] * num_arms

    def select_arm(self) -> int:
        samples = [
            random.gauss(self.post_mu[i], math.sqrt(self.post_var[i]))
            for i in range(self.num_arms)
        ]
        return max(range(self.num_arms), key=lambda i: samples[i])

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        n = self.counts[arm]
        x_bar = self.sum_rewards[arm] / n
        # 后验更新（共轭高斯）
        prior_prec = 1.0 / self.prior_sigma_sq
        lik_prec = n / self.reward_var
        post_prec = prior_prec + lik_prec
        self.post_mu[arm] = (prior_prec * self.prior_mu + lik_prec * x_bar) / post_prec
        self.post_var[arm] = 1.0 / post_prec


class LinUCB(ExplorationStrategy):
    """
    Linear Contextual Bandit (LinUCB)
    奖励模型: r_a = x^T θ_a + noise
    UCB: x^T θ̂_a + α * sqrt(x^T A_a^{-1} x)
    其中 A_a = d*I + Σ x*x^T, b_a = Σ r*x
    θ̂_a = A_a^{-1} b_a
    """

    def __init__(self, num_arms: int, context_dim: int, alpha: float = 1.0):
        super().__init__(num_arms)
        self.context_dim = context_dim
        self.alpha = alpha
        # 每个臂的 A_a (d×d) 和 b_a (d)
        d = context_dim
        self.A = [[(1.0 if i == j else 0.0) for j in range(d)] for i in range(d)]
        self.A_arms = [self._copy_matrix(self.A) for _ in range(num_arms)]
        self.b_arms = [[0.0] * d for _ in range(num_arms)]
        self.theta_arms = [[0.0] * d for _ in range(num_arms)]

    def _copy_matrix(self, m):
        return [row[:] for row in m]

    def _mat_vec_mul(self, m, v):
        return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]

    def _vec_dot(self, u, v):
        return sum(a * b for a, b in zip(u, v))

    def _outer(self, u, v):
        return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]

    def _mat_add(self, A, B):
        d = len(A)
        return [[A[i][j] + B[i][j] for j in range(d)] for i in range(d)]

    def _mat_inv(self, m):
        """高斯消元法求逆"""
        n = len(m)
        # 增广矩阵
        aug = [m[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            # 选主元
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                continue
            for j in range(2 * n):
                aug[col][j] /= pivot
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        return [row[n:] for row in aug]

    def select_arm(self, context: List[float] = None) -> int:
        if context is None:
            return 0
        best_arm = 0
        best_val = -float('inf')
        for a in range(self.num_arms):
            A_inv = self._mat_inv(self.A_arms[a])
            theta = self._mat_vec_mul(A_inv, self.b_arms[a])
            self.theta_arms[a] = theta
            pred = self._vec_dot(theta, context)
            # sqrt(x^T A^{-1} x)
            Ax = self._mat_vec_mul(A_inv, context)
            uncertainty = self.alpha * math.sqrt(max(0, self._vec_dot(context, Ax)))
            val = pred + uncertainty
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: List[float] = None) -> None:
        if context is None:
            return
        outer = self._outer(context, context)
        self.A_arms[arm] = self._mat_add(self.A_arms[arm], outer)
        for i in range(self.context_dim):
            self.b_arms[arm][i] += reward * context[i]


class LinearThompsonSampling(ExplorationStrategy):
    """
    Linear Thompson Sampling
    先验: θ_a ~ N(0, λ^{-1} I)
    后验: θ_a | data ~ N(μ_a, V_a)
    V_a^{-1} = λ*I + Σ x*x^T
    μ_a = V_a * Σ r*x
    """

    def __init__(self, num_arms: int, context_dim: int, lam: float = 1.0):
        super().__init__(num_arms)
        self.context_dim = context_dim
        self.lam = lam
        d = context_dim
        self.V_inv = [
            [[(lam if i == j else 0.0) for j in range(d)] for i in range(d)]
            for _ in range(num_arms)
        ]
        self.b_arms = [[0.0] * d for _ in range(num_arms)]
        self.mu_arms = [[0.0] * d for _ in range(num_arms)]

    def _mat_vec_mul(self, m, v):
        return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]

    def _vec_dot(self, u, v):
        return sum(a * b for a, b in zip(u, v))

    def _outer(self, u, v):
        return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]

    def _mat_add(self, A, B):
        d = len(A)
        return [[A[i][j] + B[i][j] for j in range(d)] for i in range(d)]

    def _mat_inv(self, m):
        n = len(m)
        aug = [m[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                continue
            for j in range(2 * n):
                aug[col][j] /= pivot
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        return [row[n:] for row in aug]

    def _sample_multivariate_normal(self, mu, cov):
        """从多元正态分布采样（Cholesky分解）"""
        d = len(mu)
        # Cholesky分解
        L = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = cov[i][i] - s
                    L[i][j] = math.sqrt(max(val, 1e-12))
                else:
                    L[i][j] = (cov[i][j] - s) / L[j][j] if L[j][j] > 1e-12 else 0.0
        # L * z
        z = [random.gauss(0, 1) for _ in range(d)]
        sample = [sum(L[i][j] * z[j] for j in range(d)) for i in range(d)]
        return [sample[i] + mu[i] for i in range(d)]

    def select_arm(self, context: List[float] = None) -> int:
        if context is None:
            return 0
        best_arm = 0
        best_val = -float('inf')
        for a in range(self.num_arms):
            V = self._mat_inv(self.V_inv[a])
            mu = self._mat_vec_mul(V, self.b_arms[a])
            self.mu_arms[a] = mu
            theta_sample = self._sample_multivariate_normal(mu, V)
            val = self._vec_dot(theta_sample, context)
            if val > best_val:
                best_val = val
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: List[float] = None) -> None:
        if context is None:
            return
        outer = self._outer(context, context)
        self.V_inv[arm] = self._mat_add(self.V_inv[arm], outer)
        for i in range(self.context_dim):
            self.b_arms[arm][i] += reward * context[i]


# ============================================================
# 4. RegretAnalysis - 遗憾分析
# ============================================================

class RegretAnalysis:
    """
    遗憾分析与理论界计算
    累积遗憾: R(T) = Σ_{t=1}^{T} (μ* - μ_{a_t})
    """

    def __init__(self, env: BanditEnvironment):
        self.env = env
        self.rewards_history: List[float] = []
        self.arm_history: List[int] = []
        self.optimal_reward = env.optimal_reward

    def record(self, arm: int, reward: float) -> None:
        self.arm_history.append(arm)
        self.rewards_history.append(reward)

    def cumulative_regret(self) -> List[float]:
        """计算每步的累积遗憾"""
        regret = []
        cum = 0.0
        for arm in self.arm_history:
            cum += self.env.get_regret(arm)
            regret.append(cum)
        return regret

    def instantaneous_regret(self) -> List[float]:
        return [self.env.get_regret(arm) for arm in self.arm_history]

    def gap_based_bound(self, T: int) -> float:
        """
        基于gap的遗憾上界
        Lai-Robbins下界: R(T) >= (Σ_i Δ_i / KL(ν_i, ν*)) * ln(T)
        其中 Δ_i = μ* - μ_i, KL为KL散度
        """
        gaps = []
        for i in range(self.env.num_arms):
            if i != self.env.optimal_arm:
                gap = self.env.get_gap(i)
                if gap > 1e-10:
                    # Bernoulli KL散度近似
                    p_opt = self.env.optimal_reward
                    p_i = self.env.arms[i].expected_reward()
                    kl = self._kl_bernoulli(p_i, p_opt)
                    if kl > 1e-10:
                        gaps.append(gap / kl)
        if not gaps:
            return 0.0
        bound = sum(gaps) * math.log(max(T, 1))
        return bound

    @staticmethod
    def _kl_bernoulli(p: float, q: float) -> float:
        """计算两个伯努利分布之间的KL散度"""
        p = max(1e-10, min(1 - 1e-10, p))
        q = max(1e-10, min(1 - 1e-10, q))
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def best_arm_identification(self, confidence: float = 0.95) -> Tuple[int, bool]:
        """
        最优臂识别：基于经验均值和Hoeffding界
        返回 (识别的最优臂, 是否有足够置信度)
        """
        arm_counts: Dict[int, int] = {}
        arm_sums: Dict[int, float] = {}
        for arm, reward in zip(self.arm_history, self.rewards_history):
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
            arm_sums[arm] = arm_sums.get(arm, 0.0) + reward

        if not arm_counts:
            return 0, False

        T = len(self.arm_history)
        # Hoeffding置信区间半径
        delta = 1 - confidence
        epsilon = math.sqrt(math.log(1 / delta) / (2 * T)) if T > 0 else 1.0

        best_arm = max(arm_counts.keys(),
                       key=lambda a: arm_sums[a] / arm_counts[a])
        best_mean = arm_sums[best_arm] / arm_counts[best_arm]

        # 检查是否所有其他臂的置信上界低于最优臂的置信下界
        all_lower = True
        for arm in arm_counts:
            if arm == best_arm:
                continue
            mean = arm_sums[arm] / arm_counts[arm]
            if mean + epsilon >= best_mean - epsilon:
                all_lower = False
                break

        return best_arm, all_lower

    def summary(self) -> Dict[str, Any]:
        if not self.rewards_history:
            return {"total_regret": 0, "avg_reward": 0, "num_pulls": 0}
        cum_regret = self.cumulative_regret()
        return {
            "total_regret": cum_regret[-1] if cum_regret else 0,
            "avg_reward": sum(self.rewards_history) / len(self.rewards_history),
            "num_pulls": len(self.rewards_history),
            "final_regret_rate": cum_regret[-1] / len(self.rewards_history) if cum_regret else 0,
        }


# ============================================================
# 5. BanditConfig - 配置
# ============================================================

@dataclass
class BanditConfig:
    """老虎机实验配置"""
    num_arms: int = 10
    context_dim: int = 5
    horizon: int = 1000
    seed: int = 42
    algorithm: str = "ucb1"
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    num_trials: int = 5
    non_stationary: bool = False
    drift_rate: float = 0.01


# ============================================================
# 6. MultiArmedBandit - 主编排器
# ============================================================

class MultiArmedBandit:
    """
    多臂老虎机主编排器
    支持运行实验、比较算法、生成遗憾曲线数据、统计显著性检验
    """

    ALGORITHMS = {
        "epsilon_greedy": EpsilonGreedy,
        "ucb1": UCB1,
        "ucb2": UCB2,
        "thompson": ThompsonSampling,
        "exp3": EXP3,
        "exp3s": EXP3S,
        "bayes_ucb": BayesUCB,
        "gaussian_ts": GaussianTS,
    }

    def __init__(self, config: BanditConfig):
        self.config = config
        random.seed(config.seed)

    def _create_arms(self) -> List[BanditArm]:
        """创建随机臂"""
        arms = []
        for _ in range(self.config.num_arms):
            p = random.uniform(0.1, 0.9)
            arms.append(BernoulliArm(p))
        return arms

    def _create_strategy(self, name: str, num_arms: int,
                         params: Dict[str, Any]) -> ExplorationStrategy:
        """创建探索策略"""
        if name == "epsilon_greedy":
            return EpsilonGreedy(
                num_arms,
                epsilon=params.get("epsilon", 0.1),
                decay=params.get("decay", 0.999),
                min_epsilon=params.get("min_epsilon", 0.01),
            )
        elif name == "ucb1":
            return UCB1(num_arms, c=params.get("c", 1.0))
        elif name == "ucb2":
            return UCB2(num_arms, alpha=params.get("alpha", 0.1))
        elif name == "thompson":
            return ThompsonSampling(num_arms)
        elif name == "exp3":
            return EXP3(num_arms, gamma=params.get("gamma", None))
        elif name == "exp3s":
            return EXP3S(num_arms, gamma=params.get("gamma", None),
                         beta=params.get("beta", 0.0))
        elif name == "bayes_ucb":
            return BayesUCB(num_arms)
        elif name == "gaussian_ts":
            return GaussianTS(num_arms,
                              prior_mu=params.get("prior_mu", 0.0),
                              prior_sigma_sq=params.get("prior_sigma_sq", 1.0))
        else:
            raise ValueError(f"未知算法: {name}")

    def run_experiment(self, algorithm: str = None,
                       arms: List[BanditArm] = None) -> Dict[str, Any]:
        """运行单次实验"""
        algo_name = algorithm or self.config.algorithm
        arms = arms or self._create_arms()
        env = BanditEnvironment(arms, seed=self.config.seed)
        strategy = self._create_strategy(
            algo_name, env.num_arms, self.config.algorithm_params
        )
        analysis = RegretAnalysis(env)

        rewards = []
        regrets = []
        for t in range(self.config.horizon):
            arm = strategy.select_arm()
            reward = env.pull(arm)
            strategy.update(arm, reward)
            analysis.record(arm, reward)
            rewards.append(reward)
            regrets.append(env.get_regret(arm))

        cum_regret = analysis.cumulative_regret()
        return {
            "algorithm": algo_name,
            "rewards": rewards,
            "cumulative_regret": cum_regret,
            "total_regret": cum_regret[-1] if cum_regret else 0,
            "avg_reward": sum(rewards) / len(rewards),
            "arm_pulls": analysis.summary()["num_pulls"],
            "optimal_arm": env.optimal_arm,
            "optimal_reward": env.optimal_reward,
        }

    def compare_algorithms(
        self, algorithms: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """比较多个算法"""
        algorithms = algorithms or list(self.ALGORITHMS.keys())
        arms = self._create_arms()
        results = {}
        for algo in algorithms:
            trial_results = []
            for trial in range(self.config.num_trials):
                random.seed(self.config.seed + trial)
                result = self.run_experiment(algo, arms=arms)
                trial_results.append(result)
            # 聚合
            avg_regret = [
                sum(tr["cumulative_regret"][t] for tr in trial_results)
                / self.config.num_trials
                for t in range(self.config.horizon)
            ]
            avg_reward = sum(tr["avg_reward"] for tr in trial_results) / self.config.num_trials
            results[algo] = {
                "avg_cumulative_regret": avg_regret,
                "avg_reward": avg_reward,
                "avg_total_regret": avg_regret[-1] if avg_regret else 0,
                "trials": trial_results,
            }
        return results

    def regret_curve_data(self, algorithms: List[str] = None) -> Dict[str, List[float]]:
        """生成遗憾曲线数据（用于绘图）"""
        comparison = self.compare_algorithms(algorithms)
        return {
            algo: data["avg_cumulative_regret"]
            for algo, data in comparison.items()
        }

    def statistical_test(self, results: Dict[str, List[float]],
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        统计显著性检验（配对t检验近似）
        比较两种算法的最终累积遗憾
        """
        algo_names = list(results.keys())
        if len(algo_names) != 2:
            return {"error": "需要恰好两个算法进行比较"}

        a_data = results[algo_names[0]]
        b_data = results[algo_names[1]]

        n = min(len(a_data), len(b_data))
        diffs = [a_data[i] - b_data[i] for i in range(n)]

        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0
        se = math.sqrt(var_d / n) if n > 0 else 0

        # t统计量
        t_stat = mean_d / se if se > 0 else 0

        # 简化p值判断（t分布近似）
        significant = abs(t_stat) > 2.0  # 粗略的95%置信度

        return {
            "algorithm_a": algo_names[0],
            "algorithm_b": algo_names[1],
            "mean_difference": mean_d,
            "std_error": se,
            "t_statistic": t_stat,
            "significant_at_alpha": significant,
            "alpha": alpha,
        }


# ============================================================
# 7. AdvancedBandit - 高级老虎机
# ============================================================

class CascadingBandit:
    """
    级联老虎机 (Cascading Bandit)
    用户从上到下浏览排序列表，点击第一个感兴趣的项
    基于位置模型: P(click at position k) = θ_k * P(relevant)
    """

    def __init__(self, num_items: int, list_size: int,
                 position_probs: List[float] = None):
        self.num_items = num_items
        self.list_size = list_size
        # 位置注意力概率
        self.position_probs = position_probs or [
            1.0 / (k + 1) for k in range(list_size)
        ]
        # 每个物品的相关性（真实参数）
        self.relevances = [random.uniform(0.1, 0.9) for _ in range(num_items)]
        # UCB统计
        self.counts = [0] * num_items
        self.values = [0.0] * num_items
        self.total_pulls = 0

    def select_list(self) -> List[int]:
        """选择要展示的物品列表"""
        # UCB1评分
        scores = []
        for i in range(self.num_items):
            if self.counts[i] == 0:
                scores.append(float('inf'))
            else:
                ucb = self.values[i] + math.sqrt(
                    2 * math.log(max(self.total_pulls, 1)) / self.counts[i]
                )
                scores.append(ucb)
        # 选择top-k
        indices = sorted(range(self.num_items), key=lambda i: -scores[i])
        return indices[:self.list_size]

    def simulate_click(self, item_list: List[int]) -> Tuple[int, float]:
        """模拟用户点击，返回(点击位置, 奖励)"""
        for k, item in enumerate(item_list):
            if random.random() < self.position_probs[k] * self.relevances[item]:
                return k, 1.0
        return -1, 0.0

    def update(self, item_list: List[int], click_pos: int, reward: float) -> None:
        """更新统计"""
        self.total_pulls += 1
        if click_pos >= 0:
            clicked_item = item_list[click_pos]
            self.counts[clicked_item] += 1
            self.values[clicked_item] += (
                (reward - self.values[clicked_item]) / self.counts[clicked_item]
            )
        # 未点击的物品也获得负反馈
        for k, item in enumerate(item_list):
            if k <= click_pos or click_pos < 0:
                self.counts[item] += 1
                self.values[item] += (
                    (0.0 - self.values[item]) / self.counts[item]
                )


class CombinatorialBandit:
    """
    组合老虎机 (Combinatorial Bandit)
    每次选择一个臂的子集 S ⊆ {1,...,K}, |S| = m
    奖励为子集中各臂奖励之和
    使用CUCB算法: 对每个臂独立使用UCB，选择top-m
    """

    def __init__(self, num_arms: int, subset_size: int):
        self.num_arms = num_arms
        self.subset_size = min(subset_size, num_arms)
        # 真实奖励
        self.true_means = [random.uniform(0.1, 0.9) for _ in range(num_arms)]
        # UCB统计
        self.counts = [0] * num_arms
        self.values = [0.0] * num_arms
        self.total_pulls = 0

    def select_subset(self) -> List[int]:
        """选择子集"""
        scores = []
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                scores.append(float('inf'))
            else:
                ucb = self.values[i] + math.sqrt(
                    2 * math.log(max(self.total_pulls, 1)) / self.counts[i]
                )
                scores.append(ucb)
        indices = sorted(range(self.num_arms), key=lambda i: -scores[i])
        return indices[:self.subset_size]

    def pull(self, subset: List[int]) -> float:
        """拉动子集，返回总奖励"""
        total = 0.0
        for arm in subset:
            r = 1.0 if random.random() < self.true_means[arm] else 0.0
            total += r
            self.counts[arm] += 1
            self.values[arm] += (r - self.values[arm]) / self.counts[arm]
        self.total_pulls += 1
        return total

    def optimal_reward(self) -> float:
        """最优子集的理论奖励"""
        top = sorted(range(self.num_arms),
                     key=lambda i: -self.true_means[i])[:self.subset_size]
        return sum(self.true_means[i] for i in top)


class HierarchicalBandit:
    """
    层次老虎机 (Hierarchical Bandit)
    嵌套结构：先选择类别，再选择类别内的臂
    两层UCB策略
    """

    def __init__(self, num_categories: int, arms_per_category: int):
        self.num_categories = num_categories
        self.arms_per_category = arms_per_category
        # 类别级UCB
        self.cat_counts = [0] * num_categories
        self.cat_values = [0.0] * num_categories
        # 臂级UCB
        self.arm_counts = [
            [0] * arms_per_category for _ in range(num_categories)
        ]
        self.arm_values = [
            [0.0] * arms_per_category for _ in range(num_categories)
        ]
        # 真实参数
        self.cat_means = [random.uniform(0.3, 0.7) for _ in range(num_categories)]
        self.arm_means = [
            [random.uniform(0.1, 0.9) for _ in range(arms_per_category)]
            for _ in range(num_categories)
        ]
        self.total_pulls = 0

    def select_arm(self) -> Tuple[int, int]:
        """返回 (类别, 臂)"""
        # 类别选择
        cat = 0
        if self.total_pulls == 0:
            cat = 0
        else:
            best_val = -float('inf')
            for c in range(self.num_categories):
                if self.cat_counts[c] == 0:
                    cat = c
                    break
                ucb = self.cat_values[c] + math.sqrt(
                    2 * math.log(self.total_pulls) / self.cat_counts[c]
                )
                if ucb > best_val:
                    best_val = ucb
                    cat = c

        # 臂选择
        arm = 0
        best_val = -float('inf')
        for a in range(self.arms_per_category):
            if self.arm_counts[cat][a] == 0:
                arm = a
                break
            n = self.arm_counts[cat][a]
            ucb = self.arm_values[cat][a] + math.sqrt(
                2 * math.log(max(self.cat_counts[cat], 1)) / n
            )
            if ucb > best_val:
                best_val = ucb
                arm = a

        return cat, arm

    def pull(self, cat: int, arm: int) -> float:
        """拉动臂，奖励 = 类别均值 * 臂均值 + noise"""
        base = self.cat_means[cat] * self.arm_means[cat][arm]
        reward = 1.0 if random.random() < base else 0.0

        # 更新
        self.arm_counts[cat][arm] += 1
        n = self.arm_counts[cat][arm]
        self.arm_values[cat][arm] += (reward - self.arm_values[cat][arm]) / n

        self.cat_counts[cat] += 1
        nc = self.cat_counts[cat]
        self.cat_values[cat] += (reward - self.cat_values[cat]) / nc

        self.total_pulls += 1
        return reward


class SleepingBandit:
    """
    休眠老虎机 (Sleeping Bandit)
    某些臂在某些时刻不可用
    使用Modified UCB: 仅对可用臂计算UCB
    """

    def __init__(self, num_arms: int, availability_prob: float = 0.7):
        self.num_arms = num_arms
        self.availability_prob = availability_prob
        self.true_means = [random.uniform(0.1, 0.9) for _ in range(num_arms)]
        self.counts = [0] * num_arms
        self.values = [0.0] * num_arms
        self.total_pulls = 0

    def get_available_arms(self) -> List[int]:
        """获取当前可用的臂"""
        return [
            i for i in range(self.num_arms)
            if random.random() < self.availability_prob
        ]

    def select_arm(self, available: List[int]) -> int:
        """从可用臂中选择"""
        if not available:
            return random.randint(0, self.num_arms - 1)

        best_arm = available[0]
        best_val = -float('inf')
        for arm in available:
            if self.counts[arm] == 0:
                return arm
            ucb = self.values[arm] + math.sqrt(
                2 * math.log(max(self.total_pulls, 1)) / self.counts[arm]
            )
            if ucb > best_val:
                best_val = ucb
                best_arm = arm
        return best_arm

    def pull(self, arm: int) -> float:
        reward = 1.0 if random.random() < self.true_means[arm] else 0.0
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.total_pulls += 1
        return reward


class AdversarialBandit:
    """
    对抗性老虎机 (Adversarial Bandit)
    使用EXP3算法处理最坏情况
    遗憾界: O(sqrt(K*T*ln(K)))
    """

    def __init__(self, num_arms: int, gamma: Optional[float] = None):
        self.num_arms = num_arms
        if gamma is None:
            self.gamma = min(1.0, math.sqrt(math.log(num_arms) / num_arms))
        else:
            self.gamma = gamma
        self.weights = [1.0] * num_arms
        self.cumulative_regret = 0.0
        self.total_pulls = 0
        # 对抗性奖励生成器（使用固定种子保证可复现）
        self.adversary_rng = random.Random(123)

    def select_arm(self) -> int:
        total_w = sum(self.weights)
        probs = [
            (1 - self.gamma) * w / total_w + self.gamma / self.num_arms
            for w in self.weights
        ]
        self._probs = probs
        r = random.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return self.num_arms - 1

    def adversarial_reward(self, arm: int) -> float:
        """对抗性奖励：对手试图最小化玩家收益"""
        # 对手选择给非选中臂高奖励
        rewards = [self.adversary_rng.random() for _ in range(self.num_arms)]
        rewards[arm] *= 0.1  # 压低选中臂的奖励
        return rewards[arm]

    def update(self, arm: int, reward: float) -> None:
        probs = getattr(self, '_probs',
                        [1.0 / self.num_arms] * self.num_arms)
        x = reward / probs[arm] if probs[arm] > 0 else 0
        self.weights[arm] *= math.exp(self.gamma * x / self.num_arms)
        max_w = max(self.weights)
        if max_w > 1e100:
            self.weights = [w / max_w for w in self.weights]
        # 假设最优固定臂的奖励
        best_expected = 0.5  # 对抗性设定下的保守估计
        self.cumulative_regret += best_expected - reward
        self.total_pulls += 1

    def regret_bound(self, T: int) -> float:
        """EXP3理论遗憾上界"""
        K = self.num_arms
        return 2.0 * math.sqrt(K * T * math.log(K))


# ============================================================
# 8. BanditOptimizer - 超参数优化
# ============================================================

class BanditOptimizer:
    """
    基于老虎机的超参数优化
    将超参数搜索建模为多臂老虎机问题
    """

    def __init__(self, param_space: Dict[str, List[Any]],
                 seed: int = 42):
        """
        param_space: {参数名: [候选值列表]}
        """
        self.param_space = param_space
        self.seed = seed
        random.seed(seed)
        # 展开所有候选配置
        self.configurations = self._expand_configs()
        self.num_configs = len(self.configurations)
        # UCB统计
        self.counts = [0] * self.num_configs
        self.values = [0.0] * self.num_configs
        self.total_evals = 0
        self.best_config = None
        self.best_score = -float('inf')
        self.history: List[Dict[str, Any]] = []

    def _expand_configs(self) -> List[Dict[str, Any]]:
        """展开参数空间为所有组合"""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        configs = []
        for combo in self._cartesian_product(values):
            configs.append(dict(zip(keys, combo)))
        return configs

    @staticmethod
    def _cartesian_product(lst: List[List[Any]]) -> List[Tuple]:
        """计算笛卡尔积"""
        if not lst:
            return [()]
        result = [[]]
        for pool in lst:
            new_result = []
            for x in result:
                for y in pool:
                    new_result.append(x + [y])
            result = new_result
        return [tuple(r) for r in result]

    def suggest(self) -> Dict[str, Any]:
        """建议下一个要评估的配置（UCB策略）"""
        # 初始化：每个配置至少评估一次
        for i in range(self.num_configs):
            if self.counts[i] == 0:
                return self.configurations[i]

        # UCB选择
        best_idx = 0
        best_val = -float('inf')
        for i in range(self.num_configs):
            ucb = self.values[i] + math.sqrt(
                2 * math.log(max(self.total_evals, 1)) / self.counts[i]
            )
            if ucb > best_val:
                best_val = ucb
                best_idx = i
        return self.configurations[best_idx]

    def observe(self, config: Dict[str, Any], score: float) -> None:
        """记录配置的评估结果"""
        idx = self.configurations.index(config)
        self.counts[idx] += 1
        n = self.counts[idx]
        self.values[idx] += (score - self.values[idx]) / n
        self.total_evals += 1

        if score > self.best_score:
            self.best_score = score
            self.best_config = config

        self.history.append({
            "config": config,
            "score": score,
            "total_evals": self.total_evals,
        })

    def successive_elimination(self, objective: Callable,
                               n_rounds: int = 10) -> Dict[str, Any]:
        """
        连续消除算法 (Successive Elimination)
        每轮评估所有存活配置，消除表现差的
        """
        alive = list(range(self.num_configs))
        evals_per_round = max(1, n_rounds // 5)

        for round_idx in range(n_rounds):
            if len(alive) <= 1:
                break

            # 评估每个存活配置
            round_scores = {}
            for idx in alive:
                scores = []
                for _ in range(evals_per_round):
                    score = objective(self.configurations[idx])
                    scores.append(score)
                    self.observe(self.configurations[idx], score)
                round_scores[idx] = sum(scores) / len(scores)

            # 消除：保留均值 + 置信区间最大的
            threshold = max(round_scores.values())
            epsilon = math.sqrt(math.log(n_rounds) / (2 * evals_per_round))
            alive = [
                idx for idx in alive
                if round_scores[idx] >= threshold - 2 * epsilon
            ]

        if alive:
            best_idx = max(alive, key=lambda i: self.values[i])
            return self.configurations[best_idx]
        return self.best_config or self.configurations[0]

    def racing(self, objective: Callable,
               max_evals: int = 100,
               confidence: float = 0.95) -> Dict[str, Any]:
        """
        竞速算法 (Racing Algorithm)
        逐步增加评估次数，逐步淘汰劣质配置
        使用Hoeffding界进行统计检验
        """
        alive = list(range(self.num_configs))
        eval_count = 0
        round_size = max(1, self.num_configs)

        while len(alive) > 1 and eval_count < max_evals:
            # 每轮评估
            round_scores: Dict[int, List[float]] = {i: [] for i in alive}
            for idx in alive:
                for _ in range(round_size):
                    if eval_count >= max_evals:
                        break
                    score = objective(self.configurations[idx])
                    round_scores[idx].append(score)
                    self.observe(self.configurations[idx], score)
                    eval_count += 1

            # 统计检验
            means = {i: sum(s) / len(s) for i, s in round_scores.items() if s}
            best_mean = max(means.values()) if means else 0

            # Hoeffding界
            n = round_size
            delta = 1 - confidence
            epsilon = math.sqrt(math.log(len(alive) / delta) / (2 * n)) if n > 0 else 1.0

            alive = [
                idx for idx in alive
                if idx in means and means[idx] >= best_mean - epsilon
            ]

            round_size = max(1, round_size // 2)

        if alive:
            best_idx = max(alive, key=lambda i: self.values[i])
            return self.configurations[best_idx]
        return self.best_config or self.configurations[0]

    def best(self) -> Tuple[Dict[str, Any], float]:
        """返回最佳配置及其得分"""
        if self.best_config is None and self.num_configs > 0:
            self.best_config = self.configurations[0]
            self.best_score = self.values[0]
        return self.best_config or {}, self.best_score
