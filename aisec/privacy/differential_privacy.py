"""
Differential Privacy Module - 差分隐私模块

提供完整的差分隐私实现，包括：
- 高斯机制 (Gaussian Mechanism)
- 拉普拉斯机制 (Laplace Mechanism)
- 指数机制 (Exponential Mechanism)
- RDP 会计 (Renyi Differential Privacy Accountant)
- 隐私预算跟踪 (Privacy Budget Tracking)
- DP-SGD 梯度裁剪 (Gradient Clipping)
- 组合定理 (Composition Theorems)

All implementations use pure Python standard library with complete type annotations.
"""

from __future__ import annotations

import math
import random
import hashlib
import time
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    List, Dict, Optional, Tuple, Callable, Any, Union,
    Sequence, Set, NamedTuple
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPConfig:
    """差分隐私全局配置"""
    epsilon: float = 1.0
    delta: float = 1e-5
    mechanism: str = "gaussian"  # "gaussian" | "laplace" | "exponential"
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1
    max_gradient_norm: float = 1.0
    target_delta: float = 1e-5
    orders: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)

    def validate(self) -> None:
        """验证配置参数合法性"""
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if not (0 <= self.delta < 1):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {self.clip_norm}")
        if self.noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be positive, got {self.noise_multiplier}")
        if self.max_gradient_norm <= 0:
            raise ValueError(f"max_gradient_norm must be positive, got {self.max_gradient_norm}")


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

class MechanismType(Enum):
    """差分隐私机制类型"""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EXPONENTIAL = "exponential"


class GaussianMechanism:
    """高斯机制: 添加 N(0, sigma^2) 噪声

    对于 (epsilon, delta)-差分隐私:
        sigma >= clip_norm * sqrt(2 * ln(1.25 / delta)) / epsilon
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")
        self.epsilon: float = epsilon
        self.delta: float = delta
        self.sensitivity: float = sensitivity
        self.sigma: float = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """根据 epsilon, delta, sensitivity 计算标准差"""
        return self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, value: float) -> float:
        """对单个值添加高斯噪声"""
        noise = random.gauss(0.0, self.sigma)
        return value + noise

    def add_noise_vector(self, values: Sequence[float]) -> List[float]:
        """对向量添加高斯噪声"""
        return [self.add_noise(v) for v in values]

    def compute_privacy_loss(self, num_queries: int = 1) -> Tuple[float, float]:
        """计算组合后的隐私损失 (advanced composition)"""
        eps_composed = self.epsilon * math.sqrt(2.0 * num_queries * math.log(1.0 / self.delta))
        delta_composed = self.delta * num_queries
        return (eps_composed, min(delta_composed, 1.0 - 1e-15))

    def get_rdp(self, alpha: float) -> float:
        """计算 Renyi 散度 (RDP) 上界

        对于高斯机制: RDP(alpha) = alpha * sensitivity^2 / (2 * sigma^2)
        """
        if alpha < 1.0:
            raise ValueError(f"Renyi order alpha must be >= 1, got {alpha}")
        return alpha * (self.sensitivity ** 2) / (2.0 * self.sigma ** 2)


class LaplaceMechanism:
    """拉普拉斯机制: 添加 Lap(0, b) 噪声

    对于 epsilon-差分隐私:
        b = sensitivity / epsilon
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")
        self.epsilon: float = epsilon
        self.sensitivity: float = sensitivity
        self.scale: float = self.sensitivity / self.epsilon

    def _sample_laplace(self, scale: float) -> float:
        """采样拉普拉斯分布 Lap(0, scale)"""
        u = random.uniform(-0.5, 0.5)
        return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    def add_noise(self, value: float) -> float:
        """对单个值添加拉普拉斯噪声"""
        noise = self._sample_laplace(self.scale)
        return value + noise

    def add_noise_vector(self, values: Sequence[float]) -> List[float]:
        """对向量添加拉普拉斯噪声"""
        return [self.add_noise(v) for v in values]

    def compute_privacy_loss(self, num_queries: int = 1) -> float:
        """计算组合后的隐私损失 (basic composition)"""
        return self.epsilon * num_queries

    def get_rdp(self, alpha: float) -> float:
        """计算 RDP 上界

        对于拉普拉斯机制: RDP(alpha) = alpha / (2 * epsilon)
        """
        if alpha < 1.0:
            raise ValueError(f"Renyi order alpha must be >= 1, got {alpha}")
        return alpha / (2.0 * self.epsilon)


class ExponentialMechanism:
    """指数机制: 用于非数值查询的差分隐私选择

    选择概率与 exp(epsilon * score(x, r) / (2 * delta_sensitivity)) 成正比
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")
        self.epsilon: float = epsilon
        self.sensitivity: float = sensitivity

    def select(
        self,
        candidates: Sequence[Any],
        score_fn: Callable[[Any], float],
    ) -> Any:
        """使用指数机制从候选集中选择一个元素

        Args:
            candidates: 候选元素列表
            score_fn: 评分函数, 对每个候选返回一个分数

        Returns:
            被选中的候选元素
        """
        if not candidates:
            raise ValueError("candidates must not be empty")

        scores: List[float] = [score_fn(c) for c in candidates]
        max_score: float = max(scores)

        # 计算概率: proportional to exp(epsilon * (score - max_score) / (2 * sensitivity))
        weights: List[float] = []
        for s in scores:
            exponent = self.epsilon * (s - max_score) / (2.0 * self.sensitivity)
            weights.append(math.exp(exponent))

        # 归一化并采样
        total_weight: float = sum(weights)
        if total_weight == 0:
            return random.choice(candidates)

        probabilities: List[float] = [w / total_weight for w in weights]
        r: float = random.random()
        cumulative: float = 0.0
        for i, p in enumerate(probabilities):
            cumulative += p
            if r <= cumulative:
                return candidates[i]
        return candidates[-1]

    def select_top_k(
        self,
        candidates: Sequence[Any],
        score_fn: Callable[[Any], float],
        k: int = 1,
    ) -> List[Any]:
        """选择 top-k 元素 (带隐私保证的排序)"""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k >= len(candidates):
            return list(candidates)

        remaining: List[Any] = list(candidates)
        selected: List[Any] = []

        for _ in range(k):
            choice = self.select(remaining, score_fn)
            selected.append(choice)
            remaining.remove(choice)

        return selected


# ---------------------------------------------------------------------------
# RDP Accountant
# ---------------------------------------------------------------------------

@dataclass
class RDPRecord:
    """单次 RDP 记录"""
    step: int
    alpha: float
    rdp_value: float
    sigma: float
    num_samples: int
    timestamp: float = field(default_factory=time.time)


class RDPAccountant:
    """Renyi Differential Privacy 会计

    基于 "Renyi Differential Privacy of the Sampled Gaussian Mechanism" 论文实现。
    跟踪多次 DP-SGD 迭代的累积隐私损失。
    """

    def __init__(
        self,
        orders: Optional[Sequence[float]] = None,
        target_delta: float = 1e-5,
    ) -> None:
        self.orders: List[float] = list(orders or [
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0
        ])
        self.target_delta: float = target_delta
        self._rdp_sum: Dict[float, float] = {a: 0.0 for a in self.orders}
        self._steps: int = 0
        self._history: List[RDPRecord] = []

    def _compute_rdp_poisson(
        self,
        sigma: float,
        q: float,
        alpha: float,
    ) -> float:
        """计算泊松子采样的 RDP

        Args:
            sigma: 噪声乘子
            q: 采样率 (batch_size / dataset_size)
            alpha: Renyi 阶数
        """
        if q == 0:
            return 0.0

        # 使用 RDP 的近似公式
        # RDP(alpha) = q^2 * alpha / (2 * sigma^2) + small correction
        rdp_gaussian = alpha / (2.0 * sigma * sigma)
        rdp_sampled = q * q * rdp_gaussian

        # 对于较大的 alpha, 添加修正项
        if alpha > 2.0:
            log_term = math.log(1.0 / q) if q > 0 else 0.0
            correction = q * (alpha - 1.0) / (alpha * sigma * sigma) * log_term
            rdp_sampled += correction

        return rdp_sampled

    def accumulate(
        self,
        sigma: float,
        q: float,
        num_steps: int = 1,
    ) -> None:
        """累积多步 RDP

        Args:
            sigma: 噪声乘子 (noise_multiplier)
            q: 采样率
            num_steps: 训练步数
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not (0 < q <= 1):
            raise ValueError(f"q must be in (0, 1], got {q}")

        for _ in range(num_steps):
            self._steps += 1
            for alpha in self.orders:
                rdp_val = self._compute_rdp_poisson(sigma, q, alpha)
                self._rdp_sum[alpha] += rdp_val
                self._history.append(RDPRecord(
                    step=self._steps,
                    alpha=alpha,
                    rdp_value=rdp_val,
                    sigma=sigma,
                    num_samples=int(q * 10000),
                ))

    def _rdp_to_epsilon(self, alpha: float, rdp: float, delta: float) -> float:
        """将 RDP 转换为 (epsilon, delta)-DP

        epsilon = rdp + log(1/delta) / (alpha - 1)
        """
        if alpha <= 1.0:
            return float('inf')
        return rdp + math.log(1.0 / delta) / (alpha - 1.0)

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """获取当前最优 epsilon (在所有 alpha 阶数中取最小值)

        Args:
            delta: 目标 delta, 默认使用初始化时的 target_delta

        Returns:
            最优 epsilon 值
        """
        d = delta if delta is not None else self.target_delta
        if d <= 0 or d >= 1:
            raise ValueError(f"delta must be in (0, 1), got {d}")

        best_epsilon: float = float('inf')
        for alpha in self.orders:
            rdp_total = self._rdp_sum[alpha]
            eps = self._rdp_to_epsilon(alpha, rdp_total, d)
            best_epsilon = min(best_epsilon, eps)

        return best_epsilon

    def get_epsilon_for_alpha(
        self,
        alpha: float,
        delta: Optional[float] = None,
    ) -> float:
        """获取指定 alpha 阶数的 epsilon"""
        d = delta if delta is not None else self.target_delta
        rdp_total = self._rdp_sum.get(alpha, 0.0)
        return self._rdp_to_epsilon(alpha, rdp_total, d)

    def get_privacy_budget_remaining(
        self,
        target_epsilon: float,
        delta: Optional[float] = None,
    ) -> float:
        """获取剩余隐私预算"""
        current_eps = self.get_epsilon(delta)
        return max(0.0, target_epsilon - current_eps)

    def get_history(self) -> List[RDPRecord]:
        """获取 RDP 累积历史"""
        return list(self._history)

    @property
    def steps(self) -> int:
        """已累积的步数"""
        return self._steps

    def reset(self) -> None:
        """重置会计"""
        self._rdp_sum = {a: 0.0 for a in self.orders}
        self._steps = 0
        self._history.clear()

    def get_status(self) -> Dict[str, Any]:
        """获取当前会计状态"""
        return {
            "steps": self._steps,
            "epsilon": self.get_epsilon(),
            "delta": self.target_delta,
            "orders_used": self.orders,
            "rdp_totals": dict(self._rdp_sum),
        }


# ---------------------------------------------------------------------------
# Privacy Budget
# ---------------------------------------------------------------------------

class BudgetExhaustedError(Exception):
    """隐私预算耗尽异常"""
    pass


@dataclass
class BudgetEntry:
    """预算使用记录"""
    timestamp: float
    epsilon_spent: float
    delta_spent: float
    mechanism: str
    description: str = ""


class PrivacyBudget:
    """隐私预算管理器

    跟踪和管理差分隐私预算的使用情况。
    """

    def __init__(
        self,
        total_epsilon: float = 10.0,
        total_delta: float = 1e-3,
        strict: bool = True,
    ) -> None:
        if total_epsilon <= 0:
            raise ValueError(f"total_epsilon must be positive, got {total_epsilon}")
        if not (0 < total_delta < 1):
            raise ValueError(f"total_delta must be in (0, 1), got {total_delta}")
        self.total_epsilon: float = total_epsilon
        self.total_delta: float = total_delta
        self.strict: bool = strict
        self._spent_epsilon: float = 0.0
        self._spent_delta: float = 0.0
        self._entries: List[BudgetEntry] = []

    @property
    def remaining_epsilon(self) -> float:
        """剩余 epsilon 预算"""
        return max(0.0, self.total_epsilon - self._spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        """剩余 delta 预算"""
        return max(0.0, self.total_delta - self._spent_delta)

    @property
    def spent_epsilon(self) -> float:
        """已花费的 epsilon"""
        return self._spent_epsilon

    @property
    def spent_delta(self) -> float:
        """已花费的 delta"""
        return self._spent_delta

    def check(self, epsilon: float, delta: float = 0.0) -> bool:
        """检查是否有足够预算

        Args:
            epsilon: 需要的 epsilon
            delta: 需要的 delta

        Returns:
            True 如果有足够预算
        """
        return (
            self._spent_epsilon + epsilon <= self.total_epsilon
            and self._spent_delta + delta <= self.total_delta
        )

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "unknown",
        description: str = "",
    ) -> None:
        """花费隐私预算

        Args:
            epsilon: 花费的 epsilon
            delta: 花费的 delta
            mechanism: 使用的机制名称
            description: 描述信息

        Raises:
            BudgetExhaustedError: 预算不足且 strict=True
        """
        if epsilon < 0 or delta < 0:
            raise ValueError("epsilon and delta must be non-negative")

        if not self.check(epsilon, delta):
            if self.strict:
                raise BudgetExhaustedError(
                    f"Privacy budget exhausted. "
                    f"Requested: eps={epsilon}, delta={delta}. "
                    f"Remaining: eps={self.remaining_epsilon:.6f}, "
                    f"delta={self.remaining_delta:.10f}"
                )
            # 非严格模式: 只花费到剩余预算为止
            epsilon = min(epsilon, self.remaining_epsilon)
            delta = min(delta, self.remaining_delta)

        self._spent_epsilon += epsilon
        self._spent_delta += delta
        self._entries.append(BudgetEntry(
            timestamp=time.time(),
            epsilon_spent=epsilon,
            delta_spent=delta,
            mechanism=mechanism,
            description=description,
        ))

    def get_entries(self) -> List[BudgetEntry]:
        """获取所有预算使用记录"""
        return list(self._entries)

    def get_utilization(self) -> Dict[str, float]:
        """获取预算使用率"""
        return {
            "epsilon_utilization": self._spent_epsilon / self.total_epsilon if self.total_epsilon > 0 else 0.0,
            "delta_utilization": self._spent_delta / self.total_delta if self.total_delta > 0 else 0.0,
            "epsilon_remaining": self.remaining_epsilon,
            "delta_remaining": self.remaining_delta,
        }

    def reset(self) -> None:
        """重置预算"""
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._entries.clear()


# ---------------------------------------------------------------------------
# Gradient Clipper (DP-SGD)
# ---------------------------------------------------------------------------

@dataclass
class GradientStats:
    """梯度统计信息"""
    original_norm: float
    clipped_norm: float
    clip_ratio: float
    num_elements: int


class GradientClipper:
    """DP-SGD 梯度裁剪器

    实现差分隐私随机梯度下降中的梯度裁剪和噪声添加。
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        mechanism: str = "gaussian",
    ) -> None:
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be non-negative, got {noise_multiplier}")
        self.max_norm: float = max_norm
        self.noise_multiplier: float = noise_multiplier
        self.mechanism: str = mechanism
        self._clip_count: int = 0
        self._total_norm: float = 0.0

    def _compute_l2_norm(self, gradient: Sequence[float]) -> float:
        """计算 L2 范数"""
        return math.sqrt(sum(g * g for g in gradient))

    def clip_gradient(self, gradient: List[float]) -> Tuple[List[float], GradientStats]:
        """裁剪梯度到最大范数

        Args:
            gradient: 原始梯度向量

        Returns:
            (裁剪后的梯度, 统计信息)
        """
        if not gradient:
            return ([], GradientStats(0.0, 0.0, 0.0, 0))

        norm = self._compute_l2_norm(gradient)
        self._total_norm += norm

        if norm <= self.max_norm:
            return (list(gradient), GradientStats(norm, norm, 1.0, len(gradient)))

        clip_ratio = self.max_norm / norm
        clipped = [g * clip_ratio for g in gradient]
        self._clip_count += 1

        return (clipped, GradientStats(norm, self.max_norm, clip_ratio, len(gradient)))

    def clip_and_noise(
        self,
        gradient: List[float],
    ) -> Tuple[List[float], GradientStats]:
        """裁剪并添加噪声

        Args:
            gradient: 原始梯度向量

        Returns:
            (带噪声的裁剪梯度, 统计信息)
        """
        clipped, stats = self.clip_gradient(gradient)

        if self.noise_multiplier > 0 and clipped:
            sigma = self.max_norm * self.noise_multiplier
            if self.mechanism == "gaussian":
                noised = [g + random.gauss(0.0, sigma) for g in clipped]
            elif self.mechanism == "laplace":
                scale = self.max_norm * self.noise_multiplier
                noised = [g + self._sample_laplace(scale) for g in clipped]
            else:
                noised = clipped
        else:
            noised = clipped

        return (noised, stats)

    @staticmethod
    def _sample_laplace(scale: float) -> float:
        """采样拉普拉斯分布"""
        u = random.uniform(-0.5, 0.5)
        return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    def clip_batch(
        self,
        gradients: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[GradientStats]]:
        """批量裁剪梯度

        Args:
            gradients: 梯度列表

        Returns:
            (裁剪后的梯度列表, 统计信息列表)
        """
        results: List[Tuple[List[float], GradientStats]] = []
        for grad in gradients:
            results.append(self.clip_gradient(list(grad)))
        clipped_grads = [r[0] for r in results]
        stats = [r[1] for r in results]
        return (clipped_grads, stats)

    def aggregate_and_noise(
        self,
        gradients: Sequence[Sequence[float]],
    ) -> Tuple[List[float], GradientStats]:
        """聚合多个梯度并添加噪声

        先裁剪每个梯度, 然后取平均, 最后添加噪声。
        """
        if not gradients:
            return ([], GradientStats(0.0, 0.0, 0.0, 0))

        clipped_grads, stats_list = self.clip_batch(gradients)
        n = len(clipped_grads)

        # 平均
        num_dims = len(clipped_grads[0])
        averaged: List[float] = []
        for d in range(num_dims):
            avg = sum(g[d] for g in clipped_grads if d < len(g)) / n
            averaged.append(avg)

        # 添加噪声
        if self.noise_multiplier > 0:
            sigma = self.max_norm * self.noise_multiplier / n
            noised = [a + random.gauss(0.0, sigma) for a in averaged]
        else:
            noised = averaged

        avg_original_norm = sum(s.original_norm for s in stats_list) / n
        return (noised, GradientStats(
            original_norm=avg_original_norm,
            clipped_norm=self.max_norm,
            clip_ratio=self.max_norm / avg_original_norm if avg_original_norm > 0 else 1.0,
            num_elements=num_dims,
        ))

    def get_stats(self) -> Dict[str, Any]:
        """获取裁剪统计"""
        return {
            "clip_count": self._clip_count,
            "total_norm_accumulated": self._total_norm,
            "max_norm": self.max_norm,
            "noise_multiplier": self.noise_multiplier,
            "mechanism": self.mechanism,
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self._clip_count = 0
        self._total_norm = 0.0


# ---------------------------------------------------------------------------
# Composition Theorems
# ---------------------------------------------------------------------------

class CompositionType(Enum):
    """组合类型"""
    BASIC = "basic"
    ADVANCED = "advanced"
    OPTIMAL = "optimal"
    RDP = "rdp"


@dataclass
class CompositionResult:
    """组合结果"""
    epsilon: float
    delta: float
    composition_type: CompositionType
    num_mechanisms: int
    details: Dict[str, Any] = field(default_factory=dict)


class DPComposer:
    """差分隐私组合定理

    支持基本组合、高级组合和 RDP 组合。
    """

    @staticmethod
    def basic_composition(
        mechanisms: Sequence[Tuple[float, float]],
    ) -> CompositionResult:
        """基本组合定理

        如果 M1 是 (eps1, delta1)-DP, M2 是 (eps2, delta2)-DP,
        则 M1 o M2 是 (eps1+eps2, delta1+delta2)-DP。

        Args:
            mechanisms: [(epsilon, delta), ...] 列表

        Returns:
            组合后的隐私保证
        """
        total_eps = sum(e for e, _ in mechanisms)
        total_delta = sum(d for _, d in mechanisms)

        if total_delta >= 1.0:
            total_delta = 1.0 - 1e-15

        return CompositionResult(
            epsilon=total_eps,
            delta=total_delta,
            composition_type=CompositionType.BASIC,
            num_mechanisms=len(mechanisms),
            details={"individual_epsilons": [e for e, _ in mechanisms]},
        )

    @staticmethod
    def advanced_composition(
        mechanisms: Sequence[Tuple[float, float]],
        delta: float = 1e-5,
    ) -> CompositionResult:
        """高级组合定理

        k 个 (epsilon, delta)-DP 机制的组合满足:
            (epsilon * sqrt(2*k*ln(1/delta)) + k*epsilon*(e^epsilon - 1), k*delta)-DP

        Args:
            mechanisms: [(epsilon, delta), ...] 列表
            delta: 目标 delta

        Returns:
            组合后的隐私保证
        """
        k = len(mechanisms)
        if k == 0:
            return CompositionResult(
                epsilon=0.0, delta=0.0,
                composition_type=CompositionType.ADVANCED,
                num_mechanisms=0,
            )

        avg_eps = sum(e for e, _ in mechanisms) / k
        total_delta = sum(d for _, d in mechanisms)

        if avg_eps <= 0:
            return CompositionResult(
                epsilon=0.0, delta=total_delta,
                composition_type=CompositionType.ADVANCED,
                num_mechanisms=k,
            )

        sqrt_term = math.sqrt(2.0 * k * math.log(1.0 / delta))
        exp_term = k * avg_eps * (math.exp(avg_eps) - 1.0)
        composed_eps = avg_eps * sqrt_term + exp_term

        return CompositionResult(
            epsilon=composed_eps,
            delta=min(total_delta + delta, 1.0 - 1e-15),
            composition_type=CompositionType.ADVANCED,
            num_mechanisms=k,
            details={
                "k": k,
                "avg_epsilon": avg_eps,
                "sqrt_term": avg_eps * sqrt_term,
                "exp_term": exp_term,
            },
        )

    @staticmethod
    def rdp_composition(
        rdp_values: Sequence[Tuple[float, float]],
        target_delta: float = 1e-5,
        orders: Optional[Sequence[float]] = None,
    ) -> CompositionResult:
        """RDP 组合

        多个机制的 RDP 值相加, 然后转换为 (epsilon, delta)-DP。

        Args:
            rdp_values: [(alpha, rdp_value), ...] 每个机制的 RDP
            target_delta: 目标 delta
            orders: 要考虑的 Renyi 阶数

        Returns:
            组合后的隐私保证
        """
        if not rdp_values:
            return CompositionResult(
                epsilon=0.0, delta=0.0,
                composition_type=CompositionType.RDP,
                num_mechanisms=0,
            )

        # 按 alpha 分组求和
        rdp_sums: Dict[float, float] = {}
        for alpha, rdp_val in rdp_values:
            rdp_sums[alpha] = rdp_sums.get(alpha, 0.0) + rdp_val

        # 在所有 alpha 中找最优 epsilon
        best_epsilon: float = float('inf')
        best_alpha: float = 1.0

        for alpha, rdp_total in rdp_sums.items():
            if alpha <= 1.0:
                continue
            eps = rdp_total + math.log(1.0 / target_delta) / (alpha - 1.0)
            if eps < best_epsilon:
                best_epsilon = eps
                best_alpha = alpha

        return CompositionResult(
            epsilon=best_epsilon,
            delta=target_delta,
            composition_type=CompositionType.RDP,
            num_mechanisms=len(rdp_values),
            details={
                "best_alpha": best_alpha,
                "rdp_totals": dict(rdp_sums),
            },
        )

    @staticmethod
    def optimal_composition(
        mechanisms: Sequence[Tuple[float, float]],
        target_delta: float = 1e-5,
    ) -> CompositionResult:
        """最优组合 (在基本组合和高级组合中取更紧的界)

        Args:
            mechanisms: [(epsilon, delta), ...] 列表
            target_delta: 目标 delta

        Returns:
            组合后的隐私保证
        """
        basic = DPComposer.basic_composition(mechanisms)
        advanced = DPComposer.advanced_composition(mechanisms, target_delta)

        if basic.epsilon <= advanced.epsilon:
            return basic
        return advanced

    @staticmethod
    def parallel_composition(
        mechanisms: Sequence[Tuple[float, float]],
        num_partitions: int,
    ) -> CompositionResult:
        """并行组合定理

        如果数据集被分为不相交的 k 个分区, 每个分区上应用 (eps_i, delta_i)-DP,
        则整体满足 (max(eps_i), sum(delta_i))-DP。

        Args:
            mechanisms: 每个分区的 (epsilon, delta)
            num_partitions: 分区数量

        Returns:
            组合后的隐私保证
        """
        if not mechanisms:
            return CompositionResult(
                epsilon=0.0, delta=0.0,
                composition_type=CompositionType.BASIC,
                num_mechanisms=0,
            )

        max_eps = max(e for e, _ in mechanisms)
        sum_delta = sum(d for _, d in mechanisms)

        return CompositionResult(
            epsilon=max_eps,
            delta=min(sum_delta, 1.0 - 1e-15),
            composition_type=CompositionType.BASIC,
            num_mechanisms=num_partitions,
            details={"partition_count": num_partitions},
        )

    @staticmethod
    def post_process(
        composed: CompositionResult,
        function_sensitivity: float = 0.0,
    ) -> CompositionResult:
        """后处理定理

        差分隐私机制的结果经过任何(数据无关的)后处理函数,
        仍然保持相同的隐私保证。

        Args:
            composed: 已组合的隐私保证
            function_sensitivity: 后处理函数的敏感度(0 表示纯后处理)

        Returns:
            后处理后的隐私保证
        """
        return CompositionResult(
            epsilon=composed.epsilon,
            delta=composed.delta,
            composition_type=composed.composition_type,
            num_mechanisms=composed.num_mechanisms,
            details={
                **composed.details,
                "post_processing_sensitivity": function_sensitivity,
            },
        )


# ---------------------------------------------------------------------------
# Main Differential Privacy Facade
# ---------------------------------------------------------------------------

class DifferentialPrivacy:
    """差分隐私统一接口

    提供高级 API, 整合所有差分隐私组件。
    """

    def __init__(self, config: Optional[DPConfig] = None) -> None:
        self.config: DPConfig = config or DPConfig()
        self.config.validate()

        self.gaussian: GaussianMechanism = GaussianMechanism(
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            sensitivity=1.0,
        )
        self.laplace: LaplaceMechanism = LaplaceMechanism(
            epsilon=self.config.epsilon,
            sensitivity=1.0,
        )
        self.exponential: ExponentialMechanism = ExponentialMechanism(
            epsilon=self.config.epsilon,
            sensitivity=1.0,
        )
        self.accountant: RDPAccountant = RDPAccountant(
            orders=self.config.orders,
            target_delta=self.config.target_delta,
        )
        self.budget: PrivacyBudget = PrivacyBudget(
            total_epsilon=self.config.epsilon * 10,
            total_delta=self.config.target_delta * 10,
        )
        self.clipper: GradientClipper = GradientClipper(
            max_norm=self.config.max_gradient_norm,
            noise_multiplier=self.config.noise_multiplier,
        )
        self.composer: DPComposer = DPComposer()

    def privatize_value(
        self,
        value: float,
        mechanism: Optional[str] = None,
    ) -> float:
        """对单个值添加隐私噪声

        Args:
            value: 原始值
            mechanism: 机制名称, 默认使用配置中的机制

        Returns:
            带噪声的值
        """
        mech = mechanism or self.config.mechanism
        if mech == "gaussian":
            return self.gaussian.add_noise(value)
        elif mech == "laplace":
            return self.laplace.add_noise(value)
        else:
            raise ValueError(f"Unknown mechanism: {mech}")

    def privatize_vector(
        self,
        values: Sequence[float],
        mechanism: Optional[str] = None,
    ) -> List[float]:
        """对向量添加隐私噪声"""
        mech = mechanism or self.config.mechanism
        if mech == "gaussian":
            return self.gaussian.add_noise_vector(values)
        elif mech == "laplace":
            return self.laplace.add_noise_vector(values)
        else:
            raise ValueError(f"Unknown mechanism: {mech}")

    def privatize_query(
        self,
        candidates: Sequence[Any],
        score_fn: Callable[[Any], float],
    ) -> Any:
        """使用指数机制进行隐私保护的选择"""
        return self.exponential.select(candidates, score_fn)

    def dp_sgd_step(
        self,
        gradients: Sequence[Sequence[float]],
        sampling_rate: float,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """执行一步 DP-SGD

        Args:
            gradients: 梯度列表 (每个样本的梯度)
            sampling_rate: 采样率 q = batch_size / dataset_size

        Returns:
            (聚合后的带噪声梯度, 统计信息)
        """
        aggregated, stats = self.clipper.aggregate_and_noise(gradients)

        # 更新 RDP 会计
        self.accountant.accumulate(
            sigma=self.config.noise_multiplier,
            q=sampling_rate,
            num_steps=1,
        )

        # 更新预算
        current_eps = self.accountant.get_epsilon()
        self.budget.spend(
            epsilon=current_eps / max(self.accountant.steps, 1),
            delta=self.config.target_delta / max(self.accountant.steps, 1),
            mechanism="dp_sgd",
            description=f"DP-SGD step {self.accountant.steps}",
        )

        info = self.get_status()
        info["gradient_stats"] = {
            "original_norm": stats.original_norm,
            "clipped_norm": stats.clipped_norm,
            "clip_ratio": stats.clip_ratio,
            "num_elements": stats.num_elements,
        }
        return (aggregated, info)

    def get_privacy_report(self) -> Dict[str, Any]:
        """获取完整隐私报告"""
        return {
            "config": {
                "epsilon": self.config.epsilon,
                "delta": self.config.delta,
                "mechanism": self.config.mechanism,
                "noise_multiplier": self.config.noise_multiplier,
                "max_gradient_norm": self.config.max_gradient_norm,
            },
            "accountant": self.accountant.get_status(),
            "budget": self.budget.get_utilization(),
            "clipper": self.clipper.get_stats(),
        }

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态摘要"""
        return {
            "steps": self.accountant.steps,
            "current_epsilon": self.accountant.get_epsilon(),
            "budget_remaining": self.budget.remaining_epsilon,
            "budget_utilization": self.budget.get_utilization(),
        }

    def reset(self) -> None:
        """重置所有组件"""
        self.accountant.reset()
        self.budget.reset()
        self.clipper.reset_stats()
