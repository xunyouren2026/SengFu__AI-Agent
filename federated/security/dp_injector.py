"""
Differential Privacy Injector Module
差分隐私注入器模块

实现联邦学习中的差分隐私机制，包括：
1. 梯度裁剪（L2范数裁剪）
2. 高斯机制（Gaussian Mechanism）
3. 拉普拉斯机制（Laplace Mechanism）
4. RDP隐私预算记账（Renyi Differential Privacy Accountant）
5. 隐私预算管理与组合定理
6. 自适应噪声调整

Author: AGI Unified Framework
"""

import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


# ============== 差分隐私配置 ==============

@dataclass
class DPConfig:
    """
    差分隐私配置

    Attributes:
        epsilon: 隐私预算上界，越小隐私保护越强
        delta: 隐私失败概率，通常设为 1/n 或更小
        noise_multiplier: 噪声乘数 sigma = noise_multiplier * max_grad_norm
        max_grad_norm: 梯度裁剪阈值（L2范数上界）
        mechanism: 噪声机制类型 ('gaussian' 或 'laplace')
        clip_per_layer: 是否按层独立裁剪（默认为全局裁剪）
        secure_rng_seed: 安全随机数种子（可选）
    """
    epsilon: float = 8.0
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    mechanism: str = "gaussian"
    clip_per_layer: bool = False
    secure_rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(f"epsilon 必须 > 0，当前值: {self.epsilon}")
        if not (0 < self.delta < 1):
            raise ValueError(f"delta 必须在 (0, 1) 之间，当前值: {self.delta}")
        if self.noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier 必须 > 0，当前值: {self.noise_multiplier}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm 必须 > 0，当前值: {self.max_grad_norm}")
        if self.mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"mechanism 必须为 'gaussian' 或 'laplace'，当前值: {self.mechanism}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "mechanism": self.mechanism,
            "clip_per_layer": self.clip_per_layer,
        }


# ============== 梯度裁剪器 ==============

class GradientClipper:
    """
    梯度裁剪器

    按全局L2范数对梯度进行裁剪，将梯度范数限制在 max_grad_norm 以内。
    这是差分隐私的基本步骤，通过裁剪来控制敏感度（sensitivity）。

    裁剪公式:
        如果 ||g||_2 > C，则 g' = g * (C / ||g||_2)
        否则 g' = g

    其中 C = max_grad_norm
    """

    def __init__(self, max_grad_norm: float = 1.0, clip_per_layer: bool = False) -> None:
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm 必须 > 0，当前值: {max_grad_norm}")
        self._max_grad_norm = max_grad_norm
        self._clip_per_layer = clip_per_layer
        self._clip_history: List[float] = []

    @property
    def max_grad_norm(self) -> float:
        return self._max_grad_norm

    def clip_gradient(
        self,
        gradient: List[float],
        max_norm: Optional[float] = None,
    ) -> List[float]:
        """
        对单个梯度向量进行L2范数裁剪

        Args:
            gradient: 原始梯度向量
            max_norm: 可选的裁剪阈值，默认使用初始化时的值

        Returns:
            裁剪后的梯度向量
        """
        if not gradient:
            return gradient

        C = max_norm if max_norm is not None else self._max_grad_norm
        norm = self._compute_l2_norm(gradient)

        if norm > C:
            scale = C / norm
            clipped = [g * scale for g in gradient]
        else:
            clipped = list(gradient)

        self._clip_history.append(norm)
        return clipped

    def clip_layer_gradients(
        self,
        layers: List[List[float]],
        max_norm: Optional[float] = None,
    ) -> List[List[float]]:
        """
        对多层梯度进行裁剪

        如果 clip_per_layer=True，则对每一层独立裁剪；
        否则将所有层的梯度展平后按全局范数裁剪。

        Args:
            layers: 多层梯度列表，每层为一个向量
            max_norm: 可选的裁剪阈值

        Returns:
            裁剪后的多层梯度
        """
        if not layers:
            return layers

        if self._clip_per_layer:
            return [self.clip_gradient(layer, max_norm) for layer in layers]

        # 全局裁剪：展平所有梯度，计算全局范数
        flat = self._flatten(layers)
        global_norm = self._compute_l2_norm(flat)
        C = max_norm if max_norm is not None else self._max_grad_norm

        if global_norm > C:
            scale = C / global_norm
            return [[g * scale for g in layer] for layer in layers]

        return [list(layer) for layer in layers]

    def get_clip_ratio(self) -> float:
        """
        获取裁剪比例（被裁剪的梯度占总数的比例）

        Returns:
            裁剪比例，范围 [0, 1]
        """
        if not self._clip_history:
            return 0.0
        C = self._max_grad_norm
        clipped_count = sum(1 for norm in self._clip_history if norm > C)
        return clipped_count / len(self._clip_history)

    def get_average_norm(self) -> float:
        """获取历史平均梯度范数"""
        if not self._clip_history:
            return 0.0
        return sum(self._clip_history) / len(self._clip_history)

    def reset_history(self) -> None:
        """重置裁剪历史"""
        self._clip_history.clear()

    @staticmethod
    def _compute_l2_norm(vector: List[float]) -> float:
        """计算向量的L2范数"""
        return math.sqrt(sum(x * x for x in vector))

    @staticmethod
    def _flatten(layers: List[List[float]]) -> List[float]:
        """将多层梯度展平为一维向量"""
        result: List[float] = []
        for layer in layers:
            result.extend(layer)
        return result


# ============== 高斯机制 ==============

class GaussianMechanism:
    """
    高斯机制

    向查询结果添加校准的高斯噪声以满足 (epsilon, delta)-差分隐私。

    对于敏感度为 S 的查询函数 f，添加 N(0, sigma^2) 噪声：
        sigma = S * sqrt(2 * ln(1.25 / delta)) / epsilon

    这保证了 (epsilon, delta)-差分隐私。

    在联邦学习中，敏感度 S = 2 * max_grad_norm（因为相邻数据集的梯度差最多为 2C）。
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon 必须 > 0，当前值: {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta 必须在 (0, 1) 之间，当前值: {delta}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity 必须 > 0，当前值: {sensitivity}")

        self._epsilon = epsilon
        self._delta = delta
        self._sensitivity = sensitivity

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    def compute_sigma(self) -> float:
        """
        计算高斯噪声的标准差

        使用标准公式:
            sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

        Returns:
            噪声标准差 sigma
        """
        return self._sensitivity * math.sqrt(2.0 * math.log(1.25 / self._delta)) / self._epsilon

    def add_noise(self, value: float) -> float:
        """
        向标量值添加高斯噪声

        Args:
            value: 原始值

        Returns:
            加噪后的值
        """
        sigma = self.compute_sigma()
        noise = random.gauss(0.0, sigma)
        return value + noise

    def add_noise_vector(self, vector: List[float]) -> List[float]:
        """
        向向量添加独立同分布的高斯噪声

        Args:
            vector: 原始向量

        Returns:
            加噪后的向量
        """
        sigma = self.compute_sigma()
        return [v + random.gauss(0.0, sigma) for v in vector]

    def add_noise_with_multiplier(
        self,
        vector: List[float],
        noise_multiplier: float,
        norm_bound: float,
    ) -> List[float]:
        """
        使用噪声乘数添加噪声

        在联邦学习中常用形式:
            sigma = noise_multiplier * norm_bound

        Args:
            vector: 原始梯度向量
            noise_multiplier: 噪声乘数
            norm_bound: 梯度范数上界（裁剪阈值）

        Returns:
            加噪后的向量
        """
        sigma = noise_multiplier * norm_bound
        return [v + random.gauss(0.0, sigma) for v in vector]

    def effective_epsilon(self, steps: int, noise_multiplier: float) -> float:
        """
        给定步数和噪声乘数，计算有效 epsilon

        使用高斯机制的组合定理近似:
            epsilon ≈ sqrt(2 * ln(1.25/delta)) * steps / noise_multiplier

        Args:
            steps: 训练步数
            noise_multiplier: 噪声乘数

        Returns:
            有效 epsilon
        """
        if noise_multiplier <= 0:
            return float("inf")
        return math.sqrt(2.0 * math.log(1.25 / self._delta)) * steps / noise_multiplier


# ============== 拉普拉斯机制 ==============

class LaplaceMechanism:
    """
    拉普拉斯机制

    向查询结果添加校准的拉普拉斯噪声以满足 epsilon-差分隐私。

    对于敏感度为 S 的查询函数 f，添加 Laplace(0, b) 噪声：
        b = S / epsilon

    拉普拉斯机制提供纯 epsilon-差分隐私（不需要 delta 参数），
    但噪声通常比高斯机制更大。
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon 必须 > 0，当前值: {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity 必须 > 0，当前值: {sensitivity}")

        self._epsilon = epsilon
        self._sensitivity = sensitivity

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    def compute_scale(self) -> float:
        """
        计算拉普拉斯分布的尺度参数

        公式: b = sensitivity / epsilon

        Returns:
            拉普拉斯尺度参数 b
        """
        return self._sensitivity / self._epsilon

    def add_noise(self, value: float) -> float:
        """
        向标量值添加拉普拉斯噪声

        使用逆变换采样生成拉普拉斯随机变量:
            X = b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
        其中 U ~ Uniform(0, 1)

        Args:
            value: 原始值

        Returns:
            加噪后的值
        """
        b = self.compute_scale()
        noise = self._sample_laplace(0.0, b)
        return value + noise

    def add_noise_vector(self, vector: List[float]) -> List[float]:
        """
        向向量添加独立同分布的拉普拉斯噪声

        Args:
            vector: 原始向量

        Returns:
            加噪后的向量
        """
        b = self.compute_scale()
        return [v + self._sample_laplace(0.0, b) for v in vector]

    @staticmethod
    def _sample_laplace(loc: float, scale: float) -> float:
        """
        生成拉普拉斯随机变量

        使用逆变换采样法:
            X = mu - b * sign(U) * ln(1 - 2|U|)
        其中 U ~ Uniform(-0.5, 0.5)

        Args:
            loc: 位置参数（均值）
            scale: 尺度参数 b

        Returns:
            拉普拉斯随机变量
        """
        u = random.uniform(-0.5, 0.5)
        return loc - scale * (1 if u > 0 else -1) * math.log(1.0 - 2.0 * abs(u))

    def effective_epsilon(self, steps: int) -> float:
        """
        给定步数，计算组合后的有效 epsilon

        对于串行组合的拉普拉斯机制:
            epsilon_total = steps * epsilon_per_step

        Args:
            steps: 训练步数

        Returns:
            有效 epsilon
        """
        return steps * self._epsilon


# ============== RDP 隐私预算记账 ==============

class RDPAccountant:
    """
    Renyi Differential Privacy (RDP) 隐私预算记账器

    RDP 提供了比标准组合定理更紧的隐私边界。
    对于高斯机制，alpha 阶 RDP 为:
        RDP(alpha) = alpha / (2 * sigma^2)

    通过跟踪多个 alpha 阶的 RDP 值，然后转换为 (epsilon, delta)-DP，
    可以获得更精确的隐私保证。

    转换公式（从 RDP 到 (epsilon, delta)-DP）:
        epsilon = min_alpha [ RDP(alpha) + log(1/delta) / (alpha - 1) ]

    参考文献:
        M. Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
    """

    # 用于搜索最优 alpha 的范围
    DEFAULT_ALPHA_RANGE: Tuple[float, float] = (2.0, 256.0)
    DEFAULT_ALPHA_STEPS: int = 64

    def __init__(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        sample_rate: float = 1.0,
    ) -> None:
        """
        Args:
            noise_multiplier: 噪声乘数 sigma / max_grad_norm
            max_grad_norm: 梯度裁剪阈值
            delta: 目标 delta
            sample_rate: 每步采样率（子采样率）
        """
        if noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier 必须 > 0，当前值: {noise_multiplier}")
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm 必须 > 0，当前值: {max_grad_norm}")

        self._noise_multiplier = noise_multiplier
        self._max_grad_norm = max_grad_norm
        self._delta = delta
        self._sample_rate = sample_rate
        self._steps: int = 0

        # 存储每个 alpha 阶的累积 RDP 值
        self._rdp_values: Dict[float, float] = {}

    @property
    def noise_multiplier(self) -> float:
        return self._noise_multiplier

    @property
    def steps(self) -> int:
        return self._steps

    def accumulate_step(self, num_steps: int = 1) -> None:
        """
        记录一步（或多步）隐私消耗

        每步的 RDP 贡献为:
            RDP(alpha) = alpha / (2 * sigma^2) * q^2 * alpha / (alpha - 1)

        其中 q = sample_rate, sigma = noise_multiplier

        简化（当 q=1 时）:
            RDP(alpha) = alpha / (2 * sigma^2)

        Args:
            num_steps: 步数
        """
        sigma = self._noise_multiplier
        q = self._sample_rate

        # 在一系列 alpha 值上计算 RDP
        alphas = self._get_alpha_grid()

        for alpha in alphas:
            # Poisson 子采样 RDP（当 q < 1 时使用更紧的界）
            if q < 1.0:
                rdp_per_step = self._compute_subsampled_rdp(alpha, sigma, q)
            else:
                # 无子采样: RDP(alpha) = alpha / (2 * sigma^2)
                rdp_per_step = alpha / (2.0 * sigma * sigma)

            if alpha not in self._rdp_values:
                self._rdp_values[alpha] = 0.0
            self._rdp_values[alpha] += rdp_per_step * num_steps

        self._steps += num_steps

    def _compute_subsampled_rdp(
        self, alpha: float, sigma: float, q: float
    ) -> float:
        """
        计算子采样 RDP

        使用 Abadi et al. 的定理 7:
            RDP_subsampled(alpha) = log(1 + q*(e^{RDP(alpha)} - 1)) / (alpha - 1)

        简化近似:
            当 q 较小时，RDP_subsampled ≈ q^2 * alpha / (2 * sigma^2)

        Args:
            alpha: Renyi 阶数
            sigma: 噪声标准差
            q: 采样率

        Returns:
            子采样 RDP 值
        """
        # 基础 RDP（无子采样）
        base_rdp = alpha / (2.0 * sigma * sigma)

        # 使用简化公式: q^2 * alpha / (2 * sigma^2) * alpha / (alpha - 1)
        if alpha <= 1.0:
            return float("inf")
        return q * q * base_rdp * alpha / (alpha - 1.0)

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        将累积的 RDP 转换为 (epsilon, delta)-DP

        使用最优 alpha 搜索:
            epsilon = min_alpha [ RDP(alpha) + log(1/delta) / (alpha - 1) ]

        Args:
            delta: 目标 delta，默认使用初始化时的值

        Returns:
            最优 epsilon
        """
        if self._steps == 0:
            return 0.0

        target_delta = delta if delta is not None else self._delta
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"delta 必须在 (0, 1) 之间，当前值: {target_delta}")

        log_term = math.log(1.0 / target_delta)

        best_epsilon = float("inf")

        for alpha, rdp_value in self._rdp_values.items():
            if alpha <= 1.0:
                continue
            # 转换公式
            eps = rdp_value + log_term / (alpha - 1.0)
            best_epsilon = min(best_epsilon, eps)

        return best_epsilon

    def get_privacy_spent(self) -> Dict[str, Any]:
        """
        获取已消耗的隐私预算

        Returns:
            包含 epsilon、delta、steps 等信息的字典
        """
        epsilon = self.get_epsilon()
        return {
            "epsilon": epsilon,
            "delta": self._delta,
            "steps": self._steps,
            "noise_multiplier": self._noise_multiplier,
            "sample_rate": self._sample_rate,
        }

    def _get_alpha_grid(self) -> List[float]:
        """
        生成 alpha 搜索网格

        使用对数间隔的 alpha 值，在 [2, 256] 范围内。

        Returns:
            alpha 值列表
        """
        low, high = self.DEFAULT_ALPHA_RANGE
        steps = self.DEFAULT_ALPHA_STEPS
        # 对数间隔
        log_low = math.log(low)
        log_high = math.log(high)
        return [math.exp(log_low + (log_high - log_low) * i / (steps - 1)) for i in range(steps)]

    def reset(self) -> None:
        """重置记账器"""
        self._steps = 0
        self._rdp_values.clear()


# ============== 隐私预算管理 ==============

class PrivacyBudget:
    """
    隐私预算管理器

    跟踪和分配隐私预算，防止超支。
    支持按轮次、按客户端、按操作类型分配预算。

    隐私预算一旦耗尽，应停止训练或降低噪声精度。
    """

    def __init__(
        self,
        total_epsilon: float = 8.0,
        total_delta: float = 1e-5,
        num_rounds: int = 100,
    ) -> None:
        if total_epsilon <= 0:
            raise ValueError(f"total_epsilon 必须 > 0，当前值: {total_epsilon}")
        if not (0 < total_delta < 1):
            raise ValueError(f"total_delta 必须在 (0, 1) 之间，当前值: {total_delta}")

        self._total_epsilon = total_epsilon
        self._total_delta = total_delta
        self._num_rounds = num_rounds

        self._spent_epsilon: float = 0.0
        self._spent_delta: float = 0.0
        self._current_round: int = 0

        # 每轮的预算分配记录
        self._round_budgets: Dict[int, Dict[str, float]] = {}

        # 客户端级别的预算跟踪
        self._client_budgets: Dict[str, Dict[str, float]] = {}

    @property
    def total_epsilon(self) -> float:
        return self._total_epsilon

    @property
    def total_delta(self) -> float:
        return self._total_delta

    @property
    def spent_epsilon(self) -> float:
        return self._spent_epsilon

    @property
    def spent_delta(self) -> float:
        return self._spent_delta

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self._total_epsilon - self._spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self._total_delta - self._spent_delta)

    @property
    def budget_exhausted(self) -> bool:
        return self._spent_epsilon >= self._total_epsilon

    def allocate_round_budget(self, round_id: int) -> Dict[str, float]:
        """
        为一轮训练分配隐私预算

        使用均匀分配策略:
            epsilon_per_round = (total_epsilon - spent) / remaining_rounds

        Args:
            round_id: 轮次编号

        Returns:
            分配的预算 {"epsilon": ..., "delta": ...}
        """
        self._current_round = round_id
        remaining_rounds = max(1, self._num_rounds - round_id)

        eps_budget = self.remaining_epsilon / remaining_rounds
        delta_budget = self.remaining_delta / remaining_rounds

        self._round_budgets[round_id] = {
            "epsilon": eps_budget,
            "delta": delta_budget,
        }

        return {"epsilon": eps_budget, "delta": delta_budget}

    def spend(self, epsilon: float, delta: float, round_id: Optional[int] = None) -> bool:
        """
        消耗隐私预算

        Args:
            epsilon: 消耗的 epsilon
            delta: 消耗的 delta
            round_id: 关联的轮次（可选）

        Returns:
            是否成功消耗（预算是否充足）
        """
        if self._spent_epsilon + epsilon > self._total_epsilon:
            logger.warning(
                f"隐私预算不足: 需要 {epsilon:.4f}, "
                f"剩余 {self.remaining_epsilon:.4f}"
            )
            return False

        self._spent_epsilon += epsilon
        self._spent_delta += delta

        if round_id is not None and round_id in self._round_budgets:
            self._round_budgets[round_id]["spent_epsilon"] = (
                self._round_budgets[round_id].get("spent_epsilon", 0.0) + epsilon
            )
            self._round_budgets[round_id]["spent_delta"] = (
                self._round_budgets[round_id].get("spent_delta", 0.0) + delta
            )

        return True

    def allocate_client_budget(
        self, client_id: str, epsilon: float, delta: float
    ) -> bool:
        """
        为客户端分配预算

        Args:
            client_id: 客户端ID
            epsilon: 分配的 epsilon
            delta: 分配的 delta

        Returns:
            是否分配成功
        """
        if self.remaining_epsilon < epsilon:
            return False

        self._client_budgets[client_id] = {
            "allocated_epsilon": epsilon,
            "allocated_delta": delta,
            "spent_epsilon": 0.0,
            "spent_delta": 0.0,
        }
        return True

    def get_client_status(self, client_id: str) -> Optional[Dict[str, float]]:
        """获取客户端预算状态"""
        return self._client_budgets.get(client_id)

    def get_summary(self) -> Dict[str, Any]:
        """
        获取预算使用摘要

        Returns:
            包含预算使用情况的字典
        """
        return {
            "total_epsilon": self._total_epsilon,
            "total_delta": self._total_delta,
            "spent_epsilon": self._spent_epsilon,
            "spent_delta": self._spent_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "utilization": self._spent_epsilon / self._total_epsilon if self._total_epsilon > 0 else 0.0,
            "current_round": self._current_round,
            "budget_exhausted": self.budget_exhausted,
        }

    def reset(self) -> None:
        """重置预算管理器"""
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._current_round = 0
        self._round_budgets.clear()
        self._client_budgets.clear()


# ============== 自适应噪声调整 ==============

class AdaptiveNoise:
    """
    自适应噪声调整器

    根据剩余隐私预算动态调整噪声大小。
    当剩余预算充足时使用较小噪声（提高精度），
    当预算接近耗尽时增大噪声（保护隐私）。

    调整策略:
    - 线性衰减: noise_multiplier = base * (total_eps / remaining_eps)^gamma
    - 其中 gamma 控制衰减速率
    """

    def __init__(
        self,
        base_noise_multiplier: float = 1.1,
        min_noise_multiplier: float = 0.5,
        max_noise_multiplier: float = 10.0,
        decay_rate: float = 1.0,
    ) -> None:
        """
        Args:
            base_noise_multiplier: 基础噪声乘数
            min_noise_multiplier: 最小噪声乘数下界
            max_noise_multiplier: 最大噪声乘数上界
            decay_rate: 衰减速率 gamma，控制预算耗尽时噪声增长速度
        """
        if base_noise_multiplier <= 0:
            raise ValueError(f"base_noise_multiplier 必须 > 0")
        self._base = base_noise_multiplier
        self._min = min_noise_multiplier
        self._max = max_noise_multiplier
        self._decay_rate = decay_rate

    def compute_noise_multiplier(
        self,
        spent_epsilon: float,
        total_epsilon: float,
    ) -> float:
        """
        根据已消耗的预算计算当前噪声乘数

        公式:
            noise_mult = base * (total_eps / max(remaining_eps, eps_min))^gamma

        其中 eps_min 是一个很小的值防止除零。

        Args:
            spent_epsilon: 已消耗的 epsilon
            total_epsilon: 总 epsilon

        Returns:
            调整后的噪声乘数
        """
        remaining = max(total_epsilon - spent_epsilon, 1e-8)
        ratio = total_epsilon / remaining
        adjusted = self._base * (ratio ** self._decay_rate)

        # 限制在 [min, max] 范围内
        return max(self._min, min(self._max, adjusted))

    def compute_noise_multiplier_for_round(
        self,
        current_round: int,
        total_rounds: int,
    ) -> float:
        """
        根据当前轮次计算噪声乘数

        使用轮次比例来估计预算消耗进度。

        Args:
            current_round: 当前轮次
            total_rounds: 总轮次

        Returns:
            调整后的噪声乘数
        """
        if total_rounds <= 0:
            return self._base

        progress = current_round / total_rounds
        spent_estimate = progress  # 假设均匀消耗
        return self.compute_noise_multiplier(spent_estimate, 1.0)


# ============== 隐私预算组合定理 ==============

class ComposePrivacy:
    """
    隐私预算组合定理

    实现多种隐私预算组合方法，用于计算多个差分隐私机制的组合隐私保证。

    支持的组合定理:
    1. 基础串行组合 (Basic Serial Composition)
    2. 高级组合 (Advanced Composition)
    3. 并行组合 (Parallel Composition)
    4. RDP 组合 (RDP Composition)
    """

    @staticmethod
    def basic_serial_composition(
        epsilons: List[float],
        deltas: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        基础串行组合定理

        对于 k 个 (epsilon_i, delta_i)-DP 机制的串行组合:
            epsilon_total = sum(epsilon_i)
            delta_total = sum(delta_i)

        这是最简单但最松的组合界。

        Args:
            epsilons: 各机制的 epsilon 列表
            deltas: 各机制的 delta 列表（可选，默认全为 0）

        Returns:
            (epsilon_total, delta_total)
        """
        if not epsilons:
            return (0.0, 0.0)

        total_eps = sum(epsilons)
        total_delta = sum(deltas) if deltas else 0.0

        return (total_eps, total_delta)

    @staticmethod
    def advanced_composition(
        epsilons: List[float],
        delta: float,
        k: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        高级组合定理

        对于 k 个 epsilon-DP 机制（相同的 epsilon），组合后满足:
            epsilon_total = epsilon * sqrt(2*k * ln(1/delta')) + k * epsilon * (e^epsilon - 1)

        其中 delta' 是组合后的 delta 参数。

        简化版本（当所有 epsilon_i 相同时）:
            epsilon_total ≈ epsilon * sqrt(2*k*ln(1/delta)) + k*epsilon^2

        Args:
            epsilons: 各机制的 epsilon 列表
            delta: 目标 delta
            k: 机制数量（默认为 len(epsilons)）

        Returns:
            (epsilon_total, delta)
        """
        if not epsilons:
            return (0.0, delta)

        if k is None:
            k = len(epsilons)

        # 使用平均 epsilon
        avg_eps = sum(epsilons) / len(epsilons)

        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta 必须在 (0, 1) 之间，当前值: {delta}")

        # 高级组合公式
        term1 = avg_eps * math.sqrt(2.0 * k * math.log(1.0 / delta))
        term2 = k * avg_eps * avg_eps  # (e^eps - 1) ≈ eps 当 eps 较小时

        total_eps = term1 + term2
        return (total_eps, delta)

    @staticmethod
    def parallel_composition(
        epsilon: float,
        delta: float,
        k: int,
    ) -> Tuple[float, float]:
        """
        并行组合定理

        如果 k 个机制作用于不相交的数据子集，则:
            epsilon_total = max(epsilon_i)
            delta_total = sum(delta_i)

        这是组合定理中最紧的界。

        Args:
            epsilon: 每个 epsilon-DP 机制的 epsilon（假设相同）
            delta: 每个机制的 delta
            k: 机制数量

        Returns:
            (epsilon, k * delta)
        """
        return (epsilon, k * delta)

    @staticmethod
    def rdp_composition(
        rdp_values: List[float],
        alphas: List[float],
        delta: float,
    ) -> Tuple[float, float]:
        """
        RDP 组合

        对于多个 RDP 机制的组合:
            RDP_total(alpha) = sum(RDP_i(alpha))

        然后转换为 (epsilon, delta)-DP:
            epsilon = min_alpha [ RDP_total(alpha) + ln(1/delta) / (alpha - 1) ]

        Args:
            rdp_values: 各 alpha 阶的累积 RDP 值列表
            alphas: 对应的 alpha 值列表
            delta: 目标 delta

        Returns:
            (epsilon, delta)
        """
        if not rdp_values or not alphas:
            return (0.0, delta)

        if len(rdp_values) != len(alphas):
            raise ValueError("rdp_values 和 alphas 长度必须相同")

        log_term = math.log(1.0 / delta) if delta > 0 else float("inf")
        best_epsilon = float("inf")

        for alpha, rdp_val in zip(alphas, rdp_values):
            if alpha <= 1.0:
                continue
            eps = rdp_val + log_term / (alpha - 1.0)
            best_epsilon = min(best_epsilon, eps)

        return (best_epsilon, delta)

    @staticmethod
    def moments_accountant_gaussian(
        noise_multiplier: float,
        steps: int,
        delta: float,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        """
        基于矩会计的高斯机制组合

        使用 RDP 框架计算高斯机制在多步训练后的隐私保证。

        对于 sigma（噪声乘数），q（采样率），T（步数）:
            RDP(alpha) = T * q^2 * alpha / (2 * sigma^2) * alpha / (alpha - 1)

        然后 alpha 最优搜索得到 epsilon。

        Args:
            noise_multiplier: 噪声乘数
            steps: 训练步数
            delta: 目标 delta
            sample_rate: 采样率

        Returns:
            (epsilon, delta)
        """
        if steps == 0:
            return (0.0, delta)

        sigma = noise_multiplier
        q = sample_rate
        log_term = math.log(1.0 / delta)

        best_epsilon = float("inf")

        # 搜索最优 alpha
        for i in range(1, 65):
            alpha = 2.0 + (256.0 - 2.0) * (i - 1) / 63.0

            if alpha <= 1.0:
                continue

            # RDP per step
            if q < 1.0:
                rdp_per_step = (q * q * alpha * alpha) / (2.0 * sigma * sigma * (alpha - 1.0))
            else:
                rdp_per_step = alpha / (2.0 * sigma * sigma)

            total_rdp = rdp_per_step * steps
            eps = total_rdp + log_term / (alpha - 1.0)
            best_epsilon = min(best_epsilon, eps)

        return (best_epsilon, delta)


# ============== 差分隐私注入器（主类） ==============

class DPInjector:
    """
    差分隐私注入器

    组合梯度裁剪、噪声添加和隐私记账功能，
    为联邦学习提供端到端的差分隐私保护。

    使用流程:
        1. 初始化 DPInjector（配置 epsilon、delta 等）
        2. 对每个梯度更新调用 inject() 方法
        3. 自动完成裁剪 -> 加噪 -> 记账

    示例:
        config = DPConfig(epsilon=8.0, delta=1e-5, noise_multiplier=1.1)
        injector = DPInjector(config)
        clipped_noised = injector.inject(gradient)
    """

    def __init__(
        self,
        config: Optional[DPConfig] = None,
        budget: Optional[PrivacyBudget] = None,
        adaptive_noise: Optional[AdaptiveNoise] = None,
    ) -> None:
        """
        Args:
            config: 差分隐私配置
            budget: 隐私预算管理器（可选）
            adaptive_noise: 自适应噪声调整器（可选）
        """
        self._config = config or DPConfig()
        self._budget = budget
        self._adaptive_noise = adaptive_noise

        # 初始化组件
        self._clipper = GradientClipper(
            max_grad_norm=self._config.max_grad_norm,
            clip_per_layer=self._config.clip_per_layer,
        )

        sensitivity = 2.0 * self._config.max_grad_norm  # L2 敏感度

        if self._config.mechanism == "gaussian":
            self._mechanism = GaussianMechanism(
                epsilon=self._config.epsilon,
                delta=self._config.delta,
                sensitivity=sensitivity,
            )
        else:
            self._mechanism = LaplaceMechanism(
                epsilon=self._config.epsilon,
                sensitivity=sensitivity,
            )

        self._accountant = RDPAccountant(
            noise_multiplier=self._config.noise_multiplier,
            max_grad_norm=self._config.max_grad_norm,
            delta=self._config.delta,
        )

        self._total_steps: int = 0
        self._current_noise_multiplier: float = self._config.noise_multiplier

    @property
    def config(self) -> DPConfig:
        return self._config

    @property
    def steps(self) -> int:
        return self._total_steps

    def inject(
        self,
        gradient: List[float],
        noise_multiplier: Optional[float] = None,
    ) -> List[float]:
        """
        对梯度注入差分隐私噪声

        执行流程:
        1. 梯度裁剪（L2范数裁剪到 max_grad_norm）
        2. 添加校准噪声（高斯或拉普拉斯）
        3. 更新隐私记账

        Args:
            gradient: 原始梯度向量
            noise_multiplier: 可选的自定义噪声乘数

        Returns:
            裁剪并加噪后的梯度
        """
        if not gradient:
            return gradient

        # 检查预算
        if self._budget and self._budget.budget_exhausted:
            logger.warning("隐私预算已耗尽，跳过噪声注入")
            return gradient

        # Step 1: 梯度裁剪
        clipped = self._clipper.clip_gradient(gradient)

        # Step 2: 确定噪声乘数
        nm = noise_multiplier
        if nm is None:
            if self._adaptive_noise is not None and self._budget is not None:
                nm = self._adaptive_noise.compute_noise_multiplier(
                    self._budget.spent_epsilon,
                    self._budget.total_epsilon,
                )
            else:
                nm = self._config.noise_multiplier

        self._current_noise_multiplier = nm

        # Step 3: 添加噪声
        if isinstance(self._mechanism, GaussianMechanism):
            noised = self._mechanism.add_noise_with_multiplier(
                clipped, nm, self._config.max_grad_norm
            )
        else:
            noised = self._mechanism.add_noise_vector(clipped)

        # Step 4: 更新记账
        self._accountant.accumulate_step()
        self._total_steps += 1

        # Step 5: 更新预算
        if self._budget is not None:
            # 使用 RDP 记账器计算实际消耗
            spent = self._accountant.get_privacy_spent()
            if self._total_steps == 1:
                self._budget.spend(spent["epsilon"], spent["delta"])
            else:
                # 增量更新
                prev_eps = self._accountant.get_epsilon() - spent["epsilon"] / self._total_steps
                self._budget.spend(spent["epsilon"] - prev_eps, spent["delta"] / self._total_steps)

        return noised

    def inject_layers(
        self,
        layers: List[List[float]],
        noise_multiplier: Optional[float] = None,
    ) -> List[List[float]]:
        """
        对多层梯度注入差分隐私噪声

        Args:
            layers: 多层梯度列表
            noise_multiplier: 可选的自定义噪声乘数

        Returns:
            裁剪并加噪后的多层梯度
        """
        if not layers:
            return layers

        # 全局裁剪
        clipped = self._clipper.clip_layer_gradients(layers)

        # 确定噪声乘数
        nm = noise_multiplier or self._config.noise_multiplier

        # 对每层添加噪声
        if isinstance(self._mechanism, GaussianMechanism):
            noised = [
                self._mechanism.add_noise_with_multiplier(layer, nm, self._config.max_grad_norm)
                for layer in clipped
            ]
        else:
            noised = [self._mechanism.add_noise_vector(layer) for layer in clipped]

        # 更新记账
        self._accountant.accumulate_step()
        self._total_steps += 1

        return noised

    def get_privacy_report(self) -> Dict[str, Any]:
        """
        获取隐私报告

        Returns:
            包含隐私保证信息的字典
        """
        accountant_report = self._accountant.get_privacy_spent()
        report: Dict[str, Any] = {
            "mechanism": self._config.mechanism,
            "total_steps": self._total_steps,
            "noise_multiplier": self._current_noise_multiplier,
            "max_grad_norm": self._config.max_grad_norm,
            "clip_ratio": self._clipper.get_clip_ratio(),
            "average_norm": self._clipper.get_average_norm(),
            **accountant_report,
        }

        if self._budget is not None:
            report["budget"] = self._budget.get_summary()

        return report

    def reset(self) -> None:
        """重置注入器状态"""
        self._total_steps = 0
        self._accountant.reset()
        self._clipper.reset_history()
        if self._budget is not None:
            self._budget.reset()


# ============== 模块入口 ==============

if __name__ == "__main__":
    print("=== 差分隐私注入器演示 ===\n")

    # 1. 梯度裁剪
    print("1. 梯度裁剪:")
    clipper = GradientClipper(max_grad_norm=1.0)
    grad = [0.5, 1.5, -2.0, 0.8, 3.0]
    clipped = clipper.clip_gradient(grad)
    print(f"   原始梯度范数: {GradientClipper._compute_l2_norm(grad):.4f}")
    print(f"   裁剪后范数:   {GradientClipper._compute_l2_norm(clipped):.4f}")

    # 2. 高斯机制
    print("\n2. 高斯机制:")
    gm = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
    print(f"   sigma = {gm.compute_sigma():.4f}")
    val = 10.0
    noised_vals = [gm.add_noise(val) for _ in range(5)]
    print(f"   原始值: {val}, 加噪值: {[f'{v:.4f}' for v in noised_vals]}")

    # 3. 拉普拉斯机制
    print("\n3. 拉普拉斯机制:")
    lm = LaplaceMechanism(epsilon=1.0, sensitivity=2.0)
    print(f"   scale b = {lm.compute_scale():.4f}")
    noised_vals_l = [lm.add_noise(val) for _ in range(5)]
    print(f"   原始值: {val}, 加噪值: {[f'{v:.4f}' for v in noised_vals_l]}")

    # 4. RDP 记账
    print("\n4. RDP 隐私记账:")
    accountant = RDPAccountant(noise_multiplier=1.1, delta=1e-5)
    for step in range(1, 101):
        accountant.accumulate_step()
        if step in (10, 50, 100):
            eps = accountant.get_epsilon()
            print(f"   步数 {step:3d}: epsilon = {eps:.4f}")

    # 5. 隐私预算组合
    print("\n5. 隐私预算组合定理:")
    eps_list = [1.0, 2.0, 1.5, 0.5]
    basic = ComposePrivacy.basic_serial_composition(eps_list)
    advanced = ComposePrivacy.advanced_composition(eps_list, delta=1e-5)
    parallel = ComposePrivacy.parallel_composition(epsilon=1.0, delta=1e-6, k=4)
    print(f"   串行组合: epsilon = {basic[0]:.4f}")
    print(f"   高级组合: epsilon = {advanced[0]:.4f}")
    print(f"   并行组合: epsilon = {parallel[0]:.4f}, delta = {parallel[1]:.2e}")

    # 6. DP 注入器
    print("\n6. DP 注入器:")
    config = DPConfig(epsilon=8.0, delta=1e-5, noise_multiplier=1.1, max_grad_norm=1.0)
    injector = DPInjector(config)
    gradient = [0.3, -0.5, 0.8, -0.2, 0.1]
    for i in range(5):
        result = injector.inject(gradient)
        print(f"   第 {i+1} 次注入: {[f'{v:.4f}' for v in result]}")
    report = injector.get_privacy_report()
    print(f"   隐私报告: epsilon={report['epsilon']:.4f}, steps={report['total_steps']}")

    print("\n=== 演示完成 ===")
