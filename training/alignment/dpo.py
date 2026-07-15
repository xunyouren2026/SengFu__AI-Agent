"""
DPO (Direct Preference Optimization) 完整实现
=============================================
纯 Python 实现，仅依赖 math/random，无外部依赖。

实现以下算法:
  - DPO  (Direct Preference Optimization)
  - IPO  (Identity Preference Optimization)
  - KTO  (Kahneman-Tversky Optimization)
  - ORPO (Odds Ratio Preference Optimization)
  - SimPO (Simple Preference Optimization)

数学符号约定:
  π_θ   : 可训练策略模型
  π_ref : 冻结参考策略
  y_w   : 被偏好的回答 (chosen)
  y_l   : 被拒绝的回答 (rejected)
  x     : 提示 (prompt)
  β     : KL 散度惩罚系数
  σ     : sigmoid 函数
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any


# ============================================================================
#  基础数学工具函数
# ============================================================================

def _zeros(rows: int, cols: int) -> List[List[float]]:
    """创建 rows x cols 的零矩阵"""
    return [[0.0] * cols for _ in range(rows)]


def _randn(rows: int, cols: int, std: float = 0.02) -> List[List[float]]:
    """创建 rows x cols 的正态随机矩阵 (Box-Muller 变换)"""
    mat = _zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            u1 = random.random() or 1e-10
            u2 = random.random()
            mat[i][j] = std * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mat


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵乘法 A (m x k) @ B (k x n) -> (m x n)"""
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    C = _zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def _transpose(A: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def _add_mat(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """逐元素矩阵加法"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _scale_mat(s: float, A: List[List[float]]) -> List[List[float]]:
    """标量乘矩阵"""
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    """矩阵 (m x n) 乘向量 (n,) -> (m,)"""
    m = len(A)
    n = len(v)
    result = [0.0] * m
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        result[i] = s
    return result


def _softmax(logits: List[float]) -> List[float]:
    """数值稳定的 softmax"""
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _log_softmax(logits: List[float]) -> List[float]:
    """数值稳定的 log-softmax"""
    max_val = max(logits)
    shifted = [x - max_val for x in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    return [s - log_sum_exp for s in shifted]


def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def _log_sigmoid(x: float) -> float:
    """数值稳定的 log(sigmoid(x))"""
    if x >= 0:
        return -math.log(1.0 + math.exp(-x))
    else:
        return x - math.log(1.0 + math.exp(x))


def _tanh(x: float) -> float:
    """tanh 激活函数"""
    return math.tanh(x)


def _vec_tanh(v: List[float]) -> List[float]:
    return [math.tanh(x) for x in v]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================================
#  简单语言模型 (纯 Python 前馈 LM)
# ============================================================================

class SimpleLM:
    """
    简单前馈语言模型，用于 DPO 演示。

    结构: token_embedding -> mean_pool -> hidden (tanh) -> output logits
    参数:
      embed: (vocab_size, embed_dim)
      W_hidden: (embed_dim, hidden_dim)
      b_hidden: (hidden_dim,)
      W_out: (hidden_dim, vocab_size)
      b_out: (vocab_size,)
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 32,
                 hidden_dim: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        rng = random.Random(seed)

        # 初始化参数
        self.embed = _randn(vocab_size, embed_dim, std=0.02)
        self.W_hidden = _randn(embed_dim, hidden_dim, std=0.02)
        self.b_hidden = [0.0] * hidden_dim
        self.W_out = _randn(hidden_dim, vocab_size, std=0.02)
        self.b_out = [0.0] * vocab_size

        # 梯度缓冲
        self.grad_embed = _zeros(vocab_size, embed_dim)
        self.grad_W_hidden = _zeros(embed_dim, hidden_dim)
        self.grad_b_hidden = [0.0] * hidden_dim
        self.grad_W_out = _zeros(hidden_dim, vocab_size)
        self.grad_b_out = [0.0] * vocab_size

        # 缓存前向传播中间值
        self._cache: Dict[str, Any] = {}

    def _embed_tokens(self, token_ids: List[int]) -> List[float]:
        """对 token 序列做 embedding 平均池化"""
        if not token_ids:
            return [0.0] * self.embed_dim
        pooled = [0.0] * self.embed_dim
        for tid in token_ids:
            for d in range(self.embed_dim):
                pooled[d] += self.embed[tid][d]
        n = len(token_ids)
        return [p / n for p in pooled]

    def forward(self, token_ids: List[int]) -> List[float]:
        """
        前向传播: token_ids -> logits (vocab_size,)

        Returns:
          logits: 长度为 vocab_size 的对数几率向量
        """
        # 1. Embedding + mean pool
        pooled = self._embed_tokens(token_ids)

        # 2. Hidden layer: W_hidden^T @ pooled + b_hidden -> tanh
        hidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = self.b_hidden[j]
            for d in range(self.embed_dim):
                s += pooled[d] * self.W_hidden[d][j]
            hidden[j] = _tanh(s)

        # 3. Output layer: W_out^T @ hidden + b_out -> logits
        logits = [0.0] * self.vocab_size
        for v in range(self.vocab_size):
            s = self.b_out[v]
            for j in range(self.hidden_dim):
                s += hidden[j] * self.W_out[j][v]
            logits[v] = s

        # 缓存中间值供反向传播使用
        self._cache = {
            "token_ids": token_ids,
            "pooled": pooled,
            "hidden": hidden,
            "logits": logits,
        }
        return logits

    def log_probs(self, token_ids: List[int]) -> List[float]:
        """
        计算给定 token 序列的对数概率。

        对于序列 [t_0, t_1, ..., t_{T-1}]，计算:
          log p(t_0|t_<0) + log p(t_1|t_<1) + ... + log p(t_{T-1}|t_<T-1)

        使用自回归方式: 逐步以 [t_0..t_{k-1}] 为输入预测 t_k。
        """
        total_log_prob = 0.0
        for k in range(1, len(token_ids)):
            prefix = token_ids[:k]
            logits = self.forward(prefix)
            lp = _log_softmax(logits)
            total_log_prob += lp[token_ids[k]]
        return total_log_prob

    def sequence_log_prob(self, token_ids: List[int]) -> float:
        """
        计算整个序列的对数概率 (自回归)。
        等价于 log π_θ(y|x)，其中 x 为空或前缀。
        """
        if len(token_ids) <= 1:
            return 0.0
        return self.log_probs(token_ids)

    def _backward_from_logits_grad(self, d_logits: List[float]) -> None:
        """
        从 logits 的梯度反向传播，更新所有参数梯度。

        d_logits: (vocab_size,) 对 logits 的梯度
        """
        hidden = self._cache["hidden"]
        pooled = self._cache["pooled"]
        token_ids = self._cache["token_ids"]

        # --- 输出层梯度 ---
        # d_W_out[j][v] += hidden[j] * d_logits[v]
        # d_b_out[v] += d_logits[v]
        # d_hidden[j] += W_out[j][v] * d_logits[v]
        d_hidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            for v in range(self.vocab_size):
                self.grad_W_out[j][v] += hidden[j] * d_logits[v]
                self.grad_b_out[v] += d_logits[v]
                d_hidden[j] += self.W_out[j][v] * d_logits[v]

        # --- tanh 反向传播 ---
        # d_pre_hidden[j] = d_hidden[j] * (1 - hidden[j]^2)
        d_pre_hidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            d_pre_hidden[j] = d_hidden[j] * (1.0 - hidden[j] ** 2)

        # --- 隐藏层梯度 ---
        # d_W_hidden[d][j] += pooled[d] * d_pre_hidden[j]
        # d_b_hidden[j] += d_pre_hidden[j]
        # d_pooled[d] += W_hidden[d][j] * d_pre_hidden[j]
        d_pooled = [0.0] * self.embed_dim
        for d in range(self.embed_dim):
            for j in range(self.hidden_dim):
                self.grad_W_hidden[d][j] += pooled[d] * d_pre_hidden[j]
                self.grad_b_hidden[j] += d_pre_hidden[j]
                d_pooled[d] += self.W_hidden[d][j] * d_pre_hidden[j]

        # --- Embedding 梯度 (mean pooling 反向) ---
        n = max(len(token_ids), 1)
        grad_scale = 1.0 / n
        for tid in token_ids:
            for d in range(self.embed_dim):
                self.grad_embed[tid][d] += d_pooled[d] * grad_scale

    def backward_on_token(self, target_token: int, grad_scale: float = 1.0) -> None:
        """
        对单个目标 token 计算负对数似然梯度并反向传播。

        loss = -log_softmax(logits)[target_token]
        d_logits[v] = softmax(logits)[v] - (v == target_token)
        """
        logits = self._cache["logits"]
        probs = _softmax(logits)
        d_logits = list(probs)
        d_logits[target_token] -= 1.0
        d_logits = [g * grad_scale for g in d_logits]
        self._backward_from_logits_grad(d_logits)

    def backward_on_log_prob_ratio(self, ref_log_prob: float,
                                    policy_log_prob: float,
                                    grad_scale: float = 1.0) -> None:
        """
        对 log π_θ(y|x) 的梯度进行反向传播。

        给定 DPO 损失中 log π_θ(y|x) 的梯度为 grad_scale，
        需要对每个自回归步骤传播。

        d_loss/d_logits_t[v] = softmax(logits_t)[v] - δ(v, t_k)
        然后乘以 grad_scale。
        """
        token_ids = self._cache.get("token_ids", [])
        # 对序列中每个 token 位置做反向传播
        # 重新计算每一步的前向传播
        for k in range(1, len(token_ids)):
            prefix = token_ids[:k]
            logits = self.forward(prefix)
            probs = _softmax(logits)
            d_logits = list(probs)
            d_logits[token_ids[k]] -= 1.0
            d_logits = [g * grad_scale for g in d_logits]
            self._backward_from_logits_grad(d_logits)

    def zero_grad(self) -> None:
        """清零所有梯度"""
        vs, ed, hd = self.vocab_size, self.embed_dim, self.hidden_dim
        self.grad_embed = _zeros(vs, ed)
        self.grad_W_hidden = _zeros(ed, hd)
        self.grad_b_hidden = [0.0] * hd
        self.grad_W_out = _zeros(hd, vs)
        self.grad_b_out = [0.0] * vs

    def step(self, lr: float, max_grad_norm: float = 1.0) -> float:
        """
        SGD 参数更新，返回梯度范数。

        包含梯度裁剪以防止梯度爆炸。
        """
        # 计算总梯度范数
        total_norm_sq = 0.0
        for i in range(self.vocab_size):
            for j in range(self.embed_dim):
                total_norm_sq += self.grad_embed[i][j] ** 2
        for i in range(self.embed_dim):
            for j in range(self.hidden_dim):
                total_norm_sq += self.grad_W_hidden[i][j] ** 2
        for j in range(self.hidden_dim):
            total_norm_sq += self.grad_b_hidden[j] ** 2
        for j in range(self.hidden_dim):
            for v in range(self.vocab_size):
                total_norm_sq += self.grad_W_out[j][v] ** 2
        for v in range(self.vocab_size):
            total_norm_sq += self.grad_b_out[v] ** 2

        total_norm = math.sqrt(total_norm_sq) + 1e-12
        clip_coef = min(1.0, max_grad_norm / total_norm) if total_norm > max_grad_norm else 1.0

        # 应用梯度裁剪并更新参数
        for i in range(self.vocab_size):
            for j in range(self.embed_dim):
                self.embed[i][j] -= lr * clip_coef * self.grad_embed[i][j]
        for i in range(self.embed_dim):
            for j in range(self.hidden_dim):
                self.W_hidden[i][j] -= lr * clip_coef * self.grad_W_hidden[i][j]
        for j in range(self.hidden_dim):
            self.b_hidden[j] -= lr * clip_coef * self.grad_b_hidden[j]
        for j in range(self.hidden_dim):
            for v in range(self.vocab_size):
                self.W_out[j][v] -= lr * clip_coef * self.grad_W_out[j][v]
        for v in range(self.vocab_size):
            self.b_out[v] -= lr * clip_coef * self.grad_b_out[v]

        return total_norm

    def get_state(self) -> Dict[str, Any]:
        """获取模型参数的深拷贝"""
        return {
            "embed": [row[:] for row in self.embed],
            "W_hidden": [row[:] for row in self.W_hidden],
            "b_hidden": self.b_hidden[:],
            "W_out": [row[:] for row in self.W_out],
            "b_out": self.b_out[:],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """从状态字典加载参数"""
        self.embed = [row[:] for row in state["embed"]]
        self.W_hidden = [row[:] for row in state["W_hidden"]]
        self.b_hidden = state["b_hidden"][:]
        self.W_out = [row[:] for row in state["W_out"]]
        self.b_out = state["b_out"][:]


# ============================================================================
#  1. DPOConfig - 配置数据类
# ============================================================================

@dataclass
class DPOConfig:
    """
    DPO 训练配置。

    Attributes:
        beta: KL 散度惩罚系数 β。控制策略与参考模型的偏离程度。
              较大的 β 使策略更接近参考模型，较小的 β 允许更大的偏离。
              经验值通常在 0.1 ~ 0.5 之间。
        learning_rate: 策略模型的学习率。
        batch_size: 每次更新的批次大小。
        max_length: 输入序列的最大 token 数。
        epochs: 训练轮数。
        max_grad_norm: 梯度裁剪的最大范数。
        vocab_size: 词表大小。
        embed_dim: embedding 维度。
        hidden_dim: 隐藏层维度。
        seed: 随机种子。
        label_smoothing: 标签平滑系数 (0 表示不使用)。
        loss_type: 损失函数类型 ("dpo", "ipo", "kto", "orpo", "simpo")。
    """
    beta: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 4
    max_length: int = 128
    epochs: int = 3
    max_grad_norm: float = 1.0
    vocab_size: int = 256
    embed_dim: int = 32
    hidden_dim: int = 64
    seed: int = 42
    label_smoothing: float = 0.0
    loss_type: str = "dpo"
    # IPO 特有参数
    ipo_beta_sq_penalty: bool = True
    # KTO 特有参数
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0
    # ORPO 特有参数
    orpo_lambda: float = 1.0
    # SimPO 特有参数
    simpo_gamma: float = 0.5


# ============================================================================
#  2. ReferenceModel - 冻结参考策略
# ============================================================================

class ReferenceModel:
    """
    冻结的参考策略模型 π_ref。

    在 DPO 中，参考模型提供基准对数概率 log π_ref(y|x)，
    用于计算策略比率 log(π_θ/π_ref)，从而隐式定义奖励函数:

        r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))

    参考模型在训练过程中不更新参数。
    """

    def __init__(self, model: SimpleLM):
        """
        从一个 SimpleLM 创建冻结的参考模型。

        Args:
            model: 将被冻结复制的语言模型。
        """
        self.model = SimpleLM(
            vocab_size=model.vocab_size,
            embed_dim=model.embed_dim,
            hidden_dim=model.hidden_dim,
        )
        self.model.load_state(model.get_state())
        self.vocab_size = model.vocab_size

    def compute_log_probs(self, token_ids: List[int]) -> float:
        """
        计算参考模型下序列的对数概率。

        log π_ref(y|x) = Σ_{t=1}^{T} log p_ref(y_t | y_{<t}, x)

        Args:
            token_ids: 完整的 token 序列 [x_0, ..., x_m, y_0, ..., y_n]。

        Returns:
            序列的对数概率 (标量)。
        """
        return self.model.sequence_log_prob(token_ids)

    def compute_log_probs_autoregressive(self, token_ids: List[int]) -> List[float]:
        """
        逐步计算每个 token 的对数概率。

        Returns:
            长度为 len(token_ids)-1 的列表，每项为 log p(t_k | t_{<k})。
        """
        log_probs_list = []
        for k in range(1, len(token_ids)):
            prefix = token_ids[:k]
            logits = self.model.forward(prefix)
            lp = _log_softmax(logits)
            log_probs_list.append(lp[token_ids[k]])
        return log_probs_list


# ============================================================================
#  3. DPOLoss - 核心 DPO 损失计算
# ============================================================================

class DPOLoss:
    """
    DPO 损失函数的完整实现。

    核心公式:
        L_DPO = -E_{(x, y_w, y_l)} [
            log σ(β * (log(π_θ(y_w|x) / π_ref(y_w|x))
                      - log(π_θ(y_l|x) / π_ref(y_l|x))))
        ]

    其中:
        σ(z) = 1 / (1 + exp(-z))  为 sigmoid 函数
        β     为 KL 惩罚系数
        y_w   为被偏好的回答 (chosen)
        y_l   为被拒绝的回答 (rejected)

    隐式奖励:
        r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))
    """

    def __init__(self, config: DPOConfig):
        self.beta = config.beta
        self.label_smoothing = config.label_smoothing

    def compute_loss(
        self,
        policy_chosen_logps: float,
        policy_rejected_logps: float,
        ref_chosen_logps: float,
        ref_rejected_logps: float,
    ) -> Tuple[float, float, float]:
        """
        计算单个样本的 DPO 损失。

        Args:
            policy_chosen_logps:  log π_θ(y_w|x)
            policy_rejected_logps: log π_θ(y_l|x)
            ref_chosen_logps:     log π_ref(y_w|x)
            ref_rejected_logps:   log π_ref(y_l|x)

        Returns:
            (loss, chosen_reward, rejected_reward)
            loss: 标量 DPO 损失
            chosen_reward: 隐式奖励 r(x, y_w)
            rejected_reward: 隐式奖励 r(x, y_l)
        """
        # 计算对数比率
        # log(π_θ(y_w|x) / π_ref(y_w|x)) = log π_θ(y_w|x) - log π_ref(y_w|x)
        log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
        log_ratio_rejected = policy_rejected_logps - ref_rejected_logps

        # 隐式奖励: r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))
        chosen_reward = self.beta * log_ratio_chosen
        rejected_reward = self.beta * log_ratio_rejected

        # DPO 损失: -log σ(β * (log_ratio_chosen - log_ratio_rejected))
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)

        if self.label_smoothing > 0:
            # 标签平滑: loss = -(1-ε)*log σ(logits) - ε*log(1-σ(logits))
            # 即: loss = (1-ε)*log(1+exp(-logits)) + ε*log(1+exp(logits))
            loss = (
                (1.0 - self.label_smoothing) * _log_sigmoid(logits)
                + self.label_smoothing * _log_sigmoid(-logits)
            )
            loss = -loss
        else:
            loss = -_log_sigmoid(logits)

        return loss, chosen_reward, rejected_reward

    def compute_policy_kl(
        self,
        policy_logps: float,
        ref_logps: float,
    ) -> float:
        """
        估计策略与参考模型之间的 KL 散度。

        KL(π_θ || π_ref) ≈ E_x[log π_θ(y|x) - log π_ref(y|x)]

        这是 DPO 中隐式正则化项的估计。

        Args:
            policy_logps: log π_θ(y|x)
            ref_logps: log π_ref(y|x)

        Returns:
            KL 散度估计值 (非负)。
        """
        kl = policy_logps - ref_logps
        # KL 散度理论上非负，但由于数值误差可能为微小负值
        return max(kl, 0.0)

    def get_chosen_rewards(
        self,
        policy_chosen_logps: float,
        ref_chosen_logps: float,
    ) -> float:
        """
        提取被偏好回答的隐式奖励。

        r(x, y_w) = β * log(π_θ(y_w|x) / π_ref(y_w|x))

        Args:
            policy_chosen_logps: log π_θ(y_w|x)
            ref_chosen_logps: log π_ref(y_w|x)

        Returns:
            chosen reward 标量值。
        """
        return self.beta * (policy_chosen_logps - ref_chosen_logps)

    def get_rejected_rewards(
        self,
        policy_rejected_logps: float,
        ref_rejected_logps: float,
    ) -> float:
        """
        提取被拒绝回答的隐式奖励。

        r(x, y_l) = β * log(π_θ(y_l|x) / π_ref(y_l|x))

        Args:
            policy_rejected_logps: log π_θ(y_l|x)
            ref_rejected_logps: log π_ref(y_l|x)

        Returns:
            rejected reward 标量值。
        """
        return self.beta * (policy_rejected_logps - ref_rejected_logps)

    def compute_gradient_signal(
        self,
        policy_chosen_logps: float,
        policy_rejected_logps: float,
        ref_chosen_logps: float,
        ref_rejected_logps: float,
    ) -> Tuple[float, float]:
        """
        计算 DPO 损失对 log π_θ 的梯度信号。

        ∂L/∂log π_θ(y_w|x) = -β * σ(-β * Δ)
        ∂L/∂log π_θ(y_l|x) = +β * σ(-β * Δ)

        其中 Δ = log_ratio_chosen - log_ratio_rejected

        Returns:
            (grad_chosen, grad_rejected) 梯度信号。
        """
        log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
        log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
        delta = log_ratio_chosen - log_ratio_rejected
        sigmoid_neg = _sigmoid(-self.beta * delta)

        grad_chosen = -self.beta * sigmoid_neg
        grad_rejected = self.beta * sigmoid_neg

        return grad_chosen, grad_rejected


# ============================================================================
#  4. DPOTrainer - 主 DPO 训练器
# ============================================================================

class DPOTrainer:
    """
    DPO 训练器，实现完整的 DPO 训练流程。

    训练步骤:
        1. 对每个偏好对 (x, y_w, y_l):
           a. 计算 log π_θ(y_w|x), log π_θ(y_l|x)
           b. 计算 log π_ref(y_w|x), log π_ref(y_l|x)
           c. 计算 DPO 损失
           d. 反向传播并更新策略参数
        2. 参考模型参数保持不变

    训练目标:
        min_θ L_DPO(θ) = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x)
                                       - log π_θ(y_l|x)/π_ref(y_l|x)))]
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.beta = config.beta
        self.lr = config.learning_rate

        # 初始化策略模型
        self.policy = SimpleLM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
        )

        # 初始化参考模型 (冻结)
        self.ref_model = ReferenceModel(self.policy)

        # DPO 损失计算器
        self.dpo_loss = DPOLoss(config)

        # 训练日志
        self.log_history: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[int]:
        """简单的字符级 tokenization"""
        token_ids = [ord(c) % self.config.vocab_size for c in text]
        if len(token_ids) > self.config.max_length:
            token_ids = token_ids[:self.config.max_length]
        return token_ids

    def train_step(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict[str, float]:
        """
        执行单个 DPO 训练步骤。

        Args:
            prompt: 输入提示
            chosen: 被偏好的回答
            rejected: 被拒绝的回答

        Returns:
            包含损失和奖励指标的字典。
        """
        # Tokenize
        prompt_ids = self._tokenize(prompt)
        chosen_ids = self._tokenize(chosen)
        rejected_ids = self._tokenize(rejected)

        # 完整序列: prompt + response
        chosen_full = prompt_ids + chosen_ids
        rejected_full = prompt_ids + rejected_ids

        # 计算策略模型的对数概率
        policy_chosen_logps = self.policy.sequence_log_prob(chosen_full)
        policy_rejected_logps = self.policy.sequence_log_prob(rejected_full)

        # 计算参考模型的对数概率
        ref_chosen_logps = self.ref_model.compute_log_probs(chosen_full)
        ref_rejected_logps = self.ref_model.compute_log_probs(rejected_full)

        # 计算 DPO 损失
        loss, chosen_reward, rejected_reward = self.dpo_loss.compute_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

        # 计算梯度信号
        grad_chosen, grad_rejected = self.dpo_loss.compute_gradient_signal(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

        # 反向传播
        self.policy.zero_grad()

        # 对 chosen 序列做反向传播
        if len(chosen_full) > 1:
            for k in range(1, len(chosen_full)):
                prefix = chosen_full[:k]
                logits = self.policy.forward(prefix)
                probs = _softmax(logits)
                d_logits = list(probs)
                d_logits[chosen_full[k]] -= 1.0
                d_logits = [g * grad_chosen for g in d_logits]
                self.policy._backward_from_logits_grad(d_logits)

        # 对 rejected 序列做反向传播
        if len(rejected_full) > 1:
            for k in range(1, len(rejected_full)):
                prefix = rejected_full[:k]
                logits = self.policy.forward(prefix)
                probs = _softmax(logits)
                d_logits = list(probs)
                d_logits[rejected_full[k]] -= 1.0
                d_logits = [g * grad_rejected for g in d_logits]
                self.policy._backward_from_logits_grad(d_logits)

        # 更新参数
        grad_norm = self.policy.step(self.lr, self.config.max_grad_norm)

        # 计算 KL 散度
        kl_chosen = self.dpo_loss.compute_policy_kl(
            policy_chosen_logps, ref_chosen_logps,
        )
        kl_rejected = self.dpo_loss.compute_policy_kl(
            policy_rejected_logps, ref_rejected_logps,
        )

        metrics = {
            "loss": loss,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin": chosen_reward - rejected_reward,
            "kl_chosen": kl_chosen,
            "kl_rejected": kl_rejected,
            "grad_norm": grad_norm,
        }
        return metrics

    def train_epoch(
        self,
        dataset: PreferenceDataset,
    ) -> Dict[str, float]:
        """
        训练一个完整的 epoch。

        Args:
            dataset: 偏好数据集

        Returns:
            该 epoch 的平均指标。
        """
        pairs = dataset.get_pairs()
        random.shuffle(pairs)

        total_loss = 0.0
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0
        total_reward_margin = 0.0
        total_kl = 0.0
        n_steps = 0

        # 批量训练
        for i in range(0, len(pairs), self.config.batch_size):
            batch = pairs[i:i + self.config.batch_size]

            batch_loss = 0.0
            batch_chosen_reward = 0.0
            batch_rejected_reward = 0.0
            batch_reward_margin = 0.0
            batch_kl = 0.0

            for pair in batch:
                metrics = self.train_step(
                    pair.prompt, pair.chosen, pair.rejected,
                )
                batch_loss += metrics["loss"]
                batch_chosen_reward += metrics["chosen_reward"]
                batch_rejected_reward += metrics["rejected_reward"]
                batch_reward_margin += metrics["reward_margin"]
                batch_kl += metrics["kl_chosen"]

            bs = len(batch)
            total_loss += batch_loss / bs
            total_chosen_reward += batch_chosen_reward / bs
            total_rejected_reward += batch_rejected_reward / bs
            total_reward_margin += batch_reward_margin / bs
            total_kl += batch_kl / bs
            n_steps += 1

        avg_metrics = {
            "epoch_loss": total_loss / max(n_steps, 1),
            "epoch_chosen_reward": total_chosen_reward / max(n_steps, 1),
            "epoch_rejected_reward": total_rejected_reward / max(n_steps, 1),
            "epoch_reward_margin": total_reward_margin / max(n_steps, 1),
            "epoch_kl": total_kl / max(n_steps, 1),
        }
        self.log_history.append(avg_metrics)
        return avg_metrics

    def train(
        self,
        dataset: PreferenceDataset,
    ) -> List[Dict[str, float]]:
        """
        完整的 DPO 训练流程。

        Args:
            dataset: 偏好数据集

        Returns:
            每个 epoch 的训练日志列表。
        """
        print(f"[DPO] 开始训练: {len(dataset)} 样本, "
              f"{self.config.epochs} epochs, β={self.beta}, lr={self.lr}")

        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(dataset)
            print(f"[DPO] Epoch {epoch + 1}/{self.config.epochs} | "
                  f"loss={metrics['epoch_loss']:.4f} | "
                  f"reward_margin={metrics['epoch_reward_margin']:.4f} | "
                  f"kl={metrics['epoch_kl']:.4f}")

        print("[DPO] 训练完成")
        return self.log_history

    def generate_comparisons(
        self,
        prompts: List[str],
        num_generations: int = 2,
        temperature: float = 1.0,
        max_tokens: int = 32,
    ) -> List[PreferencePair]:
        """
        从当前策略生成偏好对。

        对每个 prompt 生成多个回答，然后根据对数概率排序，
        将概率最高的作为 chosen，其余作为 rejected。

        Args:
            prompts: 提示列表
            num_generations: 每个 prompt 生成的回答数
            temperature: 采样温度
            max_tokens: 最大生成 token 数

        Returns:
            生成的偏好对列表。
        """
        pairs = []
        for prompt in prompts:
            prompt_ids = self._tokenize(prompt)
            generations = []

            for _ in range(num_generations):
                # 自回归生成
                current = list(prompt_ids)
                for _ in range(max_tokens):
                    logits = self.policy.forward(current)
                    # 温度缩放
                    scaled = [l / max(temperature, 1e-8) for l in logits]
                    probs = _softmax(scaled)

                    # 采样
                    r = random.random()
                    cumulative = 0.0
                    sampled_token = 0
                    for idx, p in enumerate(probs):
                        cumulative += p
                        if cumulative >= r:
                            sampled_token = idx
                            break
                    current.append(sampled_token)

                # 计算对数概率
                logp = self.policy.sequence_log_prob(current)
                generations.append((current, logp))

            # 按对数概率排序
            generations.sort(key=lambda x: x[1], reverse=True)

            if len(generations) >= 2:
                best_tokens = generations[0][0]
                worst_tokens = generations[-1][0]

                # 将 token IDs 转回文本 (简化: 直接用 chr)
                chosen_text = "".join(
                    chr(t) if 32 <= t < 127 else " " for t in best_tokens[len(prompt_ids):]
                )
                rejected_text = "".join(
                    chr(t) if 32 <= t < 127 else " " for t in worst_tokens[len(prompt_ids):]
                )

                pair = PreferencePair(
                    prompt=prompt,
                    chosen=chosen_text,
                    rejected=rejected_text,
                )
                pairs.append(pair)

        return pairs


# ============================================================================
#  5. IPOTrainer - Identity Preference Optimization
# ============================================================================

class IPOTrainer:
    """
    IPO (Identity Preference Optimization) 训练器。

    IPO 通过添加正则化项解决 DPO 在离线数据上的过拟合问题:

        L_IPO = -E[log σ(β * (log_π_ratio_w - log_π_ratio_l))
                   - (β²/2) * (log²_π_ratio_w + log²_π_ratio_l)]

    其中:
        log_π_ratio_w = log(π_θ(y_w|x) / π_ref(y_w|x))
        log_π_ratio_l = log(π_θ(y_l|x) / π_ref(y_l|x))

    相比 DPO，IPO 的正则化项 (β²/2)(log²_π_ratio_w + log²_π_ratio_l)
    防止策略比率无限增大，从而在有限数据上获得更好的泛化性。

    参考: Azar et al., "A General Theoretical Paradigm to Understand Learning
          from Human Feedback", 2023.
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.beta = config.beta
        self.lr = config.learning_rate

        self.policy = SimpleLM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
        )
        self.ref_model = ReferenceModel(self.policy)
        self.log_history: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[int]:
        return [ord(c) % self.config.vocab_size for c in text][:self.config.max_length]

    def compute_ipo_loss(
        self,
        policy_chosen_logps: float,
        policy_rejected_logps: float,
        ref_chosen_logps: float,
        ref_rejected_logps: float,
    ) -> Tuple[float, float, float]:
        """
        计算 IPO 损失。

        L_IPO = -log σ(β * Δ) + (β²/2) * (r_w² + r_l²)

        其中:
            Δ = log_π_ratio_w - log_π_ratio_l
            r_w = log(π_θ(y_w|x) / π_ref(y_w|x))
            r_l = log(π_θ(y_l|x) / π_ref(y_l|x))

        Returns:
            (loss, chosen_reward, rejected_reward)
        """
        log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
        log_ratio_rejected = policy_rejected_logps - ref_rejected_logps

        chosen_reward = self.beta * log_ratio_chosen
        rejected_reward = self.beta * log_ratio_rejected

        delta = self.beta * (log_ratio_chosen - log_ratio_rejected)

        # IPO 损失 = -log σ(β * Δ) + (β²/2) * (r_w² + r_l²)
        logistic_loss = -_log_sigmoid(delta)
        reg_loss = 0.5 * (log_ratio_chosen ** 2 + log_ratio_rejected ** 2)

        loss = logistic_loss + reg_loss

        return loss, chosen_reward, rejected_reward

    def compute_ipo_gradient(
        self,
        policy_chosen_logps: float,
        policy_rejected_logps: float,
        ref_chosen_logps: float,
        ref_rejected_logps: float,
    ) -> Tuple[float, float]:
        """
        计算 IPO 损失对 log π_θ 的梯度。

        ∂L/∂log π_θ(y_w|x) = -β * σ(-β * Δ) + β * r_w
        ∂L/∂log π_θ(y_l|x) = +β * σ(-β * Δ) + β * r_l

        Returns:
            (grad_chosen, grad_rejected)
        """
        log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
        log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
        delta = log_ratio_chosen - log_ratio_rejected
        sigmoid_neg = _sigmoid(-self.beta * delta)

        grad_chosen = -self.beta * sigmoid_neg + self.beta * log_ratio_chosen
        grad_rejected = self.beta * sigmoid_neg + self.beta * log_ratio_rejected

        return grad_chosen, grad_rejected

    def train_step(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict[str, float]:
        """执行单个 IPO 训练步骤。"""
        prompt_ids = self._tokenize(prompt)
        chosen_full = prompt_ids + self._tokenize(chosen)
        rejected_full = prompt_ids + self._tokenize(rejected)

        policy_chosen_logps = self.policy.sequence_log_prob(chosen_full)
        policy_rejected_logps = self.policy.sequence_log_prob(rejected_full)
        ref_chosen_logps = self.ref_model.compute_log_probs(chosen_full)
        ref_rejected_logps = self.ref_model.compute_log_probs(rejected_full)

        loss, chosen_reward, rejected_reward = self.compute_ipo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

        grad_chosen, grad_rejected = self.compute_ipo_gradient(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

        # 反向传播
        self.policy.zero_grad()
        self._backprop_sequence(chosen_full, grad_chosen)
        self._backprop_sequence(rejected_full, grad_rejected)
        grad_norm = self.policy.step(self.lr, self.config.max_grad_norm)

        return {
            "loss": loss,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin": chosen_reward - rejected_reward,
            "grad_norm": grad_norm,
        }

    def _backprop_sequence(self, token_ids: List[int], grad_scale: float) -> None:
        """对序列做反向传播。"""
        if len(token_ids) <= 1:
            return
        for k in range(1, len(token_ids)):
            prefix = token_ids[:k]
            logits = self.policy.forward(prefix)
            probs = _softmax(logits)
            d_logits = list(probs)
            d_logits[token_ids[k]] -= 1.0
            d_logits = [g * grad_scale for g in d_logits]
            self.policy._backward_from_logits_grad(d_logits)

    def train(self, dataset: PreferenceDataset) -> List[Dict[str, float]]:
        """完整的 IPO 训练流程。"""
        print(f"[IPO] 开始训练: {len(dataset)} 样本, "
              f"{self.config.epochs} epochs, β={self.beta}")

        for epoch in range(self.config.epochs):
            pairs = dataset.get_pairs()
            random.shuffle(pairs)
            epoch_loss = 0.0
            epoch_margin = 0.0
            n = 0

            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i:i + self.config.batch_size]
                for pair in batch:
                    m = self.train_step(pair.prompt, pair.chosen, pair.rejected)
                    epoch_loss += m["loss"]
                    epoch_margin += m["reward_margin"]
                    n += 1

            avg_loss = epoch_loss / max(n, 1)
            avg_margin = epoch_margin / max(n, 1)
            self.log_history.append({
                "epoch_loss": avg_loss,
                "epoch_reward_margin": avg_margin,
            })
            print(f"[IPO] Epoch {epoch + 1}/{self.config.epochs} | "
                  f"loss={avg_loss:.4f} | margin={avg_margin:.4f}")

        print("[IPO] 训练完成")
        return self.log_history


# ============================================================================
#  6. KTOTrainer - Kahneman-Tversky Optimization
# ============================================================================

class KTOTrainer:
    """
    KTO (Kahneman-Tversky Optimization) 训练器。

    KTO 不需要配对的偏好数据，而是使用单独的 "期望" 和 "不期望" 标签:

        对于期望输出 y+:
            L_KTO^+ = -log σ(β * (log(π_θ(y+|x)/π_ref(y+|x)) - z_ref))

        对于不期望输出 y-:
            L_KTO^- = -log σ(-β * (log(π_θ(y-|x)/π_ref(y-|x)) - z_ref))

    其中 z_ref 是参考模型下隐式奖励的基线:
        z_ref = E_ref[β * log(π_θ(y|x)/π_ref(y|x))]

    KTO 基于前景理论 (Prospect Theory) 的思想:
    - 期望输出: 最大化收益
    - 不期望输出: 最小化损失

    参考: Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic
          Optimization", 2024.
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.beta = config.beta
        self.lr = config.learning_rate
        self.desirable_weight = config.kto_desirable_weight
        self.undesirable_weight = config.kto_undesirable_weight

        self.policy = SimpleLM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
        )
        self.ref_model = ReferenceModel(self.policy)

        # 基线 z_ref: 使用运行均值估计
        self.z_ref: float = 0.0
        self.z_ref_momentum: float = 0.9

        self.log_history: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[int]:
        return [ord(c) % self.config.vocab_size for c in text][:self.config.max_length]

    def compute_kto_loss(
        self,
        policy_logps: float,
        ref_logps: float,
        is_desirable: bool,
    ) -> Tuple[float, float]:
        """
        计算 KTO 损失。

        Args:
            policy_logps: log π_θ(y|x)
            ref_logps: log π_ref(y|x)
            is_desirable: 是否为期望输出

        Returns:
            (loss, implicit_reward)
        """
        log_ratio = policy_logps - ref_logps
        implicit_reward = self.beta * log_ratio

        if is_desirable:
            # L_KTO^+ = -log σ(β * log_ratio - z_ref)
            logits = implicit_reward - self.z_ref
            loss = -_log_sigmoid(logits)
        else:
            # L_KTO^- = -log σ(-(β * log_ratio - z_ref))
            logits = -(implicit_reward - self.z_ref)
            loss = -_log_sigmoid(logits)

        return loss, implicit_reward

    def compute_kto_gradient(
        self,
        policy_logps: float,
        ref_logps: float,
        is_desirable: bool,
    ) -> float:
        """
        计算 KTO 损失对 log π_θ(y|x) 的梯度。

        对于期望输出:
            ∂L/∂log π_θ = -β * σ(-(β * log_ratio - z_ref))

        对于不期望输出:
            ∂L/∂log π_θ = +β * σ(β * log_ratio - z_ref)

        Returns:
            梯度信号 (标量)。
        """
        log_ratio = policy_logps - ref_logps
        margin = self.beta * log_ratio - self.z_ref

        if is_desirable:
            grad = -self.beta * _sigmoid(-margin)
        else:
            grad = self.beta * _sigmoid(margin)

        return grad

    def update_baseline(self, policy_logps: float, ref_logps: float) -> None:
        """
        使用指数移动平均更新基线 z_ref。

        z_ref ← momentum * z_ref + (1 - momentum) * β * log(π_θ/π_ref)
        """
        log_ratio = policy_logps - ref_logps
        reward = self.beta * log_ratio
        self.z_ref = (self.z_ref_momentum * self.z_ref
                      + (1.0 - self.z_ref_momentum) * reward)

    def train_step(
        self,
        prompt: str,
        response: str,
        is_desirable: bool,
    ) -> Dict[str, float]:
        """
        执行单个 KTO 训练步骤。

        Args:
            prompt: 输入提示
            response: 模型回答
            is_desirable: 是否为期望输出
        """
        prompt_ids = self._tokenize(prompt)
        response_ids = self._tokenize(response)
        full_ids = prompt_ids + response_ids

        policy_logps = self.policy.sequence_log_prob(full_ids)
        ref_logps = self.ref_model.compute_log_probs(full_ids)

        loss, implicit_reward = self.compute_kto_loss(
            policy_logps, ref_logps, is_desirable,
        )

        grad = self.compute_kto_gradient(policy_logps, ref_logps, is_desirable)

        # 更新基线
        self.update_baseline(policy_logps, ref_logps)

        # 反向传播
        self.policy.zero_grad()
        self._backprop_sequence(full_ids, grad)
        grad_norm = self.policy.step(self.lr, self.config.max_grad_norm)

        weight = self.desirable_weight if is_desirable else self.undesirable_weight

        return {
            "loss": loss * weight,
            "implicit_reward": implicit_reward,
            "z_ref": self.z_ref,
            "grad_norm": grad_norm,
            "is_desirable": is_desirable,
        }

    def _backprop_sequence(self, token_ids: List[int], grad_scale: float) -> None:
        """对序列做反向传播。"""
        if len(token_ids) <= 1:
            return
        for k in range(1, len(token_ids)):
            prefix = token_ids[:k]
            logits = self.policy.forward(prefix)
            probs = _softmax(logits)
            d_logits = list(probs)
            d_logits[token_ids[k]] -= 1.0
            d_logits = [g * grad_scale for g in d_logits]
            self.policy._backward_from_logits_grad(d_logits)

    def train(
        self,
        prompts: List[str],
        responses: List[str],
        labels: List[bool],
    ) -> List[Dict[str, float]]:
        """
        完整的 KTO 训练流程。

        Args:
            prompts: 提示列表
            responses: 回答列表
            labels: 是否为期望输出的标签列表
        """
        n_samples = len(prompts)
        print(f"[KTO] 开始训练: {n_samples} 样本, "
              f"{self.config.epochs} epochs, β={self.beta}")

        indices = list(range(n_samples))

        for epoch in range(self.config.epochs):
            random.shuffle(indices)
            epoch_loss = 0.0
            n = 0

            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                for idx in batch_idx:
                    m = self.train_step(
                        prompts[idx], responses[idx], labels[idx],
                    )
                    epoch_loss += m["loss"]
                    n += 1

            avg_loss = epoch_loss / max(n, 1)
            self.log_history.append({
                "epoch_loss": avg_loss,
                "z_ref": self.z_ref,
            })
            print(f"[KTO] Epoch {epoch + 1}/{self.config.epochs} | "
                  f"loss={avg_loss:.4f} | z_ref={self.z_ref:.4f}")

        print("[KTO] 训练完成")
        return self.log_history


# ============================================================================
#  7. ORPOTrainer - Odds Ratio Preference Optimization
# ============================================================================

class ORPOTrainer:
    """
    ORPO (Odds Ratio Preference Optimization) 训练器。

    ORPO 不需要单独的参考模型，而是在语言模型训练中直接加入偏好对齐目标:

        L_ORPO = L_NLL + λ * L_OR

    其中:
        L_NLL = -log π_θ(y_w|x)  (负对数似然损失)
        L_OR  = -log σ(log(odds_θ(y_w|x)) - log(odds_θ(y_l|x)))

        odds_θ(y|x) = π_θ(y|x) / (1 - π_θ(y|x))

    ORPO 的优势:
    - 不需要参考模型，节省显存
    - 单阶段训练 (同时学习生成和对齐)
    - odds ratio 比对数比率更稳定

    参考: Hong et al., "ORPO: Monolithic Preference Optimization without
          Reference Model", 2024.
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.lr = config.learning_rate
        self.lam = config.orpo_lambda

        self.policy = SimpleLM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
        )
        # ORPO 不需要参考模型
        self.log_history: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[int]:
        return [ord(c) % self.config.vocab_size for c in text][:self.config.max_length]

    @staticmethod
    def _log_odds(log_prob: float) -> float:
        """
        计算对数几率。

        odds = p / (1 - p)
        log_odds = log(p / (1 - p)) = log(p) - log(1 - p)

        使用 log-sum-exp 技巧保证数值稳定性:
        log(1 - p) = log(1 - exp(log_p))
        当 log_p 接近 0 时: log(1 - exp(log_p)) ≈ log(-log_p) (使用 Taylor 展开)
        """
        # log_prob 通常是负数
        if log_prob > -1e-7:
            # p ≈ 1, odds -> infinity
            return 10.0  # 裁剪
        if log_prob < -30.0:
            # p ≈ 0, odds -> 0
            return -30.0  # 裁剪

        # log(1 - exp(log_prob)) 的数值稳定计算
        # 使用恒等式: log(1 - exp(x)) = log(-expm1(x)) 当 x < 0
        log_one_minus_p = math.log(-math.expm1(log_prob))
        return log_prob - log_one_minus_p

    def compute_orpo_loss(
        self,
        chosen_logps: float,
        rejected_logps: float,
    ) -> Tuple[float, float, float, float]:
        """
        计算 ORPO 损失。

        L_ORPO = L_NLL + λ * L_OR

        L_NLL = -log π_θ(y_w|x)
        L_OR  = -log σ(log_odds(y_w) - log_odds(y_l))

        Args:
            chosen_logps: log π_θ(y_w|x)
            rejected_logps: log π_θ(y_l|x)

        Returns:
            (total_loss, nll_loss, or_loss, log_odds_margin)
        """
        # 负对数似然损失
        nll_loss = -chosen_logps

        # 对数几率
        log_odds_chosen = self._log_odds(chosen_logps)
        log_odds_rejected = self._log_odds(rejected_logps)

        # 对数几率差
        log_odds_diff = log_odds_chosen - log_odds_rejected

        # OR 损失
        or_loss = -_log_sigmoid(log_odds_diff)

        # 总损失
        total_loss = nll_loss + self.lam * or_loss

        return total_loss, nll_loss, or_loss, log_odds_diff

    def compute_orpo_gradients(
        self,
        chosen_logps: float,
        rejected_logps: float,
    ) -> Tuple[float, float]:
        """
        计算 ORPO 损失对 log π_θ 的梯度。

        ∂L_NLL/∂log π_θ(y_w) = -1

        ∂L_OR/∂log π_θ(y_w):
            令 d = log_odds(y_w) - log_odds(y_l)
            ∂L_OR/∂log_odds(y_w) = -σ(-d)
            ∂log_odds(y_w)/∂log π_θ(y_w) = 1 / (1 - π_θ(y_w))
            ∂L_OR/∂log π_θ(y_w) = -σ(-d) / (1 - π_θ(y_w))

        类似地:
            ∂L_OR/∂log π_θ(y_l) = σ(-d) / (1 - π_θ(y_l))

        Returns:
            (grad_chosen, grad_rejected)
        """
        log_odds_chosen = self._log_odds(chosen_logps)
        log_odds_rejected = self._log_odds(rejected_logps)
        log_odds_diff = log_odds_chosen - log_odds_rejected
        sigmoid_neg = _sigmoid(-log_odds_diff)

        # NLL 梯度
        nll_grad_chosen = -1.0

        # OR 梯度
        # 1 / (1 - p) = 1 / (1 - exp(log_p))
        p_chosen = math.exp(_clip(chosen_logps, -30, 0))
        p_rejected = math.exp(_clip(rejected_logps, -30, 0))

        denom_chosen = max(1.0 - p_chosen, 1e-8)
        denom_rejected = max(1.0 - p_rejected, 1e-8)

        or_grad_chosen = -sigmoid_neg / denom_chosen
        or_grad_rejected = sigmoid_neg / denom_rejected

        grad_chosen = nll_grad_chosen + self.lam * or_grad_chosen
        grad_rejected = self.lam * or_grad_rejected

        return grad_chosen, grad_rejected

    def train_step(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict[str, float]:
        """执行单个 ORPO 训练步骤。"""
        prompt_ids = self._tokenize(prompt)
        chosen_full = prompt_ids + self._tokenize(chosen)
        rejected_full = prompt_ids + self._tokenize(rejected)

        chosen_logps = self.policy.sequence_log_prob(chosen_full)
        rejected_logps = self.policy.sequence_log_prob(rejected_full)

        total_loss, nll_loss, or_loss, log_odds_margin = self.compute_orpo_loss(
            chosen_logps, rejected_logps,
        )

        grad_chosen, grad_rejected = self.compute_orpo_gradients(
            chosen_logps, rejected_logps,
        )

        # 反向传播
        self.policy.zero_grad()
        self._backprop_sequence(chosen_full, grad_chosen)
        self._backprop_sequence(rejected_full, grad_rejected)
        grad_norm = self.policy.step(self.lr, self.config.max_grad_norm)

        return {
            "loss": total_loss,
            "nll_loss": nll_loss,
            "or_loss": or_loss,
            "log_odds_margin": log_odds_margin,
            "grad_norm": grad_norm,
        }

    def _backprop_sequence(self, token_ids: List[int], grad_scale: float) -> None:
        """对序列做反向传播。"""
        if len(token_ids) <= 1:
            return
        for k in range(1, len(token_ids)):
            prefix = token_ids[:k]
            logits = self.policy.forward(prefix)
            probs = _softmax(logits)
            d_logits = list(probs)
            d_logits[token_ids[k]] -= 1.0
            d_logits = [g * grad_scale for g in d_logits]
            self.policy._backward_from_logits_grad(d_logits)

    def train(self, dataset: PreferenceDataset) -> List[Dict[str, float]]:
        """完整的 ORPO 训练流程。"""
        print(f"[ORPO] 开始训练: {len(dataset)} 样本, "
              f"{self.config.epochs} epochs, λ={self.lam}")

        for epoch in range(self.config.epochs):
            pairs = dataset.get_pairs()
            random.shuffle(pairs)
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_or = 0.0
            n = 0

            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i:i + self.config.batch_size]
                for pair in batch:
                    m = self.train_step(pair.prompt, pair.chosen, pair.rejected)
                    epoch_loss += m["loss"]
                    epoch_nll += m["nll_loss"]
                    epoch_or += m["or_loss"]
                    n += 1

            avg_loss = epoch_loss / max(n, 1)
            avg_nll = epoch_nll / max(n, 1)
            avg_or = epoch_or / max(n, 1)
            self.log_history.append({
                "epoch_loss": avg_loss,
                "epoch_nll_loss": avg_nll,
                "epoch_or_loss": avg_or,
            })
            print(f"[ORPO] Epoch {epoch + 1}/{self.config.epochs} | "
                  f"loss={avg_loss:.4f} | nll={avg_nll:.4f} | or={avg_or:.4f}")

        print("[ORPO] 训练完成")
        return self.log_history


# ============================================================================
#  8. SimPOTrainer - Simple Preference Optimization
# ============================================================================

class SimPOTrainer:
    """
    SimPO (Simple Preference Optimization) 训练器。

    SimPO 进一步简化了 DPO，去掉了参考模型，直接使用策略自身的对数概率:

        L_SimPO = -E[log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x)) - γ)]

    其中:
        γ > 0 是目标奖励间隔 (target reward margin)
        β 是长度归一化的缩放系数

    SimPO 的关键特性:
    - 不需要参考模型 (类似 ORPO)
    - 使用平均对数概率 (长度归一化) 以避免长度偏差
    - 目标间隔 γ 确保 chosen 和 rejected 之间有足够的区分度

    参考: Meng et al., "SimPO: Simple Preference Optimization with a
          Reference-Free Reward", 2024.
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.beta = config.beta
        self.gamma = config.simpo_gamma
        self.lr = config.learning_rate

        self.policy = SimpleLM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
        )
        # SimPO 不需要参考模型
        self.log_history: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[int]:
        return [ord(c) % self.config.vocab_size for c in text][:self.config.max_length]

    def _length_normalized_logps(self, token_ids: List[int]) -> float:
        """
        计算长度归一化的对数概率。

        log π_θ(y|x) / |y|

        长度归一化防止模型偏向生成更长的回答。
        """
        if len(token_ids) <= 1:
            return 0.0
        raw_logps = self.policy.sequence_log_prob(token_ids)
        # 只计算 response 部分的长度 (去掉 prompt)
        response_len = max(len(token_ids) - 1, 1)
        return raw_logps / response_len

    def compute_simpo_loss(
        self,
        chosen_logps: float,
        rejected_logps: float,
    ) -> Tuple[float, float, float]:
        """
        计算 SimPO 损失。

        L_SimPO = -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x)) - γ)

        Args:
            chosen_logps: 长度归一化的 log π_θ(y_w|x)
            rejected_logps: 长度归一化的 log π_θ(y_l|x)

        Returns:
            (loss, chosen_reward, rejected_reward)
        """
        # SimPO 隐式奖励就是长度归一化的对数概率
        chosen_reward = chosen_logps
        rejected_reward = rejected_logps

        # logits = β * (r_w - r_l) - γ
        logits = self.beta * (chosen_logps - rejected_logps) - self.gamma

        loss = -_log_sigmoid(logits)

        return loss, chosen_reward, rejected_reward

    def compute_simpo_gradients(
        self,
        chosen_logps: float,
        rejected_logps: float,
        chosen_len: int,
        rejected_len: int,
    ) -> Tuple[float, float]:
        """
        计算 SimPO 损失对 log π_θ 的梯度。

        ∂L/∂log π_θ(y_w) = -(β / |y_w|) * σ(-logits)
        ∂L/∂log π_θ(y_l) = +(β / |y_l|) * σ(-logits)

        其中 logits = β * (r_w - r_l) - γ

        Returns:
            (grad_chosen, grad_rejected)
        """
        logits = self.beta * (chosen_logps - rejected_logps) - self.gamma
        sigmoid_neg = _sigmoid(-logits)

        grad_chosen = -(self.beta / max(chosen_len, 1)) * sigmoid_neg
        grad_rejected = (self.beta / max(rejected_len, 1)) * sigmoid_neg

        return grad_chosen, grad_rejected

    def train_step(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict[str, float]:
        """执行单个 SimPO 训练步骤。"""
        prompt_ids = self._tokenize(prompt)
        chosen_full = prompt_ids + self._tokenize(chosen)
        rejected_full = prompt_ids + self._tokenize(rejected)

        # 长度归一化的对数概率
        chosen_logps = self._length_normalized_logps(chosen_full)
        rejected_logps = self._length_normalized_logps(rejected_full)

        loss, chosen_reward, rejected_reward = self.compute_simpo_loss(
            chosen_logps, rejected_logps,
        )

        # 响应长度 (不含 prompt)
        chosen_len = max(len(chosen_full) - len(prompt_ids), 1)
        rejected_len = max(len(rejected_full) - len(prompt_ids), 1)

        grad_chosen, grad_rejected = self.compute_simpo_gradients(
            chosen_logps, rejected_logps, chosen_len, rejected_len,
        )

        # 反向传播
        self.policy.zero_grad()
        self._backprop_sequence(chosen_full, grad_chosen)
        self._backprop_sequence(rejected_full, grad_rejected)
        grad_norm = self.policy.step(self.lr, self.config.max_grad_norm)

        return {
            "loss": loss,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin": chosen_reward - rejected_reward,
            "grad_norm": grad_norm,
        }

    def _backprop_sequence(self, token_ids: List[int], grad_scale: float) -> None:
        """对序列做反向传播。"""
        if len(token_ids) <= 1:
            return
        for k in range(1, len(token_ids)):
            prefix = token_ids[:k]
            logits = self.policy.forward(prefix)
            probs = _softmax(logits)
            d_logits = list(probs)
            d_logits[token_ids[k]] -= 1.0
            d_logits = [g * grad_scale for g in d_logits]
            self.policy._backward_from_logits_grad(d_logits)

    def train(self, dataset: PreferenceDataset) -> List[Dict[str, float]]:
        """完整的 SimPO 训练流程。"""
        print(f"[SimPO] 开始训练: {len(dataset)} 样本, "
              f"{self.config.epochs} epochs, β={self.beta}, γ={self.gamma}")

        for epoch in range(self.config.epochs):
            pairs = dataset.get_pairs()
            random.shuffle(pairs)
            epoch_loss = 0.0
            epoch_margin = 0.0
            n = 0

            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i:i + self.config.batch_size]
                for pair in batch:
                    m = self.train_step(pair.prompt, pair.chosen, pair.rejected)
                    epoch_loss += m["loss"]
                    epoch_margin += m["reward_margin"]
                    n += 1

            avg_loss = epoch_loss / max(n, 1)
            avg_margin = epoch_margin / max(n, 1)
            self.log_history.append({
                "epoch_loss": avg_loss,
                "epoch_reward_margin": avg_margin,
            })
            print(f"[SimPO] Epoch {epoch + 1}/{self.config.epochs} | "
                  f"loss={avg_loss:.4f} | margin={avg_margin:.4f}")

        print("[SimPO] 训练完成")
        return self.log_history


# ============================================================================
#  9. PreferenceDataset - 偏好数据管理
# ============================================================================

@dataclass
class PreferencePair:
    """
    偏好对数据结构。

    Attributes:
        prompt: 输入提示文本。
        chosen: 被偏好的回答文本。
        rejected: 被拒绝的回答文本。
        chosen_score: 被偏好回答的分数 (可选)。
        rejected_score: 被拒绝回答的分数 (可选)。
        metadata: 额外元数据 (可选)。
    """
    prompt: str
    chosen: str
    rejected: str
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KTOSample:
    """
    KTO 单样本数据结构 (非配对)。

    Attributes:
        prompt: 输入提示文本。
        response: 模型回答文本。
        is_desirable: 是否为期望输出。
    """
    prompt: str
    response: str
    is_desirable: bool


class PreferenceDataset:
    """
    偏好数据集管理类。

    支持的操作:
    - 添加偏好对
    - 随机打乱
    - 批量获取
    - 过滤和统计
    - 从 KTO 样本构造
    """

    def __init__(self, pairs: Optional[List[PreferencePair]] = None):
        """
        初始化偏好数据集。

        Args:
            pairs: 初始偏好对列表 (可为空)。
        """
        self.pairs: List[PreferencePair] = pairs if pairs is not None else []

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.pairs[idx]

    def add_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        chosen_score: Optional[float] = None,
        rejected_score: Optional[float] = None,
    ) -> None:
        """
        添加一个偏好对。

        Args:
            prompt: 输入提示
            chosen: 被偏好的回答
            rejected: 被拒绝的回答
            chosen_score: 可选分数
            rejected_score: 可选分数
        """
        pair = PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )
        self.pairs.append(pair)

    def get_pairs(self) -> List[PreferencePair]:
        """返回所有偏好对的列表。"""
        return list(self.pairs)

    def shuffle(self, seed: Optional[int] = None) -> None:
        """
        随机打乱数据集。

        Args:
            seed: 随机种子 (None 表示使用系统随机)。
        """
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.pairs)
        else:
            random.shuffle(self.pairs)

    def get_batch(self, batch_size: int, start_idx: int = 0) -> List[PreferencePair]:
        """
        获取一个批次的偏好对。

        Args:
            batch_size: 批次大小
            start_idx: 起始索引

        Returns:
            偏好对批次。
        """
        end_idx = min(start_idx + batch_size, len(self.pairs))
        return self.pairs[start_idx:end_idx]

    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[List[PreferencePair]]:
        """
        生成所有批次的迭代器。

        Args:
            batch_size: 批次大小
            shuffle: 是否在迭代前打乱

        Returns:
            批次列表的列表。
        """
        if shuffle:
            self.shuffle()

        batches = []
        for i in range(0, len(self.pairs), batch_size):
            batch = self.pairs[i:i + batch_size]
            batches.append(batch)
        return batches

    def filter_by_length(
        self,
        max_prompt_len: int = 512,
        max_response_len: int = 512,
    ) -> PreferenceDataset:
        """
        按文本长度过滤偏好对。

        Args:
            max_prompt_len: prompt 最大字符数
            max_response_len: response 最大字符数

        Returns:
            过滤后的新数据集。
        """
        filtered = [
            p for p in self.pairs
            if len(p.prompt) <= max_prompt_len
            and len(p.chosen) <= max_response_len
            and len(p.rejected) <= max_response_len
        ]
        return PreferenceDataset(filtered)

    def split(self, ratio: float, seed: int = 42) -> Tuple[PreferenceDataset, PreferenceDataset]:
        """
        将数据集按比例分割为训练集和验证集。

        Args:
            ratio: 训练集比例 (0 < ratio < 1)
            seed: 随机种子

        Returns:
            (train_dataset, val_dataset)
        """
        rng = random.Random(seed)
        indices = list(range(len(self.pairs)))
        rng.shuffle(indices)

        split_idx = int(len(self.pairs) * ratio)
        train_pairs = [self.pairs[i] for i in indices[:split_idx]]
        val_pairs = [self.pairs[i] for i in indices[split_idx:]]

        return PreferenceDataset(train_pairs), PreferenceDataset(val_pairs)

    def statistics(self) -> Dict[str, Any]:
        """
        计算数据集统计信息。

        Returns:
            包含统计信息的字典。
        """
        if not self.pairs:
            return {"n_pairs": 0}

        prompt_lens = [len(p.prompt) for p in self.pairs]
        chosen_lens = [len(p.chosen) for p in self.pairs]
        rejected_lens = [len(p.rejected) for p in self.pairs]

        return {
            "n_pairs": len(self.pairs),
            "avg_prompt_len": sum(prompt_lens) / len(prompt_lens),
            "avg_chosen_len": sum(chosen_lens) / len(chosen_lens),
            "avg_rejected_len": sum(rejected_lens) / len(rejected_lens),
            "max_prompt_len": max(prompt_lens),
            "max_chosen_len": max(chosen_lens),
            "max_rejected_len": max(rejected_lens),
            "has_scores": any(p.chosen_score is not None for p in self.pairs),
        }

    @classmethod
    def from_kto_samples(
        cls,
        desirable_samples: List[KTOSample],
        undesirable_samples: List[KTOSample],
    ) -> PreferenceDataset:
        """
        从 KTO 样本构造偏好对数据集。

        将期望输出与不期望输出配对 (基于 prompt 匹配)。

        Args:
            desirable_samples: 期望样本列表
            undesirable_samples: 不期望样本列表

        Returns:
            构造的偏好数据集。
        """
        # 按 prompt 分组
        desirable_by_prompt: Dict[str, List[str]] = {}
        for s in desirable_samples:
            desirable_by_prompt.setdefault(s.prompt, []).append(s.response)

        undesirable_by_prompt: Dict[str, List[str]] = {}
        for s in undesirable_samples:
            undesirable_by_prompt.setdefault(s.prompt, []).append(s.response)

        pairs = []
        for prompt, chosen_list in desirable_by_prompt.items():
            if prompt in undesirable_by_prompt:
                for chosen in chosen_list:
                    for rejected in undesirable_by_prompt[prompt]:
                        pairs.append(PreferencePair(
                            prompt=prompt,
                            chosen=chosen,
                            rejected=rejected,
                        ))

        return cls(pairs)

    @classmethod
    def create_synthetic(
        cls,
        n_pairs: int = 100,
        prompt_pool: Optional[List[str]] = None,
        seed: int = 42,
    ) -> PreferenceDataset:
        """
        创建合成偏好数据集 (用于测试和演示)。

        Args:
            n_pairs: 偏好对数量
            prompt_pool: 可选的 prompt 池
            seed: 随机种子

        Returns:
            合成偏好数据集。
        """
        rng = random.Random(seed)

        if prompt_pool is None:
            prompt_pool = [
                "What is machine learning?",
                "Explain quantum computing.",
                "Write a poem about nature.",
                "How does photosynthesis work?",
                "Describe the solar system.",
                "What is artificial intelligence?",
                "Explain relativity.",
                "How do vaccines work?",
                "What causes climate change?",
                "Describe the water cycle.",
            ]

        pairs = []
        for _ in range(n_pairs):
            prompt = rng.choice(prompt_pool)

            # 生成 chosen (较长的回答)
            n_words_chosen = rng.randint(10, 30)
            chosen_words = []
            for _ in range(n_words_chosen):
                word_len = rng.randint(3, 10)
                word = "".join(chr(rng.randint(97, 122)) for _ in range(word_len))
                chosen_words.append(word)
            chosen = " ".join(chosen_words)

            # 生成 rejected (较短的或重复的回答)
            n_words_rejected = rng.randint(3, 8)
            rejected_words = []
            for _ in range(n_words_rejected):
                word_len = rng.randint(3, 7)
                word = "".join(chr(rng.randint(97, 122)) for _ in range(word_len))
                rejected_words.append(word)
            rejected = " ".join(rejected_words)

            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
            ))

        return cls(pairs)


# ============================================================================
#  工具函数: 运行所有算法的演示
# ============================================================================

def run_dpo_demo() -> None:
    """运行 DPO 及其变体的演示。"""
    print("=" * 70)
    print("DPO 及其变体算法演示")
    print("=" * 70)

    # 创建合成数据集
    dataset = PreferenceDataset.create_synthetic(n_pairs=20, seed=42)
    stats = dataset.statistics()
    print(f"\n数据集统计: {stats['n_pairs']} 偏好对")
    print(f"  平均 prompt 长度: {stats['avg_prompt_len']:.1f}")
    print(f"  平均 chosen 长度: {stats['avg_chosen_len']:.1f}")
    print(f"  平均 rejected 长度: {stats['avg_rejected_len']:.1f}")

    # 1. DPO
    print("\n" + "-" * 50)
    config = DPOConfig(
        beta=0.1,
        learning_rate=1e-3,
        batch_size=4,
        epochs=2,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=32,
        seed=42,
    )
    trainer = DPOTrainer(config)
    trainer.train(dataset)

    # 2. IPO
    print("\n" + "-" * 50)
    ipo_config = DPOConfig(
        beta=0.1,
        learning_rate=1e-3,
        batch_size=4,
        epochs=2,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=32,
        seed=42,
    )
    ipo_trainer = IPOTrainer(ipo_config)
    ipo_trainer.train(dataset)

    # 3. KTO
    print("\n" + "-" * 50)
    kto_config = DPOConfig(
        beta=0.1,
        learning_rate=1e-3,
        batch_size=4,
        epochs=2,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=32,
        seed=42,
    )
    kto_trainer = KTOTrainer(kto_config)

    # 构造 KTO 数据
    prompts, responses, labels = [], [], []
    for pair in dataset.get_pairs():
        prompts.append(pair.prompt)
        responses.append(pair.chosen)
        labels.append(True)
        prompts.append(pair.prompt)
        responses.append(pair.rejected)
        labels.append(False)
    kto_trainer.train(prompts, responses, labels)

    # 4. ORPO
    print("\n" + "-" * 50)
    orpo_config = DPOConfig(
        beta=0.1,
        learning_rate=1e-3,
        batch_size=4,
        epochs=2,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=32,
        seed=42,
        orpo_lambda=1.0,
    )
    orpo_trainer = ORPOTrainer(orpo_config)
    orpo_trainer.train(dataset)

    # 5. SimPO
    print("\n" + "-" * 50)
    simpo_config = DPOConfig(
        beta=0.1,
        learning_rate=1e-3,
        batch_size=4,
        epochs=2,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=32,
        seed=42,
        simpo_gamma=0.5,
    )
    simpo_trainer = SimPOTrainer(simpo_config)
    simpo_trainer.train(dataset)

    print("\n" + "=" * 70)
    print("所有算法演示完成")
    print("=" * 70)


if __name__ == "__main__":
    run_dpo_demo()
