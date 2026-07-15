"""
RLHF (Reinforcement Learning from Human Feedback) - 完整实现
============================================================

纯Python实现，仅依赖 math/random，无外部依赖。

包含组件:
1. RLHFConfig - 配置数据类
2. RewardModel - 奖励模型 (RewardNetwork + BradleyTerryLoss)
3. KLController - KL散度控制器 (Adaptive + Fixed)
4. PPOTrainer - PPO训练器 (Buffer, ActorCritic, GAE, PPO-Clip, 训练循环)
5. RejectionSampling - 拒绝采样
6. RLHFTrainer - 主编排器

数学公式:
- PPO-Clip: L = min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A) - β*KL
- GAE: A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}, δ_t = r_t + γV(s_{t+1}) - V(s_t)
- KL: KL(π||π_ref) = E[log π(a|s) - log π_ref(a|s)]
- Bradley-Terry: P(y=1) = σ(r(x,y_chosen) - r(x,y_rejected))
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ============================================================================
# 工具函数
# ============================================================================

def sigmoid(x: float) -> float:
    """数值稳定的sigmoid函数: σ(x) = 1 / (1 + exp(-x))"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def log_sigmoid(x: float) -> float:
    """数值稳定的log(sigmoid(x)): log(σ(x)) = -log(1 + exp(-x))"""
    if x >= 0:
        return -math.log(1.0 + math.exp(-x))
    else:
        return x - math.log(1.0 + math.exp(x))


def log_softmax(x: List[float]) -> List[float]:
    """数值稳定的log-softmax: log(softmax(x_i)) = x_i - log(Σ exp(x_j))"""
    max_x = max(x)
    sum_exp = sum(math.exp(xi - max_x) for xi in x)
    log_sum_exp = max_x + math.log(sum_exp)
    return [xi - log_sum_exp for xi in x]


def softmax(x: List[float]) -> List[float]:
    """数值稳定的softmax: softmax(x_i) = exp(x_i) / Σ exp(x_j)"""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def entropy_from_probs(probs: List[float]) -> float:
    """计算概率分布的熵: H(p) = -Σ p_i * log(p_i)"""
    return -sum(p * math.log(p) if p > 1e-12 else 0.0 for p in probs)


def kl_divergence_from_log_probs(log_p: List[float], log_q: List[float]) -> float:
    """KL散度: KL(p||q) = Σ p_i * (log(p_i) - log(q_i))"""
    total = 0.0
    for lp, lq in zip(log_p, log_q):
        p = math.exp(lp)
        if p > 1e-12:
            total += p * (lp - lq)
    return total


def clip(value: float, min_val: float, max_val: float) -> float:
    """将值限制在[min_val, max_val]范围内"""
    return max(min_val, min(max_val, value))


def he_uniform_init(fan_in: int) -> float:
    """He均匀初始化: U(-sqrt(6/fan_in), sqrt(6/fan_in))"""
    limit = math.sqrt(6.0 / fan_in)
    return random.uniform(-limit, limit)


def xavier_uniform_init(fan_in: int, fan_out: int) -> float:
    """Xavier均匀初始化: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))"""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(-limit, limit)


def zeros(shape: List[int]) -> List[List[float]]:
    """创建零矩阵"""
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [zeros(shape[1:]) for _ in range(shape[0])]


def zeros_like(matrix: List[List[float]]) -> List[List[float]]:
    """创建与输入同形状的零矩阵"""
    if not matrix:
        return []
    if isinstance(matrix[0], list):
        return [[0.0 for _ in row] for row in matrix]
    return [0.0 for _ in matrix]


def mat_vec_mul(mat: List[List[float]], vec: List[float]) -> List[float]:
    """矩阵-向量乘法: result_i = Σ_j mat[i][j] * vec[j]"""
    return [sum(m_ij * v_j for m_ij, v_j in zip(row, vec)) for row in mat]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    """向量加法"""
    return [ai + bi for ai, bi in zip(a, b)]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    """向量减法"""
    return [ai - bi for ai, bi in zip(a, b)]


def vec_scale(v: List[float], s: float) -> List[float]:
    """向量标量乘法"""
    return [vi * s for vi in v]


def vec_dot(a: List[float], b: List[float]) -> float:
    """向量点积"""
    return sum(ai * bi for ai, bi in zip(a, b))


def vec_norm(v: List[float]) -> float:
    """向量L2范数"""
    return math.sqrt(sum(vi * vi for vi in v))


def outer_product(a: List[float], b: List[float]) -> List[List[float]]:
    """外积: result[i][j] = a[i] * b[j]"""
    return [[ai * bj for bj in b] for ai in a]


def transpose(mat: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    if not mat:
        return []
    n_rows = len(mat)
    n_cols = len(mat[0])
    return [[mat[i][j] for i in range(n_rows)] for j in range(n_cols)]


def mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵加法"""
    return [[a_ij + b_ij for a_ij, b_ij in zip(row_a, row_b)]
            for row_a, row_b in zip(a, b)]


def mat_scale(mat: List[List[float]], s: float) -> List[List[float]]:
    """矩阵标量乘法"""
    return [[m_ij * s for m_ij in row] for row in mat]


def mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法: C[i][j] = Σ_k a[i][k] * b[k][j]"""
    n = len(a)
    m = len(b[0])
    k = len(b)
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += a[i][p] * b[p][j]
            result[i][j] = s
    return result


def layer_norm(x: List[float], gamma: List[float], beta: List[float], eps: float = 1e-5) -> List[float]:
    """层归一化: y = γ * (x - μ) / σ + β"""
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    std = math.sqrt(var + eps)
    x_norm = [(xi - mean) / std for xi in x]
    return [gamma[i] * x_norm[i] + beta[i] for i in range(n)]


# ============================================================================
# 1. RLHFConfig - 配置数据类
# ============================================================================

@dataclass
class RLHFConfig:
    """
    RLHF训练配置数据类
    包含所有超参数
    """
    # --- 通用参数 ---
    seed: int = 42
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 3
    max_seq_length: int = 512
    vocab_size: int = 50257
    embedding_dim: int = 768
    hidden_dim: int = 1024
    num_layers: int = 6

    # --- 奖励模型参数 ---
    reward_model_lr: float = 5e-5
    reward_model_hidden_dim: int = 1024
    reward_model_num_layers: int = 3
    reward_model_dropout: float = 0.1
    reward_model_epochs: int = 1
    reward_model_batch_size: int = 32
    reward_model_weight_decay: float = 0.01
    reward_model_gradient_clip: float = 1.0
    reward_model_margin: float = 0.0  # 用于margin-based BT loss

    # --- KL控制器参数 ---
    kl_coeff: float = 0.1  # 初始KL系数 β
    kl_target: float = 6.0  # 目标KL散度
    kl_horizon: int = 10000  # 自适应KL控制的时间步
    kl_adaptive: bool = True  # 是否使用自适应KL控制
    kl_strategy: str = "adaptive"  # "adaptive" 或 "fixed"

    # --- PPO参数 ---
    ppo_epochs: int = 4  # 每批数据的PPO更新轮数
    ppo_clip_epsilon: float = 0.2  # PPO裁剪范围 ε
    ppo_gamma: float = 0.99  # 折扣因子 γ
    ppo_gae_lambda: float = 0.95  # GAE参数 λ
    ppo_entropy_coeff: float = 0.01  # 熵正则化系数
    ppo_value_coeff: float = 0.5  # 价值损失系数
    ppo_max_grad_norm: float = 0.5  # 梯度裁剪
    ppo_policy_lr: float = 1e-6  # 策略学习率
    ppo_value_lr: float = 1e-5  # 价值函数学习率
    ppo_mini_batch_size: int = 16  # PPO小批量大小
    ppo_norm_advantage: bool = True  # 是否归一化优势
    ppo_clip_value_loss: bool = True  # 是否裁剪价值损失
    ppo_value_clip_epsilon: float = 0.2  # 价值损失裁剪范围

    # --- 拒绝采样参数 ---
    rejection_sampling_num_candidates: int = 8  # 每个prompt生成的候选数
    rejection_sampling_top_k: int = 2  # 选择的最佳候选数
    rejection_sampling_temperature: float = 0.7  # 采样温度
    rejection_sampling_threshold: float = 0.0  # 最低奖励阈值

    # --- 训练循环参数 ---
    num_rollout_steps: int = 128  # 每次rollout的步数
    num_training_steps: int = 10000  # 总训练步数
    reward_model_training_freq: int = 1  # 奖励模型训练频率
    ppo_update_freq: int = 1  # PPO更新频率
    eval_freq: int = 100  # 评估频率
    save_freq: int = 500  # 保存频率
    log_freq: int = 10  # 日志频率

    # --- Adam优化器参数 ---
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 0.0

    # --- 早停参数 ---
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    def validate(self) -> bool:
        """验证配置参数的合法性"""
        assert 0 < self.learning_rate <= 1.0, "learning_rate must be in (0, 1]"
        assert 0 < self.batch_size, "batch_size must be positive"
        assert 0 < self.ppo_clip_epsilon < 1.0, "ppo_clip_epsilon must be in (0, 1)"
        assert 0 < self.ppo_gamma <= 1.0, "ppo_gamma must be in (0, 1]"
        assert 0 < self.ppo_gae_lambda <= 1.0, "ppo_gae_lambda must be in (0, 1]"
        assert self.kl_strategy in ("adaptive", "fixed"), "kl_strategy must be 'adaptive' or 'fixed'"
        assert self.reward_model_num_layers >= 1, "reward_model_num_layers must be >= 1"
        assert self.ppo_epochs >= 1, "ppo_epochs must be >= 1"
        return True


# ============================================================================
# 2. RewardModel - 奖励模型
# ============================================================================

class LinearLayer:
    """
    全连接线性层: y = Wx + b
    支持前向传播和反向传播
    """

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # Xavier初始化权重
        self.weights: List[List[float]] = [
            [xavier_uniform_init(in_features, out_features) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        # 偏置初始化为0
        self.biases: List[float] = [0.0] * out_features
        # 梯度
        self.grad_weights: List[List[float]] = zeros([out_features, in_features])
        self.grad_biases: List[float] = [0.0] * out_features
        # 缓存
        self._input: Optional[List[float]] = None

    def forward(self, x: List[float]) -> List[float]:
        """前向传播: y = Wx + b"""
        self._input = x[:]
        output = []
        for i in range(self.out_features):
            val = self.biases[i]
            for j in range(self.in_features):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad_output: List[float]) -> List[float]:
        """反向传播"""
        # grad_weights[i][j] = grad_output[i] * input[j]
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.grad_weights[i][j] = grad_output[i] * self._input[j]
            self.grad_biases[i] = grad_output[i]
        # grad_input[j] = Σ_i grad_output[i] * weights[i][j]
        grad_input = [0.0] * self.in_features
        for j in range(self.in_features):
            s = 0.0
            for i in range(self.out_features):
                s += grad_output[i] * self.weights[i][j]
            grad_input[j] = s
        return grad_input

    def parameters(self) -> Tuple[List[List[float]], List[float]]:
        return self.weights, self.biases

    def gradients(self) -> Tuple[List[List[float]], List[float]]:
        return self.grad_weights, self.grad_biases


class ReLU:
    """ReLU激活函数: f(x) = max(0, x)"""

    def __init__(self):
        self._input: Optional[List[float]] = None

    def forward(self, x: List[float]) -> List[float]:
        self._input = x[:]
        return [max(0.0, xi) for xi in x]

    def backward(self, grad_output: List[float]) -> List[float]:
        return [grad_output[i] * (1.0 if self._input[i] > 0.0 else 0.0)
                for i in range(len(self._input))]


class TanhActivation:
    """Tanh激活函数: f(x) = tanh(x)"""

    def __init__(self):
        self._output: Optional[List[float]] = None

    def forward(self, x: List[float]) -> List[float]:
        self._output = [math.tanh(xi) for xi in x]
        return self._output[:]

    def backward(self, grad_output: List[float]) -> List[float]:
        # tanh'(x) = 1 - tanh(x)^2
        return [grad_output[i] * (1.0 - self._output[i] ** 2)
                for i in range(len(self._output))]


class GELUActivation:
    """
    GELU激活函数: GELU(x) = x * Φ(x)
    其中 Φ(x) 是标准正态分布的CDF
    近似: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """

    def __init__(self):
        self._input: Optional[List[float]] = None
        self._output: Optional[List[float]] = None

    def forward(self, x: List[float]) -> List[float]:
        self._input = x[:]
        self._output = []
        for xi in x:
            cdf = 0.5 * (1.0 + math.erf(xi / math.sqrt(2.0)))
            self._output.append(xi * cdf)
        return self._output[:]

    def backward(self, grad_output: List[float]) -> List[float]:
        grad_input = []
        sqrt_2pi = math.sqrt(2.0 / math.pi)
        for i, xi in enumerate(self._input):
            pdf = math.exp(-0.5 * xi * xi) / math.sqrt(2.0 * math.pi)
            cdf = 0.5 * (1.0 + math.erf(xi / math.sqrt(2.0)))
            # GELU'(x) = Φ(x) + x * φ(x)
            gelu_grad = cdf + xi * pdf
            grad_input.append(grad_output[i] * gelu_grad)
        return grad_input


class Dropout:
    """
    Dropout层: 训练时以概率p随机置零，推理时不操作
    """

    def __init__(self, p: float = 0.1):
        self.p = p
        self._mask: Optional[List[float]] = None
        self.training: bool = True

    def forward(self, x: List[float]) -> List[float]:
        if not self.training or self.p <= 0.0:
            return x[:]
        self._mask = [1.0 if random.random() > self.p else 0.0 for _ in x]
        scale = 1.0 / (1.0 - self.p)
        return [xi * mi * scale for xi, mi in zip(x, self._mask)]

    def backward(self, grad_output: List[float]) -> List[float]:
        if self._mask is None:
            return grad_output[:]
        scale = 1.0 / (1.0 - self.p)
        return [gi * mi * scale for gi, mi in zip(grad_output, self._mask)]


class LayerNormModule:
    """层归一化模块"""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma: List[float] = [1.0] * dim
        self.beta: List[float] = [0.0] * dim
        self.grad_gamma: List[float] = [0.0] * dim
        self.grad_beta: List[float] = [0.0] * dim
        self._x_norm: Optional[List[float]] = None
        self._std: float = 1.0

    def forward(self, x: List[float]) -> List[float]:
        n = len(x)
        mean = sum(x) / n
        var = sum((xi - mean) ** 2 for xi in x) / n
        self._std = math.sqrt(var + self.eps)
        self._x_norm = [(xi - mean) / self._std for xi in x]
        return [self.gamma[i] * self._x_norm[i] + self.beta[i] for i in range(n)]

    def backward(self, grad_output: List[float]) -> List[float]:
        n = len(grad_output)
        # 对gamma和beta的梯度
        self.grad_gamma = [grad_output[i] * self._x_norm[i] for i in range(n)]
        self.grad_beta = grad_output[:]

        # 对输入的梯度
        dx_norm = [grad_output[i] * self.gamma[i] for i in range(n)]
        dvar = sum(dx_norm[i] * (self._x_norm[i]) * (-0.5) / (self._std ** 2)
                   for i in range(n))
        dmean = sum(-dx_norm[i] / self._std for i in range(n))
        grad_input = [
            dx_norm[i] / self._std + dvar * 2.0 * self._x_norm[i] / n + dmean / n
            for i in range(n)
        ]
        return grad_input


class RewardNetwork:
    """
    奖励网络: MLP将(prompt, response)映射为标量奖励值

    架构:
        输入 -> [Linear -> GELU -> LayerNorm -> Dropout] * N -> Linear -> 标量输出

    前向传播:
        h_0 = Embedding(prompt, response)
        h_{l+1} = Dropout(GELU(LayerNorm(W_l * h_l + b_l)))
        r = W_out * h_L + b_out
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024,
                 num_layers: int = 3, dropout: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # 构建MLP层
        self.layers: List[LinearLayer] = []
        self.activations: List[GELUActivation] = []
        self.norms: List[LayerNormModule] = []
        self.dropouts: List[Dropout] = []

        # 输入层
        self.layers.append(LinearLayer(input_dim, hidden_dim))
        self.activations.append(GELUActivation())
        self.norms.append(LayerNormModule(hidden_dim))
        self.dropouts.append(Dropout(dropout))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(LinearLayer(hidden_dim, hidden_dim))
            self.activations.append(GELUActivation())
            self.norms.append(LayerNormModule(hidden_dim))
            self.dropouts.append(Dropout(dropout))

        # 输出层 (标量)
        self.output_layer = LinearLayer(hidden_dim, 1)

        # 缓存
        self._layer_inputs: List[List[float]] = []
        self._layer_outputs: List[List[float]] = []

    def forward(self, x: List[float]) -> float:
        """
        前向传播: (prompt, response) -> 标量奖励

        Args:
            x: 输入特征向量 (prompt和response的拼接/编码)

        Returns:
            标量奖励值
        """
        self._layer_inputs = []
        self._layer_outputs = []

        h = x[:]
        for i in range(self.num_layers):
            self._layer_inputs.append(h[:])
            h = self.layers[i].forward(h)
            h = self.activations[i].forward(h)
            h = self.norms[i].forward(h)
            h = self.dropouts[i].forward(h)
            self._layer_outputs.append(h[:])

        output = self.output_layer.forward(h)
        return output[0]

    def backward(self, grad_reward: float) -> List[float]:
        """
        反向传播: 计算所有参数的梯度

        Args:
            grad_reward: 奖励值的梯度 (标量)

        Returns:
            输入的梯度
        """
        # 输出层反向
        grad = self.output_layer.backward([grad_reward])

        # 逐层反向
        for i in range(self.num_layers - 1, -1, -1):
            grad = self.dropouts[i].backward(grad)
            grad = self.norms[i].backward(grad)
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)

        return grad

    def get_all_parameters(self) -> List[Tuple]:
        """获取所有参数及其梯度"""
        params = []
        for i in range(self.num_layers):
            params.append((self.layers[i].weights, self.layers[i].grad_weights))
            params.append((self.layers[i].biases, self.layers[i].grad_biases))
            params.append((self.norms[i].gamma, self.norms[i].grad_gamma))
            params.append((self.norms[i].beta, self.norms[i].grad_beta))
        params.append((self.output_layer.weights, self.output_layer.grad_weights))
        params.append((self.output_layer.biases, self.output_layer.grad_biases))
        return params

    def zero_grad(self):
        """清零所有梯度"""
        for i in range(self.num_layers):
            self.layers[i].grad_weights = zeros([self.layers[i].out_features, self.layers[i].in_features])
            self.layers[i].grad_biases = [0.0] * self.layers[i].out_features
            self.norms[i].grad_gamma = [0.0] * self.norms[i].dim
            self.norms[i].grad_beta = [0.0] * self.norms[i].dim
        self.output_layer.grad_weights = zeros([1, self.output_layer.in_features])
        self.output_layer.grad_biases = [0.0]


class BradleyTerryLoss:
    """
    Bradley-Terry模型损失函数

    模型: P(chosen > rejected | prompt) = σ(r(x, y_chosen) - r(x, y_rejected))

    损失: L = -log(σ(r_chosen - r_rejected))
         = -log_sigmoid(r_chosen - r_rejected)

    梯度:
        ∂L/∂r_chosen = -σ(-(r_chosen - r_rejected)) = σ(r_rejected - r_chosen)
        ∂L/∂r_rejected = σ(r_chosen - r_rejected)

    支持margin变体:
        L_margin = -log(σ(r_chosen - r_rejected - margin))
    """

    def __init__(self, margin: float = 0.0):
        self.margin = margin

    def compute_loss(self, reward_chosen: float, reward_rejected: float) -> float:
        """
        计算Bradley-Terry损失

        L = -log(σ(r_chosen - r_rejected - margin))

        Args:
            reward_chosen: 被偏好回答的奖励值
            reward_rejected: 被拒绝回答的奖励值

        Returns:
            标量损失值
        """
        diff = reward_chosen - reward_rejected - self.margin
        return -log_sigmoid(diff)

    def compute_gradients(self, reward_chosen: float, reward_rejected: float
                          ) -> Tuple[float, float]:
        """
        计算损失对奖励值的梯度

        ∂L/∂r_chosen = σ(r_rejected - r_chosen + margin)
        ∂L/∂r_rejected = -σ(r_rejected - r_chosen + margin) = σ(r_chosen - r_rejected - margin)

        推导:
            L = -log(σ(d)), d = r_chosen - r_rejected - margin
            ∂L/∂d = -σ(-d) = σ(d) - 1
            ∂L/∂r_chosen = ∂L/∂d * ∂d/∂r_chosen = (σ(d) - 1) * 1 = σ(d) - 1
            ∂L/∂r_rejected = ∂L/∂d * ∂d/∂r_rejected = (σ(d) - 1) * (-1) = 1 - σ(d)

        Args:
            reward_chosen: 被偏好回答的奖励值
            reward_rejected: 被拒绝回答的奖励值

        Returns:
            (grad_chosen, grad_rejected) 梯度元组
        """
        diff = reward_chosen - reward_rejected - self.margin
        sig = sigmoid(diff)
        grad_chosen = sig - 1.0  # = -σ(-diff)
        grad_rejected = 1.0 - sig  # = σ(-diff)
        return grad_chosen, grad_rejected

    def compute_accuracy(self, reward_chosen: float, reward_rejected: float) -> int:
        """计算预测准确率 (1 if correct, 0 if wrong)"""
        return 1 if reward_chosen > reward_rejected else 0


class RewardModel:
    """
    奖励模型训练器

    组合RewardNetwork和BradleyTerryLoss，提供完整的训练流程:
    1. 对每个偏好对 (prompt, chosen, rejected):
       - r_c = RewardNetwork(prompt, chosen)
       - r_r = RewardNetwork(prompt, rejected)
       - L = -log(σ(r_c - r_r))
    2. 反向传播计算梯度
    3. 更新参数

    训练目标: 最小化 Bradley-Terry 损失
    """

    def __init__(self, input_dim: int, config: Optional[RLHFConfig] = None):
        self.config = config or RLHFConfig()
        self.network = RewardNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.reward_model_hidden_dim,
            num_layers=self.config.reward_model_num_layers,
            dropout=self.config.reward_model_dropout
        )
        self.loss_fn = BradleyTerryLoss(margin=self.config.reward_model_margin)

        # Adam优化器状态
        self._adam_t = 0  # 时间步
        self._adam_m: Dict[str, List] = {}  # 一阶矩
        self._adam_v: Dict[str, List] = {}  # 二阶矩
        self._adam_initialized = False

        # 训练统计
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        self.num_samples = 0

    def _init_adam_state(self):
        """初始化Adam优化器状态"""
        self._adam_t = 0
        self._adam_m = {}
        self._adam_v = {}
        param_idx = 0
        for i in range(self.network.num_layers):
            # 权重
            key_w = f"layer_{i}_w"
            self._adam_m[key_w] = zeros_like(self.network.layers[i].weights)
            self._adam_v[key_w] = zeros_like(self.network.layers[i].weights)
            # 偏置
            key_b = f"layer_{i}_b"
            self._adam_m[key_b] = [0.0] * len(self.network.layers[i].biases)
            self._adam_v[key_b] = [0.0] * len(self.network.layers[i].biases)
            # LayerNorm gamma
            key_g = f"norm_{i}_g"
            self._adam_m[key_g] = [0.0] * len(self.network.norms[i].gamma)
            self._adam_v[key_g] = [0.0] * len(self.network.norms[i].gamma)
            # LayerNorm beta
            key_b2 = f"norm_{i}_b"
            self._adam_m[key_b2] = [0.0] * len(self.network.norms[i].beta)
            self._adam_v[key_b2] = [0.0] * len(self.network.norms[i].beta)
        # 输出层
        key_ow = "output_w"
        self._adam_m[key_ow] = zeros_like(self.network.output_layer.weights)
        self._adam_v[key_ow] = zeros_like(self.network.output_layer.weights)
        key_ob = "output_b"
        self._adam_m[key_ob] = [0.0] * len(self.network.output_layer.biases)
        self._adam_v[key_ob] = [0.0] * len(self.network.output_layer.biases)
        self._adam_initialized = True

    def _adam_update_matrix(self, param: List[List[float]], grad: List[List[float]],
                            m: List[List[float]], v: List[List[float]]) -> List[List[float]]:
        """Adam更新矩阵参数"""
        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        eps = self.config.adam_epsilon
        lr = self.config.reward_model_lr
        weight_decay = self.config.reward_model_weight_decay

        new_param = []
        for i in range(len(param)):
            new_row = []
            for j in range(len(param[i])):
                g = grad[i][j] + weight_decay * param[i][j]
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g
                m_hat = m[i][j] / (1 - beta1 ** self._adam_t)
                v_hat = v[i][j] / (1 - beta2 ** self._adam_t)
                new_val = param[i][j] - lr * m_hat / (math.sqrt(v_hat) + eps)
                new_row.append(new_val)
            new_param.append(new_row)
        return new_param

    def _adam_update_vector(self, param: List[float], grad: List[float],
                            m: List[float], v: List[float]) -> List[float]:
        """Adam更新向量参数"""
        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        eps = self.config.adam_epsilon
        lr = self.config.reward_model_lr
        weight_decay = self.config.reward_model_weight_decay

        new_param = []
        for i in range(len(param)):
            g = grad[i] + weight_decay * param[i]
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat = m[i] / (1 - beta1 ** self._adam_t)
            v_hat = v[i] / (1 - beta2 ** self._adam_t)
            new_val = param[i] - lr * m_hat / (math.sqrt(v_hat) + eps)
            new_param.append(new_val)
        return new_param

    def _clip_gradient_matrix(self, grad: List[List[float]], max_norm: float) -> List[List[float]]:
        """梯度裁剪 (按全局范数)"""
        total_norm_sq = 0.0
        for row in grad:
            for g in row:
                total_norm_sq += g * g
        total_norm = math.sqrt(total_norm_sq)
        if total_norm > max_norm:
            scale = max_norm / total_norm
            return [[g * scale for g in row] for row in grad]
        return grad

    def _clip_gradient_vector(self, grad: List[float], max_norm: float) -> List[float]:
        """梯度裁剪 (按全局范数)"""
        total_norm = vec_norm(grad)
        if total_norm > max_norm:
            scale = max_norm / total_norm
            return [g * scale for g in grad]
        return grad

    def forward(self, prompt_features: List[float], response_features: List[float]) -> float:
        """
        前向传播: 计算奖励值

        将prompt和response特征拼接后输入网络

        Args:
            prompt_features: prompt的特征向量
            response_features: response的特征向量

        Returns:
            标量奖励值
        """
        combined = prompt_features + response_features
        return self.network.forward(combined)

    def train_step(self, prompt_features: List[float],
                   chosen_features: List[float],
                   rejected_features: List[float]) -> Dict[str, float]:
        """
        单步训练

        1. 前向传播计算 r_chosen 和 r_rejected
        2. 计算 Bradley-Terry 损失
        3. 反向传播计算梯度
        4. Adam优化器更新参数

        Args:
            prompt_features: prompt特征
            chosen_features: 被偏好回答的特征
            rejected_features: 被拒绝回答的特征

        Returns:
            训练指标字典
        """
        if not self._adam_initialized:
            self._init_adam_state()

        self._adam_t += 1

        # 1. 前向传播
        combined_chosen = prompt_features + chosen_features
        combined_rejected = prompt_features + rejected_features

        r_chosen = self.network.forward(combined_chosen)
        r_rejected = self.network.forward(combined_rejected)

        # 2. 计算损失
        loss = self.loss_fn.compute_loss(r_chosen, r_rejected)
        accuracy = self.loss_fn.compute_accuracy(r_chosen, r_rejected)

        # 3. 计算梯度
        grad_chosen, grad_rejected = self.loss_fn.compute_gradients(r_chosen, r_rejected)

        # 反向传播 chosen
        self.network.zero_grad()
        combined_chosen_input = prompt_features + chosen_features
        self.network.forward(combined_chosen_input)
        self.network.backward(grad_chosen)

        # 保存chosen的梯度
        saved_grads = []
        for i in range(self.network.num_layers):
            saved_grads.append((
                [row[:] for row in self.network.layers[i].grad_weights],
                self.network.layers[i].grad_biases[:],
                self.network.norms[i].grad_gamma[:],
                self.network.norms[i].grad_beta[:]
            ))
        saved_output_grad_w = [row[:] for row in self.network.output_layer.grad_weights]
        saved_output_grad_b = self.network.output_layer.grad_biases[:]

        # 反向传播 rejected 并累加梯度
        self.network.zero_grad()
        combined_rejected_input = prompt_features + rejected_features
        self.network.forward(combined_rejected_input)
        self.network.backward(grad_rejected)

        # 累加梯度
        for i in range(self.network.num_layers):
            sg_w, sg_b, sg_g, sg_b2 = saved_grads[i]
            for ii in range(len(self.network.layers[i].grad_weights)):
                for jj in range(len(self.network.layers[i].grad_weights[ii])):
                    self.network.layers[i].grad_weights[ii][jj] += sg_w[ii][jj]
            for ii in range(len(self.network.layers[i].grad_biases)):
                self.network.layers[i].grad_biases[ii] += sg_b[ii]
            for ii in range(len(self.network.norms[i].grad_gamma)):
                self.network.norms[i].grad_gamma[ii] += sg_g[ii]
            for ii in range(len(self.network.norms[i].grad_beta)):
                self.network.norms[i].grad_beta[ii] += sg_b2[ii]
        for ii in range(len(self.network.output_layer.grad_weights)):
            for jj in range(len(self.network.output_layer.grad_weights[ii])):
                self.network.output_layer.grad_weights[ii][jj] += saved_output_grad_w[ii][jj]
        for ii in range(len(self.network.output_layer.grad_biases)):
            self.network.output_layer.grad_biases[ii] += saved_output_grad_b[ii]

        # 4. 梯度裁剪
        clip_norm = self.config.reward_model_gradient_clip
        for i in range(self.network.num_layers):
            self.network.layers[i].grad_weights = self._clip_gradient_matrix(
                self.network.layers[i].grad_weights, clip_norm)
            self.network.layers[i].grad_biases = self._clip_gradient_vector(
                self.network.layers[i].grad_biases, clip_norm)

        # 5. Adam更新
        for i in range(self.network.num_layers):
            key_w = f"layer_{i}_w"
            key_b = f"layer_{i}_b"
            key_g = f"norm_{i}_g"
            key_b2 = f"norm_{i}_b"
            self.network.layers[i].weights = self._adam_update_matrix(
                self.network.layers[i].weights, self.network.layers[i].grad_weights,
                self._adam_m[key_w], self._adam_v[key_w])
            self.network.layers[i].biases = self._adam_update_vector(
                self.network.layers[i].biases, self.network.layers[i].grad_biases,
                self._adam_m[key_b], self._adam_v[key_b])
            self.network.norms[i].gamma = self._adam_update_vector(
                self.network.norms[i].gamma, self.network.norms[i].grad_gamma,
                self._adam_m[key_g], self._adam_v[key_g])
            self.network.norms[i].beta = self._adam_update_vector(
                self.network.norms[i].beta, self.network.norms[i].grad_beta,
                self._adam_m[key_b2], self._adam_v[key_b2])

        key_ow = "output_w"
        key_ob = "output_b"
        self.network.output_layer.weights = self._adam_update_matrix(
            self.network.output_layer.weights, self.network.output_layer.grad_weights,
            self._adam_m[key_ow], self._adam_v[key_ow])
        self.network.output_layer.biases = self._adam_update_vector(
            self.network.output_layer.biases, self.network.output_layer.grad_biases,
            self._adam_m[key_ob], self._adam_v[key_ob])

        # 更新统计
        self.total_loss += loss
        self.total_accuracy += accuracy
        self.num_samples += 1

        return {
            "loss": loss,
            "accuracy": accuracy,
            "reward_chosen": r_chosen,
            "reward_rejected": r_rejected,
            "reward_margin": r_chosen - r_rejected
        }

    def train_epoch(self, preference_data: List[Tuple[List[float], List[float], List[float]]]
                    ) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            preference_data: [(prompt_features, chosen_features, rejected_features), ...]

        Returns:
            平均训练指标
        """
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        self.num_samples = 0

        # 随机打乱数据
        indices = list(range(len(preference_data)))
        random.shuffle(indices)

        batch_size = self.config.reward_model_batch_size
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            for idx in batch_indices:
                prompt_feat, chosen_feat, rejected_feat = preference_data[idx]
                self.train_step(prompt_feat, chosen_feat, rejected_feat)

        avg_loss = self.total_loss / max(self.num_samples, 1)
        avg_acc = self.total_accuracy / max(self.num_samples, 1)

        return {
            "avg_loss": avg_loss,
            "avg_accuracy": avg_acc,
            "num_samples": self.num_samples
        }

    def predict(self, prompt_features: List[float],
                response_features: List[float]) -> float:
        """推理模式: 计算奖励值 (关闭dropout)"""
        for d in self.network.dropouts:
            d.training = False
        reward = self.forward(prompt_features, response_features)
        for d in self.network.dropouts:
            d.training = True
        return reward


# ============================================================================
# 3. KLController - KL散度控制器
# ============================================================================

class KLController(ABC):
    """
    KL散度控制器抽象基类

    KL(π_θ || π_ref) = E_{x~π_θ}[log π_θ(x) - log π_ref(x)]

    控制器的目标: 调整KL惩罚系数 β，使得实际KL散度接近目标值
    """

    @abstractmethod
    def compute_kl_penalty(self, log_probs_policy: List[float],
                           log_probs_ref: List[float]) -> float:
        """
        计算KL散度惩罚

        KL(p||q) = Σ p_i * (log(p_i) - log(q_i))

        Args:
            log_probs_policy: 策略模型的log概率
            log_probs_ref: 参考模型的log概率

        Returns:
            KL散度值
        """
        pass

    def compute_kl_divergence(self, log_probs_policy: List[float],
                              log_probs_ref: List[float]) -> float:
        """
        计算KL散度（别名方法）

        默认实现调用 compute_kl_penalty，子类可重写以提供不同的计算逻辑。

        KL(π_θ || π_ref) = Σ exp(log_p_i) * (log_p_i - log_q_i)

        使用数值稳定的方式计算，避免数值溢出：
            对 log_p - log_q 进行截断，限制在 [-20, 20] 范围内

        Args:
            log_probs_policy: 策略模型的log概率列表
            log_probs_ref: 参考模型的log概率列表

        Returns:
            KL散度值（非负数）
        """
        if len(log_probs_policy) != len(log_probs_ref):
            raise ValueError(
                f"log_probs_policy 和 log_probs_ref 长度不一致: "
                f"{len(log_probs_policy)} vs {len(log_probs_ref)}"
            )

        kl_total = 0.0
        for log_p, log_q in zip(log_probs_policy, log_probs_ref):
            # 计算逐token的KL散度分量
            diff = log_p - log_q
            # 截断防止数值溢出
            diff = max(-20.0, min(20.0, diff))
            # p_i = exp(log_p_i)
            p = math.exp(log_p)
            kl_total += p * diff

        # 确保KL散度非负（数值误差可能导致微小负值）
        return max(0.0, kl_total)

    @abstractmethod
    def update(self, kl_divergence: float) -> float:
        """
        根据当前KL散度更新惩罚系数

        Args:
            kl_divergence: 当前观测到的KL散度

        Returns:
            更新后的KL系数 β
        """
        pass

    @abstractmethod
    def get_kl_coeff(self) -> float:
        """获取当前KL系数"""
        pass


class AdaptiveKLController(KLController):
    """
    自适应KL控制器

    基于PPO论文中的自适应机制:
        β_{t+1} = β_t * (1 + δ)  if KL > KL_target * (1 + tolerance)
        β_{t+1} = β_t / (1 + δ)  if KL < KL_target / (1 + tolerance)
        否则 β_{t+1} = β_t

    其中 δ 是调整步长，tolerance 是容差比例。

    使用指数移动平均来平滑KL散度估计:
        KL_ema = α * KL_ema + (1 - α) * KL_current
    """

    def __init__(self, init_kl_coeff: float = 0.1, target_kl: float = 6.0,
                 horizon: int = 10000):
        self.kl_coeff = init_kl_coeff
        self.target_kl = target_kl
        self.horizon = horizon
        self.step_count = 0

        # 指数移动平均
        self.kl_ema: Optional[float] = None
        self.ema_alpha = 0.01  # 平滑系数

        # 调整参数
        self.delta = 1.5  # 调整步长因子
        self.tolerance = 0.1  # 容差比例

        # 历史记录
        self.kl_history: List[float] = []
        self.coeff_history: List[float] = []

    def compute_kl_penalty(self, log_probs_policy: List[float],
                           log_probs_ref: List[float]) -> float:
        """
        计算KL散度: KL(π_θ || π_ref) = Σ p_i * (log(p_i) - log(q_i))

        使用数值稳定的计算方式:
        对于每个token位置t:
            kl_t = exp(log_p_t) * (log_p_t - log_q_t)
        KL = Σ_t kl_t

        Args:
            log_probs_policy: 策略log概率列表
            log_probs_ref: 参考log概率列表

        Returns:
            KL散度值
        """
        assert len(log_probs_policy) == len(log_probs_ref), \
            "log_probs_policy and log_probs_ref must have same length"

        kl_total = 0.0
        for log_p, log_q in zip(log_probs_policy, log_probs_ref):
            # 数值稳定: 当 log_p - log_q 过大时截断
            diff = log_p - log_q
            diff = clip(diff, -20.0, 20.0)
            p = math.exp(log_p)
            kl_total += p * diff

        # 更新EMA
        if self.kl_ema is None:
            self.kl_ema = kl_total
        else:
            self.kl_ema = self.ema_alpha * kl_total + (1 - self.ema_alpha) * self.kl_ema

        self.kl_history.append(kl_total)
        self.step_count += 1

        return kl_total

    def update(self, kl_divergence: float) -> float:
        """
        根据当前KL散度自适应更新惩罚系数

        策略:
        - 如果 KL > target * (1 + tolerance): 增大 β (惩罚更重)
        - 如果 KL < target / (1 + tolerance): 减小 β (允许更多偏离)
        - 否则: 保持不变

        调整幅度与偏离程度成正比:
            factor = 1 + delta * |KL - target| / target

        Args:
            kl_divergence: 当前KL散度

        Returns:
            更新后的KL系数
        """
        target = self.target_kl
        tol = self.tolerance

        if kl_divergence > target * (1 + tol):
            # KL过大，增大惩罚
            ratio = kl_divergence / target
            factor = 1.0 + self.delta * (ratio - 1.0)
            factor = min(factor, 2.0)  # 限制最大增长
            self.kl_coeff *= factor
        elif kl_divergence < target / (1 + tol):
            # KL过小，减小惩罚
            ratio = target / max(kl_divergence, 1e-8)
            factor = 1.0 + self.delta * (ratio - 1.0)
            factor = min(factor, 2.0)
            self.kl_coeff /= factor
        # 否则保持不变

        # 限制KL系数范围
        self.kl_coeff = clip(self.kl_coeff, 1e-6, 100.0)

        self.coeff_history.append(self.kl_coeff)

        return self.kl_coeff

    def get_kl_coeff(self) -> float:
        """获取当前KL系数"""
        return self.kl_coeff

    def get_kl_stats(self) -> Dict[str, float]:
        """获取KL统计信息"""
        return {
            "current_kl_coeff": self.kl_coeff,
            "target_kl": self.target_kl,
            "kl_ema": self.kl_ema if self.kl_ema is not None else 0.0,
            "recent_kl": self.kl_history[-1] if self.kl_history else 0.0,
            "mean_kl": sum(self.kl_history[-100:]) / max(len(self.kl_history[-100:]), 1),
            "step_count": self.step_count
        }


class FixedKLController(KLController):
    """
    固定KL控制器

    使用恒定的KL惩罚系数 β，不进行自适应调整。
    适用于对KL约束要求不严格或已知最优β的场景。
    """

    def __init__(self, kl_coeff: float = 0.1):
        self.kl_coeff = kl_coeff
        self.total_kl = 0.0
        self.kl_count = 0
        self.kl_history: List[float] = []

    def compute_kl_penalty(self, log_probs_policy: List[float],
                           log_probs_ref: List[float]) -> float:
        """
        计算KL散度: KL(π_θ || π_ref) = Σ p_i * (log(p_i) - log(q_i))

        Args:
            log_probs_policy: 策略log概率
            log_probs_ref: 参考log概率

        Returns:
            KL散度值
        """
        kl_total = 0.0
        for log_p, log_q in zip(log_probs_policy, log_probs_ref):
            diff = log_p - log_q
            diff = clip(diff, -20.0, 20.0)
            p = math.exp(log_p)
            kl_total += p * diff

        self.total_kl += kl_total
        self.kl_count += 1
        self.kl_history.append(kl_total)

        return kl_total

    def update(self, kl_divergence: float) -> float:
        """
        固定控制器不做更新

        Args:
            kl_divergence: 当前KL散度 (忽略)

        Returns:
            固定的KL系数
        """
        return self.kl_coeff

    def get_kl_coeff(self) -> float:
        """获取固定的KL系数"""
        return self.kl_coeff

    def get_kl_stats(self) -> Dict[str, float]:
        """获取KL统计信息"""
        return {
            "current_kl_coeff": self.kl_coeff,
            "mean_kl": self.total_kl / max(self.kl_count, 1),
            "total_kl": self.total_kl,
            "kl_count": self.kl_count
        }


# ============================================================================
# 4. PPOTrainer - PPO训练器
# ============================================================================

@dataclass
class RolloutSample:
    """单个rollout样本"""
    prompt_features: List[float]
    response_features: List[float]
    action_log_probs: List[float]  # 每个token的log概率
    rewards: List[float]  # 每步的奖励
    values: List[float]  # 每步的价值估计
    dones: List[bool]  # 是否结束
    ref_log_probs: List[float]  # 参考模型的log概率


class PolicyRolloutBuffer:
    """
    PPO策略rollout缓冲区

    存储 (prompt, response, log_prob, reward, value) 用于PPO更新

    数据结构:
        - prompts: List[List[float]]  - prompt特征
        - responses: List[List[float]] - response特征
        - action_log_probs: List[List[float]] - 策略log概率
        - ref_log_probs: List[List[float]] - 参考log概率
        - rewards: List[List[float]] - 每步奖励
        - values: List[List[float]] - 价值估计
        - dones: List[List[bool]] - 终止标志
        - advantages: List[List[float]] - 优势函数 (计算后填充)
        - returns: List[List[float]] - 回报 (计算后填充)
    """

    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
        self.prompts: List[List[float]] = []
        self.responses: List[List[float]] = []
        self.action_log_probs: List[List[float]] = []
        self.ref_log_probs: List[List[float]] = []
        self.rewards: List[List[float]] = []
        self.values: List[List[float]] = []
        self.dones: List[List[bool]] = []
        self.advantages: List[List[float]] = []
        self.returns: List[List[float]] = []

    def push(self, sample: RolloutSample):
        """添加一个rollout样本"""
        if len(self.prompts) >= self.buffer_size:
            # 替换最旧的样本
            self.prompts.pop(0)
            self.responses.pop(0)
            self.action_log_probs.pop(0)
            self.ref_log_probs.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.dones.pop(0)
            self.advantages.pop(0)
            self.returns.pop(0)

        self.prompts.append(sample.prompt_features)
        self.responses.append(sample.response_features)
        self.action_log_probs.append(sample.action_log_probs)
        self.ref_log_probs.append(sample.ref_log_probs)
        self.rewards.append(sample.rewards)
        self.values.append(sample.values)
        self.dones.append(sample.dones)
        self.advantages.append([])
        self.returns.append([])

    def clear(self):
        """清空缓冲区"""
        self.prompts.clear()
        self.responses.clear()
        self.action_log_probs.clear()
        self.ref_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

    def __len__(self):
        return len(self.prompts)

    def is_ready(self) -> bool:
        """缓冲区是否有足够数据"""
        return len(self.prompts) >= self.buffer_size // 2


class GeneralizedAdvantageEstimation:
    """
    广义优势估计 (GAE)

    GAE(γ, λ) 计算公式:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - d_t) - V(s_t)
        A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}

    其中:
        - γ (gamma): 折扣因子
        - λ (gae_lambda): GAE参数，控制偏差-方差权衡
        - δ_t: TD误差
        - d_t: 终止标志
        - V(s_t): 状态价值

    λ=1 时退化为蒙特卡洛估计 (无偏差，高方差)
    λ=0 时退化为TD(0) (有偏差，低方差)
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute(self, rewards: List[float], values: List[float],
                dones: List[bool], next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        计算GAE优势和回报

        算法 (反向递推):
            gae_t = 0
            for t = T-1, T-2, ..., 0:
                δ_t = r_t + γ * V(s_{t+1}) * (1 - d_t) - V(s_t)
                gae_t = δ_t + γ * λ * (1 - d_t) * gae_t
                A_t = gae_t
                R_t = A_t + V(s_t)

        Args:
            rewards: 每步奖励 [r_0, r_1, ..., r_{T-1}]
            values: 每步价值 [V(s_0), V(s_1), ..., V(s_{T-1})]
            dones: 终止标志 [d_0, d_1, ..., d_{T-1}]
            next_value: 最后一步之后的状态价值 V(s_T)

        Returns:
            (advantages, returns) 优势和回报列表
        """
        T = len(rewards)
        assert T == len(values) == len(dones), "rewards, values, dones must have same length"

        advantages = [0.0] * T
        returns = [0.0] * T

        gae = 0.0
        for t in reversed(range(T)):
            # 下一个状态的价值
            if t == T - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]

            # 是否终止
            mask = 0.0 if dones[t] else 1.0

            # TD误差: δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
            delta = rewards[t] + self.gamma * next_v * mask - values[t]

            # GAE递推: A_t = δ_t + γ * λ * mask * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

            # 回报: R_t = A_t + V(s_t)
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def compute_batch(self, batch_rewards: List[List[float]],
                      batch_values: List[List[float]],
                      batch_dones: List[List[bool]]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        批量计算GAE

        Args:
            batch_rewards: 批次奖励
            batch_values: 批次价值
            batch_dones: 批次终止标志

        Returns:
            (batch_advantages, batch_returns)
        """
        batch_advantages = []
        batch_returns = []
        for rewards, values, dones in zip(batch_rewards, batch_values, batch_dones):
            advs, rets = self.compute(rewards, values, dones)
            batch_advantages.append(advs)
            batch_returns.append(rets)
        return batch_advantages, batch_returns


class PPOActorCritic:
    """
    PPO Actor-Critic 网络

    架构:
        共享骨干: [Linear -> GELU -> LayerNorm] * N
        策略头 (Actor): Linear -> Softmax -> log概率
        价值头 (Critic): Linear -> 标量价值

    前向传播:
        h = Backbone(x)
        logits = Actor(h)  -> π(a|s) = softmax(logits)
        value = Critic(h)  -> V(s)
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512,
                 num_layers: int = 3):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 共享骨干网络
        self.backbone_layers: List[LinearLayer] = []
        self.backbone_norms: List[LayerNormModule] = []
        self.backbone_activations: List[GELUActivation] = []

        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.backbone_layers.append(LinearLayer(in_d, hidden_dim))
            self.backbone_norms.append(LayerNormModule(hidden_dim))
            self.backbone_activations.append(GELUActivation())

        # 策略头 (Actor)
        self.policy_head = LinearLayer(hidden_dim, action_dim)

        # 价值头 (Critic)
        self.value_head = LinearLayer(hidden_dim, 1)

        # 缓存
        self._backbone_outputs: List[List[float]] = []
        self._last_logits: Optional[List[float]] = None
        self._last_value: Optional[float] = None
        self._last_log_probs: Optional[List[float]] = None

    def forward(self, x: List[float]) -> Tuple[List[float], float, List[float]]:
        """
        前向传播

        Args:
            x: 输入特征

        Returns:
            (log_probs, value, logits) 策略log概率、价值、原始logits
        """
        self._backbone_outputs = []

        # 共享骨干
        h = x[:]
        for i in range(len(self.backbone_layers)):
            h = self.backbone_layers[i].forward(h)
            h = self.backbone_norms[i].forward(h)
            h = self.backbone_activations[i].forward(h)
            self._backbone_outputs.append(h[:])

        # 策略头
        logits = self.policy_head.forward(h)
        log_probs = log_softmax(logits)
        probs = softmax(logits)

        # 价值头
        value = self.value_head.forward(h)[0]

        self._last_logits = logits[:]
        self._last_value = value
        self._last_log_probs = log_probs[:]

        return log_probs, value, logits

    def evaluate_actions(self, x: List[float], actions: List[int]
                         ) -> Tuple[List[float], float, float]:
        """
        评估给定动作的log概率和价值

        Args:
            x: 输入特征
            actions: 动作索引列表

        Returns:
            (action_log_probs, value, entropy)
        """
        log_probs, value, logits = self.forward(x)
        action_log_probs = [log_probs[a] for a in actions]

        # 计算策略熵
        probs = softmax(logits)
        ent = entropy_from_probs(probs)

        return action_log_probs, value, ent

    def get_action(self, x: List[float]) -> Tuple[int, float, float]:
        """
        采样动作

        Args:
            x: 输入特征

        Returns:
            (action, log_prob, value)
        """
        log_probs, value, logits = self.forward(x)
        probs = softmax(logits)

        # 按概率采样
        r = random.random()
        cumsum = 0.0
        action = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                action = i
                break

        return action, log_probs[action], value

    def backward_policy(self, grad_log_probs: List[float]) -> List[float]:
        """
        策略头反向传播

        Args:
            grad_log_probs: log概率的梯度

        Returns:
            骨干输出的梯度
        """
        # log_softmax的梯度
        probs = softmax(self._last_logits)
        grad_logits = [0.0] * self.action_dim
        for i in range(self.action_dim):
            for j in range(self.action_dim):
                if i == j:
                    grad_logits[i] += grad_log_probs[j] * probs[i] * (1.0 - probs[i])
                else:
                    grad_logits[i] += grad_log_probs[j] * (-probs[i] * probs[j])

        # 策略头反向
        grad = self.policy_head.backward(grad_logits)
        return grad

    def backward_value(self, grad_value: float) -> List[float]:
        """
        价值头反向传播

        Args:
            grad_value: 价值的梯度

        Returns:
            骨干输出的梯度
        """
        return self.value_head.backward([grad_value])

    def backward_backbone(self, grad: List[float]) -> List[float]:
        """
        骨干网络反向传播

        Args:
            grad: 骨干输出的梯度

        Returns:
            输入的梯度
        """
        for i in range(len(self.backbone_layers) - 1, -1, -1):
            grad = self.backbone_activations[i].backward(grad)
            grad = self.backbone_norms[i].backward(grad)
            grad = self.backbone_layers[i].backward(grad)
        return grad

    def zero_grad(self):
        """清零所有梯度"""
        for layer in self.backbone_layers:
            layer.grad_weights = zeros([layer.out_features, layer.in_features])
            layer.grad_biases = [0.0] * layer.out_features
        for norm in self.backbone_norms:
            norm.grad_gamma = [0.0] * norm.dim
            norm.grad_beta = [0.0] * norm.dim
        self.policy_head.grad_weights = zeros([self.policy_head.out_features, self.policy_head.in_features])
        self.policy_head.grad_biases = [0.0] * self.policy_head.out_features
        self.value_head.grad_weights = zeros([self.value_head.out_features, self.value_head.in_features])
        self.value_head.grad_biases = [0.0]


class PPOClipObjective:
    """
    PPO-Clip 目标函数

    核心公式:
        L^{CLIP}(θ) = E_t[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

    其中:
        r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) = exp(log_π_θ - log_π_{θ_old})
        A_t: 优势函数
        ε: 裁剪范围

    总损失:
        L(θ) = -L^{CLIP}(θ) + c_1 * L^{VF}(θ) - c_2 * H[π_θ](s_t)

    其中:
        L^{VF}(θ) = (V_θ(s_t) - V_t^{target})^2  (价值损失)
        H[π_θ](s_t): 策略熵 (鼓励探索)
        c_1: 价值损失系数
        c_2: 熵系数
    """

    def __init__(self, clip_epsilon: float = 0.2, value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01, clip_value_loss: bool = True,
                 value_clip_epsilon: float = 0.2):
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_value_loss = clip_value_loss
        self.value_clip_epsilon = value_clip_epsilon

    def compute_ratio(self, new_log_prob: float, old_log_prob: float) -> float:
        """
        计算重要性采样比率: r(θ) = exp(log_π_new - log_π_old)

        Args:
            new_log_prob: 新策略的log概率
            old_log_prob: 旧策略的log概率

        Returns:
            重要性采样比率
        """
        return math.exp(new_log_prob - old_log_prob)

    def compute_policy_loss(self, ratio: float, advantage: float) -> Tuple[float, float]:
        """
        计算PPO-Clip策略损失

        L^{CLIP} = min(r * A, clip(r, 1-ε, 1+ε) * A)

        注意: 当 A > 0 时，我们希望 r 接近1 (不增大太多)
              当 A < 0 时，我们希望 r 接近1 (不减小太多)

        Args:
            ratio: 重要性采样比率 r(θ)
            advantage: 优势函数值 A

        Returns:
            (clipped_loss, unclipped_loss) 裁剪后的损失和未裁剪的损失
        """
        # 未裁剪目标: r * A
        unclipped = ratio * advantage

        # 裁剪后的比率
        clipped_ratio = clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

        # 裁剪目标: clip(r) * A
        clipped = clipped_ratio * advantage

        # PPO目标: 取最小值 (悲观估计)
        ppo_objective = min(unclipped, clipped)

        # 损失是目标的负值 (因为我们要最大化目标)
        loss = -ppo_objective

        return loss, -unclipped

    def compute_value_loss(self, predicted_value: float, target_value: float,
                           old_value: Optional[float] = None) -> float:
        """
        计算价值函数损失

        L^{VF} = (V_θ(s) - V^{target})^2

        可选裁剪:
            L^{VF,clip} = max((V - V^{target})^2,
                             (clip(V, V_old - ε, V_old + ε) - V^{target})^2)

        Args:
            predicted_value: 预测价值
            target_value: 目标价值 (GAE回报)
            old_value: 旧价值估计 (用于裁剪)

        Returns:
            价值损失
        """
        value_loss = (predicted_value - target_value) ** 2

        if self.clip_value_loss and old_value is not None:
            clipped_value = clip(predicted_value,
                                 old_value - self.value_clip_epsilon,
                                 old_value + self.value_clip_epsilon)
            clipped_loss = (clipped_value - target_value) ** 2
            value_loss = max(value_loss, clipped_loss)

        return value_loss

    def compute_total_loss(self, policy_loss: float, value_loss: float,
                           entropy: float, kl_penalty: float = 0.0) -> Tuple[float, Dict[str, float]]:
        """
        计算总损失

        L(θ) = L^{CLIP} + c_1 * L^{VF} - c_2 * H[π] + β * KL

        Args:
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy: 策略熵
            kl_penalty: KL散度惩罚

        Returns:
            (total_loss, loss_components)
        """
        total = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy + kl_penalty

        components = {
            "total_loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": -self.entropy_coeff * entropy,
            "entropy": entropy,
            "kl_penalty": kl_penalty,
            "approx_kl": 0.0  # 将在外部计算
        }

        return total, components

    def compute_approx_kl(self, new_log_probs: List[float],
                          old_log_probs: List[float]) -> float:
        """
        计算近似KL散度 (用于监控)

        KL_approx ≈ (r(θ) - 1) - log(r(θ))
                  = (exp(log_new - log_old) - 1) - (log_new - log_old)

        这是KL散度的一阶泰勒近似

        Args:
            new_log_probs: 新策略log概率
            old_log_probs: 旧策略log概率

        Returns:
            近似KL散度
        """
        kl = 0.0
        for new_lp, old_lp in zip(new_log_probs, old_log_probs):
            log_ratio = new_lp - old_lp
            ratio = math.exp(log_ratio)
            kl += (ratio - 1.0) - log_ratio
        return kl / max(len(new_log_probs), 1)


class PPOTrainer:
    """
    PPO训练器

    完整的PPO训练循环:
    1. 收集rollout数据 (使用当前策略)
    2. 计算GAE优势和回报
    3. 多轮PPO更新 (mini-batch SGD)
    4. 更新KL控制器

    训练流程:
        for epoch in range(num_epochs):
            for step in range(num_steps):
                # 收集rollout
                rollout = collect_rollout(policy)
                buffer.push(rollout)

                # 计算GAE
                advantages, returns = gae.compute(rewards, values, dones)

                # PPO更新
                for ppo_epoch in range(ppo_epochs):
                    for mini_batch in get_mini_batches(buffer):
                        # 前向传播
                        new_log_probs, new_values, entropy = actor_critic.evaluate(mini_batch)
                        # 计算比率
                        ratios = exp(new_log_probs - old_log_probs)
                        # PPO-Clip损失
                        policy_loss = ppo_clip(ratios, advantages)
                        value_loss = mse(new_values, returns)
                        # 总损失
                        loss = policy_loss + c1*value_loss - c2*entropy
                        # 反向传播 + 更新
                        loss.backward()
                        optimizer.step()
    """

    def __init__(self, input_dim: int, action_dim: int, config: Optional[RLHFConfig] = None):
        self.config = config or RLHFConfig()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # 核心组件
        self.actor_critic = PPOActorCritic(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        )
        self.buffer = PolicyRolloutBuffer(buffer_size=self.config.num_rollout_steps)
        self.gae = GeneralizedAdvantageEstimation(
            gamma=self.config.ppo_gamma,
            gae_lambda=self.config.ppo_gae_lambda
        )
        self.ppo_objective = PPOClipObjective(
            clip_epsilon=self.config.ppo_clip_epsilon,
            value_coeff=self.config.ppo_value_coeff,
            entropy_coeff=self.config.ppo_entropy_coeff,
            clip_value_loss=self.config.ppo_clip_value_loss,
            value_clip_epsilon=self.config.ppo_value_clip_epsilon
        )

        # KL控制器
        if self.config.kl_strategy == "adaptive":
            self.kl_controller = AdaptiveKLController(
                init_kl_coeff=self.config.kl_coeff,
                target_kl=self.config.kl_target,
                horizon=self.config.kl_horizon
            )
        else:
            self.kl_controller = FixedKLController(kl_coeff=self.config.kl_coeff)

        # Adam优化器状态
        self._policy_adam_t = 0
        self._value_adam_t = 0
        self._policy_adam_initialized = False
        self._value_adam_initialized = False
        self._adam_m: Dict[str, Any] = {}
        self._adam_v: Dict[str, Any] = {}

        # 训练统计
        self.training_stats: List[Dict[str, float]] = []
        self.step_count = 0

    def _init_adam(self):
        """初始化Adam优化器"""
        self._adam_m = {}
        self._adam_v = {}

        ac = self.actor_critic
        for i, layer in enumerate(ac.backbone_layers):
            key_w = f"bb_{i}_w"
            key_b = f"bb_{i}_b"
            self._adam_m[key_w] = zeros_like(layer.weights)
            self._adam_v[key_w] = zeros_like(layer.weights)
            self._adam_m[key_b] = [0.0] * len(layer.biases)
            self._adam_v[key_b] = [0.0] * len(layer.biases)
            key_ng = f"bn_{i}_g"
            key_nb = f"bn_{i}_b"
            self._adam_m[key_ng] = [0.0] * len(ac.backbone_norms[i].gamma)
            self._adam_v[key_ng] = [0.0] * len(ac.backbone_norms[i].gamma)
            self._adam_m[key_nb] = [0.0] * len(ac.backbone_norms[i].beta)
            self._adam_v[key_nb] = [0.0] * len(ac.backbone_norms[i].beta)

        key_pw = "policy_w"
        key_pb = "policy_b"
        self._adam_m[key_pw] = zeros_like(ac.policy_head.weights)
        self._adam_v[key_pw] = zeros_like(ac.policy_head.weights)
        self._adam_m[key_pb] = [0.0] * len(ac.policy_head.biases)
        self._adam_v[key_pb] = [0.0] * len(ac.policy_head.biases)

        key_vw = "value_w"
        key_vb = "value_b"
        self._adam_m[key_vw] = zeros_like(ac.value_head.weights)
        self._adam_v[key_vw] = zeros_like(ac.value_head.weights)
        self._adam_m[key_vb] = [0.0] * len(ac.value_head.biases)
        self._adam_v[key_vb] = [0.0] * len(ac.value_head.biases)

        self._policy_adam_initialized = True
        self._value_adam_initialized = True

    def _adam_step_matrix(self, key: str, param: List[List[float]],
                          grad: List[List[float]], lr: float):
        """Adam更新矩阵参数"""
        if not self._policy_adam_initialized:
            self._init_adam()

        self._policy_adam_t += 1
        t = self._policy_adam_t
        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        eps = self.config.adam_epsilon

        m = self._adam_m[key]
        v = self._adam_v[key]

        for i in range(len(param)):
            for j in range(len(param[i])):
                g = grad[i][j]
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g
                m_hat = m[i][j] / (1 - beta1 ** t)
                v_hat = v[i][j] / (1 - beta2 ** t)
                param[i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

    def _adam_step_vector(self, key: str, param: List[float],
                          grad: List[float], lr: float):
        """Adam更新向量参数"""
        if not self._policy_adam_initialized:
            self._init_adam()

        t = self._policy_adam_t
        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        eps = self.config.adam_epsilon

        m = self._adam_m[key]
        v = self._adam_v[key]

        for i in range(len(param)):
            g = grad[i]
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            param[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

    def _clip_grad(self, grad: List[float], max_norm: float) -> List[float]:
        """梯度裁剪"""
        norm = vec_norm(grad)
        if norm > max_norm:
            scale = max_norm / norm
            return [g * scale for g in grad]
        return grad

    def _clip_grad_matrix(self, grad: List[List[float]], max_norm: float) -> List[List[float]]:
        """矩阵梯度裁剪"""
        total_sq = sum(g * g for row in grad for g in row)
        norm = math.sqrt(total_sq)
        if norm > max_norm:
            scale = max_norm / norm
            return [[g * scale for g in row] for row in grad]
        return grad

    def collect_rollout(self, prompts: List[List[float]],
                        reward_fn: Callable[[List[float], List[float]], float],
                        ref_policy_fn: Optional[Callable[[List[float]], List[float]]] = None,
                        num_steps: Optional[int] = None) -> List[RolloutSample]:
        """
        收集rollout数据

        使用当前策略生成response，计算奖励和价值估计

        Args:
            prompts: prompt特征列表
            reward_fn: 奖励函数 (prompt_features, response_features) -> reward
            ref_policy_fn: 参考策略函数 (用于计算KL)
            num_steps: rollout步数

        Returns:
            rollout样本列表
        """
        num = num_steps or self.config.num_rollout_steps
        samples = []

        for step_i in range(min(num, len(prompts))):
            prompt = prompts[step_i]

            # 使用策略采样response (简化: 使用随机特征模拟)
            # 在实际实现中，这里会是自回归生成
            response = self._generate_response(prompt)

            # 计算策略log概率和价值 (使用prompt作为状态输入)
            log_probs, value, logits = self.actor_critic.forward(prompt)

            # 参考策略log概率
            if ref_policy_fn is not None:
                ref_log_probs = ref_policy_fn(prompt)
            else:
                # 使用均匀分布作为参考
                ref_log_probs = [-math.log(self.action_dim)] * len(log_probs)

            # 计算奖励
            reward = reward_fn(prompt, response)

            # 创建rollout样本
            sample = RolloutSample(
                prompt_features=prompt,
                response_features=response,
                action_log_probs=[log_probs[0]],  # 简化: 取第一个token的log_prob
                rewards=[reward],
                values=[value],
                dones=[step_i == num - 1],
                ref_log_probs=[ref_log_probs[0]]
            )
            samples.append(sample)

        return samples

    def _generate_response(self, prompt: List[float]) -> List[float]:
        """
        使用当前策略生成response

        简化实现: 使用策略网络的输出作为response特征
        在实际实现中，这会是自回归token生成
        """
        # 使用策略网络生成response特征
        log_probs, value, logits = self.actor_critic.forward(prompt)
        probs = softmax(logits)

        # 采样response，维度与prompt相同 (因为奖励模型需要 prompt+response 拼接)
        response_dim = len(prompt)
        response = []
        for _ in range(response_dim):
            # 基于概率采样
            r = random.random()
            cumsum = 0.0
            val = 0.0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    val = logits[i] * 0.1
                    break
            response.append(val + random.gauss(0, 0.01))

        return response

    def compute_advantages_and_returns(self) -> None:
        """
        对缓冲区中的所有数据计算GAE优势和回报
        """
        for i in range(len(self.buffer)):
            rewards = self.buffer.rewards[i]
            values = self.buffer.values[i]
            dones = self.buffer.dones[i]

            advantages, returns = self.gae.compute(rewards, values, dones)
            self.buffer.advantages[i] = advantages
            self.buffer.returns[i] = returns

    def normalize_advantages(self):
        """归一化优势函数 (减均值除标准差)"""
        all_advantages = []
        for advs in self.buffer.advantages:
            all_advantages.extend(advs)

        if not all_advantages:
            return

        mean = sum(all_advantages) / len(all_advantages)
        var = sum((a - mean) ** 2 for a in all_advantages) / len(all_advantages)
        std = math.sqrt(var + 1e-8)

        for i in range(len(self.buffer.advantages)):
            self.buffer.advantages[i] = [(a - mean) / std for a in self.buffer.advantages[i]]

    def ppo_update_step(self, batch_indices: List[int]) -> Dict[str, float]:
        """
        单步PPO更新

        对mini-batch执行:
        1. 前向传播获取新策略的log概率和价值
        2. 计算重要性采样比率 r(θ) = exp(log_π_new - log_π_old)
        3. 计算PPO-Clip损失
        4. 计算价值损失
        5. 计算熵奖励
        6. 计算KL惩罚
        7. 反向传播和参数更新

        Args:
            batch_indices: mini-batch索引

        Returns:
            训练指标
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_approx_kl = 0.0
        num_samples = 0

        for idx in batch_indices:
            if idx >= len(self.buffer):
                continue

            prompt = self.buffer.prompts[idx]
            response = self.buffer.responses[idx]
            old_log_probs = self.buffer.action_log_probs[idx]
            ref_log_probs = self.buffer.ref_log_probs[idx]
            advantages = self.buffer.advantages[idx]
            returns = self.buffer.returns[idx]
            old_values = self.buffer.values[idx]

            if not advantages:
                continue

            # 1. 前向传播 (使用prompt作为状态输入)
            new_log_probs, new_value, logits = self.actor_critic.forward(prompt)
            probs = softmax(logits)
            entropy = entropy_from_probs(probs)

            # 2. 计算比率
            new_lp = new_log_probs[0] if new_log_probs else 0.0
            old_lp = old_log_probs[0] if old_log_probs else 0.0
            ratio = self.ppo_objective.compute_ratio(new_lp, old_lp)

            # 3. PPO-Clip策略损失
            advantage = advantages[0] if advantages else 0.0
            policy_loss, _ = self.ppo_objective.compute_policy_loss(ratio, advantage)

            # 4. 价值损失
            target_return = returns[0] if returns else 0.0
            old_val = old_values[0] if old_values else 0.0
            value_loss = self.ppo_objective.compute_value_loss(new_value, target_return, old_val)

            # 5. KL惩罚
            ref_lp = ref_log_probs[0] if ref_log_probs else 0.0
            kl_penalty = self.kl_controller.get_kl_coeff() * (
                new_lp - ref_lp
            )

            # 6. 总损失
            total_loss, components = self.ppo_objective.compute_total_loss(
                policy_loss, value_loss, entropy, kl_penalty
            )

            # 7. 反向传播
            self.actor_critic.zero_grad()

            # 策略梯度: dL/d(log_prob_a) * d(log_prob_a)/d(logits)
            # 其中 log_prob_a = log_softmax(logits)[a], a=0 (简化)
            # PPO-Clip梯度: dL/d(r) * dr/d(log_pi_new) = dL/d(r) * r
            # 其中 r = exp(log_pi_new - log_pi_old)
            ratio = self.ppo_objective.compute_ratio(new_lp, old_lp)
            if advantage >= 0:
                # 当A>0时，裁剪上限: grad = min(r, 1+eps) * A
                clipped_ratio = min(ratio, 1.0 + self.config.ppo_clip_epsilon)
                grad_policy = clipped_ratio * advantage
            else:
                # 当A<0时，裁剪下限: grad = max(r, 1-eps) * A
                clipped_ratio = max(ratio, 1.0 - self.config.ppo_clip_epsilon)
                grad_policy = clipped_ratio * advantage

            # 构建log_probs梯度向量 (仅对action=0有梯度)
            grad_log_probs = [0.0] * self.action_dim
            grad_log_probs[0] = grad_policy

            # 价值梯度
            grad_val = 2.0 * self.config.ppo_value_coeff * (new_value - target_return)

            # 反向传播策略
            grad_from_policy = self.actor_critic.backward_policy(grad_log_probs)
            grad_from_value = self.actor_critic.backward_value(grad_val)

            # 合并梯度到骨干
            combined_grad = vec_add(grad_from_policy, grad_from_value)
            self.actor_critic.backward_backbone(combined_grad)

            # 8. 参数更新
            self._update_parameters()

            # 统计
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy
            total_kl += abs(new_lp - ref_lp)
            num_samples += 1

        if num_samples == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / num_samples,
            "value_loss": total_value_loss / num_samples,
            "entropy": total_entropy / num_samples,
            "kl_divergence": total_kl / num_samples,
            "num_samples": num_samples
        }

    def _update_parameters(self):
        """使用Adam更新所有参数"""
        if not self._policy_adam_initialized:
            self._init_adam()

        ac = self.actor_critic
        lr_policy = self.config.ppo_policy_lr
        lr_value = self.config.ppo_value_lr

        # 更新骨干网络
        for i in range(len(ac.backbone_layers)):
            self._adam_step_matrix(
                f"bb_{i}_w", ac.backbone_layers[i].weights,
                ac.backbone_layers[i].grad_weights, lr_policy)
            self._adam_step_vector(
                f"bb_{i}_b", ac.backbone_layers[i].biases,
                ac.backbone_layers[i].grad_biases, lr_policy)
            self._adam_step_vector(
                f"bn_{i}_g", ac.backbone_norms[i].gamma,
                ac.backbone_norms[i].grad_gamma, lr_policy)
            self._adam_step_vector(
                f"bn_{i}_b", ac.backbone_norms[i].beta,
                ac.backbone_norms[i].grad_beta, lr_policy)

        # 更新策略头
        self._adam_step_matrix(
            "policy_w", ac.policy_head.weights,
            ac.policy_head.grad_weights, lr_policy)
        self._adam_step_vector(
            "policy_b", ac.policy_head.biases,
            ac.policy_head.grad_biases, lr_policy)

        # 更新价值头
        self._adam_step_matrix(
            "value_w", ac.value_head.weights,
            ac.value_head.grad_weights, lr_value)
        self._adam_step_vector(
            "value_b", ac.value_head.biases,
            ac.value_head.grad_biases, lr_value)

    def train(self, prompts: List[List[float]],
              reward_fn: Callable[[List[float], List[float]], float],
              ref_policy_fn: Optional[Callable] = None,
              num_steps: Optional[int] = None) -> Dict[str, float]:
        """
        完整的PPO训练循环

        流程:
        1. 收集rollout数据
        2. 计算GAE优势和回报
        3. (可选) 归一化优势
        4. 多轮PPO更新
        5. 更新KL控制器
        6. 清空缓冲区

        Args:
            prompts: prompt特征列表
            reward_fn: 奖励函数
            ref_policy_fn: 参考策略函数
            num_steps: 训练步数

        Returns:
            训练统计
        """
        num = num_steps or self.config.num_training_steps

        # 1. 收集rollout
        samples = self.collect_rollout(prompts, reward_fn, ref_policy_fn, num)
        for sample in samples:
            self.buffer.push(sample)

        # 2. 计算GAE
        self.compute_advantages_and_returns()

        # 3. 归一化优势
        if self.config.ppo_norm_advantage:
            self.normalize_advantages()

        # 4. PPO更新
        all_indices = list(range(len(self.buffer)))
        mini_batch_size = self.config.ppo_mini_batch_size
        epoch_stats = []

        for ppo_epoch in range(self.config.ppo_epochs):
            random.shuffle(all_indices)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_kl = 0.0
            epoch_count = 0

            for start in range(0, len(all_indices), mini_batch_size):
                batch_idx = all_indices[start:start + mini_batch_size]
                stats = self.ppo_update_step(batch_idx)

                if stats:
                    epoch_policy_loss += stats.get("policy_loss", 0.0)
                    epoch_value_loss += stats.get("value_loss", 0.0)
                    epoch_entropy += stats.get("entropy", 0.0)
                    epoch_kl += stats.get("kl_divergence", 0.0)
                    epoch_count += 1

            if epoch_count > 0:
                epoch_stats.append({
                    "ppo_epoch": ppo_epoch,
                    "policy_loss": epoch_policy_loss / epoch_count,
                    "value_loss": epoch_value_loss / epoch_count,
                    "entropy": epoch_entropy / epoch_count,
                    "kl_divergence": epoch_kl / epoch_count
                })

        # 5. 更新KL控制器
        avg_kl = sum(s.get("kl_divergence", 0.0) for s in epoch_stats) / max(len(epoch_stats), 1)
        self.kl_controller.update(avg_kl)

        # 6. 清空缓冲区
        self.buffer.clear()

        # 汇总统计
        final_stats = {
            "step": self.step_count,
            "avg_policy_loss": sum(s.get("policy_loss", 0.0) for s in epoch_stats) / max(len(epoch_stats), 1),
            "avg_value_loss": sum(s.get("value_loss", 0.0) for s in epoch_stats) / max(len(epoch_stats), 1),
            "avg_entropy": sum(s.get("entropy", 0.0) for s in epoch_stats) / max(len(epoch_stats), 1),
            "avg_kl": avg_kl,
            "kl_coeff": self.kl_controller.get_kl_coeff(),
            "num_rollout_samples": len(samples),
            "ppo_epochs_completed": len(epoch_stats)
        }

        self.training_stats.append(final_stats)
        self.step_count += 1

        return final_stats

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.training_stats:
            return {"status": "no training yet"}

        recent = self.training_stats[-10:]
        return {
            "total_steps": self.step_count,
            "recent_avg_policy_loss": sum(s["avg_policy_loss"] for s in recent) / len(recent),
            "recent_avg_value_loss": sum(s["avg_value_loss"] for s in recent) / len(recent),
            "recent_avg_entropy": sum(s["avg_entropy"] for s in recent) / len(recent),
            "recent_avg_kl": sum(s["avg_kl"] for s in recent) / len(recent),
            "current_kl_coeff": self.kl_controller.get_kl_coeff(),
            "kl_stats": self.kl_controller.get_kl_stats()
        }


# ============================================================================
# 5. RejectionSampling - 拒绝采样
# ============================================================================

class RejectionSampling:
    """
    拒绝采样 (Rejection Sampling)

    算法:
    1. 对每个prompt，使用策略生成N个候选response
    2. 使用奖励模型对每个候选评分
    3. 选择得分最高的K个response
    4. 在选中的response上进行监督微调

    这是一种简单但有效的对齐方法，可以作为PPO的预处理步骤。

    流程:
        for each prompt:
            candidates = [policy.generate(prompt) for _ in range(N)]
            scored = [(r, c) for c in candidates for r in reward_model(prompt, c)]
            best = top_k(scored, K)
            fine_tune_on(best)
    """

    def __init__(self, config: Optional[RLHFConfig] = None):
        self.config = config or RLHFConfig()
        self.num_candidates = self.config.rejection_sampling_num_candidates
        self.top_k = self.config.rejection_sampling_top_k
        self.temperature = self.config.rejection_sampling_temperature
        self.threshold = self.config.rejection_sampling_threshold

        # 统计
        self.total_generated = 0
        self.total_accepted = 0
        self.acceptance_rate = 0.0
        self.score_distribution: List[float] = []

    def generate_candidates(self, prompt_features: List[float],
                            policy_fn: Callable[[List[float], float], List[float]],
                            num_candidates: Optional[int] = None) -> List[Tuple[List[float], float]]:
        """
        生成候选response

        使用带温度的采样从策略生成多个候选

        Args:
            prompt_features: prompt特征
            policy_fn: 策略函数 (prompt, temperature) -> response_features
            num_candidates: 候选数量

        Returns:
            [(response_features, temperature), ...] 候选列表
        """
        num = num_candidates or self.num_candidates
        candidates = []

        for _ in range(num):
            # 使用温度采样
            response = policy_fn(prompt_features, self.temperature)
            candidates.append((response, self.temperature))
            self.total_generated += 1

        return candidates

    def select_best(self, prompt_features: List[float],
                    candidates: List[Tuple[List[float], float]],
                    reward_fn: Callable[[List[float], List[float]], float],
                    top_k: Optional[int] = None,
                    threshold: Optional[float] = None) -> List[Tuple[List[float], float, float]]:
        """
        从候选中选择最佳response

        1. 使用奖励模型对每个候选评分
        2. 过滤低于阈值的候选
        3. 按分数排序，选择top-k

        Args:
            prompt_features: prompt特征
            candidates: 候选列表 [(response, temperature), ...]
            reward_fn: 奖励函数
            top_k: 选择数量
            threshold: 最低奖励阈值

        Returns:
            [(response, temperature, score), ...] 按分数降序排列
        """
        k = top_k or self.top_k
        thresh = threshold if threshold is not None else self.threshold

        # 评分
        scored_candidates = []
        for response, temp in candidates:
            score = reward_fn(prompt_features, response)
            scored_candidates.append((response, temp, score))
            self.score_distribution.append(score)

        # 过滤阈值
        if thresh > 0:
            scored_candidates = [(r, t, s) for r, t, s in scored_candidates if s >= thresh]

        # 按分数降序排序
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        # 选择top-k
        best = scored_candidates[:k]
        self.total_accepted += len(best)

        # 更新接受率
        if self.total_generated > 0:
            self.acceptance_rate = self.total_accepted / self.total_generated

        return best

    def train_on_best(self, prompt_features: List[float],
                      best_candidates: List[Tuple[List[float], float, float]],
                      policy_update_fn: Callable[[List[float], List[float], float], None]) -> Dict[str, float]:
        """
        在最佳候选上进行训练

        使用选中的response更新策略:
        L = -Σ log π(a_i | s) * w_i

        其中 w_i 是基于奖励的权重:
        w_i = softmax(scores / temperature)_i

        这相当于加权监督学习，高分样本获得更大权重。

        Args:
            prompt_features: prompt特征
            best_candidates: 最佳候选 [(response, temperature, score), ...]
            policy_update_fn: 策略更新函数 (prompt, response, weight) -> None

        Returns:
            训练统计
        """
        if not best_candidates:
            return {"status": "no candidates passed threshold"}

        # 计算权重 (基于分数的softmax权重)
        scores = [s for _, _, s in best_candidates]
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / self.temperature) for s in scores]
        sum_exp = sum(exp_scores)
        weights = [e / sum_exp for e in exp_scores]

        # 加权训练
        total_loss = 0.0
        for (response, temp, score), weight in zip(best_candidates, weights):
            # 计算负对数似然损失
            nll_loss = -math.log(max(weight, 1e-10))
            total_loss += nll_loss * weight

            # 更新策略
            policy_update_fn(prompt_features, response, weight)

        avg_loss = total_loss / len(best_candidates)
        avg_score = sum(scores) / len(scores)
        max_score_val = max(scores)
        min_score_val = min(scores)

        return {
            "avg_loss": avg_loss,
            "avg_score": avg_score,
            "max_score": max_score_val,
            "min_score": min_score_val,
            "num_candidates": len(best_candidates),
            "acceptance_rate": self.acceptance_rate
        }

    def rejection_sampling_step(self, prompt_features: List[float],
                                policy_fn: Callable[[List[float], float], List[float]],
                                reward_fn: Callable[[List[float], List[float]], float],
                                policy_update_fn: Callable[[List[float], List[float], float], None]
                                ) -> Dict[str, float]:
        """
        完整的拒绝采样步骤

        1. 生成候选
        2. 选择最佳
        3. 在最佳上训练

        Args:
            prompt_features: prompt特征
            policy_fn: 策略函数
            reward_fn: 奖励函数
            policy_update_fn: 策略更新函数

        Returns:
            训练统计
        """
        # 1. 生成候选
        candidates = self.generate_candidates(prompt_features, policy_fn)

        # 2. 选择最佳
        best = self.select_best(prompt_features, candidates, reward_fn)

        # 3. 训练
        stats = self.train_on_best(prompt_features, best, policy_update_fn)

        return stats

    def batch_rejection_sampling(self, prompts: List[List[float]],
                                 policy_fn: Callable[[List[float], float], List[float]],
                                 reward_fn: Callable[[List[float], List[float]], float],
                                 policy_update_fn: Callable[[List[float], List[float], float], None]
                                 ) -> Dict[str, float]:
        """
        批量拒绝采样

        Args:
            prompts: prompt列表
            policy_fn: 策略函数
            reward_fn: 奖励函数
            policy_update_fn: 策略更新函数

        Returns:
            汇总统计
        """
        total_loss = 0.0
        total_score = 0.0
        total_accepted = 0
        num_prompts = len(prompts)

        for prompt in prompts:
            stats = self.rejection_sampling_step(
                prompt, policy_fn, reward_fn, policy_update_fn
            )
            total_loss += stats.get("avg_loss", 0.0)
            total_score += stats.get("avg_score", 0.0)
            total_accepted += stats.get("num_candidates", 0)

        return {
            "avg_loss": total_loss / max(num_prompts, 1),
            "avg_score": total_score / max(num_prompts, 1),
            "total_accepted": total_accepted,
            "num_prompts": num_prompts,
            "overall_acceptance_rate": total_accepted / max(self.total_generated, 1),
            "score_mean": sum(self.score_distribution) / max(len(self.score_distribution), 1),
            "score_std": self._compute_score_std()
        }

    def _compute_score_std(self) -> float:
        """计算分数标准差"""
        if len(self.score_distribution) < 2:
            return 0.0
        mean = sum(self.score_distribution) / len(self.score_distribution)
        var = sum((s - mean) ** 2 for s in self.score_distribution) / len(self.score_distribution)
        return math.sqrt(var)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "acceptance_rate": self.acceptance_rate,
            "num_candidates": self.num_candidates,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "score_mean": sum(self.score_distribution) / max(len(self.score_distribution), 1),
            "score_std": self._compute_score_std(),
            "score_min": min(self.score_distribution) if self.score_distribution else 0.0,
            "score_max": max(self.score_distribution) if self.score_distribution else 0.0
        }


# ============================================================================
# 6. RLHFTrainer - 主编排器
# ============================================================================

class RLHFTrainer:
    """
    RLHF训练主编排器

    完整的RLHF训练流程:
    1. 预训练奖励模型 (使用人类偏好数据)
    2. (可选) 拒绝采样预处理
    3. 收集策略rollout
    4. 使用PPO + 奖励模型 + KL惩罚进行对齐训练
    5. 评估和日志

    训练流程:
        Phase 1: 奖励模型训练
            for epoch in range(reward_epochs):
                for batch in preference_data:
                    loss = bradley_terry_loss(r_chosen, r_rejected)
                    reward_model.update(loss)

        Phase 2: PPO对齐训练
            for step in range(num_steps):
                # 收集rollout
                rollouts = policy.generate(prompts)
                rewards = reward_model.predict(rollouts)
                kl_penalty = kl_controller.compute(policy, ref_policy)

                # PPO更新
                for epoch in range(ppo_epochs):
                    ppo_update(policy, rollouts, rewards, kl_penalty)

                # 更新KL控制器
                kl_controller.update(current_kl)

                # 评估
                if step % eval_freq == 0:
                    evaluate(policy, reward_model)
    """

    def __init__(self, input_dim: int, action_dim: int,
                 config: Optional[RLHFConfig] = None):
        self.config = config or RLHFConfig()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # 设置随机种子
        random.seed(self.config.seed)

        # 初始化组件
        self.reward_model = RewardModel(
            input_dim=input_dim * 2,  # prompt + response
            config=self.config
        )
        self.ppo_trainer = PPOTrainer(
            input_dim=input_dim,
            action_dim=action_dim,
            config=self.config
        )
        self.rejection_sampling = RejectionSampling(config=self.config)

        # KL控制器 (PPO训练器内部已创建，这里也保留引用)
        if self.config.kl_strategy == "adaptive":
            self.kl_controller: KLController = AdaptiveKLController(
                init_kl_coeff=self.config.kl_coeff,
                target_kl=self.config.kl_target,
                horizon=self.config.kl_horizon
            )
        else:
            self.kl_controller = FixedKLController(kl_coeff=self.config.kl_coeff)

        # 训练状态
        self.current_phase = "initialization"
        self.global_step = 0
        self.reward_model_trained = False
        self.training_history: List[Dict[str, Any]] = []

    def pretrain_reward_model(self, preference_data: List[Tuple[List[float], List[float], List[float]]],
                              num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Phase 1: 预训练奖励模型

        使用Bradley-Terry模型从人类偏好数据训练奖励模型

        对每个偏好对 (prompt, chosen, rejected):
            r_c = RewardNetwork(prompt, chosen)
            r_r = RewardNetwork(prompt, rejected)
            L = -log(σ(r_c - r_r))

        Args:
            preference_data: [(prompt_features, chosen_features, rejected_features), ...]
            num_epochs: 训练轮数

        Returns:
            训练统计
        """
        self.current_phase = "reward_model_training"
        epochs = num_epochs or self.config.reward_model_epochs

        epoch_stats = []
        for epoch in range(epochs):
            stats = self.reward_model.train_epoch(preference_data)
            stats["epoch"] = epoch
            epoch_stats.append(stats)

            # 日志
            if (epoch + 1) % max(self.config.log_freq, 1) == 0:
                pass  # 实际实现中会记录日志

        self.reward_model_trained = True

        # 汇总
        final_stats = {
            "phase": "reward_model_training",
            "total_epochs": epochs,
            "final_loss": epoch_stats[-1]["avg_loss"] if epoch_stats else 0.0,
            "final_accuracy": epoch_stats[-1]["avg_accuracy"] if epoch_stats else 0.0,
            "epoch_history": epoch_stats
        }

        self.training_history.append(final_stats)
        return final_stats

    def ppo_alignment(self, prompts: List[List[float]],
                      num_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Phase 2: PPO对齐训练

        使用PPO + 奖励模型 + KL惩罚进行策略优化

        训练目标:
            L(θ) = E_t[min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]
                   - c_1 * L^{VF}(θ) + c_2 * H[π_θ] - β * KL(π_θ || π_ref)

        Args:
            prompts: prompt特征列表
            num_steps: 训练步数

        Returns:
            训练统计
        """
        self.current_phase = "ppo_alignment"
        steps = num_steps or self.config.num_training_steps

        # 奖励函数: 使用训练好的奖励模型
        def reward_fn(prompt_feat: List[float], response_feat: List[float]) -> float:
            return self.reward_model.predict(prompt_feat, response_feat)

        # 参考策略: 返回均匀分布的log概率
        def ref_policy_fn(x: List[float]) -> List[float]:
            return [-math.log(self.action_dim)] * self.action_dim

        step_stats = []
        for step in range(steps):
            # 随机选择一批prompt
            batch_prompts = random.sample(
                prompts, min(self.config.batch_size, len(prompts))
            )

            # PPO训练步
            stats = self.ppo_trainer.train(
                prompts=batch_prompts,
                reward_fn=reward_fn,
                ref_policy_fn=ref_policy_fn,
                num_steps=min(self.config.num_rollout_steps, len(batch_prompts))
            )

            stats["global_step"] = self.global_step
            step_stats.append(stats)
            self.global_step += 1

            # 定期评估
            if (step + 1) % self.config.eval_freq == 0:
                eval_stats = self._evaluate(batch_prompts)
                stats["evaluation"] = eval_stats

        # 汇总
        final_stats = {
            "phase": "ppo_alignment",
            "total_steps": steps,
            "avg_policy_loss": sum(s.get("avg_policy_loss", 0.0) for s in step_stats) / max(len(step_stats), 1),
            "avg_value_loss": sum(s.get("avg_value_loss", 0.0) for s in step_stats) / max(len(step_stats), 1),
            "avg_entropy": sum(s.get("avg_entropy", 0.0) for s in step_stats) / max(len(step_stats), 1),
            "avg_kl": sum(s.get("avg_kl", 0.0) for s in step_stats) / max(len(step_stats), 1),
            "final_kl_coeff": self.kl_controller.get_kl_coeff(),
            "step_history": step_stats
        }

        self.training_history.append(final_stats)
        return final_stats

    def rejection_sampling_phase(self, prompts: List[List[float]],
                                 policy_fn: Callable[[List[float], float], List[float]],
                                 policy_update_fn: Callable[[List[float], List[float], float], None]
                                 ) -> Dict[str, Any]:
        """
        Phase 1.5: 拒绝采样预处理 (可选)

        在PPO之前使用拒绝采样进行初步对齐

        Args:
            prompts: prompt列表
            policy_fn: 策略函数
            policy_update_fn: 策略更新函数

        Returns:
            训练统计
        """
        self.current_phase = "rejection_sampling"

        def reward_fn(prompt_feat: List[float], response_feat: List[float]) -> float:
            return self.reward_model.predict(prompt_feat, response_feat)

        stats = self.rejection_sampling.batch_rejection_sampling(
            prompts, policy_fn, reward_fn, policy_update_fn
        )

        stats["phase"] = "rejection_sampling"
        self.training_history.append(stats)
        return stats

    def full_training_pipeline(self,
                               preference_data: List[Tuple[List[float], List[float], List[float]]],
                               alignment_prompts: List[List[float]],
                               policy_fn: Optional[Callable] = None,
                               policy_update_fn: Optional[Callable] = None,
                               use_rejection_sampling: bool = False) -> Dict[str, Any]:
        """
        完整的RLHF训练流程

        Phase 1: 预训练奖励模型
        Phase 1.5: (可选) 拒绝采样
        Phase 2: PPO对齐训练

        Args:
            preference_data: 偏好数据
            alignment_prompts: 对齐训练的prompt
            policy_fn: 策略函数 (用于拒绝采样)
            policy_update_fn: 策略更新函数 (用于拒绝采样)
            use_rejection_sampling: 是否使用拒绝采样

        Returns:
            完整训练统计
        """
        self.current_phase = "full_pipeline"

        # Phase 1: 预训练奖励模型
        print("[RLHF] Phase 1: Training reward model...")
        reward_stats = self.pretrain_reward_model(preference_data)
        print(f"[RLHF] Reward model training complete. "
              f"Loss: {reward_stats['final_loss']:.4f}, "
              f"Accuracy: {reward_stats['final_accuracy']:.4f}")

        # Phase 1.5: (可选) 拒绝采样
        if use_rejection_sampling and policy_fn is not None and policy_update_fn is not None:
            print("[RLHF] Phase 1.5: Rejection sampling...")
            rs_stats = self.rejection_sampling_phase(
                alignment_prompts, policy_fn, policy_update_fn
            )
            print(f"[RLHF] Rejection sampling complete. "
                  f"Acceptance rate: {rs_stats.get('overall_acceptance_rate', 0):.4f}")

        # Phase 2: PPO对齐训练
        print("[RLHF] Phase 2: PPO alignment training...")
        ppo_stats = self.ppo_alignment(alignment_prompts)
        print(f"[RLHF] PPO training complete. "
              f"Policy loss: {ppo_stats['avg_policy_loss']:.4f}, "
              f"KL: {ppo_stats['avg_kl']:.4f}")

        # 最终汇总
        final_summary = {
            "phase": "full_pipeline_complete",
            "reward_model_final_loss": reward_stats['final_loss'],
            "reward_model_final_accuracy": reward_stats['final_accuracy'],
            "ppo_avg_policy_loss": ppo_stats['avg_policy_loss'],
            "ppo_avg_value_loss": ppo_stats['avg_value_loss'],
            "ppo_avg_entropy": ppo_stats['avg_entropy'],
            "ppo_avg_kl": ppo_stats['avg_kl'],
            "ppo_final_kl_coeff": ppo_stats['final_kl_coeff'],
            "total_global_steps": self.global_step,
            "training_history_phases": len(self.training_history)
        }

        self.training_history.append(final_summary)
        return final_summary

    def _evaluate(self, prompts: List[List[float]]) -> Dict[str, float]:
        """
        评估当前策略

        使用奖励模型对策略生成的response进行评分

        Args:
            prompts: 评估用的prompt列表

        Returns:
            评估指标
        """
        total_reward = 0.0
        total_kl = 0.0
        num_eval = min(10, len(prompts))

        for i in range(num_eval):
            prompt = prompts[i]
            # 使用策略生成response
            response = self.ppo_trainer._generate_response(prompt)

            # 计算奖励
            reward = self.reward_model.predict(prompt, response)
            total_reward += reward

            # 计算KL (使用prompt作为状态输入)
            log_probs, _, _ = self.ppo_trainer.actor_critic.forward(prompt)
            ref_log_probs = [-math.log(self.action_dim)] * len(log_probs)
            kl = self.kl_controller.compute_kl_penalty(log_probs, ref_log_probs)
            total_kl += kl

        return {
            "avg_reward": total_reward / max(num_eval, 1),
            "avg_kl": total_kl / max(num_eval, 1),
            "num_eval_samples": num_eval
        }

    def get_training_report(self) -> Dict[str, Any]:
        """
        获取完整的训练报告

        Returns:
            包含所有训练阶段统计的字典
        """
        report = {
            "config": {
                "seed": self.config.seed,
                "kl_strategy": self.config.kl_strategy,
                "kl_coeff_init": self.config.kl_coeff,
                "kl_target": self.config.kl_target,
                "ppo_clip_epsilon": self.config.ppo_clip_epsilon,
                "ppo_gamma": self.config.ppo_gamma,
                "ppo_gae_lambda": self.config.ppo_gae_lambda,
                "ppo_epochs": self.config.ppo_epochs,
                "reward_model_epochs": self.config.reward_model_epochs,
            },
            "current_phase": self.current_phase,
            "global_step": self.global_step,
            "reward_model_trained": self.reward_model_trained,
            "num_phases_completed": len(self.training_history),
            "kl_controller_stats": self.kl_controller.get_kl_stats(),
            "ppo_training_summary": self.ppo_trainer.get_training_summary(),
            "rejection_sampling_stats": self.rejection_sampling.get_stats(),
        }

        # 各阶段统计
        for i, phase_stats in enumerate(self.training_history):
            phase_name = phase_stats.get("phase", f"phase_{i}")
            report[f"phase_{phase_name}"] = phase_stats

        return report

    def save_state(self) -> Dict[str, Any]:
        """
        保存训练状态 (序列化为字典)

        Returns:
            包含所有模型参数和优化器状态的字典
        """
        state = {
            "config": vars(self.config),
            "global_step": self.global_step,
            "current_phase": self.current_phase,
            "reward_model_trained": self.reward_model_trained,
            "kl_controller_type": type(self.kl_controller).__name__,
            "kl_coeff": self.kl_controller.get_kl_coeff(),
        }

        # 保存奖励模型参数
        rm_net = self.reward_model.network
        reward_model_params = {
            "num_layers": rm_net.num_layers,
            "hidden_dim": rm_net.hidden_dim,
            "input_dim": rm_net.input_dim,
        }
        for i in range(rm_net.num_layers):
            reward_model_params[f"layer_{i}_weights"] = rm_net.layers[i].weights
            reward_model_params[f"layer_{i}_biases"] = rm_net.layers[i].biases
            reward_model_params[f"norm_{i}_gamma"] = rm_net.norms[i].gamma
            reward_model_params[f"norm_{i}_beta"] = rm_net.norms[i].beta
        reward_model_params["output_weights"] = rm_net.output_layer.weights
        reward_model_params["output_biases"] = rm_net.output_layer.biases
        state["reward_model"] = reward_model_params

        return state

    def load_state(self, state: Dict[str, Any]):
        """
        从字典加载训练状态

        Args:
            state: 保存的状态字典
        """
        self.global_step = state.get("global_step", 0)
        self.current_phase = state.get("current_phase", "initialization")
        self.reward_model_trained = state.get("reward_model_trained", False)

        # 加载奖励模型参数
        if "reward_model" in state:
            rm_params = state["reward_model"]
            rm_net = self.reward_model.network
            for i in range(rm_net.num_layers):
                rm_net.layers[i].weights = rm_params[f"layer_{i}_weights"]
                rm_net.layers[i].biases = rm_params[f"layer_{i}_biases"]
                rm_net.norms[i].gamma = rm_params[f"norm_{i}_gamma"]
                rm_net.norms[i].beta = rm_params[f"norm_{i}_beta"]
            rm_net.output_layer.weights = rm_params["output_weights"]
            rm_net.output_layer.biases = rm_params["output_biases"]


# ============================================================================
# 辅助函数: 数据生成和测试
# ============================================================================

def generate_synthetic_preference_data(num_samples: int, input_dim: int,
                                        quality_gap: float = 1.0,
                                        noise_std: float = 0.3) -> List[Tuple[List[float], List[float], List[float]]]:
    """
    生成合成偏好数据用于测试

    chosen response的特征质量高于rejected response

    Args:
        num_samples: 样本数量
        input_dim: 特征维度
        quality_gap: chosen和rejected之间的质量差距
        noise_std: 噪声标准差

    Returns:
        [(prompt_features, chosen_features, rejected_features), ...]
    """
    data = []
    for _ in range(num_samples):
        prompt = [random.gauss(0, 1) for _ in range(input_dim)]
        # chosen质量更高
        chosen = [random.gauss(quality_gap, noise_std) for _ in range(input_dim)]
        # rejected质量较低
        rejected = [random.gauss(-quality_gap * 0.5, noise_std) for _ in range(input_dim)]
        data.append((prompt, chosen, rejected))
    return data


def generate_synthetic_prompts(num_prompts: int, input_dim: int) -> List[List[float]]:
    """
    生成合成的prompt特征

    Args:
        num_prompts: prompt数量
        input_dim: 特征维度

    Returns:
        prompt特征列表
    """
    return [[random.gauss(0, 1) for _ in range(input_dim)] for _ in range(num_prompts)]


def demo_training():
    """
    RLHF训练演示

    展示完整的RLHF训练流程:
    1. 生成合成数据
    2. 训练奖励模型
    3. PPO对齐训练
    """
    print("=" * 70)
    print("RLHF (Reinforcement Learning from Human Feedback) 训练演示")
    print("=" * 70)

    # 配置
    config = RLHFConfig(
        seed=42,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        reward_model_hidden_dim=64,
        reward_model_num_layers=2,
        reward_model_epochs=3,
        reward_model_batch_size=8,
        reward_model_lr=1e-3,
        ppo_policy_lr=1e-3,
        ppo_value_lr=1e-3,
        ppo_epochs=2,
        ppo_mini_batch_size=4,
        num_rollout_steps=8,
        num_training_steps=3,
        batch_size=8,
        kl_coeff=0.05,
        kl_target=3.0,
        ppo_clip_epsilon=0.2,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_entropy_coeff=0.01,
        ppo_value_coeff=0.5,
        rejection_sampling_num_candidates=4,
        rejection_sampling_top_k=1,
        log_freq=1,
        eval_freq=1,
    )

    input_dim = 32
    action_dim = 16

    # 生成合成数据
    print("\n[1] 生成合成数据...")
    preference_data = generate_synthetic_preference_data(
        num_samples=50, input_dim=input_dim, quality_gap=1.0
    )
    alignment_prompts = generate_synthetic_prompts(num_prompts=20, input_dim=input_dim)
    print(f"    偏好数据: {len(preference_data)} 对")
    print(f"    对齐prompts: {len(alignment_prompts)} 个")

    # 创建RLHF训练器
    print("\n[2] 初始化RLHF训练器...")
    trainer = RLHFTrainer(input_dim=input_dim, action_dim=action_dim, config=config)
    print("    组件初始化完成:")
    print(f"    - RewardNetwork: input_dim={input_dim*2}, hidden={config.reward_model_hidden_dim}, "
          f"layers={config.reward_model_num_layers}")
    print(f"    - PPOActorCritic: input_dim={input_dim}, action_dim={action_dim}, "
          f"hidden={config.hidden_dim}, layers={config.num_layers}")
    print(f"    - KLController: type={config.kl_strategy}, coeff={config.kl_coeff}, "
          f"target={config.kl_target}")
    print(f"    - PPO-Clip: epsilon={config.ppo_clip_epsilon}, "
          f"gamma={config.ppo_gamma}, lambda={config.ppo_gae_lambda}")

    # Phase 1: 训练奖励模型
    print("\n[3] Phase 1: 训练奖励模型...")
    reward_stats = trainer.pretrain_reward_model(preference_data, num_epochs=3)
    print(f"    最终损失: {reward_stats['final_loss']:.4f}")
    print(f"    最终准确率: {reward_stats['final_accuracy']:.4f}")

    # 测试奖励模型
    print("\n[4] 测试奖励模型...")
    test_prompt = [random.gauss(0, 1) for _ in range(input_dim)]
    good_response = [random.gauss(1.0, 0.3) for _ in range(input_dim)]
    bad_response = [random.gauss(-0.5, 0.3) for _ in range(input_dim)]
    good_reward = trainer.reward_model.predict(test_prompt, good_response)
    bad_reward = trainer.reward_model.predict(test_prompt, bad_response)
    print(f"    好回答奖励: {good_reward:.4f}")
    print(f"    差回答奖励: {bad_reward:.4f}")
    print(f"    奖励差距: {good_reward - bad_reward:.4f}")

    # Phase 2: PPO对齐训练
    print("\n[5] Phase 2: PPO对齐训练...")
    ppo_stats = trainer.ppo_alignment(alignment_prompts, num_steps=3)
    print(f"    平均策略损失: {ppo_stats['avg_policy_loss']:.4f}")
    print(f"    平均价值损失: {ppo_stats['avg_value_loss']:.4f}")
    print(f"    平均熵: {ppo_stats['avg_entropy']:.4f}")
    print(f"    平均KL: {ppo_stats['avg_kl']:.4f}")
    print(f"    最终KL系数: {ppo_stats['final_kl_coeff']:.6f}")

    # 拒绝采样演示
    print("\n[6] 拒绝采样演示...")
    rs = RejectionSampling(config=config)

    def simple_policy(prompt: List[float], temp: float) -> List[float]:
        return [random.gauss(0, 1) for _ in range(input_dim)]

    def simple_reward(prompt: List[float], response: List[float]) -> float:
        return sum(p * r for p, r in zip(prompt, response)) / len(prompt)

    def simple_update(prompt: List[float], response: List[float], weight: float):
        pass  # 简化: 不实际更新

    rs_stats = rs.rejection_sampling_step(
        test_prompt, simple_policy, simple_reward, simple_update
    )
    print(f"    生成候选数: {config.rejection_sampling_num_candidates}")
    print(f"    接受候选数: {rs_stats.get('num_candidates', 0)}")
    print(f"    平均分数: {rs_stats.get('avg_score', 0):.4f}")

    # GAE演示
    print("\n[7] GAE计算演示...")
    gae = GeneralizedAdvantageEstimation(gamma=0.99, gae_lambda=0.95)
    rewards = [1.0, 0.5, 0.2, -0.1, 0.8]
    values = [0.5, 0.6, 0.4, 0.3, 0.5]
    dones = [False, False, False, False, True]
    advantages, returns = gae.compute(rewards, values, dones)
    print(f"    奖励: {rewards}")
    print(f"    价值: {[f'{v:.3f}' for v in values]}")
    print(f"    优势: {[f'{a:.3f}' for a in advantages]}")
    print(f"    回报: {[f'{r:.3f}' for r in returns]}")

    # PPO-Clip演示
    print("\n[8] PPO-Clip目标函数演示...")
    ppo_clip = PPOClipObjective(clip_epsilon=0.2)
    for ratio, adv in [(0.8, 1.0), (1.2, 1.0), (1.5, 1.0), (0.5, -1.0), (1.3, -1.0)]:
        loss, unclipped = ppo_clip.compute_policy_loss(ratio, adv)
        print(f"    ratio={ratio:.1f}, advantage={adv:.1f} -> "
              f"clipped_loss={loss:.4f}, unclipped_loss={unclipped:.4f}")

    # KL控制器演示
    print("\n[9] 自适应KL控制器演示...")
    adaptive_kl = AdaptiveKLController(init_kl_coeff=0.1, target_kl=6.0)
    print(f"    初始KL系数: {adaptive_kl.get_kl_coeff():.6f}")
    test_log_p = [math.log(0.3), math.log(0.5), math.log(0.2)]
    test_log_q = [math.log(0.33), math.log(0.34), math.log(0.33)]
    for i in range(5):
        kl_val = adaptive_kl.compute_kl_penalty(test_log_p, test_log_q)
        new_coeff = adaptive_kl.update(kl_val)
        print(f"    Step {i+1}: KL={kl_val:.4f}, coeff={new_coeff:.6f}")

    # 完整流程
    print("\n[10] 完整RLHF流程演示...")
    trainer2 = RLHFTrainer(input_dim=16, action_dim=8, config=RLHFConfig(
        seed=42, hidden_dim=32, num_layers=2,
        reward_model_hidden_dim=32, reward_model_num_layers=2,
        reward_model_epochs=2, reward_model_batch_size=8,
        ppo_epochs=2, ppo_mini_batch_size=4,
        num_rollout_steps=4, num_training_steps=2,
        batch_size=4, log_freq=1, eval_freq=1,
    ))
    pref_data = generate_synthetic_preference_data(20, 16)
    align_prompts = generate_synthetic_prompts(8, 16)
    full_stats = trainer2.full_training_pipeline(pref_data, align_prompts)
    print(f"    奖励模型最终损失: {full_stats['reward_model_final_loss']:.4f}")
    print(f"    奖励模型最终准确率: {full_stats['reward_model_final_accuracy']:.4f}")
    print(f"    PPO平均策略损失: {full_stats['ppo_avg_policy_loss']:.4f}")
    print(f"    PPO平均KL: {full_stats['ppo_avg_kl']:.4f}")
    print(f"    总全局步数: {full_stats['total_global_steps']}")

    # 训练报告
    print("\n[11] 训练报告...")
    report = trainer2.get_training_report()
    print(f"    完成阶段数: {report['num_phases_completed']}")
    print(f"    全局步数: {report['global_step']}")

    print("\n" + "=" * 70)
    print("RLHF训练演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    demo_training()
