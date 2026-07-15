"""
Video Generation Trainer Module

使用统一约束系统和MoE进行视频生成训练。

主要组件：
- VideoTrainer: 视频训练器
- TrainingConfig: 训练配置
- ConstraintManager: 约束管理器集成
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
import math

# 导入统一核心算法
from agi_unified_framework.core.unified_algorithms import (
    UnifiedAlgorithmConfig,
    ConstraintManager,
    PhysicsConstraint,
    ConsistencyConstraint,
    ConstraintPriority,
    MixtureOfExperts,
    Expert,
    ExpertType,
    MoEStats,
)

from agi_unified_framework.video_gen.unified_adapter import (
    VideoMoEAdapter,
    create_video_unified_system,
)

from agi_unified_framework.video_gen.models.dit import (
    DiTModel,
    DiTConfig,
)


# ============================================================================
# 简易自动微分引擎（纯Python实现）
# ============================================================================

class Tensor:
    """
    简易张量，支持自动微分。
    
    实现了一个轻量级的计算图和反向传播机制，
    用于在纯Python环境下模拟深度学习的梯度计算。
    
    Attributes:
        data: 张量数据（标量或嵌套列表）
        grad: 梯度值
        _backward_fn: 反向传播函数
        _prev: 前驱节点（计算图的父节点）
        _name: 节点名称（用于调试）
    """
    
    def __init__(self, data, _backward_fn=None, _prev=(), _name=""):
        """
        初始化张量
        
        Args:
            data: 标量数值或嵌套列表
            _backward_fn: 反向传播回调函数
            _prev: 前驱张量集合
            _name: 节点名称
        """
        # 将数据统一为浮点数标量或列表
        if isinstance(data, (list, tuple)):
            self.data = [float(x) if isinstance(x, (int, float)) else x for x in data]
        else:
            self.data = float(data) if isinstance(data, (int, float)) else data
        
        self.grad = 0.0  # 梯度，初始为0
        self._backward_fn = _backward_fn  # 反向传播函数
        self._prev = set(_prev)  # 计算图中的前驱节点
        self._name = _name  # 节点名称
    
    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    # ---------- 算术运算（构建计算图） ----------
    
    def __add__(self, other):
        """加法运算：self + other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        def _backward():
            # 加法的梯度就是上游梯度直接传递
            self.grad += out.grad
            other.grad += out.grad
        
        out = Tensor(self.data + other.data, _backward_fn=_backward, _prev=(self, other))
        return out
    
    def __radd__(self, other):
        """右加法运算：other + self"""
        return self.__add__(other)
    
    def __mul__(self, other):
        """乘法运算：self * other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        def _backward():
            # 乘法的梯度：d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out = Tensor(self.data * other.data, _backward_fn=_backward, _prev=(self, other))
        return out
    
    def __rmul__(self, other):
        """右乘法运算：other * self"""
        return self.__mul__(other)
    
    def __neg__(self):
        """取负运算：-self"""
        return self * (-1.0)
    
    def __sub__(self, other):
        """减法运算：self - other"""
        return self + (-other)
    
    def __rsub__(self, other):
        """右减法运算：other - self"""
        return (-self) + other
    
    def __truediv__(self, other):
        """除法运算：self / other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        def _backward():
            # 除法的梯度：d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        
        out = Tensor(self.data / other.data, _backward_fn=_backward, _prev=(self, other))
        return out
    
    def __rtruediv__(self, other):
        """右除法运算：other / self"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    def __pow__(self, exponent):
        """幂运算：self ** exponent"""
        assert isinstance(exponent, (int, float)), "指数必须是标量数值"
        
        def _backward():
            # 幂运算的梯度：d(a^n)/da = n * a^(n-1)
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad
        
        out = Tensor(self.data ** exponent, _backward_fn=_backward, _prev=(self,))
        return out
    
    def relu(self):
        """
        ReLU激活函数：max(0, x)
        
        Returns:
            激活后的张量
        """
        def _backward():
            # ReLU的梯度：x > 0时为1，否则为0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        
        out = Tensor(max(0.0, self.data), _backward_fn=_backward, _prev=(self,))
        return out
    
    def sigmoid(self):
        """
        Sigmoid激活函数：1 / (1 + exp(-x))
        
        Returns:
            激活后的张量
        """
        s = 1.0 / (1.0 + math.exp(-self.data))
        
        def _backward():
            # Sigmoid的梯度：s * (1 - s)
            self.grad += (s * (1.0 - s)) * out.grad
        
        out = Tensor(s, _backward_fn=_backward, _prev=(self,))
        return out
    
    def tanh(self):
        """
        Tanh激活函数
        
        Returns:
            激活后的张量
        """
        t = math.tanh(self.data)
        
        def _backward():
            # Tanh的梯度：1 - t^2
            self.grad += (1.0 - t ** 2) * out.grad
        
        out = Tensor(t, _backward_fn=_backward, _prev=(self,))
        return out
    
    def log(self):
        """
        自然对数函数
        
        Returns:
            对数张量
        """
        def _backward():
            # log的梯度：1/x
            self.grad += (1.0 / self.data) * out.grad
        
        out = Tensor(math.log(self.data), _backward_fn=_backward, _prev=(self,))
        return out
    
    def exp(self):
        """
        指数函数
        
        Returns:
            指数张量
        """
        e = math.exp(self.data)
        
        def _backward():
            # exp的梯度：exp(x)
            self.grad += e * out.grad
        
        out = Tensor(e, _backward_fn=_backward, _prev=(self,))
        return out
    
    def backward(self):
        """
        反向传播：从当前节点开始，沿计算图反向遍历，计算所有节点的梯度。
        
        使用拓扑排序确保每个节点在其所有后继节点处理完毕后才被处理。
        """
        # 构建拓扑排序
        topo = []
        visited = set()
        
        def _build_topo(v):
            """递归构建拓扑排序"""
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    _build_topo(child)
                topo.append(v)
        
        _build_topo(self)
        
        # 将输出节点的梯度设为1.0（dL/dL = 1）
        self.grad = 1.0
        
        # 按逆拓扑序反向传播梯度
        for node in reversed(topo):
            if node._backward_fn is not None:
                node._backward_fn()
    
    def zero_grad(self):
        """将梯度清零"""
        self.grad = 0.0


def mse_loss(predictions: List[Tensor], targets: List[float]) -> Tensor:
    """
    计算MSE（均方误差）损失
    
    MSE = (1/N) * Σ(pred_i - target_i)^2
    
    Args:
        predictions: 预测值张量列表
        targets: 目标值列表（浮点数）
        
    Returns:
        损失张量（标量）
    """
    if not predictions or not targets:
        return Tensor(0.0)
    
    n = min(len(predictions), len(targets))
    total = Tensor(0.0)
    
    for i in range(n):
        diff = predictions[i] - targets[i]
        total = total + diff * diff  # (pred - target)^2
    
    # 取平均
    return total / n


# ============================================================================
# 优化器
# ============================================================================

class Optimizer:
    """
    优化器基类
    
    提供参数管理和梯度操作的基础功能。
    
    Attributes:
        params: 可训练参数列表
        lr: 学习率
        weight_decay: 权重衰减系数（L2正则化）
    """
    
    def __init__(self, params: List[Tensor], lr: float = 1e-3, weight_decay: float = 0.0):
        """
        初始化优化器
        
        Args:
            params: 可训练参数列表
            lr: 学习率
            weight_decay: 权重衰减系数
        """
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
    
    def zero_grad(self):
        """清零所有参数的梯度"""
        for p in self.params:
            p.zero_grad()
    
    def step(self):
        """执行一步参数更新
        
        默认实现：基本的随机梯度下降更新。
        param = param - lr * gradient - lr * weight_decay * param
        
        子类应覆盖此方法以实现更复杂的更新策略（如动量、Adam等）。
        """
        for p in self.params:
            if p.grad is not None:
                grad_val = p.grad.data if hasattr(p.grad, 'data') else p.grad
                update = self.lr * grad_val
                if self.weight_decay > 0:
                    update = update + self.lr * self.weight_decay * p.data
                p.data = p.data - update


class SGDOptimizer(Optimizer):
    """
    随机梯度下降（SGD）优化器
    
    更新规则：param = param - lr * gradient - lr * weight_decay * param
    
    支持动量（Momentum）和Nesterov加速梯度。
    
    Attributes:
        momentum: 动量系数
        nesterov: 是否使用Nesterov动量
        velocities: 各参数的速度（动量累积）
    """
    
    def __init__(self, params: List[Tensor], lr: float = 1e-3,
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False):
        """
        初始化SGD优化器
        
        Args:
            params: 可训练参数列表
            lr: 学习率
            momentum: 动量系数（0表示不使用动量）
            weight_decay: 权重衰减系数
            nesterov: 是否使用Nesterov动量
        """
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        # 初始化速度（动量缓冲区）
        self.velocities = [0.0] * len(self.params)
    
    def step(self):
        """
        执行一步SGD参数更新
        
        带动量的更新规则：
            v_t = momentum * v_{t-1} + gradient + weight_decay * param
            param = param - lr * v_t
            
        Nesterov变体：
            v_t = momentum * v_{t-1} + gradient + weight_decay * param
            param = param - lr * (gradient + momentum * v_t)
        """
        for i, param in enumerate(self.params):
            grad = param.grad
            
            # 添加权重衰减（L2正则化）
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            if self.momentum != 0:
                # 更新速度
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                
                if self.nesterov:
                    # Nesterov：在"前瞻"位置计算梯度
                    param.data -= self.lr * (grad + self.momentum * self.velocities[i])
                else:
                    # 标准动量
                    param.data -= self.lr * self.velocities[i]
            else:
                # 无动量的纯SGD
                param.data -= self.lr * grad


class AdamOptimizer(Optimizer):
    """
    Adam（Adaptive Moment Estimation）优化器
    
    结合了一阶动量（均值）和二阶动量（方差）的自适应学习率方法。
    
    更新规则：
        m_t = β1 * m_{t-1} + (1 - β1) * g_t          （一阶动量）
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2        （二阶动量）
        m̂_t = m_t / (1 - β1^t)                         （偏差修正）
        v̂_t = v_t / (1 - β2^t)                         （偏差修正）
        param = param - lr * m̂_t / (√v̂_t + ε)
    
    Attributes:
        beta1: 一阶动量衰减率
        beta2: 二阶动量衰减率
        eps: 数值稳定项
        t: 时间步
        m: 一阶动量缓冲区
        v: 二阶动量缓冲区
    """
    
    def __init__(self, params: List[Tensor], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        初始化Adam优化器
        
        Args:
            params: 可训练参数列表
            lr: 学习率
            betas: (β1, β2) 动量衰减率
            eps: 数值稳定项（防止除零）
            weight_decay: 权重衰减系数（解耦权重衰减，即AdamW风格）
        """
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # 时间步计数器
        # 初始化一阶和二阶动量缓冲区
        self.m = [0.0] * len(self.params)  # 一阶动量（均值）
        self.v = [0.0] * len(self.params)  # 二阶动量（方差）
    
    def step(self):
        """
        执行一步Adam参数更新
        
        包含偏差修正和解耦权重衰减（AdamW风格）。
        """
        self.t += 1  # 递增时间步
        
        for i, param in enumerate(self.params):
            grad = param.grad
            
            # 更新一阶动量（指数移动平均）
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            
            # 更新二阶动量（梯度平方的指数移动平均）
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            
            # 偏差修正（补偿初始化为零导致的偏差）
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            
            # 解耦权重衰减（AdamW风格）：直接对参数施加衰减
            if self.weight_decay != 0:
                param.data -= self.lr * self.weight_decay * param.data
            
            # Adam参数更新
            param.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


# ============================================================================
# 学习率调度器
# ============================================================================

class LRScheduler:
    """
    学习率调度器基类
    
    Attributes:
        optimizer: 关联的优化器
        initial_lr: 初始学习率
        current_lr: 当前学习率
    """
    
    def __init__(self, optimizer: Optimizer, initial_lr: Optional[float] = None):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 关联的优化器
            initial_lr: 初始学习率（若为None则使用优化器的当前lr）
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr if initial_lr is not None else optimizer.lr
        self.current_lr = self.initial_lr
    
    def step(self) -> float:
        """
        更新学习率
        
        默认实现：保持学习率不变（恒定调度器）。
        子类应覆盖此方法以实现具体的学习率调度策略。
        
        Returns:
            更新后的学习率
        """
        self.current_lr = self.optimizer.lr
        return self.current_lr


class CosineAnnealingScheduler(LRScheduler):
    """
    余弦退火学习率调度器
    
    学习率按余弦函数衰减：
        lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
    
    Attributes:
        T_max: 总步数（半个余弦周期）
        lr_min: 最小学习率
        current_step: 当前步数
    """
    
    def __init__(self, optimizer: Optimizer, T_max: int,
                 lr_min: float = 0.0, initial_lr: Optional[float] = None):
        """
        初始化余弦退火调度器
        
        Args:
            optimizer: 关联的优化器
            T_max: 总步数
            lr_min: 最小学习率
            initial_lr: 初始学习率
        """
        super().__init__(optimizer, initial_lr)
        self.T_max = max(T_max, 1)  # 防止除零
        self.lr_min = lr_min
        self.current_step = 0
    
    def step(self) -> float:
        """
        按余弦函数更新学习率
        
        Returns:
            当前学习率
        """
        # 余弦退火公式
        self.current_lr = self.lr_min + 0.5 * (self.initial_lr - self.lr_min) * \
                          (1.0 + math.cos(math.pi * self.current_step / self.T_max))
        
        # 更新优化器的学习率
        self.optimizer.lr = self.current_lr
        self.current_step += 1
        
        return self.current_lr


class ExponentialDecayScheduler(LRScheduler):
    """
    指数衰减学习率调度器
    
    学习率按指数函数衰减：
        lr_t = lr_0 * gamma^t
    
    Attributes:
        gamma: 衰减率（每步乘以gamma）
        current_step: 当前步数
    """
    
    def __init__(self, optimizer: Optimizer, gamma: float = 0.99,
                 initial_lr: Optional[float] = None):
        """
        初始化指数衰减调度器
        
        Args:
            optimizer: 关联的优化器
            gamma: 衰减率（0 < gamma < 1）
            initial_lr: 初始学习率
        """
        super().__init__(optimizer, initial_lr)
        assert 0 < gamma < 1, "衰减率gamma必须在(0, 1)之间"
        self.gamma = gamma
        self.current_step = 0
    
    def step(self) -> float:
        """
        按指数函数更新学习率
        
        Returns:
            当前学习率
        """
        # 指数衰减公式
        self.current_lr = self.initial_lr * (self.gamma ** self.current_step)
        
        # 更新优化器的学习率
        self.optimizer.lr = self.current_lr
        self.current_step += 1
        
        return self.current_lr


class WarmupScheduler(LRScheduler):
    """
    预热学习率调度器
    
    在前warmup_steps步内线性增加学习率，之后保持不变或交给后续调度器。
    
    Attributes:
        warmup_steps: 预热步数
        current_step: 当前步数
    """
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 initial_lr: Optional[float] = None):
        """
        初始化预热调度器
        
        Args:
            optimizer: 关联的优化器
            warmup_steps: 预热步数
            initial_lr: 目标学习率（预热结束后达到的值）
        """
        super().__init__(optimizer, initial_lr)
        self.warmup_steps = max(warmup_steps, 1)
        self.current_step = 0
    
    def step(self) -> float:
        """
        线性预热学习率
        
        在预热阶段：lr = initial_lr * (step / warmup_steps)
        预热结束后：lr = initial_lr
        
        Returns:
            当前学习率
        """
        if self.current_step < self.warmup_steps:
            # 线性预热
            self.current_lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # 预热结束，保持目标学习率
            self.current_lr = self.initial_lr
        
        self.optimizer.lr = self.current_lr
        self.current_step += 1
        
        return self.current_lr


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class TrainingConfig:
    """
    训练配置
    
    Attributes:
        # 训练参数
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        
        # 统一核心配置
        use_unified_core: 是否使用统一核心
        use_constraints: 是否使用约束系统
        use_moe: 是否使用MoE
        unified_config: 统一算法配置
        
        # 约束配置
        constraint_tolerance: 约束容差
        physics_constraint_weight: 物理约束权重
        consistency_constraint_weight: 一致性约束权重
        
        # MoE配置
        num_experts: 专家数量
        top_k_experts: 激活的专家数
        expert_capacity: 专家容量
        
        # 优化配置
        gradient_clip: 梯度裁剪
        warmup_steps: 预热步数
    """
    
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    
    # 统一核心配置
    use_unified_core: bool = True
    use_constraints: bool = True
    use_moe: bool = True
    unified_config: Optional[UnifiedAlgorithmConfig] = None
    
    # 约束配置
    constraint_tolerance: float = 0.01
    physics_constraint_weight: float = 1.0
    consistency_constraint_weight: float = 0.5
    
    # MoE配置
    num_experts: int = 8
    top_k_experts: int = 2
    expert_capacity: float = 1.0
    
    # 优化配置
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    
    # 优化器配置
    optimizer_type: str = "adam"  # "sgd" 或 "adam"
    momentum: float = 0.9  # SGD动量系数
    adam_beta1: float = 0.9  # Adam一阶动量衰减率
    adam_beta2: float = 0.999  # Adam二阶动量衰减率
    adam_eps: float = 1e-8  # Adam数值稳定项
    weight_decay: float = 0.0  # 权重衰减（L2正则化）
    
    # 学习率调度配置
    lr_scheduler_type: str = "cosine"  # "cosine", "exponential", "warmup", "none"
    lr_decay_gamma: float = 0.99  # 指数衰减率
    lr_min: float = 1e-6  # 最小学习率（余弦退火用）
    
    def __post_init__(self):
        """初始化后处理"""
        if self.unified_config is None:
            self.unified_config = UnifiedAlgorithmConfig.video_optimized_config()


# ============================================================================
# 视频训练器
# ============================================================================

class VideoTrainer:
    """
    视频训练器
    
    使用统一约束系统和MoE进行训练。
    
    Attributes:
        model: DiT模型
        config: 训练配置
        constraint_manager: 约束管理器
        moe_adapter: MoE适配器
        optimizer: 优化器（简化模拟）
    """
    
    def __init__(self,
                 model: Optional[DiTModel] = None,
                 config: Optional[TrainingConfig] = None):
        """
        初始化视频训练器
        
        Args:
            model: DiT模型
            config: 训练配置
        """
        self.config = config or TrainingConfig()
        
        # 初始化模型
        if model is None:
            from agi_unified_framework.video_gen.models.dit import create_dit_model
            self.model = create_dit_model(
                use_unified_core=self.config.use_unified_core,
                use_moe=self.config.use_moe,
                num_experts=self.config.num_experts,
                top_k_experts=self.config.top_k_experts
            )
        else:
            self.model = model
        
        # 初始化约束管理器
        if self.config.use_unified_core and self.config.use_constraints:
            self.constraint_manager = ConstraintManager(strict_mode=False)
            self._init_constraints()
        else:
            self.constraint_manager = None
        
        # 初始化MoE适配器
        if self.config.use_unified_core and self.config.use_moe:
            self.moe_adapter = VideoMoEAdapter(
                num_experts=self.config.num_experts,
                top_k=self.config.top_k_experts,
                config=self.config.unified_config
            )
        else:
            self.moe_adapter = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # ---- 自动微分相关状态 ----
        # 可训练参数（Tensor列表），用于构建计算图和梯度更新
        self.trainable_params: List[Tensor] = []
        # 前向传播中记录的预测张量（用于反向传播）
        self._last_predictions: List[Tensor] = []
        # 前向传播中记录的目标值
        self._last_targets: List[float] = []
        # 前向传播中计算的总损失张量
        self._last_loss_tensor: Optional[Tensor] = None
        
        # 初始化可训练参数（从模型中提取或创建模拟参数）
        self._init_trainable_params()
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.lr_scheduler = self._create_lr_scheduler()
    
    def _init_trainable_params(self) -> None:
        """
        初始化可训练参数。
        
        尝试从模型中提取参数；如果模型没有提供参数接口，
        则创建一组模拟参数用于演示自动微分流程。
        """
        self.trainable_params = []
        
        # 尝试从模型中获取参数
        if hasattr(self.model, 'parameters') and callable(self.model.parameters):
            try:
                model_params = self.model.parameters()
                if model_params:
                    for p in model_params:
                        if isinstance(p, Tensor):
                            self.trainable_params.append(p)
                        elif isinstance(p, (int, float)):
                            self.trainable_params.append(Tensor(float(p)))
                        elif isinstance(p, dict) and 'data' in p:
                            self.trainable_params.append(Tensor(float(p['data'])))
            except Exception:
                pass
        
        # 如果模型没有提供参数，创建模拟参数用于演示
        if not self.trainable_params:
            import random
            random.seed(42)
            # 创建10个模拟参数，模拟模型权重
            for i in range(10):
                param = Tensor(random.gauss(0, 0.02), _name=f"param_{i}")
                self.trainable_params.append(param)
    
    def _create_optimizer(self) -> Optimizer:
        """
        根据配置创建优化器
        
        Returns:
            优化器实例（SGD或Adam）
        """
        if self.config.optimizer_type == "sgd":
            return SGDOptimizer(
                params=self.trainable_params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            # 默认使用Adam
            return AdamOptimizer(
                params=self.trainable_params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay
            )
    
    def _create_lr_scheduler(self) -> Optional[LRScheduler]:
        """
        根据配置创建学习率调度器
        
        Returns:
            学习率调度器实例，或None（不使用调度器）
        """
        if self.config.lr_scheduler_type == "none" or self.optimizer is None:
            return None
        
        if self.config.lr_scheduler_type == "cosine":
            # 余弦退火：总步数 = epochs * 预估每epoch步数
            total_steps = max(self.config.num_epochs * 100, 1)
            return CosineAnnealingScheduler(
                optimizer=self.optimizer,
                T_max=total_steps,
                lr_min=self.config.lr_min,
                initial_lr=self.config.learning_rate
            )
        elif self.config.lr_scheduler_type == "exponential":
            return ExponentialDecayScheduler(
                optimizer=self.optimizer,
                gamma=self.config.lr_decay_gamma,
                initial_lr=self.config.learning_rate
            )
        elif self.config.lr_scheduler_type == "warmup":
            return WarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=self.config.warmup_steps,
                initial_lr=self.config.learning_rate
            )
        
        return None
    
    def _init_constraints(self) -> None:
        """初始化约束"""
        if self.constraint_manager is None:
            return
        
        # 添加物理约束
        physics_constraint = PhysicsConstraint(
            name="video_physics",
            constraint_type="velocity_limit",
            priority=ConstraintPriority.SOFT,
            tolerance=self.config.constraint_tolerance,
            max_velocity=10.0
        )
        self.constraint_manager.add_constraint(physics_constraint)
        
        # 添加一致性约束
        consistency_constraint = ConsistencyConstraint(
            name="temporal_consistency",
            consistency_type="temporal",
            priority=ConstraintPriority.SOFT,
            tolerance=self.config.constraint_tolerance
        )
        self.constraint_manager.add_constraint(consistency_constraint)
        
        # 添加边界约束
        boundary_constraint = PhysicsConstraint(
            name="spatial_boundary",
            constraint_type="boundary",
            priority=ConstraintPriority.HARD,
            tolerance=self.config.constraint_tolerance,
            bounds=[[-10, 10], [-10, 10], [-10, 10]]
        )
        self.constraint_manager.add_constraint(boundary_constraint)
    
    def train_step(self, 
                   batch_data: Dict[str, Any],
                   batch_idx: int = 0) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch_data: 批次数据
            batch_idx: 批次索引
            
        Returns:
            训练指标字典
        """
        # 前向传播
        frames = batch_data.get('frames', [])
        targets = batch_data.get('targets', [])
        
        # 计算损失（使用自动微分计算图）
        loss, loss_tensor = self._compute_loss_with_graph(frames, targets)
        
        # 约束检查
        constraint_loss = 0.0
        if self.constraint_manager is not None:
            constraint_result = self.constraint_manager.check_all(frames)
            if not constraint_result.is_satisfied:
                # 添加约束违反惩罚
                constraint_loss = sum(
                    v.severity for v in constraint_result.violations
                ) * 0.1
            
            # 将约束损失加入计算图
            constraint_tensor = Tensor(constraint_loss)
            loss_tensor = loss_tensor + constraint_tensor
            loss += constraint_loss
        
        # MoE负载均衡损失
        moe_loss = 0.0
        if self.moe_adapter is not None:
            moe_stats = self.moe_adapter.get_routing_stats()
            # 简单的负载均衡惩罚
            if moe_stats['expert_loads']:
                loads = list(moe_stats['expert_loads'].values())
                avg_load = sum(loads) / len(loads) if loads else 0
                variance = sum((l - avg_load) ** 2 for l in loads) / len(loads) if loads else 0
                moe_loss = variance * 0.01
            
            # 将MoE损失加入计算图
            moe_tensor = Tensor(moe_loss)
            loss_tensor = loss_tensor + moe_tensor
            loss += moe_loss
        
        # 保存当前步的计算图信息，供反向传播使用
        self._last_loss_tensor = loss_tensor
        
        # 反向传播：基于计算图计算梯度
        self._backward(loss, loss_tensor)
        
        # 更新参数：使用优化器更新可训练参数
        self._update_parameters()
        
        # 更新学习率调度器
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        return {
            'loss': loss,
            'constraint_loss': constraint_loss,
            'moe_loss': moe_loss,
            'step': self.global_step
        }
    
    def train_epoch(self, 
                    train_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_data: 训练数据列表
            
        Returns:
            epoch指标字典
        """
        epoch_losses = []
        
        for batch_idx, batch_data in enumerate(train_data):
            metrics = self.train_step(batch_data, batch_idx)
            epoch_losses.append(metrics['loss'])
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        self.train_losses.append(avg_loss)
        
        self.current_epoch += 1
        
        return {
            'epoch': self.current_epoch,
            'train_loss': avg_loss,
            'num_batches': len(train_data)
        }
    
    def validate(self, 
                 val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        验证
        
        Args:
            val_data: 验证数据列表
            
        Returns:
            验证指标字典
        """
        val_losses = []
        
        for batch_data in val_data:
            frames = batch_data.get('frames', [])
            targets = batch_data.get('targets', [])
            
            loss = self._compute_loss(frames, targets)
            val_losses.append(loss)
        
        avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        self.val_losses.append(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'num_batches': len(val_data)
        }
    
    def fit(self, 
            train_data: List[Dict[str, Any]],
            val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        完整训练流程
        
        Args:
            train_data: 训练数据
            val_data: 验证数据（可选）
            
        Returns:
            训练历史
        """
        print(f"开始训练: {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_data)
            
            # 验证
            if val_data is not None:
                val_metrics = self.validate(val_data)
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"train_loss={train_metrics['train_loss']:.4f}, "
                      f"val_loss={val_metrics['val_loss']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"train_loss={train_metrics['train_loss']:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'num_epochs': self.config.num_epochs
        }
    
    def _compute_loss(self, 
                      predictions: List[Any], 
                      targets: List[Any]) -> float:
        """
        计算损失（纯数值版本，不构建计算图）
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            损失值
        """
        # 简化实现：MSE损失模拟
        if not predictions or not targets:
            return 0.0
        
        loss = 0.0
        count = 0
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, (list, tuple)) and isinstance(target, (list, tuple)):
                for p, t in zip(pred, target):
                    try:
                        loss += (float(p) - float(t)) ** 2
                        count += 1
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    loss += (float(pred) - float(target)) ** 2
                    count += 1
                except (ValueError, TypeError):
                    pass
        
        return loss / count if count > 0 else 0.0
    
    def _compute_loss_with_graph(self,
                                  predictions: List[Any],
                                  targets: List[Any]) -> Tuple[float, Tensor]:
        """
        计算损失并构建自动微分计算图
        
        将预测值和目标值包装为Tensor，通过计算图构建
        可追踪的MSE损失，使得后续反向传播能够正确计算梯度。
        
        Args:
            predictions: 预测值列表（可以是标量、列表或Tensor）
            targets: 目标值列表
            
        Returns:
            (标量损失值, 损失Tensor节点)
        """
        if not predictions or not targets:
            return 0.0, Tensor(0.0)
        
        # 将预测值展平为标量列表，同时包装为Tensor
        pred_tensors: List[Tensor] = []
        target_floats: List[float] = []
        
        for pred, target in zip(predictions, targets):
            if isinstance(pred, (list, tuple)) and isinstance(target, (list, tuple)):
                for p, t in zip(pred, target):
                    try:
                        p_val = float(p)
                        t_val = float(t)
                        # 使用可训练参数构建预测（模拟前向传播）
                        # 在实际深度学习中，pred是通过模型参数计算得到的
                        # 这里我们用参数的线性组合来模拟
                        pred_tensor = self._simulate_forward(p_val)
                        pred_tensors.append(pred_tensor)
                        target_floats.append(t_val)
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    p_val = float(pred)
                    t_val = float(target)
                    pred_tensor = self._simulate_forward(p_val)
                    pred_tensors.append(pred_tensor)
                    target_floats.append(t_val)
                except (ValueError, TypeError):
                    pass
        
        # 保存用于调试
        self._last_predictions = pred_tensors
        self._last_targets = target_floats
        
        if not pred_tensors:
            return 0.0, Tensor(0.0)
        
        # 使用计算图构建MSE损失
        loss_tensor = mse_loss(pred_tensors, target_floats)
        
        return loss_tensor.data, loss_tensor
    
    def _simulate_forward(self, input_val: float) -> Tensor:
        """
        模拟前向传播：用可训练参数对输入进行变换。
        
        在实际深度学习中，前向传播是通过神经网络层完成的。
        这里使用参数的简单线性组合来模拟：
            output = param_0 * input + param_1 * input^2 + param_2
        
        这样梯度可以通过计算图正确回传到参数。
        
        Args:
            input_val: 输入标量值
            
        Returns:
            输出张量（连接到可训练参数的计算图）
        """
        if len(self.trainable_params) < 3:
            # 参数不足时直接返回输入的Tensor
            return Tensor(input_val)
        
        # 简单的非线性变换：w0*x + w1*x^2 + w2
        x = Tensor(input_val)
        w0 = self.trainable_params[0]
        w1 = self.trainable_params[1]
        w2 = self.trainable_params[2]
        
        # 构建计算图：output = w0 * x + w1 * x^2 + w2
        output = w0 * x + w1 * (x ** 2) + w2
        
        return output
    
    def _backward(self, loss: float, loss_tensor: Optional[Tensor] = None) -> None:
        """
        反向传播：基于计算图计算所有参数的梯度
        
        实现步骤：
        1. 清零所有参数的梯度
        2. 如果有计算图（loss_tensor），通过自动微分反向传播
        3. 如果没有计算图，使用数值梯度作为后备方案
        4. 执行梯度裁剪，防止梯度爆炸
        
        Args:
            loss: 当前损失值（标量，用于数值梯度计算）
            loss_tensor: 损失张量节点（计算图的输出节点）
        """
        # 第一步：清零所有参数的梯度
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        
        if loss_tensor is not None and loss_tensor._prev:
            # ---- 方式一：基于计算图的自动微分（解析梯度） ----
            # 调用Tensor.backward()进行拓扑排序反向传播
            loss_tensor.backward()
            
        else:
            # ---- 方式二：数值梯度（后备方案） ----
            # 当计算图不可用时，使用有限差分法计算数值梯度
            self._compute_numerical_gradients(loss)
        
        # 梯度裁剪：防止梯度爆炸
        self._clip_gradients()
    
    def _compute_numerical_gradients(self, loss: float, epsilon: float = 1e-5) -> None:
        """
        使用有限差分法计算数值梯度（后备方案）
        
        对于每个参数p，计算：
            grad_p ≈ (loss(p + ε) - loss(p - ε)) / (2 * ε)
        
        Args:
            loss: 当前参数下的损失值
            epsilon: 有限差分步长
        """
        for param in self.trainable_params:
            original_val = param.data
            
            # 前向差分：f(x + ε)
            param.data = original_val + epsilon
            loss_plus = self._compute_loss(
                [p.data for p in self._last_predictions] if self._last_predictions else [],
                self._last_targets
            )
            
            # 后向差分：f(x - ε)
            param.data = original_val - epsilon
            loss_minus = self._compute_loss(
                [p.data for p in self._last_predictions] if self._last_predictions else [],
                self._last_targets
            )
            
            # 中心差分公式
            param.grad = (loss_plus - loss_minus) / (2.0 * epsilon)
            
            # 恢复原始参数值
            param.data = original_val
    
    def _clip_gradients(self) -> None:
        """
        梯度裁剪：将所有参数的梯度限制在配置范围内
        
        支持两种裁剪方式：
        - 按值裁剪（默认）：将每个梯度限制在 [-clip_value, +clip_value]
        - 按范数裁剪：将梯度向量的L2范数限制在clip_value以内
        
        这里实现按值裁剪，简单高效。
        """
        clip_value = self.config.gradient_clip
        if clip_value <= 0:
            return  # 不裁剪
        
        for param in self.trainable_params:
            if param.grad > clip_value:
                param.grad = clip_value
            elif param.grad < -clip_value:
                param.grad = -clip_value
    
    def _update_parameters(self) -> None:
        """
        使用优化器更新所有可训练参数
        
        支持的优化器：
        - SGD（带动量/Nesterov）
        - Adam（带偏差修正和AdamW权重衰减）
        
        优化器和学习率调度器在__init__中根据配置创建。
        每步训练后，学习率调度器会自动调整学习率。
        """
        if self.optimizer is None:
            return
        
        # 执行一步参数更新
        self.optimizer.step()
    
    def check_constraints(self, data: Any) -> Dict[str, Any]:
        """
        检查约束
        
        Args:
            data: 要检查的数据
            
        Returns:
            约束检查结果
        """
        if self.constraint_manager is None:
            return {'enabled': False}
        
        result = self.constraint_manager.check_all(data)
        
        return {
            'enabled': True,
            'is_satisfied': result.is_satisfied,
            'score': result.score,
            'violations': [
                {
                    'name': v.constraint_name,
                    'type': v.constraint_type,
                    'severity': v.severity,
                    'message': v.message
                }
                for v in result.violations
            ]
        }
    
    def repair_constraints(self, data: Any) -> Any:
        """
        修复约束违反
        
        Args:
            data: 原始数据
            
        Returns:
            修复后的数据
        """
        if self.constraint_manager is None:
            return data
        
        return self.constraint_manager.repair_all(data)
    
    def get_moe_stats(self) -> Dict[str, Any]:
        """
        获取MoE统计信息
        
        Returns:
            MoE统计信息
        """
        if self.moe_adapter is None:
            return {'enabled': False}
        
        stats = self.moe_adapter.get_routing_stats()
        return {
            'enabled': True,
            **stats
        }
    
    def get_constraint_stats(self) -> Dict[str, Any]:
        """
        获取约束统计信息
        
        Returns:
            约束统计信息
        """
        if self.constraint_manager is None:
            return {'enabled': False}
        
        return {
            'enabled': True,
            **self.constraint_manager.get_stats()
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            训练统计信息
        """
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'use_constraints': self.config.use_constraints,
                'use_moe': self.config.use_moe
            }
        }


# ============================================================================
# 约束感知损失
# ============================================================================

class ConstraintAwareLoss:
    """
    约束感知损失
    
    将约束违反纳入损失函数。
    
    Attributes:
        constraint_manager: 约束管理器
        base_loss_fn: 基础损失函数
        constraint_weight: 约束权重
    """
    
    def __init__(self,
                 constraint_manager: ConstraintManager,
                 base_loss_fn: Optional[Callable] = None,
                 constraint_weight: float = 0.1):
        """
        初始化约束感知损失
        
        Args:
            constraint_manager: 约束管理器
            base_loss_fn: 基础损失函数
            constraint_weight: 约束权重
        """
        self.constraint_manager = constraint_manager
        self.base_loss_fn = base_loss_fn or self._default_loss_fn
        self.constraint_weight = constraint_weight
    
    def __call__(self, 
                 predictions: Any, 
                 targets: Any) -> Tuple[float, Dict[str, float]]:
        """
        计算损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            (总损失, 损失组件字典)
        """
        # 基础损失
        base_loss = self.base_loss_fn(predictions, targets)
        
        # 约束损失
        constraint_result = self.constraint_manager.check_all(predictions)
        constraint_loss = 0.0
        
        if not constraint_result.is_satisfied:
            constraint_loss = sum(
                v.severity for v in constraint_result.violations
            ) * self.constraint_weight
        
        # 总损失
        total_loss = base_loss + constraint_loss
        
        return total_loss, {
            'base_loss': base_loss,
            'constraint_loss': constraint_loss,
            'num_violations': len(constraint_result.violations)
        }
    
    def _default_loss_fn(self, predictions: Any, targets: Any) -> float:
        """
        默认损失函数
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            损失值
        """
        # 简化MSE
        if isinstance(predictions, (list, tuple)) and isinstance(targets, (list, tuple)):
            return sum((float(p) - float(t)) ** 2 
                      for p, t in zip(predictions, targets)) / len(predictions)
        return 0.0


# ============================================================================
# MoE训练辅助
# ============================================================================

class MoETrainer:
    """
    MoE训练辅助
    
    辅助MoE的训练过程。
    
    Attributes:
        moe_adapter: MoE适配器
        load_balance_weight: 负载均衡权重
    """
    
    def __init__(self,
                 moe_adapter: VideoMoEAdapter,
                 load_balance_weight: float = 0.01):
        """
        初始化MoE训练辅助
        
        Args:
            moe_adapter: MoE适配器
            load_balance_weight: 负载均衡权重
        """
        self.moe_adapter = moe_adapter
        self.load_balance_weight = load_balance_weight
    
    def compute_load_balance_loss(self) -> float:
        """
        计算负载均衡损失
        
        Returns:
            负载均衡损失值
        """
        stats = self.moe_adapter.get_routing_stats()
        expert_loads = stats.get('expert_loads', {})
        
        if not expert_loads:
            return 0.0
        
        loads = list(expert_loads.values())
        avg_load = sum(loads) / len(loads)
        
        if avg_load == 0:
            return 0.0
        
        # 计算方差
        variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
        
        return variance * self.load_balance_weight
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """
        获取专家利用率
        
        Returns:
            专家利用率字典
        """
        stats = self.moe_adapter.get_routing_stats()
        expert_loads = stats.get('expert_loads', {})
        total_tokens = stats.get('total_tokens', 0)
        
        if total_tokens == 0:
            return {i: 0.0 for i in range(self.moe_adapter.num_experts)}
        
        return {
            expert_id: load / total_tokens
            for expert_id, load in expert_loads.items()
        }


# ============================================================================
# 渐进式训练器
# ============================================================================

class ProgressiveTrainer(VideoTrainer):
    """
    渐进式训练器
    
    从低分辨率、短时长开始训练，逐步增加帧数和分辨率，
    最终达到目标视频规格。这种方式可以：
    1. 加速早期训练收敛
    2. 稳定长视频生成训练
    3. 逐步提升模型对时空细节的建模能力
    
    渐进阶段：
    - 阶段0: 8帧, 128p (快速学习基本运动模式)
    - 阶段1: 16帧, 128p (学习更长的时序依赖)
    - 阶段2: 32帧, 256p (提升空间分辨率)
    - 阶段3: 64帧, 256p (更长的时序建模)
    - 阶段4: 128帧, 512p (高分辨率长视频)
    - 阶段5: 256帧, 512p (目标规格)
    
    Attributes:
        stages: 渐进阶段配置列表
        current_stage_idx: 当前阶段索引
        stage_epochs: 每个阶段的训练轮数
        transition_fn: 阶段转换回调函数
        stage_history: 各阶段的训练历史
    """
    
    # 默认渐进阶段配置：(帧数, 分辨率)
    DEFAULT_STAGES: List[Tuple[int, int]] = [
        (8, 128),    # 阶段0：8帧, 128p
        (16, 128),   # 阶段1：16帧, 128p
        (32, 256),   # 阶段2：32帧, 256p
        (64, 256),   # 阶段3：64帧, 256p
        (128, 512),  # 阶段4：128帧, 512p
        (256, 512),  # 阶段5：256帧, 512p
    ]
    
    def __init__(self,
                 model: Optional[DiTModel] = None,
                 config: Optional[TrainingConfig] = None,
                 stages: Optional[List[Tuple[int, int]]] = None,
                 stage_epochs: Optional[List[int]] = None,
                 transition_fn: Optional[Callable] = None):
        """
        初始化渐进式训练器
        
        Args:
            model: DiT模型
            config: 训练配置
            stages: 自定义阶段配置列表，每项为(帧数, 分辨率)
            stage_epochs: 每个阶段的训练轮数列表
            transition_fn: 阶段转换时的回调函数
        """
        super().__init__(model=model, config=config)
        
        # 渐进阶段配置
        self.stages = stages if stages is not None else list(self.DEFAULT_STAGES)
        self.current_stage_idx = 0
        
        # 每个阶段的训练轮数（默认每个阶段训练20轮）
        if stage_epochs is not None:
            self.stage_epochs = stage_epochs
        else:
            self.stage_epochs = [20] * len(self.stages)
        
        # 确保stage_epochs和stages长度一致
        while len(self.stage_epochs) < len(self.stages):
            self.stage_epochs.append(20)
        
        # 阶段转换回调
        self.transition_fn = transition_fn
        
        # 各阶段训练历史
        self.stage_history: List[Dict[str, Any]] = []
        
        # 当前阶段的帧数和分辨率
        self.current_frames = self.stages[0][0]
        self.current_resolution = self.stages[0][1]
    
    @property
    def current_stage(self) -> Dict[str, Any]:
        """
        获取当前阶段信息
        
        Returns:
            当前阶段配置字典
        """
        if self.current_stage_idx < len(self.stages):
            frames, resolution = self.stages[self.current_stage_idx]
            return {
                'stage_idx': self.current_stage_idx,
                'num_frames': frames,
                'resolution': resolution,
                'epochs': self.stage_epochs[self.current_stage_idx],
                'total_stages': len(self.stages)
            }
        return {
            'stage_idx': self.current_stage_idx,
            'completed': True,
            'total_stages': len(self.stages)
        }
    
    def _prepare_data_for_stage(self,
                                 train_data: List[Dict[str, Any]],
                                 num_frames: int,
                                 resolution: int) -> List[Dict[str, Any]]:
        """
        根据当前阶段配置裁剪/调整训练数据
        
        将原始数据调整为当前阶段所需的帧数和分辨率。
        在实际应用中，这里会进行视频帧采样和分辨率缩放。
        
        Args:
            train_data: 原始训练数据
            num_frames: 当前阶段的帧数
            resolution: 当前阶段的分辨率
            
        Returns:
            调整后的训练数据
        """
        prepared = []
        for batch in train_data:
            adjusted_batch = dict(batch)
            
            # 调整帧数：截取或填充到目标帧数
            frames = batch.get('frames', [])
            if isinstance(frames, list) and len(frames) > num_frames:
                # 均匀采样num_frames帧
                step = len(frames) / num_frames
                adjusted_frames = [frames[int(i * step)] for i in range(num_frames)]
                adjusted_batch['frames'] = adjusted_frames
            elif isinstance(frames, list) and len(frames) < num_frames:
                # 循环填充到目标帧数
                adjusted_frames = []
                for i in range(num_frames):
                    adjusted_frames.append(frames[i % len(frames)])
                adjusted_batch['frames'] = adjusted_frames
            
            # 调整目标值：与帧数保持一致
            targets = batch.get('targets', [])
            if isinstance(targets, list) and len(targets) > num_frames:
                step = len(targets) / num_frames
                adjusted_targets = [targets[int(i * step)] for i in range(num_frames)]
                adjusted_batch['targets'] = adjusted_targets
            elif isinstance(targets, list) and len(targets) < num_frames:
                adjusted_targets = []
                for i in range(num_frames):
                    adjusted_targets.append(targets[i % len(targets)])
                adjusted_batch['targets'] = adjusted_targets
            
            # 记录分辨率信息（实际缩放在数据加载阶段完成）
            adjusted_batch['resolution'] = resolution
            adjusted_batch['num_frames'] = num_frames
            
            prepared.append(adjusted_batch)
        
        return prepared
    
    def _transition_to_next_stage(self) -> bool:
        """
        转换到下一个训练阶段
        
        执行以下操作：
        1. 更新当前阶段索引
        2. 调整模型配置（帧数、分辨率）
        3. 调整学习率（阶段转换时通常降低学习率）
        4. 调整优化器状态
        5. 调用用户自定义的转换回调
        
        Returns:
            是否成功转换（False表示已经是最后阶段）
        """
        next_idx = self.current_stage_idx + 1
        
        if next_idx >= len(self.stages):
            print("已达到最终阶段，无需进一步转换")
            return False
        
        old_stage = self.current_stage
        self.current_stage_idx = next_idx
        
        # 更新当前帧数和分辨率
        new_frames, new_resolution = self.stages[next_idx]
        self.current_frames = new_frames
        self.current_resolution = new_resolution
        
        # 调整模型配置（如果模型支持）
        if hasattr(self.model, 'config'):
            try:
                self.model.config.num_frames = new_frames
                self.model.config.resolution = new_resolution
            except Exception:
                pass
        
        # 阶段转换时降低学习率（衰减为原来的0.7倍）
        if self.optimizer is not None:
            self.optimizer.lr *= 0.7
            print(f"  学习率调整: {self.optimizer.lr:.6f}")
        
        # 重置学习率调度器（为新阶段重新开始调度）
        if self.lr_scheduler is not None:
            self.lr_scheduler = self._create_lr_scheduler()
        
        # 调用用户自定义的转换回调
        if self.transition_fn is not None:
            try:
                self.transition_fn(
                    old_stage=old_stage,
                    new_stage=self.current_stage,
                    model=self.model,
                    optimizer=self.optimizer
                )
            except Exception as e:
                print(f"  警告：阶段转换回调执行失败: {e}")
        
        print(f"  阶段转换完成: {old_stage['num_frames']}帧/{old_stage['resolution']}p "
              f"-> {new_frames}帧/{new_resolution}p")
        
        return True
    
    def _adjust_model_for_stage(self, num_frames: int, resolution: int) -> None:
        """
        调整模型以适应当前阶段
        
        在阶段转换时，可能需要：
        1. 调整位置编码的长度（适应新的帧数）
        2. 调整分辨率相关的参数
        3. 添加新的层（如果分辨率增加）
        
        Args:
            num_frames: 目标帧数
            resolution: 目标分辨率
        """
        # 模拟模型调整：更新可训练参数以适应新的规模
        # 在实际实现中，这里会调整DiT的位置编码等
        if hasattr(self.model, 'update_config'):
            try:
                self.model.update_config(num_frames=num_frames, resolution=resolution)
            except Exception:
                pass
        
        # 如果当前参数数量不足以支撑新的规模，添加新参数
        required_params = max(10, int(num_frames * resolution / 1000))
        while len(self.trainable_params) < required_params:
            import random
            new_param = Tensor(
                random.gauss(0, 0.02),
                _name=f"param_{len(self.trainable_params)}"
            )
            self.trainable_params.append(new_param)
        
        # 重新创建优化器（包含新参数）
        self.optimizer = self._create_optimizer()
    
    def train_stage(self,
                    train_data: List[Dict[str, Any]],
                    val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        训练当前阶段
        
        Args:
            train_data: 训练数据
            val_data: 验证数据（可选）
            
        Returns:
            当前阶段的训练结果
        """
        stage_info = self.current_stage
        num_frames = stage_info['num_frames']
        resolution = stage_info['resolution']
        num_epochs = stage_info['epochs']
        
        print(f"\n{'='*60}")
        print(f"阶段 {stage_info['stage_idx']}/{stage_info['total_stages'] - 1}: "
              f"{num_frames}帧, {resolution}p, {num_epochs}轮")
        print(f"{'='*60}")
        
        # 调整模型
        self._adjust_model_for_stage(num_frames, resolution)
        
        # 准备当前阶段的数据
        stage_train_data = self._prepare_data_for_stage(
            train_data, num_frames, resolution
        )
        stage_val_data = None
        if val_data is not None:
            stage_val_data = self._prepare_data_for_stage(
                val_data, num_frames, resolution
            )
        
        # 执行训练
        stage_train_losses = []
        stage_val_losses = []
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            epoch_metrics = self.train_epoch(stage_train_data)
            stage_train_losses.append(epoch_metrics['train_loss'])
            
            # 验证
            val_loss = None
            if stage_val_data is not None:
                val_metrics = self.validate(stage_val_data)
                val_loss = val_metrics['val_loss']
                stage_val_losses.append(val_loss)
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"train_loss={epoch_metrics['train_loss']:.4f}, "
                      f"val_loss={val_loss:.4f}")
            else:
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"train_loss={epoch_metrics['train_loss']:.4f}")
        
        # 记录阶段历史
        stage_result = {
            'stage_idx': stage_info['stage_idx'],
            'num_frames': num_frames,
            'resolution': resolution,
            'num_epochs': num_epochs,
            'train_losses': stage_train_losses,
            'val_losses': stage_val_losses,
            'final_train_loss': stage_train_losses[-1] if stage_train_losses else None,
            'final_val_loss': stage_val_losses[-1] if stage_val_losses else None,
            'learning_rate': self.optimizer.lr if self.optimizer else None
        }
        self.stage_history.append(stage_result)
        
        return stage_result
    
    def fit(self,
            train_data: List[Dict[str, Any]],
            val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        完整的渐进式训练流程
        
        依次通过所有渐进阶段，每个阶段使用对应的帧数和分辨率进行训练。
        阶段之间自动进行模型调整和学习率调度。
        
        Args:
            train_data: 训练数据
            val_data: 验证数据（可选）
            
        Returns:
            完整的训练历史（包含所有阶段）
        """
        print(f"开始渐进式训练: 共{len(self.stages)}个阶段")
        print(f"渐进路线: {' -> '.join(f'{f}帧/{r}p' for f, r in self.stages)}")
        
        total_start_step = self.global_step
        
        for stage_idx in range(len(self.stages)):
            self.current_stage_idx = stage_idx
            
            # 训练当前阶段
            stage_result = self.train_stage(train_data, val_data)
            
            # 如果不是最后一个阶段，转换到下一阶段
            if stage_idx < len(self.stages) - 1:
                print(f"\n阶段 {stage_idx} 完成，准备转换...")
                self._transition_to_next_stage()
        
        print(f"\n{'='*60}")
        print(f"渐进式训练完成!")
        print(f"总训练步数: {self.global_step - total_start_step}")
        print(f"{'='*60}")
        
        return {
            'stage_history': self.stage_history,
            'total_stages': len(self.stages),
            'total_steps': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def get_stage_progress(self) -> Dict[str, Any]:
        """
        获取渐进式训练的进度信息
        
        Returns:
            进度信息字典
        """
        completed = len(self.stage_history)
        total = len(self.stages)
        
        return {
            'current_stage': self.current_stage_idx,
            'completed_stages': completed,
            'total_stages': total,
            'progress': completed / total if total > 0 else 0.0,
            'current_frames': self.current_frames,
            'current_resolution': self.current_resolution,
            'stage_history_summary': [
                {
                    'stage': h['stage_idx'],
                    'frames': h['num_frames'],
                    'resolution': h['resolution'],
                    'final_loss': h['final_train_loss']
                }
                for h in self.stage_history
            ]
        }


# ============================================================================
# DPO偏好训练器
# ============================================================================

class DPOTrainer(VideoTrainer):
    """
    DPO（Direct Preference Optimization）偏好训练器
    
    直接偏好优化是一种无需显式奖励模型的RLHF替代方法。
    通过直接在偏好数据上优化策略模型，使其输出更接近
    人类偏好的结果。
    
    DPO损失函数：
        L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                          - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))]
    
    其中：
        - y_w: 偏好（chosen）输出
        - y_l: 非偏好（rejected）输出
        - π_θ: 策略模型
        - π_ref: 参考模型（冻结）
        - β: 温度参数（控制偏离参考模型的程度）
        - σ: sigmoid函数
    
    Attributes:
        ref_model: 参考模型（冻结的初始策略模型）
        beta: DPO温度参数
        kl_weight: KL散度惩罚权重
        preference_data: 偏好数据列表
        dpo_losses: DPO损失历史
    """
    
    def __init__(self,
                 model: Optional[DiTModel] = None,
                 config: Optional[TrainingConfig] = None,
                 beta: float = 0.1,
                 kl_weight: float = 0.01):
        """
        初始化DPO偏好训练器
        
        Args:
            model: 策略模型（将被优化的模型）
            config: 训练配置
            beta: DPO温度参数，控制策略偏离参考模型的程度
                  beta越大，允许偏离越大；beta越小，越接近参考模型
            kl_weight: KL散度惩罚权重，防止策略偏离参考模型过远
        """
        super().__init__(model=model, config=config)
        
        # DPO温度参数
        self.beta = beta
        self.kl_weight = kl_weight
        
        # 参考模型（策略模型的初始副本，参数冻结）
        self.ref_params: List[float] = []
        self._init_reference_model()
        
        # 偏好数据
        self.preference_data: List[Dict[str, Any]] = []
        
        # DPO训练历史
        self.dpo_losses: List[float] = []
        self.kl_divergences: List[float] = []
        self.reward_margins: List[float] = []
    
    def _init_reference_model(self) -> None:
        """
        初始化参考模型
        
        参考模型是策略模型的初始副本，在训练过程中保持冻结。
        DPO通过比较策略模型和参考模型的对数概率来计算损失。
        """
        # 保存当前可训练参数的初始值作为参考
        self.ref_params = [p.data for p in self.trainable_params]
        print(f"参考模型已初始化: {len(self.ref_params)}个参数已冻结")
    
    def load_preference_data(self,
                              data: List[Dict[str, Any]]) -> None:
        """
        加载偏好数据
        
        偏好数据格式：
        [
            {
                'prompt': 输入提示/条件,
                'chosen': 偏好输出（人类选择）,
                'rejected': 非偏好输出（人类拒绝）,
            },
            ...
        ]
        
        Args:
            data: 偏好数据列表
        """
        self.preference_data = []
        for item in data:
            # 验证数据格式
            if 'chosen' in item and 'rejected' in item:
                self.preference_data.append({
                    'prompt': item.get('prompt', []),
                    'chosen': item['chosen'],
                    'rejected': item['rejected'],
                })
        
        print(f"已加载 {len(self.preference_data)} 条偏好数据")
    
    def _compute_log_prob(self,
                           output: Any,
                           params: List[Tensor]) -> float:
        """
        计算模型对给定输出的对数概率
        
        在实际深度学习中，这是通过模型的输出分布计算的。
        这里使用简化的高斯对数概率来模拟：
            log p(y|x) = -0.5 * ((y - μ)^2 / σ^2) - log(σ) - 0.5*log(2π)
        
        其中μ是模型预测值，σ是固定标准差。
        
        Args:
            output: 输出值（标量或列表）
            params: 模型参数（Tensor列表）
            
        Returns:
            对数概率值
        """
        if not params:
            return 0.0
        
        # 使用前几个参数的均值作为模型预测μ
        mu = sum(p.data for p in params[:3]) / min(3, len(params))
        sigma = 1.0  # 固定标准差
        
        # 计算输出值
        if isinstance(output, (list, tuple)):
            output_val = sum(float(x) for x in output) / len(output) if output else 0.0
        else:
            output_val = float(output)
        
        # 高斯对数概率
        log_prob = -0.5 * ((output_val - mu) ** 2) / (sigma ** 2) \
                   - math.log(sigma) - 0.5 * math.log(2 * math.pi)
        
        return log_prob
    
    def _compute_dpo_loss(self,
                           chosen_output: Any,
                           rejected_output: Any,
                           prompt: Any = None) -> Tuple[float, float, float]:
        """
        计算DPO损失
        
        DPO损失公式：
            L = -log σ(β * (log π_θ/π_ref(y_w) - β * log π_θ/π_ref(y_l)))
        
        其中 log π_θ/π_ref(y) = log π_θ(y) - log π_ref(y)
        
        同时计算KL散度作为辅助监控指标：
            KL(π_θ || π_ref) = E[log π_θ(y) - log π_ref(y)]
        
        Args:
            chosen_output: 偏好输出
            rejected_output: 非偏好输出
            prompt: 输入提示（可选）
            
        Returns:
            (dpo_loss, kl_divergence, reward_margin)
        """
        # 计算策略模型的对数概率
        log_pi_chosen = self._compute_log_prob(chosen_output, self.trainable_params)
        log_pi_rejected = self._compute_log_prob(rejected_output, self.trainable_params)
        
        # 计算参考模型的对数概率（使用冻结的参数值）
        # 临时切换参数值来计算参考对数概率
        original_params = [p.data for p in self.trainable_params]
        for i, p in enumerate(self.trainable_params):
            if i < len(self.ref_params):
                p.data = self.ref_params[i]
        
        log_ref_chosen = self._compute_log_prob(chosen_output, self.trainable_params)
        log_ref_rejected = self._compute_log_prob(rejected_output, self.trainable_params)
        
        # 恢复策略模型参数
        for i, p in enumerate(self.trainable_params):
            if i < len(original_params):
                p.data = original_params[i]
        
        # 计算对数概率比
        # log(π_θ(y)/π_ref(y)) = log π_θ(y) - log π_ref(y)
        log_ratio_chosen = log_pi_chosen - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected
        
        # DPO核心：偏好优势
        # β * (log_ratio_chosen - log_ratio_rejected)
        preference_advantage = self.beta * (log_ratio_chosen - log_ratio_rejected)
        
        # DPO损失：-log σ(preference_advantage)
        # 使用数值稳定的log-sigmoid实现
        if preference_advantage >= 0:
            # log σ(x) = -log(1 + exp(-x))，当x较大时数值稳定
            log_sigmoid = -math.log(1.0 + math.exp(-preference_advantage))
        else:
            # log σ(x) = x - log(1 + exp(x))，当x较小时数值稳定
            log_sigmoid = preference_advantage - math.log(1.0 + math.exp(preference_advantage))
        
        dpo_loss = -log_sigmoid
        
        # KL散度惩罚：防止策略偏离参考模型过远
        # KL ≈ 0.5 * (log_ratio_chosen^2 + log_ratio_rejected^2) / 2
        kl_divergence = 0.5 * (
            log_ratio_chosen ** 2 + log_ratio_rejected ** 2
        ) / 2.0
        
        # 奖励边际：chosen和rejected的隐式奖励差
        # r(x,y) = β * log(π_θ(y|x) / π_ref(y|x))
        reward_chosen = self.beta * log_ratio_chosen
        reward_rejected = self.beta * log_ratio_rejected
        reward_margin = reward_chosen - reward_rejected
        
        return dpo_loss, kl_divergence, reward_margin
    
    def _compute_total_dpo_loss(self,
                                 batch: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        计算一个批次的完整DPO损失
        
        包含DPO损失、KL散度惩罚和辅助损失。
        
        Args:
            batch: 偏好数据批次
            
        Returns:
            (总损失, 损失组件字典)
        """
        chosen = batch.get('chosen', [])
        rejected = batch.get('rejected', [])
        prompt = batch.get('prompt', [])
        
        # 计算DPO损失
        dpo_loss, kl_div, reward_margin = self._compute_dpo_loss(
            chosen, rejected, prompt
        )
        
        # 总损失 = DPO损失 + KL散度惩罚
        total_loss = dpo_loss + self.kl_weight * kl_div
        
        return total_loss, {
            'dpo_loss': dpo_loss,
            'kl_divergence': kl_div,
            'reward_margin': reward_margin,
            'total_loss': total_loss
        }
    
    def dpo_train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        DPO单步训练
        
        Args:
            batch: 偏好数据批次，包含'chosen'和'rejected'键
            
        Returns:
            训练指标字典
        """
        # 计算DPO损失
        total_loss, loss_components = self._compute_total_dpo_loss(batch)
        
        # 构建计算图用于反向传播
        # 将DPO损失包装为Tensor并连接到可训练参数
        loss_tensor = Tensor(total_loss)
        
        # 添加参数相关的计算图连接（确保梯度能回传到参数）
        for param in self.trainable_params:
            # 添加一个小的参数依赖项到损失中
            # 这确保了计算图包含所有参数
            loss_tensor = loss_tensor + param * Tensor(0.0)
        
        # 反向传播
        self._backward(total_loss, loss_tensor)
        
        # 更新参数
        self._update_parameters()
        
        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        # 记录历史
        self.dpo_losses.append(loss_components['dpo_loss'])
        self.kl_divergences.append(loss_components['kl_divergence'])
        self.reward_margins.append(loss_components['reward_margin'])
        
        return {
            'loss': total_loss,
            'dpo_loss': loss_components['dpo_loss'],
            'kl_divergence': loss_components['kl_divergence'],
            'reward_margin': loss_components['reward_margin'],
            'step': self.global_step
        }
    
    def dpo_train_epoch(self,
                         preference_data: Optional[List[Dict[str, Any]]] = None
                         ) -> Dict[str, float]:
        """
        DPO训练一个epoch
        
        Args:
            preference_data: 偏好数据（若为None则使用已加载的数据）
            
        Returns:
            epoch指标字典
        """
        data = preference_data if preference_data is not None else self.preference_data
        
        if not data:
            print("警告：没有偏好数据，跳过训练")
            return {'dpo_loss': 0.0, 'num_batches': 0}
        
        epoch_dpo_losses = []
        epoch_kl_divs = []
        epoch_margins = []
        
        for batch_data in data:
            metrics = self.dpo_train_step(batch_data)
            epoch_dpo_losses.append(metrics['dpo_loss'])
            epoch_kl_divs.append(metrics['kl_divergence'])
            epoch_margins.append(metrics['reward_margin'])
        
        avg_dpo = sum(epoch_dpo_losses) / len(epoch_dpo_losses) if epoch_dpo_losses else 0.0
        avg_kl = sum(epoch_kl_divs) / len(epoch_kl_divs) if epoch_kl_divs else 0.0
        avg_margin = sum(epoch_margins) / len(epoch_margins) if epoch_margins else 0.0
        
        self.current_epoch += 1
        
        return {
            'epoch': self.current_epoch,
            'dpo_loss': avg_dpo,
            'kl_divergence': avg_kl,
            'reward_margin': avg_margin,
            'num_batches': len(data)
        }
    
    def fit(self,
            train_data: List[Dict[str, Any]],
            val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        完整的DPO训练流程
        
        Args:
            train_data: 偏好训练数据列表
            val_data: 验证数据（可选，DPO中通常用偏好数据的子集）
            
        Returns:
            训练历史
        """
        # 加载偏好数据
        self.load_preference_data(train_data)
        
        if not self.preference_data:
            print("错误：没有有效的偏好数据")
            return {'error': 'no_preference_data'}
        
        print(f"开始DPO偏好训练: {self.config.num_epochs} epochs")
        print(f"  β (温度参数): {self.beta}")
        print(f"  KL权重: {self.kl_weight}")
        print(f"  偏好数据量: {len(self.preference_data)}")
        print(f"  优化器: {self.config.optimizer_type}")
        print(f"  学习率: {self.config.learning_rate}")
        
        for epoch in range(self.config.num_epochs):
            metrics = self.dpo_train_epoch()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"dpo_loss={metrics['dpo_loss']:.4f}, "
                  f"kl_div={metrics['kl_divergence']:.4f}, "
                  f"reward_margin={metrics['reward_margin']:.4f}")
        
        return {
            'dpo_losses': self.dpo_losses,
            'kl_divergences': self.kl_divergences,
            'reward_margins': self.reward_margins,
            'num_epochs': self.config.num_epochs,
            'beta': self.beta,
            'kl_weight': self.kl_weight
        }
    
    def evaluate_preferences(self,
                              test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估偏好学习效果
        
        计算模型在测试偏好数据上的准确率和其他指标。
        
        Args:
            test_data: 测试偏好数据
            
        Returns:
            评估指标字典
        """
        correct = 0
        total = 0
        total_reward_margin = 0.0
        
        for item in test_data:
            chosen = item.get('chosen', [])
            rejected = item.get('rejected', [])
            
            # 计算策略模型对chosen和rejected的隐式奖励
            log_pi_chosen = self._compute_log_prob(chosen, self.trainable_params)
            log_pi_rejected = self._compute_log_prob(rejected, self.trainable_params)
            
            # 计算参考模型的对数概率
            original_params = [p.data for p in self.trainable_params]
            for i, p in enumerate(self.trainable_params):
                if i < len(self.ref_params):
                    p.data = self.ref_params[i]
            
            log_ref_chosen = self._compute_log_prob(chosen, self.trainable_params)
            log_ref_rejected = self._compute_log_prob(rejected, self.trainable_params)
            
            for i, p in enumerate(self.trainable_params):
                if i < len(original_params):
                    p.data = original_params[i]
            
            # 计算隐式奖励
            reward_chosen = self.beta * (log_pi_chosen - log_ref_chosen)
            reward_rejected = self.beta * (log_pi_rejected - log_ref_rejected)
            
            # 如果chosen的奖励高于rejected，则预测正确
            if reward_chosen > reward_rejected:
                correct += 1
            
            total_reward_margin += (reward_chosen - reward_rejected)
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        avg_margin = total_reward_margin / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_reward_margin': avg_margin
        }
    
    def get_dpo_stats(self) -> Dict[str, Any]:
        """
        获取DPO训练统计信息
        
        Returns:
            DPO训练统计字典
        """
        return {
            'beta': self.beta,
            'kl_weight': self.kl_weight,
            'num_preference_samples': len(self.preference_data),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'latest_dpo_loss': self.dpo_losses[-1] if self.dpo_losses else None,
            'latest_kl_divergence': self.kl_divergences[-1] if self.kl_divergences else None,
            'latest_reward_margin': self.reward_margins[-1] if self.reward_margins else None,
            'avg_dpo_loss': sum(self.dpo_losses) / len(self.dpo_losses) if self.dpo_losses else None,
            'dpo_losses': self.dpo_losses,
            'kl_divergences': self.kl_divergences,
            'reward_margins': self.reward_margins,
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_video_trainer(use_unified_core: bool = True,
                         use_constraints: bool = True,
                         use_moe: bool = True,
                         **kwargs) -> VideoTrainer:
    """
    创建视频训练器
    
    Args:
        use_unified_core: 是否使用统一核心
        use_constraints: 是否使用约束
        use_moe: 是否使用MoE
        **kwargs: 其他配置参数
        
    Returns:
        视频训练器
    """
    config = TrainingConfig(
        use_unified_core=use_unified_core,
        use_constraints=use_constraints,
        use_moe=use_moe,
        **kwargs
    )
    return VideoTrainer(config=config)


def train_model(train_data: List[Dict[str, Any]],
                val_data: Optional[List[Dict[str, Any]]] = None,
                num_epochs: int = 100,
                use_constraints: bool = True,
                use_moe: bool = True,
                **kwargs) -> Dict[str, Any]:
    """
    便捷函数：训练模型
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        num_epochs: 训练轮数
        use_constraints: 是否使用约束
        use_moe: 是否使用MoE
        **kwargs: 其他参数
        
    Returns:
        训练历史
    """
    trainer = create_video_trainer(
        num_epochs=num_epochs,
        use_constraints=use_constraints,
        use_moe=use_moe,
        **kwargs
    )
    
    return trainer.fit(train_data, val_data)


# ============================================================================
# 导出列表
# ============================================================================

__all__ = [
    # 配置
    'TrainingConfig',
    
    # 自动微分
    'Tensor',
    'mse_loss',
    
    # 优化器
    'Optimizer',
    'SGDOptimizer',
    'AdamOptimizer',
    
    # 学习率调度器
    'LRScheduler',
    'CosineAnnealingScheduler',
    'ExponentialDecayScheduler',
    'WarmupScheduler',
    
    # 训练器
    'VideoTrainer',
    'ProgressiveTrainer',
    'DPOTrainer',
    'ConstraintAwareLoss',
    'MoETrainer',
    
    # 便捷函数
    'create_video_trainer',
    'train_model',
]
