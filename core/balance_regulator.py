"""
调节层 - 内省、正则化与反思 (Balance Regulator Layer)

胜复学架构的第四层，负责模型的自我监控、动态调节与反思生成。
通过内省计算不确定性，动态调整正则化强度，并在必要时触发反思机制。

核心功能:
- 内省计算: 不确定性估计、KL散度分析
- 动态正则化: L2、Dropout强度自适应调整
- 学习率调节: 基于训练状态的智能调整
- 反思生成: 当系统检测到异常时生成反思文本
- 防胜过亢: 在高置信度时增加探索
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    NamedTuple, Protocol, runtime_checkable
)
from enum import Enum, auto
from collections import deque
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


# ============================================================
# 1. 类型定义与数据结构
# ============================================================

class IntrospectionLevel(Enum):
    """内省级别"""
    NONE = auto()      # 无内省
    LIGHT = auto()     # 轻量级（仅监控）
    STANDARD = auto()  # 标准（监控+简单分析）
    DEEP = auto()      # 深度（完整分析+反思）


class RegularizationType(Enum):
    """正则化类型"""
    L2 = auto()
    DROPOUT = auto()
    EWC = auto()       # Elastic Weight Consolidation
    MAX_NORM = auto()  # 最大范数约束
    SPECTRAL = auto()  # 谱范数正则化


class ReflectionTrigger(Enum):
    """反思触发条件"""
    UNCERTAINTY_SPIKE = auto()   # 不确定性激增
    LOSS_PLATEAU = auto()        # 损失平台期
    GRADIENT_VANISH = auto()     # 梯度消失
    GRADIENT_EXPLODE = auto()    # 梯度爆炸
    OVERCONFIDENCE = auto()      # 过度自信
    HALT_WARNING = auto()        # 郁值预警


@dataclass
class IntrospectionResult:
    """内省结果数据结构"""
    uncertainty: float                    # 不确定性估计值
    kl_divergence: float                  # KL散度
    confidence: float                     # 置信度
    entropy: float                        # 预测熵
    gradient_norm: float                  # 梯度范数
    parameter_norm: float                 # 参数范数
    effective_lr: float                   # 有效学习率
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'uncertainty': self.uncertainty,
            'kl_divergence': self.kl_divergence,
            'confidence': self.confidence,
            'entropy': self.entropy,
            'gradient_norm': self.gradient_norm,
            'parameter_norm': self.parameter_norm,
            'effective_lr': self.effective_lr,
            'timestamp': self.timestamp,
            **self.metadata
        }


@dataclass
class RegularizationState:
    """正则化状态"""
    l2_lambda: float = 1e-4
    dropout_rate: float = 0.1
    ewc_lambda: float = 1e4
    max_norm: float = 1.0
    spectral_lambda: float = 1e-3
    current_strength: float = 1.0  # 当前整体正则化强度


@dataclass
class ReflectionRecord:
    """反思记录"""
    trigger: ReflectionTrigger
    content: str
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    action_taken: Optional[str] = None


@dataclass
class BalanceConfig:
    """调节层配置"""
    # 内省配置
    introspection_level: IntrospectionLevel = IntrospectionLevel.STANDARD
    mc_dropout_iterations: int = 10
    uncertainty_threshold: float = 0.5
    kl_divergence_threshold: float = 1.0

    # 正则化配置
    base_l2_lambda: float = 1e-4
    base_dropout_rate: float = 0.1
    max_regularization_multiplier: float = 5.0
    min_regularization_multiplier: float = 0.1

    # 学习率配置
    base_lr: float = 1e-3
    lr_adjustment_factor: float = 0.5
    min_lr: float = 1e-6
    max_lr: float = 1e-1

    # 反思配置
    reflection_enabled: bool = True
    reflection_history_size: int = 100
    reflection_trigger_threshold: float = 0.7

    # 防胜过亢配置
    overconfidence_threshold: float = 0.9
    exploration_boost: float = 0.2

    # 郁值响应配置
    halt_warning_threshold: float = 0.7
    halt_critical_threshold: float = 0.9


# ============================================================
# 2. 不确定性估计器
# ============================================================

class UncertaintyEstimator(ABC):
    """不确定性估计器抽象基类"""

    @abstractmethod
    def estimate(self, model: nn.Module, data: torch.Tensor) -> float:
        """
        估计模型在给定数据上的不确定性

        Args:
            model: 神经网络模型
            data: 输入数据

        Returns:
            不确定性分数（越高表示越不确定）
        """
        pass


class MCDropoutEstimator(UncertaintyEstimator):
    """
    MC Dropout不确定性估计器

    通过多次前向传播（启用dropout）来估计预测不确定性。
    不确定性 = 预测方差
    """

    def __init__(self, num_iterations: int = 10):
        self.num_iterations = num_iterations

    def estimate(self, model: nn.Module, data: torch.Tensor) -> float:
        """
        使用MC Dropout估计不确定性

        Args:
            model: 模型
            data: 输入数据 [batch_size, ...]

        Returns:
            不确定性分数
        """
        model.train()  # 保持train模式以启用dropout

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                output = model(data)
                if output.dim() > 1 and output.shape[-1] > 1:
                    # 分类任务：取softmax概率
                    probs = F.softmax(output, dim=-1)
                    predictions.append(probs)
                else:
                    # 回归任务
                    predictions.append(output)

        if not predictions:
            return 0.0

        # 堆叠预测结果
        stacked = torch.stack(predictions)  # [num_iterations, batch_size, num_classes]

        # 计算预测方差作为不确定性
        mean_pred = stacked.mean(dim=0)  # [batch_size, num_classes]
        variance = stacked.var(dim=0).mean().item()  # 标量

        # 同时考虑预测熵
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1).mean().item()

        # 综合不确定性
        uncertainty = variance + 0.1 * entropy

        return uncertainty


class EnsembleUncertaintyEstimator(UncertaintyEstimator):
    """
    集成不确定性估计器

    使用模型快照集成来估计不确定性。
    """

    def __init__(self, num_snapshots: int = 5):
        self.num_snapshots = num_snapshots
        self.snapshots: List[Dict[str, torch.Tensor]] = []

    def add_snapshot(self, model: nn.Module):
        """添加模型快照"""
        snapshot = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.snapshots.append(snapshot)

        # 保持固定数量的快照
        if len(self.snapshots) > self.num_snapshots:
            self.snapshots.pop(0)

    def estimate(self, model: nn.Module, data: torch.Tensor) -> float:
        """使用集成估计不确定性"""
        if len(self.snapshots) < 2:
            return 0.0

        original_state = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        predictions = []
        model.eval()

        with torch.no_grad():
            # 使用原始模型
            output = model(data)
            if output.dim() > 1 and output.shape[-1] > 1:
                predictions.append(F.softmax(output, dim=-1))
            else:
                predictions.append(output)

            # 使用每个快照
            for snapshot in self.snapshots:
                for name, param in model.named_parameters():
                    if name in snapshot:
                        param.data.copy_(snapshot[name])

                output = model(data)
                if output.dim() > 1 and output.shape[-1] > 1:
                    predictions.append(F.softmax(output, dim=-1))
                else:
                    predictions.append(output)

        # 恢复原始状态
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])

        # 计算集成不确定性
        stacked = torch.stack(predictions)
        variance = stacked.var(dim=0).mean().item()

        return variance


class GradientUncertaintyEstimator(UncertaintyEstimator):
    """
    基于梯度的不确定性估计器

    使用Fisher信息矩阵的对角线近似来估计参数不确定性。
    """

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples

    def estimate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """
        使用Fisher信息估计不确定性

        Args:
            model: 模型
            data_loader: 数据加载器

        Returns:
            Fisher信息迹的平均值（作为不确定性度量）
        """
        model.eval()
        fisher_trace = 0.0
        num_params = 0
        sample_count = 0

        for batch in data_loader:
            if sample_count >= self.num_samples:
                break

            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(next(model.parameters()).device)

            model.zero_grad()
            outputs = model(inputs)

            # 使用预测分布
            if outputs.dim() > 1 and outputs.shape[-1] > 1:
                log_probs = F.log_softmax(outputs, dim=-1)
                # 采样预测
                probs = F.softmax(outputs, dim=-1)
                sampled_classes = torch.multinomial(probs, 1).squeeze(-1)

                for i in range(min(inputs.size(0), self.num_samples - sample_count)):
                    loss = -log_probs[i, sampled_classes[i]]
                    loss.backward(retain_graph=True)

                    # 累积梯度平方
                    for param in model.parameters():
                        if param.grad is not None:
                            fisher_trace += (param.grad ** 2).sum().item()
                            num_params += param.numel()

                    sample_count += 1
                    if sample_count >= self.num_samples:
                        break

            model.zero_grad()

        if num_params > 0:
            return fisher_trace / num_params
        return 0.0


# ============================================================
# 3. KL散度计算器
# ============================================================

class KLDivergenceCalculator:
    """
    KL散度计算器

    计算当前模型分布与历史分布之间的KL散度，用于检测分布漂移。
    """

    @staticmethod
    def compute_gaussian_kl(
        mu_current: torch.Tensor,
        sigma_current: torch.Tensor,
        mu_historical: torch.Tensor,
        sigma_historical: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两个高斯分布之间的KL散度

        KL(N(mu_current, sigma_current) || N(mu_historical, sigma_historical))

        Args:
            mu_current: 当前均值
            sigma_current: 当前标准差
            mu_historical: 历史均值
            sigma_historical: 历史标准差

        Returns:
            KL散度值
        """
        sigma_c = sigma_current.clamp(min=1e-8)
        sigma_h = sigma_historical.clamp(min=1e-8)

        kl = (
            torch.log(sigma_h / sigma_c) +
            (sigma_c ** 2 + (mu_current - mu_historical) ** 2) / (2 * sigma_h ** 2) -
            0.5
        )

        return kl.sum()

    @staticmethod
    def compute_categorical_kl(
        p_current: torch.Tensor,
        p_historical: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两个分类分布之间的KL散度

        Args:
            p_current: 当前概率分布
            p_historical: 历史概率分布

        Returns:
            KL散度值
        """
        p_c = p_current.clamp(min=1e-10)
        p_h = p_historical.clamp(min=1e-10)

        kl = (p_c * (torch.log(p_c) - torch.log(p_h))).sum()
        return kl

    @staticmethod
    def compute_model_kl(
        model_current: nn.Module,
        model_historical: nn.Module,
        sample_input: torch.Tensor
    ) -> float:
        """
        计算两个模型输出分布之间的KL散度

        Args:
            model_current: 当前模型
            model_historical: 历史模型
            sample_input: 样本输入

        Returns:
            KL散度值
        """
        model_current.eval()
        model_historical.eval()

        with torch.no_grad():
            output_c = model_current(sample_input)
            output_h = model_historical(sample_input)

            if output_c.dim() > 1 and output_c.shape[-1] > 1:
                # 分类任务
                p_c = F.softmax(output_c, dim=-1)
                p_h = F.softmax(output_h, dim=-1)
                kl = (p_c * (torch.log(p_c + 1e-10) - torch.log(p_h + 1e-10))).sum(dim=-1).mean()
            else:
                # 回归任务 - 使用高斯近似
                mu_c = output_c.mean()
                sigma_c = output_c.std().clamp(min=1e-8)
                mu_h = output_h.mean()
                sigma_h = output_h.std().clamp(min=1e-8)

                kl = (
                    torch.log(sigma_h / sigma_c) +
                    (sigma_c ** 2 + (mu_c - mu_h) ** 2) / (2 * sigma_h ** 2) - 0.5
                )

        return kl.item()


# ============================================================
# 4. 学习率调节器
# ============================================================

class LearningRateAdjuster:
    """
    学习率调节器

    基于训练状态智能调整学习率。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        max_lr: float = 1e-1,
        adjustment_factor: float = 0.5
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adjustment_factor = adjustment_factor

        self.lr_history: List[float] = []
        self.loss_history: deque = deque(maxlen=10)
        self.plateau_count: int = 0

    def get_current_lr(self) -> float:
        """获取当前学习率"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return self.base_lr

    def set_lr(self, lr: float):
        """设置学习率"""
        lr = max(self.min_lr, min(self.max_lr, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr_history.append(lr)

    def adjust_based_on_loss(self, current_loss: float):
        """
        基于损失变化调整学习率

        Args:
            current_loss: 当前损失值
        """
        self.loss_history.append(current_loss)

        if len(self.loss_history) < 5:
            return

        # 检测损失平台期
        recent_losses = list(self.loss_history)[-5:]
        loss_variance = sum((l - sum(recent_losses)/len(recent_losses))**2
                           for l in recent_losses) / len(recent_losses)

        if loss_variance < 1e-6:  # 平台期
            self.plateau_count += 1
            if self.plateau_count >= 3:
                # 降低学习率
                current_lr = self.get_current_lr()
                new_lr = current_lr * self.adjustment_factor
                self.set_lr(new_lr)
                self.plateau_count = 0
        else:
            self.plateau_count = max(0, self.plateau_count - 1)

    def adjust_based_on_gradient(self, gradient_norm: float):
        """
        基于梯度范数调整学习率

        Args:
            gradient_norm: 梯度范数
        """
        if gradient_norm > 10.0:  # 梯度爆炸
            current_lr = self.get_current_lr()
            self.set_lr(current_lr * 0.5)
        elif gradient_norm < 1e-6:  # 梯度消失
            current_lr = self.get_current_lr()
            self.set_lr(min(current_lr * 2.0, self.max_lr))

    def adjust_based_on_uncertainty(self, uncertainty: float, threshold: float = 0.5):
        """
        基于不确定性调整学习率

        高不确定性时降低学习率以增加稳定性

        Args:
            uncertainty: 不确定性值
            threshold: 不确定性阈值
        """
        if uncertainty > threshold:
            current_lr = self.get_current_lr()
            new_lr = current_lr * (1 - 0.3 * (uncertainty - threshold))
            self.set_lr(new_lr)

    def boost_for_exploration(self, boost_factor: float = 2.0):
        """
        临时提升学习率以促进探索

        Args:
            boost_factor: 提升倍数
        """
        current_lr = self.get_current_lr()
        self.set_lr(min(current_lr * boost_factor, self.max_lr))


# ============================================================
# 5. 反思生成器
# ============================================================

class ReflectionGenerator:
    """
    反思生成器

    当系统检测到异常状态时，生成结构化的反思文本。
    """

    def __init__(self, config: BalanceConfig):
        self.config = config
        self.reflection_history: deque = deque(maxlen=config.reflection_history_size)
        self.reflection_templates = self._load_templates()

    def _load_templates(self) -> Dict[ReflectionTrigger, List[str]]:
        """加载反思模板"""
        return {
            ReflectionTrigger.UNCERTAINTY_SPIKE: [
                "检测到不确定性激增（{uncertainty:.3f} > {threshold:.3f}）。"
                "模型可能对当前输入缺乏信心。建议：增加数据多样性或降低学习率。",
                "不确定性异常升高至{uncertainty:.3f}。"
                "可能存在分布外样本。建议：启用更严格的正则化。",
            ],
            ReflectionTrigger.LOSS_PLATEAU: [
                "损失函数进入平台期（方差 < 1e-6）。"
                "模型可能陷入局部最优。建议：增加学习率或引入扰动。",
                "检测到损失停滞。考虑使用学习率预热或更换优化器。",
            ],
            ReflectionTrigger.GRADIENT_VANISH: [
                "检测到梯度消失（范数 < 1e-6）。"
                "反向传播信号过弱。建议：检查激活函数或添加残差连接。",
                "梯度范数异常低。可能需要调整网络架构或初始化策略。",
            ],
            ReflectionTrigger.GRADIENT_EXPLODE: [
                "检测到梯度爆炸（范数 > {gradient_norm:.1f}）。"
                "已自动应用梯度裁剪。建议：降低学习率或使用更稳定的优化器。",
                "梯度范数过高（{gradient_norm:.1f}）。"
                "考虑使用权重衰减或梯度累积。",
            ],
            ReflectionTrigger.OVERCONFIDENCE: [
                "检测到过度自信（置信度 {confidence:.3f} > {threshold:.3f}）。"
                "模型可能出现过拟合。建议：增加Dropout或数据增强。",
                "置信度过高可能暗示过拟合。建议启用更多探索机制。",
            ],
            ReflectionTrigger.HALT_WARNING: [
                "郁值预警（{halt_value:.3f} > {threshold:.3f}）。"
                "系统可能进入不稳定状态。建议：增加正则化强度并降低学习率。",
                "检测到郁值升高。启动防御性调节机制。",
            ],
        }

    def generate(
        self,
        trigger: ReflectionTrigger,
        metrics: Dict[str, float],
        context: Optional[str] = None
    ) -> str:
        """
        生成反思文本

        Args:
            trigger: 触发条件
            metrics: 相关指标
            context: 额外上下文

        Returns:
            反思文本
        """
        templates = self.reflection_templates.get(trigger, ["系统状态异常，建议检查。"])
        template = random.choice(templates)

        try:
            reflection = template.format(**metrics)
        except KeyError:
            reflection = template

        if context:
            reflection += f"\n上下文: {context}"

        # 添加建议行动
        action = self._suggest_action(trigger, metrics)
        reflection += f"\n建议行动: {action}"

        # 记录反思
        record = ReflectionRecord(
            trigger=trigger,
            content=reflection,
            metrics=metrics,
            action_taken=action
        )
        self.reflection_history.append(record)

        return reflection

    def _suggest_action(
        self,
        trigger: ReflectionTrigger,
        metrics: Dict[str, float]
    ) -> str:
        """根据触发条件建议行动"""
        actions = {
            ReflectionTrigger.UNCERTAINTY_SPIKE: "increase_regularization",
            ReflectionTrigger.LOSS_PLATEAU: "boost_learning_rate",
            ReflectionTrigger.GRADIENT_VANISH: "check_architecture",
            ReflectionTrigger.GRADIENT_EXPLODE: "clip_gradients",
            ReflectionTrigger.OVERCONFIDENCE: "increase_dropout",
            ReflectionTrigger.HALT_WARNING: "defensive_regulation",
        }
        return actions.get(trigger, "monitor_closely")

    def get_reflection_summary(self, n_recent: int = 5) -> str:
        """获取最近反思的摘要"""
        recent = list(self.reflection_history)[-n_recent:]
        if not recent:
            return "暂无反思记录。"

        summary_parts = []
        for record in recent:
            summary_parts.append(
                f"[{time.strftime('%H:%M:%S', time.localtime(record.timestamp))}] "
                f"{record.trigger.name}: {record.action_taken}"
            )

        return "\n".join(summary_parts)


# ============================================================
# 6. 主调节器类
# ============================================================

class BalanceRegulator:
    """
    平衡调节器

    胜复学架构的调节层核心类，负责内省、正则化调节和反思生成。

    Attributes:
        config: 调节层配置
        uncertainty_estimator: 不确定性估计器
        lr_adjuster: 学习率调节器
        reflection_generator: 反思生成器
        kl_calculator: KL散度计算器
        regularization_state: 当前正则化状态
        introspection_history: 内省历史记录
    """

    def __init__(self, config: Optional[BalanceConfig] = None):
        """
        初始化平衡调节器

        Args:
            config: 调节层配置，使用默认配置如果为None
        """
        self.config = config or BalanceConfig()

        # 初始化组件
        self.uncertainty_estimator = MCDropoutEstimator(
            self.config.mc_dropout_iterations
        )
        self.kl_calculator = KLDivergenceCalculator()
        self.reflection_generator = ReflectionGenerator(self.config)

        # 状态
        self.regularization_state = RegularizationState(
            l2_lambda=self.config.base_l2_lambda,
            dropout_rate=self.config.base_dropout_rate
        )
        self.lr_adjuster: Optional[LearningRateAdjuster] = None
        self.introspection_history: deque = deque(maxlen=100)
        self.historical_params: List[Dict[str, torch.Tensor]] = []
        self.halt_value_history: deque = deque(maxlen=50)

        # 统计
        self.regulation_count: int = 0
        self.reflection_count: int = 0

    def attach_optimizer(self, optimizer: Optimizer):
        """
        附加优化器以进行学习率调节

        Args:
            optimizer: PyTorch优化器
        """
        self.lr_adjuster = LearningRateAdjuster(
            optimizer=optimizer,
            base_lr=self.config.base_lr,
            min_lr=self.config.min_lr,
            max_lr=self.config.max_lr,
            adjustment_factor=self.config.lr_adjustment_factor
        )

    def introspect(
        self,
        model: nn.Module,
        history: Optional[Dict[str, Any]] = None
    ) -> IntrospectionResult:
        """
        执行内省分析

        分析模型的当前状态，包括不确定性、梯度、参数分布等。

        Args:
            model: 要分析的模型
            history: 训练历史数据（可选）

        Returns:
            内省结果
        """
        if self.config.introspection_level == IntrospectionLevel.NONE:
            return IntrospectionResult(
                uncertainty=0.0,
                kl_divergence=0.0,
                confidence=0.0,
                entropy=0.0,
                gradient_norm=0.0,
                parameter_norm=0.0,
                effective_lr=self.lr_adjuster.get_current_lr() if self.lr_adjuster else self.config.base_lr
            )

        model.eval()

        # 计算参数范数
        parameter_norm = sum(
            p.norm().item() for p in model.parameters()
        ) / sum(1 for _ in model.parameters())

        # 计算梯度范数
        gradient_norm = 0.0
        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                gradient_norm += p.grad.norm().item() ** 2
                has_grad = True
        if has_grad:
            gradient_norm = math.sqrt(gradient_norm)

        # 获取学习率
        effective_lr = (
            self.lr_adjuster.get_current_lr()
            if self.lr_adjuster else self.config.base_lr
        )

        # 轻量级内省到此为止
        if self.config.introspection_level == IntrospectionLevel.LIGHT:
            result = IntrospectionResult(
                uncertainty=0.0,
                kl_divergence=0.0,
                confidence=0.0,
                entropy=0.0,
                gradient_norm=gradient_norm,
                parameter_norm=parameter_norm,
                effective_lr=effective_lr
            )
            self.introspection_history.append(result)
            return result

        # 标准/深度内省：计算更多指标
        # 使用随机样本估计不确定性
        sample_input = torch.randn(1, 10)  # 简化示例
        try:
            uncertainty = self.uncertainty_estimator.estimate(model, sample_input)
        except Exception:
            uncertainty = 0.0

        # 计算KL散度（如果有历史参数）
        kl_divergence = 0.0
        if self.historical_params and self.config.introspection_level == IntrospectionLevel.DEEP:
            kl_divergence = self._compute_average_kl(model)

        # 计算置信度和熵
        confidence, entropy = self._compute_prediction_stats(model, sample_input)

        result = IntrospectionResult(
            uncertainty=uncertainty,
            kl_divergence=kl_divergence,
            confidence=confidence,
            entropy=entropy,
            gradient_norm=gradient_norm,
            parameter_norm=parameter_norm,
            effective_lr=effective_lr,
            metadata={
                'num_historical_params': len(self.historical_params),
                'regulation_count': self.regulation_count
            }
        )

        self.introspection_history.append(result)
        return result

    def compute_uncertainty(
        self,
        model: nn.Module,
        data: torch.Tensor
    ) -> float:
        """
        计算MC Dropout不确定性

        Args:
            model: 模型
            data: 输入数据

        Returns:
            不确定性分数
        """
        return self.uncertainty_estimator.estimate(model, data)

    def compute_kl_divergence(
        self,
        current: nn.Module,
        historical: Union[nn.Module, Dict[str, torch.Tensor]]
    ) -> float:
        """
        计算当前模型与历史模型之间的KL散度

        Args:
            current: 当前模型
            historical: 历史模型或参数字典

        Returns:
            KL散度值
        """
        sample_input = torch.randn(1, 10)

        if isinstance(historical, nn.Module):
            return self.kl_calculator.compute_model_kl(current, historical, sample_input)
        else:
            # 从参数字典创建临时模型
            return self._compute_kl_from_params(current, historical, sample_input)

    def _compute_kl_from_params(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        sample_input: torch.Tensor
    ) -> float:
        """从参数字典计算KL散度"""
        original_state = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        # 加载历史参数
        for name, param in model.named_parameters():
            if name in params:
                param.data.copy_(params[name])

        # 创建历史模型副本
        historical_model = copy.deepcopy(model)

        # 恢复当前参数
        for name, param in model.named_parameters():
            if name in original_state:
                param.data.copy_(original_state[name])

        return self.kl_calculator.compute_model_kl(model, historical_model, sample_input)

    def _compute_average_kl(self, model: nn.Module) -> float:
        """计算与历史参数的平均KL散度"""
        if not self.historical_params:
            return 0.0

        kls = []
        for hist_params in self.historical_params[-5:]:  # 只比较最近的5个
            kl = self._compute_kl_from_params(model, hist_params, torch.randn(1, 10))
            kls.append(kl)

        return sum(kls) / len(kls) if kls else 0.0

    def _compute_prediction_stats(
        self,
        model: nn.Module,
        sample_input: torch.Tensor
    ) -> Tuple[float, float]:
        """计算预测统计信息"""
        model.eval()
        with torch.no_grad():
            try:
                output = model(sample_input)
                if output.dim() > 1 and output.shape[-1] > 1:
                    probs = F.softmax(output, dim=-1)
                    confidence = probs.max(dim=-1)[0].mean().item()
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                else:
                    confidence = 0.5
                    entropy = 0.0
            except Exception:
                confidence = 0.5
                entropy = 0.0

        return confidence, entropy

    def adjust_regularization(self, halt_value: float):
        """
        根据郁值调整正则化强度

        当郁值升高时，增加正则化以防止过拟合和不稳定。

        Args:
            halt_value: 当前郁值（0-1之间，越高表示越需要调节）
        """
        self.halt_value_history.append(halt_value)

        # 计算调节因子
        if halt_value > self.config.halt_critical_threshold:
            multiplier = self.config.max_regularization_multiplier
        elif halt_value > self.config.halt_warning_threshold:
            # 线性插值
            ratio = (halt_value - self.config.halt_warning_threshold) / \
                   (self.config.halt_critical_threshold - self.config.halt_warning_threshold)
            multiplier = 1.0 + ratio * (self.config.max_regularization_multiplier - 1.0)
        else:
            multiplier = 1.0

        # 应用调节
        self.regularization_state.current_strength = multiplier
        self.regularization_state.l2_lambda = self.config.base_l2_lambda * multiplier
        self.regularization_state.dropout_rate = min(
            self.config.base_dropout_rate * multiplier,
            0.5  # 最大dropout率
        )

        self.regulation_count += 1

    def apply_l2_regularization(self, model: nn.Module) -> torch.Tensor:
        """
        应用L2正则化

        Args:
            model: 模型

        Returns:
            L2正则化损失
        """
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += param.pow(2).sum()

        return self.regularization_state.l2_lambda * l2_loss

    def apply_ewc_loss(
        self,
        base_loss: torch.Tensor,
        ewc: Any  # ElasticWeightConsolidation
    ) -> torch.Tensor:
        """
        应用EWC正则化损失

        Args:
            base_loss: 基础损失
            ewc: EWC实例

        Returns:
            总损失（基础损失 + EWC损失）
        """
        if hasattr(ewc, 'compute_total_loss'):
            total_loss, stats = ewc.compute_total_loss(base_loss)
            return total_loss
        return base_loss

    def generate_reflection(self, metrics: Dict[str, float]) -> Optional[str]:
        """
        根据指标生成反思

        Args:
            metrics: 包含各种指标的字典
                - uncertainty: 不确定性
                - confidence: 置信度
                - gradient_norm: 梯度范数
                - halt_value: 郁值

        Returns:
            反思文本，如果不需要反思则返回None
        """
        if not self.config.reflection_enabled:
            return None

        # 检查各种触发条件
        triggers = []

        if metrics.get('uncertainty', 0) > self.config.uncertainty_threshold:
            triggers.append(ReflectionTrigger.UNCERTAINTY_SPIKE)

        if metrics.get('confidence', 0) > self.config.overconfidence_threshold:
            triggers.append(ReflectionTrigger.OVERCONFIDENCE)

        gradient_norm = metrics.get('gradient_norm', 0)
        if gradient_norm > 10.0:
            triggers.append(ReflectionTrigger.GRADIENT_EXPLODE)
        elif gradient_norm < 1e-6:
            triggers.append(ReflectionTrigger.GRADIENT_VANISH)

        halt_value = metrics.get('halt_value', 0)
        if halt_value > self.config.halt_warning_threshold:
            triggers.append(ReflectionTrigger.HALT_WARNING)

        if not triggers:
            return None

        # 选择最严重的触发条件
        priority = [
            ReflectionTrigger.HALT_WARNING,
            ReflectionTrigger.GRADIENT_EXPLODE,
            ReflectionTrigger.UNCERTAINTY_SPIKE,
            ReflectionTrigger.OVERCONFIDENCE,
            ReflectionTrigger.GRADIENT_VANISH,
        ]

        selected_trigger = None
        for p in priority:
            if p in triggers:
                selected_trigger = p
                break

        if selected_trigger is None:
            selected_trigger = triggers[0]

        reflection = self.reflection_generator.generate(
            selected_trigger,
            metrics
        )

        self.reflection_count += 1
        return reflection

    def regulate(self, state_coefficient: float) -> Dict[str, Any]:
        """
        主调节函数

        根据状态系数执行综合调节。

        Args:
            state_coefficient: 状态系数（0-1之间，越高表示需要越强的调节）

        Returns:
            调节结果字典
        """
        results = {
            'state_coefficient': state_coefficient,
            'actions_taken': [],
            'regularization_multiplier': 1.0,
            'learning_rate_adjusted': False,
            'reflection_generated': None
        }

        # 1. 调整正则化
        self.adjust_regularization(state_coefficient)
        results['regularization_multiplier'] = self.regularization_state.current_strength
        results['actions_taken'].append('adjust_regularization')

        # 2. 调整学习率
        if self.lr_adjuster and state_coefficient > self.config.halt_warning_threshold:
            current_lr = self.lr_adjuster.get_current_lr()
            new_lr = current_lr * (1 - 0.3 * state_coefficient)
            self.lr_adjuster.set_lr(new_lr)
            results['learning_rate_adjusted'] = True
            results['actions_taken'].append('adjust_learning_rate')

        # 3. 生成反思
        if self.config.reflection_enabled and state_coefficient > self.config.reflection_trigger_threshold:
            metrics = {
                'halt_value': state_coefficient,
                'threshold': self.config.halt_warning_threshold
            }
            reflection = self.generate_reflection(metrics)
            if reflection:
                results['reflection_generated'] = reflection
                results['actions_taken'].append('generate_reflection')

        return results

    def prevent_overconfidence(self, confidence: float) -> Dict[str, Any]:
        """
        防胜过亢机制

        当模型过于自信时，增加探索以防止过拟合。

        Args:
            confidence: 模型置信度

        Returns:
            调节动作
        """
        actions = {
            'confidence': confidence,
            'actions_taken': [],
            'exploration_boosted': False
        }

        if confidence > self.config.overconfidence_threshold:
            # 增加Dropout
            self.regularization_state.dropout_rate = min(
                self.regularization_state.dropout_rate * 1.5,
                0.5
            )
            actions['actions_taken'].append('increase_dropout')

            # 临时提升学习率以促进探索
            if self.lr_adjuster:
                self.lr_adjuster.boost_for_exploration(
                    1 + self.config.exploration_boost
                )
                actions['exploration_boosted'] = True
                actions['actions_taken'].append('boost_learning_rate')

            # 生成反思
            if self.config.reflection_enabled:
                metrics = {
                    'confidence': confidence,
                    'threshold': self.config.overconfidence_threshold
                }
                reflection = self.reflection_generator.generate(
                    ReflectionTrigger.OVERCONFIDENCE,
                    metrics
                )
                actions['reflection'] = reflection

        return actions

    def save_model_checkpoint(self, model: nn.Module):
        """
        保存模型检查点用于KL散度计算

        Args:
            model: 模型
        """
        params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.historical_params.append(params)

        # 限制历史记录大小
        max_history = 10
        if len(self.historical_params) > max_history:
            self.historical_params = self.historical_params[-max_history:]

    def get_status(self) -> Dict[str, Any]:
        """获取调节器状态摘要"""
        return {
            'regulation_count': self.regulation_count,
            'reflection_count': self.reflection_count,
            'introspection_history_size': len(self.introspection_history),
            'historical_params_count': len(self.historical_params),
            'current_regularization_strength': self.regularization_state.current_strength,
            'current_l2_lambda': self.regularization_state.l2_lambda,
            'current_dropout_rate': self.regularization_state.dropout_rate,
            'recent_reflections': self.reflection_generator.get_reflection_summary(3)
            if hasattr(self, 'reflection_generator') else "N/A"
        }

    def reset(self):
        """重置调节器状态"""
        self.regularization_state = RegularizationState(
            l2_lambda=self.config.base_l2_lambda,
            dropout_rate=self.config.base_dropout_rate
        )
        self.introspection_history.clear()
        self.historical_params.clear()
        self.halt_value_history.clear()
        self.regulation_count = 0
        self.reflection_count = 0


# ============================================================
# 7. 便捷函数
# ============================================================

def create_balance_regulator(
    base_lr: float = 1e-3,
    introspection_level: str = "standard"
) -> BalanceRegulator:
    """
    创建平衡调节器的便捷函数

    Args:
        base_lr: 基础学习率
        introspection_level: 内省级别 ("none", "light", "standard", "deep")

    Returns:
        配置好的BalanceRegulator实例
    """
    level_map = {
        "none": IntrospectionLevel.NONE,
        "light": IntrospectionLevel.LIGHT,
        "standard": IntrospectionLevel.STANDARD,
        "deep": IntrospectionLevel.DEEP
    }

    config = BalanceConfig(
        base_lr=base_lr,
        introspection_level=level_map.get(introspection_level, IntrospectionLevel.STANDARD)
    )

    return BalanceRegulator(config)


def quick_introspect(model: nn.Module) -> Dict[str, float]:
    """
    快速内省函数

    Args:
        model: 模型

    Returns:
        关键指标字典
    """
    regulator = BalanceRegulator(BalanceConfig(introspection_level=IntrospectionLevel.LIGHT))
    result = regulator.introspect(model)
    return result.to_dict()
