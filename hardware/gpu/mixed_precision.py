"""
Mixed Precision Training - FP16/BF16混合精度训练

模块路径: hardware/gpu/mixed_precision.py

提供混合精度训练功能，支持FP16和BF16格式，
自动损失缩放和梯度缩放，提高训练速度和内存效率。
"""

import logging
import warnings
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class PrecisionType(Enum):
    """精度类型枚举"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class MixedPrecisionConfig:
    """混合精度训练配置"""
    precision: str = "fp16"  # fp16, bf16, fp32
    enabled: bool = True
    loss_scale: Optional[float] = None  # 静态损失缩放值
    dynamic_loss_scale: bool = True  # 动态损失缩放
    init_scale: float = 2.0**16  # 初始缩放值
    growth_factor: float = 2.0  # 增长因子
    backoff_factor: float = 0.5  # 回退因子
    growth_interval: int = 2000  # 增长间隔
    enabled_backwards: bool = True  # 启用反向传播优化


class MixedPrecisionTrainer:
    """
    混合精度训练器
    
    管理FP16/BF16混合精度训练，包括自动损失缩放和梯度缩放。
    """
    
    def __init__(self, config: Optional[MixedPrecisionConfig] = None, precision: str = "fp16"):
        """
        初始化混合精度训练器
        
        Args:
            config: 混合精度配置
            precision: 精度类型（fp16/bf16/fp32）
        """
        if config is None:
            config = MixedPrecisionConfig(precision=precision)
        self.config = config
        
        self._precision = PrecisionType(config.precision)
        self._enabled = config.enabled and self._precision != PrecisionType.FP32
        
        # 检查设备支持
        if not torch.cuda.is_available():
            self._enabled = False
            warnings.warn("CUDA not available, mixed precision disabled")
        
        # 检查BF16支持
        if self._precision == PrecisionType.BF16:
            if not torch.cuda.is_bf16_supported():
                warnings.warn("BF16 not supported on this device, falling back to FP16")
                self._precision = PrecisionType.FP16
        
        # 初始化GradScaler（仅用于FP16）
        self._scaler: Optional[GradScaler] = None
        if self._enabled and self._precision == PrecisionType.FP16:
            self._scaler = GradScaler(
                init_scale=config.init_scale,
                growth_factor=config.growth_factor,
                backoff_factor=config.backoff_factor,
                growth_interval=config.growth_interval,
                enabled=config.dynamic_loss_scale
            )
        
        self._step_count = 0
        self._overflow_count = 0
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        准备模型进行混合精度训练
        
        Args:
            model: PyTorch模型
            
        Returns:
            准备好的模型
        """
        if not self._enabled:
            return model
        
        # 将模型转换为适当的精度
        if self._precision == PrecisionType.FP16:
            # FP16: 保持FP32主权重，前向传播使用FP16
            pass
        elif self._precision == PrecisionType.BF16:
            # BF16: 可以直接使用BF16权重
            model = model.to(torch.bfloat16)
        
        return model
    
    def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """
        准备优化器
        
        Args:
            optimizer: PyTorch优化器
            
        Returns:
            准备好的优化器
        """
        # 优化器通常不需要特殊处理
        return optimizer
    
    @contextmanager
    def autocast_context(self):
        """
        自动类型转换上下文
        
        在上下文中自动将操作转换为低精度。
        """
        if not self._enabled:
            yield
            return
        
        dtype = torch.float16 if self._precision == PrecisionType.FP16 else torch.bfloat16
        
        with autocast(device_type='cuda', dtype=dtype, enabled=self._enabled):
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        缩放损失
        
        Args:
            loss: 原始损失
            
        Returns:
            缩放后的损失
        """
        if not self._enabled or self._scaler is None:
            return loss
        
        return self._scaler.scale(loss)
    
    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """
        反向传播
        
        Args:
            loss: 损失张量
            **kwargs: 其他参数
        """
        if self._scaler is not None:
            self._scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)
    
    def step(self, optimizer: Optimizer) -> bool:
        """
        执行优化步骤
        
        Args:
            optimizer: 优化器
            
        Returns:
            是否成功执行（没有溢出）
        """
        self._step_count += 1
        
        if self._scaler is not None:
            # 检查梯度溢出
            self._scaler.unscale_(optimizer)
            
            # 裁剪梯度（可选）
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # 更新权重
            self._scaler.step(optimizer)
            self._scaler.update()
            
            # 检查是否溢出
            if self._scaler.get_scale() < self.config.init_scale:
                self._overflow_count += 1
                return False
        else:
            optimizer.step()
        
        return True
    
    def zero_grad(self, optimizer: Optimizer, set_to_none: bool = False) -> None:
        """
        清零梯度
        
        Args:
            optimizer: 优化器
            set_to_none: 是否将梯度设为None
        """
        optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        获取状态字典
        
        Returns:
            状态字典
        """
        state = {
            "step_count": self._step_count,
            "overflow_count": self._overflow_count,
            "precision": self._precision.value
        }
        
        if self._scaler is not None:
            state["scaler_state"] = self._scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载状态字典
        
        Args:
            state_dict: 状态字典
        """
        self._step_count = state_dict.get("step_count", 0)
        self._overflow_count = state_dict.get("overflow_count", 0)
        
        if self._scaler is not None and "scaler_state" in state_dict:
            self._scaler.load_state_dict(state_dict["scaler_state"])
    
    def get_scale(self) -> float:
        """获取当前损失缩放值"""
        if self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "step_count": self._step_count,
            "overflow_count": self._overflow_count,
            "current_scale": self.get_scale(),
            "precision": self._precision.value,
            "enabled": self._enabled
        }
    
    @property
    def enabled(self) -> bool:
        """混合精度是否启用"""
        return self._enabled
    
    @property
    def precision(self) -> PrecisionType:
        """当前精度类型"""
        return self._precision


class FP16Trainer(MixedPrecisionTrainer):
    """
    FP16训练器
    
    专门用于FP16混合精度训练。
    """
    
    def __init__(self, **kwargs):
        super().__init__(precision="fp16", **kwargs)


class BF16Trainer(MixedPrecisionTrainer):
    """
    BF16训练器
    
    专门用于BF16混合精度训练。
    """
    
    def __init__(self, **kwargs):
        super().__init__(precision="bf16", **kwargs)


class MasterWeightOptimizer:
    """
    主权重优化器包装器
    
    在FP16训练中保持FP32主权重，提高数值稳定性。
    """
    
    def __init__(self, optimizer: Optimizer, model: nn.Module):
        """
        初始化主权重优化器
        
        Args:
            optimizer: 基础优化器
            model: 模型
        """
        self.optimizer = optimizer
        self.model = model
        
        # 创建FP32主权重副本
        self.master_params = []
        self.param_map = {}
        
        for param in model.parameters():
            if param.requires_grad:
                master_param = param.clone().float().detach()
                master_param.requires_grad = True
                self.master_params.append(master_param)
                self.param_map[id(master_param)] = param
        
        # 更新优化器的参数
        self.optimizer.param_groups = []
        self.optimizer.add_param_group({'params': self.master_params})
    
    def zero_grad(self, set_to_none: bool = False):
        """清零梯度"""
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
    
    def step(self, closure=None):
        """执行优化步骤"""
        # 将FP16梯度复制到FP32主权重
        for master_param in self.master_params:
            model_param = self.param_map[id(master_param)]
            if model_param.grad is not None:
                if master_param.grad is None:
                    master_param.grad = model_param.grad.float()
                else:
                    master_param.grad.copy_(model_param.grad)
        
        # 更新FP32主权重
        loss = self.optimizer.step(closure)
        
        # 将更新后的FP32权重复制回FP16模型
        for master_param in self.master_params:
            model_param = self.param_map[id(master_param)]
            model_param.data.copy_(master_param.data)
        
        return loss
    
    def state_dict(self):
        """获取状态字典"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.optimizer.load_state_dict(state_dict)


# 便捷的上下文管理器
@contextmanager
def autocast(precision: str = "fp16", enabled: bool = True):
    """
    自动类型转换上下文管理器
    
    Args:
        precision: 精度类型（fp16/bf16）
        enabled: 是否启用
    """
    if not enabled or not torch.cuda.is_available():
        yield
        return
    
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    
    with autocast(device_type='cuda', dtype=dtype):
        yield


# 训练循环辅助函数
def train_step_with_amp(
    model: nn.Module,
    optimizer: Optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: Callable,
    trainer: MixedPrecisionTrainer,
    gradient_clip_val: Optional[float] = None
) -> Tuple[torch.Tensor, bool]:
    """
    使用混合精度的训练步骤
    
    Args:
        model: 模型
        optimizer: 优化器
        data: 输入数据
        target: 目标标签
        criterion: 损失函数
        trainer: 混合精度训练器
        gradient_clip_val: 梯度裁剪值
        
    Returns:
        (损失值, 是否成功)
    """
    model.train()
    optimizer.zero_grad()
    
    with trainer.autocast_context():
        output = model(data)
        loss = criterion(output, target)
    
    trainer.backward(loss)
    
    # 梯度裁剪
    if gradient_clip_val is not None:
        if trainer._scaler is not None:
            trainer._scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
    
    success = trainer.step(optimizer)
    
    return loss.detach(), success


# 模型转换函数
def convert_to_fp16(model: nn.Module) -> nn.Module:
    """
    将模型转换为FP16
    
    Args:
        model: 模型
        
    Returns:
        FP16模型
    """
    return model.half()


def convert_to_bf16(model: nn.Module) -> nn.Module:
    """
    将模型转换为BF16
    
    Args:
        model: 模型
        
    Returns:
        BF16模型
    """
    return model.to(torch.bfloat16)


def convert_to_fp32(model: nn.Module) -> nn.Module:
    """
    将模型转换为FP32
    
    Args:
        model: 模型
        
    Returns:
        FP32模型
    """
    return model.float()


# 检查函数
def is_fp16_supported() -> bool:
    """检查FP16是否支持"""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 5


def is_bf16_supported() -> bool:
    """检查BF16是否支持"""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()


def get_optimal_precision() -> str:
    """
    获取最优精度类型
    
    Returns:
        最优精度类型字符串
    """
    if is_bf16_supported():
        return "bf16"
    elif is_fp16_supported():
        return "fp16"
    return "fp32"


# 工具函数
def get_precision_info() -> Dict[str, Any]:
    """获取精度支持信息"""
    return {
        "cuda_available": torch.cuda.is_available(),
        "fp16_supported": is_fp16_supported(),
        "bf16_supported": is_bf16_supported(),
        "optimal_precision": get_optimal_precision(),
        "device_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    }


def print_precision_info() -> None:
    """打印精度支持信息"""
    info = get_precision_info()
    
    print("Mixed Precision Support:")
    print(f"  CUDA Available: {info['cuda_available']}")
    print(f"  FP16 Supported: {info['fp16_supported']}")
    print(f"  BF16 Supported: {info['bf16_supported']}")
    print(f"  Optimal Precision: {info['optimal_precision']}")
    if info['device_capability']:
        print(f"  Device Capability: {info['device_capability']}")


# 兼容性函数
def enable_amp(
    model: nn.Module,
    optimizer: Optimizer,
    precision: str = "fp16",
    **kwargs
) -> Tuple[nn.Module, Optimizer, MixedPrecisionTrainer]:
    """
    启用自动混合精度
    
    Args:
        model: 模型
        optimizer: 优化器
        precision: 精度类型
        **kwargs: 其他参数
        
    Returns:
        (模型, 优化器, 训练器)
    """
    config = MixedPrecisionConfig(precision=precision, **kwargs)
    trainer = MixedPrecisionTrainer(config)
    
    model = trainer.prepare_model(model)
    optimizer = trainer.prepare_optimizer(optimizer)
    
    return model, optimizer, trainer
