"""
Gradient Checkpointing - 梯度检查点节省内存

模块路径: hardware/gpu/gradient_checkpointing.py

提供梯度检查点功能，通过以计算换内存的方式显著降低训练时的显存占用。
支持多种检查点策略和自定义检查点函数。
"""

import logging
import warnings
from typing import Optional, List, Callable, Any, Dict, Set, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """梯度检查点配置"""
    enabled: bool = True
    preserve_rng_state: bool = True
    use_reentrant: bool = True  # 使用重入检查点
    debug: bool = False
    # 检查点策略
    strategy: str = "selective"  # selective, full, none
    # 哪些层使用检查点
    checkpoint_layers: List[str] = field(default_factory=list)
    # 内存节省目标（相对于原始内存）
    memory_target: float = 0.5


class GradientCheckpointingManager:
    """
    梯度检查点管理器
    
    管理模型的梯度检查点，提供灵活的配置和监控功能。
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        初始化梯度检查点管理器
        
        Args:
            config: 检查点配置
        """
        self.config = config or CheckpointConfig()
        self._enabled = self.config.enabled
        self._checkpointed_modules: Set[str] = set()
        self._original_forwards: Dict[str, Callable] = {}
        self._memory_stats: Dict[str, Any] = {
            "checkpoint_count": 0,
            "memory_saved": 0,
            "recompute_time": 0.0
        }
    
    def enable_checkpointing(
        self,
        model: nn.Module,
        layer_types: Optional[List[type]] = None,
        layer_names: Optional[List[str]] = None
    ) -> nn.Module:
        """
        为模型启用梯度检查点
        
        Args:
            model: PyTorch模型
            layer_types: 要应用检查点的层类型列表
            layer_names: 要应用检查点的层名称列表
            
        Returns:
            修改后的模型
        """
        if not self._enabled:
            return model
        
        layer_types = layer_types or [nn.TransformerEncoderLayer, nn.TransformerDecoderLayer]
        
        for name, module in model.named_modules():
            should_checkpoint = False
            
            # 检查层类型
            if any(isinstance(module, t) for t in layer_types):
                should_checkpoint = True
            
            # 检查层名称
            if layer_names and any(ln in name for ln in layer_names):
                should_checkpoint = True
            
            # 检查配置中的层
            if self.config.checkpoint_layers and any(ln in name for ln in self.config.checkpoint_layers):
                should_checkpoint = True
            
            if should_checkpoint:
                self._apply_checkpoint_to_module(module, name)
        
        logger.info(f"Gradient checkpointing enabled for {len(self._checkpointed_modules)} modules")
        return model
    
    def _apply_checkpoint_to_module(self, module: nn.Module, name: str) -> None:
        """
        为单个模块应用梯度检查点
        
        Args:
            module: 要修改的模块
            name: 模块名称
        """
        if name in self._checkpointed_modules:
            return
        
        # 保存原始前向函数
        self._original_forwards[name] = module.forward
        
        # 包装前向函数
        original_forward = module.forward
        
        @wraps(original_forward)
        def checkpointed_forward(*args, **kwargs):
            if self.training and self._enabled:
                self._memory_stats["checkpoint_count"] += 1
                return checkpoint(
                    original_forward,
                    *args,
                    use_reentrant=self.config.use_reentrant,
                    preserve_rng_state=self.config.preserve_rng_state,
                    **kwargs
                )
            return original_forward(*args, **kwargs)
        
        module.forward = checkpointed_forward
        self._checkpointed_modules.add(name)
        
        if self.config.debug:
            logger.debug(f"Applied checkpoint to: {name}")
    
    def disable_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        禁用模型的梯度检查点
        
        Args:
            model: PyTorch模型
            
        Returns:
            恢复后的模型
        """
        for name in self._checkpointed_modules:
            module = self._get_module_by_name(model, name)
            if module and name in self._original_forwards:
                module.forward = self._original_forwards[name]
        
        self._checkpointed_modules.clear()
        self._original_forwards.clear()
        logger.info("Gradient checkpointing disabled")
        return model
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """通过名称获取模块"""
        for n, m in model.named_modules():
            if n == name:
                return m
        return None
    
    @contextmanager
    def checkpoint_context(self, enabled: bool = True):
        """
        检查点上下文管理器
        
        Args:
            enabled: 是否启用检查点
        """
        old_enabled = self._enabled
        self._enabled = enabled
        try:
            yield self
        finally:
            self._enabled = old_enabled
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检查点统计信息"""
        return self._memory_stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._memory_stats = {
            "checkpoint_count": 0,
            "memory_saved": 0,
            "recompute_time": 0.0
        }
    
    @property
    def enabled(self) -> bool:
        """检查点是否启用"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """设置检查点启用状态"""
        self._enabled = value


class CheckpointedSequential(nn.Sequential):
    """
    支持梯度检查点的Sequential容器
    
    对整个序列应用检查点。
    """
    
    def __init__(self, *args, num_segments: int = 1, **kwargs):
        """
        初始化
        
        Args:
            *args: 传递给Sequential的参数
            num_segments: 将序列分割为多少段进行检查点
            **kwargs: 其他参数
        """
        super().__init__(*args)
        self.num_segments = num_segments
        self._checkpoint_enabled = True
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training and self._checkpoint_enabled and self.num_segments > 1:
            return checkpoint_sequential(
                self,
                self.num_segments,
                input,
                use_reentrant=True
            )
        return super().forward(input)
    
    def enable_checkpoint(self):
        """启用检查点"""
        self._checkpoint_enabled = True
    
    def disable_checkpoint(self):
        """禁用检查点"""
        self._checkpoint_enabled = False


class SelectiveCheckpointing:
    """
    选择性梯度检查点
    
    根据内存使用情况动态决定是否使用检查点。
    """
    
    def __init__(
        self,
        memory_threshold: float = 0.9,
        checkpoint_ratio: float = 0.5
    ):
        """
        初始化选择性检查点
        
        Args:
            memory_threshold: 内存使用阈值
            checkpoint_ratio: 检查点应用比例
        """
        self.memory_threshold = memory_threshold
        self.checkpoint_ratio = checkpoint_ratio
        self._memory_usage_history: List[float] = []
    
    def should_checkpoint(self, module: nn.Module) -> bool:
        """
        判断是否应该对模块应用检查点
        
        Args:
            module: 要判断的模块
            
        Returns:
            是否应该应用检查点
        """
        if not torch.cuda.is_available():
            return False
        
        # 获取当前内存使用情况
        allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        self._memory_usage_history.append(allocated)
        
        # 如果内存使用超过阈值，应用检查点
        if allocated > self.memory_threshold:
            return True
        
        # 按比例随机应用检查点
        import random
        return random.random() < self.checkpoint_ratio
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        if not self._memory_usage_history:
            return {}
        
        return {
            "avg_memory_usage": sum(self._memory_usage_history) / len(self._memory_usage_history),
            "max_memory_usage": max(self._memory_usage_history),
            "min_memory_usage": min(self._memory_usage_history)
        }


class ActivationCheckpointing:
    """
    激活值检查点
    
    专门用于检查点激活值，支持更细粒度的控制。
    """
    
    def __init__(self):
        self._checkpointed_activations: Dict[str, torch.Tensor] = {}
        self._enabled = False
    
    def checkpoint_activation(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> torch.Tensor:
        """
        检查点激活值
        
        Args:
            tensor: 要检查点的张量
            name: 检查点名称
            
        Returns:
            检查点后的张量
        """
        if not self._enabled:
            return tensor
        
        # 使用detach和requires_grad来创建检查点
        checkpointed = tensor.detach()
        checkpointed.requires_grad = tensor.requires_grad
        
        self._checkpointed_activations[name] = checkpointed
        
        # 注册钩子以在反向传播时恢复梯度
        if tensor.requires_grad:
            def hook(grad):
                return grad
            checkpointed.register_hook(hook)
        
        return checkpointed
    
    def clear_checkpoints(self):
        """清除所有检查点"""
        self._checkpointed_activations.clear()
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value


def checkpoint_function(
    func: Callable,
    *args,
    preserve_rng_state: bool = True,
    use_reentrant: bool = True,
    **kwargs
) -> Any:
    """
    检查点包装函数
    
    对任意函数应用梯度检查点。
    
    Args:
        func: 要包装的函数
        args: 位置参数
        preserve_rng_state: 是否保留随机数状态
        use_reentrant: 是否使用重入检查点
        kwargs: 关键字参数
        
    Returns:
        函数返回值
    """
    return checkpoint(
        func,
        *args,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
        **kwargs
    )


class LayerwiseCheckpointing:
    """
    逐层梯度检查点
    
    为Transformer等模型的每一层单独应用检查点。
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_type: type = nn.TransformerEncoderLayer,
        every_n_layers: int = 1
    ):
        """
        初始化逐层检查点
        
        Args:
            model: 模型
            layer_type: 要检查点的层类型
            every_n_layers: 每多少层应用一次检查点
        """
        self.model = model
        self.layer_type = layer_type
        self.every_n_layers = every_n_layers
        self._layer_count = 0
        self._enabled = False
    
    def apply(self) -> nn.Module:
        """应用逐层检查点"""
        self._enabled = True
        self._layer_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_type):
                self._layer_count += 1
                if self._layer_count % self.every_n_layers == 0:
                    self._wrap_layer(module, name)
        
        logger.info(f"Applied layerwise checkpointing to {self._layer_count} layers")
        return self.model
    
    def _wrap_layer(self, layer: nn.Module, name: str):
        """包装单个层"""
        original_forward = layer.forward
        
        @wraps(original_forward)
        def checkpointed_forward(*args, **kwargs):
            if self._enabled and self.model.training:
                return checkpoint(
                    original_forward,
                    *args,
                    use_reentrant=True,
                    **kwargs
                )
            return original_forward(*args, **kwargs)
        
        layer.forward = checkpointed_forward
    
    def enable(self):
        """启用检查点"""
        self._enabled = True
    
    def disable(self):
        """禁用检查点"""
        self._enabled = False


# 便捷的装饰器
def checkpoint_wrapper(
    preserve_rng_state: bool = True,
    use_reentrant: bool = True
):
    """
    梯度检查点装饰器
    
    Args:
        preserve_rng_state: 是否保留随机数状态
        use_reentrant: 是否使用重入检查点
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return checkpoint(
                func,
                *args,
                use_reentrant=use_reentrant,
                preserve_rng_state=preserve_rng_state,
                **kwargs
            )
        return wrapper
    return decorator


# 上下文管理器
@contextmanager
def enable_gradient_checkpointing(model: nn.Module, **kwargs):
    """
    梯度检查点上下文管理器
    
    Args:
        model: 要应用检查点的模型
        **kwargs: 传递给CheckpointConfig的参数
    """
    config = CheckpointConfig(**kwargs)
    manager = GradientCheckpointingManager(config)
    
    try:
        model = manager.enable_checkpointing(model)
        yield manager
    finally:
        model = manager.disable_checkpointing(model)


# 内存估算工具
def estimate_memory_savings(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """
    估算梯度检查点的内存节省
    
    Args:
        model: 模型
        input_shape: 输入形状
        dtype: 数据类型
        
    Returns:
        内存统计字典
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # 记录初始内存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建输入
    dummy_input = torch.randn(input_shape, dtype=dtype, device="cuda")
    
    # 前向传播（无检查点）
    model.train()
    output = model(dummy_input)
    if isinstance(output, tuple):
        output = output[0]
    
    # 反向传播
    loss = output.sum()
    loss.backward()
    
    memory_without_checkpoint = torch.cuda.max_memory_allocated()
    
    # 清理
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 应用检查点
    manager = GradientCheckpointingManager()
    model = manager.enable_checkpointing(model)
    
    # 再次前向传播（有检查点）
    output = model(dummy_input)
    if isinstance(output, tuple):
        output = output[0]
    
    loss = output.sum()
    loss.backward()
    
    memory_with_checkpoint = torch.cuda.max_memory_allocated()
    
    # 恢复模型
    model = manager.disable_checkpointing(model)
    
    return {
        "memory_without_checkpoint_mb": memory_without_checkpoint / 1024**2,
        "memory_with_checkpoint_mb": memory_with_checkpoint / 1024**2,
        "memory_saved_mb": (memory_without_checkpoint - memory_with_checkpoint) / 1024**2,
        "memory_saved_percent": (memory_without_checkpoint - memory_with_checkpoint) / memory_without_checkpoint * 100
    }


def configure_checkpointing(
    model: nn.Module,
    strategy: str = "selective",
    **kwargs
) -> GradientCheckpointingManager:
    """
    配置梯度检查点
    
    Args:
        model: 模型
        strategy: 策略（selective, full, none）
        **kwargs: 其他配置参数
        
    Returns:
        配置好的管理器
    """
    config = CheckpointConfig(strategy=strategy, **kwargs)
    manager = GradientCheckpointingManager(config)
    
    if strategy != "none":
        manager.enable_checkpointing(model)
    
    return manager
