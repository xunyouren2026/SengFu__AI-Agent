"""
Tensor Cores - Tensor Core优化

模块路径: hardware/gpu/tensor_cores.py

提供Tensor Core优化功能，包括:
- 自动利用Tensor Core进行矩阵乘法加速
- TF32、FP16、BF16格式支持
- 内存布局优化
- 性能分析和调优工具
"""

import logging
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TensorCoreConfig:
    """Tensor Core配置"""
    enabled: bool = True
    allow_tf32: bool = True  # 启用TF32
    matmul_precision: str = "high"  # high, highest
    use_fp16: bool = True
    use_bf16: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


class TensorCoreOptimizer:
    """
    Tensor Core优化器
    
    自动配置PyTorch以利用Tensor Core进行加速计算。
    """
    
    def __init__(self, config: Optional[TensorCoreConfig] = None, enabled: bool = True):
        """
        初始化Tensor Core优化器
        
        Args:
            config: Tensor Core配置
            enabled: 是否启用
        """
        if config is None:
            config = TensorCoreConfig()
        self.config = config
        self.config.enabled = enabled and self._check_tensor_core_support()
        
        self._original_settings: Dict[str, Any] = {}
        self._applied = False
    
    def _check_tensor_core_support(self) -> bool:
        """检查Tensor Core支持"""
        if not torch.cuda.is_available():
            return False
        
        capability = torch.cuda.get_device_capability()
        # Tensor Core需要计算能力 >= 7.0 (Volta及更高)
        return capability[0] >= 7
    
    def apply(self) -> None:
        """应用Tensor Core优化设置"""
        if not self.config.enabled or self._applied:
            return
        
        # 保存原始设置
        self._original_settings['allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
        self._original_settings['allow_fp16_reduced_precision_reduction'] = \
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        self._original_settings['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        self._original_settings['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        
        # 应用新设置
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for Tensor Cores")
        
        # 设置矩阵乘法精度
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision(self.config.matmul_precision)
            logger.info(f"FP32 matmul precision set to: {self.config.matmul_precision}")
        
        # 设置cuDNN
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
        
        self._applied = True
        logger.info("Tensor Core optimizations applied")
    
    def restore(self) -> None:
        """恢复原始设置"""
        if not self._applied:
            return
        
        if 'allow_tf32' in self._original_settings:
            torch.backends.cuda.matmul.allow_tf32 = self._original_settings['allow_tf32']
            torch.backends.cudnn.allow_tf32 = self._original_settings['allow_tf32']
        
        if 'cudnn_benchmark' in self._original_settings:
            torch.backends.cudnn.benchmark = self._original_settings['cudnn_benchmark']
        
        if 'cudnn_deterministic' in self._original_settings:
            torch.backends.cudnn.deterministic = self._original_settings['cudnn_deterministic']
        
        self._applied = False
        logger.info("Tensor Core settings restored")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        优化模型以利用Tensor Core
        
        Args:
            model: PyTorch模型
            
        Returns:
            优化后的模型
        """
        if not self.config.enabled:
            return model
        
        self.apply()
        
        # 确保模型参数对齐到8的倍数（Tensor Core要求）
        self._ensure_alignment(model)
        
        # 编译模型（如果可用）
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        return model
    
    def _ensure_alignment(self, model: nn.Module) -> None:
        """
        确保模型参数对齐到8的倍数
        
        Tensor Core要求输入维度是8的倍数以获得最佳性能。
        """
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                # 检查最后两个维度
                shape = list(param.shape)
                needs_padding = False
                
                for i in range(-2, 0):
                    if shape[i] % 8 != 0:
                        needs_padding = True
                        break
                
                if needs_padding:
                    logger.debug(f"Parameter {name} may benefit from padding to multiples of 8")
    
    @contextmanager
    def optimized_context(self):
        """Tensor Core优化上下文管理器"""
        self.apply()
        try:
            yield self
        finally:
            self.restore()
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """获取设备Tensor Core信息"""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        capability = torch.cuda.get_device_capability()
        has_tensor_cores = capability[0] >= 7
        
        info = {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": capability,
            "has_tensor_cores": has_tensor_cores,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32_enabled": torch.backends.cudnn.allow_tf32,
        }
        
        # 获取TFLOPS估算
        if has_tensor_cores:
            info["tensor_core_generation"] = TensorCoreOptimizer._get_tensor_core_generation(capability)
        
        return info
    
    @staticmethod
    def _get_tensor_core_generation(capability: Tuple[int, int]) -> str:
        """获取Tensor Core代数"""
        major, minor = capability
        if major == 7:
            return "1st Gen (Volta)"
        elif major == 8:
            return "3rd Gen (Ampere)"
        elif major == 9:
            return "4th Gen (Hopper)"
        elif major >= 10:
            return "Future Gen"
        return "Unknown"


class TensorCoreLinear(nn.Module):
    """
    Tensor Core优化的线性层
    
    确保输入输出维度对齐到8的倍数以充分利用Tensor Core。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # 对齐到8的倍数
        self._in_features_aligned = self._align_to_8(in_features)
        self._out_features_aligned = self._align_to_8(out_features)
        self._in_features = in_features
        self._out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(
            self._out_features_aligned,
            self._in_features_aligned,
            device=device,
            dtype=dtype
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self._out_features_aligned, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def _align_to_8(self, n: int) -> int:
        """对齐到8的倍数"""
        return ((n + 7) // 8) * 8
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 如果需要，对输入进行填充
        if input.shape[-1] != self._in_features_aligned:
            padding_size = self._in_features_aligned - input.shape[-1]
            input = F.pad(input, (0, padding_size))
        
        output = F.linear(input, self.weight, self.bias)
        
        # 裁剪输出到原始大小
        if output.shape[-1] != self._out_features:
            output = output[..., :self._out_features]
        
        return output


class TensorCoreConv2d(nn.Module):
    """
    Tensor Core优化的卷积层
    
    确保通道数对齐到8的倍数。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # 对齐通道数到8的倍数
        self._in_channels_aligned = self._align_to_8(in_channels)
        self._out_channels_aligned = self._align_to_8(out_channels)
        self._in_channels = in_channels
        self._out_channels = out_channels
        
        self.conv = nn.Conv2d(
            self._in_channels_aligned,
            self._out_channels_aligned,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
    
    def _align_to_8(self, n: int) -> int:
        """对齐到8的倍数"""
        return ((n + 7) // 8) * 8
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 填充输入通道
        if input.shape[1] != self._in_channels_aligned:
            padding_size = self._in_channels_aligned - input.shape[1]
            input = F.pad(input, (0, 0, 0, 0, 0, padding_size))
        
        output = self.conv(input)
        
        # 裁剪输出通道
        if output.shape[1] != self._out_channels:
            output = output[:, :self._out_channels]
        
        return output


def enable_tf32() -> None:
    """启用TF32"""
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, cannot enable TF32")
        return
    
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        warnings.warn(f"TF32 requires compute capability >= 8.0, current: {capability}")
        return
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 enabled")


def disable_tf32() -> None:
    """禁用TF32"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logger.info("TF32 disabled")


def set_matmul_precision(precision: str) -> None:
    """
    设置矩阵乘法精度
    
    Args:
        precision: 'high' 或 'highest'
    """
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(precision)
        logger.info(f"FP32 matmul precision set to: {precision}")


@contextmanager
def tf32_context(enabled: bool = True):
    """
    TF32上下文管理器
    
    Args:
        enabled: 是否启用TF32
    """
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


# 性能基准测试
def benchmark_matmul(
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    dtype: torch.dtype = torch.float32,
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    基准测试矩阵乘法
    
    Args:
        m, n, k: 矩阵维度
        dtype: 数据类型
        iterations: 迭代次数
        warmup: 预热次数
        
    Returns:
        性能统计
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device("cuda")
    
    # 创建矩阵
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    
    # 预热
    for _ in range(warmup):
        _ = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    
    # 测试
    import time
    start = time.time()
    
    for _ in range(iterations):
        _ = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算TFLOPS
    flops = 2 * m * n * k * iterations
    tflops = flops / elapsed / 1e12
    
    return {
        "total_time_ms": elapsed * 1000,
        "avg_time_ms": elapsed * 1000 / iterations,
        "tflops": tflops,
        "dtype": str(dtype)
    }


def compare_precision_performance(
    m: int = 4096,
    n: int = 4096,
    k: int = 4096
) -> Dict[str, Dict[str, float]]:
    """
    比较不同精度的性能
    
    Args:
        m, n, k: 矩阵维度
        
    Returns:
        各精度的性能比较
    """
    results = {}
    
    # FP32
    results["fp32"] = benchmark_matmul(m, n, k, torch.float32)
    
    # TF32 (如果可用)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        with tf32_context(True):
            results["tf32"] = benchmark_matmul(m, n, k, torch.float32)
    
    # FP16
    if torch.cuda.is_available():
        results["fp16"] = benchmark_matmul(m, n, k, torch.float16)
    
    # BF16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        results["bf16"] = benchmark_matmul(m, n, k, torch.bfloat16)
    
    return results


def print_tensor_core_info() -> None:
    """打印Tensor Core信息"""
    info = TensorCoreOptimizer.get_device_info()
    
    print("Tensor Core Information:")
    print(f"  CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"  Device: {info['device_name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Has Tensor Cores: {info['has_tensor_cores']}")
        
        if info['has_tensor_cores']:
            print(f"  Tensor Core Generation: {info['tensor_core_generation']}")
        
        print(f"  TF32 Enabled: {info['tf32_enabled']}")
        print(f"  cuDNN TF32 Enabled: {info['cudnn_tf32_enabled']}")


def optimize_for_tensor_cores(model: nn.Module) -> nn.Module:
    """
    为Tensor Core优化模型
    
    Args:
        model: 模型
        
    Returns:
        优化后的模型
    """
    optimizer = TensorCoreOptimizer()
    return optimizer.optimize_model(model)


# 检查函数
def has_tensor_cores() -> bool:
    """检查是否有Tensor Core支持"""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 7


def get_optimal_dtype() -> torch.dtype:
    """
    获取最优数据类型
    
    Returns:
        最优的torch.dtype
    """
    if not torch.cuda.is_available():
        return torch.float32
    
    capability = torch.cuda.get_device_capability()
    
    if capability[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif capability[0] >= 7:
        return torch.float16
    
    return torch.float32
