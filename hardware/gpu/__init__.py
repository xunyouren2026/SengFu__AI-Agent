"""
GPU Optimization Module - AGI Unified Framework

模块路径: hardware/gpu/__init__.py

该模块提供GPU加速优化功能，包括:
- CUDA流和事件管理 (cuda_optimizer)
- Flash Attention高效注意力计算 (flash_attention)
- 梯度检查点节省内存 (gradient_checkpointing)
- 算子融合优化性能 (kernel_fusion)
- GPU内存池管理 (memory_pool)
- FP16/BF16混合精度训练 (mixed_precision)
- DDP/DeepSpeed多GPU训练 (multi_gpu)
- Tensor Core优化 (tensor_cores)
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.warning("PyTorch not available. GPU optimization features disabled.")


@dataclass
class GPUConfig:
    """GPU配置数据类"""
    device_id: int = 0
    num_gpus: int = 1
    use_cuda: bool = True
    use_tensor_cores: bool = True
    mixed_precision: str = "fp16"  # fp16, bf16, fp32
    gradient_checkpointing: bool = False
    memory_efficient: bool = True
    flash_attention: bool = True
    kernel_fusion: bool = True


class GPUManager:
    """
    GPU管理器 - 统一管理所有GPU优化功能
    
    提供统一的接口来配置和使用各种GPU优化技术。
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        """
        初始化GPU管理器
        
        Args:
            config: GPU配置对象，如果为None则使用默认配置
        """
        self.config = config or GPUConfig()
        self._initialized = False
        self._optimizers: Dict[str, Any] = {}
        
        # 检查CUDA可用性
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU optimization")
        if not CUDA_AVAILABLE:
            logger.warning("CUDA not available. GPU features will be limited.")
    
    def initialize(self) -> None:
        """初始化所有GPU优化组件"""
        if self._initialized:
            return
        
        # 设置默认CUDA设备
        if CUDA_AVAILABLE:
            torch.cuda.set_device(self.config.device_id)
            logger.info(f"CUDA device set to: {self.config.device_id}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(self.config.device_id)}")
        
        self._initialized = True
    
    def get_device(self) -> torch.device:
        """获取当前计算设备"""
        if CUDA_AVAILABLE and self.config.use_cuda:
            return torch.device(f"cuda:{self.config.device_id}")
        return torch.device("cpu")
    
    def get_cuda_optimizer(self):
        """获取CUDA优化器实例"""
        if "cuda_optimizer" not in self._optimizers:
            from .cuda_optimizer import CUDAOptimizer
            self._optimizers["cuda_optimizer"] = CUDAOptimizer(
                device_id=self.config.device_id
            )
        return self._optimizers["cuda_optimizer"]
    
    def get_flash_attention(self):
        """获取Flash Attention实例"""
        if "flash_attention" not in self._optimizers:
            from .flash_attention import FlashAttention
            self._optimizers["flash_attention"] = FlashAttention(
                use_flash=self.config.flash_attention
            )
        return self._optimizers["flash_attention"]
    
    def get_gradient_checkpointing(self):
        """获取梯度检查点管理器"""
        if "gradient_checkpointing" not in self._optimizers:
            from .gradient_checkpointing import GradientCheckpointingManager
            self._optimizers["gradient_checkpointing"] = GradientCheckpointingManager(
                enabled=self.config.gradient_checkpointing
            )
        return self._optimizers["gradient_checkpointing"]
    
    def get_kernel_fusion(self):
        """获取算子融合优化器"""
        if "kernel_fusion" not in self._optimizers:
            from .kernel_fusion import KernelFusionOptimizer
            self._optimizers["kernel_fusion"] = KernelFusionOptimizer(
                enabled=self.config.kernel_fusion
            )
        return self._optimizers["kernel_fusion"]
    
    def get_memory_pool(self):
        """获取内存池管理器"""
        if "memory_pool" not in self._optimizers:
            from .memory_pool import GPUMemoryPool
            self._optimizers["memory_pool"] = GPUMemoryPool(
                device_id=self.config.device_id
            )
        return self._optimizers["memory_pool"]
    
    def get_mixed_precision(self):
        """获取混合精度训练器"""
        if "mixed_precision" not in self._optimizers:
            from .mixed_precision import MixedPrecisionTrainer
            self._optimizers["mixed_precision"] = MixedPrecisionTrainer(
                precision=self.config.mixed_precision
            )
        return self._optimizers["mixed_precision"]
    
    def get_multi_gpu(self):
        """获取多GPU训练器"""
        if "multi_gpu" not in self._optimizers:
            from .multi_gpu import MultiGPUTrainer
            self._optimizers["multi_gpu"] = MultiGPUTrainer(
                num_gpus=self.config.num_gpus
            )
        return self._optimizers["multi_gpu"]
    
    def get_tensor_cores(self):
        """获取Tensor Core优化器"""
        if "tensor_cores" not in self._optimizers:
            from .tensor_cores import TensorCoreOptimizer
            self._optimizers["tensor_cores"] = TensorCoreOptimizer(
                enabled=self.config.use_tensor_cores
            )
        return self._optimizers["tensor_cores"]
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        对模型应用所有启用的优化
        
        Args:
            model: PyTorch模型
            
        Returns:
            优化后的模型
        """
        if not self._initialized:
            self.initialize()
        
        # 应用梯度检查点
        if self.config.gradient_checkpointing:
            gc = self.get_gradient_checkpointing()
            model = gc.enable_checkpointing(model)
        
        # 应用混合精度
        if self.config.mixed_precision in ["fp16", "bf16"]:
            mp = self.get_mixed_precision()
            model = mp.prepare_model(model)
        
        # 应用Tensor Core优化
        if self.config.use_tensor_cores:
            tc = self.get_tensor_cores()
            model = tc.optimize_model(model)
        
        # 应用算子融合
        if self.config.kernel_fusion:
            kf = self.get_kernel_fusion()
            model = kf.optimize_model(model)
        
        logger.info("Model optimization completed")
        return model
    
    def cleanup(self) -> None:
        """清理GPU资源"""
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
        self._optimizers.clear()
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """检查管理器是否已初始化"""
        return self._initialized
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """获取GPU信息"""
        if not CUDA_AVAILABLE:
            return {"cuda_available": False}
        
        info = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "num_gpus": torch.cuda.device_count(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor
            })
        
        return info


# 便捷的模块级函数
def create_gpu_manager(config: Optional[GPUConfig] = None) -> GPUManager:
    """创建GPU管理器实例"""
    return GPUManager(config)


def get_default_device() -> torch.device:
    """获取默认计算设备"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def synchronize_device(device: Optional[torch.device] = None) -> None:
    """同步CUDA设备"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        if device is None or device.type == "cuda":
            torch.cuda.synchronize(device)


# 版本信息
__version__ = "1.0.0"
__all__ = [
    "GPUConfig",
    "GPUManager",
    "create_gpu_manager",
    "get_default_device",
    "synchronize_device",
    "TORCH_AVAILABLE",
    "CUDA_AVAILABLE"
]
