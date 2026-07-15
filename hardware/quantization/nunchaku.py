"""
Nunchaku: 4-bit/8-bit混合精度量化
动态精度切换，支持多种后端

基于混合精度量化技术，实现高效的模型压缩和推理加速
支持动态精度切换，根据层的重要性自动选择量化精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any, Union, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class NunchakuConfig:
    """Nunchaku量化配置"""
    # 量化配置
    quantization_mode: str = "mixed"  # "fp16", "int8", "int4", "mixed"
    mixed_precision_layers: Optional[List[str]] = None  # 使用高精度（8bit）的层
    
    # 4-bit配置
    int4_group_size: int = 128  # 4-bit量化的组大小
    int4_quant_type: str = "nf4"  # "nf4" 或 "fp4"
    int4_use_double_quant: bool = True  # 使用双重量化
    
    # 8-bit配置
    int8_block_size: int = 256  # 8-bit量化的块大小
    int8_threshold: float = 6.0  # 离群值阈值
    
    # 动态精度切换
    enable_dynamic_switch: bool = True  # 启用动态精度切换
    dynamic_switch_threshold: float = 0.95  # 切换阈值
    
    # 后端配置
    backend: str = "auto"  # "auto", "cuda", "cpu", "triton"
    use_marlin: bool = False  # 使用Marlin内核
    use_exllama: bool = False  # 使用ExLlama内核
    
    # 校准配置
    calibration_samples: int = 128  # 校准样本数
    calibration_batch_size: int = 1  # 校准批次大小


class QuantizationUtils:
    """量化工具函数"""
    
    @staticmethod
    def quantize_to_int8(
        tensor: torch.Tensor,
        block_size: int = 256,
        threshold: float = 6.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        8-bit量化
        
        Args:
            tensor: 输入张量
            block_size: 量化块大小
            threshold: 离群值阈值
            
        Returns:
            (量化后的张量, 缩放因子, 零点)
        """
        orig_shape = tensor.shape
        tensor = tensor.view(-1, block_size)
        
        # 计算每块的min/max
        min_val = tensor.min(dim=1, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0]
        
        # 检测离群值
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        outlier_mask = (tensor - mean).abs() > threshold * std
        
        # 计算缩放因子和零点
        scale = (max_val - min_val) / 255.0
        zero_point = -min_val / scale
        
        # 量化
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 255).to(torch.uint8)
        
        # 保留离群值的原始精度
        quantized = torch.where(outlier_mask, tensor, quantized.float())
        
        return quantized.view(orig_shape), scale.view(-1), zero_point.view(-1)
    
    @staticmethod
    def dequantize_from_int8(
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: int = 256
    ) -> torch.Tensor:
        """
        8-bit反量化
        
        Args:
            quantized: 量化后的张量
            scale: 缩放因子
            zero_point: 零点
            block_size: 块大小
            
        Returns:
            反量化后的张量
        """
        orig_shape = quantized.shape
        quantized = quantized.view(-1, block_size)
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        
        # 反量化
        dequantized = (quantized.float() - zero_point) * scale
        
        return dequantized.view(orig_shape)
    
    @staticmethod
    def quantize_to_int4(
        tensor: torch.Tensor,
        group_size: int = 128,
        quant_type: str = "nf4"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        4-bit量化（支持NF4和FP4）
        
        Args:
            tensor: 输入张量
            group_size: 量化组大小
            quant_type: 量化类型 "nf4" 或 "fp4"
            
        Returns:
            (量化后的张量, 缩放因子, 零点)
        """
        orig_shape = tensor.shape
        tensor = tensor.view(-1, group_size)
        
        # 计算每组的min/max
        min_val = tensor.min(dim=1, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0]
        
        # 计算缩放因子和零点
        if quant_type == "nf4":
            # Normal Float 4
            scale = (max_val - min_val) / 15.0
            zero_point = -min_val / scale
        else:
            # FP4
            scale = (max_val - min_val) / 15.0
            zero_point = -min_val / scale
        
        # 量化到4-bit（存储为8-bit，但只使用前4位）
        quantized = torch.round(tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
        
        # 打包两个4-bit值到一个字节
        packed = torch.zeros(tensor.shape[0], tensor.shape[1] // 2, dtype=torch.uint8, device=tensor.device)
        packed = (quantized[:, 0::2] << 4) | quantized[:, 1::2]
        
        return packed, scale.view(-1), zero_point.view(-1)
    
    @staticmethod
    def dequantize_from_int4(
        packed: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        group_size: int = 128,
        quant_type: str = "nf4"
    ) -> torch.Tensor:
        """
        4-bit反量化
        
        Args:
            packed: 打包的4-bit张量
            scale: 缩放因子
            zero_point: 零点
            group_size: 组大小
            quant_type: 量化类型
            
        Returns:
            反量化后的张量
        """
        # 解包
        unpacked = torch.zeros(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8, device=packed.device)
        unpacked[:, 0::2] = (packed >> 4) & 0x0F
        unpacked[:, 1::2] = packed & 0x0F
        
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        
        # 反量化
        dequantized = (unpacked.float() - zero_point) * scale
        
        return dequantized


class NunchakuLinear(nn.Module):
    """
    Nunchaku量化线性层
    支持4-bit/8-bit混合精度
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantization_mode: str = "int4",
        config: Optional[NunchakuConfig] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_mode = quantization_mode
        self.config = config or NunchakuConfig()
        
        # 注册量化参数
        if quantization_mode == "int8":
            self.register_buffer('weight_scale', torch.zeros(out_features))
            self.register_buffer('weight_zero_point', torch.zeros(out_features))
            self.register_buffer('quantized_weight', torch.zeros(out_features, in_features, dtype=torch.uint8))
        elif quantization_mode == "int4":
            self.register_buffer('weight_scale', torch.zeros(out_features))
            self.register_buffer('weight_zero_point', torch.zeros(out_features))
            # 4-bit权重打包存储
            self.register_buffer('quantized_weight', torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 动态精度切换状态
        self.current_precision = quantization_mode
        self.activation_stats = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        # 动态精度切换
        if self.config.enable_dynamic_switch and self.training:
            self._update_activation_stats(x)
            self._maybe_switch_precision()
        
        # 根据当前精度选择计算方式
        if self.current_precision == "int8":
            return self._forward_int8(x)
        elif self.current_precision == "int4":
            return self._forward_int4(x)
        else:
            return self._forward_fp16(x)
    
    def _forward_fp16(self, x: torch.Tensor) -> torch.Tensor:
        """FP16前向传播"""
        return F.linear(x, self.weight, self.bias)
    
    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        """8-bit量化前向传播"""
        # 反量化权重
        weight = QuantizationUtils.dequantize_from_int8(
            self.quantized_weight,
            self.weight_scale,
            self.weight_zero_point,
            self.config.int8_block_size
        )
        
        # 量化激活（可选）
        if self.config.enable_dynamic_switch:
            x_quantized, x_scale, x_zero = QuantizationUtils.quantize_to_int8(
                x, self.config.int8_block_size
            )
            x = QuantizationUtils.dequantize_from_int8(
                x_quantized, x_scale, x_zero, self.config.int8_block_size
            )
        
        return F.linear(x, weight, self.bias)
    
    def _forward_int4(self, x: torch.Tensor) -> torch.Tensor:
        """4-bit量化前向传播"""
        # 反量化权重
        weight = QuantizationUtils.dequantize_from_int4(
            self.quantized_weight,
            self.weight_scale,
            self.weight_zero_point,
            self.config.int4_group_size,
            self.config.int4_quant_type
        )
        
        return F.linear(x, weight, self.bias)
    
    def _update_activation_stats(self, x: torch.Tensor):
        """更新激活统计信息"""
        with torch.no_grad():
            stats = {
                'mean': x.mean().item(),
                'std': x.std().item(),
                'max': x.max().item(),
                'min': x.min().item()
            }
            self.activation_stats.append(stats)
            
            # 限制统计历史长度
            if len(self.activation_stats) > 100:
                self.activation_stats.pop(0)
    
    def _maybe_switch_precision(self):
        """根据统计信息可能切换精度"""
        if len(self.activation_stats) < 10:
            return
        
        # 计算最近激活的方差
        recent_stats = self.activation_stats[-10:]
        stds = [s['std'] for s in recent_stats]
        avg_std = np.mean(stds)
        
        # 如果方差大，切换到高精度
        if avg_std > self.config.dynamic_switch_threshold and self.current_precision == "int4":
            self.current_precision = "int8"
        elif avg_std < self.config.dynamic_switch_threshold * 0.5 and self.current_precision == "int8":
            self.current_precision = "int4"
    
    def quantize_weight(self, weight: torch.Tensor):
        """量化权重"""
        if self.quantization_mode == "int8":
            quantized, scale, zero_point = QuantizationUtils.quantize_to_int8(
                weight, self.config.int8_block_size, self.config.int8_threshold
            )
            self.quantized_weight.copy_(quantized)
            self.weight_scale.copy_(scale)
            self.weight_zero_point.copy_(zero_point)
        elif self.quantization_mode == "int4":
            quantized, scale, zero_point = QuantizationUtils.quantize_to_int4(
                weight, self.config.int4_group_size, self.config.int4_quant_type
            )
            self.quantized_weight.copy_(quantized)
            self.weight_scale.copy_(scale)
            self.weight_zero_point.copy_(zero_point)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quantization_mode: str = "int4",
        config: Optional[NunchakuConfig] = None
    ) -> 'NunchakuLinear':
        """
        从普通线性层创建Nunchaku线性层
        
        Args:
            linear: 普通线性层
            quantization_mode: 量化模式
            config: 配置
            
        Returns:
            Nunchaku线性层
        """
        nunchaku_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            quantization_mode=quantization_mode,
            config=config
        )
        
        # 量化权重
        nunchaku_linear.quantize_weight(linear.weight.data)
        
        if linear.bias is not None:
            nunchaku_linear.bias.data.copy_(linear.bias.data)
        
        return nunchaku_linear


class NunchakuModel(nn.Module):
    """
    Nunchaku量化模型包装器
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: NunchakuConfig
    ):
        super().__init__()
        self.model = model
        self.config = config
        
        # 应用量化
        self._apply_quantization()
    
    def _apply_quantization(self):
        """应用量化到模型"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 决定量化模式
                quant_mode = self._get_quantization_mode_for_layer(name)
                
                # 替换为Nunchaku线性层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                
                nunchaku_linear = NunchakuLinear.from_linear(
                    module, quant_mode, self.config
                )
                setattr(parent, child_name, nunchaku_linear)
    
    def _get_quantization_mode_for_layer(self, layer_name: str) -> str:
        """
        获取层的量化模式
        
        Args:
            layer_name: 层名称
            
        Returns:
            量化模式
        """
        if self.config.quantization_mode != "mixed":
            return self.config.quantization_mode
        
        # 混合精度：某些层使用8-bit，其他使用4-bit
        if self.config.mixed_precision_layers:
            for pattern in self.config.mixed_precision_layers:
                if pattern in layer_name:
                    return "int8"
        
        # 默认使用4-bit
        return "int4"
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.model(*args, **kwargs)
    
    def save_quantized(self, save_path: str):
        """保存量化模型"""
        state_dict = {}
        config_dict = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, NunchakuLinear):
                state_dict[name] = {
                    'quantized_weight': module.quantized_weight,
                    'weight_scale': module.weight_scale,
                    'weight_zero_point': module.weight_zero_point,
                    'bias': module.bias if module.bias is not None else None,
                    'quantization_mode': module.quantization_mode
                }
        
        config_dict = {
            'quantization_mode': self.config.quantization_mode,
            'int4_group_size': self.config.int4_group_size,
            'int4_quant_type': self.config.int4_quant_type,
            'int8_block_size': self.config.int8_block_size
        }
        
        torch.save({
            'state_dict': state_dict,
            'config': config_dict
        }, save_path)
    
    def load_quantized(self, load_path: str):
        """加载量化模型"""
        checkpoint = torch.load(load_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        for name, params in state_dict.items():
            module = self.model.get_submodule(name)
            if isinstance(module, NunchakuLinear):
                module.quantized_weight.copy_(params['quantized_weight'])
                module.weight_scale.copy_(params['weight_scale'])
                module.weight_zero_point.copy_(params['weight_zero_point'])
                if params['bias'] is not None and module.bias is not None:
                    module.bias.copy_(params['bias'])


class NunchakuQuantizer:
    """
    Nunchaku量化器
    用于模型量化和校准
    """
    
    def __init__(self, config: NunchakuConfig):
        self.config = config
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.utils.data.DataLoader] = None
    ) -> NunchakuModel:
        """
        量化模型
        
        Args:
            model: 原始模型
            calibration_data: 校准数据
            
        Returns:
            量化后的模型
        """
        # 如果有校准数据，先进行校准
        if calibration_data is not None:
            self._calibrate_model(model, calibration_data)
        
        # 创建量化模型
        quantized_model = NunchakuModel(model, self.config)
        
        return quantized_model
    
    def _calibrate_model(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ):
        """
        校准模型以确定最佳量化参数
        
        Args:
            model: 模型
            calibration_data: 校准数据
        """
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_data):
                if batch_idx >= self.config.calibration_samples:
                    break
                
                # 前向传播以收集统计信息
                if isinstance(batch, dict):
                    _ = model(**batch)
                else:
                    _ = model(batch)
    
    def evaluate_quantization_quality(
        self,
        original_model: nn.Module,
        quantized_model: NunchakuModel,
        test_data: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        评估量化质量
        
        Args:
            original_model: 原始模型
            quantized_model: 量化模型
            test_data: 测试数据
            
        Returns:
            评估指标
        """
        original_model.eval()
        quantized_model.eval()
        
        total_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_data:
                # 原始模型输出
                if isinstance(batch, dict):
                    orig_output = original_model(**batch)
                    quant_output = quantized_model(**batch)
                else:
                    orig_output = original_model(batch)
                    quant_output = quantized_model(batch)
                
                # 计算MSE
                if isinstance(orig_output, tuple):
                    orig_output = orig_output[0]
                if isinstance(quant_output, tuple):
                    quant_output = quant_output[0]
                
                mse = F.mse_loss(orig_output, quant_output).item()
                total_mse += mse
                total_samples += 1
        
        avg_mse = total_mse / total_samples if total_samples > 0 else 0.0
        
        return {
            'mse': avg_mse,
            'psnr': -10 * np.log10(avg_mse) if avg_mse > 0 else float('inf'),
            'samples_evaluated': total_samples
        }


def quantize_model_with_nunchaku(
    model: nn.Module,
    quantization_mode: str = "mixed",
    mixed_precision_layers: Optional[List[str]] = None,
    calibration_data: Optional[torch.utils.data.DataLoader] = None
) -> NunchakuModel:
    """
    使用Nunchaku量化模型的便捷函数
    
    Args:
        model: 原始模型
        quantization_mode: 量化模式
        mixed_precision_layers: 混合精度层列表
        calibration_data: 校准数据
        
    Returns:
        量化后的模型
    """
    config = NunchakuConfig(
        quantization_mode=quantization_mode,
        mixed_precision_layers=mixed_precision_layers
    )
    
    quantizer = NunchakuQuantizer(config)
    return quantizer.quantize_model(model, calibration_data)


def get_model_size_info(model: nn.Module) -> Dict[str, float]:
    """
    获取模型大小信息
    
    Args:
        model: 模型
        
    Returns:
        大小信息
    """
    total_params = 0
    total_bytes = 0
    
    for param in model.parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()
    
    # 计算缓冲区大小
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    
    return {
        'total_params': total_params,
        'total_size_mb': total_bytes / (1024 ** 2),
        'total_size_gb': total_bytes / (1024 ** 3)
    }


def compare_model_sizes(
    original_model: nn.Module,
    quantized_model: NunchakuModel
) -> Dict[str, Any]:
    """
    比较原始模型和量化模型的大小
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        
    Returns:
        比较结果
    """
    orig_info = get_model_size_info(original_model)
    quant_info = get_model_size_info(quantized_model)
    
    compression_ratio = orig_info['total_size_mb'] / quant_info['total_size_mb']
    
    return {
        'original': orig_info,
        'quantized': quant_info,
        'compression_ratio': compression_ratio,
        'space_saved_mb': orig_info['total_size_mb'] - quant_info['total_size_mb'],
        'space_saved_percentage': (1 - quant_info['total_size_mb'] / orig_info['total_size_mb']) * 100
    }
