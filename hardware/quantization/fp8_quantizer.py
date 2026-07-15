"""
FP8 (8-bit Floating Point) 量化实现

模块路径: hardware/quantization/fp8_quantizer.py

支持NVIDIA H100 Hopper架构的FP8格式:
- E4M3: 1位符号, 4位指数, 3位尾数 (动态范围更大)
- E5M2: 1位符号, 5位指数, 2位尾数 (精度更高)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json


@dataclass
class FP8Config:
    """FP8量化配置"""
    format: str = "e4m3"  # "e4m3" or "e5m2"
    per_tensor_scale: bool = False
    per_channel_scale: bool = True
    compute_dtype: torch.dtype = torch.float16
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"  # "max" or "most_recent"
    device: str = "cuda"


class FP8Tensor:
    """FP8张量表示"""

    # E4M3格式常量 (FP8 with 4-bit exponent, 3-bit mantissa)
    E4M3_MAX = 448.0
    E4M3_MIN = -448.0
    E4M3_EXP_BIAS = 7

    # E5M2格式常量 (FP8 with 5-bit exponent, 2-bit mantissa)
    E5M2_MAX = 57344.0
    E5M2_MIN = -57344.0
    E5M2_EXP_BIAS = 15

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        format: str = "e4m3",
        device: str = "cuda"
    ):
        self.data = data  # uint8存储
        self.scale = scale
        self.format = format
        self.device = device

        if format == "e4m3":
            self.max_val = self.E4M3_MAX
            self.min_val = self.E4M3_MIN
        else:
            self.max_val = self.E5M2_MAX
            self.min_val = self.E5M2_MIN

    @classmethod
    def from_float(
        cls,
        x: torch.Tensor,
        format: str = "e4m3",
        scale: Optional[torch.Tensor] = None,
        per_channel: bool = False
    ) -> "FP8Tensor":
        """从浮点张量创建FP8张量"""
        if scale is None:
            scale = cls.compute_scale(x, format, per_channel)

        # 缩放
        x_scaled = x / scale

        # 钳制到FP8范围
        max_val = cls.E4M3_MAX if format == "e4m3" else cls.E5M2_MAX
        min_val = cls.E4M3_MIN if format == "e4m3" else cls.E5M2_MIN
        x_scaled = torch.clamp(x_scaled, min_val, max_val)

        # 量化到FP8
        data = cls.float_to_fp8(x_scaled, format)

        return cls(data, scale, format, x.device.type)

    def to_float(self) -> torch.Tensor:
        """反量化为浮点"""
        x_normalized = self.fp8_to_float(self.data, self.format)
        return x_normalized * self.scale

    @staticmethod
    def compute_scale(
        x: torch.Tensor,
        format: str = "e4m3",
        per_channel: bool = False
    ) -> torch.Tensor:
        """计算量化缩放因子"""
        max_val = FP8Tensor.E4M3_MAX if format == "e4m3" else FP8Tensor.E5M2_MAX

        if per_channel and x.dim() >= 2:
            # 按输出通道计算缩放
            amax = x.abs().amax(dim=list(range(1, x.dim())), keepdim=True)
        else:
            amax = x.abs().max()

        # 缩放因子 = amax / FP8最大值
        scale = amax / max_val
        scale = torch.clamp(scale, min=1e-12)

        return scale

    @staticmethod
    def float_to_fp8(x: torch.Tensor, format: str = "e4m3") -> torch.Tensor:
        """
        将浮点数量化为FP8 (uint8存储)

        模拟FP8量化过程，实际硬件支持时可直接使用
        """
        if format == "e4m3":
            return FP8Tensor._float_to_e4m3(x)
        else:
            return FP8Tensor._float_to_e5m2(x)

    @staticmethod
    def fp8_to_float(x: torch.Tensor, format: str = "e4m3") -> torch.Tensor:
        """将FP8反量化为浮点"""
        if format == "e4m3":
            return FP8Tensor._e4m3_to_float(x)
        else:
            return FP8Tensor._e5m2_to_float(x)

    @staticmethod
    def _float_to_e4m3(x: torch.Tensor) -> torch.Tensor:
        """量化到E4M3格式"""
        # 符号位
        sign = (x < 0).to(torch.uint8)
        x_abs = x.abs()

        # 指数和尾数计算 (简化版)
        # 实际实现应该使用查找表或硬件指令
        log2_val = torch.log2(x_abs + 1e-12)
        exp = torch.clamp(torch.floor(log2_val) + FP8Tensor.E4M3_EXP_BIAS, 0, 15)

        # 计算尾数
        mantissa_val = x_abs / (2.0 ** (exp.float() - FP8Tensor.E4M3_EXP_BIAS)) - 1.0
        mantissa = torch.clamp(torch.round(mantissa_val * 8), 0, 7).to(torch.uint8)

        # 组合: 符号(1) + 指数(4) + 尾数(3)
        fp8_val = (sign << 7) | (exp.to(torch.uint8) << 3) | mantissa

        return fp8_val

    @staticmethod
    def _e4m3_to_float(x: torch.Tensor) -> torch.Tensor:
        """从E4M3格式反量化"""
        x = x.to(torch.uint8)

        # 提取符号、指数、尾数
        sign = (x >> 7) & 0x1
        exp = ((x >> 3) & 0xF).float()
        mantissa = (x & 0x7).float()

        # 计算值
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - FP8Tensor.E4M3_EXP_BIAS))

        # 应用符号
        sign_f = sign.float() * -2.0 + 1.0  # 0->1, 1->-1
        value = value * sign_f

        return value

    @staticmethod
    def _float_to_e5m2(x: torch.Tensor) -> torch.Tensor:
        """量化到E5M2格式"""
        sign = (x < 0).to(torch.uint8)
        x_abs = x.abs()

        log2_val = torch.log2(x_abs + 1e-12)
        exp = torch.clamp(torch.floor(log2_val) + FP8Tensor.E5M2_EXP_BIAS, 0, 31)

        mantissa_val = x_abs / (2.0 ** (exp.float() - FP8Tensor.E5M2_EXP_BIAS)) - 1.0
        mantissa = torch.clamp(torch.round(mantissa_val * 4), 0, 3).to(torch.uint8)

        fp8_val = (sign << 7) | (exp.to(torch.uint8) << 2) | mantissa

        return fp8_val

    @staticmethod
    def _e5m2_to_float(x: torch.Tensor) -> torch.Tensor:
        """从E5M2格式反量化"""
        x = x.to(torch.uint8)

        sign = (x >> 7) & 0x1
        exp = ((x >> 2) & 0x1F).float()
        mantissa = (x & 0x3).float()

        value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - FP8Tensor.E5M2_EXP_BIAS))

        sign_f = sign.float() * -2.0 + 1.0
        value = value * sign_f

        return value


class FP8Linear(nn.Module):
    """FP8线性层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        format: str = "e4m3",
        per_channel_scale: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.format = format
        self.per_channel_scale = per_channel_scale
        self.compute_dtype = compute_dtype

        # FP8量化权重
        self.register_buffer(
            'weight_fp8',
            torch.zeros((out_features, in_features), dtype=torch.uint8, device=device)
        )

        # 权重缩放因子
        if per_channel_scale:
            self.register_buffer(
                'weight_scale',
                torch.ones(out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_buffer(
                'weight_scale',
                torch.tensor(1.0, dtype=torch.float32, device=device)
            )

        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.bias = None

        # 运行时统计
        self.register_buffer('input_amax_history', torch.zeros(1024, device=device))
        self.register_buffer('weight_amax_history', torch.zeros(1024, device=device))
        self.amax_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 量化输入到FP8 (运行时)
        x_fp8 = self._quantize_activation(x)

        # 反量化权重
        weight = self._dequantize_weight()

        # 反量化输入
        x_dequant = x_fp8.to_float().to(self.compute_dtype)

        # 矩阵乘法
        output = F.linear(x_dequant, weight, self.bias)

        return output

    def _quantize_activation(self, x: torch.Tensor) -> FP8Tensor:
        """量化激活值"""
        # 更新amax历史
        amax = x.abs().max()
        self.input_amax_history[self.amax_counter % len(self.input_amax_history)] = amax
        self.amax_counter += 1

        return FP8Tensor.from_float(
            x, self.format, per_channel=False
        )

    def _dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        weight_fp8 = FP8Tensor(
            self.weight_fp8,
            self.weight_scale,
            self.format,
            self.weight_fp8.device.type
        )
        return weight_fp8.to_float().to(self.compute_dtype)

    def quantize_weight(self, weight: torch.Tensor):
        """量化权重"""
        fp8_tensor = FP8Tensor.from_float(
            weight,
            self.format,
            per_channel=self.per_channel_scale
        )

        self.weight_fp8.copy_(fp8_tensor.data)
        self.weight_scale.copy_(fp8_tensor.scale)


class DelayedScaling:
    """延迟缩放管理器"""

    def __init__(
        self,
        amax_history_len: int = 1024,
        amax_compute_algo: str = "max"
    ):
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.amax_history: Dict[str, torch.Tensor] = {}
        self.scale: Dict[str, torch.Tensor] = {}
        self.scale_inv: Dict[str, torch.Tensor] = {}

    def update_amax(self, name: str, amax: torch.Tensor):
        """更新amax历史"""
        if name not in self.amax_history:
            self.amax_history[name] = torch.zeros(
                self.amax_history_len,
                device=amax.device
            )

        # 滚动更新
        self.amax_history[name] = torch.roll(self.amax_history[name], 1)
        self.amax_history[name][0] = amax

    def compute_scale(self, name: str, format: str = "e4m3"):
        """计算缩放因子"""
        if name not in self.amax_history:
            return

        if self.amax_compute_algo == "max":
            amax = self.amax_history[name].max()
        else:
            amax = self.amax_history[name][0]

        max_val = FP8Tensor.E4M3_MAX if format == "e4m3" else FP8Tensor.E5M2_MAX
        scale = amax / max_val
        scale = torch.clamp(scale, min=1e-12)

        self.scale[name] = scale
        self.scale_inv[name] = 1.0 / scale


class FP8Quantizer:
    """FP8量化器"""

    def __init__(self, config: Optional[FP8Config] = None):
        self.config = config or FP8Config()
        self.delayed_scaling = DelayedScaling(
            amax_history_len=self.config.amax_history_len,
            amax_compute_algo=self.config.amax_compute_algo
        )

    def quantize_model(
        self,
        model: nn.Module,
        module_filter: Optional[callable] = None
    ) -> nn.Module:
        """量化整个模型"""
        if module_filter is None:
            module_filter = lambda m: isinstance(m, nn.Linear)

        for name, module in list(model.named_modules()):
            if module_filter(module):
                fp8_linear = self._create_fp8_linear(module)

                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, fp8_linear)
                else:
                    setattr(model, child_name, fp8_linear)

        return model

    def _create_fp8_linear(self, linear: nn.Linear) -> FP8Linear:
        """创建FP8线性层"""
        fp8_linear = FP8Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            format=self.config.format,
            per_channel_scale=self.config.per_channel_scale,
            compute_dtype=self.config.compute_dtype,
            device=linear.weight.device.type
        )

        # 量化权重
        fp8_linear.quantize_weight(linear.weight.data)

        if linear.bias is not None:
            fp8_linear.bias.copy_(linear.bias.data)

        return fp8_linear

    def quantize_tensor(self, x: torch.Tensor) -> FP8Tensor:
        """量化单个张量"""
        return FP8Tensor.from_float(
            x,
            self.config.format,
            per_channel=self.config.per_channel_scale
        )

    def save_quantized_model(self, model: nn.Module, save_path: str):
        """保存量化模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        state_dict = {}
        for name, param in model.state_dict().items():
            state_dict[name] = param.cpu()

        torch.save(state_dict, save_path / "model.pt")

        config_dict = {
            'format': self.config.format,
            'per_channel_scale': self.config.per_channel_scale,
            'compute_dtype': str(self.config.compute_dtype)
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


def quantize_fp8(
    model: nn.Module,
    format: str = "e4m3",
    per_channel: bool = True,
    **kwargs
) -> nn.Module:
    """
    FP8量化便捷函数

    Args:
        model: 待量化模型
        format: FP8格式 ("e4m3" 或 "e5m2")
        per_channel: 是否按通道量化
        **kwargs: 其他配置参数

    Returns:
        量化后的模型
    """
    config = FP8Config(format=format, per_channel_scale=per_channel, **kwargs)
    quantizer = FP8Quantizer(config)
    return quantizer.quantize_model(model)


def cast_to_fp8(
    x: torch.Tensor,
    format: str = "e4m3",
    scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将张量转换为FP8格式

    Args:
        x: 输入张量
        format: FP8格式
        scale: 可选的缩放因子

    Returns:
        (fp8_data, scale) 元组
    """
    fp8_tensor = FP8Tensor.from_float(x, format, scale)
    return fp8_tensor.data, fp8_tensor.scale


def cast_from_fp8(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    format: str = "e4m3"
) -> torch.Tensor:
    """
    从FP8格式转换回浮点

    Args:
        x_fp8: FP8数据
        scale: 缩放因子
        format: FP8格式

    Returns:
        浮点张量
    """
    fp8_tensor = FP8Tensor(x_fp8, scale, format, x_fp8.device.type)
    return fp8_tensor.to_float()
