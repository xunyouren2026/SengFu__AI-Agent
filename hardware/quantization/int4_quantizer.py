"""
INT4 量化实现

模块路径: hardware/quantization/int4_quantizer.py

提供4-bit整数量化支持，包括:
- 对称/非对称量化
- 按通道/按张量量化
- 分组量化 (Group-wise Quantization)
- 动态/静态量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json


@dataclass
class INT4Config:
    """INT4量化配置"""
    symmetric: bool = True
    per_channel: bool = True
    group_size: int = 128
    dynamic: bool = False
    zero_point: bool = False
    round_method: str = "round"  # "round", "floor", "ceil"
    observer_method: str = "minmax"  # "minmax", "mse", "entropy"
    percentile: float = 99.99
    storage_dtype: torch.dtype = torch.int8  # 存储为int8，实际使用4-bit
    compute_dtype: torch.dtype = torch.float16
    device: str = "cuda"


class INT4Tensor:
    """INT4张量表示"""

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        group_size: int = 128,
        symmetric: bool = True
    ):
        """
        Args:
            data: 量化后的数据 (int8存储，实际4-bit)
            scale: 缩放因子
            zero_point: 零点 (非对称量化时使用)
            group_size: 分组大小
            symmetric: 是否对称量化
        """
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = group_size
        self.symmetric = symmetric

    @classmethod
    def from_float(
        cls,
        x: torch.Tensor,
        config: INT4Config,
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None
    ) -> "INT4Tensor":
        """从浮点张量创建INT4张量"""
        if scale is None:
            scale, zero_point = cls.compute_quantization_params(x, config)

        # 量化
        x_quant = cls.quantize(x, scale, zero_point, config.symmetric)

        return cls(x_quant, scale, zero_point, config.group_size, config.symmetric)

    def to_float(self) -> torch.Tensor:
        """反量化为浮点"""
        return self.dequantize(self.data, self.scale, self.zero_point, self.symmetric)

    @staticmethod
    def compute_quantization_params(
        x: torch.Tensor,
        config: INT4Config
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """计算量化参数"""
        if config.per_channel and x.dim() >= 2:
            # 按输出通道计算
            dims = list(range(1, x.dim()))
            x_min = x.amin(dim=dims, keepdim=True)
            x_max = x.amax(dim=dims, keepdim=True)
        else:
            x_min = x.min()
            x_max = x.max()

        # 应用百分位数
        if config.observer_method == "percentile":
            x_min = torch.quantile(x.flatten(), (100 - config.percentile) / 100)
            x_max = torch.quantile(x.flatten(), config.percentile / 100)

        if config.symmetric:
            # 对称量化
            max_abs = torch.maximum(x_max.abs(), x_min.abs())
            scale = max_abs / 7.0  # 4-bit有符号范围: -8 to 7, 实际使用 -7 to 7
            scale = torch.clamp(scale, min=1e-8)
            zero_point = None
        else:
            # 非对称量化
            scale = (x_max - x_min) / 15.0  # 4-bit无符号范围: 0 to 15
            scale = torch.clamp(scale, min=1e-8)
            zero_point = -x_min / scale
            zero_point = zero_point.round().clamp(0, 15)

        return scale, zero_point

    @staticmethod
    def quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        symmetric: bool
    ) -> torch.Tensor:
        """量化浮点到INT4"""
        if symmetric:
            x_quant = torch.round(x / scale)
            x_quant = torch.clamp(x_quant, -8, 7)
        else:
            x_quant = torch.round(x / scale) + zero_point
            x_quant = torch.clamp(x_quant, 0, 15)

        return x_quant.to(torch.int8)

    @staticmethod
    def dequantize(
        x_quant: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        symmetric: bool
    ) -> torch.Tensor:
        """反量化INT4到浮点"""
        if symmetric:
            x = x_quant.float() * scale
        else:
            x = (x_quant.float() - zero_point) * scale

        return x


class INT4Linear(nn.Module):
    """INT4量化线性层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[INT4Config] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.config = config or INT4Config()
        self.in_features = in_features
        self.out_features = out_features

        # 处理分组
        if self.config.group_size > 0 and in_features % self.config.group_size != 0:
            padding = self.config.group_size - (in_features % self.config.group_size)
            self.padded_in_features = in_features + padding
        else:
            self.padded_in_features = in_features
            padding = 0
        self.padding = padding

        if self.config.group_size > 0:
            self.num_groups = self.padded_in_features // self.config.group_size
        else:
            self.num_groups = 1

        # 量化权重 (使用int8存储，打包2个4-bit值)
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, (self.padded_in_features + 1) // 2), dtype=torch.int8, device=device)
        )

        # 缩放因子
        if self.config.per_channel:
            self.register_buffer(
                'scales',
                torch.ones((out_features, self.num_groups), dtype=torch.float16, device=device)
            )
        else:
            self.register_buffer(
                'scales',
                torch.ones(self.num_groups, dtype=torch.float16, device=device)
            )

        # 零点 (非对称量化)
        if not self.config.symmetric:
            if self.config.per_channel:
                self.register_buffer(
                    'zero_points',
                    torch.zeros((out_features, self.num_groups), dtype=torch.int8, device=device)
                )
            else:
                self.register_buffer(
                    'zero_points',
                    torch.zeros(self.num_groups, dtype=torch.int8, device=device)
                )
        else:
            self.zero_points = None

        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=torch.float16, device=device)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 反量化权重
        weight = self.dequantize_weight()

        # 处理padding
        if self.padding > 0:
            weight = weight[:, :self.in_features]

        # 矩阵乘法
        output = F.linear(x, weight, self.bias)
        return output

    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        # 解包权重
        weight = self.unpack_weights()

        # 按组反量化
        if self.config.group_size > 0:
            weight = weight.reshape(self.out_features, self.num_groups, -1)

            if self.config.per_channel:
                scales = self.scales.unsqueeze(-1)
            else:
                scales = self.scales.view(1, -1, 1)

            if self.zero_points is not None:
                if self.config.per_channel:
                    zeros = self.zero_points.unsqueeze(-1)
                else:
                    zeros = self.zero_points.view(1, -1, 1)
                weight = (weight - zeros) * scales
            else:
                weight = weight * scales

            weight = weight.reshape(self.out_features, self.padded_in_features)
        else:
            if self.zero_points is not None:
                weight = (weight - self.zero_points) * self.scales
            else:
                weight = weight * self.scales

        return weight.float()

    def unpack_weights(self) -> torch.Tensor:
        """解包4-bit权重"""
        weight = torch.zeros(
            (self.out_features, self.padded_in_features),
            dtype=torch.int8,
            device=self.qweight.device
        )

        # 解包：低4位和高4位
        weight[:, ::2] = self.qweight & 0x0F
        weight[:, 1::2] = (self.qweight >> 4) & 0x0F

        # 处理符号 (对称量化时，值8-15表示负数)
        if self.config.symmetric:
            weight = weight.where(weight <= 7, weight - 16)

        return weight

    def pack_weights(self, weight: torch.Tensor):
        """打包权重到4-bit"""
        # 转换到无符号表示
        if self.config.symmetric:
            weight = weight.where(weight >= 0, weight + 16)

        # 打包
        packed = torch.zeros_like(self.qweight)
        packed = weight[:, ::2].to(torch.int8) | (weight[:, 1::2].to(torch.int8) << 4)

        self.qweight.copy_(packed)

    def quantize_weight(self, weight: torch.Tensor):
        """量化权重"""
        # 处理padding
        if self.padding > 0:
            weight = F.pad(weight, (0, self.padding))

        # 按组量化
        scales_list = []
        zeros_list = []

        for g in range(self.num_groups):
            if self.config.group_size > 0:
                start = g * self.config.group_size
                end = (g + 1) * self.config.group_size
                w_group = weight[:, start:end]
            else:
                w_group = weight

            # 计算量化参数
            scale, zero_point = INT4Tensor.compute_quantization_params(
                w_group, self.config
            )

            scales_list.append(scale)
            if zero_point is not None:
                zeros_list.append(zero_point)

            # 量化
            w_quant = INT4Tensor.quantize(
                w_group, scale, zero_point, self.config.symmetric
            )

            # 存储
            if self.config.group_size > 0:
                weight[:, start:end] = w_quant.float()
            else:
                weight = w_quant.float()

        # 存储参数
        if self.config.per_channel:
            self.scales = torch.stack(scales_list, dim=1).half()
        else:
            self.scales = torch.stack(scales_list).half()

        if zeros_list:
            if self.config.per_channel:
                self.zero_points = torch.stack(zeros_list, dim=1).to(torch.int8)
            else:
                self.zero_points = torch.stack(zeros_list).to(torch.int8)

        # 打包权重
        self.pack_weights(weight.to(torch.int8))


class INT4Quantizer:
    """INT4量化器"""

    def __init__(self, config: Optional[INT4Config] = None):
        self.config = config or INT4Config()
        self.observers: Dict[str, Any] = {}

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None,
        module_filter: Optional[Callable[[nn.Module], bool]] = None
    ) -> nn.Module:
        """
        量化模型

        Args:
            model: 待量化模型
            calibration_data: 校准数据 (静态量化时使用)
            module_filter: 模块过滤函数

        Returns:
            量化后的模型
        """
        if module_filter is None:
            module_filter = lambda m: isinstance(m, nn.Linear)

        # 静态量化：收集统计信息
        if not self.config.dynamic and calibration_data is not None:
            self._calibrate(model, calibration_data, module_filter)

        # 替换模块
        for name, module in list(model.named_modules()):
            if module_filter(module):
                int4_linear = self._create_int4_linear(module)

                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, int4_linear)
                else:
                    setattr(model, child_name, int4_linear)

        return model

    def _calibrate(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        module_filter: Callable[[nn.Module], bool]
    ):
        """校准：收集统计信息"""
        model.eval()

        # 注册钩子收集统计
        stats = {}
        handles = []

        def get_hook(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                if name not in stats:
                    stats[name] = {'min': [], 'max': []}
                stats[name]['min'].append(input.min().item())
                stats[name]['max'].append(input.max().item())
            return hook

        for name, module in model.named_modules():
            if module_filter(module):
                handle = module.register_forward_hook(get_hook(name))
                handles.append(handle)

        with torch.no_grad():
            for data in calibration_data:
                _ = model(data)

        for handle in handles:
            handle.remove()

        self.observers = stats

    def _create_int4_linear(self, linear: nn.Linear) -> INT4Linear:
        """创建INT4线性层"""
        int4_linear = INT4Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=self.config,
            device=linear.weight.device.type
        )

        # 量化权重
        int4_linear.quantize_weight(linear.weight.data)

        if linear.bias is not None:
            int4_linear.bias.copy_(linear.bias.data)

        return int4_linear

    def quantize_tensor(self, x: torch.Tensor) -> INT4Tensor:
        """量化单个张量"""
        return INT4Tensor.from_float(x, self.config)

    def save_quantized_model(self, model: nn.Module, save_path: str):
        """保存量化模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        state_dict = {}
        for name, param in model.state_dict().items():
            state_dict[name] = param.cpu()

        torch.save(state_dict, save_path / "model.pt")

        config_dict = {
            'symmetric': self.config.symmetric,
            'per_channel': self.config.per_channel,
            'group_size': self.config.group_size,
            'dynamic': self.config.dynamic
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


def quantize_int4(
    model: nn.Module,
    calibration_data: Optional[List[torch.Tensor]] = None,
    symmetric: bool = True,
    per_channel: bool = True,
    group_size: int = 128,
    **kwargs
) -> nn.Module:
    """
    INT4量化便捷函数

    Args:
        model: 待量化模型
        calibration_data: 校准数据
        symmetric: 是否对称量化
        per_channel: 是否按通道量化
        group_size: 分组大小
        **kwargs: 其他配置参数

    Returns:
        量化后的模型
    """
    config = INT4Config(
        symmetric=symmetric,
        per_channel=per_channel,
        group_size=group_size,
        **kwargs
    )
    quantizer = INT4Quantizer(config)
    return quantizer.quantize_model(model, calibration_data)


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    """
    将INT4值打包到INT8

    Args:
        x: INT4张量 (值范围0-15或-8到7)

    Returns:
        打包后的INT8张量
    """
    # 确保是偶数长度
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))

    # 打包
    packed = (x[..., ::2] & 0x0F) | ((x[..., 1::2] & 0x0F) << 4)
    return packed


def unpack_int4(x: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
    """
    从INT8解包INT4值

    Args:
        x: 打包的INT8张量
        symmetric: 是否对称量化

    Returns:
        解包后的INT4张量
    """
    # 解包
    low = x & 0x0F
    high = (x >> 4) & 0x0F

    # 交错合并
    unpacked = torch.stack([low, high], dim=-1).flatten(-2)

    # 处理符号
    if symmetric:
        unpacked = torch.where(unpacked > 7, unpacked - 16, unpacked)

    return unpacked
