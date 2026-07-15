"""
INT8 量化实现

模块路径: hardware/quantization/int8_quantizer.py

提供8-bit整数量化支持，包括:
- 对称/非对称量化
- 按通道/按张量量化
- 动态/静态量化
- 感知训练量化 (QAT)
- 后训练量化 (PTQ)
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
class INT8Config:
    """INT8量化配置"""
    symmetric: bool = True
    per_channel: bool = True
    dynamic: bool = False
    observer_method: str = "minmax"  # "minmax", "moving_average", "histogram"
    quant_min: int = -128
    quant_max: int = 127
    eps: float = 1e-8
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class INT8Tensor:
    """INT8张量表示"""

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        symmetric: bool = True
    ):
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.symmetric = symmetric

    @classmethod
    def from_float(
        cls,
        x: torch.Tensor,
        config: INT8Config,
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None
    ) -> "INT8Tensor":
        """从浮点张量创建INT8张量"""
        if scale is None:
            scale, zero_point = cls.compute_quantization_params(x, config)

        x_quant = cls.quantize(x, scale, zero_point, config)
        return cls(x_quant, scale, zero_point, config.symmetric)

    def to_float(self) -> torch.Tensor:
        """反量化为浮点"""
        return self.dequantize(self.data, self.scale, self.zero_point, self.symmetric)

    @staticmethod
    def compute_quantization_params(
        x: torch.Tensor,
        config: INT8Config
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

        if config.symmetric:
            # 对称量化
            max_abs = torch.maximum(x_max.abs(), x_min.abs())
            scale = max_abs / 127.0
            scale = torch.clamp(scale, min=config.eps)
            zero_point = None
        else:
            # 非对称量化
            scale = (x_max - x_min) / 255.0
            scale = torch.clamp(scale, min=config.eps)
            zero_point = -x_min / scale - 128
            zero_point = zero_point.round().clamp(-128, 127)

        return scale, zero_point

    @staticmethod
    def quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        config: INT8Config
    ) -> torch.Tensor:
        """量化浮点到INT8"""
        if config.symmetric:
            x_quant = torch.round(x / scale)
            x_quant = torch.clamp(x_quant, -128, 127)
        else:
            x_quant = torch.round(x / scale) + zero_point
            x_quant = torch.clamp(x_quant, -128, 127)

        return x_quant.to(torch.int8)

    @staticmethod
    def dequantize(
        x_quant: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        symmetric: bool
    ) -> torch.Tensor:
        """反量化INT8到浮点"""
        if symmetric:
            x = x_quant.float() * scale
        else:
            x = (x_quant.float() - zero_point) * scale

        return x


class INT8Linear(nn.Module):
    """INT8量化线性层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[INT8Config] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.config = config or INT8Config()
        self.in_features = in_features
        self.out_features = out_features

        # INT8量化权重
        self.register_buffer(
            'weight_int8',
            torch.zeros((out_features, in_features), dtype=torch.int8, device=device)
        )

        # 缩放因子
        if self.config.per_channel:
            self.register_buffer(
                'weight_scale',
                torch.ones(out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_buffer(
                'weight_scale',
                torch.tensor(1.0, dtype=torch.float32, device=device)
            )

        # 零点 (非对称量化)
        if not self.config.symmetric:
            if self.config.per_channel:
                self.register_buffer(
                    'weight_zero_point',
                    torch.zeros(out_features, dtype=torch.int32, device=device)
                )
            else:
                self.register_buffer(
                    'weight_zero_point',
                    torch.tensor(0, dtype=torch.int32, device=device)
                )
        else:
            self.weight_zero_point = None

        # 输入量化参数 (静态量化时使用)
        self.register_buffer('input_scale', torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer('input_zero_point', torch.tensor(0, dtype=torch.int32, device=device))

        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=self.config.dtype, device=device)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.config.dynamic:
            # 动态量化：运行时计算输入量化参数
            x_scale, x_zero_point = INT8Tensor.compute_quantization_params(x, self.config)
            x_quant = INT8Tensor.quantize(x, x_scale, x_zero_point, self.config)
            x_dequant = INT8Tensor.dequantize(x_quant, x_scale, x_zero_point, self.config.symmetric)
        else:
            # 静态量化：使用预计算的参数
            x_dequant = x

        # 反量化权重
        weight = self.dequantize_weight()

        # 矩阵乘法
        output = F.linear(x_dequant, weight, self.bias)
        return output

    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        if self.config.per_channel:
            scale = self.weight_scale.view(-1, 1)
        else:
            scale = self.weight_scale

        if self.weight_zero_point is not None:
            if self.config.per_channel:
                zero = self.weight_zero_point.view(-1, 1)
            else:
                zero = self.weight_zero_point
            weight = (self.weight_int8.float() - zero) * scale
        else:
            weight = self.weight_int8.float() * scale

        return weight.to(self.config.dtype)

    def quantize_weight(self, weight: torch.Tensor):
        """量化权重"""
        scale, zero_point = INT8Tensor.compute_quantization_params(
            weight, self.config
        )

        self.weight_scale.copy_(scale.view_as(self.weight_scale))
        if zero_point is not None:
            self.weight_zero_point.copy_(zero_point.view_as(self.weight_zero_point))

        self.weight_int8.copy_(
            INT8Tensor.quantize(weight, scale, zero_point, self.config)
        )


class FakeQuantize(nn.Module):
    """伪量化模块 (用于QAT)"""

    def __init__(
        self,
        config: INT8Config,
        observer_method: str = "moving_average"
    ):
        super().__init__()
        self.config = config
        self.observer_method = observer_method

        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0))
        self.register_buffer('observer_enabled', torch.tensor(1, dtype=torch.uint8))
        self.register_buffer('fake_quant_enabled', torch.tensor(1, dtype=torch.uint8))

        # 移动平均统计
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.observer_enabled:
            self._update_observer(x)

        if self.fake_quant_enabled:
            return self._fake_quantize(x)

        return x

    def _update_observer(self, x: torch.Tensor):
        """更新观察统计"""
        if self.observer_method == "minmax":
            self.min_val = torch.min(self.min_val, x.min())
            self.max_val = torch.max(self.max_val, x.max())
        elif self.observer_method == "moving_average":
            momentum = 0.01
            self.min_val = self.min_val * (1 - momentum) + x.min() * momentum
            self.max_val = self.max_val * (1 - momentum) + x.max() * momentum

        # 重新计算量化参数
        if self.config.symmetric:
            max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs())
            self.scale = max_abs / 127.0
            self.scale = torch.clamp(self.scale, min=self.config.eps)
            self.zero_point = torch.tensor(0)
        else:
            self.scale = (self.max_val - self.min_val) / 255.0
            self.scale = torch.clamp(self.scale, min=self.config.eps)
            self.zero_point = -self.min_val / self.scale - 128
            self.zero_point = self.zero_point.round().clamp(-128, 127)

    def _fake_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """伪量化"""
        if self.config.symmetric:
            x_quant = torch.round(x / self.scale)
            x_quant = torch.clamp(x_quant, -128, 127)
            x_dequant = x_quant * self.scale
        else:
            x_quant = torch.round(x / self.scale) + self.zero_point
            x_quant = torch.clamp(x_quant, -128, 127)
            x_dequant = (x_quant - self.zero_point) * self.scale

        # 模拟量化噪声
        return x + (x_dequant - x).detach()


class INT8Quantizer:
    """INT8量化器"""

    def __init__(self, config: Optional[INT8Config] = None):
        self.config = config or INT8Config()
        self.observers: Dict[str, Any] = {}

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None,
        module_filter: Optional[Callable[[nn.Module], bool]] = None
    ) -> nn.Module:
        """
        量化模型 (PTQ)

        Args:
            model: 待量化模型
            calibration_data: 校准数据
            module_filter: 模块过滤函数

        Returns:
            量化后的模型
        """
        if module_filter is None:
            module_filter = lambda m: isinstance(m, nn.Linear)

        # 校准
        if calibration_data is not None and not self.config.dynamic:
            self._calibrate(model, calibration_data, module_filter)

        # 替换模块
        for name, module in list(model.named_modules()):
            if module_filter(module):
                int8_linear = self._create_int8_linear(module)

                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, int8_linear)
                else:
                    setattr(model, child_name, int8_linear)

        return model

    def prepare_qat(
        self,
        model: nn.Module,
        module_filter: Optional[Callable[[nn.Module], bool]] = None
    ) -> nn.Module:
        """
        准备感知训练量化 (QAT)

        Args:
            model: 待量化模型
            module_filter: 模块过滤函数

        Returns:
            准备好的模型
        """
        if module_filter is None:
            module_filter = lambda m: isinstance(m, (nn.Linear, nn.Conv2d))

        # 插入伪量化模块
        for name, module in list(model.named_modules()):
            if module_filter(module):
                # 在权重和激活上添加伪量化
                fake_quant_w = FakeQuantize(self.config, "moving_average")
                fake_quant_a = FakeQuantize(self.config, "moving_average")

                # 包装模块
                wrapped = QuantWrapper(module, fake_quant_w, fake_quant_a)

                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, wrapped)
                else:
                    setattr(model, child_name, wrapped)

        return model

    def _calibrate(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        module_filter: Callable[[nn.Module], bool]
    ):
        """校准"""
        model.eval()

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

    def _create_int8_linear(self, linear: nn.Linear) -> INT8Linear:
        """创建INT8线性层"""
        int8_linear = INT8Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=self.config,
            device=linear.weight.device.type
        )

        int8_linear.quantize_weight(linear.weight.data)

        if linear.bias is not None:
            int8_linear.bias.copy_(linear.bias.data)

        return int8_linear

    def convert_qat_to_quantized(self, model: nn.Module) -> nn.Module:
        """将QAT模型转换为量化模型"""
        for name, module in list(model.named_modules()):
            if isinstance(module, QuantWrapper):
                # 提取量化参数并创建INT8层
                int8_linear = self._create_int8_linear_from_qat(module)

                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, int8_linear)
                else:
                    setattr(model, child_name, int8_linear)

        return model

    def _create_int8_linear_from_qat(self, wrapper: "QuantWrapper") -> INT8Linear:
        """从QAT包装器创建INT8层"""
        linear = wrapper.module

        int8_linear = INT8Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=self.config,
            device=linear.weight.device.type
        )

        # 使用伪量化的参数
        int8_linear.weight_scale.copy_(wrapper.fake_quant_w.scale)
        if int8_linear.weight_zero_point is not None:
            int8_linear.weight_zero_point.copy_(wrapper.fake_quant_w.zero_point)

        # 量化权重
        int8_linear.weight_int8.copy_(
            INT8Tensor.quantize(
                linear.weight.data,
                wrapper.fake_quant_w.scale,
                wrapper.fake_quant_w.zero_point,
                self.config
            )
        )

        if linear.bias is not None:
            int8_linear.bias.copy_(linear.bias.data)

        return int8_linear

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
            'dynamic': self.config.dynamic
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


class QuantWrapper(nn.Module):
    """量化包装器 (用于QAT)"""

    def __init__(
        self,
        module: nn.Module,
        fake_quant_w: FakeQuantize,
        fake_quant_a: FakeQuantize
    ):
        super().__init__()
        self.module = module
        self.fake_quant_w = fake_quant_w
        self.fake_quant_a = fake_quant_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 量化激活
        x = self.fake_quant_a(x)

        # 量化权重并计算
        if isinstance(self.module, nn.Linear):
            weight = self.fake_quant_w(self.module.weight)
            output = F.linear(x, weight, self.module.bias)
        elif isinstance(self.module, nn.Conv2d):
            weight = self.fake_quant_w(self.module.weight)
            output = F.conv2d(
                x, weight, self.module.bias,
                self.module.stride, self.module.padding,
                self.module.dilation, self.module.groups
            )
        else:
            output = self.module(x)

        return output


def quantize_int8(
    model: nn.Module,
    calibration_data: Optional[List[torch.Tensor]] = None,
    symmetric: bool = True,
    per_channel: bool = True,
    dynamic: bool = False,
    **kwargs
) -> nn.Module:
    """
    INT8量化便捷函数

    Args:
        model: 待量化模型
        calibration_data: 校准数据
        symmetric: 是否对称量化
        per_channel: 是否按通道量化
        dynamic: 是否动态量化
        **kwargs: 其他配置参数

    Returns:
        量化后的模型
    """
    config = INT8Config(
        symmetric=symmetric,
        per_channel=per_channel,
        dynamic=dynamic,
        **kwargs
    )
    quantizer = INT8Quantizer(config)
    return quantizer.quantize_model(model, calibration_data)


def prepare_qat(
    model: nn.Module,
    symmetric: bool = True,
    per_channel: bool = True,
    **kwargs
) -> nn.Module:
    """
    准备QAT便捷函数

    Args:
        model: 待量化模型
        symmetric: 是否对称量化
        per_channel: 是否按通道量化
        **kwargs: 其他配置参数

    Returns:
        准备好的模型
    """
    config = INT8Config(symmetric=symmetric, per_channel=per_channel, **kwargs)
    quantizer = INT8Quantizer(config)
    return quantizer.prepare_qat(model)
