"""
AWQ (Activation-aware Weight Quantization) 量化实现

模块路径: hardware/quantization/awq.py

基于 "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
论文实现，通过考虑激活值分布来保护重要的权重通道。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json


@dataclass
class AWQConfig:
    """AWQ量化配置"""
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"
    scaling_factor: float = 0.5
    n_grid: int = 20
    max_shrink: float = 0.0
    apply_scale: bool = True
    apply_clip: bool = True
    device: str = "cuda"


class AWQLinear(nn.Module):
    """AWQ量化线性层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.device = device

        # 计算分组数
        self.num_groups = in_features // group_size

        # 量化权重存储 (int32打包存储)
        pack_factor = 32 // bits
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // pack_factor), dtype=torch.int32, device=device)
        )

        # 缩放因子和零点
        self.register_buffer(
            'scales',
            torch.zeros((out_features, self.num_groups), dtype=torch.float16, device=device)
        )

        if bits < 8:
            self.register_buffer(
                'qzeros',
                torch.zeros((out_features, self.num_groups), dtype=torch.int32, device=device)
            )
        else:
            self.register_buffer(
                'zeros',
                torch.zeros((out_features, self.num_groups), dtype=torch.float16, device=device)
            )

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

        # 执行矩阵乘法
        output = F.linear(x, weight, self.bias)
        return output

    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重到浮点"""
        # 解包量化权重
        weight = self.unpack_weights()

        # 应用缩放和零点
        if self.bits < 8:
            zeros = self.unpack_zeros()
        else:
            zeros = self.zeros

        # 扩展维度以匹配
        weight = weight.reshape(self.out_features, self.num_groups, -1)
        scales = self.scales.unsqueeze(-1)
        zeros = zeros.unsqueeze(-1)

        # 反量化
        weight = (weight - zeros) * scales
        weight = weight.reshape(self.out_features, self.in_features)

        return weight.float()

    def unpack_weights(self) -> torch.Tensor:
        """解包量化权重"""
        pack_factor = 32 // self.bits
        mask = (1 << self.bits) - 1

        weight = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.int32,
            device=self.device
        )

        for i in range(pack_factor):
            weight[:, i::pack_factor] = (self.qweight >> (i * self.bits)) & mask

        return weight

    def unpack_zeros(self) -> torch.Tensor:
        """解包零点"""
        pack_factor = 32 // self.bits
        mask = (1 << self.bits) - 1

        zeros = torch.zeros(
            (self.out_features, self.num_groups),
            dtype=torch.int32,
            device=self.device
        )

        for i in range(pack_factor):
            zeros[:, i::pack_factor] = (self.qzeros >> (i * self.bits)) & mask

        return zeros

    def pack_weights(self, weight: torch.Tensor):
        """打包量化权重"""
        pack_factor = 32 // self.bits
        self.qweight.zero_()

        for i in range(pack_factor):
            self.qweight |= (weight[:, i::pack_factor] << (i * self.bits))


class AWQQuantizer:
    """AWQ量化器"""

    SUPPORTED_BITS = [4, 8]

    def __init__(self, config: Optional[AWQConfig] = None):
        self.config = config or AWQConfig()
        self.scales_dict: Dict[str, torch.Tensor] = {}
        self.clip_dict: Dict[str, torch.Tensor] = {}

    def quantize(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        module_filter: Optional[Callable[[nn.Module], bool]] = None
    ) -> nn.Module:
        """
        量化模型

        Args:
            model: 待量化模型
            calibration_data: 校准数据列表
            module_filter: 模块过滤函数

        Returns:
            量化后的模型
        """
        if module_filter is None:
            module_filter = lambda m: isinstance(m, nn.Linear)

        # 第一步：搜索最佳缩放因子
        self._search_scales(model, calibration_data, module_filter)

        # 第二步：应用缩放并量化
        self._apply_quantization(model, module_filter)

        return model

    def _search_scales(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        module_filter: Callable[[nn.Module], bool]
    ):
        """搜索最佳缩放因子"""
        model.eval()

        # 收集激活值
        activations = {}
        handles = []

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                activations[name] = input.detach()
            return hook

        # 注册钩子
        for name, module in model.named_modules():
            if module_filter(module):
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)

        # 前向传播收集激活
        with torch.no_grad():
            for data in calibration_data[:min(10, len(calibration_data))]:
                _ = model(data)

        # 移除钩子
        for handle in handles:
            handle.remove()

        # 为每个模块搜索缩放因子
        for name, module in model.named_modules():
            if module_filter(module) and name in activations:
                scale = self._search_module_scale(module, activations[name])
                self.scales_dict[name] = scale

    def _search_module_scale(
        self,
        module: nn.Module,
        activation: torch.Tensor
    ) -> torch.Tensor:
        """
        为单个模块搜索最佳缩放因子

        基于激活值分布计算每个通道的重要性，并搜索最佳缩放。
        """
        weight = module.weight.data

        # 计算激活值的平均幅度 (作为通道重要性指标)
        act_mean = activation.abs().mean(dim=0)

        # 计算权重和激活的联合重要性
        importance = weight.abs() * act_mean.unsqueeze(0)

        # 按组聚合重要性
        group_size = self.config.group_size
        num_groups = weight.shape[1] // group_size

        importance = importance.reshape(weight.shape[0], num_groups, group_size)
        group_importance = importance.sum(dim=-1)

        # 搜索最佳缩放因子
        best_scales = torch.ones(
            (weight.shape[0], num_groups),
            device=weight.device,
            dtype=weight.dtype
        )

        # 网格搜索
        grid_range = torch.linspace(
            0, 1, self.config.n_grid,
            device=weight.device
        )

        for group_idx in range(num_groups):
            best_loss = float('inf')
            best_scale = 1.0

            for ratio in grid_range:
                scale = 1 + ratio * self.config.scaling_factor

                # 模拟量化
                w_group = weight[:, group_idx * group_size:(group_idx + 1) * group_size]
                w_scaled = w_group * scale

                # 量化并反量化
                w_quant = self._simulate_quantize(w_scaled)
                w_dequant = w_quant / scale

                # 计算损失
                loss = (w_group - w_dequant).pow(2).mean()

                if loss < best_loss:
                    best_loss = loss
                    best_scale = scale

            best_scales[:, group_idx] = best_scale

        return best_scales

    def _simulate_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """模拟量化过程"""
        max_val = weight.abs().amax(dim=-1, keepdim=True)
        scales = max_val / (2 ** (self.config.bits - 1) - 1)

        quantized = torch.clamp(
            torch.round(weight / scales),
            -(2 ** (self.config.bits - 1)),
            2 ** (self.config.bits - 1) - 1
        )

        return quantized * scales

    def _apply_quantization(
        self,
        model: nn.Module,
        module_filter: Callable[[nn.Module], bool]
    ):
        """应用量化"""
        for name, module in list(model.named_modules()):
            if module_filter(module) and name in self.scales_dict:
                # 创建AWQ线性层
                awq_linear = self._create_awq_linear(module, self.scales_dict[name])

                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, awq_linear)
                else:
                    setattr(model, child_name, awq_linear)

    def _create_awq_linear(
        self,
        linear: nn.Linear,
        scales: torch.Tensor
    ) -> AWQLinear:
        """从普通线性层创建AWQ线性层"""
        awq_linear = AWQLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=self.config.bits,
            group_size=self.config.group_size,
            bias=linear.bias is not None,
            device=linear.weight.device.type
        )

        # 应用缩放并量化权重
        weight = linear.weight.data

        # 按组应用缩放
        num_groups = weight.shape[1] // self.config.group_size
        weight_scaled = weight.clone()

        for g in range(num_groups):
            start = g * self.config.group_size
            end = (g + 1) * self.config.group_size
            weight_scaled[:, start:end] *= scales[:, g:g+1]

        # 量化权重
        self._quantize_to_awq(weight_scaled, awq_linear)

        # 复制偏置
        if linear.bias is not None:
            awq_linear.bias.copy_(linear.bias.data)

        return awq_linear

    def _quantize_to_awq(self, weight: torch.Tensor, awq_linear: AWQLinear):
        """将权重量化并存储到AWQ线性层"""
        num_groups = weight.shape[1] // self.config.group_size

        scales_list = []
        zeros_list = []
        quantized_groups = []

        for g in range(num_groups):
            start = g * self.config.group_size
            end = (g + 1) * self.config.group_size
            w_group = weight[:, start:end]

            # 计算缩放和零点
            w_min = w_group.min(dim=-1, keepdim=True)[0]
            w_max = w_group.max(dim=-1, keepdim=True)[0]

            if self.config.zero_point:
                scales = (w_max - w_min) / (2 ** self.config.bits - 1)
                zeros = -w_min / scales
                zeros = zeros.squeeze().round().clamp(0, 2 ** self.config.bits - 1)
            else:
                max_val = w_group.abs().amax(dim=-1, keepdim=True)
                scales = max_val / (2 ** (self.config.bits - 1) - 1)
                zeros = torch.zeros(weight.shape[0], device=weight.device)

            # 量化
            w_quant = torch.clamp(
                torch.round(w_group / scales.unsqueeze(-1)) + zeros.unsqueeze(-1),
                0, 2 ** self.config.bits - 1
            )

            scales_list.append(scales.squeeze())
            zeros_list.append(zeros)
            quantized_groups.append(w_quant)

        # 存储参数
        awq_linear.scales = torch.stack(scales_list, dim=1).half()

        if self.config.bits < 8:
            awq_linear.qzeros = torch.stack(zeros_list, dim=1).int()
        else:
            awq_linear.zeros = torch.stack(zeros_list, dim=1).half()

        # 合并并打包量化权重
        full_quantized = torch.cat(quantized_groups, dim=1).int()
        awq_linear.pack_weights(full_quantized)

    def save_quantized_model(self, model: nn.Module, save_path: str):
        """保存量化后的模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存模型状态
        state_dict = {}
        for name, param in model.state_dict().items():
            state_dict[name] = param.cpu()

        torch.save(state_dict, save_path / "model.pt")

        # 保存配置
        config_dict = {
            'bits': self.config.bits,
            'group_size': self.config.group_size,
            'zero_point': self.config.zero_point,
            'version': self.config.version
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load_quantized_model(self, load_path: str, model_class: type) -> nn.Module:
        """加载量化后的模型"""
        load_path = Path(load_path)

        # 加载配置
        with open(load_path / "config.json", 'r') as f:
            config_dict = json.load(f)
            self.config = AWQConfig(**config_dict)

        # 加载模型
        model = model_class()
        state_dict = torch.load(load_path / "model.pt", map_location='cpu')
        model.load_state_dict(state_dict)

        return model


def quantize_awq(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    **kwargs
) -> nn.Module:
    """
    AWQ量化便捷函数

    Args:
        model: 待量化模型
        calibration_data: 校准数据
        bits: 量化位数
        group_size: 分组大小
        **kwargs: 其他配置参数

    Returns:
        量化后的模型
    """
    config = AWQConfig(bits=bits, group_size=group_size, **kwargs)
    quantizer = AWQQuantizer(config)
    return quantizer.quantize(model, calibration_data)
