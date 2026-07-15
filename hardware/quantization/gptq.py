"""
GPTQ (General-purpose Post-Training Quantization) 量化实现

模块路径: hardware/quantization/gptq.py

基于OBS (Optimal Brain Surgeon) 框架的逐层量化方法，
通过Hessian矩阵信息来最小化量化误差。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import time


@dataclass
class GPTQConfig:
    """GPTQ量化配置"""
    bits: int = 4
    group_size: int = 128
    damp_percent: float = 0.01
    static_groups: bool = False
    sym: bool = True
    true_sequential: bool = True
    act_order: bool = False  # 是否按激活值排序
    percdamp: float = 0.01
    block_size: int = 128
    device: str = "cuda"


class GPTQLinear(nn.Module):
    """GPTQ量化线性层"""

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
        if in_features % group_size != 0:
            padding = group_size - (in_features % group_size)
            self.padded_in_features = in_features + padding
        else:
            self.padded_in_features = in_features
            padding = 0
        self.padding = padding

        self.num_groups = self.padded_in_features // group_size

        # 量化权重存储
        pack_factor = 32 // bits
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, self.padded_in_features // pack_factor), dtype=torch.int32, device=device)
        )

        # 缩放因子
        self.register_buffer(
            'scales',
            torch.zeros((out_features, self.num_groups), dtype=torch.float16, device=device)
        )

        # 零点
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

        # g_idx用于act_order
        self.register_buffer(
            'g_idx',
            torch.zeros(self.padded_in_features, dtype=torch.int32, device=device)
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
        weight = self.dequantize_weight()

        # 处理padding
        if self.padding > 0:
            weight = weight[:, :self.in_features]

        output = F.linear(x, weight, self.bias)
        return output

    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        # 解包权重
        weight = self.unpack_weights()

        # 应用g_idx重排序
        if self.g_idx.max() > 0:
            # 根据g_idx重新组织权重
            weight = weight[:, torch.argsort(self.g_idx)]

        # 应用缩放和零点
        if self.bits < 8:
            zeros = self.unpack_zeros()
        else:
            zeros = self.zeros

        # 按组反量化
        weight = weight.reshape(self.out_features, self.num_groups, -1)
        scales = self.scales.unsqueeze(-1)
        zeros = zeros.unsqueeze(-1)

        weight = (weight - zeros) * scales
        weight = weight.reshape(self.out_features, self.padded_in_features)

        return weight.float()

    def unpack_weights(self) -> torch.Tensor:
        """解包量化权重"""
        pack_factor = 32 // self.bits
        mask = (1 << self.bits) - 1

        weight = torch.zeros(
            (self.out_features, self.padded_in_features),
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
        """打包权重"""
        pack_factor = 32 // self.bits
        self.qweight.zero_()

        for i in range(pack_factor):
            self.qweight |= (weight[:, i::pack_factor] << (i * self.bits))


class GPTQQuantizer:
    """GPTQ量化器"""

    def __init__(self, config: Optional[GPTQConfig] = None):
        self.config = config or GPTQConfig()
        self.H = {}  # Hessian矩阵缓存
        self.quantizers = {}

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
            calibration_data: 校准数据
            module_filter: 模块过滤函数

        Returns:
            量化后的模型
        """
        if module_filter is None:
            module_filter = lambda m: isinstance(m, nn.Linear)

        # 收集层输入统计
        self._collect_statistics(model, calibration_data, module_filter)

        # 逐层量化
        self._quantize_layers(model, module_filter)

        return model

    def _collect_statistics(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        module_filter: Callable[[nn.Module], bool]
    ):
        """收集Hessian矩阵统计信息"""
        model.eval()

        # 存储每层的输入
        layer_inputs = {}
        handles = []

        def get_hook(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                if name not in layer_inputs:
                    layer_inputs[name] = []
                layer_inputs[name].append(input.detach().cpu())
            return hook

        # 注册钩子
        for name, module in model.named_modules():
            if module_filter(module):
                handle = module.register_forward_hook(get_hook(name))
                handles.append(handle)

        # 前向传播
        with torch.no_grad():
            for data in calibration_data:
                _ = model(data)

        # 移除钩子
        for handle in handles:
            handle.remove()

        # 计算Hessian矩阵
        for name, inputs in layer_inputs.items():
            # 合并所有输入
            all_inputs = torch.cat(inputs, dim=0)

            # 计算Hessian: H = X^T * X / n
            n_samples = all_inputs.shape[0]
            H = torch.zeros((all_inputs.shape[-1], all_inputs.shape[-1]))

            for i in range(n_samples):
                x = all_inputs[i]
                H += torch.outer(x, x)

            H = H / n_samples

            # 添加阻尼
            damp = self.config.percdamp * torch.mean(torch.diag(H))
            H += torch.eye(H.shape[0]) * damp

            self.H[name] = H

    def _quantize_layers(
        self,
        model: nn.Module,
        module_filter: Callable[[nn.Module], bool]
    ):
        """量化所有层"""
        for name, module in list(model.named_modules()):
            if module_filter(module) and name in self.H:
                print(f"Quantizing {name}...")

                # 执行GPTQ量化
                quantized_module = self._gptq_quantize(module, self.H[name])

                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, quantized_module)
                else:
                    setattr(model, child_name, quantized_module)

    def _gptq_quantize(
        self,
        layer: nn.Linear,
        H: torch.Tensor
    ) -> GPTQLinear:
        """
        对单个层执行GPTQ量化

        基于OBS (Optimal Brain Surgeon) 的最优量化算法
        """
        W = layer.weight.data.clone()
        device = W.device

        # 处理padding
        if W.shape[1] % self.config.group_size != 0:
            padding = self.config.group_size - (W.shape[1] % self.config.group_size)
            W = F.pad(W, (0, padding))
        else:
            padding = 0

        # 将Hessian移到相同设备
        H = H.to(device)

        # Cholesky分解
        try:
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except:
            # 如果Cholesky失败，使用伪逆
            H_inv = torch.linalg.pinv(H)

        # 创建量化层
        gptq_layer = GPTQLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bits=self.config.bits,
            group_size=self.config.group_size,
            bias=layer.bias is not None,
            device=device.type
        )
        gptq_layer.padding = padding

        # 按组执行量化
        num_groups = W.shape[1] // self.config.group_size

        scales_list = []
        zeros_list = []

        for g in range(num_groups):
            start = g * self.config.group_size
            end = (g + 1) * self.config.group_size
            W_group = W[:, start:end]

            # 计算缩放和零点
            w_min = W_group.min(dim=1, keepdim=True)[0]
            w_max = W_group.max(dim=1, keepdim=True)[0]

            if self.config.sym:
                max_val = torch.maximum(w_max.abs(), w_min.abs())
                scales = max_val / (2 ** (self.config.bits - 1) - 1)
                scales = torch.clamp(scales, min=1e-8)
                zeros = torch.zeros_like(scales)
            else:
                scales = (w_max - w_min) / (2 ** self.config.bits - 1)
                scales = torch.clamp(scales, min=1e-8)
                zeros = -w_min / scales
                zeros = zeros.round().clamp(0, 2 ** self.config.bits - 1)

            # 量化
            W_quant = torch.clamp(
                torch.round(W_group / scales) + zeros,
                0, 2 ** self.config.bits - 1
            )

            # 计算量化误差
            W_dequant = (W_quant - zeros) * scales
            error = W_group - W_dequant

            # OBS更新: 将误差传播到未量化的权重
            if end < W.shape[1]:
                H_inv_block = H_inv[start:end, end:]
                H_inv_diag = torch.diag(H_inv[start:end, start:end])

                # 误差补偿
                update = error @ (H_inv_block / H_inv_diag.unsqueeze(1))
                W[:, end:] -= update

            scales_list.append(scales.squeeze(1))
            zeros_list.append(zeros.squeeze(1))

            # 存储量化后的权重
            W[:, start:end] = W_quant

        # 设置量化参数
        gptq_layer.scales = torch.stack(scales_list, dim=1).half()

        if self.config.bits < 8:
            gptq_layer.qzeros = torch.stack(zeros_list, dim=1).int()
        else:
            gptq_layer.zeros = torch.stack(zeros_list, dim=1).half()

        # 打包权重
        gptq_layer.pack_weights(W.int())

        # 复制偏置
        if layer.bias is not None:
            gptq_layer.bias.copy_(layer.bias.data)

        return gptq_layer

    def save_quantized_model(self, model: nn.Module, save_path: str):
        """保存量化模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        state_dict = {}
        for name, param in model.state_dict().items():
            state_dict[name] = param.cpu()

        torch.save(state_dict, save_path / "model.pt")

        config_dict = {
            'bits': self.config.bits,
            'group_size': self.config.group_size,
            'sym': self.config.sym,
            'act_order': self.config.act_order
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load_quantized_model(self, load_path: str, model_class: type) -> nn.Module:
        """加载量化模型"""
        load_path = Path(load_path)

        with open(load_path / "config.json", 'r') as f:
            config_dict = json.load(f)
            self.config = GPTQConfig(**config_dict)

        model = model_class()
        state_dict = torch.load(load_path / "model.pt", map_location='cpu')
        model.load_state_dict(state_dict)

        return model


def quantize_gptq(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    **kwargs
) -> nn.Module:
    """
    GPTQ量化便捷函数

    Args:
        model: 待量化模型
        calibration_data: 校准数据
        bits: 量化位数
        group_size: 分组大小
        **kwargs: 其他配置参数

    Returns:
        量化后的模型
    """
    config = GPTQConfig(bits=bits, group_size=group_size, **kwargs)
    quantizer = GPTQQuantizer(config)
    return quantizer.quantize(model, calibration_data)
