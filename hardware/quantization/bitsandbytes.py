"""
BitsAndBytes 量化实现

模块路径: hardware/quantization/bitsandbytes.py

提供8-bit和4-bit量化支持，包括:
- 8-bit 块级量化 (LLM.int8())
- 4-bit Normal Float (NF4) 量化
- 4-bit Double Quantization
- 混合精度推理
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
class BitsAndBytesConfig:
    """BitsAndBytes量化配置"""
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: torch.dtype = torch.uint8
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    device: str = "cuda"


class Linear8bitLt(nn.Module):
    """8-bit线性层 (LLM.int8)"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        has_fp16_weights: bool = False,
        threshold: float = 6.0,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_fp16_weights = has_fp16_weights
        self.threshold = threshold

        # 8-bit量化权重 (int8)
        self.register_buffer(
            'weight',
            torch.zeros((out_features, in_features), dtype=torch.int8, device=device)
        )

        # 量化统计信息 (每128个元素的块)
        self.block_size = 128
        num_blocks = (in_features + self.block_size - 1) // self.block_size
        self.register_buffer(
            'weight_scale',
            torch.zeros((out_features, num_blocks), dtype=torch.float16, device=device)
        )
        self.register_buffer(
            'weight_zero_point',
            torch.zeros((out_features, num_blocks), dtype=torch.float16, device=device)
        )

        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=torch.float16, device=device)
            )
        else:
            self.bias = None

        # 用于异常值处理的FP16权重
        self.register_buffer(
            'weight_fp16',
            torch.zeros((out_features, in_features), dtype=torch.float16, device=device)
        )

        # 异常值掩码
        self.register_buffer(
            'outlier_mask',
            torch.zeros((out_features, in_features), dtype=torch.bool, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，使用混合精度分解"""
        # 分离异常值
        x_outliers = x * self.outlier_mask.unsqueeze(0).expand(x.shape[0], -1, -1)
        x_normal = x - x_outliers

        # 正常值使用8-bit计算
        weight_dequant = self.dequantize_weight()
        output_normal = F.linear(x_normal, weight_dequant, None)

        # 异常值使用FP16计算
        if self.outlier_mask.any():
            output_outliers = F.linear(x_outliers, self.weight_fp16, None)
            output = output_normal + output_outliers
        else:
            output = output_normal

        if self.bias is not None:
            output = output + self.bias

        return output

    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        weight_float = torch.zeros_like(self.weight, dtype=torch.float16)

        for i in range(self.weight_scale.shape[1]):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, self.weight.shape[1])

            block = self.weight[:, start:end].float()
            scale = self.weight_scale[:, i:i+1]
            zero = self.weight_zero_point[:, i:i+1]

            weight_float[:, start:end] = (block - zero) * scale

        return weight_float

    def quantize_weight(self, weight: torch.Tensor):
        """量化FP16权重到8-bit"""
        # 检测异常值
        outlier_threshold = weight.abs().mean() * self.threshold
        self.outlier_mask = weight.abs() > outlier_threshold

        # 保存异常值
        self.weight_fp16.copy_(weight * self.outlier_mask)

        # 量化正常值
        weight_to_quant = weight * (~self.outlier_mask)

        for i in range(self.weight_scale.shape[1]):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, weight.shape[1])

            block = weight_to_quant[:, start:end]

            # 计算每块的缩放和零点
            w_min = block.min(dim=1, keepdim=True)[0]
            w_max = block.max(dim=1, keepdim=True)[0]

            scale = (w_max - w_min) / 255.0
            scale = torch.clamp(scale, min=1e-8)
            zero = w_min

            # 量化
            quantized = torch.clamp(
                torch.round((block - zero) / scale),
                0, 255
            ).to(torch.int8)

            self.weight[:, start:end].copy_(quantized)
            self.weight_scale[:, i].copy_(scale.squeeze().half())
            self.weight_zero_point[:, i].copy_(zero.squeeze().half())


class Linear4bit(nn.Module):
    """4-bit线性层 (NF4/FP4)"""

    # NF4量化级别 (基于正态分布)
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])

    # FP4量化级别
    FP4_E2M1_LEVELS = torch.tensor([
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, float('nan')  # NaN用于表示0
    ])

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str = "nf4",
        use_double_quant: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.use_double_quant = use_double_quant
        self.compute_dtype = compute_dtype

        # 4-bit权重打包存储 (2个4-bit值打包到1个uint8)
        self.register_buffer(
            'weight',
            torch.zeros((out_features, (in_features + 1) // 2), dtype=torch.uint8, device=device)
        )

        # 块级量化参数
        self.block_size = 64
        num_blocks = (in_features + self.block_size - 1) // self.block_size

        # 一级量化参数 (absmax)
        self.register_buffer(
            'absmax',
            torch.zeros((out_features, num_blocks), dtype=torch.float16, device=device)
        )

        # 双重量化参数
        if use_double_quant:
            dq_block_size = 256
            dq_num_blocks = (num_blocks + dq_block_size - 1) // dq_block_size

            self.register_buffer(
                'absmax_code',
                torch.zeros((out_features, (num_blocks + 1) // 2), dtype=torch.uint8, device=device)
            )
            self.register_buffer(
                'absmax_scale',
                torch.zeros((out_features, dq_num_blocks), dtype=torch.float16, device=device)
            )
            self.register_buffer(
                'absmax_zero_point',
                torch.zeros((out_features, dq_num_blocks), dtype=torch.float16, device=device)
            )

        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        weight_dequant = self.dequantize_weight()
        output = F.linear(x.to(self.compute_dtype), weight_dequant, self.bias)
        return output

    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        # 解包4-bit权重
        weight_4bit = self.unpack_weights()

        # 反量化
        if self.quant_type == "nf4":
            levels = self.NF4_LEVELS.to(weight_4bit.device)
        else:
            levels = self.FP4_E2M1_LEVELS.to(weight_4bit.device)

        # 映射到量化级别
        weight_normalized = levels[weight_4bit.long()]

        # 应用absmax缩放
        absmax_full = self.get_absmax()

        # 扩展absmax以匹配权重维度
        weight_dequant = torch.zeros(
            (self.out_features, self.in_features),
            dtype=self.compute_dtype,
            device=weight_4bit.device
        )

        for i in range(absmax_full.shape[1]):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, self.in_features)
            weight_dequant[:, start:end] = (
                weight_normalized[:, start:end] * absmax_full[:, i:i+1]
            )

        return weight_dequant

    def get_absmax(self) -> torch.Tensor:
        """获取解压缩后的absmax"""
        if not self.use_double_quant:
            return self.absmax

        # 反量化absmax
        absmax_dequant = torch.zeros_like(self.absmax)

        dq_block_size = 256
        for i in range(self.absmax_scale.shape[1]):
            start = i * dq_block_size
            end = min((i + 1) * dq_block_size, self.absmax.shape[1])

            # 解包absmax_code
            code_packed = self.absmax_code[:, start:end]
            code_unpacked = torch.zeros(
                (self.out_features, end - start),
                dtype=torch.int32,
                device=code_packed.device
            )
            code_unpacked[:, ::2] = code_packed & 0x0F
            code_unpacked[:, 1::2] = (code_packed >> 4) & 0x0F

            # 反量化
            absmax_dequant[:, start:end] = (
                code_unpacked.float() - self.absmax_zero_point[:, i:i+1]
            ) * self.absmax_scale[:, i:i+1]

        return absmax_dequant

    def unpack_weights(self) -> torch.Tensor:
        """解包4-bit权重到int32"""
        weight_unpacked = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.int32,
            device=self.weight.device
        )

        # 解包：低4位和高4位
        weight_unpacked[:, ::2] = self.weight & 0x0F
        weight_unpacked[:, 1::2] = (self.weight >> 4) & 0x0F

        return weight_unpacked

    def quantize_weight(self, weight: torch.Tensor):
        """量化FP16权重到4-bit"""
        num_blocks = (self.in_features + self.block_size - 1) // self.block_size

        absmax_list = []

        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, self.in_features)
            block = weight[:, start:end]

            # 计算absmax
            absmax = block.abs().max(dim=1, keepdim=True)[0]
            absmax = torch.clamp(absmax, min=1e-8)
            absmax_list.append(absmax.squeeze(1))

            # 归一化
            block_normalized = block / absmax

            # 量化到4-bit
            if self.quant_type == "nf4":
                quantized_block = self._quantize_nf4(block_normalized)
            else:
                quantized_block = self._quantize_fp4(block_normalized)

            # 打包存储
            if i == 0:
                packed = torch.zeros(
                    (self.out_features, (self.in_features + 1) // 2),
                    dtype=torch.uint8,
                    device=weight.device
                )

            # 存储到对应位置
            for j in range(end - start):
                col = start + j
                if col % 2 == 0:
                    packed[:, col // 2] = quantized_block[:, j].to(torch.uint8)
                else:
                    packed[:, col // 2] |= (quantized_block[:, j].to(torch.uint8) << 4)

        self.weight.copy_(packed)

        # 存储或双重量化absmax
        absmax_tensor = torch.stack(absmax_list, dim=1)

        if self.use_double_quant:
            self._double_quantize_absmax(absmax_tensor)
        else:
            self.absmax.copy_(absmax_tensor.half())

    def _quantize_nf4(self, x: torch.Tensor) -> torch.Tensor:
        """量化到NF4"""
        levels = self.NF4_LEVELS.to(x.device)

        # 找到最近的量化级别
        x_expanded = x.unsqueeze(-1)
        levels_expanded = levels.view(1, 1, -1)

        distances = torch.abs(x_expanded - levels_expanded)
        indices = torch.argmin(distances, dim=-1)

        return indices.to(torch.int32)

    def _quantize_fp4(self, x: torch.Tensor) -> torch.Tensor:
        """量化到FP4 E2M1"""
        levels = self.FP4_E2M1_LEVELS.to(x.device)

        x_expanded = x.unsqueeze(-1)
        levels_expanded = levels.view(1, 1, -1)

        distances = torch.abs(x_expanded - levels_expanded)
        indices = torch.argmin(distances, dim=-1)

        return indices.to(torch.int32)

    def _double_quantize_absmax(self, absmax: torch.Tensor):
        """对absmax进行双重量化"""
        dq_block_size = 256
        num_blocks = absmax.shape[1]
        dq_num_blocks = (num_blocks + dq_block_size - 1) // dq_block_size

        absmax_quantized = torch.zeros_like(absmax)

        for i in range(dq_num_blocks):
            start = i * dq_block_size
            end = min((i + 1) * dq_block_size, num_blocks)
            block = absmax[:, start:end]

            # 量化到8-bit
            block_min = block.min(dim=1, keepdim=True)[0]
            block_max = block.max(dim=1, keepdim=True)[0]
            scale = (block_max - block_min) / 255.0
            scale = torch.clamp(scale, min=1e-8)
            zero = block_min

            quantized = torch.clamp(
                torch.round((block - zero) / scale),
                0, 255
            ).to(torch.int32)

            absmax_quantized[:, start:end] = quantized

            self.absmax_scale[:, i].copy_(scale.squeeze().half())
            self.absmax_zero_point[:, i].copy_(zero.squeeze().half())

        # 打包absmax_code
        packed = torch.zeros(
            (self.out_features, (num_blocks + 1) // 2),
            dtype=torch.uint8,
            device=absmax.device
        )

        for i in range(num_blocks):
            if i % 2 == 0:
                packed[:, i // 2] = absmax_quantized[:, i].to(torch.uint8)
            else:
                packed[:, i // 2] |= (absmax_quantized[:, i].to(torch.uint8) << 4)

        self.absmax_code.copy_(packed)


class BitsAndBytesQuantizer:
    """BitsAndBytes量化器"""

    def __init__(self, config: Optional[BitsAndBytesConfig] = None):
        self.config = config or BitsAndBytesConfig()
        self.quantized_modules: Dict[str, nn.Module] = {}

    def quantize_model(
        self,
        model: nn.Module,
        module_filter: Optional[callable] = None
    ) -> nn.Module:
        """
        量化整个模型

        Args:
            model: 待量化模型
            module_filter: 模块过滤函数

        Returns:
            量化后的模型
        """
        if module_filter is None:
            module_filter = lambda name, m: (
                isinstance(m, nn.Linear) and
                (self.config.llm_int8_skip_modules is None or
                 not any(skip in name for skip in self.config.llm_int8_skip_modules))
            )

        # 替换线性层
        for name, module in list(model.named_modules()):
            if module_filter(name, module):
                quantized_module = self._create_quantized_linear(module)

                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, quantized_module)
                else:
                    setattr(model, child_name, quantized_module)

                self.quantized_modules[name] = quantized_module

        return model

    def _create_quantized_linear(self, linear: nn.Linear) -> nn.Module:
        """从普通线性层创建量化线性层"""
        if self.config.load_in_8bit:
            quantized = Linear8bitLt(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=linear.bias is not None,
                threshold=self.config.llm_int8_threshold,
                device=linear.weight.device.type
            )
            quantized.quantize_weight(linear.weight.data)

        elif self.config.load_in_4bit:
            quantized = Linear4bit(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=linear.bias is not None,
                quant_type=self.config.bnb_4bit_quant_type,
                use_double_quant=self.config.bnb_4bit_use_double_quant,
                compute_dtype=self.config.bnb_4bit_compute_dtype,
                device=linear.weight.device.type
            )
            quantized.quantize_weight(linear.weight.data)

        else:
            return linear

        if linear.bias is not None:
            quantized.bias.copy_(linear.bias.data)

        return quantized

    def save_pretrained(self, model: nn.Module, save_path: str):
        """保存量化模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_dict = {
            'load_in_8bit': self.config.load_in_8bit,
            'load_in_4bit': self.config.load_in_4bit,
            'bnb_4bit_quant_type': self.config.bnb_4bit_quant_type,
            'bnb_4bit_use_double_quant': self.config.bnb_4bit_use_double_quant,
            'llm_int8_threshold': self.config.llm_int8_threshold,
        }

        with open(save_path / 'quant_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        # 保存模型状态
        state_dict = {}
        for name, param in model.state_dict().items():
            state_dict[name] = param.cpu()

        torch.save(state_dict, save_path / 'model.pt')


def quantize_bnb_8bit(
    model: nn.Module,
    threshold: float = 6.0,
    skip_modules: Optional[List[str]] = None
) -> nn.Module:
    """8-bit量化便捷函数"""
    config = BitsAndBytesConfig(
        load_in_8bit=True,
        load_in_4bit=False,
        llm_int8_threshold=threshold,
        llm_int8_skip_modules=skip_modules
    )
    quantizer = BitsAndBytesQuantizer(config)
    return quantizer.quantize_model(model)


def quantize_bnb_4bit(
    model: nn.Module,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: torch.dtype = torch.float16
) -> nn.Module:
    """4-bit量化便捷函数"""
    config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype
    )
    quantizer = BitsAndBytesQuantizer(config)
    return quantizer.quantize_model(model)
