"""
AGI统一框架 - 模型压缩与加速
实现量化、剪枝、知识蒸馏、神经架构压缩等技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from dataclasses import dataclass
import math


# ==================== 量化 ====================

@dataclass
class QuantizationConfig:
    """量化配置"""
    bits: int = 8
    quant_type: str = "symmetric"  # symmetric, asymmetric
    per_channel: bool = True
    range_learning: bool = False


class Quantizer(nn.Module):
    """基础量化器"""
    
    def __init__(self, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
            
    def quantize(self, x: torch.Tensor, scale: torch.Tensor,
                 zero_point: torch.Tensor) -> torch.Tensor:
        """量化"""
        x_q = x / scale + zero_point
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        return torch.round(x_q)
    
    def dequantize(self, x_q: torch.Tensor, scale: torch.Tensor,
                   zero_point: torch.Tensor) -> torch.Tensor:
        """反量化"""
        return (x_q - zero_point) * scale


class MinMaxObserver(nn.Module):
    """MinMax观察器"""
    
    def __init__(self, symmetric: bool = True):
        super().__init__()
        self.symmetric = symmetric
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        
    def update(self, x: torch.Tensor):
        """更新统计"""
        self.min_val = torch.min(self.min_val, x.min())
        self.max_val = torch.max(self.max_val, x.max())
        
    def calculate_qparams(self, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算量化参数"""
        if self.symmetric:
            max_val = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
            scale = max_val / (2 ** (bits - 1) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (self.max_val - self.min_val) / (2 ** bits - 1)
            zero_point = -self.min_val / scale
            
        return scale, zero_point


class QuantizedLinear(nn.Module):
    """量化线性层"""
    
    def __init__(self, in_features: int, out_features: int,
                 bits: int = 8, symmetric: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 量化权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # 量化参数
        self.weight_scale = nn.Parameter(torch.ones(out_features))
        self.weight_zero_point = nn.Parameter(torch.zeros(out_features))
        
        self.quantizer = Quantizer(bits, symmetric)
        self.observer = MinMaxObserver(symmetric)
        
        self.quantized = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            # 量化权重
            w_q = self.quantizer.quantize(
                self.weight, self.weight_scale.unsqueeze(1),
                self.weight_zero_point.unsqueeze(1)
            )
            w = self.quantizer.dequantize(
                w_q, self.weight_scale.unsqueeze(1),
                self.weight_zero_point.unsqueeze(1)
            )
        else:
            # 训练时更新观察器
            self.observer.update(self.weight)
            w = self.weight
            
        return F.linear(x, w, self.bias)
    
    def quantize_(self):
        """执行量化"""
        scale, zp = self.observer.calculate_qparams()
        self.weight_scale.data = scale
        self.weight_zero_point.data = zp
        self.quantized = True


class DynamicQuantizer:
    """动态量化"""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.quantizer = Quantizer(bits, symmetric=True)
        
    def quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """动态量化张量"""
        max_val = x.abs().max()
        scale = max_val / (2 ** (self.bits - 1) - 1)
        
        x_q = self.quantizer.quantize(x, scale, torch.zeros_like(scale))
        return x_q, scale


class QATLinear(nn.Module):
    """量化感知训练线性层"""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.bits = bits
        self.weight_scale = nn.Parameter(torch.ones(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 伪量化权重
        w = self._fake_quantize(self.weight, self.weight_scale.unsqueeze(1))
        
        # 伪量化激活
        x = self._fake_quantize_activation(x)
        
        return F.linear(x, w, self.bias)
    
    def _fake_quantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """伪量化"""
        x_q = x / scale
        x_q = torch.clamp(x_q, -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        x_q = torch.round(x_q)
        return x_q * scale
    
    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """激活伪量化"""
        scale = x.abs().max() / (2 ** (self.bits - 1) - 1) + 1e-6
        return self._fake_quantize(x, scale.expand(x.shape[-1]))


# ==================== 剪枝 ====================

class Pruner:
    """基础剪枝器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.masks: Dict[str, torch.Tensor] = {}
        
    def compute_masks(self, amount: float):
        """计算剪枝掩码 - 默认实现：基于权重幅度的全局剪枝"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 将权重展平并排序
                weights = param.data.abs().flatten()
                if weights.numel() == 0:
                    continue

                # 计算剪枝阈值：保留 (1-amount) 比例的权重
                sorted_weights, _ = torch.sort(weights)
                idx = int(amount * len(sorted_weights))
                idx = min(idx, len(sorted_weights) - 1)
                threshold = sorted_weights[idx].item()

                # 创建二值掩码：大于阈值的权重保留
                mask = (param.data.abs() > threshold).float()
                self.masks[name] = mask
    
    def apply_masks(self):
        """应用掩码"""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
                
    def get_sparsity(self) -> float:
        """获取稀疏度"""
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
                
        return zero_params / total_params if total_params > 0 else 0.0


class MagnitudePruner(Pruner):
    """幅度剪枝"""
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        
    def compute_masks(self, amount: float):
        """基于权重幅度计算掩码"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 计算阈值
                threshold = self._compute_threshold(param, amount)
                
                # 创建掩码
                mask = (param.abs() > threshold).float()
                self.masks[name] = mask
                
    def _compute_threshold(self, param: torch.Tensor, amount: float) -> float:
        """计算剪枝阈值"""
        weights = param.abs().flatten()
        sorted_weights, _ = torch.sort(weights)
        
        idx = int(amount * len(sorted_weights))
        threshold = sorted_weights[idx].item()
        
        return threshold


class GradientPruner(Pruner):
    """梯度剪枝"""
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        
    def compute_masks(self, amount: float):
        """基于梯度幅度计算掩码"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.grad is not None:
                # 使用梯度*权重的幅度
                importance = param.abs() * param.grad.abs()
                
                threshold = self._compute_threshold(importance, amount)
                mask = (importance > threshold).float()
                self.masks[name] = mask
                
    def _compute_threshold(self, importance: torch.Tensor, amount: float) -> float:
        """计算阈值"""
        flat = importance.flatten()
        sorted_vals, _ = torch.sort(flat)
        
        idx = int(amount * len(sorted_vals))
        return sorted_vals[idx].item()


class StructuredPruner(Pruner):
    """结构化剪枝（剪枝整个通道/滤波器）"""
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.pruned_indices: Dict[str, List[int]] = {}
        
    def compute_masks(self, amount: float):
        """计算结构化掩码"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # 计算每个输出通道的L1范数
                if param.dim() == 4:  # 卷积层
                    importance = param.abs().sum(dim=(1, 2, 3))
                elif param.dim() == 2:  # 线性层
                    importance = param.abs().sum(dim=1)
                else:
                    continue
                    
                # 选择要保留的通道
                num_keep = int((1 - amount) * len(importance))
                _, keep_indices = torch.topk(importance, num_keep)
                
                # 创建掩码
                mask = torch.zeros_like(param)
                if param.dim() == 4:
                    mask[keep_indices] = 1
                else:
                    mask[keep_indices] = 1
                    
                self.masks[name] = mask
                self.pruned_indices[name] = keep_indices.tolist()


class GlobalPruner(Pruner):
    """全局剪枝"""
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        
    def compute_masks(self, amount: float):
        """全局剪枝"""
        # 收集所有权重
        all_weights = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.abs().flatten())
                
        all_weights = torch.cat(all_weights)
        
        # 计算全局阈值
        sorted_weights, _ = torch.sort(all_weights)
        idx = int(amount * len(sorted_weights))
        threshold = sorted_weights[idx].item()
        
        # 创建掩码
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = (param.abs() > threshold).float()
                self.masks[name] = mask


# ==================== 知识蒸馏 ====================

class DistillationLoss(nn.Module):
    """知识蒸馏损失"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # 软标签损失
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # 组合损失
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class AttentionDistillation(nn.Module):
    """注意力蒸馏"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, student_attn: torch.Tensor,
                teacher_attn: torch.Tensor) -> torch.Tensor:
        """计算注意力蒸馏损失"""
        return F.mSE_loss(student_attn, teacher_attn)


class FeatureDistillation(nn.Module):
    """特征蒸馏"""
    
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        
        # 投影层（如果维度不匹配）
        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()
            
    def forward(self, student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> torch.Tensor:
        """计算特征蒸馏损失"""
        student_features = self.projector(student_features)
        return F.mse_loss(student_features, teacher_features)


# ==================== 模型压缩工具 ====================

def count_parameters(model: nn.Module) -> int:
    """计算参数数量"""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """计算非零参数数量"""
    return sum((p != 0).sum().item() for p in model.parameters())


def compute_compression_ratio(model: nn.Module) -> float:
    """计算压缩比"""
    total = count_parameters(model)
    nonzero = count_nonzero_parameters(model)
    return total / nonzero if nonzero > 0 else 1.0


def estimate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """估算FLOPs"""
    total_flops = 0
    
    def hook(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            batch_size = input[0].size(0)
            out_h = output.size(2)
            out_w = output.size(3)
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            flops = batch_size * out_h * out_w * module.in_channels * \
                    module.out_channels * kernel_ops / module.groups
            total_flops += flops
        elif isinstance(module, nn.Linear):
            flops = input[0].size(0) * module.in_features * module.out_features
            total_flops += flops
            
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook))
        
    # 前向传播
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)
    with torch.no_grad():
        model(dummy_input)
        
    # 移除钩子
    for h in hooks:
        h.remove()
        
    return int(total_flops)


def model_size_mb(model: nn.Module) -> float:
    """计算模型大小（MB）"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


# ==================== 神经网络二值化 ====================

class BinaryLinear(nn.Module):
    """二值线性层"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 二值化权重
        w_b = self.weight.sign()
        
        # 二值化激活
        x_b = x.sign()
        
        return F.linear(x_b, w_b, self.bias)


class TernaryLinear(nn.Module):
    """三值线性层 (+1, 0, -1)"""
    
    def __init__(self, in_features: int, out_features: int, threshold: float = 0.05):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 三值化权重
        w_t = self._ternarize(self.weight)
        
        return F.linear(x, w_t, self.bias)
    
    def _ternarize(self, w: torch.Tensor) -> torch.Tensor:
        """三值化"""
        # 计算缩放因子
        delta = self.threshold * w.abs().max()
        
        # 三值化
        w_t = torch.zeros_like(w)
        w_t[w > delta] = 1
        w_t[w < -delta] = -1
        
        # 缩放
        alpha = w.abs().sum() / (w != 0).sum()
        
        return alpha * w_t


# ==================== 低秩分解 ====================

class LowRankLinear(nn.Module):
    """低秩分解线性层"""
    
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # W ≈ U @ V
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 低秩矩阵乘法
        return F.linear(F.linear(x, self.V), self.U, self.bias)
    
    def get_full_weight(self) -> torch.Tensor:
        """获取完整权重矩阵"""
        return self.U @ self.V


def decompose_linear(layer: nn.Linear, rank: int) -> LowRankLinear:
    """将线性层分解为低秩形式"""
    # SVD分解
    U, S, V = torch.svd(layer.weight.data)
    
    # 截断
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:, :rank]
    
    # 创建低秩层
    low_rank = LowRankLinear(layer.in_features, layer.out_features, rank)
    
    low_rank.U.data = U_r @ torch.diag(S_r.sqrt())
    low_rank.V.data = torch.diag(S_r.sqrt()) @ V_r.T
    low_rank.bias.data = layer.bias.data
    
    return low_rank
