"""
AGI统一框架 - 神经网络层
实现各种神经网络层：卷积、池化、归一化、激活函数等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
import numpy as np


# ==================== 卷积层 ====================

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = False):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GhostConv(nn.Module):
    """Ghost卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 ratio: int = 2, dw_size: int = 3):
        super().__init__()
        
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        
        self.primary_conv = nn.Conv2d(
            in_channels, init_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        
        self.cheap_operation = nn.Conv2d(
            init_channels, new_channels, dw_size,
            stride=1, padding=dw_size // 2, groups=init_channels, bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return self.bn(out)


class DynamicConv(nn.Module):
    """动态卷积"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, num_experts: int = 4,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        
        # 专家卷积核
        self.weight = nn.Parameter(torch.randn(
            num_experts, out_channels, in_channels, kernel_size, kernel_size
        ))
        
        # 路由网络
        self.router = nn.Linear(in_channels, num_experts)
        
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.size()
        
        # 计算路由权重
        x_pooled = F.adaptive_avg_pool2d(x, 1).view(batch_size, -1)
        routing_weights = F.softmax(self.router(x_pooled), dim=1)
        
        # 组合专家权重
        weight = torch.einsum('bk,koihw->boihw', routing_weights, self.weight)
        
        # 应用卷积
        x = x.view(1, batch_size * self.in_channels, h, w)
        weight = weight.view(batch_size * self.out_channels, self.in_channels,
                            self.kernel_size, self.kernel_size)
        
        out = F.conv2d(x, weight, stride=self.stride, 
                      padding=self.padding, groups=batch_size)
        
        return out.view(batch_size, self.out_channels, -1, w)


class DeformableConv(nn.Module):
    """可变形卷积 (简化版)"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 偏移量预测
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=3, padding=1
        )
        
        # 主卷积
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 预测偏移量
        offset = self.offset_conv(x)
        
        # 应用可变形卷积 (简化实现)
        # 实际实现需要使用CUDA kernel或torchvision.ops.deform_conv2d
        return self.conv(x)


# ==================== 池化层 ====================

class MixedPooling(nn.Module):
    """混合池化"""
    
    def __init__(self, kernel_size: int = 2, stride: int = 2, 
                 alpha: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride)
        avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride)
        return self.alpha * max_pool + (1 - self.alpha) * avg_pool


class SpatialPyramidPooling(nn.Module):
    """空间金字塔池化"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 pool_sizes: List[int] = [1, 2, 4]):
        super().__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        
        features = [x]
        for stage in self.stages:
            features.append(F.interpolate(stage(x), size=(h, w), mode='bilinear'))
            
        return torch.cat(features, dim=1)


class AttentionPooling(nn.Module):
    """注意力池化"""
    
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x)
        return (x * attn).sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)


# ==================== 归一化层 ====================

class LayerNorm(nn.Module):
    """层归一化"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


class GroupNorm(nn.Module):
    """组归一化"""
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class InstanceNorm(nn.Module):
    """实例归一化"""
    
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
            
        return x


class RMSNorm(nn.Module):
    """RMS归一化"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AdaptiveInstanceNorm(nn.Module):
    """自适应实例归一化"""
    
    def __init__(self, style_dim: int, num_channels: int):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        
        self.style_scale = nn.Linear(style_dim, num_channels)
        self.style_bias = nn.Linear(style_dim, num_channels)
        
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        
        scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)
        
        return scale * x + bias


# ==================== 激活函数 ====================

class Swish(nn.Module):
    """Swish激活函数"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish激活函数"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class GELU(nn.Module):
    """高斯误差线性单元"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class HardSwish(nn.Module):
    """Hard Swish"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3) / 6


class SELU(nn.Module):
    """SELU激活函数"""
    
    def __init__(self):
        super().__init__()
        self.alpha = 1.67326324235437728481704
        self.scale = 1.05070098735548049341933
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.elu(x, self.alpha)


# ==================== 注意力模块 ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca = self.channel_attention(x).unsqueeze(2).unsqueeze(3)
        x = x * ca
        
        # 空间注意力
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        
        return x


class ECA(nn.Module):
    """Efficient Channel Attention"""
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        
        # 自适应核大小
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


# ==================== 残差块 ====================

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """瓶颈块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, expansion: int = 4):
        super().__init__()
        
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out


class InvertedResidual(nn.Module):
    """倒残差块 (MobileNetV2)"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
            
        layers.extend([
            # 深度卷积
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 逐点卷积
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


# ==================== 密集连接 ====================

class DenseLayer(nn.Module):
    """密集层"""
    
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        return out


class DenseBlock(nn.Module):
    """密集块"""
    
    def __init__(self, num_layers: int, in_channels: int, 
                 growth_rate: int, bn_size: int = 4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    """过渡层"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x
