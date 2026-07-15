"""
独立专家网络模块

提供专家基类和多种专家实现（MLP、CNN、Transformer），
支持专家路由接口和统一专家网络管理。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import math


class ExpertBase(nn.Module, ABC):
    """
    专家基类
    
    所有专家网络的基类，定义统一的接口。
    """
    
    def __init__(self, input_dim: int, output_dim: int, expert_id: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_id = expert_id
        self._expert_type = "base"
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, ..., input_dim)
        
        Returns:
            输出张量，形状为 (batch_size, ..., output_dim)
        """
        pass
    
    @property
    def expert_type(self) -> str:
        """获取专家类型"""
        return self._expert_type
    
    @property
    def num_parameters(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_expert_info(self) -> Dict[str, Any]:
        """获取专家信息"""
        return {
            'expert_id': self.expert_id,
            'expert_type': self.expert_type,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_parameters': self.num_parameters
        }


class MLPExpert(ExpertBase):
    """
    MLP专家网络
    
    多层感知机专家，适用于一般特征变换。
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 use_layer_norm: bool = True,
                 expert_id: Optional[int] = None):
        """
        初始化MLP专家
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度，默认为input_dim * 4
            num_layers: 层数
            activation: 激活函数类型
            dropout: Dropout概率
            use_layer_norm: 是否使用LayerNorm
            expert_id: 专家ID
        """
        super().__init__(input_dim, output_dim, expert_id)
        self._expert_type = "mlp"
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_layer_norm and i < num_layers - 1:
                self.norms.append(nn.LayerNorm(dims[i + 1]))
            
            if dropout > 0 and i < num_layers - 1:
                self.dropouts.append(nn.Dropout(dropout))
        
        # 激活函数
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Callable:
        """获取激活函数"""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
            'swish': F.silu
        }
        return activations.get(activation, F.relu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 最后一层不加激活和归一化
            if i < self.num_layers - 1:
                if self.use_layer_norm:
                    x = self.norms[i](x)
                
                x = self.activation(x)
                
                if self.dropout_rate > 0:
                    x = self.dropouts[i](x)
        
        return x


class CNNExpert(ExpertBase):
    """
    CNN专家网络
    
    卷积神经网络专家，适用于图像和空间特征处理。
    """
    
    def __init__(self,
                 input_channels: int,
                 output_dim: int,
                 hidden_channels: List[int] = None,
                 kernel_sizes: Union[int, List[int]] = 3,
                 strides: Union[int, List[int]] = 1,
                 paddings: Union[int, List[int]] = 1,
                 use_batch_norm: bool = True,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 pool_type: str = "adaptive_avg",
                 expert_id: Optional[int] = None):
        """
        初始化CNN专家
        
        Args:
            input_channels: 输入通道数
            output_dim: 输出维度
            hidden_channels: 隐藏层通道数列表
            kernel_sizes: 卷积核大小
            strides: 步长
            paddings: 填充
            use_batch_norm: 是否使用BatchNorm
            activation: 激活函数类型
            dropout: Dropout概率
            pool_type: 池化类型
            expert_id: 专家ID
        """
        # CNN专家需要特殊处理input_dim，先调用父类初始化
        super().__init__(input_channels, output_dim, expert_id)
        self._expert_type = "cnn"
        self.input_channels = input_channels
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        
        # 统一参数为列表
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(hidden_channels)
        if isinstance(strides, int):
            strides = [strides] * len(hidden_channels)
        if isinstance(paddings, int):
            paddings = [paddings] * len(hidden_channels)
        
        self.hidden_channels = hidden_channels
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        
        in_ch = input_channels
        for i, (out_ch, k, s, p) in enumerate(zip(hidden_channels, kernel_sizes, strides, paddings)):
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(out_ch))
            if dropout > 0:
                self.dropouts.append(nn.Dropout2d(dropout))
            in_ch = out_ch
        
        # 激活函数
        self.activation = self._get_activation(activation)
        
        # 池化层
        if pool_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Linear(hidden_channels[-1], output_dim)
    
    def _get_activation(self, activation: str) -> Callable:
        """获取激活函数"""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
        }
        return activations.get(activation, F.relu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 卷积层
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            
            x = self.activation(x)
            
            if self.dropout_rate > 0:
                x = self.dropouts[i](x)
        
        # 池化
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class TransformerExpert(ExpertBase):
    """
    Transformer专家网络
    
    基于Transformer架构的专家，适用于序列数据处理。
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 ff_dim: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 activation: str = "gelu",
                 expert_id: Optional[int] = None):
        """
        初始化Transformer专家
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            ff_dim: 前馈网络维度
            dropout: Dropout概率
            max_seq_len: 最大序列长度
            activation: 激活函数类型
            expert_id: 专家ID
        """
        super().__init__(input_dim, output_dim, expert_id)
        self._expert_type = "transformer"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # 输入嵌入
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, output_dim)
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 输出投影
        x = self.output_projection(x)
        
        return x


class ExpertRouter(nn.Module):
    """
    专家路由器
    
    根据输入特征动态选择激活的专家。
    """
    
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 top_k: int = 2,
                 noise_std: float = 1.0):
        """
        初始化专家路由器
        
        Args:
            input_dim: 输入维度
            num_experts: 专家数量
            top_k: 选择的专家数量
            noise_std: 路由噪声标准差
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # 路由网络
        self.router = nn.Linear(input_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        路由前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
        
        Returns:
            expert_weights: 专家权重，形状为 (batch_size, top_k)
            expert_indices: 专家索引，形状为 (batch_size, top_k)
        """
        # 计算路由logits
        router_logits = self.router(x)
        
        # 添加噪声（训练时）
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # 选择top-k专家
        expert_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=-1),
            self.top_k,
            dim=-1
        )
        
        # 归一化权重
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return expert_weights, expert_indices


class MixtureOfExperts(nn.Module):
    """
    混合专家网络 (MoE)
    
    集成多个专家和路由机制的统一框架。
    """
    
    def __init__(self,
                 experts: List[ExpertBase],
                 router: ExpertRouter,
                 aggregate_method: str = "weighted_sum"):
        """
        初始化混合专家网络
        
        Args:
            experts: 专家列表
            router: 专家路由器
            aggregate_method: 聚合方法 ("weighted_sum" 或 "concat")
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.aggregate_method = aggregate_method
        
        assert len(experts) == router.num_experts, "专家数量必须与路由器输出匹配"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        batch_size = x.size(0)
        
        # 获取路由权重和索引
        if x.dim() > 2:
            # 对于序列数据，使用第一个token或平均池化
            router_input = x.mean(dim=1) if x.dim() == 3 else x.view(batch_size, -1)
        else:
            router_input = x
        
        expert_weights, expert_indices = self.router(router_input)
        
        # 收集各专家的输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            output = expert(x)
            expert_outputs.append(output)
        
        # 聚合专家输出
        if self.aggregate_method == "weighted_sum":
            # 加权求和
            output = torch.zeros_like(expert_outputs[0])
            for i in range(self.router.top_k):
                weight = expert_weights[:, i:i+1]
                idx = expert_indices[:, i]
                for b in range(batch_size):
                    output[b] += weight[b] * expert_outputs[idx[b]][b]
        else:
            # 拼接
            selected_outputs = [expert_outputs[idx[0]] for idx in expert_indices]
            output = torch.cat(selected_outputs, dim=-1)
        
        return output
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """获取专家利用率统计"""
        utilization = {}
        for i, expert in enumerate(self.experts):
            utilization[i] = {
                'type': expert.expert_type,
                'parameters': expert.num_parameters
            }
        return utilization


# 便捷函数
def create_mlp_expert(input_dim: int, output_dim: int, **kwargs) -> MLPExpert:
    """创建MLP专家"""
    return MLPExpert(input_dim, output_dim, **kwargs)


def create_cnn_expert(input_channels: int, output_dim: int, **kwargs) -> CNNExpert:
    """创建CNN专家"""
    return CNNExpert(input_channels, output_dim, **kwargs)


def create_transformer_expert(input_dim: int, output_dim: int, **kwargs) -> TransformerExpert:
    """创建Transformer专家"""
    return TransformerExpert(input_dim, output_dim, **kwargs)


def create_moe_from_experts(experts: List[ExpertBase], 
                            input_dim: int,
                            top_k: int = 2) -> MixtureOfExperts:
    """
    从专家列表创建MoE
    
    Args:
        experts: 专家列表
        input_dim: 输入维度
        top_k: 选择的专家数量
    
    Returns:
        混合专家网络
    """
    router = ExpertRouter(input_dim, len(experts), top_k=top_k)
    return MixtureOfExperts(experts, router)
