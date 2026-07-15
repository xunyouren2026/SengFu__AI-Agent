"""
专家路由门控网络模块

提供专家路由门控网络实现，包括：
- ExpertGate: 学习输入到专家的映射
- 可学习噪声：训练时添加Gumbel噪声
- 稀疏路由：只激活Top-K专家
- 重要性估计：估计每个专家的重要性

关键算法：
- 线性变换 + Softmax
- Top-K掩码：只保留前K个最大值
- Gumbel-Softmax（可选）：可微分离散采样
- 噪声调度：随训练步数衰减
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler(ABC):
    """
    噪声调度器基类
    
    用于控制训练时添加的噪声强度，随训练步数衰减。
    """
    
    def __init__(self, initial_noise: float = 1.0, min_noise: float = 0.0):
        """
        初始化噪声调度器
        
        Args:
            initial_noise: 初始噪声强度
            min_noise: 最小噪声强度
        """
        self.initial_noise = initial_noise
        self.min_noise = min_noise
        self.current_step = 0
    
    @abstractmethod
    def get_noise(self, step: Optional[int] = None) -> float:
        """
        获取当前噪声强度
        
        Args:
            step: 当前训练步数，None则使用内部计数
        
        Returns:
            当前噪声强度
        """
        pass
    
    def step(self) -> None:
        """前进一步"""
        self.current_step += 1
    
    def reset(self) -> None:
        """重置步数"""
        self.current_step = 0


class LinearNoiseScheduler(NoiseScheduler):
    """
    线性噪声调度器
    
    噪声强度随步数线性衰减。
    """
    
    def __init__(self,
                 initial_noise: float = 1.0,
                 min_noise: float = 0.0,
                 total_steps: int = 10000,
                 warmup_steps: int = 0):
        """
        初始化线性噪声调度器
        
        Args:
            initial_noise: 初始噪声强度
            min_noise: 最小噪声强度
            total_steps: 总训练步数
            warmup_steps: 预热步数（期间保持初始噪声）
        """
        super().__init__(initial_noise, min_noise)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def get_noise(self, step: Optional[int] = None) -> float:
        """获取当前噪声强度"""
        step = step if step is not None else self.current_step
        
        if step < self.warmup_steps:
            return self.initial_noise
        
        effective_step = step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps
        
        if effective_step >= effective_total:
            return self.min_noise
        
        progress = effective_step / effective_total
        noise = self.initial_noise - progress * (self.initial_noise - self.min_noise)
        return max(noise, self.min_noise)


class ExponentialNoiseScheduler(NoiseScheduler):
    """
    指数噪声调度器
    
    噪声强度随步数指数衰减。
    """
    
    def __init__(self,
                 initial_noise: float = 1.0,
                 min_noise: float = 0.0,
                 decay_rate: float = 0.9999):
        """
        初始化指数噪声调度器
        
        Args:
            initial_noise: 初始噪声强度
            min_noise: 最小噪声强度
            decay_rate: 衰减率
        """
        super().__init__(initial_noise, min_noise)
        self.decay_rate = decay_rate
    
    def get_noise(self, step: Optional[int] = None) -> float:
        """获取当前噪声强度"""
        step = step if step is not None else self.current_step
        noise = self.initial_noise * (self.decay_rate ** step)
        return max(noise, self.min_noise)


class CosineNoiseScheduler(NoiseScheduler):
    """
    余弦噪声调度器
    
    噪声强度随步数按余弦曲线衰减。
    """
    
    def __init__(self,
                 initial_noise: float = 1.0,
                 min_noise: float = 0.0,
                 total_steps: int = 10000):
        """
        初始化余弦噪声调度器
        
        Args:
            initial_noise: 初始噪声强度
            min_noise: 最小噪声强度
            total_steps: 总训练步数
        """
        super().__init__(initial_noise, min_noise)
        self.total_steps = total_steps
    
    def get_noise(self, step: Optional[int] = None) -> float:
        """获取当前噪声强度"""
        step = step if step is not None else self.current_step
        
        if step >= self.total_steps:
            return self.min_noise
        
        progress = step / self.total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        noise = self.min_noise + (self.initial_noise - self.min_noise) * cosine_decay
        return noise


class ExpertGate(nn.Module):
    """
    专家路由门控网络
    
    学习输入到专家的映射，支持：
    - Top-K稀疏路由
    - Gumbel噪声（训练时）
    - 噪声调度
    - 重要性估计
    
    Attributes:
        input_dim: 输入维度
        num_experts: 专家数量
        top_k: 选择的Top-K专家数
        use_gumbel: 是否使用Gumbel-Softmax
        temperature: Gumbel-Softmax温度参数
    """
    
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 top_k: int = 2,
                 use_gumbel: bool = False,
                 temperature: float = 1.0,
                 noise_scheduler: Optional[NoiseScheduler] = None,
                 use_bias: bool = True,
                 dropout: float = 0.0):
        """
        初始化专家门控网络
        
        Args:
            input_dim: 输入特征维度
            num_experts: 专家数量
            top_k: 每次前向传播选择的专家数量
            use_gumbel: 是否使用Gumbel-Softmax进行可微分离散采样
            temperature: Gumbel-Softmax温度参数（越低越接近one-hot）
            noise_scheduler: 噪声调度器，None则使用固定噪声
            use_bias: 门控线性层是否使用偏置
            dropout: 输入dropout概率
        
        Raises:
            ValueError: 当参数不合法时
        """
        super().__init__()
        
        if input_dim <= 0:
            raise ValueError(f"input_dim必须为正数，得到{input_dim}")
        if num_experts <= 0:
            raise ValueError(f"num_experts必须为正数，得到{num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k必须在[1, {num_experts}]范围内，得到{top_k}")
        if temperature <= 0:
            raise ValueError(f"temperature必须为正数，得到{temperature}")
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.use_bias = use_bias
        
        # 门控线性层
        self.gate = nn.Linear(input_dim, num_experts, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 噪声调度器
        self.noise_scheduler = noise_scheduler
        
        # 统计信息
        self.register_buffer('total_tokens', torch.tensor(0, dtype=torch.long))
        self.register_buffer('expert_counts', torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer('accumulated_importance', torch.zeros(num_experts))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化权重"""
        # 使用较小的初始值，使初始路由更均匀
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        if self.use_bias and self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
    
    def _add_gumbel_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """
        添加Gumbel噪声
        
        Args:
            logits: 原始logits
        
        Returns:
            添加噪声后的logits
        """
        # Gumbel(0, 1)噪声: -log(-log(U))，U ~ Uniform(0, 1)
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)
        return logits + gumbel_noise
    
    def _gumbel_softmax(self,
                       logits: torch.Tensor,
                       hard: bool = False) -> torch.Tensor:
        """
        Gumbel-Softmax采样
        
        Args:
            logits: 原始logits
            hard: 是否返回硬离散样本（前向时one-hot，反向时soft）
        
        Returns:
            Gumbel-Softmax采样结果
        """
        gumbel_logits = self._add_gumbel_noise(logits)
        soft_probs = F.softmax(gumbel_logits / self.temperature, dim=-1)
        
        if hard:
            # 硬采样：前向用argmax，反向用soft
            hard_probs = torch.zeros_like(soft_probs)
            hard_probs.scatter_(-1, soft_probs.argmax(dim=-1, keepdim=True), 1.0)
            # 使用straight-through estimator
            probs = hard_probs - soft_probs.detach() + soft_probs
        else:
            probs = soft_probs
        
        return probs
    
    def _compute_top_k_mask(self,
                           logits: torch.Tensor,
                           top_k: Optional[int] = None) -> torch.Tensor:
        """
        计算Top-K掩码
        
        Args:
            logits: 原始logits
            top_k: 选择的专家数，None则使用self.top_k
        
        Returns:
            Top-K掩码，形状与logits相同，Top-K位置为1，其余为0
        """
        k = top_k if top_k is not None else self.top_k
        
        # 获取Top-K索引
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # 创建掩码
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_k_indices, 1.0)
        
        return mask
    
    def forward(self,
                x: torch.Tensor,
                return_importance: bool = False) -> Union[
                    Tuple[torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, ..., input_dim)
            return_importance: 是否返回重要性估计
        
        Returns:
            如果return_importance为False:
                - expert_weights: 专家权重，形状为 (batch_size, ..., num_experts)
                - expert_indices: 选中的专家索引，形状为 (batch_size, ..., top_k)
            如果return_importance为True:
                - expert_weights: 专家权重
                - expert_indices: 选中的专家索引
                - importance: 专家重要性估计，形状为 (num_experts,)
        
        Raises:
            ValueError: 当输入维度不匹配时
        """
        # 检查输入维度
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入维度不匹配: 期望{self.input_dim}, 得到{x.shape[-1]}"
            )
        
        original_shape = x.shape[:-1]
        x_flat = x.view(-1, self.input_dim)
        batch_size = x_flat.size(0)
        
        # Dropout
        if self.dropout is not None:
            x_flat = self.dropout(x_flat)
        
        # 计算门控logits
        logits = self.gate(x_flat)  # (batch_size, num_experts)
        
        # 添加噪声（训练时）
        if self.training:
            if self.noise_scheduler is not None:
                noise_std = self.noise_scheduler.get_noise()
                self.noise_scheduler.step()
            else:
                noise_std = 1.0
            
            if noise_std > 0:
                noise = torch.randn_like(logits) * noise_std
                logits = logits + noise
        
        # 计算路由概率
        if self.use_gumbel and self.training:
            router_probs = self._gumbel_softmax(logits, hard=True)
        else:
            router_probs = F.softmax(logits, dim=-1)
        
        # Top-K选择
        top_k_mask = self._compute_top_k_mask(logits, self.top_k)
        
        # 应用Top-K掩码
        masked_probs = router_probs * top_k_mask
        
        # 重新归一化
        weights_sum = masked_probs.sum(dim=-1, keepdim=True)
        expert_weights = masked_probs / (weights_sum + 1e-10)
        
        # 获取选中的专家索引
        _, expert_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # 更新统计信息
        if self.training:
            self.total_tokens += batch_size
            for i in range(self.num_experts):
                count = (expert_indices == i).sum()
                self.expert_counts[i] += count
            
            # 累积重要性（使用路由概率的均值）
            self.accumulated_importance += router_probs.mean(dim=0).detach()
        
        # 恢复原始形状
        expert_weights = expert_weights.view(*original_shape, self.num_experts)
        expert_indices = expert_indices.view(*original_shape, self.top_k)
        
        if return_importance:
            importance = self.get_expert_importance()
            return expert_weights, expert_indices, importance
        
        return expert_weights, expert_indices
    
    def get_expert_importance(self) -> torch.Tensor:
        """
        获取专家重要性估计
        
        重要性基于专家被选中的频率和路由概率。
        
        Returns:
            专家重要性，形状为 (num_experts,)
        """
        if self.total_tokens == 0:
            return torch.ones(self.num_experts) / self.num_experts
        
        # 使用累积的路由概率作为重要性
        importance = self.accumulated_importance / (self.accumulated_importance.sum() + 1e-10)
        return importance
    
    def get_expert_utilization(self) -> torch.Tensor:
        """
        获取专家利用率
        
        Returns:
            每个专家的利用率，形状为 (num_experts,)
        """
        if self.total_tokens == 0:
            return torch.zeros(self.num_experts)
        
        total_selections = self.total_tokens * self.top_k
        utilization = self.expert_counts.float() / total_selections
        return utilization
    
    def compute_load_balancing_loss(self) -> torch.Tensor:
        """
        计算负载均衡辅助损失
        
        鼓励均匀使用所有专家。
        
        Returns:
            负载均衡损失
        """
        utilization = self.get_expert_utilization()
        
        # 理想均匀分布
        uniform = 1.0 / self.num_experts
        
        # L2距离
        loss = ((utilization - uniform) ** 2).sum()
        
        return loss
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取门控网络统计信息
        
        Returns:
            包含统计信息的字典
        """
        importance = self.get_expert_importance()
        utilization = self.get_expert_utilization()
        
        stats = {
            'total_tokens': self.total_tokens.item(),
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'expert_counts': self.expert_counts.tolist(),
            'expert_importance': importance.tolist(),
            'expert_utilization': utilization.tolist(),
            'temperature': self.temperature,
            'use_gumbel': self.use_gumbel,
        }
        
        # 计算负载均衡损失
        stats['load_balancing_loss'] = self.compute_load_balancing_loss().item()
        
        # 计算熵（路由不确定性）
        if self.total_tokens > 0:
            entropy = -(utilization * torch.log(utilization + 1e-10)).sum()
            stats['routing_entropy'] = entropy.item()
        
        # 噪声调度器状态
        if self.noise_scheduler is not None:
            stats['current_noise'] = self.noise_scheduler.get_noise()
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.total_tokens.zero_()
        self.expert_counts.zero_()
        self.accumulated_importance.zero_()
        if self.noise_scheduler is not None:
            self.noise_scheduler.reset()
    
    def set_temperature(self, temperature: float) -> None:
        """
        设置Gumbel-Softmax温度
        
        Args:
            temperature: 新的温度值
        
        Raises:
            ValueError: 当温度值不合法时
        """
        if temperature <= 0:
            raise ValueError(f"temperature必须为正数，得到{temperature}")
        self.temperature = temperature
    
    def get_gate_weights(self) -> torch.Tensor:
        """
        获取门控权重
        
        Returns:
            门控线性层权重
        """
        return self.gate.weight.data.clone()
    
    def set_gate_weights(self, weights: torch.Tensor) -> None:
        """
        设置门控权重
        
        Args:
            weights: 新的权重
        
        Raises:
            ValueError: 当权重形状不匹配时
        """
        if weights.shape != self.gate.weight.shape:
            raise ValueError(
                f"权重形状不匹配: 期望{self.gate.weight.shape}, 得到{weights.shape}"
            )
        self.gate.weight.data.copy_(weights)


class MultiHeadExpertGate(nn.Module):
    """
    多专家门控网络
    
    使用多个门控头，每个头负责不同的输入子空间或任务。
    """
    
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 num_heads: int = 4,
                 top_k: int = 2,
                 head_dim: Optional[int] = None,
                 use_gumbel: bool = False,
                 temperature: float = 1.0,
                 dropout: float = 0.0):
        """
        初始化多专家门控网络
        
        Args:
            input_dim: 输入维度
            num_experts: 专家数量
            num_heads: 门控头数量
            top_k: 每个头选择的专家数
            head_dim: 每个头的维度，None则自动计算
            use_gumbel: 是否使用Gumbel-Softmax
            temperature: Gumbel-Softmax温度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.top_k = top_k
        
        if head_dim is None:
            head_dim = input_dim // num_heads
        self.head_dim = head_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, num_heads * head_dim)
        
        # 多个门控头
        self.gates = nn.ModuleList([
            ExpertGate(
                head_dim,
                num_experts,
                top_k=top_k,
                use_gumbel=use_gumbel,
                temperature=temperature,
                dropout=dropout
            )
            for _ in range(num_heads)
        ])
        
        # 头聚合权重
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
    
    def forward(self,
                x: torch.Tensor,
                return_importance: bool = False) -> Union[
                    Tuple[torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
        """
        前向传播
        
        Args:
            x: 输入张量
            return_importance: 是否返回重要性
        
        Returns:
            聚合后的专家权重、索引和重要性（可选）
        """
        batch_size = x.size(0)
        
        # 投影到多个头
        projected = self.input_projection(x)
        projected = projected.view(batch_size, self.num_heads, self.head_dim)
        
        # 每个头计算路由
        all_weights = []
        all_indices = []
        all_importance = []
        
        for i, gate in enumerate(self.gates):
            head_input = projected[:, i, :]
            if return_importance:
                weights, indices, importance = gate(
                    head_input,
                    return_importance=True
                )
                all_importance.append(importance)
            else:
                weights, indices = gate(head_input)
            all_weights.append(weights)
            all_indices.append(indices)
        
        # 聚合权重（加权平均）
        head_weights = F.softmax(self.head_weights, dim=0)
        aggregated_weights = sum(
            w * hw for w, hw in zip(all_weights, head_weights)
        )
        
        # 聚合索引（取并集）
        aggregated_indices = torch.cat(all_indices, dim=-1)
        # 去重并取前top_k
        unique_indices = torch.unique(aggregated_indices, dim=-1)
        if unique_indices.size(-1) > self.top_k:
            aggregated_indices = unique_indices[..., :self.top_k]
        else:
            # 填充
            padding = torch.zeros(
                *unique_indices.shape[:-1],
                self.top_k - unique_indices.size(-1),
                dtype=unique_indices.dtype,
                device=unique_indices.device
            )
            aggregated_indices = torch.cat([unique_indices, padding], dim=-1)
        
        if return_importance:
            aggregated_importance = sum(
                imp * hw for imp, hw in zip(all_importance, head_weights)
            )
            return aggregated_weights, aggregated_indices, aggregated_importance
        
        return aggregated_weights, aggregated_indices
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'head_weights': F.softmax(self.head_weights, dim=0).tolist(),
            'per_head_stats': [gate.get_stats() for gate in self.gates]
        }
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        for gate in self.gates:
            gate.reset_stats()


def create_gate(input_dim: int,
                num_experts: int,
                top_k: int = 2,
                noise_type: str = "none",
                **kwargs) -> ExpertGate:
    """
    创建门控网络的便捷函数
    
    Args:
        input_dim: 输入维度
        num_experts: 专家数量
        top_k: 选择的专家数
        noise_type: 噪声类型 ("none", "linear", "exponential", "cosine")
        **kwargs: 其他参数
    
    Returns:
        配置好的ExpertGate实例
    """
    noise_scheduler = None
    
    if noise_type == "linear":
        noise_scheduler = LinearNoiseScheduler(
            initial_noise=kwargs.get('initial_noise', 1.0),
            min_noise=kwargs.get('min_noise', 0.0),
            total_steps=kwargs.get('total_steps', 10000),
            warmup_steps=kwargs.get('warmup_steps', 0)
        )
    elif noise_type == "exponential":
        noise_scheduler = ExponentialNoiseScheduler(
            initial_noise=kwargs.get('initial_noise', 1.0),
            min_noise=kwargs.get('min_noise', 0.0),
            decay_rate=kwargs.get('decay_rate', 0.9999)
        )
    elif noise_type == "cosine":
        noise_scheduler = CosineNoiseScheduler(
            initial_noise=kwargs.get('initial_noise', 1.0),
            min_noise=kwargs.get('min_noise', 0.0),
            total_steps=kwargs.get('total_steps', 10000)
        )
    
    return ExpertGate(
        input_dim=input_dim,
        num_experts=num_experts,
        top_k=top_k,
        noise_scheduler=noise_scheduler,
        use_gumbel=kwargs.get('use_gumbel', False),
        temperature=kwargs.get('temperature', 1.0),
        use_bias=kwargs.get('use_bias', True),
        dropout=kwargs.get('dropout', 0.0)
    )
