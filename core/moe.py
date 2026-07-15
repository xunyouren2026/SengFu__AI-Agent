"""
混合专家网络（MoE）主实现模块

提供完整的MoE（Mixture of Experts）实现，包括：
- MixtureOfExperts类：管理多个专家网络
- 专家路由：根据输入动态选择Top-K专家
- 负载均衡损失：防止专家坍缩
- 专家容量限制：防止单个专家过载
- 支持不同专家类型（MLP、CNN、Transformer）

关键算法：
- 门控网络输出路由概率
- Top-K选择 + 噪声注入（训练时）
- 负载均衡辅助损失：loss_aux = α * Σ(f_i * P_i)
- 专家容量：capacity = (total_tokens / num_experts) * capacity_factor
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert import ExpertBase, MLPExpert, CNNExpert, TransformerExpert
from .gate import ExpertGate, create_gate


class ExpertCapacityLimiter:
    """
    专家容量限制器
    
    防止单个专家过载，通过设置容量上限并处理溢出token。
    """
    
    def __init__(self,
                 num_experts: int,
                 capacity_factor: float = 1.0,
                 drop_tokens: bool = False):
        """
        初始化容量限制器
        
        Args:
            num_experts: 专家数量
            capacity_factor: 容量因子，控制每个专家的容量
            drop_tokens: 是否丢弃溢出token，False则使用备用专家
        """
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # 溢出统计
        self.overflow_count = 0
        self.total_tokens = 0
    
    def compute_capacity(self, num_tokens: int) -> int:
        """
        计算每个专家的容量
        
        Args:
            num_tokens: 总token数
        
        Returns:
            每个专家的容量
        """
        base_capacity = num_tokens / self.num_experts
        capacity = int(base_capacity * self.capacity_factor)
        return max(capacity, 1)  # 至少为1
    
    def apply_capacity_constraint(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用容量约束
        
        Args:
            expert_indices: 专家索引，形状为 (batch_size, top_k)
            expert_weights: 专家权重，形状为 (batch_size, num_experts) 或 (batch_size, top_k)
            capacity: 每个专家的容量
        
        Returns:
            - constrained_indices: 约束后的专家索引
            - constrained_weights: 约束后的专家权重
            - overflow_mask: 溢出掩码，标记哪些token被溢出处理
        """
        batch_size, top_k = expert_indices.shape
        device = expert_indices.device
        
        # 展平处理
        flat_indices = expert_indices.view(-1)
        
        # 处理不同形状的expert_weights
        if expert_weights.dim() == 2 and expert_weights.shape[1] == self.num_experts:
            # 从完整权重中提取选中的top_k权重
            flat_weights = torch.zeros(batch_size * top_k, device=device)
            for i in range(batch_size):
                for j in range(top_k):
                    expert_id = expert_indices[i, j]
                    flat_weights[i * top_k + j] = expert_weights[i, expert_id]
        else:
            flat_weights = expert_weights.view(-1)
        
        # 为每个token创建唯一标识
        token_ids = torch.arange(batch_size, device=device).repeat_interleave(top_k)
        position_ids = torch.arange(top_k, device=device).repeat(batch_size)
        
        # 按专家分组统计
        constrained_indices = torch.full_like(flat_indices, -1)
        constrained_weights = torch.zeros_like(flat_weights)
        overflow_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for expert_id in range(self.num_experts):
            # 找到分配给该专家的所有token
            mask = flat_indices == expert_id
            expert_token_ids = token_ids[mask]
            expert_position_ids = position_ids[mask]
            expert_weights_list = flat_weights[mask]
            
            if len(expert_token_ids) == 0:
                continue
            
            # 按权重排序，优先处理权重高的
            sorted_indices = torch.argsort(expert_weights_list, descending=True)
            
            # 应用容量限制
            if len(sorted_indices) > capacity:
                # 标记溢出的token
                overflow_token_ids = expert_token_ids[sorted_indices[capacity:]]
                for tid in overflow_token_ids.unique():
                    overflow_mask[tid] = True
                
                # 只保留容量内的
                kept_indices = sorted_indices[:capacity]
            else:
                kept_indices = sorted_indices
            
            # 更新约束后的索引和权重
            for idx in kept_indices:
                token_idx = expert_token_ids[idx]
                pos_idx = expert_position_ids[idx]
                flat_idx = token_idx * top_k + pos_idx
                constrained_indices[flat_idx] = expert_id
                constrained_weights[flat_idx] = expert_weights_list[idx]
        
        # 恢复形状
        constrained_indices = constrained_indices.view(batch_size, top_k)
        constrained_weights = constrained_weights.view(batch_size, top_k)
        
        # 归一化权重
        weight_sum = constrained_weights.sum(dim=-1, keepdim=True)
        constrained_weights = constrained_weights / (weight_sum + 1e-10)
        
        # 更新统计
        self.total_tokens += batch_size
        self.overflow_count += overflow_mask.sum().item()
        
        return constrained_indices, constrained_weights, overflow_mask
    
    def get_overflow_rate(self) -> float:
        """
        获取溢出率
        
        Returns:
            溢出token比例
        """
        if self.total_tokens == 0:
            return 0.0
        return self.overflow_count / self.total_tokens
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.overflow_count = 0
        self.total_tokens = 0


class MixtureOfExperts(nn.Module):
    """
    混合专家网络（MoE）主类
    
    管理多个专家网络，根据输入动态选择Top-K专家进行计算。
    支持负载均衡损失和专家容量限制。
    
    Attributes:
        num_experts: 专家数量
        top_k: 每次选择的专家数
        input_dim: 输入维度
        output_dim: 输出维度
        capacity_factor: 容量因子
        aux_loss_coef: 辅助损失系数
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 expert_type: str = "mlp",
                 expert_config: Optional[Dict[str, Any]] = None,
                 gate_config: Optional[Dict[str, Any]] = None,
                 capacity_factor: float = 1.25,
                 aux_loss_coef: float = 0.01,
                 use_capacity_limit: bool = True,
                 aggregate_method: str = "weighted_sum"):
        """
        初始化混合专家网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_experts: 专家数量
            top_k: 每次前向传播选择的专家数
            expert_type: 专家类型 ("mlp", "cnn", "transformer", "custom")
            expert_config: 专家配置字典
            gate_config: 门控网络配置字典
            capacity_factor: 容量因子，控制每个专家的最大token数
            aux_loss_coef: 负载均衡辅助损失系数
            use_capacity_limit: 是否使用容量限制
            aggregate_method: 输出聚合方法 ("weighted_sum", "mean", "max")
        
        Raises:
            ValueError: 当参数不合法时
        """
        super().__init__()
        
        if input_dim <= 0:
            raise ValueError(f"input_dim必须为正数，得到{input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim必须为正数，得到{output_dim}")
        if num_experts <= 0:
            raise ValueError(f"num_experts必须为正数，得到{num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k必须在[1, {num_experts}]范围内，得到{top_k}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.use_capacity_limit = use_capacity_limit
        self.aggregate_method = aggregate_method
        self.expert_type = expert_type
        
        # 专家配置
        expert_config = expert_config or {}
        
        # 创建专家
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = self._create_expert(expert_type, input_dim, output_dim, expert_config, i)
            self.experts.append(expert)
        
        # 创建门控网络
        gate_config = gate_config or {}
        self.gate = create_gate(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            **gate_config
        )
        
        # 容量限制器
        if use_capacity_limit:
            self.capacity_limiter = ExpertCapacityLimiter(
                num_experts=num_experts,
                capacity_factor=capacity_factor
            )
        else:
            self.capacity_limiter = None
        
        # 统计信息
        self.register_buffer('forward_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_aux_loss', torch.tensor(0.0))
    
    def _create_expert(self,
                      expert_type: str,
                      input_dim: int,
                      output_dim: int,
                      config: Dict[str, Any],
                      expert_id: int) -> ExpertBase:
        """
        创建专家实例
        
        Args:
            expert_type: 专家类型
            input_dim: 输入维度
            output_dim: 输出维度
            config: 专家配置
            expert_id: 专家ID
        
        Returns:
            专家实例
        
        Raises:
            ValueError: 当专家类型不支持时
        """
        if expert_type == "mlp":
            return MLPExpert(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.get('hidden_dim'),
                num_layers=config.get('num_layers', 2),
                activation=config.get('activation', 'relu'),
                dropout=config.get('dropout', 0.0),
                use_layer_norm=config.get('use_layer_norm', True),
                expert_id=expert_id
            )
        elif expert_type == "cnn":
            return CNNExpert(
                input_channels=config.get('input_channels', input_dim),
                output_dim=output_dim,
                hidden_channels=config.get('hidden_channels', [64, 128, 256]),
                kernel_sizes=config.get('kernel_sizes', 3),
                use_batch_norm=config.get('use_batch_norm', True),
                activation=config.get('activation', 'relu'),
                dropout=config.get('dropout', 0.0),
                expert_id=expert_id
            )
        elif expert_type == "transformer":
            return TransformerExpert(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.get('hidden_dim', 512),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 6),
                ff_dim=config.get('ff_dim', 2048),
                dropout=config.get('dropout', 0.1),
                max_seq_len=config.get('max_seq_len', 512),
                activation=config.get('activation', 'gelu'),
                expert_id=expert_id
            )
        else:
            raise ValueError(f"不支持的专家类型: {expert_type}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, ..., input_dim)
        
        Returns:
            - output: 输出张量，形状为 (batch_size, ..., output_dim)
            - aux_outputs: 辅助输出字典，包含路由信息和损失
        
        Raises:
            ValueError: 当输入维度不匹配时
        """
        # 检查输入维度
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入维度不匹配: 期望{self.input_dim}, 得到{x.shape[-1]}"
            )
        
        original_shape = x.shape[:-1]
        batch_size = x.view(-1, self.input_dim).size(0)
        
        # 准备路由输入（取平均或第一个token）
        if x.dim() == 3:  # 序列数据 (batch, seq, dim)
            router_input = x.mean(dim=1)
        elif x.dim() == 2:  # 普通数据 (batch, dim)
            router_input = x
        else:
            router_input = x.view(batch_size, -1).mean(dim=-1, keepdim=True)
            router_input = router_input.expand(-1, self.input_dim)
        
        # 获取路由决策
        expert_weights, expert_indices = self.gate(router_input)
        
        # 应用容量限制
        overflow_mask = None
        if self.use_capacity_limit and self.capacity_limiter is not None:
            capacity = self.capacity_limiter.compute_capacity(batch_size)
            expert_indices, expert_weights, overflow_mask = \
                self.capacity_limiter.apply_capacity_constraint(
                    expert_indices, expert_weights, capacity
                )
        
        # 收集专家输出
        expert_outputs = self._compute_expert_outputs(x, expert_indices, expert_weights)
        
        # 聚合输出
        output = self._aggregate_outputs(expert_outputs, expert_weights, expert_indices)
        
        # 恢复原始形状
        output = output.view(*original_shape, self.output_dim)
        
        # 计算辅助损失
        aux_loss = self._compute_aux_loss(expert_weights, expert_indices, batch_size)
        
        # 更新统计
        self.forward_count += 1
        self.total_aux_loss += aux_loss.detach()
        
        # 构建辅助输出
        aux_outputs = {
            'expert_weights': expert_weights,
            'expert_indices': expert_indices,
            'aux_loss': aux_loss,
            'overflow_mask': overflow_mask,
        }
        
        return output, aux_outputs
    
    def _compute_expert_outputs(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        计算各专家的输出
        
        Args:
            x: 输入张量
            expert_indices: 专家索引
            expert_weights: 专家权重
        
        Returns:
            专家输出字典，key为专家ID
        """
        batch_size = x.size(0)
        expert_outputs = {}
        
        # 确定哪些专家被使用
        used_experts = expert_indices.unique().tolist()
        
        for expert_id in used_experts:
            if expert_id < 0:  # 跳过无效索引
                continue
            
            expert = self.experts[expert_id]
            output = expert(x)
            expert_outputs[expert_id] = output
        
        return expert_outputs
    
    def _aggregate_outputs(
        self,
        expert_outputs: Dict[int, torch.Tensor],
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        聚合专家输出
        
        Args:
            expert_outputs: 专家输出字典
            expert_weights: 专家权重
            expert_indices: 专家索引
        
        Returns:
            聚合后的输出
        """
        batch_size = expert_weights.size(0)
        device = expert_weights.device
        
        # 获取输出形状
        first_output = next(iter(expert_outputs.values()))
        output_shape = first_output.shape[1:]
        
        if self.aggregate_method == "weighted_sum":
            # 加权求和
            output = torch.zeros(batch_size, *output_shape, device=device)
            
            for i in range(self.top_k):
                weight = expert_weights[:, i:i+1]
                idx = expert_indices[:, i]
                
                for b in range(batch_size):
                    expert_id = idx[b].item()
                    if expert_id in expert_outputs:
                        output[b] += weight[b] * expert_outputs[expert_id][b]
        
        elif self.aggregate_method == "mean":
            # 平均
            output = torch.zeros(batch_size, *output_shape, device=device)
            count = torch.zeros(batch_size, 1, device=device)
            
            for i in range(self.top_k):
                idx = expert_indices[:, i]
                for b in range(batch_size):
                    expert_id = idx[b].item()
                    if expert_id in expert_outputs:
                        output[b] += expert_outputs[expert_id][b]
                        count[b] += 1
            
            output = output / (count.unsqueeze(-1) + 1e-10)
        
        elif self.aggregate_method == "max":
            # 最大值
            outputs_list = []
            
            for i in range(self.top_k):
                idx = expert_indices[:, i]
                batch_outputs = []
                for b in range(batch_size):
                    expert_id = idx[b].item()
                    if expert_id in expert_outputs:
                        batch_outputs.append(expert_outputs[expert_id][b:b+1])
                    else:
                        batch_outputs.append(torch.zeros(1, *output_shape, device=device))
                outputs_list.append(torch.cat(batch_outputs, dim=0))
            
            output = torch.stack(outputs_list, dim=0).max(dim=0)[0]
        
        else:
            raise ValueError(f"不支持的聚合方法: {self.aggregate_method}")
        
        return output
    
    def _compute_aux_loss(
        self,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        计算辅助损失（负载均衡损失）
        
        基于Switch Transformer的负载均衡损失：
        loss = α * N * Σ(f_i * P_i)
        其中f_i是专家i的选中频率，P_i是平均路由概率
        
        Args:
            expert_weights: 专家权重
            expert_indices: 专家索引
            batch_size: 批次大小
        
        Returns:
            辅助损失
        """
        device = expert_weights.device
        
        # 计算每个专家的选中频率 f_i
        expert_counts = torch.zeros(self.num_experts, device=device)
        for i in range(self.num_experts):
            expert_counts[i] = (expert_indices == i).float().sum()
        
        # 归一化频率
        f = expert_counts / (batch_size * self.top_k + 1e-10)
        
        # 计算平均路由概率 P_i
        P = torch.zeros(self.num_experts, device=device)
        for i in range(self.top_k):
            for j in range(self.num_experts):
                mask = expert_indices[:, i] == j
                if mask.any():
                    P[j] += expert_weights[mask, i].mean()
        P = P / (self.top_k + 1e-10)
        
        # 负载均衡损失
        aux_loss = self.num_experts * (f * P).sum()
        
        return self.aux_loss_coef * aux_loss
    
    def get_aux_loss(self) -> torch.Tensor:
        """
        获取累积的辅助损失
        
        Returns:
            平均辅助损失
        """
        if self.forward_count == 0:
            return torch.tensor(0.0)
        return self.total_aux_loss / self.forward_count.item()
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """
        获取专家统计信息
        
        Returns:
            包含专家统计信息的字典
        """
        stats = {
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'expert_type': self.expert_type,
            'forward_count': self.forward_count.item(),
            'aux_loss': self.get_aux_loss().item(),
            'gate_stats': self.gate.get_stats(),
        }
        
        # 专家参数统计
        expert_params = []
        for expert in self.experts:
            expert_params.append({
                'expert_id': expert.expert_id,
                'expert_type': expert.expert_type,
                'num_parameters': expert.num_parameters
            })
        stats['experts'] = expert_params
        
        # 容量限制统计
        if self.capacity_limiter is not None:
            stats['overflow_rate'] = self.capacity_limiter.get_overflow_rate()
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.forward_count.zero_()
        self.total_aux_loss.zero_()
        self.gate.reset_stats()
        if self.capacity_limiter is not None:
            self.capacity_limiter.reset_stats()
    
    def get_expert_parameters(self, expert_id: int) -> nn.Parameter:
        """
        获取指定专家的参数
        
        Args:
            expert_id: 专家ID
        
        Returns:
            专家参数
        
        Raises:
            ValueError: 当专家ID不合法时
        """
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"专家ID必须在[0, {self.num_experts})范围内")
        return self.experts[expert_id].parameters()
    
    def freeze_expert(self, expert_id: int) -> None:
        """
        冻结指定专家
        
        Args:
            expert_id: 专家ID
        """
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"专家ID必须在[0, {self.num_experts})范围内")
        for param in self.experts[expert_id].parameters():
            param.requires_grad = False
    
    def unfreeze_expert(self, expert_id: int) -> None:
        """
        解冻指定专家
        
        Args:
            expert_id: 专家ID
        """
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"专家ID必须在[0, {self.num_experts})范围内")
        for param in self.experts[expert_id].parameters():
            param.requires_grad = True


class SwitchMoE(MixtureOfExperts):
    """
    Switch MoE变体
    
    每次只选择1个专家（Top-1路由），计算效率更高。
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_experts: int = 8,
                 expert_type: str = "mlp",
                 expert_config: Optional[Dict[str, Any]] = None,
                 gate_config: Optional[Dict[str, Any]] = None,
                 capacity_factor: float = 1.0,
                 aux_loss_coef: float = 0.01,
                 use_capacity_limit: bool = True):
        """
        初始化Switch MoE
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_experts: 专家数量
            expert_type: 专家类型
            expert_config: 专家配置
            gate_config: 门控配置
            capacity_factor: 容量因子
            aux_loss_coef: 辅助损失系数
            use_capacity_limit: 是否使用容量限制
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            top_k=1,  # Switch MoE使用Top-1
            expert_type=expert_type,
            expert_config=expert_config,
            gate_config=gate_config,
            capacity_factor=capacity_factor,
            aux_loss_coef=aux_loss_coef,
            use_capacity_limit=use_capacity_limit,
            aggregate_method="weighted_sum"
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Switch MoE前向传播（优化版本）
        
        由于只选择1个专家，可以直接索引而不需要加权聚合。
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"输入维度不匹配: 期望{self.input_dim}, 得到{x.shape[-1]}"
            )
        
        original_shape = x.shape[:-1]
        x_flat = x.view(-1, self.input_dim)
        batch_size = x_flat.size(0)
        
        # 准备路由输入
        if x.dim() == 3:
            router_input = x.mean(dim=1)
        elif x.dim() == 2:
            router_input = x
        else:
            router_input = x_flat
        
        # 获取路由决策
        expert_weights, expert_indices = self.gate(router_input)
        
        # 应用容量限制
        overflow_mask = None
        if self.use_capacity_limit and self.capacity_limiter is not None:
            capacity = self.capacity_limiter.compute_capacity(batch_size)
            expert_indices, expert_weights, overflow_mask = \
                self.capacity_limiter.apply_capacity_constraint(
                    expert_indices, expert_weights, capacity
                )
        
        # Switch MoE优化：直接计算选中的专家
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for expert_id in range(self.num_experts):
            mask = expert_indices[:, 0] == expert_id
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_id](expert_input)
                output[mask] = expert_output
        
        # 恢复形状
        output = output.view(*original_shape, self.output_dim)
        
        # 计算辅助损失
        aux_loss = self._compute_aux_loss(expert_weights, expert_indices, batch_size)
        
        self.forward_count += 1
        self.total_aux_loss += aux_loss.detach()
        
        aux_outputs = {
            'expert_weights': expert_weights,
            'expert_indices': expert_indices,
            'aux_loss': aux_loss,
            'overflow_mask': overflow_mask,
        }
        
        return output, aux_outputs


class SparseMoE(MixtureOfExperts):
    """
    稀疏MoE变体
    
    支持更激进的稀疏性，可以配置非常小的top_k（如0.5表示只处理50%的token）。
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 expert_type: str = "mlp",
                 expert_config: Optional[Dict[str, Any]] = None,
                 gate_config: Optional[Dict[str, Any]] = None,
                 capacity_factor: float = 1.25,
                 aux_loss_coef: float = 0.01,
                 use_capacity_limit: bool = True,
                 token_dropout: float = 0.0):
        """
        初始化稀疏MoE
        
        Args:
            token_dropout: token丢弃率，用于进一步增加稀疏性
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_type=expert_type,
            expert_config=expert_config,
            gate_config=gate_config,
            capacity_factor=capacity_factor,
            aux_loss_coef=aux_loss_coef,
            use_capacity_limit=use_capacity_limit,
            aggregate_method="weighted_sum"
        )
        self.token_dropout = token_dropout
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        稀疏MoE前向传播
        
        支持token级别的dropout以增加稀疏性。
        """
        if self.training and self.token_dropout > 0:
            # 随机丢弃部分token
            mask = torch.rand(x.size(0), device=x.device) > self.token_dropout
            if not mask.all():
                # 对保留的token进行MoE计算
                output, aux_outputs = super().forward(x[mask])
                
                # 恢复完整输出
                full_output = torch.zeros(
                    x.size(0), *output.shape[1:],
                    device=output.device, dtype=output.dtype
                )
                full_output[mask] = output
                
                # 对丢弃的token使用残差连接或零填充
                full_output[~mask] = x[~mask, :self.output_dim] if self.input_dim == self.output_dim else 0
                
                return full_output, aux_outputs
        
        return super().forward(x)


def create_moe(input_dim: int,
               output_dim: int,
               num_experts: int = 8,
               top_k: int = 2,
               variant: str = "standard",
               **kwargs) -> MixtureOfExperts:
    """
    创建MoE网络的便捷函数
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        num_experts: 专家数量
        top_k: 选择的专家数
        variant: MoE变体 ("standard", "switch", "sparse")
        **kwargs: 其他配置参数
    
    Returns:
        MoE实例
    
    Raises:
        ValueError: 当变体类型不支持时
    """
    if variant == "standard":
        return MixtureOfExperts(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            top_k=top_k,
            **kwargs
        )
    elif variant == "switch":
        return SwitchMoE(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            **kwargs
        )
    elif variant == "sparse":
        return SparseMoE(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            top_k=top_k,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的MoE变体: {variant}")


# 向后兼容的别名
MoE = MixtureOfExperts
