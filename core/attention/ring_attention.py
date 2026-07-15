"""
Ring Attention: 环形注意力机制
支持超长上下文，块级并行计算

基于论文 "Ring Attention with Blockwise Transformers for Near-Infinite Context"
通过环形通信模式实现超长序列的注意力计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from dataclasses import dataclass
import torch.distributed as dist


@dataclass
class RingAttentionConfig:
    """环形注意力配置"""
    block_size: int = 1024  # 块大小
    num_heads: int = 8  # 注意力头数
    head_dim: int = 64  # 每个头的维度
    dropout: float = 0.0  # dropout概率
    causal: bool = True  # 是否因果注意力
    use_flash_attn: bool = False  # 是否使用Flash Attention
    scale_factor: Optional[float] = None  # 缩放因子


class RingAttention(nn.Module):
    """
    环形注意力机制
    支持超长上下文的块级并行注意力计算
    """
    
    def __init__(self, config: RingAttentionConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.causal = config.causal
        self.scale_factor = config.scale_factor or (self.head_dim ** -0.5)
        
        # 线性投影层
        self.q_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
        self.k_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
        self.v_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.num_heads * config.head_dim)
        
        self.dropout_layer = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            
        Returns:
            输出隐藏状态和注意力权重
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 线性投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用环形注意力计算
        if seq_len > self.block_size:
            # 长序列使用环形注意力
            attn_output = self._ring_attention_forward(q, k, v, attention_mask)
        else:
            # 短序列使用标准注意力
            attn_output = self._standard_attention(q, k, v, attention_mask)
        
        # 重塑并投影输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.o_proj(attn_output)
        output = self.dropout_layer(output)
        
        return output, None
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        标准注意力计算（用于短序列）
        
        Args:
            q: 查询 [batch_size, num_heads, seq_len, head_dim]
            k: 键 [batch_size, num_heads, seq_len, head_dim]
            v: 值 [batch_size, num_heads, seq_len, head_dim]
            attention_mask: 注意力掩码
            
        Returns:
            注意力输出
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        
        # 应用因果掩码
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 应用注意力掩码
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def _ring_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        环形注意力前向传播
        
        将序列分成块，通过环形通信模式计算注意力
        
        Args:
            q: 查询 [batch_size, num_heads, seq_len, head_dim]
            k: 键 [batch_size, num_heads, seq_len, head_dim]
            v: 值 [batch_size, num_heads, seq_len, head_dim]
            attention_mask: 注意力掩码
            
        Returns:
            注意力输出
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # 初始化输出和累加器
        output = torch.zeros_like(q)
        
        # 将KV分成块
        k_blocks = list(torch.split(k, self.block_size, dim=2))
        v_blocks = list(torch.split(v, self.block_size, dim=2))
        
        # 环形注意力：每个查询块与所有KV块计算注意力
        for query_block_idx in range(num_blocks):
            q_start = query_block_idx * self.block_size
            q_end = min(q_start + self.block_size, seq_len)
            q_block = q[:, :, q_start:q_end, :]
            
            # 初始化该查询块的累加器
            block_output = torch.zeros_like(q_block)
            max_score = torch.full((batch_size, num_heads, q_end - q_start, 1), float('-inf'), device=q.device)
            sum_exp = torch.zeros((batch_size, num_heads, q_end - q_start, 1), device=q.device)
            
            # 与所有KV块计算注意力（环形遍历）
            for kv_block_idx in range(num_blocks):
                k_block = k_blocks[kv_block_idx]
                v_block = v_blocks[kv_block_idx]
                
                # 计算块间注意力分数
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale_factor
                
                # 应用因果掩码
                if self.causal:
                    kv_start = kv_block_idx * self.block_size
                    kv_end = min(kv_start + self.block_size, seq_len)
                    
                    # 创建因果掩码
                    q_indices = torch.arange(q_start, q_end, device=q.device).view(-1, 1)
                    kv_indices = torch.arange(kv_start, kv_end, device=q.device).view(1, -1)
                    causal_mask = q_indices < kv_indices
                    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # 在线softmax更新
                block_max = scores.max(dim=-1, keepdim=True).values
                new_max = torch.maximum(max_score, block_max)
                
                # 更新累加器
                exp_scores = torch.exp(scores - new_max)
                sum_exp = sum_exp * torch.exp(max_score - new_max) + exp_scores.sum(dim=-1, keepdim=True)
                
                # 更新输出
                block_output = block_output * torch.exp(max_score - new_max) + torch.matmul(exp_scores, v_block)
                max_score = new_max
            
            # 归一化
            block_output = block_output / (sum_exp + 1e-8)
            output[:, :, q_start:q_end, :] = block_output
        
        return output


class RingAttentionLayer(nn.Module):
    """
    环形注意力层
    包含注意力、前馈网络和残差连接
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 1024,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        
        # 配置
        config = RingAttentionConfig(
            block_size=block_size,
            num_heads=num_heads,
            head_dim=self.head_dim,
            dropout=dropout
        )
        
        # 注意力层
        self.attention = RingAttention(config)
        
        # Layer Norm
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            
        Returns:
            输出隐藏状态
        """
        # 自注意力 + 残差
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, _ = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        
        # 前馈网络 + 残差
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states


class DistributedRingAttention(nn.Module):
    """
    分布式环形注意力
    支持多GPU并行处理超长序列
    """
    
    def __init__(
        self,
        config: RingAttentionConfig,
        process_group: Optional[Any] = None
    ):
        super().__init__()
        self.config = config
        self.process_group = process_group or dist.group.WORLD
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed training is not initialized")
        
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        
        # 本地注意力模块
        self.local_attention = RingAttention(config)
    
    def forward(
        self,
        local_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        分布式前向传播
        
        Args:
            local_hidden_states: 本地隐藏状态
            attention_mask: 注意力掩码
            
        Returns:
            本地输出
        """
        batch_size, local_seq_len, hidden_size = local_hidden_states.shape
        
        # 获取本地QKV
        q = self.local_attention.q_proj(local_hidden_states)
        k = self.local_attention.k_proj(local_hidden_states)
        v = self.local_attention.v_proj(local_hidden_states)
        
        # 重塑为多头形式
        q = q.view(batch_size, local_seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, local_seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, local_seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        # 环形通信：每个rank发送KV给下一个rank，接收来自上一个rank的KV
        output = self._ring_communication(q, k, v)
        
        # 重塑并投影输出
        output = output.transpose(1, 2).contiguous().view(batch_size, local_seq_len, hidden_size)
        output = self.local_attention.o_proj(output)
        
        return output
    
    def _ring_communication(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        环形通信模式
        
        每个rank与所有其他rank交换KV，计算局部注意力
        
        Args:
            q: 查询
            k: 键
            v: 值
            
        Returns:
            注意力输出
        """
        batch_size, num_heads, local_seq_len, head_dim = q.shape
        
        # 初始化输出和累加器
        output = torch.zeros_like(q)
        max_score = torch.full((batch_size, num_heads, local_seq_len, 1), float('-inf'), device=q.device)
        sum_exp = torch.zeros((batch_size, num_heads, local_seq_len, 1), device=q.device)
        
        # 当前KV
        current_k = k
        current_v = v
        
        # 环形遍历
        for step in range(self.world_size):
            # 计算与当前KV的注意力
            scores = torch.matmul(q, current_k.transpose(-2, -1)) * self.config.scale_factor
            
            # 在线softmax更新
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_score, block_max)
            
            exp_scores = torch.exp(scores - new_max)
            sum_exp = sum_exp * torch.exp(max_score - new_max) + exp_scores.sum(dim=-1, keepdim=True)
            output = output * torch.exp(max_score - new_max) + torch.matmul(exp_scores, current_v)
            max_score = new_max
            
            # 发送KV给下一个rank，接收来自上一个rank的KV
            if step < self.world_size - 1:
                send_to = (self.rank + 1) % self.world_size
                recv_from = (self.rank - 1 + self.world_size) % self.world_size
                
                # 准备发送缓冲区
                send_k = current_k.contiguous()
                send_v = current_v.contiguous()
                recv_k = torch.empty_like(send_k)
                recv_v = torch.empty_like(send_v)
                
                # 发送和接收
                send_op = dist.P2POp(dist.isend, send_k, send_to, group=self.process_group)
                recv_op = dist.P2POp(dist.irecv, recv_k, recv_from, group=self.process_group)
                
                dist.batch_isend_irecv([send_op, recv_op])
                
                # 同样处理V
                send_op_v = dist.P2POp(dist.isend, send_v, send_to, group=self.process_group)
                recv_op_v = dist.P2POp(dist.irecv, recv_v, recv_from, group=self.process_group)
                
                dist.batch_isend_irecv([send_op_v, recv_op_v])
                
                current_k = recv_k
                current_v = recv_v
        
        # 归一化
        output = output / (sum_exp + 1e-8)
        
        return output


class BlockwiseRingAttention(nn.Module):
    """
    块级环形注意力
    用于单设备上的超长序列处理
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 1024,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = hidden_size // num_heads
        
        config = RingAttentionConfig(
            block_size=block_size,
            num_heads=num_heads,
            head_dim=self.head_dim,
            dropout=dropout
        )
        
        self.attention = RingAttention(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        块级前向传播
        
        将长序列分块处理，通过梯度检查点减少内存占用
        
        Args:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            
        Returns:
            输出隐藏状态
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if seq_len <= self.block_size:
            # 短序列直接处理
            return self.attention(hidden_states, attention_mask)[0]
        
        # 长序列分块处理
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        outputs = []
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, seq_len)
            
            # 提取块
            block = hidden_states[:, start:end, :]
            
            # 使用梯度检查点
            if self.training:
                block_output = torch.utils.checkpoint.checkpoint(
                    self._forward_block,
                    block,
                    attention_mask[:, start:end] if attention_mask is not None else None
                )
            else:
                block_output = self._forward_block(
                    block,
                    attention_mask[:, start:end] if attention_mask is not None else None
                )
            
            outputs.append(block_output)
        
        # 拼接输出
        return torch.cat(outputs, dim=1)
    
    def _forward_block(
        self,
        block: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """处理单个块"""
        return self.attention(block, block_mask)[0]


def create_ring_attention_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    causal: bool = True
) -> torch.Tensor:
    """
    创建环形注意力掩码
    
    Args:
        seq_len: 序列长度
        block_size: 块大小
        device: 设备
        causal: 是否因果掩码
        
    Returns:
        注意力掩码
    """
    if not causal:
        return None
    
    # 创建因果掩码
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, 0.0)
    
    return mask.unsqueeze(0).unsqueeze(0)


def test_ring_attention():
    """测试环形注意力"""
    # 配置
    config = RingAttentionConfig(
        block_size=512,
        num_heads=8,
        head_dim=64,
        causal=True
    )
    
    # 创建模型
    model = RingAttention(config)
    
    # 测试数据
    batch_size = 2
    seq_len = 2048
    hidden_size = config.num_heads * config.head_dim
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    output, _ = model(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print("Ring Attention test passed!")
    
    return output


if __name__ == "__main__":
    test_ring_attention()
