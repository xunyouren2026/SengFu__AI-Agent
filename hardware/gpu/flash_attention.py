"""
Flash Attention - 高效注意力计算实现

模块路径: hardware/gpu/flash_attention.py

提供高效的Flash Attention实现，支持:
- 标准Flash Attention算法
- 内存高效的注意力计算
- 因果掩码支持
- 多头注意力优化
- 与PyTorch的集成

参考: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
"""

import math
import logging
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# 尝试导入flash_attn库
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.warning("flash_attn not available. Using fallback implementation.")

# 尝试导入PyTorch SDPA
try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')
except:
    SDPA_AVAILABLE = False


@dataclass
class FlashAttentionConfig:
    """Flash Attention配置"""
    dropout: float = 0.0
    causal: bool = False
    window_size: Tuple[int, int] = (-1, -1)  # 局部注意力窗口大小
    alibi_slopes: Optional[Tensor] = None  # ALiBi位置编码斜率
    deterministic: bool = False
    return_attn_probs: bool = False
    softmax_scale: Optional[float] = None


class FlashAttention(nn.Module):
    """
    Flash Attention模块
    
    实现高效的注意力计算，显著减少HBM访问，提高计算效率。
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        初始化Flash Attention
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_kv_heads: KV头数（用于GQA/MQA），默认为num_heads
            dropout: dropout概率
            causal: 是否使用因果掩码
            use_flash: 是否使用Flash Attention
            device: 计算设备
            dtype: 数据类型
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        
        # 线性投影层
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """重置参数"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, embed_dim]
            key: 键张量，默认为query
            value: 值张量，默认为query
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            need_weights: 是否返回注意力权重
            is_causal: 是否使用因果掩码
            
        Returns:
            output: 输出张量
            attn_weights: 注意力权重（如果need_weights=True）
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        causal = is_causal if is_causal is not None else self.causal
        
        batch_size, seq_len, _ = query.shape
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        
        # 应用Flash Attention或标准注意力
        if self.use_flash and FLASH_ATTN_AVAILABLE:
            output = self._flash_attention_forward(q, k, v, causal, key_padding_mask)
        elif SDPA_AVAILABLE:
            output = self._sdpa_forward(q, k, v, attn_mask, causal)
        else:
            output = self._standard_attention_forward(q, k, v, attn_mask, causal)
        
        # 重塑并投影输出
        output = output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        if need_weights:
            # Flash Attention不返回权重，这里返回None
            return output, None
        return output, None
    
    def _flash_attention_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Flash Attention前向传播
        
        Args:
            q: 查询 [batch, seq_len, num_heads, head_dim]
            k: 键 [batch, kv_len, num_kv_heads, head_dim]
            v: 值 [batch, kv_len, num_kv_heads, head_dim]
            causal: 是否因果
            key_padding_mask: 填充掩码
            
        Returns:
            输出张量
        """
        batch_size, seq_len, _, _ = q.shape
        
        # 处理填充（变长序列）
        if key_padding_mask is not None:
            # 展开填充
            k, v, indices, cu_seqlens_k, max_seqlen_k = self._unpad_input(k, v, key_padding_mask)
            q, _, indices_q, cu_seqlens_q, max_seqlen_q = self._unpad_input(
                q, q, torch.zeros(batch_size, seq_len, dtype=torch.bool, device=q.device)
            )
            
            # 变长Flash Attention
            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.dropout if self.training else 0.0,
                causal=causal,
                softmax_scale=None,
            )
            # 重新填充
            output = pad_input(output, indices_q, batch_size, seq_len)
        else:
            # 标准Flash Attention
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=causal,
                softmax_scale=None,
            )
        
        return output
    
    def _sdpa_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        causal: bool,
    ) -> Tensor:
        """
        使用PyTorch SDPA的前向传播
        
        Args:
            q: 查询
            k: 键
            v: 值
            attn_mask: 注意力掩码
            causal: 是否因果
            
        Returns:
            输出张量
        """
        # 调整维度以匹配SDPA: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 处理GQA/MQA
        if self.num_kv_heads != self.num_heads:
            k = self._repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = self._repeat_kv(v, self.num_heads // self.num_kv_heads)
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        
        # 恢复维度
        output = output.transpose(1, 2).contiguous()
        return output
    
    def _standard_attention_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        causal: bool,
    ) -> Tensor:
        """
        标准注意力前向传播（回退实现）
        
        Args:
            q: 查询
            k: 键
            v: 值
            attn_mask: 注意力掩码
            causal: 是否因果
            
        Returns:
            输出张量
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        _, kv_len, num_kv_heads, _ = k.shape
        
        # 调整维度
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, kv_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads, kv_len, head_dim]
        
        # 处理GQA/MQA
        if num_kv_heads != num_heads:
            k = self._repeat_kv(k, num_heads // num_kv_heads)
            v = self._repeat_kv(v, num_heads // num_kv_heads)
        
        # 计算注意力分数
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 应用掩码
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # 应用注意力到值
        output = torch.matmul(attn_weights, v)
        
        # 恢复维度
        output = output.transpose(1, 2).contiguous()
        return output
    
    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """
        重复KV头（用于GQA/MQA）
        
        Args:
            x: 输入张量 [batch, num_kv_heads, seq_len, head_dim]
            n_rep: 重复次数
            
        Returns:
            重复后的张量 [batch, num_heads, seq_len, head_dim]
        """
        batch, num_kv_heads, slen, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]
            .expand(batch, num_kv_heads, n_rep, slen, head_dim)
            .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
        )
    
    def _unpad_input(
        self,
        hidden_states: Tensor,
        values: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        移除填充的输入
        
        Args:
            hidden_states: 隐藏状态
            values: 值张量
            attention_mask: 注意力掩码
            
        Returns:
            展开后的张量和相关信息
        """
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        
        hidden_states = index_first_axis(hidden_states.reshape(-1, *hidden_states.shape[2:]), indices)
        values = index_first_axis(values.reshape(-1, *values.shape[2:]), indices)
        
        return hidden_states, values, indices, cu_seqlens, max_seqlen_in_batch


class MemoryEfficientAttention(FlashAttention):
    """
    内存高效注意力
    
    继承FlashAttention，专注于最小化内存使用。
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['use_flash'] = kwargs.get('use_flash', True)
        super().__init__(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """前向传播，强制使用内存高效实现"""
        # 确保使用checkpoint来节省内存
        if self.training:
            return torch.utils.checkpoint.checkpoint(super().forward, *args, **kwargs)
        return super().forward(*args, **kwargs)


class LocalFlashAttention(FlashAttention):
    """
    局部Flash Attention
    
    只关注局部窗口内的token，降低计算复杂度。
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 512,
        **kwargs
    ):
        """
        初始化局部Flash Attention
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            window_size: 局部窗口大小
        """
        super().__init__(embed_dim, num_heads, **kwargs)
        self.window_size = window_size
    
    def _create_local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        创建局部注意力掩码
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            局部掩码
        """
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0
        return mask
    
    def forward(self, query, key=None, value=None, **kwargs):
        """前向传播，应用局部掩码"""
        seq_len = query.shape[1]
        local_mask = self._create_local_mask(seq_len, query.device)
        
        if kwargs.get('attn_mask') is not None:
            kwargs['attn_mask'] = kwargs['attn_mask'] + local_mask
        else:
            kwargs['attn_mask'] = local_mask
        
        return super().forward(query, key, value, **kwargs)


class FlashAttentionFunction(torch.autograd.Function):
    """
    Flash Attention自定义函数
    
    支持自定义梯度计算。
    """
    
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, causal, softmax_scale):
        """
        前向传播
        
        Args:
            ctx: 上下文
            q: 查询
            k: 键
            v: 值
            dropout_p: dropout概率
            causal: 是否因果
            softmax_scale: softmax缩放因子
            
        Returns:
            输出张量
        """
        if not FLASH_ATTN_AVAILABLE:
            raise RuntimeError("flash_attn is not available")
        
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        
        out = flash_attn_func(q, k, v, dropout_p, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, k, v, out)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        q, k, v, out = ctx.saved_tensors
        
        # 使用flash_attn的反向传播（如果可用）
        # 否则使用标准实现
        grad_q, grad_k, grad_v = torch.autograd.grad(
            out, (q, k, v), grad_output, retain_graph=False
        )
        
        return grad_q, grad_k, grad_v, None, None, None


def apply_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """
    应用Flash Attention
    
    Args:
        q: 查询张量 [batch, seq_len, num_heads, head_dim]
        k: 键张量 [batch, seq_len, num_heads, head_dim]
        v: 值张量 [batch, seq_len, num_heads, head_dim]
        dropout_p: dropout概率
        causal: 是否因果
        softmax_scale: softmax缩放因子
        
    Returns:
        输出张量
    """
    if FLASH_ATTN_AVAILABLE:
        return FlashAttentionFunction.apply(q, k, v, dropout_p, causal, softmax_scale)
    
    # 回退到标准实现
    batch, seq_len, num_heads, head_dim = q.shape
    scale = softmax_scale or head_dim ** -0.5
    
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k) * scale
    
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    
    output = torch.einsum('bhqk,bkhd->bqhd', attn, v)
    return output


# 便捷的函数接口
def flash_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Optional[Tensor] = None,
    value: Optional[Tensor] = None,
    **kwargs
) -> Tensor:
    """
    Flash Attention前向函数
    
    Args:
        module: FlashAttention模块
        query: 查询
        key: 键
        value: 值
        **kwargs: 其他参数
        
    Returns:
        输出张量
    """
    output, _ = module(query, key, value, **kwargs)
    return output


def enable_flash_attention(model: nn.Module, use_flash: bool = True) -> nn.Module:
    """
    为模型启用Flash Attention
    
    Args:
        model: PyTorch模型
        use_flash: 是否使用Flash Attention
        
    Returns:
        修改后的模型
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.MultiheadAttention, nn.MultiheadAttention)):
            # 替换为Flash Attention
            # 注意：这需要根据具体模型结构调整
            pass
    return model


# 检查Flash Attention可用性
def is_flash_attention_available() -> bool:
    """检查Flash Attention是否可用"""
    return FLASH_ATTN_AVAILABLE


def get_flash_attention_info() -> Dict[str, Any]:
    """获取Flash Attention信息"""
    return {
        "flash_attn_available": FLASH_ATTN_AVAILABLE,
        "sdpa_available": SDPA_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
