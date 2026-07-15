"""
AGI统一框架 - Transformer核心组件
实现完整的Transformer架构，包括编码器、解码器、注意力机制等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


# ==================== 位置编码 ====================

class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LearnablePositionEmbedding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1)]


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._build_cache(max_len)
        
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
        
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
            
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)
        
        return q_embed, k_embed
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat([x1 * cos[..., :x1.size(-1)] - x2 * sin[..., :x2.size(-1)],
                         x1 * sin[..., :x1.size(-1)] + x2 * cos[..., :x2.size(-1)]], dim=-1)


# ==================== 注意力机制 ====================

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0,
                 use_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=use_bias)
        self.k_proj = nn.Linear(dim, dim, bias=use_bias)
        self.v_proj = nn.Linear(dim, dim, bias=use_bias)
        self.out_proj = nn.Linear(dim, dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力
        output = torch.matmul(attn_probs, v)
        
        # 重塑
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(output)


class FlashAttention(nn.Module):
    """Flash Attention (内存高效实现)"""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 分块计算
        output = self._flash_attention(q, k, v, attention_mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(output)
    
    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         mask: Optional[torch.Tensor]) -> torch.Tensor:
        # 简化的Flash Attention实现
        # 实际实现需要CUDA kernel
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn + mask
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        return torch.matmul(attn, v)


class LinearAttention(nn.Module):
    """线性复杂度注意力"""
    
    def __init__(self, dim: int, num_heads: int, feature_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.feature_dim = feature_dim or self.head_dim
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 特征映射 (ELU + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 线性注意力: (Q @ (K^T @ V)) / (Q @ (K^T @ 1))
        kv = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2, keepdim=True).transpose(-2, -1)
        
        numerator = torch.matmul(q, kv)
        denominator = torch.matmul(q, k_sum) + 1e-6
        
        output = numerator / denominator
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(output)


class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力"""
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        output = torch.zeros_like(q)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            q_i = q[:, :, i:i+1, :]
            k_window = k[:, :, start:end, :]
            v_window = v[:, :, start:end, :]
            
            attn = torch.matmul(q_i, k_window.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            output[:, :, i:i+1, :] = torch.matmul(attn, v_window)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(output)


# ==================== 前馈网络 ====================

class FeedForward(nn.Module):
    """标准前馈网络"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0,
                 activation: str = "gelu"):
        super().__init__()
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GLU(nn.Module):
    """门控线性单元"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU激活函数"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ==================== Transformer层 ====================

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.0, activation: str = "gelu",
                 layer_norm_eps: float = 1e-6,
                 pre_norm: bool = True):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_dim, dropout, activation)
        
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.pre_norm = pre_norm
        
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.pre_norm:
            # Pre-LN
            x_norm = self.norm1(x)
            attn_out = self.self_attn(x_norm, x_norm, x_norm, attention_mask, key_padding_mask)
            x = x + self.dropout1(attn_out)
            
            x_norm = self.norm2(x)
            ff_out = self.feed_forward(x_norm)
            x = x + self.dropout2(ff_out)
        else:
            # Post-LN
            attn_out = self.self_attn(x, x, x, attention_mask, key_padding_mask)
            x = self.norm1(x + self.dropout1(attn_out))
            
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_out))
            
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.0, activation: str = "gelu",
                 layer_norm_eps: float = 1e-6):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_dim, dropout, activation)
        
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self attention
        x_norm = self.norm1(x)
        self_attn_out = self.self_attn(x_norm, x_norm, x_norm, self_attn_mask)
        x = x + self.dropout1(self_attn_out)
        
        # Cross attention
        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(x_norm, memory, memory, cross_attn_mask)
        x = x + self.dropout2(cross_attn_out)
        
        # Feed forward
        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout3(ff_out)
        
        return x


# ==================== 完整Transformer ====================

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, vocab_size: int, dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, max_len: int = 512,
                 dropout: float = 0.0, activation: str = "gelu",
                 pos_embedding_type: str = "sinusoidal"):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        if pos_embedding_type == "sinusoidal":
            self.pos_embedding = SinusoidalPositionEmbedding(dim, max_len)
        else:
            self.pos_embedding = LearnablePositionEmbedding(max_len, dim)
            
        self.embed_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, ff_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = self.token_embedding(input_ids)
        x = self.pos_embedding(x)
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask, key_padding_mask)
            
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, vocab_size: int, dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, max_len: int = 512,
                 dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = SinusoidalPositionEmbedding(dim, max_len)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, num_heads, ff_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = self.token_embedding(input_ids)
        x = self.pos_embedding(x)
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask)
            
        x = self.final_norm(x)
        return self.output_proj(x)


class Transformer(nn.Module):
    """完整Transformer模型"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 dim: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 ff_dim: int = 2048, max_len: int = 512,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, dim, num_heads, num_encoder_layers,
            ff_dim, max_len, dropout, activation
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, dim, num_heads, num_decoder_layers,
            ff_dim, max_len, dropout, activation
        )
        
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        memory = self.encoder(src_ids, src_mask)
        output = self.decoder(tgt_ids, memory, tgt_mask)
        
        return output
    
    def encode(self, src_ids: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(src_ids, src_mask)
    
    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(tgt_ids, memory, tgt_mask)


# ==================== 生成工具 ====================

def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """生成因果注意力掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))


def top_k_sampling(logits: torch.Tensor, k: int = 50,
                   temperature: float = 1.0) -> torch.Tensor:
    """Top-K采样"""
    logits = logits / temperature
    
    values, indices = torch.topk(logits, k, dim=-1)
    probs = F.softmax(values, dim=-1)
    
    sampled = torch.multinomial(probs, 1)
    return indices.gather(-1, sampled).squeeze(-1)


def top_p_sampling(logits: torch.Tensor, p: float = 0.9,
                   temperature: float = 1.0) -> torch.Tensor:
    """Nucleus (Top-P) 采样"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return torch.multinomial(probs, 1).squeeze(-1)


def beam_search(model: Transformer, src_ids: torch.Tensor,
                beam_width: int = 5, max_len: int = 100,
                bos_id: int = 0, eos_id: int = 2) -> torch.Tensor:
    """束搜索解码"""
    batch_size = src_ids.size(0)
    device = src_ids.device
    
    with torch.no_grad():
        memory = model.encode(src_ids)
    
    # 初始化束
    beams = [torch.tensor([[bos_id]], device=device) for _ in range(batch_size)]
    scores = [torch.tensor([0.0], device=device) for _ in range(batch_size)]
    
    for _ in range(max_len):
        all_candidates = []
        
        for i, (beam, score) in enumerate(zip(beams, scores)):
            if beam[0, -1] == eos_id:
                all_candidates.append((beam, score, i))
                continue
                
            with torch.no_grad():
                logits = model.decode(beam, memory[i:i+1])
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
            top_k_log_probs, top_k_indices = log_probs.topk(beam_width, dim=-1)
            
            for j in range(beam_width):
                new_beam = torch.cat([beam, top_k_indices[:, j:j+1]], dim=-1)
                new_score = score + top_k_log_probs[:, j]
                all_candidates.append((new_beam, new_score, i))
        
        # 选择最佳候选
        all_candidates.sort(key=lambda x: x[1].item(), reverse=True)
        
        new_beams = []
        new_scores = []
        used_batch = set()
        
        for beam, score, batch_idx in all_candidates:
            if batch_idx not in used_batch and len(new_beams) < batch_size:
                new_beams.append(beam)
                new_scores.append(score)
                used_batch.add(batch_idx)
                
        beams = new_beams
        scores = new_scores
        
        if all(beam[0, -1] == eos_id for beam in beams):
            break
    
    return beams[0]
