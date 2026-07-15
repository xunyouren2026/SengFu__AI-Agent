"""
嵌入层模块 - 完整实现
包含: Embedding, PositionalEncoding, LearnedPositionalEncoding,
      RotaryPositionalEmbedding(RoPE), ALiBi, TokenEmbedding等
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH




# =============================================================================
# PyTorch Compatibility Utilities
# =============================================================================

def _to_tensor(x, device: str = None, dtype=None, requires_grad: bool = False):
    """
    Convert input to torch.Tensor.
    
    Supports:
    - torch.Tensor: returned as-is (with optional device/dtype cast)
    - list/tuple: converted to torch.Tensor
    - numpy.ndarray: converted to torch.Tensor
    - scalar: wrapped in torch.Tensor
    
    Args:
        x: Input data (tensor, list, tuple, numpy array, or scalar)
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Target dtype (torch.float32, torch.float64, etc.)
        requires_grad: Whether to track gradients
    
    Returns:
        torch.Tensor or original type if torch is not available
    """
    if not _HAS_TORCH:
        return x
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != torch.device(device):
            x = x.to(device=device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        if requires_grad and not x.requires_grad:
            x = x.requires_grad_(requires_grad=True)
        return x
    # Convert from list/tuple/numpy
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def _to_numpy(x):
    """Convert torch.Tensor to numpy array."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


def _to_list(x):
    """Convert torch.Tensor to nested Python list."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().tolist()


def _get_device(x):
    """Get device of tensor, default to 'cpu'."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.device
    return None


def _batch_dim(x):
    """Ensure input has batch dimension. If 2D, add batch dim to make 3D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 2:
            return x.unsqueeze(0)
    return x


def _unbatch(x):
    """Remove batch dimension if it's 1. If 3D with batch=1, squeeze to 2D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 3 and x.size(0) == 1:
            return x.squeeze(0)
    return x


class Embedding:
    """
    嵌入层
    将整数索引映射到稠密向量
    
    参数:
        num_embeddings: 词汇表大小
        embedding_dim: 嵌入维度
        padding_idx: 填充索引(可选)
        max_norm: 最大范数(可选)
        scale_grad_by_freq: 是否按频率缩放梯度
        sparse: 是否使用稀疏梯度
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 scale_grad_by_freq: bool = False):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.scale_grad_by_freq = scale_grad_by_freq
        
        # 初始化权重 (num_embeddings, embedding_dim)
        std = 1.0 / math.sqrt(embedding_dim)
        self.weight = [[random.gauss(0, std) for _ in range(embedding_dim)] 
                       for _ in range(num_embeddings)]
        
        # 如果指定了padding_idx，将其设为0
        if padding_idx is not None and 0 <= padding_idx < num_embeddings:
            self.weight[padding_idx] = [0.0 for _ in range(embedding_dim)]
        
        self._input_cache = None
    
    def forward(self, indices: Union[List[int], List[List[int]]]) -> Union[Union[List[List[float]], 'torch.Tensor'], List[Union[List[List[float]], 'torch.Tensor']]]:
        """
        前向传播
        indices: 整数索引列表或二维索引矩阵
        返回: 嵌入向量
        """
        self._input_cache = indices
        
        # 检查是否是二维输入
        if indices and isinstance(indices[0], list):
            # 二维输入: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            output = []
            for batch_indices in indices:
                batch_output = []
                for idx in batch_indices:
                    if 0 <= idx < self.num_embeddings:
                        embedding = self.weight[idx][:]
                        # 应用max_norm
                        if self.max_norm is not None:
                            norm = math.sqrt(sum(e**2 for e in embedding))
                            if norm > self.max_norm:
                                scale = self.max_norm / norm
                                embedding = [e * scale for e in embedding]
                        batch_output.append(embedding)
                    else:
                        # 越界索引返回零向量
                        batch_output.append([0.0 for _ in range(self.embedding_dim)])
                output.append(batch_output)
            return output
        else:
            # 一维输入: (seq_len,) -> (seq_len, embedding_dim)
            output = []
            for idx in indices:
                if 0 <= idx < self.num_embeddings:
                    embedding = self.weight[idx][:]
                    if self.max_norm is not None:
                        norm = math.sqrt(sum(e**2 for e in embedding))
                        if norm > self.max_norm:
                            scale = self.max_norm / norm
                            embedding = [e * scale for e in embedding]
                    output.append(embedding)
                else:
                    output.append([0.0 for _ in range(self.embedding_dim)])
            return output
    
    def backward(self, grad_output: Union[Union[List[List[float]], 'torch.Tensor'], List[Union[List[List[float]], 'torch.Tensor']]]) -> Union[List[List[float]], 'torch.Tensor']:
        """
        反向传播
        返回权重梯度
        """
        indices = self._input_cache
        
        # 初始化权重梯度
        grad_weight = [[0.0 for _ in range(self.embedding_dim)] 
                       for _ in range(self.num_embeddings)]
        
        if indices and isinstance(indices[0], list):
            # 二维输入
            for b, batch_indices in enumerate(indices):
                for t, idx in enumerate(batch_indices):
                    if 0 <= idx < self.num_embeddings and idx != self.padding_idx:
                        for d in range(self.embedding_dim):
                            grad_weight[idx][d] += grad_output[b][t][d]
        else:
            # 一维输入
            for t, idx in enumerate(indices):
                if 0 <= idx < self.num_embeddings and idx != self.padding_idx:
                    for d in range(self.embedding_dim):
                        grad_weight[idx][d] += grad_output[t][d]
        
        return grad_weight
    
    def get_weight(self) -> Union[List[List[float]], 'torch.Tensor']:
        """获取权重矩阵"""
        return self.weight


class PositionalEncoding:
    """
    正弦位置编码 (原始Transformer位置编码)
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        
        # 预计算位置编码
        self.pe = self._compute_positional_encoding()
    
    def _compute_positional_encoding(self) -> Union[List[List[float]], 'torch.Tensor']:
        """计算位置编码矩阵"""
        pe = [[0.0 for _ in range(self.d_model)] for _ in range(self.max_len)]
        
        position = list(range(self.max_len))
        div_term = [1.0 / (10000 ** (2 * i / self.d_model)) for i in range(self.d_model // 2)]
        
        for pos in range(self.max_len):
            for i in range(self.d_model // 2):
                pe[pos][2 * i] = math.sin(position[pos] * div_term[i])
                pe[pos][2 * i + 1] = math.cos(position[pos] * div_term[i])
        
        # 处理奇数维度
        if self.d_model % 2 == 1:
            for pos in range(self.max_len):
                pe[pos][self.d_model - 1] = math.sin(position[pos] * div_term[-1])
        
        return pe
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        前向传播
        x: (batch_size, seq_len, d_model)
        返回: x + positional_encoding
        """
        batch_size = len(x)
        seq_len = len(x[0]) if batch_size > 0 else 0
        
        if seq_len > self.max_len:
            # 如果序列长度超过max_len，扩展位置编码
            self.max_len = seq_len
            self.pe = self._compute_positional_encoding()
        
        # 添加位置编码
        output = []
        for b in range(batch_size):
            batch_output = []
            for t in range(seq_len):
                token = [x[b][t][d] + self.pe[t][d] for d in range(self.d_model)]
                
                # 应用dropout
                if self.dropout > 0:
                    token = [t * (1 if random.random() > self.dropout else 0) for t in token]
                
                batch_output.append(token)
            output.append(batch_output)
        
        return output
    
    def get_encoding(self, seq_len: int) -> Union[List[List[float]], 'torch.Tensor']:
        """获取指定长度的位置编码"""
        return self.pe[:seq_len]


class LearnedPositionalEncoding:
    """
    可学习的位置编码
    位置编码作为可训练参数
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout
        
        # 初始化可学习的位置编码
        std = 1.0 / math.sqrt(d_model)
        self.pe = [[random.gauss(0, std) for _ in range(d_model)] 
                   for _ in range(max_len)]
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """前向传播"""
        batch_size = len(x)
        seq_len = len(x[0]) if batch_size > 0 else 0
        
        output = []
        for b in range(batch_size):
            batch_output = []
            for t in range(seq_len):
                if t < self.max_len:
                    token = [x[b][t][d] + self.pe[t][d] for d in range(self.d_model)]
                else:
                    # 超出max_len的位置使用最后一个位置编码
                    token = [x[b][t][d] + self.pe[-1][d] for d in range(self.d_model)]
                
                if self.dropout > 0:
                    token = [t * (1 if random.random() > self.dropout else 0) for t in token]
                
                batch_output.append(token)
            output.append(batch_output)
        
        return output
    
    def backward(self, grad_output: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        """反向传播"""
        batch_size = len(grad_output)
        seq_len = len(grad_output[0]) if batch_size > 0 else 0
        
        grad_pe = [[0.0 for _ in range(self.d_model)] for _ in range(self.max_len)]
        
        for b in range(batch_size):
            for t in range(min(seq_len, self.max_len)):
                for d in range(self.d_model):
                    grad_pe[t][d] += grad_output[b][t][d]
        
        return grad_pe


class RotaryPositionalEmbedding:
    """
    旋转位置嵌入 (RoPE)
    用于Transformer的位置编码，具有相对位置特性
    
    将向量旋转角度 theta * position
    """
    
    def __init__(self, d_model: int, max_len: int = 2048, base: int = 10000):
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # 预计算频率
        self.inv_freq = [1.0 / (base ** (2 * i / d_model)) for i in range(d_model // 2)]
        
        # 预计算cos和sin值
        self._compute_cos_sin_cache()
    
    def _compute_cos_sin_cache(self):
        """预计算cos和sin值"""
        self.cos_cache = [[0.0 for _ in range(self.d_model // 2)] 
                          for _ in range(self.max_len)]
        self.sin_cache = [[0.0 for _ in range(self.d_model // 2)] 
                          for _ in range(self.max_len)]
        
        for pos in range(self.max_len):
            for i in range(self.d_model // 2):
                theta = pos * self.inv_freq[i]
                self.cos_cache[pos][i] = math.cos(theta)
                self.sin_cache[pos][i] = math.sin(theta)
    
    def _rotate_half(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """将向量分成两半并旋转"""
        half = len(x) // 2
        x1 = x[:half]
        x2 = x[half:]
        return [-x2[i] for i in range(half)] + [x1[i] for i in range(half)]
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        前向传播
        x: (batch_size, seq_len, d_model)
        返回: 应用RoPE后的向量
        """
        batch_size = len(x)
        seq_len = len(x[0]) if batch_size > 0 else 0
        
        if seq_len > self.max_len:
            self.max_len = seq_len
            self._compute_cos_sin_cache()
        
        output = []
        for b in range(batch_size):
            batch_output = []
            for t in range(seq_len):
                token = x[b][t]
                
                # 应用旋转
                rotated = []
                half = self.d_model // 2
                
                for i in range(half):
                    # x1 * cos - x2 * sin
                    rotated.append(token[i] * self.cos_cache[t][i] - 
                                  token[half + i] * self.sin_cache[t][i])
                
                for i in range(half):
                    # x2 * cos + x1 * sin
                    rotated.append(token[half + i] * self.cos_cache[t][i] + 
                                  token[i] * self.sin_cache[t][i])
                
                # 处理奇数维度
                if self.d_model % 2 == 1:
                    rotated.append(token[-1])
                
                batch_output.append(rotated)
            output.append(batch_output)
        
        return output
    
    def apply_to_query_key(self, q: List[Union[List[List[float]], 'torch.Tensor']], 
                           k: List[Union[List[List[float]], 'torch.Tensor']]) -> Tuple:
        """将RoPE应用于查询和键"""
        return self.forward(q), self.forward(k)


class ALiBiPositionalEmbedding:
    """
    Attention with Linear Biases (ALiBi)
    使用线性偏置代替位置编码
    
    在注意力分数中添加 -m * |i - j| 的偏置
    """
    
    def __init__(self, num_heads: int, max_len: int = 2048):
        self.num_heads = num_heads
        self.max_len = max_len
        
        # 计算每个头的斜率
        self.slopes = self._compute_slopes()
        
        # 预计算偏置矩阵
        self._compute_bias_cache()
    
    def _compute_slopes(self) -> Union[List[float], 'torch.Tensor']:
        """计算每个头的斜率"""
        # 使用几何序列
        n = self.num_heads
        
        # 找到最大的2的幂次
        m = 1
        while m <= n:
            m *= 2
        m = m // 2
        
        slopes = []
        for i in range(n):
            if i < m:
                slopes.append(1.0 / (2 ** (i + 1)))
            else:
                slopes.append(1.0 / (2 ** (2 * m - i)))
        
        return slopes
    
    def _compute_bias_cache(self):
        """预计算偏置矩阵"""
        self.bias_cache = [[[0.0 for _ in range(self.max_len)] 
                           for _ in range(self.max_len)] 
                          for _ in range(self.num_heads)]
        
        for h in range(self.num_heads):
            for i in range(self.max_len):
                for j in range(self.max_len):
                    self.bias_cache[h][i][j] = -self.slopes[h] * abs(i - j)
    
    def get_bias(self, seq_len: int) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        获取指定长度的偏置矩阵
        返回: (num_heads, seq_len, seq_len)
        """
        if seq_len > self.max_len:
            self.max_len = seq_len
            self._compute_bias_cache()
        
        return [[self.bias_cache[h][i][:seq_len] for i in range(seq_len)] 
                for h in range(self.num_heads)]
    
    def forward(self, attention_scores: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """
        将ALiBi偏置添加到注意力分数
        attention_scores: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = len(attention_scores)
        num_heads = len(attention_scores[0]) if batch_size > 0 else 0
        seq_len = len(attention_scores[0][0]) if num_heads > 0 else 0
        
        bias = self.get_bias(seq_len)
        
        output = []
        for b in range(batch_size):
            batch_output = []
            for h in range(num_heads):
                head_output = []
                for i in range(seq_len):
                    row = [attention_scores[b][h][i][j] + bias[h][i][j] for j in range(seq_len)]
                    head_output.append(row)
                batch_output.append(head_output)
            output.append(batch_output)
        
        return output


class TokenEmbedding:
    """
    Token嵌入层
    将token ID转换为嵌入向量
    """
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # 初始化嵌入矩阵
        std = 1.0 / math.sqrt(d_model)
        self.embedding = [[random.gauss(0, std) for _ in range(d_model)] 
                         for _ in range(vocab_size)]
        
        if padding_idx is not None and 0 <= padding_idx < vocab_size:
            self.embedding[padding_idx] = [0.0 for _ in range(d_model)]
    
    def forward(self, token_ids: Union[List[int], List[List[int]]]) -> Union[Union[List[List[float]], 'torch.Tensor'], List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        if token_ids and isinstance(token_ids[0], list):
            # 二维输入
            output = []
            for batch_ids in token_ids:
                batch_output = []
                for idx in batch_ids:
                    if 0 <= idx < self.vocab_size:
                        batch_output.append(self.embedding[idx][:])
                    else:
                        batch_output.append([0.0 for _ in range(self.d_model)])
                output.append(batch_output)
            return output
        else:
            # 一维输入
            output = []
            for idx in token_ids:
                if 0 <= idx < self.vocab_size:
                    output.append(self.embedding[idx][:])
                else:
                    output.append([0.0 for _ in range(self.d_model)])
            return output


class SegmentEmbedding:
    """
    分段嵌入 (用于BERT等模型)
    区分不同句子的嵌入
    """
    
    def __init__(self, d_model: int, num_segments: int = 2):
        self.d_model = d_model
        self.num_segments = num_segments
        
        std = 1.0 / math.sqrt(d_model)
        self.embedding = [[random.gauss(0, std) for _ in range(d_model)] 
                         for _ in range(num_segments)]
    
    def forward(self, segment_ids: Union[List[int], List[List[int]]]) -> Union[Union[List[List[float]], 'torch.Tensor'], List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        if segment_ids and isinstance(segment_ids[0], list):
            output = []
            for batch_ids in segment_ids:
                batch_output = []
                for idx in batch_ids:
                    if 0 <= idx < self.num_segments:
                        batch_output.append(self.embedding[idx][:])
                    else:
                        batch_output.append([0.0 for _ in range(self.d_model)])
                output.append(batch_output)
            return output
        else:
            output = []
            for idx in segment_ids:
                if 0 <= idx < self.num_segments:
                    output.append(self.embedding[idx][:])
                else:
                    output.append([0.0 for _ in range(self.d_model)])
            return output


class BERTEmbedding:
    """
    BERT嵌入层
    组合token嵌入、分段嵌入和位置嵌入
    """
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512,
                 num_segments: int = 2, dropout: float = 0.1):
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.segment_embedding = SegmentEmbedding(d_model, num_segments)
        self.position_embedding = LearnedPositionalEncoding(d_model, max_len)
        self.dropout = dropout
        self.d_model = d_model
    
    def forward(self, token_ids: List[List[int]], 
                segment_ids: Optional[List[List[int]]] = None) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        前向传播
        token_ids: (batch_size, seq_len)
        segment_ids: (batch_size, seq_len)
        返回: (batch_size, seq_len, d_model)
        """
        batch_size = len(token_ids)
        seq_len = len(token_ids[0]) if batch_size > 0 else 0
        
        # Token嵌入
        token_emb = self.token_embedding.forward(token_ids)
        
        # 分段嵌入
        if segment_ids is None:
            segment_ids = [[0 for _ in range(seq_len)] for _ in range(batch_size)]
        segment_emb = self.segment_embedding.forward(segment_ids)
        
        # 位置嵌入
        position_ids = [list(range(seq_len)) for _ in range(batch_size)]
        position_emb = [[self.position_embedding.pe[t][:] for t in range(seq_len)] 
                       for _ in range(batch_size)]
        
        # 组合嵌入
        output = []
        for b in range(batch_size):
            batch_output = []
            for t in range(seq_len):
                combined = [token_emb[b][t][d] + segment_emb[b][t][d] + position_emb[b][t][d] 
                           for d in range(self.d_model)]
                
                # Dropout
                if self.dropout > 0:
                    combined = [c * (1 if random.random() > self.dropout else 0) for c in combined]
                
                batch_output.append(combined)
            output.append(batch_output)
        
        return output


class ScaledEmbedding:
    """
    缩放嵌入
    嵌入向量乘以sqrt(d_model)
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: bool = True):
        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.scale = scale
        self.embedding_dim = embedding_dim
    
    def forward(self, indices: Union[List[int], List[List[int]]]) -> Union[Union[List[List[float]], 'torch.Tensor'], List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        emb = self.embedding.forward(indices)
        
        if self.scale:
            scale_factor = math.sqrt(self.embedding_dim)
            
            if emb and isinstance(emb[0], list) and isinstance(emb[0][0], list):
                # 三维输出
                return [[[emb[b][t][d] * scale_factor for d in range(self.embedding_dim)]
                        for t in range(len(emb[b]))] for b in range(len(emb))]
            else:
                # 二维输出
                return [[emb[t][d] * scale_factor for d in range(self.embedding_dim)]
                        for t in range(len(emb))]
        
        return emb


class SinusoidalTimeEmbedding:
    """
    正弦时间嵌入 (用于扩散模型等)
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        self.half_dim = dim // 2
    
    def forward(self, timesteps: Union[List[float], 'torch.Tensor']) -> Union[List[List[float]], 'torch.Tensor']:
        """
        前向传播
        timesteps: 时间步列表
        返回: (len(timesteps), dim)
        """
        output = []
        
        for t in timesteps:
            embedding = []
            
            # 计算频率
            for i in range(self.half_dim):
                freq = 1.0 / (10000 ** (2 * i / self.dim))
                embedding.append(math.sin(t * freq))
                embedding.append(math.cos(t * freq))
            
            # 处理奇数维度
            if self.dim % 2 == 1:
                freq = 1.0 / (10000 ** (2 * self.half_dim / self.dim))
                embedding.append(math.sin(t * freq))
            
            output.append(embedding)
        
        return output


class FourierFeatureEmbedding:
    """
    傅里叶特征嵌入
    用于将低维输入映射到高维空间
    """
    
    def __init__(self, input_dim: int, embedding_dim: int, sigma: float = 1.0):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        
        # 随机采样频率
        num_frequencies = embedding_dim // 2
        std = sigma
        self.B = [[random.gauss(0, std) for _ in range(input_dim)] 
                  for _ in range(num_frequencies)]
    
    def forward(self, x: Union[List[List[float]], 'torch.Tensor']) -> Union[List[List[float]], 'torch.Tensor']:
        """
        前向传播
        x: (batch_size, input_dim)
        返回: (batch_size, embedding_dim)
        """
        batch_size = len(x)
        num_frequencies = len(self.B)
        
        output = []
        for b in range(batch_size):
            embedding = []
            
            for i in range(num_frequencies):
                # 计算 B_i * x
                dot_product = sum(self.B[i][j] * x[b][j] for j in range(self.input_dim))
                embedding.append(math.sin(2 * math.pi * dot_product))
                embedding.append(math.cos(2 * math.pi * dot_product))
            
            output.append(embedding)
        
        return output


class RelativePositionalEncoding:
    """
    相对位置编码
    用于Transformer-XL等模型
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 128):
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 初始化相对位置嵌入
        # 范围: [-max_relative_position, max_relative_position]
        vocab_size = 2 * max_relative_position + 1
        std = 1.0 / math.sqrt(d_model)
        self.embeddings = [[random.gauss(0, std) for _ in range(d_model)] 
                          for _ in range(vocab_size)]
    
    def _get_relative_position(self, i: int, j: int) -> int:
        """获取相对位置索引"""
        rel_pos = i - j
        # 裁剪到范围内
        rel_pos = max(-self.max_relative_position, min(self.max_relative_position, rel_pos))
        # 转换到非负索引
        return rel_pos + self.max_relative_position
    
    def forward(self, seq_len: int) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        获取相对位置编码矩阵
        返回: (seq_len, seq_len, d_model)
        """
        output = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                rel_pos_idx = self._get_relative_position(i, j)
                row.append(self.embeddings[rel_pos_idx][:])
            output.append(row)
        
        return output


class T5RelativePositionalEncoding:
    """
    T5风格的相对位置编码
    使用对数分桶
    """
    
    def __init__(self, num_buckets: int = 32, max_distance: int = 128):
        self.num_buckets = num_buckets
        self.max_distance = max_distance
    
    def _relative_position_bucket(self, relative_position: int) -> int:
        """将相对位置映射到桶索引"""
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        
        # 正负分开处理
        if relative_position < 0:
            relative_position = -relative_position
            bucket_offset = num_buckets // 2
        else:
            bucket_offset = 0
        
        # 对数分桶
        half_buckets = num_buckets // 2
        max_exact = half_buckets // 2
        
        if relative_position < max_exact:
            bucket = relative_position
        else:
            # 对数缩放
            relative_position = min(relative_position, max_distance)
            log_pos = math.log(relative_position / max_exact) / math.log(max_distance / max_exact)
            bucket = max_exact + int(log_pos * (half_buckets - max_exact))
        
        return min(bucket + bucket_offset, num_buckets - 1)
    
    def forward(self, seq_len: int) -> List[List[int]]:
        """
        获取相对位置桶索引矩阵
        返回: (seq_len, seq_len)
        """
        output = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                rel_pos = i - j
                bucket = self._relative_position_bucket(rel_pos)
                row.append(bucket)
            output.append(row)
        
        return output


class PositionalEncoding2D:
    """
    2D位置编码 (用于ViT等)
    """
    
    def __init__(self, d_model: int, height: int, width: int):
        self.d_model = d_model
        self.height = height
        self.width = width
        
        # 分别为行和列创建位置编码
        self.row_pe = PositionalEncoding(d_model // 2, max_len=height)
        self.col_pe = PositionalEncoding(d_model // 2, max_len=width)
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """
        前向传播
        x: (batch_size, height, width, d_model)
        """
        batch_size = len(x)
        h = len(x[0]) if batch_size > 0 else 0
        w = len(x[0][0]) if h > 0 else 0
        
        output = []
        for b in range(batch_size):
            batch_output = []
            for i in range(h):
                row_output = []
                for j in range(w):
                    # 组合行和列的位置编码
                    half = self.d_model // 2
                    combined = []
                    for d in range(half):
                        combined.append(x[b][i][j][d] + self.row_pe.pe[i][d])
                    for d in range(half):
                        combined.append(x[b][i][j][half + d] + self.col_pe.pe[j][d])
                    row_output.append(combined)
                batch_output.append(row_output)
            output.append(batch_output)
        
        return output


# 工厂函数
def embedding(num_embeddings: int, embedding_dim: int, **kwargs) -> Embedding:
    """创建嵌入层"""
    return Embedding(num_embeddings, embedding_dim, **kwargs)


def positional_encoding(d_model: int, max_len: int = 5000, **kwargs) -> PositionalEncoding:
    """创建正弦位置编码"""
    return PositionalEncoding(d_model, max_len, **kwargs)


def learned_positional_encoding(d_model: int, max_len: int = 512, **kwargs) -> LearnedPositionalEncoding:
    """创建可学习位置编码"""
    return LearnedPositionalEncoding(d_model, max_len, **kwargs)


def rotary_positional_embedding(d_model: int, max_len: int = 2048, **kwargs) -> RotaryPositionalEmbedding:
    """创建旋转位置嵌入"""
    return RotaryPositionalEmbedding(d_model, max_len, **kwargs)


def alibi_positional_embedding(num_heads: int, max_len: int = 2048) -> ALiBiPositionalEmbedding:
    """创建ALiBi位置嵌入"""
    return ALiBiPositionalEmbedding(num_heads, max_len)


def token_embedding(vocab_size: int, d_model: int, **kwargs) -> TokenEmbedding:
    """创建Token嵌入层"""
    return TokenEmbedding(vocab_size, d_model, **kwargs)


def segment_embedding(d_model: int, num_segments: int = 2) -> SegmentEmbedding:
    """创建分段嵌入层"""
    return SegmentEmbedding(d_model, num_segments)


def bert_embedding(vocab_size: int, d_model: int, **kwargs) -> BERTEmbedding:
    """创建BERT嵌入层"""
    return BERTEmbedding(vocab_size, d_model, **kwargs)


def sinusoidal_time_embedding(dim: int) -> SinusoidalTimeEmbedding:
    """创建正弦时间嵌入"""
    return SinusoidalTimeEmbedding(dim)


def fourier_feature_embedding(input_dim: int, embedding_dim: int, **kwargs) -> FourierFeatureEmbedding:
    """创建傅里叶特征嵌入"""
    return FourierFeatureEmbedding(input_dim, embedding_dim, **kwargs)


def relative_positional_encoding(d_model: int, max_relative_position: int = 128) -> RelativePositionalEncoding:
    """创建相对位置编码"""
    return RelativePositionalEncoding(d_model, max_relative_position)


def t5_relative_positional_encoding(num_buckets: int = 32, max_distance: int = 128) -> T5RelativePositionalEncoding:
    """创建T5风格相对位置编码"""
    return T5RelativePositionalEncoding(num_buckets, max_distance)


def positional_encoding_2d(d_model: int, height: int, width: int) -> PositionalEncoding2D:
    """创建2D位置编码"""
    return PositionalEncoding2D(d_model, height, width)
