"""
注意力机制模块 - 包含各种注意力机制的真实实现
包括: Scaled Dot-Product Attention, Multi-Head Attention, Linear Attention,
      Flash Attention, Multi-Query Attention, Grouped Query Attention,
      Sliding Window Attention, Sparse Attention, etc.
"""

import math
import random
from typing import Optional, Tuple, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def softmax(x: Union[List[float], 'torch.Tensor'], axis: int = -1) -> Union[List[float], 'torch.Tensor']:
    """计算softmax"""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def matmul(a: Union[List[List[float]], 'torch.Tensor'], b: Union[List[List[float]], 'torch.Tensor']) -> Union[List[List[float]], 'torch.Tensor']:
    """矩阵乘法"""
    m = len(a)
    n = len(b[0])
    k = len(b)
    result = [[sum(a[i][l] * b[l][j] for l in range(k)) for j in range(n)] for i in range(m)]
    return result


def transpose(x: Union[List[List[float]], 'torch.Tensor']) -> Union[List[List[float]], 'torch.Tensor']:
    """矩阵转置"""
    return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]



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


class Attention(ABC):
    """注意力机制基类"""
    
    @abstractmethod
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        pass
    
    def __call__(self, query, key, value, mask=None):
        return self.forward(query, key, value, mask)


class ScaledDotProductAttention(Attention):
    """
    缩放点积注意力
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout: float = 0.0):
        self.dropout = dropout
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        """
        query: [seq_len_q, d_k]
        key: [seq_len_k, d_k]
        value: [seq_len_k, d_v]
        mask: [seq_len_q, seq_len_k]
        
        返回: [seq_len_q, d_v]
        """
        d_k = len(query[0])
        
        # QK^T: [seq_len_q, seq_len_k]
        k_t = transpose(key)
        scores = matmul(query, k_t)
        
        # 缩放
        scale = 1.0 / math.sqrt(d_k)
        scores = [[s * scale for s in row] for row in scores]
        
        # 应用mask
        if mask is not None:
            scores = [[scores[i][j] + (mask[i][j] if mask[i][j] < 0 else -1e9)
                      for j in range(len(scores[0]))] for i in range(len(scores))]
        
        # Softmax
        attn_weights = [softmax(row) for row in scores]
        
        # Dropout
        if self.dropout > 0:
            attn_weights = [[w if random.random() > self.dropout else 0.0
                           for w in row] for row in attn_weights]
        
        # 乘以V
        output = matmul(attn_weights, value)
        
        return output


class MultiHeadAttention(Attention):
    """
    多头注意力
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    其中 head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.dropout = dropout
        
        # 初始化权重 (简化：使用随机初始化)
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
        
        if bias:
            self.b_q = [0.0] * d_model
            self.b_k = [0.0] * d_model
            self.b_v = [0.0] * d_model
            self.b_o = [0.0] * d_model
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None
        
        self.scaled_dot_attn = ScaledDotProductAttention(dropout)
    
    def _init_weight(self, in_features: int, out_features: int) -> Union[List[List[float]], 'torch.Tensor']:
        """Xavier初始化"""
        scale = math.sqrt(2.0 / (in_features + out_features))
        return [[random.gauss(0, scale) for _ in range(out_features)] 
                for _ in range(in_features)]
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        """
        query, key, value: [batch_size, seq_len, d_model]
        这里简化为 [seq_len, d_model]
        """
        batch_size = len(query)
        
        # 线性投影
        q = self._linear(query, self.w_q, self.b_q)  # [seq_len, d_model]
        k = self._linear(key, self.w_k, self.b_k)
        v = self._linear(value, self.w_v, self.b_v)
        
        # 分割为多头
        q = self._split_heads(q)  # [num_heads, seq_len, d_k]
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # 对每个头计算注意力
        heads = []
        for i in range(self.num_heads):
            head = self.scaled_dot_attn(q[i], k[i], v[i], mask)
            heads.append(head)
        
        # 拼接头
        concat = self._concat_heads(heads)  # [seq_len, d_model]
        
        # 输出投影
        output = self._linear(concat, self.w_o, self.b_o)
        
        return output
    
    def _linear(
        self,
        x: Union[List[List[float]], 'torch.Tensor'],
        weight: Union[List[List[float]], 'torch.Tensor'],
        bias: Optional[Union[List[float], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        """线性变换"""
        result = matmul(x, weight)
        if bias is not None:
            result = [[result[i][j] + bias[j] for j in range(len(result[0]))]
                     for i in range(len(result))]
        return result
    
    def _split_heads(self, x: Union[List[List[float]], 'torch.Tensor']) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """分割为多头"""
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        """拼接多头"""
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class MultiQueryAttention(Attention):
    """
    Multi-Query Attention (MQA)
    
    多个查询头共享单个键和值头
    减少KV cache大小，提高推理效率
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Q有多个头，K和V只有单个头
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, self.d_k)  # 单个头
        self.w_v = self._init_weight(d_model, self.d_k)  # 单个头
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_features: int, out_features: int) -> Union[List[List[float]], 'torch.Tensor']:
        scale = math.sqrt(2.0 / (in_features + out_features))
        return [[random.gauss(0, scale) for _ in range(out_features)]
                for _ in range(in_features)]
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(query)
        
        # 线性投影
        q = matmul(query, self.w_q)  # [seq_len, d_model]
        k = matmul(key, self.w_k)    # [seq_len, d_k]
        v = matmul(value, self.w_v)  # [seq_len, d_k]
        
        # 分割Q为多头
        q_heads = self._split_heads(q)
        
        # 每个头使用相同的K和V
        heads = []
        for h in range(self.num_heads):
            # 计算注意力分数
            k_t = transpose(k)
            scores = matmul(q_heads[h], k_t)
            scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
            
            if mask is not None:
                scores = [[scores[i][j] + (mask[i][j] if mask[i][j] < 0 else -1e9)
                          for j in range(len(scores[0]))] for i in range(len(scores))]
            
            attn_weights = [softmax(row) for row in scores]
            head = matmul(attn_weights, v)
            heads.append(head)
        
        # 拼接和输出投影
        concat = self._concat_heads(heads)
        output = matmul(concat, self.w_o)
        
        return output
    
    def _split_heads(self, x: Union[List[List[float]], 'torch.Tensor']) -> List[Union[List[List[float]], 'torch.Tensor']]:
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class GroupedQueryAttention(Attention):
    """
    Grouped Query Attention (GQA)
    
    将查询头分组，每组共享一个键值头
    是MQA和MHA的泛化
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0
    ):
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_heads_per_group = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Q有num_heads个头，K和V有num_kv_heads个头
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, num_kv_heads * self.d_k)
        self.w_v = self._init_weight(d_model, num_kv_heads * self.d_k)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_features: int, out_features: int) -> Union[List[List[float]], 'torch.Tensor']:
        scale = math.sqrt(2.0 / (in_features + out_features))
        return [[random.gauss(0, scale) for _ in range(out_features)]
                for _ in range(in_features)]
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(query)
        
        # 线性投影
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        # 分割为头
        q_heads = self._split_heads(q, self.num_heads)
        k_heads = self._split_heads(k, self.num_kv_heads)
        v_heads = self._split_heads(v, self.num_kv_heads)
        
        # 计算注意力
        heads = []
        for h in range(self.num_heads):
            # 找到对应的KV头
            kv_h = h // self.num_heads_per_group
            
            k_t = transpose(k_heads[kv_h])
            scores = matmul(q_heads[h], k_t)
            scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
            
            if mask is not None:
                scores = [[scores[i][j] + (mask[i][j] if mask[i][j] < 0 else -1e9)
                          for j in range(len(scores[0]))] for i in range(len(scores))]
            
            attn_weights = [softmax(row) for row in scores]
            head = matmul(attn_weights, v_heads[kv_h])
            heads.append(head)
        
        concat = self._concat_heads(heads)
        output = matmul(concat, self.w_o)
        
        return output
    
    def _split_heads(self, x: Union[List[List[float]], 'torch.Tensor'], num_heads: int) -> List[Union[List[List[float]], 'torch.Tensor']]:
        seq_len = len(x)
        d_per_head = len(x[0]) // num_heads
        result = []
        for h in range(num_heads):
            head = []
            for i in range(seq_len):
                start = h * d_per_head
                end = start + d_per_head
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(len(heads)):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class LinearAttention(Attention):
    """
    线性注意力
    
    使用核函数近似softmax，将复杂度从O(n^2)降到O(n)
    
    Attention(Q, K, V) = phi(Q) (phi(K)^T V) / phi(Q) sum(phi(K))
    其中phi是特征映射函数
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feature_map: str = 'elu'
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_map = feature_map
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_features: int, out_features: int) -> Union[List[List[float]], 'torch.Tensor']:
        scale = math.sqrt(2.0 / (in_features + out_features))
        return [[random.gauss(0, scale) for _ in range(out_features)]
                for _ in range(in_features)]
    
    def _phi(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """特征映射"""
        if self.feature_map == 'elu':
            # ELU+1: phi(x) = elu(x) + 1
            return [max(0, xi) + math.exp(min(0, xi)) - 1 + 1 for xi in x]
        elif self.feature_map == 'relu':
            # ReLU+1: phi(x) = relu(x) + 1
            return [max(0, xi) + 1 for xi in x]
        elif self.feature_map == 'softmax':
            # Softmax
            return softmax(x)
        else:
            return x
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(query)
        
        # 线性投影
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        # 分割为头
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            # 应用特征映射
            phi_q = [self._phi(vec) for vec in q_heads[h]]
            phi_k = [self._phi(vec) for vec in k_heads[h]]
            
            # 计算 phi(K)^T V: [d_k, d_k]
            phi_k_t = transpose(phi_k)
            kv = matmul(phi_k_t, v_heads[h])
            
            # 计算 sum(phi(K)): [d_k]
            sum_phi_k = [sum(phi_k[i][j] for i in range(seq_len))
                        for j in range(self.d_k)]
            
            # 计算 phi(Q) (phi(K)^T V): [seq_len, d_k]
            qkv = matmul(phi_q, kv)
            
            # 计算 phi(Q) sum(phi(K)): [seq_len, d_k]
            q_sum = [[sum(phi_q[i][j] * sum_phi_k[j] for j in range(self.d_k))]
                    for i in range(seq_len)]
            
            # 归一化
            head = [[qkv[i][j] / (q_sum[i][0] + 1e-9) for j in range(self.d_k)]
                   for i in range(seq_len)]
            heads.append(head)
        
        concat = self._concat_heads(heads)
        output = matmul(concat, self.w_o)
        
        return output
    
    def _split_heads(self, x: Union[List[List[float]], 'torch.Tensor']) -> List[Union[List[List[float]], 'torch.Tensor']]:
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class FlashAttention(Attention):
    """
    Flash Attention (简化版)
    
    通过分块计算和重计算减少内存访问
    这是概念实现，实际Flash Attention需要CUDA kernel
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 64,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        self.dropout = dropout
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_features: int, out_features: int) -> Union[List[List[float]], 'torch.Tensor']:
        scale = math.sqrt(2.0 / (in_features + out_features))
        return [[random.gauss(0, scale) for _ in range(out_features)]
                for _ in range(in_features)]
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(query)
        
        # 线性投影
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        # 分割为头
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            # 分块计算
            head = self._flash_attention_block(q_heads[h], k_heads[h], v_heads[h])
            heads.append(head)
        
        concat = self._concat_heads(heads)
        output = matmul(concat, self.w_o)
        
        return output
    
    def _flash_attention_block(
        self,
        q: Union[List[List[float]], 'torch.Tensor'],
        k: Union[List[List[float]], 'torch.Tensor'],
        v: Union[List[List[float]], 'torch.Tensor']
    ) -> Union[List[List[float]], 'torch.Tensor']:
        """分块计算注意力"""
        seq_len = len(q)
        block_size = min(self.block_size, seq_len)
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # 初始化输出和log-sum-exp
        output = [[0.0] * self.d_k for _ in range(seq_len)]
        lse = [-float('inf')] * seq_len
        
        scale = 1.0 / math.sqrt(self.d_k)
        
        # 分块计算
        for i in range(num_blocks):
            q_start = i * block_size
            q_end = min((i + 1) * block_size, seq_len)
            q_block = q[q_start:q_end]
            
            # 初始化块的累加器
            block_output = [[0.0] * self.d_k for _ in range(q_end - q_start)]
            block_lse = [-float('inf')] * (q_end - q_start)
            
            for j in range(num_blocks):
                k_start = j * block_size
                k_end = min((j + 1) * block_size, seq_len)
                k_block = k[k_start:k_end]
                v_block = v[k_start:k_end]
                
                # 计算当前块的分数
                k_t = transpose(k_block)
                scores = matmul(q_block, k_t)
                scores = [[s * scale for s in row] for row in scores]
                
                # 在线softmax更新
                for bi in range(q_end - q_start):
                    row_scores = scores[bi]
                    row_max = max(row_scores)
                    
                    # 计算新的log-sum-exp
                    new_lse = math.log(
                        math.exp(block_lse[bi]) + 
                        sum(math.exp(s - row_max) for s in row_scores)
                    ) + row_max
                    
                    # 更新输出
                    exp_scores = [math.exp(s - new_lse) for s in row_scores]
                    for bj, exp_s in enumerate(exp_scores):
                        for d in range(self.d_k):
                            block_output[bi][d] += exp_s * v_block[bj][d]
                    
                    block_lse[bi] = new_lse
            
            # 将块结果写入输出
            for bi in range(q_end - q_start):
                output[q_start + bi] = block_output[bi]
                lse[q_start + bi] = block_lse[bi]
        
        return output
    
    def _split_heads(self, x: Union[List[List[float]], 'torch.Tensor']) -> List[Union[List[List[float]], 'torch.Tensor']]:
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class SlidingWindowAttention(Attention):
    """
    滑动窗口注意力
    
    每个位置只关注局部窗口内的位置
    复杂度O(n * w)，其中w是窗口大小
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 512,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_features: int, out_features: int) -> Union[List[List[float]], 'torch.Tensor']:
        scale = math.sqrt(2.0 / (in_features + out_features))
        return [[random.gauss(0, scale) for _ in range(out_features)]
                for _ in range(in_features)]
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(query)
        half_window = self.window_size // 2
        
        # 线性投影
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        # 分割为头
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                # 确定窗口范围
                start = max(0, i - half_window)
                end = min(seq_len, i + half_window + 1)
                
                # 获取窗口内的K和V
                k_window = k_heads[h][start:end]
                v_window = v_heads[h][start:end]
                
                # 计算注意力
                q_i = [q_heads[h][i]]
                k_t = transpose(k_window)
                scores = matmul(q_i, k_t)
                scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
                
                attn_weights = [softmax(row) for row in scores]
                out_i = matmul(attn_weights, v_window)
                head.append(out_i[0])
            
            heads.append(head)
        
        concat = self._concat_heads(heads)
        output = matmul(concat, self.w_o)
        
        return output
    
    def _split_heads(self, x: Union[List[List[float]], 'torch.Tensor']) -> List[Union[List[List[float]], 'torch.Tensor']]:
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class SparseAttention(Attention):
    """
    稀疏注意力
    
    只计算特定位置的注意力，其他位置设为-inf
    支持多种稀疏模式
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sparse_pattern: str = 'strided',
        stride: int = 128,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sparse_pattern = sparse_pattern
        self.stride = stride
        self.dropout = dropout
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_features: int, out_features: int) -> Union[List[List[float]], 'torch.Tensor']:
        scale = math.sqrt(2.0 / (in_features + out_features))
        return [[random.gauss(0, scale) for _ in range(out_features)]
                for _ in range(in_features)]
    
    def _get_sparse_mask(self, seq_len: int) -> Union[List[List[float]], 'torch.Tensor']:
        """生成稀疏mask"""
        mask = [[-float('inf')] * seq_len for _ in range(seq_len)]
        
        if self.sparse_pattern == 'strided':
            # Strided pattern: 每个位置关注stride倍数的位置
            for i in range(seq_len):
                for j in range(seq_len):
                    if j % self.stride == 0 or abs(i - j) <= self.stride // 2:
                        mask[i][j] = 0.0
        
        elif self.sparse_pattern == 'fixed':
            # Fixed pattern: 每个位置关注固定的几个位置
            for i in range(seq_len):
                for j in range(seq_len):
                    if j < self.stride or abs(i - j) <= self.stride // 4:
                        mask[i][j] = 0.0
        
        elif self.sparse_pattern == 'block':
            # Block pattern: 分块注意力
            block_size = self.stride
            for i in range(seq_len):
                block_i = i // block_size
                for j in range(seq_len):
                    block_j = j // block_size
                    if block_i == block_j:
                        mask[i][j] = 0.0
        
        return mask
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(query)
        
        # 生成稀疏mask
        sparse_mask = self._get_sparse_mask(seq_len)
        
        # 合并用户提供的mask
        if mask is not None:
            for i in range(seq_len):
                for j in range(seq_len):
                    if mask[i][j] < 0:
                        sparse_mask[i][j] = -float('inf')
        
        # 线性投影
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        # 分割为头
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            k_t = transpose(k_heads[h])
            scores = matmul(q_heads[h], k_t)
            scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
            
            # 应用稀疏mask
            scores = [[scores[i][j] + sparse_mask[i][j]
                      for j in range(seq_len)] for i in range(seq_len)]
            
            attn_weights = [softmax(row) for row in scores]
            head = matmul(attn_weights, v_heads[h])
            heads.append(head)
        
        concat = self._concat_heads(heads)
        output = matmul(concat, self.w_o)
        
        return output
    
    def _split_heads(self, x: Union[List[List[float]], 'torch.Tensor']) -> List[Union[List[List[float]], 'torch.Tensor']]:
        seq_len = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(seq_len):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class CrossAttention(Attention):
    """
    交叉注意力
    
    用于编码器-解码器架构，Q来自一个序列，K和V来自另一个序列
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self,
        query: Union[List[List[float]], 'torch.Tensor'],
        key: Union[List[List[float]], 'torch.Tensor'],
        value: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        return self.mha(query, key, value, mask)


class SelfAttention(Attention):
    """
    自注意力
    
    Q、K、V来自同一个序列
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self,
        x: Union[List[List[float]], 'torch.Tensor'],
        mask: Optional[Union[List[List[float]], 'torch.Tensor']] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        return self.mha(x, x, x, mask)


class RotaryPositionalEmbedding:
    """
    旋转位置编码 (RoPE)
    
    通过旋转矩阵编码相对位置信息
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        self.inv_freq = [1.0 / (base ** (2 * i / d_model)) for i in range(d_model // 2)]
        
        # 预计算cos和sin
        self.cos_cache = []
        self.sin_cache = []
        for pos in range(max_seq_len):
            theta = [pos * freq for freq in self.inv_freq]
            self.cos_cache.append([math.cos(t) for t in theta])
            self.sin_cache.append([math.sin(t) for t in theta])
    
    def forward(
        self,
        x: Union[List[List[float]], 'torch.Tensor'],
        positions: Optional[List[int]] = None
    ) -> Union[List[List[float]], 'torch.Tensor']:
        """
        x: [seq_len, d_model]
        positions: 位置索引列表
        """
        seq_len = len(x)
        
        if positions is None:
            positions = list(range(seq_len))
        
        result = []
        for i, pos in enumerate(positions):
            if pos >= self.max_seq_len:
                pos = self.max_seq_len - 1
            
            cos_theta = self.cos_cache[pos]
            sin_theta = self.sin_cache[pos]
            
            # 应用旋转
            x_i = x[i]
            rotated = []
            for j in range(self.d_model // 2):
                x1 = x_i[2 * j]
                x2 = x_i[2 * j + 1]
                rotated.append(x1 * cos_theta[j] - x2 * sin_theta[j])
                rotated.append(x1 * sin_theta[j] + x2 * cos_theta[j])
            
            result.append(rotated)
        
        return result
    
    def __call__(self, x, positions=None):
        return self.forward(x, positions)


class ALiBiPositionalBias:
    """
    ALiBi (Attention with Linear Biases)
    
    通过线性偏置编码位置信息，无需位置编码
    """
    
    def __init__(self, num_heads: int):
        self.num_heads = num_heads
        
        # 计算每个头的斜率
        # m_h = 1 / 2^((8h+4)/n) for h in [1, n]
        self.slopes = []
        for h in range(1, num_heads + 1):
            slope = 1.0 / (2 ** ((8 * h + 4) / num_heads))
            self.slopes.append(slope)
    
    def get_bias(self, seq_len: int) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        返回: [num_heads, seq_len, seq_len] 的偏置矩阵
        """
        biases = []
        for h in range(self.num_heads):
            m = self.slopes[h]
            bias = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
            biases.append(bias)
        return biases


class RelativePositionalBias:
    """
    相对位置偏置
    
    学习相对位置的偏置值
    """
    
    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        
        # 初始化可学习的偏置
        if bidirectional:
            # -max_distance 到 max_distance
            self.bias_table = [[random.gauss(0, 0.02) 
                               for _ in range(2 * max_distance + 1)]
                              for _ in range(num_heads)]
        else:
            # 0 到 max_distance
            self.bias_table = [[random.gauss(0, 0.02)
                               for _ in range(max_distance + 1)]
                              for _ in range(num_heads)]
    
    def get_bias(self, seq_len: int) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """返回相对位置偏置"""
        biases = []
        for h in range(self.num_heads):
            bias = []
            for i in range(seq_len):
                row = []
                for j in range(seq_len):
                    rel_pos = i - j
                    if self.bidirectional:
                        rel_pos = max(-self.max_distance, min(self.max_distance, rel_pos))
                        idx = rel_pos + self.max_distance
                    else:
                        rel_pos = max(0, min(self.max_distance, abs(rel_pos)))
                        idx = rel_pos
                    row.append(self.bias_table[h][idx])
                bias.append(row)
            biases.append(bias)
        return biases


# 工厂函数
def get_attention(name: str, **kwargs) -> Attention:
    """根据名称获取注意力机制"""
    attentions = {
        'scaled_dot_product': ScaledDotProductAttention,
        'multi_head': MultiHeadAttention,
        'multi_query': MultiQueryAttention,
        'grouped_query': GroupedQueryAttention,
        'linear': LinearAttention,
        'flash': FlashAttention,
        'sliding_window': SlidingWindowAttention,
        'sparse': SparseAttention,
        'cross': CrossAttention,
        'self': SelfAttention
    }
    
    name_lower = name.lower()
    if name_lower not in attentions:
        raise ValueError(f"Unknown attention: {name}. Available: {list(attentions.keys())}")
    
    return attentions[name_lower](**kwargs)
