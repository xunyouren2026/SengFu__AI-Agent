"""
Transformer变体模块 - 包含各种高效Transformer的真实实现
包括: Reformer, Longformer, BigBird, Performer, Linformer, 
      Local Attention, Memory-efficient Transformer
"""

import math
import random
from typing import Optional, Tuple, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def softmax(x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def matmul(a, b):
    if isinstance(a[0], list) and isinstance(b[0], list):
        return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    elif isinstance(a[0], list):
        return [sum(a[i][k] * b[k] for k in range(len(b))) for i in range(len(a))]
    elif isinstance(b[0], list):
        return [sum(a[k] * b[k][j] for k in range(len(a))) for j in range(len(b[0]))]
    else:
        return sum(a[k] * b[k] for k in range(len(a)))


def transpose(x: Union[List[List[float]], 'torch.Tensor']) -> Union[List[List[float]], 'torch.Tensor']:
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


class LayerNorm:
    """层归一化"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = [1.0] * normalized_shape
        self.bias = [0.0] * normalized_shape
    
    def forward(self, x: Union[List[List[float]], 'torch.Tensor']) -> Union[List[List[float]], 'torch.Tensor']:
        result = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            std = math.sqrt(var + self.eps)
            normalized = [(row[i] - mean) / std * self.weight[i] + self.bias[i]
                         for i in range(len(row))]
            result.append(normalized)
        return result


class PositionalEncoding:
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        
        # 预计算位置编码
        self.pe = []
        for pos in range(max_len):
            row = []
            for i in range(d_model):
                if i % 2 == 0:
                    row.append(math.sin(pos / (10000 ** (i / d_model))))
                else:
                    row.append(math.cos(pos / (10000 ** ((i - 1) / d_model))))
            self.pe.append(row)
    
    def forward(self, x: Union[List[List[float]], 'torch.Tensor'], offset: int = 0) -> Union[List[List[float]], 'torch.Tensor']:
        seq_len = len(x)
        return [[x[i][j] + self.pe[offset + i][j] for j in range(self.d_model)]
                for i in range(seq_len)]


class MultiHeadAttention:
    """标准多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def forward(self, query, key, value, mask=None):
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            k_t = transpose(k_heads[h])
            scores = matmul(q_heads[h], k_t)
            scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
            
            if mask is not None:
                scores = [[scores[i][j] + mask[i][j] for j in range(len(scores[0]))]
                         for i in range(len(scores))]
            
            attn = [softmax(row) for row in scores]
            head = matmul(attn, v_heads[h])
            heads.append(head)
        
        concat = self._concat_heads(heads)
        return matmul(concat, self.w_o)
    
    def _split_heads(self, x):
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
    
    def _concat_heads(self, heads):
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class ReformerAttention:
    """
    Reformer注意力 - 使用LSH (Locality Sensitive Hashing)
    
    通过哈希将相似向量分组，只计算组内注意力
    复杂度从O(n^2)降到O(n log n)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_hashes: int = 4,
        bucket_size: int = 64,
        causal: bool = False
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_hashes = num_hashes
        self.bucket_size = bucket_size
        self.causal = causal
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
        
        # LSH投影向量
        self.hash_projections = [[random.gauss(0, 1) for _ in range(self.d_k)]
                                 for _ in range(num_hashes)]
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _lsh_hash(self, vectors: Union[List[List[float]], 'torch.Tensor']) -> List[int]:
        """LSH哈希"""
        hashes = []
        for vec in vectors:
            # 对每个哈希函数计算
            hash_vals = []
            for proj in self.hash_projections:
                # 计算投影并取整
                projection = sum(v * p for v, p in zip(vec, proj))
                hash_val = int(projection / self.bucket_size)
                hash_vals.append(hash_val)
            # 组合所有哈希值
            combined = sum(h * (2 ** i) for i, h in enumerate(hash_vals))
            hashes.append(combined)
        return hashes
    
    def forward(self, query, key, value, mask=None):
        seq_len = len(query)
        
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            # LSH哈希
            hashes = self._lsh_hash(k_heads[h])
            
            # 按哈希值排序
            sorted_indices = sorted(range(seq_len), key=lambda i: hashes[i])
            
            # 分桶计算注意力
            output = [[0.0] * self.d_k for _ in range(seq_len)]
            
            num_buckets = (seq_len + self.bucket_size - 1) // self.bucket_size
            
            for bucket_idx in range(num_buckets):
                start = bucket_idx * self.bucket_size
                end = min(start + self.bucket_size, seq_len)
                
                bucket_indices = sorted_indices[start:end]
                
                if len(bucket_indices) == 0:
                    continue
                
                # 获取桶内的Q, K, V
                q_bucket = [q_heads[h][i] for i in bucket_indices]
                k_bucket = [k_heads[h][i] for i in bucket_indices]
                v_bucket = [v_heads[h][i] for i in bucket_indices]
                
                # 计算桶内注意力
                k_t = transpose(k_bucket)
                scores = matmul(q_bucket, k_t)
                scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
                
                # 因果mask
                if self.causal:
                    for i in range(len(scores)):
                        for j in range(len(scores[0])):
                            if bucket_indices[i] < bucket_indices[j]:
                                scores[i][j] = -1e9
                
                attn = [softmax(row) for row in scores]
                bucket_output = matmul(attn, v_bucket)
                
                # 写回输出
                for i, idx in enumerate(bucket_indices):
                    output[idx] = bucket_output[i]
            
            heads.append(output)
        
        concat = self._concat_heads(heads)
        return matmul(concat, self.w_o)
    
    def _split_heads(self, x):
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
    
    def _concat_heads(self, heads):
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class LongformerAttention:
    """
    Longformer注意力 - 滑动窗口 + 全局注意力
    
    每个位置关注:
    1. 局部窗口内的位置 (滑动窗口注意力)
    2. 全局关注的位置 (全局注意力)
    
    复杂度O(n * w + n * g)，其中w是窗口大小，g是全局位置数
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 512,
        num_global_tokens: int = 1,
        dilation: int = 1
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.dilation = dilation
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def forward(self, query, key, value, global_mask=None):
        seq_len = len(query)
        half_window = self.window_size // 2
        
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        # 确定全局位置
        global_positions = set(range(min(self.num_global_tokens, seq_len)))
        if global_mask is not None:
            global_positions.update(i for i in range(seq_len) if global_mask[i])
        
        heads = []
        for h in range(self.num_heads):
            output = [[0.0] * self.d_k for _ in range(seq_len)]
            
            for i in range(seq_len):
                # 确定关注的位置
                attend_positions = set()
                
                # 局部窗口
                for j in range(max(0, i - half_window), min(seq_len, i + half_window + 1)):
                    attend_positions.add(j)
                
                # 全局位置
                attend_positions.update(global_positions)
                
                # 如果当前位置是全局的，关注所有位置
                if i in global_positions:
                    attend_positions = set(range(seq_len))
                
                attend_list = sorted(attend_positions)
                
                # 计算注意力
                q_i = q_heads[h][i]
                k_attend = [k_heads[h][j] for j in attend_list]
                v_attend = [v_heads[h][j] for j in attend_list]
                
                scores = [sum(q_i[d] * k_attend[j][d] for d in range(self.d_k)) / math.sqrt(self.d_k)
                         for j in range(len(attend_list))]
                
                attn = softmax(scores)
                
                output[i] = [sum(attn[j] * v_attend[j][d] for j in range(len(attend_list)))
                            for d in range(self.d_k)]
            
            heads.append(output)
        
        concat = self._concat_heads(heads)
        return matmul(concat, self.w_o)
    
    def _split_heads(self, x):
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
    
    def _concat_heads(self, heads):
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class BigBirdAttention:
    """
    BigBird注意力 - 稀疏注意力模式
    
    结合三种注意力:
    1. 随机注意力: 随机选择的位置
    2. 滑动窗口注意力: 局部窗口
    3. 全局注意力: 特定全局位置
    
    复杂度O(n * (w + r + g))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 64,
        num_random_blocks: int = 3,
        num_global_blocks: int = 1,
        window_blocks: int = 3
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.num_global_blocks = num_global_blocks
        self.window_blocks = window_blocks
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def forward(self, query, key, value):
        seq_len = len(query)
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            output = [[0.0] * self.d_k for _ in range(seq_len)]
            
            for block_idx in range(num_blocks):
                block_start = block_idx * self.block_size
                block_end = min(block_start + self.block_size, seq_len)
                
                # 确定关注的块
                attend_blocks = set()
                
                # 全局块
                for g in range(min(self.num_global_blocks, num_blocks)):
                    attend_blocks.add(g)
                    attend_blocks.add(num_blocks - 1 - g)
                
                # 窗口块
                for w in range(-self.window_blocks, self.window_blocks + 1):
                    neighbor = block_idx + w
                    if 0 <= neighbor < num_blocks:
                        attend_blocks.add(neighbor)
                
                # 随机块
                available = [b for b in range(num_blocks) if b not in attend_blocks]
                random_blocks = random.sample(available, min(self.num_random_blocks, len(available)))
                attend_blocks.update(random_blocks)
                
                # 如果当前块是全局块，关注所有块
                if block_idx < self.num_global_blocks or block_idx >= num_blocks - self.num_global_blocks:
                    attend_blocks = set(range(num_blocks))
                
                # 计算注意力
                for i in range(block_start, block_end):
                    attend_positions = []
                    for attend_block in attend_blocks:
                        ab_start = attend_block * self.block_size
                        ab_end = min(ab_start + self.block_size, seq_len)
                        attend_positions.extend(range(ab_start, ab_end))
                    
                    q_i = q_heads[h][i]
                    k_attend = [k_heads[h][j] for j in attend_positions]
                    v_attend = [v_heads[h][j] for j in attend_positions]
                    
                    scores = [sum(q_i[d] * k_attend[j][d] for d in range(self.d_k)) / math.sqrt(self.d_k)
                             for j in range(len(attend_positions))]
                    
                    attn = softmax(scores)
                    output[i] = [sum(attn[j] * v_attend[j][d] for j in range(len(attend_positions)))
                                for d in range(self.d_k)]
            
            heads.append(output)
        
        concat = self._concat_heads(heads)
        return matmul(concat, self.w_o)
    
    def _split_heads(self, x):
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
    
    def _concat_heads(self, heads):
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class PerformerAttention:
    """
    Performer注意力 - 使用随机特征近似
    
    使用随机特征方法(Favor+)近似softmax注意力
    复杂度O(n * d^2)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_features: int = 256,
        kernel_type: str = 'exp'
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_features = num_features
        self.kernel_type = kernel_type
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
        
        # 随机特征投影
        self.projection_matrix = [[random.gauss(0, 1) for _ in range(self.d_k)]
                                  for _ in range(num_features)]
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _kernel_feature(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """计算核特征映射"""
        # 缩放
        scale = 1.0 / math.sqrt(2 * self.d_k)
        x_scaled = [xi * scale for xi in x]
        
        # 投影
        projected = [sum(x_scaled[d] * self.projection_matrix[f][d] for d in range(self.d_k))
                    for f in range(self.num_features)]
        
        if self.kernel_type == 'exp':
            # 指数核
            norm_sq = sum(xi ** 2 for xi in x_scaled)
            feature = [math.exp(p - norm_sq / 2) for p in projected]
        elif self.kernel_type == 'relu':
            # ReLU核
            feature = [max(0, p) for p in projected]
        else:
            feature = projected
        
        return feature
    
    def forward(self, query, key, value):
        seq_len = len(query)
        
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        heads = []
        for h in range(self.num_heads):
            # 计算核特征
            q_features = [self._kernel_feature(q_heads[h][i]) for i in range(seq_len)]
            k_features = [self._kernel_feature(k_heads[h][i]) for i in range(seq_len)]
            
            # 计算K^T V (d_k x d_k)
            kv = [[sum(k_features[i][f] * v_heads[h][i][d] for i in range(seq_len))
                  for d in range(self.d_k)]
                 for f in range(self.num_features)]
            
            # 计算sum(K) (d_k,)
            sum_k = [sum(k_features[i][f] for i in range(seq_len))
                    for f in range(self.num_features)]
            
            # 计算Q (K^T V)
            qkv = [[sum(q_features[i][f] * kv[f][d] for f in range(self.num_features))
                   for d in range(self.d_k)]
                  for i in range(seq_len)]
            
            # 计算Q sum(K)
            q_sum_k = [sum(q_features[i][f] * sum_k[f] for f in range(self.num_features))
                      for i in range(seq_len)]
            
            # 归一化
            output = [[qkv[i][d] / (q_sum_k[i] + 1e-9) for d in range(self.d_k)]
                     for i in range(seq_len)]
            
            heads.append(output)
        
        concat = self._concat_heads(heads)
        return matmul(concat, self.w_o)
    
    def _split_heads(self, x):
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
    
    def _concat_heads(self, heads):
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class LinformerAttention:
    """
    Linformer注意力 - 低秩近似
    
    将K和V投影到低维空间
    复杂度O(n * k)，其中k是投影维度
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_len: int,
        k: int = 256,
        shared_projection: bool = True
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_len = seq_len
        self.k = k
        
        self.w_q = self._init_weight(d_model, d_model)
        self.w_k = self._init_weight(d_model, d_model)
        self.w_v = self._init_weight(d_model, d_model)
        self.w_o = self._init_weight(d_model, d_model)
        
        # 投影矩阵 E: n -> k
        if shared_projection:
            self.proj_k = self._init_projection(seq_len, k)
            self.proj_v = self.proj_k
        else:
            self.proj_k = self._init_projection(seq_len, k)
            self.proj_v = self._init_projection(seq_len, k)
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _init_projection(self, seq_len: int, k: int):
        # 使用简单的平均投影
        scale = 1.0 / math.sqrt(seq_len // k)
        proj = [[0.0] * k for _ in range(seq_len)]
        block_size = seq_len // k
        for i in range(seq_len):
            j = min(i // block_size, k - 1)
            proj[i][j] = scale
        return proj
    
    def forward(self, query, key, value):
        seq_len = len(query)
        
        q = matmul(query, self.w_q)
        k = matmul(key, self.w_k)
        v = matmul(value, self.w_v)
        
        # 投影K和V到低维
        # K_proj = E^T K, V_proj = F^T V
        k_proj_t = transpose(self.proj_k[:seq_len])
        v_proj_t = transpose(self.proj_v[:seq_len])
        
        k_projected = matmul(k_proj_t, k)  # k x d_model
        v_projected = matmul(v_proj_t, v)  # k x d_model
        
        q_heads = self._split_heads(q)
        k_heads = self._split_heads_projected(k_projected)
        v_heads = self._split_heads_projected(v_projected)
        
        heads = []
        for h in range(self.num_heads):
            k_t = transpose(k_heads[h])
            scores = matmul(q_heads[h], k_t)
            scores = [[s / math.sqrt(self.d_k) for s in row] for row in scores]
            
            attn = [softmax(row) for row in scores]
            head = matmul(attn, v_heads[h])
            heads.append(head)
        
        concat = self._concat_heads(heads)
        return matmul(concat, self.w_o)
    
    def _split_heads(self, x):
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
    
    def _split_heads_projected(self, x):
        k = len(x)
        result = []
        for h in range(self.num_heads):
            head = []
            for i in range(k):
                start = h * self.d_k
                end = start + self.d_k
                head.append(x[i][start:end])
            result.append(head)
        return result
    
    def _concat_heads(self, heads):
        seq_len = len(heads[0])
        result = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][i])
            result.append(concat)
        return result


class TransformerBlock:
    """Transformer块"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attention_type: str = 'standard',
        dropout: float = 0.1,
        **attention_kwargs
    ):
        self.d_model = d_model
        
        # 选择注意力类型
        if attention_type == 'standard':
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        elif attention_type == 'reformer':
            self.attention = ReformerAttention(d_model, num_heads, **attention_kwargs)
        elif attention_type == 'longformer':
            self.attention = LongformerAttention(d_model, num_heads, **attention_kwargs)
        elif attention_type == 'bigbird':
            self.attention = BigBirdAttention(d_model, num_heads, **attention_kwargs)
        elif attention_type == 'performer':
            self.attention = PerformerAttention(d_model, num_heads, **attention_kwargs)
        elif attention_type == 'linformer':
            self.attention = LinformerAttention(d_model, num_heads, **attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # FFN
        self.ffn_w1 = self._init_weight(d_model, d_ff)
        self.ffn_w2 = self._init_weight(d_ff, d_model)
    
    def _init_weight(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_out = self.attention.forward(x, x, x, mask)
        x = [[x[i][j] + attn_out[i][j] for j in range(self.d_model)] for i in range(len(x))]
        x = self.norm1.forward(x)
        
        # FFN
        ffn_hidden = matmul(x, self.ffn_w1)
        ffn_hidden = [[max(0, val) for val in row] for row in ffn_hidden]  # ReLU
        ffn_out = matmul(ffn_hidden, self.ffn_w2)
        x = [[x[i][j] + ffn_out[i][j] for j in range(self.d_model)] for i in range(len(x))]
        x = self.norm2.forward(x)
        
        return x


class EfficientTransformer:
    """
    高效Transformer - 支持多种注意力机制
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        attention_type: str = 'standard',
        vocab_size: int = 30000,
        **attention_kwargs
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 嵌入层
        self.embedding = [[random.gauss(0, 0.02) for _ in range(d_model)]
                         for _ in range(vocab_size)]
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer层
        self.layers = [
            TransformerBlock(d_model, num_heads, d_ff, attention_type, **attention_kwargs)
            for _ in range(num_layers)
        ]
        
        # 输出层
        self.output_norm = LayerNorm(d_model)
    
    def forward(self, input_ids: List[int], mask=None):
        # 嵌入
        x = [self.embedding[idx] for idx in input_ids]
        
        # 位置编码
        x = self.pos_encoding.forward(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        # 输出归一化
        x = self.output_norm.forward(x)
        
        return x


# 工厂函数
def get_efficient_transformer(name: str, **kwargs) -> EfficientTransformer:
    """根据名称获取高效Transformer"""
    attention_types = {
        'standard': 'standard',
        'reformer': 'reformer',
        'longformer': 'longformer',
        'bigbird': 'bigbird',
        'performer': 'performer',
        'linformer': 'linformer'
    }
    
    name_lower = name.lower()
    if name_lower not in attention_types:
        raise ValueError(f"Unknown transformer type: {name}. Available: {list(attention_types.keys())}")
    
    kwargs['attention_type'] = attention_types[name_lower]
    return EfficientTransformer(**kwargs)
