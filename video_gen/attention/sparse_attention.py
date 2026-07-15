"""
稀疏注意力机制实现 - 包含6种稀疏注意力机制及相关辅助类
仅使用标准库，无外部依赖
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """注意力机制配置类"""
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    window_size: int = 256
    sparse_ratio: float = 0.1
    
    def __post_init__(self):
        if self.hidden_size <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("hidden_size, num_heads, head_dim must be positive")
        if not 0 < self.sparse_ratio <= 1:
            raise ValueError("sparse_ratio must be in (0, 1]")
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError(f"hidden_size must equal num_heads * head_dim")


# 数学工具函数
def softmax(x: List[float]) -> List[float]:
    """计算softmax"""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    return [e / sum(exp_x) for e in exp_x]

def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法"""
    rows_a, cols_a, cols_b = len(a), len(a[0]), len(b[0])
    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result

def transpose(matrix: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]

def apply_mask(scores: List[List[float]], mask: List[List[bool]], 
               fill_value: float = float('-inf')) -> List[List[float]]:
    """应用掩码到注意力分数"""
    return [[scores[i][j] if mask[i][j] else fill_value for j in range(len(scores[0]))]
            for i in range(len(scores))]


# 1. 滑动窗口注意力
class SlidingWindowAttention:
    """滑动窗口注意力 - 维护固定窗口，包含sink tokens和三区缓存"""
    
    def __init__(self, config: AttentionConfig, window_size: int = 256, 
                 sink_size: int = 4, compress_ratio: float = 0.5):
        self._config = config
        self._window_size = window_size
        self._sink_size = sink_size
        self._compress_ratio = compress_ratio
        self._cache: Dict[str, Any] = {
            'sink': {'k': [], 'v': []},
            'mid': {'k': [], 'v': []},
            'recent': {'k': [], 'v': []}
        }
        self._compressor_weights: List[float] = []
        self._compressor_bias: List[float] = []
        self._init_compressor()
    
    def _init_compressor(self) -> None:
        """初始化压缩器参数"""
        compressed_dim = int(self._config.head_dim * self._compress_ratio)
        scale = math.sqrt(2.0 / (self._config.head_dim + compressed_dim))
        self._compressor_weights = [
            [random.gauss(0, scale) for _ in range(self._config.head_dim)]
            for _ in range(compressed_dim)
        ]
        self._compressor_bias = [0.0 for _ in range(compressed_dim)]
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]]) -> List[List[float]]:
        """滑动窗口注意力前向传播"""
        seq_len = len(query)
        head_dim = len(query[0])
        self._update_cache(key, value)
        all_keys = self._cache['sink']['k'] + self._cache['mid']['k'] + self._cache['recent']['k']
        all_values = self._cache['sink']['v'] + self._cache['mid']['v'] + self._cache['recent']['v']
        if not all_keys:
            all_keys, all_values = key, value
        kv_len = len(all_keys)
        output = []
        for i in range(seq_len):
            q = query[i]
            scores = []
            for j in range(kv_len):
                score = sum(q[d] * all_keys[j][d] for d in range(head_dim)) / math.sqrt(head_dim)
                in_window = (j < self._sink_size) or (j >= kv_len - self._window_size)
                scores.append(score if in_window else float('-inf'))
            attn_weights = softmax(scores)
            out = [0.0 for _ in range(head_dim)]
            for j in range(kv_len):
                for d in range(head_dim):
                    out[d] += attn_weights[j] * all_values[j][d]
            output.append(out)
        return output
    
    def _update_cache(self, new_k: List[List[float]], new_v: List[List[float]]) -> None:
        """更新三区缓存"""
        seq_len = len(new_k)
        if not self._cache['sink']['k'] and seq_len >= self._sink_size:
            self._cache['sink']['k'] = new_k[:self._sink_size]
            self._cache['sink']['v'] = new_v[:self._sink_size]
            remaining_k, remaining_v = new_k[self._sink_size:], new_v[self._sink_size:]
        else:
            remaining_k, remaining_v = new_k, new_v
        self._cache['recent']['k'].extend(remaining_k)
        self._cache['recent']['v'].extend(remaining_v)
        while len(self._cache['recent']['k']) > self._window_size:
            compress_k = self._cache['recent']['k'].pop(0)
            compress_v = self._cache['recent']['v'].pop(0)
            
            # 压缩并存储到mid
            compressed_k = self._compress_cache([compress_k])
            compressed_v = self._compress_cache([compress_v])
            
            self._cache['mid']['k'].extend(compressed_k)
            self._cache['mid']['v'].extend(compressed_v)
    
    def _compress_cache(self, cache: List[List[float]]) -> List[List[float]]:
        """使用可学习压缩器压缩缓存"""
        if not cache:
            return []
        compressed = []
        for vec in cache:
            comp_vec = []
            for i in range(len(self._compressor_weights)):
                val = sum(self._compressor_weights[i][j] * vec[j] for j in range(len(vec)))
                comp_vec.append(val + self._compressor_bias[i])
            compressed.append(comp_vec)
        return compressed
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache = {'sink': {'k': [], 'v': []}, 'mid': {'k': [], 'v': []}, 'recent': {'k': [], 'v': []}}


# 2. 块稀疏注意力
class BlockSparseAttention:
    """块稀疏注意力 - 将序列划分为固定大小的块，只计算块内注意力"""
    
    def __init__(self, config: AttentionConfig, block_size: int = 64):
        self._config = config
        self._block_size = block_size
        self._num_blocks = 0
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]]) -> List[List[float]]:
        """块稀疏注意力前向传播"""
        seq_len = len(query)
        head_dim = len(query[0])
        self._num_blocks = (seq_len + self._block_size - 1) // self._block_size
        block_mask = self._build_block_mask(seq_len)
        output = []
        for i in range(seq_len):
            q = query[i]
            scores = []
            for j in range(seq_len):
                if block_mask[i][j]:
                    score = sum(q[d] * key[j][d] for d in range(head_dim)) / math.sqrt(head_dim)
                else:
                    score = float('-inf')
                scores.append(score)
            
            attn_weights = softmax(scores)
            out = [0.0 for _ in range(head_dim)]
            for j in range(seq_len):
                for d in range(head_dim):
                    out[d] += attn_weights[j] * value[j][d]
            output.append(out)
        return output
    
    def _build_block_mask(self, seq_len: int) -> List[List[bool]]:
        """构建块稀疏掩码"""
        mask = [[False for _ in range(seq_len)] for _ in range(seq_len)]
        for i in range(seq_len):
            block_idx = i // self._block_size
            block_start = block_idx * self._block_size
            block_end = min((block_idx + 1) * self._block_size, seq_len)
            for j in range(block_start, block_end):
                mask[i][j] = True
        return mask


# 3. 步长稀疏注意力
class StridedAttention:
    """步长稀疏注意力 - 按固定步长采样key/value，保留全局信息"""
    
    def __init__(self, config: AttentionConfig, stride: int = 3):
        self._config = config
        self._stride = stride
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]]) -> List[List[float]]:
        """步长稀疏注意力前向传播"""
        seq_len = len(query)
        head_dim = len(query[0])
        sampled_key = self._stride_sample(key, self._stride)
        sampled_value = self._stride_sample(value, self._stride)
        window_size = self._stride * 2
        output = []
        for i in range(seq_len):
            q = query[i]
            visible_indices = set()
            for j in range(0, seq_len, self._stride):
                visible_indices.add(j)
            window_start = max(0, i - window_size)
            window_end = min(seq_len, i + window_size + 1)
            for j in range(window_start, window_end):
                visible_indices.add(j)
            
            visible_indices = sorted(visible_indices)
            scores = [sum(q[d] * key[j][d] for d in range(head_dim)) / math.sqrt(head_dim) for j in visible_indices]
            attn_weights = softmax(scores)
            out = [0.0 for _ in range(head_dim)]
            for idx, j in enumerate(visible_indices):
                for d in range(head_dim):
                    out[d] += attn_weights[idx] * value[j][d]
            output.append(out)
        return output
    
    def _stride_sample(self, x: List[List[float]], stride: int) -> List[List[float]]:
        """步长采样"""
        return [x[i] for i in range(0, len(x), stride)]


# 4. Performer线性注意力
class PerformerAttention:
    """Performer线性注意力 - 使用随机特征近似softmax核函数，O(N)复杂度"""
    
    def __init__(self, config: AttentionConfig, num_features: int = 256):
        self._config = config
        self._num_features = num_features
        self._projection_matrix: List[List[float]] = []
        self._init_projection()
    
    def _init_projection(self) -> None:
        """初始化随机投影矩阵"""
        scale = 1.0 / math.sqrt(self._config.hidden_size)
        self._projection_matrix = [
            [random.gauss(0, scale) for _ in range(self._config.hidden_size)]
            for _ in range(self._num_features)
        ]
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]]) -> List[List[float]]:
        """Performer线性注意力前向传播"""
        seq_len = len(query)
        head_dim = len(query[0])
        q_prime = [self._kernel_approx(q) for q in query]
        k_prime = [self._kernel_approx(k) for k in key]
        feature_dim = len(q_prime[0])
        ktv = [[0.0 for _ in range(head_dim)] for _ in range(feature_dim)]
        for i in range(seq_len):
            for f in range(feature_dim):
                for d in range(head_dim):
                    ktv[f][d] += k_prime[i][f] * value[i][d]
        sum_k = [0.0 for _ in range(feature_dim)]
        for i in range(seq_len):
            for f in range(feature_dim):
                sum_k[f] += k_prime[i][f]
        output = []
        for i in range(seq_len):
            numerator = [0.0 for _ in range(head_dim)]
            for f in range(feature_dim):
                for d in range(head_dim):
                    numerator[d] += q_prime[i][f] * ktv[f][d]
            denominator = sum(q_prime[i][f] * sum_k[f] for f in range(feature_dim))
            out = [n / denominator for n in numerator] if denominator > 1e-8 else numerator
            output.append(out)
        return output
    
    def _random_features(self, x: List[float]) -> List[float]:
        """计算随机特征投影"""
        return [sum(x[d] * proj[d] for d in range(len(x))) for proj in self._projection_matrix]
    
    def _kernel_approx(self, x: List[float]) -> List[float]:
        """核函数近似 (softmax核)"""
        norm_sq = sum(xi * xi for xi in x)
        projections = self._random_features(x)
        features = []
        for proj in projections:
            features.extend([math.sin(proj), math.cos(proj)])
        scale = math.exp(norm_sq / 2) / math.sqrt(len(features))
        return [f * scale for f in features]


# 5. 校准稀疏注意力
class CalibSparseAttention:
    """校准稀疏注意力 - 使用离线预计算的稀疏mask，选择top-k重要位置"""
    
    def __init__(self, config: AttentionConfig, top_k: int = 256):
        self._config = config
        self._top_k = top_k
        self._calib_mask: List[List[int]] = []
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]]) -> List[List[float]]:
        """校准稀疏注意力前向传播"""
        seq_len = len(query)
        head_dim = len(query[0])
        output = []
        for i in range(seq_len):
            q = query[i]
            visible_indices = self._calib_mask[i] if self._calib_mask and i < len(self._calib_mask) else list(range(seq_len))
            scores = [sum(q[d] * key[j][d] for d in range(head_dim)) / math.sqrt(head_dim) for j in visible_indices]
            attn_weights = softmax(scores)
            out = [0.0 for _ in range(head_dim)]
            for idx, j in enumerate(visible_indices):
                for d in range(head_dim):
                    out[d] += attn_weights[idx] * value[j][d]
            output.append(out)
        return output
    
    def precompute_mask(self, attention_weights: List[List[float]]) -> None:
        """预计算稀疏mask"""
        self._calib_mask = [self._select_topk(attention_weights[i], self._top_k) for i in range(len(attention_weights))]
    
    def _select_topk(self, attn: List[float], k: int) -> List[int]:
        """选择top-k位置"""
        indexed = [(attn[j], j) for j in range(len(attn))]
        indexed.sort(reverse=True)
        return [idx for _, idx in indexed[:k]]
    
    def clear_mask(self) -> None:
        """清空预计算的mask"""
        self._calib_mask = []


# 6. 动态Token路由
class DynamicTokenRouter:
    """动态Token路由 - MoE风格路由策略，为每个query动态选择相关KV子集"""
    
    def __init__(self, config: AttentionConfig, num_routes: int = 8):
        self._config = config
        self._num_routes = num_routes
        self._route_centers: List[List[float]] = []
        self._init_routes()
    
    def _init_routes(self) -> None:
        """初始化路由中心点"""
        scale = 1.0 / math.sqrt(self._config.hidden_size)
        self._route_centers = [
            [random.gauss(0, scale) for _ in range(self._config.hidden_size)]
            for _ in range(self._num_routes)
        ]
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]]) -> List[List[float]]:
        """动态Token路由前向传播"""
        seq_len = len(query)
        head_dim = len(query[0])
        key_routes = [self._route_tokens(k) for k in key]
        output = []
        for i in range(seq_len):
            q = query[i]
            q_routes = self._route_tokens(q)
            visible_indices = [j for j in range(seq_len) if set(key_routes[j]) & set(q_routes)]
            if not visible_indices:
                visible_indices = list(range(seq_len))
            scores = [sum(q[d] * key[j][d] for d in range(head_dim)) / math.sqrt(head_dim) for j in visible_indices]
            attn_weights = softmax(scores)
            out = [0.0 for _ in range(head_dim)]
            for idx, j in enumerate(visible_indices):
                for d in range(head_dim):
                    out[d] += attn_weights[idx] * value[j][d]
            output.append(out)
        return output
    
    def _route_tokens(self, x: List[float]) -> List[int]:
        """为token选择路由 (top-2)"""
        similarities = [(sum(x[d] * center[d] for d in range(len(x))), i) for i, center in enumerate(self._route_centers)]
        similarities.sort(reverse=True)
        return [idx for _, idx in similarities[:2]]
    
    def _gather_routed(self, key: List[List[float]], value: List[List[float]], 
                       routes: List[int]) -> Tuple[List[List[float]], List[List[float]]]:
        """收集指定路由的KV"""
        gathered_k, gathered_v = [], []
        for i, k in enumerate(key):
            if set(self._route_tokens(k)) & set(routes):
                gathered_k.append(k)
                gathered_v.append(value[i])
        return gathered_k, gathered_v


# 7. 相对位置偏置
class RelativePositionBias:
    """相对位置偏置 - 可学习的位置偏置表，支持最大距离限制和训练时抖动"""
    
    def __init__(self, config: AttentionConfig, max_distance: int = 128):
        self._config = config
        self._max_distance = max_distance
        self._bias_table: List[float] = [random.gauss(0, 0.02) for _ in range(2 * max_distance + 1)]
        self._training = True
    
    def forward(self, seq_len: int) -> List[List[float]]:
        """生成相对位置偏置矩阵"""
        bias = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos = max(-self._max_distance, min(self._max_distance, i - j))
                bias[i][j] = self._bias_table[rel_pos + self._max_distance]
        if self._training:
            bias = self._add_jitter(bias, 0.1)
        return bias
    
    def _add_jitter(self, bias: List[List[float]], amount: float = 0.1) -> List[List[float]]:
        """添加随机抖动"""
        return [[b + random.gauss(0, amount) for b in row] for row in bias]
    
    def set_training(self, training: bool) -> None:
        """设置训练模式"""
        self._training = training
    
    def get_bias(self, relative_pos: int) -> float:
        """获取指定相对位置的偏置"""
        rel_pos = max(-self._max_distance, min(self._max_distance, relative_pos))
        return self._bias_table[rel_pos + self._max_distance]


# 8. Flash Attention模拟
class FlashAttentionSim:
    """Flash Attention模拟 - 分块计算和在线softmax算法"""
    
    def __init__(self, config: AttentionConfig, tile_size: int = 64):
        self._config = config
        self._tile_size = tile_size
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]]) -> List[List[float]]:
        """Flash Attention前向传播"""
        return self._tiled_forward(query, key, value, self._tile_size)
    
    def _tiled_forward(self, q: List[List[float]], k: List[List[float]], 
                       v: List[List[float]], tile_size: int) -> List[List[float]]:
        """分块计算"""
        seq_len = len(q)
        head_dim = len(q[0])
        output = [[0.0 for _ in range(head_dim)] for _ in range(seq_len)]
        logsumexp = [float('-inf') for _ in range(seq_len)]
        num_kv_tiles = (seq_len + tile_size - 1) // tile_size
        
        for kv_tile_idx in range(num_kv_tiles):
            kv_start = kv_tile_idx * tile_size
            kv_end = min((kv_tile_idx + 1) * tile_size, seq_len)
            
            # 当前KV块
            k_tile = k[kv_start:kv_end]
            v_tile = v[kv_start:kv_end]
            
            # 分块遍历Q
            num_q_tiles = (seq_len + tile_size - 1) // tile_size
            
            for q_tile_idx in range(num_q_tiles):
                q_start = q_tile_idx * tile_size
                q_end = min((q_tile_idx + 1) * tile_size, seq_len)
                
                # 当前Q块
                q_tile = q[q_start:q_end]
                
                # 计算块内注意力
                block_output, block_logsumexp = self._compute_block(
                    q_tile, k_tile, v_tile, head_dim
                )
                
                # 在线合并结果
                for i, global_i in enumerate(range(q_start, q_end)):
                    # 在线softmax合并
                    old_logsumexp = logsumexp[global_i]
                    new_logsumexp = max(old_logsumexp, block_logsumexp[i])
                    
                    if old_logsumexp == float('-inf'):
                        # 第一次计算
                        output[global_i] = block_output[i]
                        logsumexp[global_i] = block_logsumexp[i]
                    else:
                        # 合并结果
                        old_scale = math.exp(old_logsumexp - new_logsumexp)
                        new_scale = math.exp(block_logsumexp[i] - new_logsumexp)
                        
                        for d in range(head_dim):
                            output[global_i][d] = (
                                output[global_i][d] * old_scale + 
                                block_output[i][d] * new_scale
                            )
                        logsumexp[global_i] = new_logsumexp
        
        return output
    
    def _compute_block(self, q_tile: List[List[float]], k_tile: List[List[float]], 
                       v_tile: List[List[float]], head_dim: int) -> Tuple[List[List[float]], List[float]]:
        """计算单个块的注意力"""
        q_size, kv_size = len(q_tile), len(k_tile)
        scores = [[sum(q_tile[i][d] * k_tile[j][d] for d in range(head_dim)) / math.sqrt(head_dim) 
                   for j in range(kv_size)] for i in range(q_size)]
        return self._online_softmax(scores, v_tile)
    
    def _online_softmax(self, block: List[List[float]], 
                        v_tile: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
        """在线softmax算法"""
        q_size, kv_size = len(block), len(block[0])
        head_dim = len(v_tile[0]) if v_tile else 0
        output, logsumexp = [], []
        for i in range(q_size):
            max_score = max(block[i])
            exp_sum = sum(math.exp(s - max_score) for s in block[i])
            lse = max_score + math.log(exp_sum) if exp_sum > 0 else max_score
            logsumexp.append(lse)
            out = [0.0 for _ in range(head_dim)]
            for j in range(kv_size):
                weight = math.exp(block[i][j] - lse)
                for d in range(head_dim):
                    out[d] += weight * v_tile[j][d]
            output.append(out)
        return output, logsumexp


# 9. 标准多头注意力
class MultiHeadAttention:
    """标准多头注意力 - 完整实现，支持注意力掩码和缩放点积"""
    
    def __init__(self, config: AttentionConfig):
        self._config = config
        self._num_heads = config.num_heads
        self._head_dim = config.head_dim
        self._wq: List[List[float]] = []
        self._wk: List[List[float]] = []
        self._wv: List[List[float]] = []
        self._wo: List[List[float]] = []
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化投影权重"""
        hidden_size = self._config.hidden_size
        scale = math.sqrt(2.0 / (hidden_size + hidden_size))
        self._wq = [[random.gauss(0, scale) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self._wk = [[random.gauss(0, scale) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self._wv = [[random.gauss(0, scale) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self._wo = [[random.gauss(0, scale) for _ in range(hidden_size)] for _ in range(hidden_size)]
    
    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]], mask: Optional[List[List[bool]]] = None) -> List[List[float]]:
        """多头注意力前向传播"""
        seq_len = len(query)
        hidden_size = len(query[0])
        q_proj = matrix_multiply(query, transpose(self._wq))
        k_proj = matrix_multiply(key, transpose(self._wk))
        v_proj = matrix_multiply(value, transpose(self._wv))
        q_heads = self._split_heads(q_proj)
        k_heads = self._split_heads(k_proj)
        v_heads = self._split_heads(v_proj)
        head_outputs = [self._scaled_dot_product(q_heads[h], k_heads[h], v_heads[h], mask) 
                        for h in range(self._num_heads)]
        concat_heads = self._concat_heads(head_outputs)
        return matrix_multiply(concat_heads, transpose(self._wo))
    
    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """将输入分割为多头"""
        seq_len = len(x)
        heads = []
        for h in range(self._num_heads):
            start, end = h * self._head_dim, (h + 1) * self._head_dim
            heads.append([x[i][start:end] for i in range(seq_len)])
        return heads
    
    def _concat_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        """合并多头"""
        seq_len = len(heads[0])
        return [list(sum((heads[h][i] for h in range(self._num_heads)), [])) for i in range(seq_len)]
    
    def _scaled_dot_product(self, q: List[List[float]], k: List[List[float]], 
                            v: List[List[float]], mask: Optional[List[List[bool]]] = None) -> List[List[float]]:
        """缩放点积注意力"""
        seq_len, head_dim = len(q), len(q[0])
        scores = [[sum(q[i][d] * k[j][d] for d in range(head_dim)) / math.sqrt(head_dim) 
                   for j in range(seq_len)] for i in range(seq_len)]
        if mask is not None:
            scores = apply_mask(scores, mask)
        attn_weights = [softmax(row) for row in scores]
        output = []
        for i in range(seq_len):
            out = [0.0 for _ in range(head_dim)]
            for j in range(seq_len):
                for d in range(head_dim):
                    out[d] += attn_weights[i][j] * v[j][d]
            output.append(out)
        return output


def create_attention_module(attention_type: str, config: AttentionConfig, **kwargs) -> Any:
    """创建注意力模块的工厂函数"""
    modules = {
        'sliding_window': SlidingWindowAttention,
        'block_sparse': BlockSparseAttention,
        'strided': StridedAttention,
        'performer': PerformerAttention,
        'calib_sparse': CalibSparseAttention,
        'dynamic_router': DynamicTokenRouter,
        'flash': FlashAttentionSim,
        'multihead': MultiHeadAttention,
    }
    if attention_type not in modules:
        raise ValueError(f"Unknown attention type: {attention_type}")
    return modules[attention_type](config, **kwargs)


# 工厂类别名
class SparseAttentionFactory:
    """稀疏注意力工厂类"""
    
    @staticmethod
    def create(attention_type: str, config: AttentionConfig, **kwargs) -> Any:
        """创建注意力模块"""
        return create_attention_module(attention_type, config, **kwargs)
    
    @staticmethod
    def get_available_types() -> List[str]:
        """获取可用的注意力类型"""
        return ['sliding_window', 'block_sparse', 'strided', 'performer', 
                'calib_sparse', 'dynamic_router', 'flash', 'multihead']


# 模块导出
__all__ = [
    'AttentionConfig',
    'SlidingWindowAttention',
    'BlockSparseAttention',
    'StridedAttention',
    'PerformerAttention',
    'CalibSparseAttention',
    'DynamicTokenRouter',
    'FlashAttentionSim',
    'MultiHeadAttention',
    'SparseAttentionFactory',
    'create_attention_module',
    'softmax',
    'matrix_multiply',
    'transpose',
    'apply_mask',
]


if __name__ == "__main__":
    config = AttentionConfig(hidden_size=64, num_heads=4, head_dim=16)
    q = [[random.gauss(0, 1) for _ in range(64)] for _ in range(8)]
    k = [[random.gauss(0, 1) for _ in range(64)] for _ in range(8)]
    v = [[random.gauss(0, 1) for _ in range(64)] for _ in range(8)]
    mha = MultiHeadAttention(config)
    out = mha.forward(q, k, v)
    print(f"Output shape: {len(out)}x{len(out[0])}")
