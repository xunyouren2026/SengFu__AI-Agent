"""
跨模态交叉注意力
实现不同模态之间的交互注意力机制
"""
from typing import Optional, Tuple, List, Dict, Any
import math
import random


class LayerNorm:
    """层归一化"""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim
        self.bias = [0.0] * dim
    
    def __call__(self, x: List[float]) -> List[float]:
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        x_norm = [(xi - mean) / math.sqrt(var + self.eps) for xi in x]
        return [w * xn + b for w, xn, b in zip(self.weight, self.bias, x_norm)]


class CrossAttention:
    """跨模态交叉注意力"""
    
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8,
                 dropout: float = 0.0, bias: bool = True):
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        scale_init = 0.02
        
        # Q投影 (来自query模态)
        self.q_proj = [[random.gauss(0, scale_init) for _ in range(query_dim)] 
                       for _ in range(query_dim)]
        
        # K, V投影 (来自key模态)
        self.k_proj = [[random.gauss(0, scale_init) for _ in range(key_dim)] 
                       for _ in range(query_dim)]
        self.v_proj = [[random.gauss(0, scale_init) for _ in range(key_dim)] 
                       for _ in range(query_dim)]
        
        # 输出投影
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(query_dim)] 
                         for _ in range(query_dim)]
        
        if bias:
            self.q_bias = [0.0] * query_dim
            self.k_bias = [0.0] * query_dim
            self.v_bias = [0.0] * query_dim
            self.out_bias = [0.0] * query_dim
        else:
            self.q_bias = self.k_bias = self.v_bias = self.out_bias = None
        
        self.dropout = dropout
    
    def _linear(self, x: List[float], weight: List[List[float]], 
                bias: Optional[List[float]] = None) -> List[float]:
        out = [sum(x[i] * weight[j][i] for i in range(len(x))) for j in range(len(weight))]
        if bias:
            out = [out[i] + bias[i] for i in range(len(out))]
        return out
    
    def __call__(self, query: List[List[float]], 
                 key: List[List[float]], 
                 value: List[List[float]],
                 attn_mask: Optional[List[List[float]]] = None) -> Tuple[List[List[float]], List[List[float]]]:
        """
        跨模态交叉注意力前向传播
        
        Args:
            query: 查询模态特征 [seq_q, query_dim]
            key: 键模态特征 [seq_k, key_dim]
            value: 值模态特征 [seq_k, key_dim]
            attn_mask: 注意力掩码
        
        Returns:
            output: 输出特征 [seq_q, query_dim]
            attn_weights: 注意力权重 [seq_q, seq_k]
        """
        # 投影
        q = [self._linear(qi, self.q_proj, self.q_bias) for qi in query]
        k = [self._linear(ki, self.k_proj, self.k_bias) for ki in key]
        v = [self._linear(vi, self.v_proj, self.v_bias) for vi in value]
        
        # 计算注意力分数
        attn_scores = []
        for i in range(len(q)):
            row = []
            for j in range(len(k)):
                score = sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale
                row.append(score)
            attn_scores.append(row)
        
        # 应用掩码
        if attn_mask is not None:
            for i in range(len(attn_scores)):
                for j in range(len(attn_scores[i])):
                    attn_scores[i][j] += attn_mask[i][j]
        
        # Softmax
        attn_weights = []
        for row in attn_scores:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            attn_weights.append([e / sum_exp for e in exp_vals])
        
        # 加权求和
        output = []
        for i in range(len(attn_weights)):
            out = [0.0] * len(v[0]) if v else []
            for j in range(len(attn_weights[i])):
                for d in range(len(out)):
                    out[d] += attn_weights[i][j] * v[j][d]
            output.append(out)
        
        # 输出投影
        output = [self._linear(o, self.out_proj, self.out_bias) for o in output]
        
        return output, attn_weights


class CrossAttentionBlock:
    """跨模态交叉注意力块"""
    
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8,
                 ff_dim: Optional[int] = None, dropout: float = 0.0):
        ff_dim = ff_dim or query_dim * 4
        
        self.norm_q = LayerNorm(query_dim)
        self.norm_kv = LayerNorm(key_dim)
        self.cross_attn = CrossAttention(query_dim, key_dim, num_heads, dropout)
        
        self.norm_ff = LayerNorm(query_dim)
        
        # 前馈网络
        scale = 0.02
        self.fc1 = [[random.gauss(0, scale) for _ in range(query_dim)] for _ in range(ff_dim)]
        self.fc2 = [[random.gauss(0, scale) for _ in range(ff_dim)] for _ in range(query_dim)]
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def _ffn(self, x: List[List[float]]) -> List[List[float]]:
        hidden = [[self._gelu(sum(xi[k] * self.fc1[j][k] for k in range(len(xi)))) 
                   for j in range(len(self.fc1))] for xi in x]
        output = [[sum(h[k] * self.fc2[j][k] for k in range(len(h))) 
                   for j in range(len(self.fc2))] for h in hidden]
        return output
    
    def __call__(self, query: List[List[float]], 
                 key: List[List[float]],
                 value: Optional[List[List[float]]] = None) -> Tuple[List[List[float]], List[List[float]]]:
        """前向传播"""
        if value is None:
            value = key
        
        # 归一化
        query_norm = [self.norm_q(qi) for qi in query]
        key_norm = [self.norm_kv(ki) for ki in key]
        value_norm = [self.norm_kv(vi) for vi in value]
        
        # 交叉注意力
        attn_out, attn_weights = self.cross_attn(query_norm, key_norm, value_norm)
        query = [[query[i][j] + attn_out[i][j] for j in range(len(query[i]))] 
                 for i in range(len(query))]
        
        # 前馈网络
        query_norm = [self.norm_ff(qi) for qi in query]
        ffn_out = self._ffn(query_norm)
        query = [[query[i][j] + ffn_out[i][j] for j in range(len(query[i]))] 
                 for i in range(len(query))]
        
        return query, attn_weights


class BidirectionalCrossAttention:
    """双向跨模态交叉注意力"""
    
    def __init__(self, dim_a: int, dim_b: int, num_heads: int = 8,
                 ff_dim: Optional[int] = None, dropout: float = 0.0):
        # A -> B 方向
        self.cross_attn_a_to_b = CrossAttentionBlock(dim_b, dim_a, num_heads, ff_dim, dropout)
        
        # B -> A 方向
        self.cross_attn_b_to_a = CrossAttentionBlock(dim_a, dim_b, num_heads, ff_dim, dropout)
    
    def __call__(self, features_a: List[List[float]], 
                 features_b: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], Dict[str, Any]]:
        """
        双向跨模态注意力
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
        
        Returns:
            updated_a: 更新后的模态A特征
            updated_b: 更新后的模态B特征
            attn_info: 注意力信息
        """
        # A -> B
        updated_b, attn_a_to_b = self.cross_attn_a_to_b(features_b, features_a)
        
        # B -> A
        updated_a, attn_b_to_a = self.cross_attn_b_to_a(features_a, features_b)
        
        attn_info = {
            'a_to_b_weights': attn_a_to_b,
            'b_to_a_weights': attn_b_to_a
        }
        
        return updated_a, updated_b, attn_info


class MultiModalCrossAttention:
    """多模态交叉注意力"""
    
    def __init__(self, modalities: Dict[str, int], num_heads: int = 8,
                 ff_dim: Optional[int] = None, dropout: float = 0.0):
        """
        Args:
            modalities: 模态名称到维度的映射，如 {'vision': 768, 'text': 512, 'audio': 256}
        """
        self.modalities = modalities
        self.modality_names = list(modalities.keys())
        
        # 为每对模态创建交叉注意力
        self.cross_attentions = {}
        for i, mod_a in enumerate(self.modality_names):
            for mod_b in self.modality_names:
                if mod_a != mod_b:
                    key = f"{mod_a}_to_{mod_b}"
                    self.cross_attentions[key] = CrossAttentionBlock(
                        modalities[mod_a], modalities[mod_b], num_heads, ff_dim, dropout
                    )
    
    def __call__(self, features: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
        """
        多模态交叉注意力
        
        Args:
            features: 各模态特征字典
        
        Returns:
            更新后的各模态特征
        """
        updated_features = {k: v.copy() for k, v in features.items()}
        
        # 对每对模态进行交叉注意力
        for mod_a in self.modality_names:
            if mod_a not in features:
                continue
            
            for mod_b in self.modality_names:
                if mod_a == mod_b or mod_b not in features:
                    continue
                
                key = f"{mod_a}_to_{mod_b}"
                if key in self.cross_attentions:
                    updated, _ = self.cross_attentions[key](
                        updated_features[mod_a], features[mod_b]
                    )
                    updated_features[mod_a] = updated
        
        return updated_features


class GatedCrossAttention:
    """门控跨模态交叉注意力"""
    
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8,
                 dropout: float = 0.0):
        self.cross_attn = CrossAttentionBlock(query_dim, key_dim, num_heads, 
                                              query_dim * 4, dropout)
        
        # 门控参数
        scale = 0.02
        self.gate = [random.gauss(0, scale) for _ in range(query_dim)]
    
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    
    def __call__(self, query: List[List[float]], 
                 key: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """门控交叉注意力"""
        attn_out, attn_weights = self.cross_attn(query, key)
        
        # 应用门控
        gate_values = [self._sigmoid(g) for g in self.gate]
        gated_out = [[attn_out[i][j] * gate_values[j] for j in range(len(attn_out[i]))] 
                     for i in range(len(attn_out))]
        
        output = [[query[i][j] + gated_out[i][j] for j in range(len(query[i]))] 
                  for i in range(len(query))]
        
        return output, attn_weights


def create_cross_attention(query_dim: int, key_dim: int, 
                           num_heads: int = 8) -> CrossAttention:
    """创建跨模态交叉注意力"""
    return CrossAttention(query_dim, key_dim, num_heads)


def create_bidirectional_cross_attention(dim_a: int, dim_b: int, 
                                         num_heads: int = 8) -> BidirectionalCrossAttention:
    """创建双向跨模态交叉注意力"""
    return BidirectionalCrossAttention(dim_a, dim_b, num_heads)
