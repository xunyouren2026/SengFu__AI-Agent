"""
Q-Former查询变换器
实现可学习的查询向量用于特征提取
"""
from typing import Optional, List, Dict, Any, Tuple
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
    """交叉注意力"""
    
    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 8):
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        scale_init = 0.02
        self.q_proj = [[random.gauss(0, scale_init) for _ in range(query_dim)] for _ in range(query_dim)]
        self.k_proj = [[random.gauss(0, scale_init) for _ in range(kv_dim)] for _ in range(query_dim)]
        self.v_proj = [[random.gauss(0, scale_init) for _ in range(kv_dim)] for _ in range(query_dim)]
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(query_dim)] for _ in range(query_dim)]
    
    def __call__(self, queries: List[List[float]], 
                 inputs: List[List[float]],
                 attention_mask: Optional[List[List[float]]] = None) -> Tuple[List[List[float]], List[List[float]]]:
        # 投影
        q = [[sum(queries[i][k] * self.q_proj[j][k] for k in range(len(queries[i])))
              for j in range(len(self.q_proj))] for i in range(len(queries))]
        k = [[sum(inputs[i][k] * self.k_proj[j][k] for k in range(len(inputs[i])))
              for j in range(len(self.k_proj))] for i in range(len(inputs))]
        v = [[sum(inputs[i][k] * self.v_proj[j][k] for k in range(len(inputs[i])))
              for j in range(len(self.v_proj))] for i in range(len(inputs))]
        
        # 注意力分数
        attn = [[sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale 
                 for j in range(len(k))] for i in range(len(q))]
        
        # 应用掩码
        if attention_mask is not None:
            for i in range(len(attn)):
                for j in range(len(attn[i])):
                    attn[i][j] += attention_mask[i][j]
        
        # Softmax
        attn_weights = []
        for row in attn:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            attn_weights.append([e / sum_exp for e in exp_vals])
        
        # 加权求和
        output = [[sum(attn_weights[i][j] * v[j][d] for j in range(len(v)))
                   for d in range(len(v[0]))] for i in range(len(attn_weights))]
        
        # 输出投影
        output = [[sum(output[i][k] * self.out_proj[j][k] for k in range(len(output[i])))
                   for j in range(len(self.out_proj))] for i in range(len(output))]
        
        return output, attn_weights


class SelfAttention:
    """自注意力"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        scale_init = 0.02
        self.qkv_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(3 * dim)]
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        n = len(x)
        c = len(x[0]) if x else 0
        
        qkv = [[sum(x[i][k] * self.qkv_proj[j][k] for k in range(c)) 
                for j in range(3 * c)] for i in range(n)]
        
        q = [[qkv[i][j] for j in range(c)] for i in range(n)]
        k = [[qkv[i][j + c] for j in range(c)] for i in range(n)]
        v = [[qkv[i][j + 2 * c] for j in range(c)] for i in range(n)]
        
        attn = [[sum(q[i][d] * k[j][d] for d in range(c)) * self.scale 
                 for j in range(n)] for i in range(n)]
        
        attn_weights = []
        for row in attn:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            attn_weights.append([e / sum_exp for e in exp_vals])
        
        output = [[sum(attn_weights[i][j] * v[j][d] for j in range(n)) 
                   for d in range(c)] for i in range(n)]
        
        output = [[sum(output[i][k] * self.out_proj[j][k] for k in range(c)) 
                   for j in range(c)] for i in range(n)]
        
        return output


class MLP:
    """前馈网络"""
    
    def __init__(self, dim: int, hidden_dim: int):
        scale = 0.02
        self.fc1 = [[random.gauss(0, scale) for _ in range(dim)] for _ in range(hidden_dim)]
        self.fc2 = [[random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(dim)]
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        hidden = [[self._gelu(sum(xi[k] * self.fc1[j][k] for k in range(len(xi)))) 
                   for j in range(len(self.fc1))] for xi in x]
        output = [[sum(h[k] * self.fc2[j][k] for k in range(len(h))) 
                   for j in range(len(self.fc2))] for h in hidden]
        return output


class QFormerLayer:
    """Q-Former层"""
    
    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        
        # 自注意力
        self.self_attn = SelfAttention(query_dim, num_heads)
        self.self_attn_norm = LayerNorm(query_dim)
        
        # 交叉注意力
        self.cross_attn = CrossAttention(query_dim, kv_dim, num_heads)
        self.cross_attn_norm_q = LayerNorm(query_dim)
        self.cross_attn_norm_kv = LayerNorm(kv_dim)
        
        # FFN
        self.ffn = MLP(query_dim, int(query_dim * mlp_ratio))
        self.ffn_norm = LayerNorm(query_dim)
    
    def __call__(self, queries: List[List[float]], 
                 inputs: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        # 自注意力
        queries_norm = [[self.self_attn_norm(q)[j] for j in range(len(q))] for q in queries]
        self_attn_out = self.self_attn(queries_norm)
        queries = [[queries[i][j] + self_attn_out[i][j] for j in range(len(queries[i]))] 
                   for i in range(len(queries))]
        
        # 交叉注意力
        queries_norm = [[self.cross_attn_norm_q(q)[j] for j in range(len(q))] for q in queries]
        inputs_norm = [[self.cross_attn_norm_kv(i)[j] for j in range(len(i))] for i in inputs]
        cross_attn_out, attn_weights = self.cross_attn(queries_norm, inputs_norm)
        queries = [[queries[i][j] + cross_attn_out[i][j] for j in range(len(queries[i]))] 
                   for i in range(len(queries))]
        
        # FFN
        queries_norm = [[self.ffn_norm(q)[j] for j in range(len(q))] for q in queries]
        ffn_out = self.ffn(queries_norm)
        queries = [[queries[i][j] + ffn_out[i][j] for j in range(len(queries[i]))] 
                   for i in range(len(queries))]
        
        return queries, attn_weights


class QFormer:
    """Q-Former查询变换器
    
    使用可学习查询向量从输入特征中提取信息
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.query_dim = config.get('query_dim', 768)
        self.kv_dim = config.get('kv_dim', 1024)
        self.num_queries = config.get('num_queries', 32)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.output_dim = config.get('output_dim', 768)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        
        scale = 0.02
        
        # 可学习查询向量
        self.queries = [[random.gauss(0, scale) for _ in range(self.query_dim)] 
                        for _ in range(self.num_queries)]
        
        # Q-Former层
        self.layers = [QFormerLayer(self.query_dim, self.kv_dim, self.num_heads, self.mlp_ratio) 
                       for _ in range(self.num_layers)]
        
        # 输出层
        self.output_norm = LayerNorm(self.query_dim)
        self.output_proj = [[random.gauss(0, scale) for _ in range(self.query_dim * self.num_queries)] 
                            for _ in range(self.output_dim)]
    
    def __call__(self, inputs: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """
        前向传播
        
        Args:
            inputs: 输入特征 [seq_len, kv_dim]
        
        Returns:
            output: 输出特征 [output_dim]
            queries: 最终的查询特征 [num_queries, query_dim]
        """
        # 初始化查询
        queries = [q.copy() for q in self.queries]
        
        # 通过各层
        for layer in self.layers:
            queries, _ = layer(queries, inputs)
        
        # 归一化
        queries = [[self.output_norm(q)[j] for j in range(len(q))] for q in queries]
        
        # 展平并投影
        flat_queries = []
        for q in queries:
            flat_queries.extend(q)
        
        output = [sum(flat_queries[k] * self.output_proj[j][k] for k in range(len(flat_queries)))
                  for j in range(self.output_dim)]
        
        return output, queries
    
    def get_attention_weights(self, inputs: List[List[float]]) -> List[List[List[float]]]:
        """获取各层的注意力权重"""
        queries = [q.copy() for q in self.queries]
        
        all_attn_weights = []
        for layer in self.layers:
            queries, attn_weights = layer(queries, inputs)
            all_attn_weights.append(attn_weights)
        
        return all_attn_weights


class QFormerWithText(QFormer):
    """带文本输入的Q-Former"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}
        
        self.text_dim = config.get('text_dim', 512)
        
        scale = 0.02
        
        # 文本交叉注意力
        self.text_cross_attn = CrossAttention(self.query_dim, self.text_dim, self.num_heads)
        self.text_cross_norm_q = LayerNorm(self.query_dim)
        self.text_cross_norm_t = LayerNorm(self.text_dim)
    
    def __call__(self, visual_inputs: List[List[float]], 
                 text_inputs: Optional[List[List[float]]] = None) -> Tuple[List[float], List[List[float]]]:
        """前向传播"""
        # 初始化查询
        queries = [q.copy() for q in self.queries]
        
        # 通过各层
        for layer in self.layers:
            queries, _ = layer(queries, visual_inputs)
        
        # 文本交叉注意力
        if text_inputs is not None:
            queries_norm = [[self.text_cross_norm_q(q)[j] for j in range(len(q))] for q in queries]
            text_norm = [[self.text_cross_norm_t(t)[j] for j in range(len(t))] for t in text_inputs]
            text_attn_out, _ = self.text_cross_attn(queries_norm, text_norm)
            queries = [[queries[i][j] + text_attn_out[i][j] for j in range(len(queries[i]))] 
                       for i in range(len(queries))]
        
        # 归一化
        queries = [[self.output_norm(q)[j] for j in range(len(q))] for q in queries]
        
        # 展平并投影
        flat_queries = []
        for q in queries:
            flat_queries.extend(q)
        
        output = [sum(flat_queries[k] * self.output_proj[j][k] for k in range(len(flat_queries)))
                  for j in range(self.output_dim)]
        
        return output, queries


def create_qformer(query_dim: int, kv_dim: int, 
                   num_queries: int = 32) -> QFormer:
    """创建Q-Former"""
    return QFormer({
        'query_dim': query_dim,
        'kv_dim': kv_dim,
        'num_queries': num_queries
    })
