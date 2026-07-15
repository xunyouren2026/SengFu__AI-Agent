"""
Perceiver重采样
实现基于Perceiver架构的特征重采样
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
        self.q_proj = [[random.gauss(0, scale_init) for _ in range(query_dim)] 
                       for _ in range(query_dim)]
        self.k_proj = [[random.gauss(0, scale_init) for _ in range(kv_dim)] 
                       for _ in range(query_dim)]
        self.v_proj = [[random.gauss(0, scale_init) for _ in range(kv_dim)] 
                       for _ in range(query_dim)]
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(query_dim)] 
                         for _ in range(query_dim)]
    
    def __call__(self, queries: List[List[float]], 
                 inputs: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        # 投影
        q = [[sum(queries[i][k] * self.q_proj[j][k] for k in range(len(queries[i])))
              for j in range(len(self.q_proj))] for i in range(len(queries))]
        k = [[sum(inputs[i][k] * self.k_proj[j][k] for k in range(len(inputs[i])))
              for j in range(len(self.k_proj))] for i in range(len(inputs))]
        v = [[sum(inputs[i][k] * self.v_proj[j][k] for k in range(len(inputs[i])))
              for j in range(len(self.v_proj))] for i in range(len(inputs))]
        
        # 注意力
        attn = [[sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale 
                 for j in range(len(k))] for i in range(len(q))]
        
        attn_weights = []
        for row in attn:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            attn_weights.append([e / sum_exp for e in exp_vals])
        
        output = [[sum(attn_weights[i][j] * v[j][d] for j in range(len(v)))
                   for d in range(len(v[0]))] for i in range(len(attn_weights))]
        
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
        self.qkv_proj = [[random.gauss(0, scale_init) for _ in range(dim)] 
                         for _ in range(3 * dim)]
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(dim)] 
                         for _ in range(dim)]
    
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


class PerceiverEncoder:
    """Perceiver编码器层"""
    
    def __init__(self, dim: int, kv_dim: int, num_heads: int = 8, 
                 num_self_attn: int = 2, mlp_ratio: float = 4.0):
        self.dim = dim
        self.kv_dim = kv_dim
        
        # 交叉注意力
        self.cross_attn = CrossAttention(dim, kv_dim, num_heads)
        self.cross_norm_q = LayerNorm(dim)
        self.cross_norm_kv = LayerNorm(kv_dim)
        self.cross_mlp = MLP(dim, int(dim * mlp_ratio))
        self.cross_mlp_norm = LayerNorm(dim)
        
        # 自注意力塔
        self.self_attn_layers = []
        for _ in range(num_self_attn):
            self.self_attn_layers.append({
                'self_attn': SelfAttention(dim, num_heads),
                'norm': LayerNorm(dim),
                'mlp': MLP(dim, int(dim * mlp_ratio)),
                'mlp_norm': LayerNorm(dim)
            })
    
    def __call__(self, latents: List[List[float]], 
                 inputs: List[List[float]]) -> List[List[float]]:
        """前向传播"""
        # 交叉注意力
        q_norm = [[self.cross_norm_q(l)[j] for j in range(len(l))] for l in latents]
        kv_norm = [[self.cross_norm_kv(i)[j] for j in range(len(i))] for i in inputs]
        
        cross_out, _ = self.cross_attn(q_norm, kv_norm)
        latents = [[latents[i][j] + cross_out[i][j] for j in range(len(latents[i]))] 
                   for i in range(len(latents))]
        
        # MLP
        latents_norm = [[self.cross_mlp_norm(l)[j] for j in range(len(l))] for l in latents]
        mlp_out = self.cross_mlp(latents_norm)
        latents = [[latents[i][j] + mlp_out[i][j] for j in range(len(latents[i]))] 
                   for i in range(len(latents))]
        
        # 自注意力塔
        for layer in self.self_attn_layers:
            # 自注意力
            latents_norm = [[layer['norm'](l)[j] for j in range(len(l))] for l in latents]
            self_out = layer['self_attn'](latents_norm)
            latents = [[latents[i][j] + self_out[i][j] for j in range(len(latents[i]))] 
                       for i in range(len(latents))]
            
            # MLP
            latents_norm = [[layer['mlp_norm'](l)[j] for j in range(len(l))] for l in latents]
            mlp_out = layer['mlp'](latents_norm)
            latents = [[latents[i][j] + mlp_out[i][j] for j in range(len(latents[i]))] 
                       for i in range(len(latents))]
        
        return latents


class PerceiverDecoder:
    """Perceiver解码器"""
    
    def __init__(self, dim: int, output_dim: int, num_heads: int = 8,
                 num_queries: int = 1, mlp_ratio: float = 4.0):
        self.dim = dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        
        scale = 0.02
        
        # 输出查询
        self.output_queries = [[random.gauss(0, scale) for _ in range(dim)] 
                               for _ in range(num_queries)]
        
        # 交叉注意力
        self.cross_attn = CrossAttention(dim, dim, num_heads)
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        
        # 输出MLP
        self.output_mlp = MLP(dim, int(dim * mlp_ratio))
        self.output_norm = LayerNorm(dim)
        
        # 最终投影
        self.final_proj = [[random.gauss(0, scale) for _ in range(dim)] 
                           for _ in range(output_dim)]
    
    def __call__(self, latents: List[List[float]]) -> List[List[float]]:
        """解码"""
        queries = [q.copy() for q in self.output_queries]
        
        # 交叉注意力
        q_norm = [[self.norm_q(q)[j] for j in range(len(q))] for q in queries]
        kv_norm = [[self.norm_kv(l)[j] for j in range(len(l))] for l in latents]
        
        cross_out, _ = self.cross_attn(q_norm, kv_norm)
        queries = [[queries[i][j] + cross_out[i][j] for j in range(len(queries[i]))] 
                   for i in range(len(queries))]
        
        # MLP
        queries_norm = [[self.output_norm(q)[j] for j in range(len(q))] for q in queries]
        mlp_out = self.output_mlp(queries_norm)
        queries = [[queries[i][j] + mlp_out[i][j] for j in range(len(queries[i]))] 
                   for i in range(len(queries))]
        
        # 最终投影
        outputs = [[sum(q[k] * self.final_proj[j][k] for k in range(len(q)))
                    for j in range(self.output_dim)] for q in queries]
        
        return outputs


class PerceiverResampler:
    """Perceiver重采样器
    
    将可变长度的输入特征重采样为固定长度的输出特征
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.input_dim = config.get('input_dim', 768)
        self.latent_dim = config.get('latent_dim', 512)
        self.num_latents = config.get('num_latents', 64)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.num_self_attn = config.get('num_self_attn', 2)
        self.output_dim = config.get('output_dim', 512)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        
        scale = 0.02
        
        # 可学习潜在向量
        self.latents = [[random.gauss(0, scale) for _ in range(self.latent_dim)] 
                        for _ in range(self.num_latents)]
        
        # 编码器层
        self.encoder_layers = [
            PerceiverEncoder(self.latent_dim, self.input_dim, self.num_heads, 
                            self.num_self_attn, self.mlp_ratio)
            for _ in range(self.num_layers)
        ]
        
        # 解码器
        self.decoder = PerceiverDecoder(self.latent_dim, self.output_dim, 
                                        self.num_heads, 1, self.mlp_ratio)
    
    def __call__(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        重采样
        
        Args:
            inputs: 输入特征 [seq_len, input_dim]
        
        Returns:
            输出特征 [1, output_dim] 或展平为 [output_dim]
        """
        # 初始化潜在向量
        latents = [l.copy() for l in self.latents]
        
        # 编码器层
        for encoder in self.encoder_layers:
            latents = encoder(latents, inputs)
        
        # 解码
        outputs = self.decoder(latents)
        
        return outputs
    
    def get_latents(self, inputs: List[List[float]]) -> List[List[float]]:
        """获取潜在表示"""
        latents = [l.copy() for l in self.latents]
        for encoder in self.encoder_layers:
            latents = encoder(latents, inputs)
        return latents


class MultiModalPerceiver:
    """多模态Perceiver
    
    处理多种模态的输入
    """
    
    def __init__(self, modalities: Dict[str, int], latent_dim: int = 512,
                 num_latents: int = 64, num_heads: int = 8,
                 num_layers: int = 4, output_dim: int = 512):
        """
        Args:
            modalities: 模态名称到输入维度的映射
        """
        self.modalities = modalities
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.output_dim = output_dim
        
        scale = 0.02
        
        # 共享潜在向量
        self.latents = [[random.gauss(0, scale) for _ in range(latent_dim)] 
                        for _ in range(num_latents)]
        
        # 各模态的编码器
        self.modality_encoders = {}
        for mod_name, mod_dim in modalities.items():
            self.modality_encoders[mod_name] = PerceiverEncoder(
                latent_dim, mod_dim, num_heads, 2, 4.0
            )
        
        # 共享自注意力层
        self.shared_layers = [
            PerceiverEncoder(latent_dim, latent_dim, num_heads, 2, 4.0)
            for _ in range(num_layers)
        ]
        
        # 输出投影
        self.output_proj = [[random.gauss(0, scale) for _ in range(latent_dim * num_latents)] 
                            for _ in range(output_dim)]
    
    def __call__(self, inputs: Dict[str, List[List[float]]]) -> List[float]:
        """
        多模态处理
        
        Args:
            inputs: 各模态输入特征
        
        Returns:
            融合后的特征
        """
        # 初始化潜在向量
        latents = [l.copy() for l in self.latents]
        
        # 各模态编码
        for mod_name, mod_input in inputs.items():
            if mod_name in self.modality_encoders:
                latents = self.modality_encoders[mod_name](latents, mod_input)
        
        # 共享处理
        for layer in self.shared_layers:
            latents = layer(latents, latents)
        
        # 展平并投影
        flat = []
        for l in latents:
            flat.extend(l)
        
        output = [sum(flat[k] * self.output_proj[j][k] for k in range(len(flat)))
                  for j in range(self.output_dim)]
        
        return output


def create_perceiver_resampler(input_dim: int, output_dim: int,
                               num_latents: int = 64) -> PerceiverResampler:
    """创建Perceiver重采样器"""
    return PerceiverResampler({
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_latents': num_latents
    })
