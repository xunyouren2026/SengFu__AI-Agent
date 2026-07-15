"""
模态适配器
实现MLP适配器和Q-Former适配器
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


class MLPAdapter:
    """MLP适配器
    
    简单的线性投影适配器，用于对齐不同模态的特征维度
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2,
                 activation: str = 'gelu',
                 dropout: float = 0.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        # 构建层
        scale = 0.02
        self.layers = []
        self.biases = []
        
        dims = [input_dim] + [self.hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            weight = [[random.gauss(0, scale) for _ in range(dims[i])] 
                      for _ in range(dims[i + 1])]
            bias = [0.0] * dims[i + 1]
            self.layers.append(weight)
            self.biases.append(bias)
        
        # 层归一化
        self.layer_norms = [LayerNorm(d) for d in dims[:-1]]
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def _relu(self, x: float) -> float:
        return max(0.0, x)
    
    def _activate(self, x: float) -> float:
        if self.activation == 'gelu':
            return self._gelu(x)
        elif self.activation == 'relu':
            return self._relu(x)
        return x
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, input_dim]
        
        Returns:
            输出特征 [batch, output_dim]
        """
        for i, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            # 线性变换
            x = [[sum(xi[k] * layer[j][k] for k in range(len(xi))) + bias[j]
                  for j in range(len(layer))] for xi in x]
            
            # 激活函数 (最后一层除外)
            if i < len(self.layers) - 1:
                x = [[self._activate(xij) for xij in xi] for xi in x]
                # 层归一化
                x = [[self.layer_norms[i](xi)[j] for j in range(len(xi))] for xi in x]
        
        return x


class ResidualAdapter:
    """残差适配器
    
    带残差连接的适配器，保持原始特征信息
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 activation: str = 'gelu', dropout: float = 0.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        scale = 0.02
        
        # 下投影
        self.down_proj = [[random.gauss(0, scale) for _ in range(input_dim)] 
                          for _ in range(hidden_dim)]
        self.down_bias = [0.0] * hidden_dim
        
        # 上投影
        self.up_proj = [[random.gauss(0, scale) for _ in range(hidden_dim)] 
                        for _ in range(input_dim)]
        self.up_bias = [0.0] * input_dim
        
        self.norm = LayerNorm(input_dim)
        self.activation = activation
        self.dropout = dropout
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """前向传播"""
        # 归一化
        x_norm = [[self.norm(xi)[j] for j in range(len(xi))] for xi in x]
        
        # 下投影
        hidden = [[self._gelu(sum(x_norm[i][k] * self.down_proj[j][k] for k in range(len(x_norm[i]))) + self.down_bias[j])
                   for j in range(len(self.down_proj))] for i in range(len(x_norm))]
        
        # 上投影
        residual = [[sum(hidden[i][k] * self.up_proj[j][k] for k in range(len(hidden[i]))) + self.up_bias[j]
                     for j in range(len(self.up_proj))] for i in range(len(hidden))]
        
        # 残差连接
        output = [[x[i][j] + residual[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        return output


class QFormerAdapter:
    """Q-Former适配器
    
    使用可学习查询向量提取特征
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_queries: int = 32,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 hidden_dim: Optional[int] = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or input_dim
        
        scale = 0.02
        
        # 可学习查询向量
        self.queries = [[random.gauss(0, scale) for _ in range(self.hidden_dim)] 
                        for _ in range(num_queries)]
        
        # Q, K, V投影
        self.q_proj = [[random.gauss(0, scale) for _ in range(self.hidden_dim)] 
                       for _ in range(self.hidden_dim)]
        self.k_proj = [[random.gauss(0, scale) for _ in range(input_dim)] 
                       for _ in range(self.hidden_dim)]
        self.v_proj = [[random.gauss(0, scale) for _ in range(input_dim)] 
                       for _ in range(self.hidden_dim)]
        self.out_proj = [[random.gauss(0, scale) for _ in range(self.hidden_dim)] 
                         for _ in range(self.hidden_dim)]
        
        # FFN
        self.ffn_fc1 = [[random.gauss(0, scale) for _ in range(self.hidden_dim)] 
                        for _ in range(self.hidden_dim * 4)]
        self.ffn_fc2 = [[random.gauss(0, scale) for _ in range(self.hidden_dim * 4)] 
                        for _ in range(self.hidden_dim)]
        
        # 输出投影
        self.output_proj = [[random.gauss(0, scale) for _ in range(self.hidden_dim * num_queries)] 
                            for _ in range(output_dim)]
        
        self.norm1 = LayerNorm(self.hidden_dim)
        self.norm2 = LayerNorm(self.hidden_dim)
        
        self.head_dim = self.hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """
        前向传播
        
        Args:
            x: 输入特征 [seq_len, input_dim]
        
        Returns:
            输出特征 [batch, output_dim]
        """
        # 初始化查询
        queries = [q.copy() for q in self.queries]
        
        # 多层处理
        for _ in range(self.num_layers):
            # Q投影
            q = [[sum(queries[i][k] * self.q_proj[j][k] for k in range(len(queries[i])))
                  for j in range(len(self.q_proj))] for i in range(len(queries))]
            
            # K, V投影
            k = [[sum(x[i][k] * self.k_proj[j][k] for k in range(len(x[i])))
                  for j in range(len(self.k_proj))] for i in range(len(x))]
            v = [[sum(x[i][k] * self.v_proj[j][k] for k in range(len(x[i])))
                  for j in range(len(self.v_proj))] for i in range(len(x))]
            
            # 注意力分数
            attn = [[sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale 
                     for j in range(len(k))] for i in range(len(q))]
            
            # Softmax
            attn_weights = []
            for row in attn:
                max_val = max(row)
                exp_vals = [math.exp(v - max_val) for v in row]
                sum_exp = sum(exp_vals)
                attn_weights.append([e / sum_exp for e in exp_vals])
            
            # 加权求和
            attn_out = [[sum(attn_weights[i][j] * v[j][d] for j in range(len(v)))
                         for d in range(len(v[0]))] for i in range(len(attn_weights))]
            
            # 输出投影
            attn_out = [[sum(attn_out[i][k] * self.out_proj[j][k] for k in range(len(attn_out[i])))
                         for j in range(len(self.out_proj))] for i in range(len(attn_out))]
            
            # 残差 + 归一化
            queries = [[queries[i][j] + attn_out[i][j] for j in range(len(queries[i]))] 
                       for i in range(len(queries))]
            queries = [[self.norm1(queries[i])[j] for j in range(len(queries[i]))] 
                       for i in range(len(queries))]
            
            # FFN
            hidden = [[self._gelu(sum(queries[i][k] * self.ffn_fc1[j][k] for k in range(len(queries[i]))))
                       for j in range(len(self.ffn_fc1))] for i in range(len(queries))]
            ffn_out = [[sum(hidden[i][k] * self.ffn_fc2[j][k] for k in range(len(hidden[i])))
                        for j in range(len(self.ffn_fc2))] for i in range(len(hidden))]
            
            queries = [[queries[i][j] + ffn_out[i][j] for j in range(len(queries[i]))] 
                       for i in range(len(queries))]
            queries = [[self.norm2(queries[i])[j] for j in range(len(queries[i]))] 
                       for i in range(len(queries))]
        
        # 展平查询
        flat_queries = []
        for q in queries:
            flat_queries.extend(q)
        
        # 输出投影
        output = [sum(flat_queries[k] * self.output_proj[j][k] for k in range(len(flat_queries)))
                  for j in range(self.output_dim)]
        
        return [output]


class ModalityAdapter:
    """模态适配器
    
    为不同模态提供统一的适配接口
    """
    
    def __init__(self, adapters: Dict[str, Any]):
        """
        Args:
            adapters: 模态名称到适配器的映射
        """
        self.adapters = adapters
    
    def __call__(self, features: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
        """
        适配各模态特征
        
        Args:
            features: 各模态特征字典
        
        Returns:
            适配后的特征字典
        """
        adapted = {}
        for modality, feat in features.items():
            if modality in self.adapters:
                adapted[modality] = self.adapters[modality](feat)
            else:
                adapted[modality] = feat
        return adapted


class SequentialAdapter:
    """顺序适配器
    
    串联多个适配器
    """
    
    def __init__(self, adapters: List[Any]):
        self.adapters = adapters
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        for adapter in self.adapters:
            x = adapter(x)
        return x


class ParallelAdapter:
    """并行适配器
    
    并联多个适配器并拼接输出
    """
    
    def __init__(self, adapters: List[Any]):
        self.adapters = adapters
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        outputs = [adapter(x) for adapter in self.adapters]
        
        # 拼接
        result = []
        for i in range(len(outputs[0])):
            combined = []
            for output in outputs:
                combined.extend(output[i])
            result.append(combined)
        
        return result


def create_mlp_adapter(input_dim: int, output_dim: int, 
                       hidden_dim: Optional[int] = None) -> MLPAdapter:
    """创建MLP适配器"""
    return MLPAdapter(input_dim, output_dim, hidden_dim)


def create_residual_adapter(input_dim: int, hidden_dim: int) -> ResidualAdapter:
    """创建残差适配器"""
    return ResidualAdapter(input_dim, hidden_dim)


def create_qformer_adapter(input_dim: int, output_dim: int, 
                           num_queries: int = 32) -> QFormerAdapter:
    """创建Q-Former适配器"""
    return QFormerAdapter(input_dim, output_dim, num_queries)
