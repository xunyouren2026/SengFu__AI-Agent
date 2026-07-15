"""
视觉投影层
实现视觉特征到语言模型空间的投影
"""
from typing import Optional, List, Dict, Any
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


class LinearProjection:
    """线性投影层"""
    
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        scale = 0.02
        self.weight = [[random.gauss(0, scale) for _ in range(input_dim)] 
                       for _ in range(output_dim)]
        self.bias = [0.0] * output_dim if bias else None
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """前向传播"""
        output = []
        for xi in x:
            out = [sum(xi[k] * self.weight[j][k] for k in range(len(xi))) 
                   for j in range(len(self.weight))]
            if self.bias:
                out = [out[i] + self.bias[i] for i in range(len(out))]
            output.append(out)
        return output


class MLPProjection:
    """MLP投影层"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2,
                 activation: str = 'gelu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.num_layers = num_layers
        self.activation = activation
        
        scale = 0.02
        
        # 构建层
        self.layers = []
        self.biases = []
        self.norms = []
        
        dims = [input_dim] + [self.hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            weight = [[random.gauss(0, scale) for _ in range(dims[i])] 
                      for _ in range(dims[i + 1])]
            bias = [0.0] * dims[i + 1]
            self.layers.append(weight)
            self.biases.append(bias)
            self.norms.append(LayerNorm(dims[i]))
    
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
        """前向传播"""
        for i, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            # 归一化
            x = [[self.norms[i](xi)[j] for j in range(len(xi))] for xi in x]
            
            # 线性变换
            x = [[sum(xi[k] * layer[j][k] for k in range(len(xi))) + bias[j]
                  for j in range(len(layer))] for xi in x]
            
            # 激活函数 (最后一层除外)
            if i < len(self.layers) - 1:
                x = [[self._activate(xij) for xij in xi] for xi in x]
        
        return x


class ResidualProjection:
    """残差投影层"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        scale = 0.02
        
        # 输入投影
        self.input_proj = [[random.gauss(0, scale) for _ in range(input_dim)] 
                           for _ in range(hidden_dim)]
        
        # 残差块
        self.residual_fc1 = [[random.gauss(0, scale) for _ in range(hidden_dim)] 
                             for _ in range(hidden_dim)]
        self.residual_fc2 = [[random.gauss(0, scale) for _ in range(hidden_dim)] 
                             for _ in range(hidden_dim)]
        
        # 输出投影
        self.output_proj = [[random.gauss(0, scale) for _ in range(hidden_dim)] 
                            for _ in range(output_dim)]
        
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """前向传播"""
        # 输入投影
        h = [[sum(xi[k] * self.input_proj[j][k] for k in range(len(xi))) 
              for j in range(len(self.input_proj))] for xi in x]
        
        # 残差块
        h_norm = [[self.norm1(hi)[j] for j in range(len(hi))] for hi in h]
        h1 = [[self._gelu(sum(h_norm[i][k] * self.residual_fc1[j][k] for k in range(len(h_norm[i])))) 
               for j in range(len(self.residual_fc1))] for i in range(len(h_norm))]
        
        h1_norm = [[self.norm2(h1i)[j] for j in range(len(h1i))] for h1i in h1]
        h2 = [[sum(h1_norm[i][k] * self.residual_fc2[j][k] for k in range(len(h1_norm[i]))) 
               for j in range(len(self.residual_fc2))] for i in range(len(h1_norm))]
        
        h = [[h[i][j] + h2[i][j] for j in range(len(h[i]))] for i in range(len(h))]
        
        # 输出投影
        output = [[sum(h[i][k] * self.output_proj[j][k] for k in range(len(h[i]))) 
                   for j in range(len(self.output_proj))] for i in range(len(h))]
        
        return output


class VisualProjector:
    """视觉投影器
    
    将视觉特征投影到语言模型的嵌入空间
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.visual_dim = config.get('visual_dim', 768)
        self.output_dim = config.get('output_dim', 4096)
        self.hidden_dim = config.get('hidden_dim', 2048)
        self.num_layers = config.get('num_layers', 2)
        self.projection_type = config.get('projection_type', 'mlp')
        
        # 根据类型创建投影层
        if self.projection_type == 'linear':
            self.proj = LinearProjection(self.visual_dim, self.output_dim)
        elif self.projection_type == 'mlp':
            self.proj = MLPProjection(self.visual_dim, self.output_dim, 
                                      self.hidden_dim, self.num_layers)
        elif self.projection_type == 'residual':
            self.proj = ResidualProjection(self.visual_dim, self.output_dim, self.hidden_dim)
        else:
            self.proj = MLPProjection(self.visual_dim, self.output_dim, self.hidden_dim)
    
    def __call__(self, visual_features: List[List[float]]) -> List[List[float]]:
        """
        投影视觉特征
        
        Args:
            visual_features: 视觉特征 [num_tokens, visual_dim]
        
        Returns:
            投影后的特征 [num_tokens, output_dim]
        """
        return self.proj(visual_features)
    
    def project_single(self, visual_feature: List[float]) -> List[float]:
        """投影单个特征向量"""
        return self.proj([visual_feature])[0]


class MultiLayerVisualProjector:
    """多层视觉投影器
    
    支持从不同视觉层提取特征并融合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.visual_dim = config.get('visual_dim', 768)
        self.output_dim = config.get('output_dim', 4096)
        self.hidden_dim = config.get('hidden_dim', 2048)
        self.num_visual_layers = config.get('num_visual_layers', 4)
        
        scale = 0.02
        
        # 各层的投影
        self.layer_projs = []
        for _ in range(self.num_visual_layers):
            proj = MLPProjection(self.visual_dim, self.hidden_dim, self.hidden_dim)
            self.layer_projs.append(proj)
        
        # 融合层
        self.fusion_weight = [random.gauss(0, scale) for _ in range(self.num_visual_layers)]
        
        # 输出投影
        self.output_proj = MLPProjection(self.hidden_dim, self.output_dim, self.hidden_dim)
    
    def __call__(self, layer_features: List[List[List[float]]]) -> List[List[float]]:
        """
        投影多层视觉特征
        
        Args:
            layer_features: 各层视觉特征 [num_layers, num_tokens, visual_dim]
        
        Returns:
            投影后的特征 [num_tokens, output_dim]
        """
        # 投影各层
        projected = []
        for i, (features, proj) in enumerate(zip(layer_features, self.layer_projs)):
            proj_feat = proj(features)
            w = self.fusion_weight[i]
            weighted = [[f * w for f in fi] for fi in proj_feat]
            projected.append(weighted)
        
        # 融合
        if projected:
            fused = [[0.0] * self.hidden_dim for _ in range(len(projected[0]))]
            for proj_feat in projected:
                for i in range(len(fused)):
                    for j in range(len(fused[i])):
                        fused[i][j] += proj_feat[i][j]
            
            # 归一化
            total_weight = sum(abs(w) for w in self.fusion_weight)
            if total_weight > 0:
                fused = [[f / total_weight for f in fi] for fi in fused]
        else:
            fused = [[0.0] * self.hidden_dim]
        
        # 输出投影
        return self.output_proj(fused)


class AdaptiveVisualProjector:
    """自适应视觉投影器
    
    根据输入动态调整投影方式
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.visual_dim = config.get('visual_dim', 768)
        self.output_dim = config.get('output_dim', 4096)
        self.hidden_dim = config.get('hidden_dim', 2048)
        
        scale = 0.02
        
        # 多种投影方式
        self.linear_proj = LinearProjection(self.visual_dim, self.output_dim)
        self.mlp_proj = MLPProjection(self.visual_dim, self.output_dim, self.hidden_dim)
        
        # 门控网络
        self.gate_proj = [[random.gauss(0, scale) for _ in range(self.visual_dim)] 
                          for _ in range(2)]
    
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
    
    def __call__(self, visual_features: List[List[float]]) -> List[List[float]]:
        """自适应投影"""
        # 计算门控值
        pooled = [sum(f[i] for f in visual_features) / len(visual_features) 
                  for i in range(len(visual_features[0]))] if visual_features else [0.0] * self.visual_dim
        
        gate_scores = [sum(pooled[k] * self.gate_proj[j][k] for k in range(len(pooled))) 
                       for j in range(2)]
        gate_scores = [self._sigmoid(s) for s in gate_scores]
        
        # 归一化
        total = sum(gate_scores)
        if total > 0:
            gate_scores = [s / total for s in gate_scores]
        
        # 加权投影
        linear_out = self.linear_proj(visual_features)
        mlp_out = self.mlp_proj(visual_features)
        
        output = [[linear_out[i][j] * gate_scores[0] + mlp_out[i][j] * gate_scores[1] 
                   for j in range(len(linear_out[i]))] for i in range(len(linear_out))]
        
        return output


def create_visual_projector(visual_dim: int, output_dim: int,
                            projection_type: str = 'mlp') -> VisualProjector:
    """创建视觉投影器"""
    return VisualProjector({
        'visual_dim': visual_dim,
        'output_dim': output_dim,
        'projection_type': projection_type
    })
