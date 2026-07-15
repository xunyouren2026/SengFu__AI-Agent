"""
晚期融合策略
实现多种多模态晚期融合方法
"""
from typing import Optional, List, Dict, Any, Tuple, Callable
import math
import random


class LateFusion:
    """晚期融合基类"""
    
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """晚期融合的默认实现
        
        将所有模态特征拼接后通过随机投影映射到输出维度。
        
        Args:
            features: 模态名称到特征向量的映射
            
        Returns:
            融合后的特征向量
        """
        if not features:
            return [0.0] * self.output_dim
        
        # 收集所有可用特征并拼接
        concatenated: List[float] = []
        for mod, feat in features.items():
            if isinstance(feat, (list, tuple)):
                concatenated.extend(feat)
        
        if not concatenated:
            return [0.0] * self.output_dim
        
        # 如果拼接维度恰好等于输出维度，直接返回
        if len(concatenated) == self.output_dim:
            return concatenated
        
        # 通过随机投影映射到目标维度
        import random as _random
        _random.seed(42)  # 保证可重复性
        scale = 0.02
        proj = [[_random.gauss(0, scale) for _ in range(len(concatenated))]
                for _ in range(self.output_dim)]
        
        output = [
            sum(concatenated[k] * proj[j][k] for k in range(len(concatenated)))
            for j in range(self.output_dim)
        ]
        return output


class ConcatenationFusion(LateFusion):
    """拼接融合
    
    直接拼接各模态特征
    """
    
    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        super().__init__(output_dim)
        self.modality_dims = modality_dims
        total_dim = sum(modality_dims.values())
        
        # 投影层
        scale = 0.02
        self.proj = [[random.gauss(0, scale) for _ in range(total_dim)] 
                     for _ in range(output_dim)]
        self.bias = [0.0] * output_dim
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """拼接并投影"""
        # 按顺序拼接
        concatenated = []
        for mod in self.modality_dims.keys():
            if mod in features:
                concatenated.extend(features[mod])
        
        # 投影
        output = [sum(concatenated[k] * self.proj[j][k] for k in range(len(concatenated))) + self.bias[j]
                  for j in range(self.output_dim)]
        
        return output


class WeightedFusion(LateFusion):
    """加权融合
    
    对各模态特征进行加权求和
    """
    
    def __init__(self, modalities: List[str], feature_dim: int, output_dim: int,
                 learnable: bool = True):
        super().__init__(output_dim)
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.learnable = learnable
        
        # 权重
        if learnable:
            self.weights = {mod: 1.0 / len(modalities) for mod in modalities}
        else:
            self.weights = {mod: 1.0 / len(modalities) for mod in modalities}
        
        # 投影层
        scale = 0.02
        self.proj = [[random.gauss(0, scale) for _ in range(feature_dim)] 
                     for _ in range(output_dim)]
        self.bias = [0.0] * output_dim
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """加权求和"""
        # 加权求和
        weighted_sum = [0.0] * self.feature_dim
        total_weight = 0.0
        
        for mod in self.modalities:
            if mod in features and len(features[mod]) == self.feature_dim:
                w = self.weights.get(mod, 0.0)
                for i in range(len(weighted_sum)):
                    weighted_sum[i] += w * features[mod][i]
                total_weight += w
        
        # 归一化
        if total_weight > 0:
            weighted_sum = [v / total_weight for v in weighted_sum]
        
        # 投影
        output = [sum(weighted_sum[k] * self.proj[j][k] for k in range(len(weighted_sum))) + self.bias[j]
                  for j in range(self.output_dim)]
        
        return output


class AttentionFusion(LateFusion):
    """注意力融合
    
    使用注意力机制动态融合各模态特征
    """
    
    def __init__(self, modalities: List[str], feature_dim: int, output_dim: int,
                 num_heads: int = 4):
        super().__init__(output_dim)
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        scale = 0.02
        
        # 查询向量
        self.query = [random.gauss(0, scale) for _ in range(feature_dim)]
        
        # 键投影
        self.key_proj = [[random.gauss(0, scale) for _ in range(feature_dim)] 
                         for _ in range(feature_dim)]
        
        # 值投影
        self.value_proj = [[random.gauss(0, scale) for _ in range(feature_dim)] 
                           for _ in range(feature_dim)]
        
        # 输出投影
        self.out_proj = [[random.gauss(0, scale) for _ in range(feature_dim)] 
                         for _ in range(output_dim)]
        self.out_bias = [0.0] * output_dim
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """注意力融合"""
        # 收集有效特征
        valid_features = []
        for mod in self.modalities:
            if mod in features and len(features[mod]) == self.feature_dim:
                valid_features.append(features[mod])
        
        if not valid_features:
            return [0.0] * self.output_dim
        
        # 计算键和值
        keys = [[sum(f[k] * self.key_proj[j][k] for k in range(len(f)))
                 for j in range(len(self.key_proj))] for f in valid_features]
        values = [[sum(f[k] * self.value_proj[j][k] for k in range(len(f)))
                   for j in range(len(self.value_proj))] for f in valid_features]
        
        # 计算注意力分数
        scores = [sum(self.query[d] * k[d] for d in range(len(self.query))) / math.sqrt(len(self.query))
                  for k in keys]
        
        # Softmax
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        attn_weights = [e / sum_exp for e in exp_scores]
        
        # 加权求和
        attended = [0.0] * len(values[0])
        for i, v in enumerate(values):
            for d in range(len(attended)):
                attended[d] += attn_weights[i] * v[d]
        
        # 输出投影
        output = [sum(attended[k] * self.out_proj[j][k] for k in range(len(attended))) + self.out_bias[j]
                  for j in range(self.output_dim)]
        
        return output


class GatedFusion(LateFusion):
    """门控融合
    
    使用门控机制动态选择各模态的贡献
    """
    
    def __init__(self, modalities: List[str], feature_dim: int, output_dim: int):
        super().__init__(output_dim)
        self.modalities = modalities
        self.feature_dim = feature_dim
        
        scale = 0.02
        
        # 门控网络
        self.gate_weights = {}
        self.gate_biases = {}
        for mod in modalities:
            self.gate_weights[mod] = [[random.gauss(0, scale) for _ in range(feature_dim)] 
                                      for _ in range(feature_dim)]
            self.gate_biases[mod] = [0.0] * feature_dim
        
        # 输出投影
        self.proj = [[random.gauss(0, scale) for _ in range(feature_dim)] 
                     for _ in range(output_dim)]
    
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """门控融合"""
        fused = [0.0] * self.feature_dim
        
        for mod in self.modalities:
            if mod not in features or len(features[mod]) != self.feature_dim:
                continue
            
            f = features[mod]
            
            # 计算门控值
            gate = [sum(f[k] * self.gate_weights[mod][j][k] for k in range(len(f))) + self.gate_biases[mod][j]
                    for j in range(self.feature_dim)]
            gate = [self._sigmoid(g) for g in gate]
            
            # 门控特征
            gated = [f[i] * gate[i] for i in range(len(f))]
            
            # 累加
            for i in range(len(fused)):
                fused[i] += gated[i]
        
        # 输出投影
        output = [sum(fused[k] * self.proj[j][k] for k in range(len(fused)))
                  for j in range(self.output_dim)]
        
        return output


class TensorFusion(LateFusion):
    """张量融合
    
    计算各模态特征的外积进行融合
    """
    
    def __init__(self, modalities: List[str], modality_dims: Dict[str, int], 
                 output_dim: int, rank: int = 32):
        super().__init__(output_dim)
        self.modalities = modalities
        self.modality_dims = modality_dims
        self.rank = rank
        
        scale = 0.02
        
        # 低秩分解参数
        self.factors = {}
        for mod in modalities:
            dim = modality_dims.get(mod, 0)
            self.factors[mod] = [[random.gauss(0, scale) for _ in range(dim)] 
                                 for _ in range(rank)]
        
        # 输出投影
        self.proj = [[random.gauss(0, scale) for _ in range(rank)] 
                     for _ in range(output_dim)]
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """张量融合"""
        # 计算各模态的低秩投影
        projections = []
        for mod in self.modalities:
            if mod in features:
                f = features[mod]
                factor = self.factors[mod]
                proj = [sum(f[k] * factor[j][k] for k in range(len(f)))
                        for j in range(self.rank)]
                projections.append(proj)
        
        if not projections:
            return [0.0] * self.output_dim
        
        # Hadamard积
        fused = projections[0]
        for proj in projections[1:]:
            fused = [fused[i] * proj[i] for i in range(len(fused))]
        
        # 输出投影
        output = [sum(fused[k] * self.proj[j][k] for k in range(len(fused)))
                  for j in range(self.output_dim)]
        
        return output


class HierarchicalFusion(LateFusion):
    """层次融合
    
    按层次逐步融合各模态特征
    """
    
    def __init__(self, fusion_hierarchy: List[List[str]], 
                 feature_dim: int, output_dim: int):
        """
        Args:
            fusion_hierarchy: 融合层次，如 [['vision', 'text'], ['audio']]
                             表示先融合vision和text，再与audio融合
        """
        super().__init__(output_dim)
        self.fusion_hierarchy = fusion_hierarchy
        self.feature_dim = feature_dim
        
        scale = 0.02
        
        # 各层的融合权重
        self.layer_weights = []
        for layer in fusion_hierarchy:
            weights = {mod: random.gauss(0, scale) for mod in layer}
            self.layer_weights.append(weights)
        
        # 输出投影
        self.proj = [[random.gauss(0, scale) for _ in range(feature_dim)] 
                     for _ in range(output_dim)]
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """层次融合"""
        current_features = features.copy()
        
        for layer_idx, layer in enumerate(self.fusion_hierarchy):
            # 获取当前层的特征
            layer_features = [current_features[mod] for mod in layer if mod in current_features]
            
            if not layer_features:
                continue
            
            # 加权融合
            weights = self.layer_weights[layer_idx]
            fused = [0.0] * self.feature_dim
            total_w = 0.0
            
            for mod, feat in zip(layer, layer_features):
                w = weights.get(mod, 1.0)
                for i in range(len(fused)):
                    fused[i] += w * feat[i]
                total_w += abs(w)
            
            if total_w > 0:
                fused = [v / total_w for v in fused]
            
            # 更新特征
            current_features[f'layer_{layer_idx}'] = fused
        
        # 获取最终特征
        final_key = f'layer_{len(self.fusion_hierarchy) - 1}'
        final_feature = current_features.get(final_key, list(current_features.values())[0] if current_features else [0.0] * self.feature_dim)
        
        # 输出投影
        output = [sum(final_feature[k] * self.proj[j][k] for k in range(len(final_feature)))
                  for j in range(self.output_dim)]
        
        return output


class EnsembleFusion(LateFusion):
    """集成融合
    
    组合多种融合策略
    """
    
    def __init__(self, fusion_methods: List[LateFusion], output_dim: int,
                 weights: Optional[List[float]] = None):
        super().__init__(output_dim)
        self.fusion_methods = fusion_methods
        self.weights = weights or [1.0 / len(fusion_methods)] * len(fusion_methods)
    
    def __call__(self, features: Dict[str, List[float]]) -> List[float]:
        """集成融合"""
        outputs = [method(features) for method in self.fusion_methods]
        
        # 加权平均
        fused = [0.0] * self.output_dim
        for i, output in enumerate(outputs):
            w = self.weights[i]
            for j in range(len(fused)):
                fused[j] += w * output[j]
        
        return fused


def create_attention_fusion(modalities: List[str], feature_dim: int, 
                            output_dim: int) -> AttentionFusion:
    """创建注意力融合"""
    return AttentionFusion(modalities, feature_dim, output_dim)


def create_gated_fusion(modalities: List[str], feature_dim: int, 
                        output_dim: int) -> GatedFusion:
    """创建门控融合"""
    return GatedFusion(modalities, feature_dim, output_dim)
