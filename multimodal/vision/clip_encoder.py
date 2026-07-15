"""
CLIP视觉编码器 - 图像嵌入提取
实现基于Vision Transformer的图像编码器
"""
from typing import Optional, Tuple, List, Dict, Any
import math
import random


class LayerNormalization:
    """层归一化"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = [1.0] * normalized_shape
        self.bias = [0.0] * normalized_shape
    
    def __call__(self, x: List[float]) -> List[float]:
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        std = math.sqrt(var + self.eps)
        normalized = [(xi - mean) / std for xi in x]
        return [w * n + b for w, n, b in zip(self.weight, self.bias, normalized)]


class MultiHeadAttention:
    """多头自注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V投影权重
        self.q_proj_weight = self._init_weight(embed_dim, embed_dim)
        self.k_proj_weight = self._init_weight(embed_dim, embed_dim)
        self.v_proj_weight = self._init_weight(embed_dim, embed_dim)
        self.out_proj_weight = self._init_weight(embed_dim, embed_dim)
        self.out_proj_bias = [0.0] * embed_dim
        
        self.dropout = dropout
    
    def _init_weight(self, rows: int, cols: int) -> List[List[float]]:
        scale = 0.02
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]
    
    def _linear(self, x: List[List[float]], weight: List[List[float]], 
                bias: Optional[List[float]] = None) -> List[List[float]]:
        """线性变换"""
        result = []
        for xi in x:
            out = []
            for j in range(len(weight)):
                val = sum(xi[k] * weight[j][k] for k in range(len(xi)))
                if bias:
                    val += bias[j]
                out.append(val)
            result.append(out)
        return result
    
    def _softmax(self, x: List[List[float]]) -> List[List[float]]:
        """Softmax函数"""
        result = []
        for row in x:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            result.append([e / sum_exp for e in exp_vals])
        return result
    
    def __call__(self, query: List[List[float]], key: List[List[float]], 
                 value: List[List[float]], 
                 attn_mask: Optional[List[List[float]]] = None) -> Tuple[List[List[float]], List[List[float]]]:
        """前向传播"""
        batch_size = len(query)
        seq_len = len(query[0]) if query else 0
        
        # 投影
        q = self._linear(query, self.q_proj_weight)
        k = self._linear(key, self.k_proj_weight)
        v = self._linear(value, self.v_proj_weight)
        
        # 重塑为多头形式 (简化处理)
        # 计算注意力分数
        attn_scores = []
        for i in range(len(q)):
            row = []
            for j in range(len(k)):
                score = sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale
                row.append(score)
            attn_scores.append(row)
        
        # 应用mask
        if attn_mask is not None:
            for i in range(len(attn_scores)):
                for j in range(len(attn_scores[i])):
                    attn_scores[i][j] += attn_mask[i][j]
        
        # Softmax
        attn_weights = self._softmax(attn_scores)
        
        # 加权求和
        output = []
        for i in range(len(attn_weights)):
            out = [0.0] * len(v[0]) if v else []
            for j in range(len(attn_weights[i])):
                for d in range(len(out)):
                    out[d] += attn_weights[i][j] * v[j][d]
            output.append(out)
        
        # 输出投影
        output = self._linear(output, self.out_proj_weight, self.out_proj_bias)
        
        return output, attn_weights


class MLP:
    """前馈神经网络"""
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        self.fc1_weight = self._init_weight(hidden_dim, embed_dim)
        self.fc1_bias = [0.0] * hidden_dim
        self.fc2_weight = self._init_weight(embed_dim, hidden_dim)
        self.fc2_bias = [0.0] * embed_dim
        self.dropout = dropout
    
    def _init_weight(self, rows: int, cols: int) -> List[List[float]]:
        scale = 0.02
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]
    
    def _gelu(self, x: float) -> float:
        """GELU激活函数"""
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """前向传播"""
        # 第一层
        hidden = []
        for xi in x:
            out = []
            for j in range(len(self.fc1_weight)):
                val = sum(xi[k] * self.fc1_weight[j][k] for k in range(len(xi)))
                val += self.fc1_bias[j]
                val = self._gelu(val)
                out.append(val)
            hidden.append(out)
        
        # 第二层
        output = []
        for h in hidden:
            out = []
            for j in range(len(self.fc2_weight)):
                val = sum(h[k] * self.fc2_weight[j][k] for k in range(len(h)))
                val += self.fc2_bias[j]
                out.append(val)
            output.append(out)
        
        return output


class TransformerBlock:
    """Transformer编码器块"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, ff_dim, dropout)
        self.norm1 = LayerNormalization(embed_dim)
        self.norm2 = LayerNormalization(embed_dim)
        self.dropout = dropout
    
    def __call__(self, x: List[List[float]], 
                 attn_mask: Optional[List[List[float]]] = None) -> List[List[float]]:
        """前向传播，带残差连接"""
        # 自注意力
        normed = [self.norm1(xi) for xi in x]
        attn_out, _ = self.attention(normed, normed, normed, attn_mask)
        x = [[xi[d] + attn_out[i][d] for d in range(len(xi))] for i, xi in enumerate(x)]
        
        # MLP
        normed = [self.norm2(xi) for xi in x]
        mlp_out = self.mlp(normed)
        x = [[xi[d] + mlp_out[i][d] for d in range(len(xi))] for i, xi in enumerate(x)]
        
        return x


class PatchEmbedding:
    """图像分块嵌入"""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim
        
        # 投影权重
        scale = 0.02
        self.proj_weight = [[random.gauss(0, scale) for _ in range(self.patch_dim)] 
                           for _ in range(embed_dim)]
        self.proj_bias = [0.0] * embed_dim
    
    def __call__(self, image: List[List[List[List[float]]]]) -> List[List[float]]:
        """
        将图像转换为patch嵌入
        image: [batch, channels, height, width]
        返回: [batch * num_patches, embed_dim]
        """
        patches = []
        batch_size = len(image)
        
        for b in range(batch_size):
            for i in range(0, self.image_size, self.patch_size):
                for j in range(0, self.image_size, self.patch_size):
                    # 提取patch
                    patch = []
                    for c in range(len(image[b])):
                        for pi in range(self.patch_size):
                            for pj in range(self.patch_size):
                                patch.append(image[b][c][i + pi][j + pj])
                    
                    # 投影
                    embedded = []
                    for k in range(self.embed_dim):
                        val = sum(patch[m] * self.proj_weight[k][m] 
                                 for m in range(len(patch)))
                        val += self.proj_bias[k]
                        embedded.append(val)
                    patches.append(embedded)
        
        return patches


class CLIPVisionEncoder:
    """CLIP视觉编码器
    
    基于Vision Transformer架构，实现图像到嵌入向量的转换
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # 默认配置 (CLIP ViT-B/16)
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.in_channels = config.get('in_channels', 3)
        self.embed_dim = config.get('embed_dim', 768)
        self.num_heads = config.get('num_heads', 12)
        self.num_layers = config.get('num_layers', 12)
        self.ff_dim = config.get('ff_dim', 3072)
        self.dropout = config.get('dropout', 0.0)
        self.output_dim = config.get('output_dim', 512)
        
        # 分块嵌入
        self.patch_embed = PatchEmbedding(
            self.image_size, self.patch_size, 
            self.in_channels, self.embed_dim
        )
        
        # 类别token和位置嵌入
        self.class_token = [random.gauss(0, 0.02) for _ in range(self.embed_dim)]
        num_positions = self.patch_embed.num_patches + 1
        self.position_embedding = [[random.gauss(0, 0.02) for _ in range(self.embed_dim)] 
                                   for _ in range(num_positions)]
        
        # Transformer块
        self.transformer_blocks = [
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ]
        
        # 最终层归一化
        self.final_norm = LayerNormalization(self.embed_dim)
        
        # 投影到输出维度
        self.proj = [[random.gauss(0, 0.02) for _ in range(self.embed_dim)] 
                     for _ in range(self.output_dim)]
    
    def _add_position_embedding(self, x: List[List[float]]) -> List[List[float]]:
        """添加位置嵌入"""
        result = []
        for i, xi in enumerate(x):
            if i < len(self.position_embedding):
                result.append([xi[d] + self.position_embedding[i][d] 
                              for d in range(len(xi))])
            else:
                result.append(xi)
        return result
    
    def encode_image(self, image: List[List[List[List[float]]]]) -> List[List[float]]:
        """
        编码图像为特征向量
        
        Args:
            image: 输入图像 [batch, channels, height, width]
        
        Returns:
            图像特征 [batch, embed_dim]
        """
        batch_size = len(image)
        
        # 分块嵌入
        patches = self.patch_embed(image)
        
        # 重塑为 [batch, num_patches, embed_dim]
        num_patches_per_image = self.patch_embed.num_patches
        batch_patches = []
        for b in range(batch_size):
            start = b * num_patches_per_image
            batch_patches.append(patches[start:start + num_patches_per_image])
        
        # 添加类别token和位置嵌入
        all_features = []
        for b in range(batch_size):
            # 添加类别token
            features = [self.class_token.copy()] + batch_patches[b]
            # 添加位置嵌入
            features = self._add_position_embedding(features)
            all_features.append(features)
        
        # 合并处理
        all_tokens = []
        for features in all_features:
            all_tokens.extend(features)
        
        # 通过Transformer块
        for block in self.transformer_blocks:
            all_tokens = block(all_tokens)
        
        # 提取类别token并归一化
        outputs = []
        for b in range(batch_size):
            cls_token_idx = b * (num_patches_per_image + 1)
            cls_token = all_tokens[cls_token_idx]
            normalized = self.final_norm(cls_token)
            
            # 投影到输出维度
            output = [sum(normalized[k] * self.proj[j][k] for k in range(len(normalized)))
                     for j in range(self.output_dim)]
            outputs.append(output)
        
        return outputs
    
    def get_intermediate_features(self, image: List[List[List[List[float]]]], 
                                  layer_indices: Optional[List[int]] = None) -> List[List[List[float]]]:
        """
        获取中间层特征
        
        Args:
            image: 输入图像
            layer_indices: 要获取的层索引列表
        
        Returns:
            各层的特征表示
        """
        if layer_indices is None:
            layer_indices = [self.num_layers - 1]
        
        batch_size = len(image)
        patches = self.patch_embed(image)
        num_patches_per_image = self.patch_embed.num_patches
        
        all_features = []
        for b in range(batch_size):
            features = [self.class_token.copy()] + patches[b * num_patches_per_image:(b + 1) * num_patches_per_image]
            features = self._add_position_embedding(features)
            all_features.append(features)
        
        all_tokens = []
        for features in all_features:
            all_tokens.extend(features)
        
        intermediate_features = []
        for i, block in enumerate(self.transformer_blocks):
            all_tokens = block(all_tokens)
            if i in layer_indices:
                intermediate_features.append([t.copy() for t in all_tokens])
        
        return intermediate_features
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'output_dim': self.output_dim
        }


class CLIPImagePreprocessor:
    """CLIP图像预处理器"""
    
    def __init__(self, image_size: int = 224, mean: Optional[List[float]] = None, 
                 std: Optional[List[float]] = None):
        self.image_size = image_size
        self.mean = mean or [0.48145466, 0.4578275, 0.40821073]
        self.std = std or [0.26862954, 0.26130258, 0.27577711]
    
    def resize(self, image: List[List[List[float]]], 
               target_size: int) -> List[List[List[float]]]:
        """双线性插值缩放"""
        h, w = len(image[0]), len(image[0][0])
        channels = len(image)
        
        new_image = [[[0.0 for _ in range(target_size)] for _ in range(target_size)] 
                     for _ in range(channels)]
        
        for c in range(channels):
            for i in range(target_size):
                for j in range(target_size):
                    # 映射回原图坐标
                    src_i = i * (h - 1) / (target_size - 1) if target_size > 1 else 0
                    src_j = j * (w - 1) / (target_size - 1) if target_size > 1 else 0
                    
                    i0, j0 = int(src_i), int(src_j)
                    i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
                    
                    # 双线性插值
                    di, dj = src_i - i0, src_j - j0
                    val = (image[c][i0][j0] * (1 - di) * (1 - dj) +
                           image[c][i0][j1] * (1 - di) * dj +
                           image[c][i1][j0] * di * (1 - dj) +
                           image[c][i1][j1] * di * dj)
                    new_image[c][i][j] = val
        
        return new_image
    
    def center_crop(self, image: List[List[List[float]]], 
                    crop_size: int) -> List[List[List[float]]]:
        """中心裁剪"""
        h, w = len(image[0]), len(image[0][0])
        channels = len(image)
        
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        
        cropped = [[[image[c][start_h + i][start_w + j] 
                    for j in range(crop_size)] 
                   for i in range(crop_size)] 
                  for c in range(channels)]
        
        return cropped
    
    def normalize(self, image: List[List[List[float]]]) -> List[List[List[float]]]:
        """归一化"""
        channels = len(image)
        normalized = [[[0.0 for _ in range(len(image[0][0]))] 
                      for _ in range(len(image[0]))] 
                     for _ in range(channels)]
        
        for c in range(channels):
            for i in range(len(image[0])):
                for j in range(len(image[0][0])):
                    normalized[c][i][j] = (image[c][i][j] - self.mean[c]) / self.std[c]
        
        return normalized
    
    def __call__(self, image: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        完整预处理流程
        
        Args:
            image: 输入图像 [channels, height, width]，值范围[0, 1]
        
        Returns:
            预处理后的图像
        """
        # 缩放
        resized = self.resize(image, self.image_size)
        # 归一化
        normalized = self.normalize(resized)
        return normalized


def create_clip_vision_encoder(model_name: str = 'ViT-B/16') -> CLIPVisionEncoder:
    """
    创建预配置的CLIP视觉编码器
    
    Args:
        model_name: 模型名称，支持 'ViT-B/16', 'ViT-B/32', 'ViT-L/14'
    
    Returns:
        配置好的CLIP视觉编码器
    """
    configs = {
        'ViT-B/16': {
            'image_size': 224, 'patch_size': 16, 'embed_dim': 768,
            'num_heads': 12, 'num_layers': 12, 'ff_dim': 3072, 'output_dim': 512
        },
        'ViT-B/32': {
            'image_size': 224, 'patch_size': 32, 'embed_dim': 768,
            'num_heads': 12, 'num_layers': 12, 'ff_dim': 3072, 'output_dim': 512
        },
        'ViT-L/14': {
            'image_size': 224, 'patch_size': 14, 'embed_dim': 1024,
            'num_heads': 16, 'num_layers': 24, 'ff_dim': 4096, 'output_dim': 768
        }
    }
    
    config = configs.get(model_name, configs['ViT-B/16'])
    return CLIPVisionEncoder(config)
