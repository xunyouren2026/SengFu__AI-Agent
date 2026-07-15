"""
SigLIP视觉编码器
实现基于Sigmoid Loss的图像-文本对齐编码器
"""
from typing import Optional, Tuple, List, Dict, Any
import math
import random


class LayerNorm:
    """层归一化"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim
        self.bias = [0.0] * dim
    
    def __call__(self, x: List[float]) -> List[float]:
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        x_norm = [(xi - mean) / math.sqrt(var + self.eps) for xi in x]
        return [w * xn + b for w, xn, b in zip(self.weight, self.bias, x_norm)]


class SigLIPAttention:
    """SigLIP多头注意力"""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        scale_init = 0.02
        # 独立的Q, K, V投影
        self.q_proj = [[random.gauss(0, scale_init) for _ in range(dim)] 
                       for _ in range(dim)]
        self.k_proj = [[random.gauss(0, scale_init) for _ in range(dim)] 
                       for _ in range(dim)]
        self.v_proj = [[random.gauss(0, scale_init) for _ in range(dim)] 
                       for _ in range(dim)]
        
        self.q_bias = [0.0] * dim if qkv_bias else None
        self.k_bias = [0.0] * dim if qkv_bias else None
        self.v_bias = [0.0] * dim if qkv_bias else None
        
        self.proj = [[random.gauss(0, scale_init) for _ in range(dim)] 
                     for _ in range(dim)]
        self.proj_bias = [0.0] * dim
        
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
    
    def _linear(self, x: List[float], weight: List[List[float]], 
                bias: Optional[List[float]] = None) -> List[float]:
        out = []
        for j in range(len(weight)):
            val = sum(x[i] * weight[j][i] for i in range(len(x)))
            if bias:
                val += bias[j]
            out.append(val)
        return out
    
    def __call__(self, x: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """返回输出和注意力权重"""
        # 计算Q, K, V
        q = [self._linear(xi, self.q_proj, self.q_bias) for xi in x]
        k = [self._linear(xi, self.k_proj, self.k_bias) for xi in x]
        v = [self._linear(xi, self.v_proj, self.v_bias) for xi in x]
        
        # 注意力分数
        attn = []
        for i in range(len(q)):
            row = []
            for j in range(len(k)):
                score = sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale
                row.append(score)
            attn.append(row)
        
        # Softmax
        attn_weights = []
        for row in attn:
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
        result = [self._linear(o, self.proj, self.proj_bias) for o in output]
        
        return result, attn_weights


class SigLIPMLP:
    """SigLIP前馈网络"""
    
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        scale = 0.02
        self.fc1 = [[random.gauss(0, scale) for _ in range(in_features)] 
                    for _ in range(hidden_features)]
        self.fc1_bias = [0.0] * hidden_features
        
        self.fc2 = [[random.gauss(0, scale) for _ in range(hidden_features)] 
                    for _ in range(in_features)]
        self.fc2_bias = [0.0] * in_features
        
        self.drop = drop
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        hidden = []
        for xi in x:
            h = [self._gelu(sum(xi[k] * self.fc1[j][k] for k in range(len(xi))) + self.fc1_bias[j])
                 for j in range(len(self.fc1))]
            hidden.append(h)
        
        output = []
        for h in hidden:
            o = [sum(h[k] * self.fc2[j][k] for k in range(len(h))) + self.fc2_bias[j]
                 for j in range(len(self.fc2))]
            output.append(o)
        
        return output


class SigLIPBlock:
    """SigLIP Transformer块"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        self.norm1 = LayerNorm(dim)
        self.attn = SigLIPAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = LayerNorm(dim)
        self.mlp = SigLIPMLP(dim, int(dim * mlp_ratio), drop)
    
    def __call__(self, x: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        # 注意力
        normed = [self.norm1(xi) for xi in x]
        attn_out, attn_weights = self.attn(normed)
        x = [[x[i][j] + attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        # MLP
        normed = [self.norm2(xi) for xi in x]
        mlp_out = self.mlp(normed)
        x = [[x[i][j] + mlp_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        return x, attn_weights


class SigLIPPatchEmbed:
    """SigLIP图像分块嵌入"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        scale = 0.02
        patch_dim = in_chans * patch_size * patch_size
        self.proj = [[random.gauss(0, scale) for _ in range(patch_dim)] 
                     for _ in range(embed_dim)]
        self.bias = [0.0] * embed_dim
    
    def __call__(self, x: List[List[List[List[float]]]]) -> List[List[float]]:
        """图像转patch嵌入"""
        B = len(x)
        H, W = self.img_size, self.img_size
        P = self.patch_size
        
        patches = []
        for b in range(B):
            for i in range(0, H, P):
                for j in range(0, W, P):
                    patch = []
                    for c in range(len(x[b])):
                        for pi in range(P):
                            for pj in range(P):
                                patch.append(x[b][c][i + pi][j + pj])
                    
                    embedded = [sum(patch[k] * self.proj[d][k] for k in range(len(patch))) + self.bias[d]
                               for d in range(len(self.proj))]
                    patches.append(embedded)
        
        return patches


class SigLIPVisionEncoder:
    """SigLIP视觉编码器
    
    使用Sigmoid Loss训练的视觉编码器，相比CLIP有更好的训练稳定性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # 模型配置
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.in_chans = config.get('in_chans', 3)
        self.embed_dim = config.get('embed_dim', 768)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 12)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.head_dim = config.get('head_dim', 64)
        self.out_dim = config.get('out_dim', 512)
        
        # 分块嵌入
        self.patch_embed = SigLIPPatchEmbed(self.img_size, self.patch_size, 
                                            self.in_chans, self.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = [random.gauss(0, 0.02) for _ in range(self.embed_dim)]
        
        # 位置嵌入
        num_pos = self.num_patches + 1
        self.pos_embed = [[random.gauss(0, 0.02) for _ in range(self.embed_dim)] 
                          for _ in range(num_pos)]
        
        # Transformer块
        self.blocks = [
            SigLIPBlock(self.embed_dim, self.num_heads, self.mlp_ratio)
            for _ in range(self.depth)
        ]
        
        # 最终归一化
        self.norm = LayerNorm(self.embed_dim)
        
        # 输出头
        self.head = [[random.gauss(0, 0.02) for _ in range(self.embed_dim)] 
                     for _ in range(self.out_dim)]
        self.head_bias = [0.0] * self.out_dim
    
    def encode_image(self, x: List[List[List[List[float]]]]) -> List[List[float]]:
        """
        编码图像为特征向量
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            图像特征 [B, out_dim]
        """
        B = len(x)
        
        # Patch嵌入
        patches = self.patch_embed(x)
        
        # 重塑
        patches_per_img = self.num_patches
        x_reshaped = []
        for b in range(B):
            x_reshaped.append(patches[b * patches_per_img:(b + 1) * patches_per_img])
        
        # 添加CLS token和位置嵌入
        all_tokens = []
        for b in range(B):
            tokens = [self.cls_token.copy()] + x_reshaped[b]
            tokens = [[tokens[i][j] + self.pos_embed[i][j] for j in range(len(tokens[i]))] 
                      for i in range(len(tokens))]
            all_tokens.extend(tokens)
        
        # 通过Transformer块
        for block in self.blocks:
            all_tokens, _ = block(all_tokens)
        
        # 归一化
        all_tokens = [self.norm(t) for t in all_tokens]
        
        # 提取CLS token
        total_tokens_per_img = self.num_patches + 1
        cls_tokens = [all_tokens[b * total_tokens_per_img] for b in range(B)]
        
        # 投影到输出维度
        outputs = []
        for cls in cls_tokens:
            out = [sum(cls[k] * self.head[j][k] for k in range(len(cls))) + self.head_bias[j]
                   for j in range(self.out_dim)]
            outputs.append(out)
        
        return outputs
    
    def get_attention_maps(self, x: List[List[List[List[float]]]]) -> List[List[List[float]]]:
        """
        获取注意力图
        
        Args:
            x: 输入图像
        
        Returns:
            各层的注意力权重
        """
        B = len(x)
        
        patches = self.patch_embed(x)
        patches_per_img = self.num_patches
        x_reshaped = []
        for b in range(B):
            x_reshaped.append(patches[b * patches_per_img:(b + 1) * patches_per_img])
        
        all_tokens = []
        for b in range(B):
            tokens = [self.cls_token.copy()] + x_reshaped[b]
            tokens = [[tokens[i][j] + self.pos_embed[i][j] for j in range(len(tokens[i]))] 
                      for i in range(len(tokens))]
            all_tokens.extend(tokens)
        
        # 收集注意力权重
        attention_maps = []
        for block in self.blocks:
            all_tokens, attn_weights = block(all_tokens)
            attention_maps.append(attn_weights)
        
        return attention_maps
    
    def get_patch_features(self, x: List[List[List[List[float]]]]) -> List[List[List[float]]]:
        """
        获取patch级别特征
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            patch特征 [B, num_patches, embed_dim]
        """
        B = len(x)
        
        patches = self.patch_embed(x)
        patches_per_img = self.num_patches
        x_reshaped = []
        for b in range(B):
            x_reshaped.append(patches[b * patches_per_img:(b + 1) * patches_per_img])
        
        all_tokens = []
        for b in range(B):
            tokens = [self.cls_token.copy()] + x_reshaped[b]
            tokens = [[tokens[i][j] + self.pos_embed[i][j] for j in range(len(tokens[i]))] 
                      for i in range(len(tokens))]
            all_tokens.extend(tokens)
        
        for block in self.blocks:
            all_tokens, _ = block(all_tokens)
        
        all_tokens = [self.norm(t) for t in all_tokens]
        
        total_tokens_per_img = self.num_patches + 1
        patch_features = []
        for b in range(B):
            start = b * total_tokens_per_img + 1
            patch_features.append(all_tokens[start:start + self.num_patches])
        
        return patch_features
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'in_chans': self.in_chans,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'head_dim': self.head_dim,
            'out_dim': self.out_dim
        }


class SigLIPPreprocessor:
    """SigLIP图像预处理器"""
    
    def __init__(self, img_size: int = 224, mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None):
        self.img_size = img_size
        self.mean = mean or [0.5, 0.5, 0.5]
        self.std = std or [0.5, 0.5, 0.5]
    
    def resize(self, image: List[List[List[float]]], size: int) -> List[List[List[float]]]:
        """调整图像大小"""
        h, w = len(image[0]), len(image[0][0])
        c = len(image)
        
        new_image = [[[0.0 for _ in range(size)] for _ in range(size)] for _ in range(c)]
        
        for ch in range(c):
            for i in range(size):
                for j in range(size):
                    src_i = i * (h - 1) / (size - 1) if size > 1 else 0
                    src_j = j * (w - 1) / (size - 1) if size > 1 else 0
                    
                    i0, j0 = int(src_i), int(src_j)
                    i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
                    
                    di, dj = src_i - i0, src_j - j0
                    val = (image[ch][i0][j0] * (1 - di) * (1 - dj) +
                           image[ch][i0][j1] * (1 - di) * dj +
                           image[ch][i1][j0] * di * (1 - dj) +
                           image[ch][i1][j1] * di * dj)
                    new_image[ch][i][j] = val
        
        return new_image
    
    def normalize(self, image: List[List[List[float]]]) -> List[List[List[float]]]:
        """归一化"""
        c = len(image)
        h, w = len(image[0]), len(image[0][0])
        
        normalized = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]
        for ch in range(c):
            for i in range(h):
                for j in range(w):
                    normalized[ch][i][j] = (image[ch][i][j] - self.mean[ch]) / self.std[ch]
        
        return normalized
    
    def __call__(self, image: List[List[List[float]]]) -> List[List[List[float]]]:
        """预处理流程"""
        resized = self.resize(image, self.img_size)
        normalized = self.normalize(resized)
        return normalized


def create_siglip_encoder(model_name: str = 'siglip_base') -> SigLIPVisionEncoder:
    """
    创建预配置的SigLIP编码器
    
    Args:
        model_name: 模型名称
    
    Returns:
        SigLIP编码器
    """
    configs = {
        'siglip_base': {
            'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
            'depth': 12, 'num_heads': 12, 'out_dim': 512
        },
        'siglip_large': {
            'img_size': 224, 'patch_size': 16, 'embed_dim': 1024,
            'depth': 24, 'num_heads': 16, 'out_dim': 768
        },
        'siglip_so400m': {
            'img_size': 224, 'patch_size': 14, 'embed_dim': 1152,
            'depth': 27, 'num_heads': 16, 'out_dim': 768
        }
    }
    
    config = configs.get(model_name, configs['siglip_base'])
    return SigLIPVisionEncoder(config)
