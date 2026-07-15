"""
DINOv2自监督视觉编码器
实现基于自监督学习的视觉特征提取器
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


class DropPath:
    """随机深度 (Stochastic Depth)"""
    
    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob
    
    def __call__(self, x: List[float], training: bool = True) -> List[float]:
        if not training or self.drop_prob == 0.0:
            return x
        if random.random() < self.drop_prob:
            return [0.0] * len(x)
        return [xi / (1 - self.drop_prob) for xi in x]


class Attention:
    """多头自注意力"""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        scale_init = 0.02
        self.qkv = [[random.gauss(0, scale_init) for _ in range(dim)] 
                    for _ in range(3 * dim)]
        self.q_bias = [0.0] * dim if qkv_bias else None
        self.v_bias = [0.0] * dim if qkv_bias else None
        
        # 输出投影
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
    
    def __call__(self, x: List[List[float]], 
                 attn_mask: Optional[List[List[float]]] = None) -> List[List[float]]:
        B, N = len(x), len(x[0]) if x else 0
        C = len(x[0]) if x and x[0] else 0
        
        # 计算QKV
        qkv = []
        for xi in x:
            qkv_i = self._linear(xi, self.qkv)
            qkv.append(qkv_i)
        
        # 分离Q, K, V (简化处理)
        q = [[qkv_i[j] for j in range(C)] for qkv_i in qkv]
        k = [[qkv_i[j + C] for j in range(C)] for qkv_i in qkv]
        v = [[qkv_i[j + 2 * C] for j in range(C)] for qkv_i in qkv]
        
        # 注意力计算
        attn = []
        for i in range(len(q)):
            row = []
            for j in range(len(k)):
                score = sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale
                row.append(score)
            attn.append(row)
        
        # Softmax
        attn_out = []
        for row in attn:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            attn_out.append([e / sum_exp for e in exp_vals])
        
        # 加权求和
        output = []
        for i in range(len(attn_out)):
            out = [0.0] * len(v[0]) if v else []
            for j in range(len(attn_out[i])):
                for d in range(len(out)):
                    out[d] += attn_out[i][j] * v[j][d]
            output.append(out)
        
        # 输出投影
        result = []
        for o in output:
            result.append(self._linear(o, self.proj, self.proj_bias))
        
        return result


class MLP:
    """前馈网络"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        scale = 0.02
        self.fc1 = [[random.gauss(0, scale) for _ in range(in_features)] 
                    for _ in range(hidden_features)]
        self.fc1_bias = [0.0] * hidden_features
        
        self.fc2 = [[random.gauss(0, scale) for _ in range(hidden_features)] 
                    for _ in range(out_features)]
        self.fc2_bias = [0.0] * out_features
        
        self.drop = drop
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        # FC1 + GELU
        hidden = []
        for xi in x:
            h = [sum(xi[k] * self.fc1[j][k] for k in range(len(xi))) + self.fc1_bias[j]
                 for j in range(len(self.fc1))]
            h = [self._gelu(hi) for hi in h]
            hidden.append(h)
        
        # FC2
        output = []
        for h in hidden:
            o = [sum(h[k] * self.fc2[j][k] for k in range(len(h))) + self.fc2_bias[j]
                 for j in range(len(self.fc2))]
            output.append(o)
        
        return output


class Block:
    """Transformer块"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, act_layer: str = 'gelu'):
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
    
    def __call__(self, x: List[List[float]], 
                 attn_mask: Optional[List[List[float]]] = None) -> List[List[float]]:
        # 注意力 + 残差
        normed = [self.norm1(xi) for xi in x]
        attn_out = self.attn(normed, attn_mask)
        x = [[x[i][j] + self.drop_path([attn_out[i][j]])[0] 
              for j in range(len(x[i]))] for i in range(len(x))]
        
        # MLP + 残差
        normed = [self.norm2(xi) for xi in x]
        mlp_out = self.mlp(normed)
        x = [[x[i][j] + self.drop_path([mlp_out[i][j]])[0] 
              for j in range(len(x[i]))] for i in range(len(x))]
        
        return x


class PatchEmbed:
    """图像分块嵌入"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        scale = 0.02
        patch_dim = in_chans * patch_size * patch_size
        self.proj = [[random.gauss(0, scale) for _ in range(patch_dim)] 
                     for _ in range(embed_dim)]
    
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
                    
                    embedded = [sum(patch[k] * self.proj[d][k] for k in range(len(patch)))
                               for d in range(len(self.proj))]
                    patches.append(embedded)
        
        return patches


class DINOv2Encoder:
    """DINOv2自监督视觉编码器
    
    实现基于自监督学习的视觉特征提取，支持全局和局部特征提取
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # 模型配置
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 14)
        self.in_chans = config.get('in_chans', 3)
        self.embed_dim = config.get('embed_dim', 768)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 12)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.out_dim = config.get('out_dim', 768)
        
        # 分块嵌入
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, 
                                       self.in_chans, self.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = [random.gauss(0, 0.02) for _ in range(self.embed_dim)]
        
        # 位置嵌入
        num_pos = self.num_patches + 1
        self.pos_embed = [[random.gauss(0, 0.02) for _ in range(self.embed_dim)] 
                          for _ in range(num_pos)]
        self.pos_drop = 0.0
        
        # Transformer块
        dpr = [x / self.depth for x in range(self.depth)]  # drop path rate
        self.blocks = [
            Block(self.embed_dim, self.num_heads, self.mlp_ratio,
                  drop_path=dpr[i]) for i in range(self.depth)
        ]
        
        # 最终归一化
        self.norm = LayerNorm(self.embed_dim)
        
        # 输出投影
        self.head = [[random.gauss(0, 0.02) for _ in range(self.embed_dim)] 
                     for _ in range(self.out_dim)]
    
    def _get_pos_embed(self) -> List[List[float]]:
        """获取位置嵌入"""
        return self.pos_embed
    
    def forward_features(self, x: List[List[List[List[float]]]]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        提取特征
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            cls_token: 全局特征 [B, embed_dim]
            patch_tokens: 局部特征 [B, num_patches, embed_dim]
        """
        B = len(x)
        
        # Patch嵌入
        x = self.patch_embed(x)
        
        # 重塑为 [B, num_patches, embed_dim]
        patches_per_img = self.num_patches
        x_reshaped = []
        for b in range(B):
            x_reshaped.append(x[b * patches_per_img:(b + 1) * patches_per_img])
        
        # 添加CLS token和位置嵌入
        pos_embed = self._get_pos_embed()
        all_tokens = []
        for b in range(B):
            tokens = [self.cls_token.copy()] + x_reshaped[b]
            tokens = [[tokens[i][j] + pos_embed[i][j] for j in range(len(tokens[i]))] 
                      for i in range(len(tokens))]
            all_tokens.extend(tokens)
        
        # 通过Transformer块
        for block in self.blocks:
            all_tokens = block(all_tokens)
        
        # 归一化
        all_tokens = [self.norm(t) for t in all_tokens]
        
        # 分离CLS token和patch tokens
        cls_tokens = []
        patch_tokens = []
        total_tokens_per_img = self.num_patches + 1
        
        for b in range(B):
            start = b * total_tokens_per_img
            cls_tokens.append(all_tokens[start])
            patch_tokens.append(all_tokens[start + 1:start + total_tokens_per_img])
        
        return cls_tokens, patch_tokens
    
    def encode_image(self, x: List[List[List[List[float]]]]) -> List[List[float]]:
        """
        编码图像为全局特征
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            全局特征 [B, out_dim]
        """
        cls_tokens, _ = self.forward_features(x)
        
        # 投影到输出维度
        outputs = []
        for cls in cls_tokens:
            out = [sum(cls[k] * self.head[j][k] for k in range(len(cls)))
                   for j in range(self.out_dim)]
            outputs.append(out)
        
        return outputs
    
    def get_local_features(self, x: List[List[List[List[float]]]]) -> List[List[List[float]]]:
        """
        获取局部特征 (patch级别)
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            局部特征 [B, num_patches, embed_dim]
        """
        _, patch_tokens = self.forward_features(x)
        return patch_tokens
    
    def get_intermediate_layers(self, x: List[List[List[List[float]]]], 
                                n: int = 1) -> List[List[List[List[float]]]]:
        """
        获取中间层特征
        
        Args:
            x: 输入图像
            n: 返回最后n层的特征
        
        Returns:
            中间层特征列表
        """
        B = len(x)
        
        x = self.patch_embed(x)
        patches_per_img = self.num_patches
        x_reshaped = []
        for b in range(B):
            x_reshaped.append(x[b * patches_per_img:(b + 1) * patches_per_img])
        
        pos_embed = self._get_pos_embed()
        all_tokens = []
        for b in range(B):
            tokens = [self.cls_token.copy()] + x_reshaped[b]
            tokens = [[tokens[i][j] + pos_embed[i][j] for j in range(len(tokens[i]))] 
                      for i in range(len(tokens))]
            all_tokens.extend(tokens)
        
        # 收集中间层输出
        intermediates = []
        for i, block in enumerate(self.blocks):
            all_tokens = block(all_tokens)
            if i >= self.depth - n:
                intermediates.append([[tj for tj in t] for t in all_tokens])
        
        return intermediates
    
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
            'out_dim': self.out_dim
        }


class DINOv2Preprocessor:
    """DINOv2图像预处理器"""
    
    def __init__(self, img_size: int = 224, mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None):
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
    
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


def create_dino_v2_encoder(model_name: str = 'dinov2_vitb14') -> DINOv2Encoder:
    """
    创建预配置的DINOv2编码器
    
    Args:
        model_name: 模型名称
    
    Returns:
        DINOv2编码器
    """
    configs = {
        'dinov2_vits14': {
            'img_size': 224, 'patch_size': 14, 'embed_dim': 384,
            'depth': 12, 'num_heads': 6, 'out_dim': 384
        },
        'dinov2_vitb14': {
            'img_size': 224, 'patch_size': 14, 'embed_dim': 768,
            'depth': 12, 'num_heads': 12, 'out_dim': 768
        },
        'dinov2_vitl14': {
            'img_size': 224, 'patch_size': 14, 'embed_dim': 1024,
            'depth': 24, 'num_heads': 16, 'out_dim': 1024
        },
        'dinov2_vitg14': {
            'img_size': 224, 'patch_size': 14, 'embed_dim': 1536,
            'depth': 40, 'num_heads': 24, 'out_dim': 1536
        }
    }
    
    config = configs.get(model_name, configs['dinov2_vitb14'])
    return DINOv2Encoder(config)
