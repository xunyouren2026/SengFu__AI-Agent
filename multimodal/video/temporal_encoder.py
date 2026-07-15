"""
时序编码器
实现3D卷积和时序注意力机制
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


class Conv3D:
    """3D卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重
        scale = 0.02
        kt, kh, kw = kernel_size
        self.weight = [[[[[random.gauss(0, scale) 
                          for _ in range(kw)] 
                         for _ in range(kh)] 
                        for _ in range(kt)] 
                       for _ in range(in_channels)] 
                      for _ in range(out_channels)]
        self.bias = [0.0] * out_channels
    
    def __call__(self, x: List[List[List[List[List[float]]]]]) -> List[List[List[List[List[float]]]]]:
        """
        3D卷积前向传播
        
        Args:
            x: 输入 [batch, channels, time, height, width]
        
        Returns:
            输出 [batch, out_channels, time', height', width']
        """
        batch = len(x)
        in_c = len(x[0]) if x else 0
        in_t = len(x[0][0]) if x and x[0] else 0
        in_h = len(x[0][0][0]) if x and x[0] and x[0][0] else 0
        in_w = len(x[0][0][0][0]) if x and x[0] and x[0][0] and x[0][0][0] else 0
        
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding
        
        out_t = (in_t + 2 * pt - kt) // st + 1
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        
        output = [[[[[0.0 for _ in range(out_w)] 
                     for _ in range(out_h)] 
                    for _ in range(out_t)] 
                   for _ in range(self.out_channels)] 
                  for _ in range(batch)]
        
        # 简化的3D卷积计算
        for b in range(batch):
            for oc in range(self.out_channels):
                for ot in range(out_t):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            val = self.bias[oc]
                            
                            for ic in range(in_c):
                                for kt_idx in range(kt):
                                    for kh_idx in range(kh):
                                        for kw_idx in range(kw):
                                            it = ot * st + kt_idx - pt
                                            ih = oh * sh + kh_idx - ph
                                            iw = ow * sw + kw_idx - pw
                                            
                                            if 0 <= it < in_t and 0 <= ih < in_h and 0 <= iw < in_w:
                                                val += (x[b][ic][it][ih][iw] * 
                                                       self.weight[oc][ic][kt_idx][kh_idx][kw_idx])
                            
                            output[b][oc][ot][oh][ow] = val
        
        return output


class TemporalConvBlock:
    """时序卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        self.conv = Conv3D(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(kernel_size // 2, 0, 0)
        )
        self.norm = LayerNorm(out_channels)
    
    def _relu(self, x: float) -> float:
        return max(0.0, x)
    
    def __call__(self, x: List[List[List[List[List[float]]]]]) -> List[List[List[List[List[float]]]]]:
        """前向传播"""
        # 卷积
        x = self.conv(x)
        
        # 归一化和激活
        batch = len(x)
        channels = len(x[0]) if x else 0
        time = len(x[0][0]) if x and x[0] else 0
        height = len(x[0][0][0]) if x and x[0] and x[0][0] else 0
        width = len(x[0][0][0][0]) if x and x[0] and x[0][0] and x[0][0][0] else 0
        
        for b in range(batch):
            for t in range(time):
                for h in range(height):
                    for w in range(width):
                        # 提取通道维度
                        feat = [x[b][c][t][h][w] for c in range(channels)]
                        # 归一化
                        feat = self.norm(feat)
                        # ReLU
                        feat = [self._relu(f) for f in feat]
                        # 写回
                        for c in range(channels):
                            x[b][c][t][h][w] = feat[c]
        
        return x


class TemporalAttention:
    """时序注意力"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        scale_init = 0.02
        self.q_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        self.k_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        self.v_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        
        self.dropout = dropout
    
    def __call__(self, x: List[List[float]], 
                 temporal_mask: Optional[List[List[float]]] = None) -> Tuple[List[List[float]], List[List[float]]]:
        """
        时序注意力
        
        Args:
            x: 输入特征 [time, dim]
            temporal_mask: 时序掩码
        
        Returns:
            output: 输出特征
            attn_weights: 注意力权重
        """
        t = len(x)
        
        # 投影
        q = [[sum(x[i][k] * self.q_proj[j][k] for k in range(len(x[i]))) 
              for j in range(len(self.q_proj))] for i in range(t)]
        k = [[sum(x[i][k] * self.k_proj[j][k] for k in range(len(x[i]))) 
              for j in range(len(self.k_proj))] for i in range(t)]
        v = [[sum(x[i][k] * self.v_proj[j][k] for k in range(len(x[i]))) 
              for j in range(len(self.v_proj))] for i in range(t)]
        
        # 注意力分数
        attn = [[sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale 
                 for j in range(t)] for i in range(t)]
        
        # 应用时序掩码
        if temporal_mask is not None:
            for i in range(len(attn)):
                for j in range(len(attn[i])):
                    attn[i][j] += temporal_mask[i][j]
        
        # Softmax
        attn_weights = []
        for row in attn:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            attn_weights.append([e / sum_exp for e in exp_vals])
        
        # 加权求和
        output = [[sum(attn_weights[i][j] * v[j][d] for j in range(t)) 
                   for d in range(len(v[0]))] for i in range(t)]
        
        # 输出投影
        output = [[sum(output[i][k] * self.out_proj[j][k] for k in range(len(output[i]))) 
                   for j in range(len(self.out_proj))] for i in range(t)]
        
        return output, attn_weights


class TemporalAttentionBlock:
    """时序注意力块"""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        self.norm1 = LayerNorm(dim)
        self.attn = TemporalAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        
        # MLP
        hidden_dim = int(dim * mlp_ratio)
        scale = 0.02
        self.fc1 = [[random.gauss(0, scale) for _ in range(dim)] for _ in range(hidden_dim)]
        self.fc2 = [[random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(dim)]
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def _mlp(self, x: List[List[float]]) -> List[List[float]]:
        hidden = [[self._gelu(sum(xi[k] * self.fc1[j][k] for k in range(len(xi)))) 
                   for j in range(len(self.fc1))] for xi in x]
        output = [[sum(h[k] * self.fc2[j][k] for k in range(len(h))) 
                   for j in range(len(self.fc2))] for h in hidden]
        return output
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """前向传播"""
        # 时序注意力
        normed = [[self.norm1(xi)[j] for j in range(len(xi))] for xi in x]
        attn_out, _ = self.attn(normed)
        x = [[x[i][j] + attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        # MLP
        normed = [[self.norm2(xi)[j] for j in range(len(xi))] for xi in x]
        mlp_out = self._mlp(normed)
        x = [[x[i][j] + mlp_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        return x


class TemporalEncoder:
    """时序编码器
    
    编码视频的时序信息
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.input_dim = config.get('input_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.output_dim = config.get('output_dim', 512)
        self.max_seq_len = config.get('max_seq_len', 64)
        self.use_conv = config.get('use_conv', True)
        
        scale = 0.02
        
        # 输入投影
        self.input_proj = [[random.gauss(0, scale) for _ in range(self.input_dim)] 
                           for _ in range(self.hidden_dim)]
        
        # 时序位置编码
        self.temporal_pos = [[random.gauss(0, scale) for _ in range(self.hidden_dim)] 
                             for _ in range(self.max_seq_len)]
        
        # 时序编码层
        if self.use_conv:
            self.conv_layers = [TemporalConvBlock(self.hidden_dim, self.hidden_dim) 
                               for _ in range(2)]
        
        self.attn_layers = [TemporalAttentionBlock(self.hidden_dim, self.num_heads) 
                           for _ in range(self.num_layers)]
        
        # 输出层
        self.norm = LayerNorm(self.hidden_dim)
        self.output_proj = [[random.gauss(0, scale) for _ in range(self.hidden_dim)] 
                            for _ in range(self.output_dim)]
    
    def __call__(self, features: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """
        编码时序特征
        
        Args:
            features: 帧特征 [num_frames, input_dim]
        
        Returns:
            global_feature: 全局时序特征 [output_dim]
            temporal_features: 时序特征 [num_frames, hidden_dim]
        """
        num_frames = len(features)
        
        # 输入投影
        x = [[sum(f[k] * self.input_proj[j][k] for k in range(len(f))) 
              for j in range(len(self.input_proj))] for f in features]
        
        # 添加时序位置编码
        for i in range(len(x)):
            if i < len(self.temporal_pos):
                x[i] = [x[i][j] + self.temporal_pos[i][j] for j in range(len(x[i]))]
        
        # 时序注意力层
        for layer in self.attn_layers:
            x = layer(x)
        
        # 归一化
        x = [[self.norm(xi)[j] for j in range(len(xi))] for xi in x]
        
        # 全局池化
        global_feat = [sum(x[i][j] for i in range(len(x))) / len(x) for j in range(len(x[0]))]
        
        # 输出投影
        global_feat = [sum(global_feat[k] * self.output_proj[j][k] for k in range(len(global_feat))) 
                       for j in range(self.output_dim)]
        
        return global_feat, x
    
    def encode_sequence(self, features: List[List[float]]) -> List[List[float]]:
        """
        编码序列特征（返回每帧的时序编码）
        
        Args:
            features: 帧特征 [num_frames, input_dim]
        
        Returns:
            时序编码特征 [num_frames, hidden_dim]
        """
        _, temporal_features = self(features)
        return temporal_features


class VideoEncoder:
    """视频编码器
    
    结合空间和时序编码
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.spatial_dim = config.get('spatial_dim', 768)
        self.temporal_dim = config.get('temporal_dim', 512)
        self.output_dim = config.get('output_dim', 512)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        
        # 时序编码器
        self.temporal_encoder = TemporalEncoder({
            'input_dim': self.spatial_dim,
            'hidden_dim': self.temporal_dim,
            'output_dim': self.output_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers
        })
    
    def __call__(self, frame_features: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """
        编码视频
        
        Args:
            frame_features: 各帧的空间特征 [num_frames, spatial_dim]
        
        Returns:
            global_feature: 全局视频特征 [output_dim]
            temporal_features: 时序特征 [num_frames, temporal_dim]
        """
        return self.temporal_encoder(frame_features)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'spatial_dim': self.spatial_dim,
            'temporal_dim': self.temporal_dim,
            'output_dim': self.output_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers
        }


def create_temporal_encoder(input_dim: int, output_dim: int,
                            num_layers: int = 4) -> TemporalEncoder:
    """创建时序编码器"""
    return TemporalEncoder({
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_layers': num_layers
    })


def create_video_encoder(spatial_dim: int, output_dim: int) -> VideoEncoder:
    """创建视频编码器"""
    return VideoEncoder({
        'spatial_dim': spatial_dim,
        'output_dim': output_dim
    })
