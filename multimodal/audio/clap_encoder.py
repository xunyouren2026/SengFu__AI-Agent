"""
CLAP音频-文本对齐编码器
实现音频与文本的对比学习对齐
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


class MultiHeadAttention:
    """多头注意力"""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        scale_init = 0.02
        self.qkv_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(3 * dim)]
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        self.dropout = dropout
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        n = len(x)
        c = len(x[0]) if x else 0
        
        # QKV投影
        qkv = [[sum(x[i][k] * self.qkv_proj[j][k] for k in range(c)) 
                for j in range(3 * c)] for i in range(n)]
        
        # 分离Q, K, V
        q = [[qkv[i][j] for j in range(c)] for i in range(n)]
        k = [[qkv[i][j + c] for j in range(c)] for i in range(n)]
        v = [[qkv[i][j + 2 * c] for j in range(c)] for i in range(n)]
        
        # 注意力分数
        attn = [[sum(q[i][d] * k[j][d] for d in range(c)) * self.scale for j in range(n)] for i in range(n)]
        
        # Softmax
        attn_weights = []
        for row in attn:
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            attn_weights.append([e / sum_exp for e in exp_vals])
        
        # 加权求和
        output = [[sum(attn_weights[i][j] * v[j][d] for j in range(n)) for d in range(c)] for i in range(n)]
        
        # 输出投影
        output = [[sum(output[i][k] * self.out_proj[j][k] for k in range(c)) for j in range(c)] for i in range(n)]
        
        return output


class TransformerBlock:
    """Transformer块"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = LayerNorm(dim)
        
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
        # 注意力
        normed = [self.norm1(xi) for xi in x]
        attn_out = self.attn(normed)
        x = [[x[i][j] + attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        # MLP
        normed = [self.norm2(xi) for xi in x]
        mlp_out = self._mlp(normed)
        x = [[x[i][j] + mlp_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        return x


class AudioFeatureExtractor:
    """音频特征提取器"""
    
    def __init__(self, sample_rate: int = 48000, n_fft: int = 1024,
                 hop_length: int = 480, n_mels: int = 64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel滤波器
        self.mel_filters = self._create_mel_filters()
    
    def _create_mel_filters(self) -> List[List[float]]:
        """创建Mel滤波器组"""
        mel_min = 0.0
        mel_max = 2595.0 * math.log10(1.0 + self.sample_rate / 2.0 / 700.0)
        
        mel_points = [mel_min + i * (mel_max - mel_min) / (self.n_mels + 1) 
                      for i in range(self.n_mels + 2)]
        hz_points = [700.0 * (10.0 ** (m / 2595.0) - 1.0) for m in mel_points]
        bin_points = [int(hz * self.n_fft / self.sample_rate) for hz in hz_points]
        
        filters = []
        for i in range(self.n_mels):
            filter_bank = [0.0] * (self.n_fft // 2 + 1)
            left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            
            for j in range(left, center):
                if j < len(filter_bank) and center > left:
                    filter_bank[j] = (j - left) / (center - left)
            for j in range(center, right):
                if j < len(filter_bank) and right > center:
                    filter_bank[j] = (right - j) / (right - center)
            
            filters.append(filter_bank)
        
        return filters
    
    def __call__(self, audio: List[float]) -> List[List[float]]:
        """提取Mel频谱"""
        n_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        
        # STFT
        stft = []
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]
            windowed = [frame[j] * 0.5 * (1 - math.cos(2 * math.pi * j / (self.n_fft - 1)))
                       for j in range(len(frame))]
            
            spectrum = []
            for k in range(self.n_fft // 2 + 1):
                real = sum(windowed[n] * math.cos(-2 * math.pi * k * n / self.n_fft) 
                          for n in range(len(windowed)))
                imag = sum(windowed[n] * math.sin(-2 * math.pi * k * n / self.n_fft) 
                          for n in range(len(windowed)))
                spectrum.append(complex(real, imag))
            stft.append(spectrum)
        
        # 功率谱
        power = [[abs(s) ** 2 for s in frame] for frame in stft]
        
        # Mel滤波
        mel_spec = []
        for mel_filter in self.mel_filters:
            mel_row = []
            for frame in power:
                val = sum(frame[k] * mel_filter[k] for k in range(len(mel_filter)))
                mel_row.append(math.log(max(val, 1e-10)))
            mel_spec.append(mel_row)
        
        return mel_spec


class CLAPAudioEncoder:
    """CLAP音频编码器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.n_mels = config.get('n_mels', 64)
        self.embed_dim = config.get('embed_dim', 512)
        self.depth = config.get('depth', 4)
        self.num_heads = config.get('num_heads', 8)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.out_dim = config.get('out_dim', 512)
        
        # 特征提取
        self.feature_extractor = AudioFeatureExtractor(n_mels=self.n_mels)
        
        # Patch嵌入
        scale = 0.02
        self.patch_embed = [[random.gauss(0, scale) for _ in range(self.n_mels)] 
                            for _ in range(self.embed_dim)]
        self.patch_bias = [0.0] * self.embed_dim
        
        # 位置嵌入
        self.pos_embed = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                          for _ in range(1000)]  # 最大长度
        
        # Transformer块
        self.blocks = [TransformerBlock(self.embed_dim, self.num_heads, self.mlp_ratio) 
                       for _ in range(self.depth)]
        
        # 最终层
        self.norm = LayerNorm(self.embed_dim)
        self.proj = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                     for _ in range(self.out_dim)]
    
    def __call__(self, audio: List[float]) -> List[float]:
        """
        编码音频为特征向量
        
        Args:
            audio: 音频波形
        
        Returns:
            音频特征向量
        """
        # 提取Mel频谱
        mel_spec = self.feature_extractor(audio)  # [n_mels, time]
        
        # 转置为 [time, n_mels]
        mel_spec = [[mel_spec[m][t] for m in range(len(mel_spec))] 
                    for t in range(len(mel_spec[0]))]
        
        # Patch嵌入
        x = [[sum(mel_spec[i][k] * self.patch_embed[j][k] for k in range(len(mel_spec[i]))) + self.patch_bias[j]
              for j in range(self.embed_dim)] for i in range(len(mel_spec))]
        
        # 添加位置嵌入
        x = [[x[i][j] + self.pos_embed[i][j] if i < len(self.pos_embed) else x[i][j]
              for j in range(len(x[i]))] for i in range(len(x))]
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 全局平均池化
        pooled = [sum(x[i][j] for i in range(len(x))) / len(x) for j in range(len(x[0]))]
        
        # 归一化和投影
        pooled = self.norm(pooled)
        output = [sum(pooled[k] * self.proj[j][k] for k in range(len(pooled))) 
                  for j in range(self.out_dim)]
        
        return output


class CLAPTextEncoder:
    """CLAP文本编码器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.vocab_size = config.get('vocab_size', 30522)
        self.embed_dim = config.get('embed_dim', 512)
        self.depth = config.get('depth', 4)
        self.num_heads = config.get('num_heads', 8)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.max_length = config.get('max_length', 77)
        self.out_dim = config.get('out_dim', 512)
        
        scale = 0.02
        
        # Token嵌入
        self.token_embed = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                            for _ in range(self.vocab_size)]
        
        # 位置嵌入
        self.pos_embed = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                          for _ in range(self.max_length)]
        
        # Transformer块
        self.blocks = [TransformerBlock(self.embed_dim, self.num_heads, self.mlp_ratio) 
                       for _ in range(self.depth)]
        
        # 最终层
        self.norm = LayerNorm(self.embed_dim)
        self.proj = [[random.gauss(0, scale) for _ in range(self.embed_dim)] 
                     for _ in range(self.out_dim)]
    
    def __call__(self, tokens: List[int]) -> List[float]:
        """
        编码文本为特征向量
        
        Args:
            tokens: token序列
        
        Returns:
            文本特征向量
        """
        # Token嵌入
        x = []
        for i, token in enumerate(tokens):
            if token < len(self.token_embed):
                emb = self.token_embed[token].copy()
            else:
                emb = [0.0] * self.embed_dim
            
            # 添加位置嵌入
            if i < len(self.pos_embed):
                emb = [emb[j] + self.pos_embed[i][j] for j in range(len(emb))]
            x.append(emb)
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 使用第一个token (类似CLS)
        pooled = x[0] if x else [0.0] * self.embed_dim
        
        # 归一化和投影
        pooled = self.norm(pooled)
        output = [sum(pooled[k] * self.proj[j][k] for k in range(len(pooled))) 
                  for j in range(self.out_dim)]
        
        return output


class CLAPEncoder:
    """CLAP音频-文本对齐编码器
    
    实现音频与文本的对比学习对齐
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.embed_dim = config.get('embed_dim', 512)
        self.out_dim = config.get('out_dim', 512)
        
        # 音频编码器
        self.audio_encoder = CLAPAudioEncoder({
            'embed_dim': self.embed_dim,
            'out_dim': self.out_dim,
            **config.get('audio_config', {})
        })
        
        # 文本编码器
        self.text_encoder = CLAPTextEncoder({
            'embed_dim': self.embed_dim,
            'out_dim': self.out_dim,
            **config.get('text_config', {})
        })
        
        # 温度参数
        self.logit_scale = 0.07  # 可学习参数
    
    def encode_audio(self, audio: List[float]) -> List[float]:
        """编码音频"""
        return self.audio_encoder(audio)
    
    def encode_text(self, tokens: List[int]) -> List[float]:
        """编码文本"""
        return self.text_encoder(tokens)
    
    def get_similarity(self, audio_features: List[float], 
                       text_features: List[float]) -> float:
        """
        计算音频-文本相似度
        
        Args:
            audio_features: 音频特征
            text_features: 文本特征
        
        Returns:
            相似度分数
        """
        # L2归一化
        audio_norm = math.sqrt(sum(x ** 2 for x in audio_features))
        text_norm = math.sqrt(sum(x ** 2 for x in text_features))
        
        if audio_norm > 0 and text_norm > 0:
            audio_normalized = [x / audio_norm for x in audio_features]
            text_normalized = [x / text_norm for x in text_features]
        else:
            return 0.0
        
        # 余弦相似度
        similarity = sum(a * t for a, t in zip(audio_normalized, text_normalized))
        
        # 缩放
        return similarity * math.exp(self.logit_scale)
    
    def get_similarity_matrix(self, audio_features: List[List[float]], 
                              text_features: List[List[float]]) -> List[List[float]]:
        """
        计算音频-文本相似度矩阵
        
        Args:
            audio_features: 多个音频特征
            text_features: 多个文本特征
        
        Returns:
            相似度矩阵
        """
        # L2归一化
        audio_norms = [math.sqrt(sum(x ** 2 for x in af)) for af in audio_features]
        text_norms = [math.sqrt(sum(x ** 2 for x in tf)) for tf in text_features]
        
        audio_normalized = [[af[j] / an if an > 0 else 0.0 for j in range(len(af))] 
                           for af, an in zip(audio_features, audio_norms)]
        text_normalized = [[tf[j] / tn if tn > 0 else 0.0 for j in range(len(tf))] 
                          for tf, tn in zip(text_features, text_norms)]
        
        # 计算相似度矩阵
        scale = math.exp(self.logit_scale)
        sim_matrix = []
        for af in audio_normalized:
            row = []
            for tf in text_normalized:
                sim = sum(a * t for a, t in zip(af, tf)) * scale
                row.append(sim)
            sim_matrix.append(row)
        
        return sim_matrix
    
    def retrieve_audio(self, query_text: List[int], 
                       audio_features: List[List[float]], 
                       top_k: int = 5) -> List[Tuple[int, float]]:
        """
        根据文本检索音频
        
        Args:
            query_text: 查询文本tokens
            audio_features: 音频特征库
            top_k: 返回top-k结果
        
        Returns:
            (索引, 相似度)列表
        """
        text_feature = self.encode_text(query_text)
        
        similarities = []
        for i, af in enumerate(audio_features):
            sim = self.get_similarity(af, text_feature)
            similarities.append((i, sim))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def retrieve_text(self, query_audio: List[float], 
                      text_features: List[List[float]], 
                      top_k: int = 5) -> List[Tuple[int, float]]:
        """
        根据音频检索文本
        
        Args:
            query_audio: 查询音频
            text_features: 文本特征库
            top_k: 返回top-k结果
        
        Returns:
            (索引, 相似度)列表
        """
        audio_feature = self.encode_audio(query_audio)
        
        similarities = []
        for i, tf in enumerate(text_features):
            sim = self.get_similarity(audio_feature, tf)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'embed_dim': self.embed_dim,
            'out_dim': self.out_dim,
            'logit_scale': self.logit_scale
        }


def create_clap_encoder(model_name: str = 'clap_base') -> CLAPEncoder:
    """
    创建预配置的CLAP编码器
    
    Args:
        model_name: 模型名称
    
    Returns:
        CLAP编码器
    """
    configs = {
        'clap_base': {
            'embed_dim': 512, 'out_dim': 512,
            'audio_config': {'depth': 4, 'num_heads': 8},
            'text_config': {'depth': 4, 'num_heads': 8}
        },
        'clap_large': {
            'embed_dim': 768, 'out_dim': 768,
            'audio_config': {'depth': 6, 'num_heads': 12},
            'text_config': {'depth': 6, 'num_heads': 12}
        }
    }
    
    config = configs.get(model_name, configs['clap_base'])
    return CLAPEncoder(config)
