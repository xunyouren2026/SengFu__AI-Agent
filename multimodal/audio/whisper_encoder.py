"""
Whisper语音编码器
实现基于Transformer的语音识别和特征提取
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
        self.q_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        self.k_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        self.v_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        self.out_proj = [[random.gauss(0, scale_init) for _ in range(dim)] for _ in range(dim)]
        
        self.dropout = dropout
    
    def _linear(self, x: List[float], weight: List[List[float]]) -> List[float]:
        return [sum(x[i] * weight[j][i] for i in range(len(x))) for j in range(len(weight))]
    
    def __call__(self, query: List[List[float]], key: List[List[float]], 
                 value: List[List[float]], 
                 key_padding_mask: Optional[List[bool]] = None) -> List[List[float]]:
        # 投影
        q = [self._linear(qi, self.q_proj) for qi in query]
        k = [self._linear(ki, self.k_proj) for ki in key]
        v = [self._linear(vi, self.v_proj) for vi in value]
        
        # 注意力分数
        attn_scores = []
        for i in range(len(q)):
            row = []
            for j in range(len(k)):
                score = sum(q[i][d] * k[j][d] for d in range(len(q[i]))) * self.scale
                if key_padding_mask and key_padding_mask[j]:
                    score = -1e9
                row.append(score)
            attn_scores.append(row)
        
        # Softmax
        attn_weights = []
        for row in attn_scores:
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
        output = [self._linear(o, self.out_proj) for o in output]
        return output


class FeedForward:
    """前馈网络"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        scale = 0.02
        self.fc1 = [[random.gauss(0, scale) for _ in range(dim)] for _ in range(hidden_dim)]
        self.fc2 = [[random.gauss(0, scale) for _ in range(hidden_dim)] for _ in range(dim)]
        self.dropout = dropout
    
    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        hidden = []
        for xi in x:
            h = [self._gelu(sum(xi[k] * self.fc1[j][k] for k in range(len(xi)))) 
                 for j in range(len(self.fc1))]
            hidden.append(h)
        
        output = []
        for h in hidden:
            o = [sum(h[k] * self.fc2[j][k] for k in range(len(h))) 
                 for j in range(len(self.fc2))]
            output.append(o)
        
        return output


class EncoderLayer:
    """Transformer编码器层"""
    
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_dim, dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.dropout = dropout
    
    def __call__(self, x: List[List[float]], 
                 key_padding_mask: Optional[List[bool]] = None) -> List[List[float]]:
        # 自注意力
        normed = [self.norm1(xi) for xi in x]
        attn_out = self.self_attn(normed, normed, normed, key_padding_mask)
        x = [[x[i][j] + attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        # 前馈网络
        normed = [self.norm2(xi) for xi in x]
        ff_out = self.feed_forward(normed)
        x = [[x[i][j] + ff_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        return x


class DecoderLayer:
    """Transformer解码器层"""
    
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.feed_forward = FeedForward(dim, ff_dim, dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.dropout = dropout
    
    def __call__(self, x: List[List[float]], memory: List[List[float]],
                 tgt_mask: Optional[List[List[float]]] = None) -> List[List[float]]:
        # 自注意力
        normed = [self.norm1(xi) for xi in x]
        self_attn_out = self.self_attn(normed, normed, normed)
        x = [[x[i][j] + self_attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        # 交叉注意力
        normed = [self.norm2(xi) for xi in x]
        cross_attn_out = self.cross_attn(normed, memory, memory)
        x = [[x[i][j] + cross_attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        # 前馈网络
        normed = [self.norm3(xi) for xi in x]
        ff_out = self.feed_forward(normed)
        x = [[x[i][j] + ff_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        return x


class AudioEncoder:
    """音频编码器 - 将音频波形转换为特征"""
    
    def __init__(self, n_mels: int = 80, n_audio_ctx: int = 1500,
                 n_audio_state: int = 512, n_audio_head: int = 8,
                 n_audio_layer: int = 6):
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        
        # 卷积层
        scale = 0.02
        self.conv1 = [[random.gauss(0, scale) for _ in range(3 * n_mels)] 
                      for _ in range(n_audio_state)]
        self.conv2 = [[random.gauss(0, scale) for _ in range(n_audio_state)] 
                      for _ in range(n_audio_state)]
        
        # 位置嵌入
        self.positional_embedding = [[random.gauss(0, scale) for _ in range(n_audio_state)] 
                                     for _ in range(n_audio_ctx)]
        
        # Transformer块
        self.layers = [EncoderLayer(n_audio_state, n_audio_head, n_audio_state * 4) 
                       for _ in range(n_audio_layer)]
        
        self.ln_post = LayerNorm(n_audio_state)
    
    def _conv1d(self, x: List[List[float]], kernel: List[List[float]], 
                stride: int = 2, padding: int = 1) -> List[List[float]]:
        """1D卷积"""
        output = []
        k_size = len(kernel[0]) if kernel else 0
        
        for k in range(len(kernel)):
            out = []
            for i in range(0, len(x) - k_size + 2 * padding, stride):
                val = 0.0
                for j in range(k_size):
                    idx = i - padding + j
                    if 0 <= idx < len(x):
                        for c in range(len(x[idx])):
                            val += x[idx][c] * kernel[k][j * len(x[idx]) // k_size + c % (len(x[idx]) // k_size)]
                out.append(val)
            output.append(out)
        
        return output
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        """
        编码音频特征
        
        Args:
            x: Mel频谱特征 [n_mels, time]
        
        Returns:
            编码后的特征 [time', n_audio_state]
        """
        # 转置为 [time, n_mels]
        x = [[x[m][t] for m in range(len(x))] for t in range(len(x[0]))]
        
        # 卷积 + GELU
        x = self._conv1d(x, self.conv1)
        x = [[0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))) 
              for v in row] for row in x]
        
        x = self._conv1d(x, self.conv2)
        x = [[0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))) 
              for v in row] for row in x]
        
        # 添加位置嵌入
        x = [[x[i][j] + self.positional_embedding[i][j] if i < len(self.positional_embedding) else x[i][j]
              for j in range(len(x[i]))] for i in range(len(x))]
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 后归一化
        x = [self.ln_post(xi) for xi in x]
        
        return x


class TextDecoder:
    """文本解码器"""
    
    def __init__(self, n_vocab: int = 51865, n_text_ctx: int = 448,
                 n_text_state: int = 512, n_text_head: int = 8,
                 n_text_layer: int = 6):
        self.n_vocab = n_vocab
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
        
        scale = 0.02
        
        # Token嵌入
        self.token_embedding = [[random.gauss(0, scale) for _ in range(n_text_state)] 
                                for _ in range(n_vocab)]
        
        # 位置嵌入
        self.positional_embedding = [[random.gauss(0, scale) for _ in range(n_text_state)] 
                                     for _ in range(n_text_ctx)]
        
        # Transformer层
        self.layers = [DecoderLayer(n_text_state, n_text_head, n_text_state * 4) 
                       for _ in range(n_text_layer)]
        
        self.ln = LayerNorm(n_text_state)
    
    def __call__(self, tokens: List[int], memory: List[List[float]]) -> List[List[float]]:
        """
        解码文本
        
        Args:
            tokens: 输入token序列
            memory: 编码器输出
        
        Returns:
            解码后的logits
        """
        # Token嵌入 + 位置嵌入
        x = []
        for i, token in enumerate(tokens):
            if token < len(self.token_embedding):
                emb = self.token_embedding[token]
            else:
                emb = [0.0] * self.n_text_state
            
            if i < len(self.positional_embedding):
                emb = [emb[j] + self.positional_embedding[i][j] for j in range(len(emb))]
            x.append(emb)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, memory)
        
        # 归一化
        x = [self.ln(xi) for xi in x]
        
        # 输出logits (与token嵌入的点积)
        logits = []
        for xi in x:
            row = [sum(xi[j] * self.token_embedding[t][j] for j in range(len(xi))) 
                   for t in range(self.n_vocab)]
            logits.append(row)
        
        return logits


class WhisperEncoder:
    """Whisper语音编码器
    
    实现语音识别和语音特征提取功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # 模型配置
        self.n_mels = config.get('n_mels', 80)
        self.n_audio_ctx = config.get('n_audio_ctx', 1500)
        self.n_audio_state = config.get('n_audio_state', 512)
        self.n_audio_head = config.get('n_audio_head', 8)
        self.n_audio_layer = config.get('n_audio_layer', 6)
        self.n_text_ctx = config.get('n_text_ctx', 448)
        self.n_text_state = config.get('n_text_state', 512)
        self.n_text_head = config.get('n_text_head', 8)
        self.n_text_layer = config.get('n_text_layer', 6)
        self.n_vocab = config.get('n_vocab', 51865)
        
        # 音频编码器
        self.encoder = AudioEncoder(
            self.n_mels, self.n_audio_ctx,
            self.n_audio_state, self.n_audio_head, self.n_audio_layer
        )
        
        # 文本解码器
        self.decoder = TextDecoder(
            self.n_vocab, self.n_text_ctx,
            self.n_text_state, self.n_text_head, self.n_text_layer
        )
    
    def encode_audio(self, mel_features: List[List[float]]) -> List[List[float]]:
        """
        编码音频Mel频谱特征
        
        Args:
            mel_features: Mel频谱 [n_mels, time]
        
        Returns:
            编码后的音频特征
        """
        return self.encoder(mel_features)
    
    def decode_text(self, tokens: List[int], 
                    audio_features: List[List[float]]) -> List[List[float]]:
        """
        解码文本
        
        Args:
            tokens: 输入token序列
            audio_features: 音频编码特征
        
        Returns:
            输出logits
        """
        return self.decoder(tokens, audio_features)
    
    def transcribe(self, mel_features: List[List[float]], 
                   max_tokens: int = 448) -> List[int]:
        """
        语音识别转录
        
        Args:
            mel_features: Mel频谱特征
            max_tokens: 最大token数
        
        Returns:
            转录的token序列
        """
        # 编码音频
        audio_features = self.encode_audio(mel_features)
        
        # 自回归解码
        tokens = [50257]  # 开始token (sot)
        
        for _ in range(max_tokens):
            logits = self.decode_text(tokens, audio_features)
            
            # 贪心解码
            next_token = max(range(len(logits[-1])), key=lambda i: logits[-1][i])
            tokens.append(next_token)
            
            # 检查结束token
            if next_token == 50257:  # eot
                break
        
        return tokens
    
    def get_audio_features(self, mel_features: List[List[float]]) -> List[List[float]]:
        """
        提取音频特征（用于其他任务）
        
        Args:
            mel_features: Mel频谱特征
        
        Returns:
            音频嵌入特征
        """
        return self.encode_audio(mel_features)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'n_mels': self.n_mels,
            'n_audio_ctx': self.n_audio_ctx,
            'n_audio_state': self.n_audio_state,
            'n_audio_head': self.n_audio_head,
            'n_audio_layer': self.n_audio_layer,
            'n_text_ctx': self.n_text_ctx,
            'n_text_state': self.n_text_state,
            'n_text_head': self.n_text_head,
            'n_text_layer': self.n_text_layer,
            'n_vocab': self.n_vocab
        }


class MelSpectrogram:
    """Mel频谱特征提取器"""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400,
                 hop_length: int = 160, n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel滤波器组
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
            
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            for j in range(left, center):
                if j < len(filter_bank) and center > left:
                    filter_bank[j] = (j - left) / (center - left)
            
            for j in range(center, right):
                if j < len(filter_bank) and right > center:
                    filter_bank[j] = (right - j) / (right - center)
            
            filters.append(filter_bank)
        
        return filters
    
    def _stft(self, audio: List[float]) -> List[List[complex]]:
        """短时傅里叶变换"""
        n_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        stft_matrix = []
        
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]
            
            # 添加汉宁窗
            windowed = [frame[j] * 0.5 * (1 - math.cos(2 * math.pi * j / (self.n_fft - 1)))
                       for j in range(len(frame))]
            
            # DFT
            spectrum = []
            for k in range(self.n_fft // 2 + 1):
                real = sum(windowed[n] * math.cos(-2 * math.pi * k * n / self.n_fft) 
                          for n in range(len(windowed)))
                imag = sum(windowed[n] * math.sin(-2 * math.pi * k * n / self.n_fft) 
                          for n in range(len(windowed)))
                spectrum.append(complex(real, imag))
            
            stft_matrix.append(spectrum)
        
        return stft_matrix
    
    def __call__(self, audio: List[float]) -> List[List[float]]:
        """
        提取Mel频谱特征
        
        Args:
            audio: 音频波形数据
        
        Returns:
            Mel频谱 [n_mels, time]
        """
        # STFT
        stft = self._stft(audio)
        
        # 功率谱
        power = [[abs(s) ** 2 for s in frame] for frame in stft]
        
        # 应用Mel滤波器
        mel_spec = []
        for mel_filter in self.mel_filters:
            mel_row = []
            for frame in power:
                val = sum(frame[k] * mel_filter[k] for k in range(len(mel_filter)))
                mel_row.append(math.log(max(val, 1e-10)))
            mel_spec.append(mel_row)
        
        return mel_spec


def create_whisper_encoder(model_name: str = 'base') -> WhisperEncoder:
    """
    创建预配置的Whisper编码器
    
    Args:
        model_name: 模型名称
    
    Returns:
        Whisper编码器
    """
    configs = {
        'tiny': {
            'n_mels': 80, 'n_audio_ctx': 1500, 'n_audio_state': 384,
            'n_audio_head': 6, 'n_audio_layer': 4,
            'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4
        },
        'base': {
            'n_mels': 80, 'n_audio_ctx': 1500, 'n_audio_state': 512,
            'n_audio_head': 8, 'n_audio_layer': 6,
            'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6
        },
        'small': {
            'n_mels': 80, 'n_audio_ctx': 1500, 'n_audio_state': 768,
            'n_audio_head': 12, 'n_audio_layer': 12,
            'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12
        },
        'medium': {
            'n_mels': 80, 'n_audio_ctx': 1500, 'n_audio_state': 1024,
            'n_audio_head': 16, 'n_audio_layer': 24,
            'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24
        },
        'large': {
            'n_mels': 128, 'n_audio_ctx': 1500, 'n_audio_state': 1280,
            'n_audio_head': 20, 'n_audio_layer': 32,
            'n_text_state': 1280, 'n_text_head': 20, 'n_text_layer': 32
        }
    }
    
    config = configs.get(model_name, configs['base'])
    return WhisperEncoder(config)
