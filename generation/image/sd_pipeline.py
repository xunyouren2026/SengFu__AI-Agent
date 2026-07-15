"""
Stable Diffusion Pipeline - Pure Python Implementation
=======================================================
纯Python实现的Stable Diffusion管线，包含文本编码、扩散过程、UNet、VAE等核心组件。
仅使用标准库，不依赖外部库。
"""

import math
import random
import hashlib
from typing import List, Tuple, Optional, Any, Dict


class GenerationResult:
    """生成结果封装类"""

    def __init__(self, images: List, prompt: str = "", negative_prompt: str = "",
                 seed: Optional[int] = None, steps: int = 50, guidance_scale: float = 7.5,
                 width: int = 512, height: int = 512):
        self.images = images
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "num_images": len(self.images),
        }


# ===========================================================================
# 工具函数
# ===========================================================================

def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法 a @ b"""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    assert cols_a == rows_b, f"矩阵维度不匹配: {cols_a} vs {rows_b}"
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _transpose(a: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    if not a:
        return []
    rows, cols = len(a), len(a[0])
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def _softmax(x: List[float]) -> List[float]:
    """Softmax函数"""
    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


def _gelu(x: float) -> float:
    """GELU激活函数"""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _sigmoid(x: float) -> float:
    """Sigmoid激活函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _relu(x: float) -> float:
    """ReLU激活函数"""
    return max(0.0, x)


def _tanh(x: float) -> float:
    """Tanh激活函数"""
    return math.tanh(x)


def _layer_norm_1d(x: List[float], eps: float = 1e-5) -> List[float]:
    """一维层归一化"""
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


def _randn(shape: List[int], seed: Optional[int] = None) -> List:
    """生成标准正态分布随机数（Box-Muller变换）"""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    def _gen():
        u1 = rng.random()
        u2 = rng.random()
        while u1 == 0:
            u1 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    if len(shape) == 1:
        return [_gen() for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[_gen() for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[_gen() for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[ _gen() for _ in range(shape[3])]
                   for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def _rand(shape: List[int], seed: Optional[int] = None) -> List:
    """生成[0,1)均匀分布随机数"""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    def _gen():
        return rng.random()

    if len(shape) == 1:
        return [_gen() for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[_gen() for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[_gen() for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[ _gen() for _ in range(shape[3])]
                   for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def _zeros(shape: List[int]) -> List:
    """生成全零张量"""
    if len(shape) == 1:
        return [0.0] * shape[0]
    elif len(shape) == 2:
        return [[0.0] * shape[1] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[0.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[0.0] * shape[3] for _ in range(shape[2])]
                  for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def _ones(shape: List[int]) -> List:
    """生成全一张量"""
    if len(shape) == 1:
        return [1.0] * shape[0]
    elif len(shape) == 2:
        return [[1.0] * shape[1] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[1.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[1.0] * shape[3] for _ in range(shape[2])]
                  for _ in range(shape[1])] for _ in range(shape[0])]
    else:
        raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def _tensor_add(a: List, b: List) -> List:
    """张量逐元素加法"""
    if isinstance(a[0], list):
        return [_tensor_add(ai, bi) for ai, bi in zip(a, b)]
    return [ai + bi for ai, bi in zip(a, b)]


def _tensor_mul(a: List, b: List) -> List:
    """张量逐元素乘法"""
    if isinstance(a[0], list):
        return [_tensor_mul(ai, bi) for ai, bi in zip(a, b)]
    return [ai * bi for ai, bi in zip(a, b)]


def _tensor_scale(a: List, s: float) -> List:
    """张量标量乘法"""
    if isinstance(a[0], list):
        return [_tensor_scale(ai, s) for ai in a]
    return [v * s for v in a]


def _tensor_shape(x: List) -> List[int]:
    """获取张量形状"""
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        if len(x) > 0:
            x = x[0]
        else:
            break
    return shape


def _linear(x: List[float], weight: List[List[float]], bias: List[float]) -> List[float]:
    """线性变换 y = xW + b, 其中weight形状为[in_features, out_features]"""
    result = _matmul([x], weight)[0]
    return [r + bias[i] for i, r in enumerate(result)]


# ===========================================================================
# 1. CLIPTextEncoder - CLIP文本编码器
# ===========================================================================

class CLIPTextEncoder:
    """
    CLIP文本编码器的纯Python实现。
    使用简化的Transformer架构对输入文本进行编码。
    """

    _vocab_size: int = 49408
    _embed_dim: int = 768
    _max_length: int = 77
    _num_heads: int = 12
    _num_layers: int = 12
    _ff_dim: int = 3072

    def __init__(self, seed: Optional[int] = 42):
        self._seed = seed
        self._vocab = self._build_vocab()
        self._positional_embeddings = self._build_positional_embeddings()
        # 延迟初始化：避免在构造函数中分配大矩阵
        self._token_embeddings = None
        self._layers = None
        self._final_ln_w = [1.0] * self._embed_dim
        self._final_ln_b = [0.0] * self._embed_dim
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """延迟初始化权重矩阵"""
        if self._initialized:
            return
        rng = random.Random(self._seed)
        scale = 0.02
        # 词嵌入矩阵 [vocab_size, embed_dim] - 使用紧凑存储
        # 仅预分配常用token的嵌入，其余按需生成
        self._token_embeddings = {}
        for token_id in self._vocab.values():
            self._token_embeddings[token_id] = [rng.gauss(0, scale) for _ in range(self._embed_dim)]
        # Transformer层权重（简化为1层以保持性能）
        self._layers = []
        for _ in range(1):  # 简化为1层
            layer = {
                "attn_q": [[rng.gauss(0, scale) for _ in range(self._embed_dim)] for _ in range(self._embed_dim)],
                "attn_k": [[rng.gauss(0, scale) for _ in range(self._embed_dim)] for _ in range(self._embed_dim)],
                "attn_v": [[rng.gauss(0, scale) for _ in range(self._embed_dim)] for _ in range(self._embed_dim)],
                "attn_out": [[rng.gauss(0, scale) for _ in range(self._embed_dim)] for _ in range(self._embed_dim)],
                "ff1": [[rng.gauss(0, scale) for _ in range(self._ff_dim)] for _ in range(self._embed_dim)],
                "ff2": [[rng.gauss(0, scale) for _ in range(self._embed_dim)] for _ in range(self._ff_dim)],
                "ln1_w": [1.0] * self._embed_dim,
                "ln1_b": [0.0] * self._embed_dim,
                "ln2_w": [1.0] * self._embed_dim,
                "ln2_b": [0.0] * self._embed_dim,
            }
            self._layers.append(layer)
        self._initialized = True

    def _build_vocab(self) -> Dict[str, int]:
        """构建简易BPE词表"""
        vocab = {}
        # 基础ASCII字符
        for i in range(256):
            vocab[chr(i)] = i + 3
        # 常见子词
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "because", "but", "and", "or", "if", "while", "that", "this", "these",
            "those", "it", "its", "he", "she", "they", "them", "his", "her",
            "their", "my", "your", "our", "what", "which", "who", "whom",
            "man", "woman", "person", "people", "child", "dog", "cat", "bird",
            "tree", "flower", "sky", "sun", "moon", "star", "water", "fire",
            "mountain", "ocean", "river", "forest", "city", "house", "building",
            "car", "road", "street", "park", "garden", "field", "cloud", "rain",
            "snow", "wind", "light", "dark", "red", "blue", "green", "yellow",
            "white", "black", "brown", "orange", "purple", "pink", "gray",
            "beautiful", "ugly", "big", "small", "tall", "short", "long", "new",
            "old", "young", "happy", "sad", "good", "bad", "hot", "cold",
            "fast", "slow", "high", "low", "deep", "wide", "narrow", "thick",
            "thin", "heavy", "light_weight", "strong", "weak", "soft", "hard",
            "smooth", "rough", "sharp", "dull", "bright", "dim", "clean", "dirty",
            "wet", "dry", "full", "empty", "open", "closed", "alive", "dead",
            "real", "fake", "natural", "artificial", "simple", "complex",
            "photo", "painting", "drawing", "sketch", "illustration", "art",
            "style", "quality", "detail", "realistic", "abstract", "modern",
            "vintage", "retro", "futuristic", "minimalist", "elaborate",
            "professional", "amateur", "studio", "outdoor", "indoor",
            "portrait", "landscape", "closeup", "wide_angle", "aerial",
            "dramatic", "serene", "vibrant", "muted", "colorful", "monochrome",
            "high_resolution", "4k", "8k", "hdr", "bokeh", "depth_of_field",
            "cinematic", "photorealistic", "hyperrealistic", "surreal",
            "fantasy", "sci-fi", "cyberpunk", "steampunk", "horror", "romantic",
            "epic", "majestic", "elegant", "graceful", "powerful", "gentle",
            "mysterious", "magical", "ethereal", "dreamy", "nostalgic",
            "peaceful", "chaotic", "dynamic", "static", "organic", "geometric",
            "textured", "smooth_surface", "reflective", "transparent", "opaque",
            "glowing", "shadow", "silhouette", "backlit", "frontlit", "ambient",
        ]
        for idx, word in enumerate(common_words):
            vocab[word] = 256 + idx
        # 特殊token
        vocab["<pad>"] = 0
        vocab["<bos>"] = 1
        vocab["<eos>"] = 2
        vocab["<unk>"] = 3
        return vocab

    def _build_positional_embeddings(self) -> List[List[float]]:
        """构建正弦位置编码"""
        pe = []
        for pos in range(self._max_length):
            row = []
            for i in range(self._embed_dim):
                if i % 2 == 0:
                    row.append(math.sin(pos / (10000 ** (i / self._embed_dim))))
                else:
                    row.append(math.cos(pos / (10000 ** ((i - 1) / self._embed_dim))))
            pe.append(row)
        return pe

    def tokenize(self, text: str) -> List[int]:
        """
        简易BPE分词。
        将输入文本分割为子词单元并映射为词表索引。
        """
        text = text.lower().strip()
        tokens = [1]  # <bos>
        # 按空格和标点分割
        words = []
        current_word = ""
        for ch in text:
            if ch in " ,.!?;:\"'()-\n\t":
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(ch)
            else:
                current_word += ch
        if current_word:
            words.append(current_word)

        for word in words:
            if word in self._vocab:
                tokens.append(self._vocab[word])
            else:
                # 逐字符回退
                for ch in word:
                    if ch in self._vocab:
                        tokens.append(self._vocab[ch])
                    else:
                        tokens.append(self._vocab.get("<unk>", 3))
            if len(tokens) >= self._max_length - 1:
                break

        # 填充到max_length
        while len(tokens) < self._max_length:
            tokens.append(0)  # <pad>
        tokens = tokens[:self._max_length]
        return tokens

    def _build_attention_mask(self, tokens: List[int]) -> List[List[float]]:
        """构建注意力掩码，padding位置为0"""
        seq_len = len(tokens)
        mask = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                if tokens[j] == 0:  # padding
                    row.append(0.0)
                else:
                    row.append(1.0)
            mask.append(row)
        return mask

    def _self_attention(self, embeddings: List[List[float]],
                        mask: List[List[float]]) -> List[List[float]]:
        """
        多头自注意力机制。
        Q, K, V = embeddings * W_q, W_k, W_v
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        """
        seq_len = len(embeddings)
        dim = len(embeddings[0])
        head_dim = dim // self._num_heads
        scale = math.sqrt(head_dim)

        # 简化：使用单头注意力（完整多头太慢）
        # 计算Q, K, V
        layer = self._layers[0]
        queries = [_linear(emb, layer["attn_q"], [0.0] * dim) for emb in embeddings]
        keys = [_linear(emb, layer["attn_k"], [0.0] * dim) for emb in embeddings]
        values = [_linear(emb, layer["attn_v"], [0.0] * dim) for emb in embeddings]

        # 计算注意力分数 QK^T / sqrt(d_k)
        attn_scores = _matmul(queries, _transpose(keys))
        attn_scores = [[s / scale for s in row] for row in attn_scores]

        # 应用掩码
        for i in range(seq_len):
            for j in range(seq_len):
                if mask[i][j] == 0:
                    attn_scores[i][j] = -1e9

        # Softmax
        attn_weights = [_softmax(row) for row in attn_scores]

        # 加权求和
        output = _matmul(attn_weights, values)

        # 输出投影
        output = [_linear(emb, layer["attn_out"], [0.0] * dim) for emb in output]
        return output

    def _cross_attention(self, query: List[List[float]],
                         context: List[List[float]],
                         mask: List[List[float]]) -> List[List[float]]:
        """
        交叉注意力机制。
        query来自一个序列，context来自另一个序列。
        """
        seq_len_q = len(query)
        seq_len_k = len(context)
        dim = len(query[0])
        scale = math.sqrt(dim)

        # 简化投影
        rng = random.Random(self._seed + 100)
        w_q = [[rng.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        w_k = [[rng.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        w_v = [[rng.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]

        queries = [_linear(q, w_q, [0.0] * dim) for q in query]
        keys = [_linear(c, w_k, [0.0] * dim) for c in context]
        values = [_linear(c, w_v, [0.0] * dim) for c in context]

        # QK^T
        attn_scores = _matmul(queries, _transpose(keys))
        attn_scores = [[s / scale for s in row] for row in attn_scores]

        # 应用掩码
        for i in range(seq_len_q):
            for j in range(seq_len_k):
                if mask[i][j] == 0:
                    attn_scores[i][j] = -1e9

        attn_weights = [_softmax(row) for row in attn_scores]
        output = _matmul(attn_weights, values)
        return output

    def _feed_forward(self, x: List[List[float]]) -> List[List[float]]:
        """
        前馈网络: FFN(x) = GELU(xW1 + b1)W2 + b2
        """
        layer = self._layers[0]
        dim = len(x[0])
        bias1 = [0.0] * self._ff_dim
        bias2 = [0.0] * dim

        result = []
        for row in x:
            hidden = _linear(row, layer["ff1"], bias1)
            hidden = [_gelu(h) for h in hidden]
            out = _linear(hidden, layer["ff2"], bias2)
            result.append(out)
        return result

    def _layer_norm(self, x: List[List[float]], w: List[float], b: List[float]) -> List[List[float]]:
        """层归一化"""
        eps = 1e-5
        result = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            std = math.sqrt(var + eps)
            normalized = [(row[i] - mean) / std * w[i] + b[i] for i in range(len(row))]
            result.append(normalized)
        return result

    def _transformer_block(self, x: List[List[float]], context: Optional[List[List[float]]],
                           mask: List[List[float]]) -> List[List[float]]:
        """
        单个Transformer块:
        x = LayerNorm(x + SelfAttention(x))
        x = LayerNorm(x + CrossAttention(x, context))
        x = LayerNorm(x + FFN(x))
        """
        # 简化：只使用第一个layer的权重
        layer = self._layers[0]

        # 自注意力 + 残差连接 + 层归一化
        attn_out = self._self_attention(x, mask)
        x = [_tensor_add(xi, ai) for xi, ai in zip(x, attn_out)]
        x = self._layer_norm(x, layer["ln1_w"], layer["ln1_b"])

        # 交叉注意力（如果有context）
        if context is not None:
            cross_out = self._cross_attention(x, context, mask)
            x = [_tensor_add(xi, ci) for xi, ci in zip(x, cross_out)]
            x = self._layer_norm(x, layer["ln1_w"], layer["ln1_b"])

        # 前馈网络 + 残差连接 + 层归一化
        ff_out = self._feed_forward(x)
        x = [_tensor_add(xi, fi) for xi, fi in zip(x, ff_out)]
        x = self._layer_norm(x, layer["ln2_w"], layer["ln2_b"])

        return x

    def encode(self, text: str) -> List[List[float]]:
        """
        文本编码主方法。
        将输入文本编码为固定长度的特征向量序列。
        """
        self._ensure_initialized()
        tokens = self.tokenize(text)
        mask = self._build_attention_mask(tokens)

        # 词嵌入 + 位置编码
        embeddings = []
        for i, token_id in enumerate(tokens):
            tok_emb = self._token_embeddings.get(token_id)
            if tok_emb is None:
                # 按需生成未知token的嵌入
                rng = random.Random(self._seed + token_id)
                tok_emb = [rng.gauss(0, 0.02) for _ in range(self._embed_dim)]
                self._token_embeddings[token_id] = tok_emb
            pos_emb = self._positional_embeddings[i]
            combined = [t + p for t, p in zip(tok_emb, pos_emb)]
            embeddings.append(combined)

        # 通过Transformer层（简化为1层以保持性能）
        embeddings = self._transformer_block(embeddings, None, mask)

        # 最终层归一化
        embeddings = self._layer_norm(embeddings, self._final_ln_w, self._final_ln_b)

        return embeddings


# ===========================================================================
# 2. GaussianDiffusion - 高斯扩散过程
# ===========================================================================

class GaussianDiffusion:
    """
    高斯扩散过程的纯Python实现。
    实现前向扩散 q(x_t|x_0) 和反向扩散 p(x_{t-1}|x_t)。
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "linear"):
        self._num_timesteps = num_timesteps
        self._schedule_type = schedule
        self._betas = []
        self._alphas = []
        self._alpha_cumprod = []
        self._sqrt_alphas_cumprod = []
        self._sqrt_one_minus_alphas_cumprod = []
        self._compute_schedule(schedule)

    def _compute_schedule(self, schedule: str) -> None:
        """
        计算噪声调度系数。
        支持 linear（线性）和 cosine（余弦）两种调度策略。
        """
        if schedule == "linear":
            # 线性调度: beta从0.00085线性增长到0.012
            beta_start = 0.00085
            beta_end = 0.012
            self._betas = [
                beta_start + (beta_end - beta_start) * i / (self._num_timesteps - 1)
                for i in range(self._num_timesteps)
            ]
        elif schedule == "cosine":
            # 余弦调度
            self._betas = []
            steps = self._num_timesteps + 1
            s = 0.008  # 偏移量
            for i in range(self._num_timesteps):
                f = math.cos(((i + 1) / steps + s) / (1 + s) * math.pi / 2) ** 2
                f_prev = math.cos((i / steps + s) / (1 + s) * math.pi / 2) ** 2
                beta = min(1.0 - f / f_prev, 0.999)
                self._betas.append(beta)
        else:
            raise ValueError(f"未知的调度类型: {schedule}")

        # 计算alpha相关系数
        self._alphas = [1.0 - b for b in self._betas]
        self._alpha_cumprod = []
        cumprod = 1.0
        for a in self._alphas:
            cumprod *= a
            self._alpha_cumprod.append(cumprod)
        self._sqrt_alphas_cumprod = [math.sqrt(a) for a in self._alpha_cumprod]
        self._sqrt_one_minus_alphas_cumprod = [math.sqrt(1.0 - a) for a in self._alpha_cumprod]

    def _extract(self, a: List[float], t: int, x_shape: List[int]) -> List[float]:
        """
        从系数列表中提取指定时间步t的系数。
        返回与x_shape兼容的广播系数。
        """
        return [a[t]] * (x_shape[0] * x_shape[1] if len(x_shape) >= 2 else x_shape[0])

    def forward_diffusion(self, x_0: List[List[float]], t: int,
                          noise: Optional[List[List[float]]] = None) -> Tuple[List[List[float]], List[List[float]]]:
        """
        前向扩散过程: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: 原始数据 [batch, channels*height*width]
            t: 时间步
            noise: 可选的预生成噪声

        Returns:
            (x_t, noise): 加噪后的数据和使用的噪声
        """
        if noise is None:
            shape = _tensor_shape(x_0)
            noise = _randn(shape)

        sqrt_alpha = self._sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self._sqrt_one_minus_alphas_cumprod[t]

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = []
        for i in range(len(x_0)):
            row = []
            for j in range(len(x_0[i])):
                val = sqrt_alpha * x_0[i][j] + sqrt_one_minus_alpha * noise[i][j]
                row.append(val)
            x_t.append(row)

        return x_t, noise

    def reverse_diffusion(self, x_t: List[List[float]], t: int,
                          model_output: List[List[float]]) -> List[List[float]]:
        """
        反向扩散一步: p(x_{t-1} | x_t)
        使用DDPM公式计算前一个时间步的样本。

        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * model_output)
                   + sigma_t * noise
        """
        beta_t = self._betas[t]
        alpha_t = self._alphas[t]
        alpha_bar_t = self._alpha_cumprod[t]
        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(1.0 - alpha_bar_t)

        # 计算均值
        pred_x0_coeff = beta_t / sqrt_one_minus_alpha_bar_t
        x_t_coeff = 1.0 / sqrt_alpha_t

        shape = _tensor_shape(x_t)
        noise = _randn(shape)

        result = []
        for i in range(len(x_t)):
            row = []
            for j in range(len(x_t[i])):
                mean = x_t_coeff * (x_t[i][j] - pred_x0_coeff * model_output[i][j])
                # 后验方差
                if t > 0:
                    sigma = math.sqrt(beta_t) * 0.5
                    val = mean + sigma * noise[i][j]
                else:
                    val = mean
                row.append(val)
            result.append(row)

        return result


# ===========================================================================
# 3. UNetMini - 简化UNet
# ===========================================================================

class UNetMini:
    """
    简化的UNet模型，用于噪声预测。
    包含编码器（下采样）、中间层和解码器（上采样）。
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 time_embed_dim: int = 256, seed: Optional[int] = 42):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._time_embed_dim = time_embed_dim
        self._seed = seed
        rng = random.Random(seed)

        # 时间嵌入MLP权重
        self._time_mlp1_w = [[rng.gauss(0, 0.02) for _ in range(time_embed_dim)] for _ in range(in_channels)]
        self._time_mlp1_b = [0.0] * time_embed_dim
        self._time_mlp2_w = [[rng.gauss(0, 0.02) for _ in range(time_embed_dim)] for _ in range(time_embed_dim)]
        self._time_mlp2_b = [0.0] * time_embed_dim

        # 下采样块配置
        self._down_blocks = [
            {"in_ch": in_channels, "out_ch": 128, "attention": False},
            {"in_ch": 128, "out_ch": 256, "attention": True},
            {"in_ch": 256, "out_ch": 512, "attention": True},
        ]

        # 上采样块配置
        self._up_blocks = [
            {"in_ch": 512, "out_ch": 256, "attention": True},
            {"in_ch": 256, "out_ch": 128, "attention": True},
            {"in_ch": 128, "out_ch": out_channels, "attention": False},
        ]

        # 中间块
        self._mid_ch = 512

        # 为每个块生成权重
        self._block_weights = {}
        for block in self._down_blocks + self._up_blocks:
            key = f"{block['in_ch']}_{block['out_ch']}"
            self._block_weights[key] = {
                "conv1": [[rng.gauss(0, 0.02) for _ in range(block['out_ch'])] for _ in range(block['in_ch'])],
                "conv1_b": [0.0] * block['out_ch'],
                "conv2": [[rng.gauss(0, 0.02) for _ in range(block['out_ch'])] for _ in range(block['out_ch'])],
                "conv2_b": [0.0] * block['out_ch'],
            }

        # 中间块权重
        self._mid_weights = {
            "conv1": [[rng.gauss(0, 0.02) for _ in range(self._mid_ch)] for _ in range(self._mid_ch)],
            "conv1_b": [0.0] * self._mid_ch,
            "conv2": [[rng.gauss(0, 0.02) for _ in range(self._mid_ch)] for _ in range(self._mid_ch)],
            "conv2_b": [0.0] * self._mid_ch,
        }

    def _time_embedding(self, t: int) -> List[float]:
        """
        时间步嵌入。
        使用正弦位置编码将时间步映射到高维空间，
        然后通过两层MLP。
        """
        half_dim = self._time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = [math.exp(i * emb) for i in range(half_dim)]

        # 正弦编码
        pos_emb = []
        for i in range(half_dim):
            pos_emb.append(math.sin(t * emb[i]))
        for i in range(half_dim):
            pos_emb.append(math.cos(t * emb[i]))

        # MLP
        hidden = _linear(pos_emb, self._time_mlp1_w, self._time_mlp1_b)
        hidden = [_gelu(h) for h in hidden]
        output = _linear(hidden, self._time_mlp2_w, self._time_mlp2_b)
        return output

    def _residual_block(self, x: List[float], embed: List[float],
                        weights: Dict) -> List[float]:
        """
        残差块: output = x + conv(relu(conv(x + embed)))
        """
        # 将embed加到输入
        x_with_embed = [xi + ei for xi, ei in zip(x, embed)] if len(x) == len(embed) else x

        # 第一个卷积
        in_dim = len(x)
        out_dim = weights["conv1_b"] if isinstance(weights["conv1_b"], list) else weights["out_ch"]
        hidden = _linear(x_with_embed, weights["conv1"], weights["conv1_b"])
        hidden = [_relu(h) for h in hidden]

        # 第二个卷积
        out = _linear(hidden, weights["conv2"], weights["conv2_b"])

        # 残差连接（如果维度匹配）
        if len(x) == len(out):
            out = [xi + oi for xi, oi in zip(x, out)]

        return out

    def _attention_block(self, x: List[float], context: Optional[List[List[float]]]) -> List[float]:
        """
        简化的注意力块。
        将输入视为单个token，与context进行交叉注意力。
        """
        dim = len(x)
        scale = math.sqrt(dim)

        if context is None:
            return x

        # Q = x, K = V = context的平均
        context_mean = []
        for j in range(len(context[0])):
            avg = sum(context[i][j] for i in range(len(context))) / len(context)
            context_mean.append(avg)

        # 简化注意力：计算x与context的相似度
        dot = sum(xi * ci for xi, ci in zip(x, context_mean)) / scale
        attn_weight = _sigmoid(dot)

        # 加权组合
        result = [xi * (1 - attn_weight) + ci * attn_weight for xi, ci in zip(x, context_mean)]
        return result

    def _downsample(self, x: List[List[float]]) -> List[List[float]]:
        """
        下采样：取每两个相邻元素的平均值。
        """
        if len(x) <= 1:
            return x
        new_len = len(x) // 2
        result = []
        for i in range(new_len):
            row = []
            for j in range(len(x[0]) // 2):
                val = (x[2 * i][2 * j] + x[2 * i][2 * j + 1] +
                       x[2 * i + 1][2 * j] + x[2 * i + 1][2 * j + 1]) / 4.0
                row.append(val)
            result.append(row)
        return result

    def _upsample(self, x: List[List[float]]) -> List[List[float]]:
        """
        上采样：最近邻插值，将每个元素复制为2x2块。
        """
        result = []
        for row in x:
            new_row = []
            for val in row:
                new_row.extend([val, val])
            result.append(new_row)
            result.append(new_row[:])
        return result

    def forward(self, x: List[List[float]], t: int,
                context: Optional[List[List[float]]] = None) -> List[List[float]]:
        """
        UNet前向传播（轻量级实现）。
        使用简化的投影和激活函数，避免大矩阵乘法。

        Args:
            x: 输入潜变量 [batch, channels]
            t: 时间步
            context: 文本条件 [seq_len, embed_dim]

        Returns:
            预测的噪声 [batch, channels]
        """
        # 时间嵌入（简化）
        t_emb = self._time_embedding(t)

        # 计算文本条件均值
        cond_mean = None
        if context is not None and len(context) > 0:
            cond_dim = len(context[0])
            cond_mean = [0.0] * cond_dim
            for row in context:
                for j in range(cond_dim):
                    cond_mean[j] += row[j]
            cond_mean = [v / len(context) for v in cond_mean]

        # 简化的UNet：使用多层非线性变换模拟编码-解码
        h = x
        skips = []

        # 编码器路径
        for block in self._down_blocks:
            out_ch = block["out_ch"]
            processed = []
            for row in h:
                # 简化投影：使用时间嵌入调制 + 非线性变换
                out = []
                for j in range(out_ch):
                    # 输入贡献
                    in_val = row[j % len(row)] if j < len(row) else 0.0
                    # 时间嵌入调制
                    t_val = t_emb[j % len(t_emb)] * 0.1
                    # 条件调制
                    c_val = 0.0
                    if cond_mean and j < len(cond_mean):
                        c_val = cond_mean[j] * 0.05
                    # 非线性组合
                    val = _gelu(in_val + t_val + c_val)
                    out.append(val)
                processed.append(out)
            h = processed
            skips.append(h)

        # 中间块
        mid_processed = []
        for row in h:
            out = []
            for j in range(self._mid_ch):
                in_val = row[j % len(row)] if j < len(row) else 0.0
                t_val = t_emb[j % len(t_emb)] * 0.1
                c_val = cond_mean[j] * 0.05 if cond_mean and j < len(cond_mean) else 0.0
                val = _gelu(in_val + t_val + c_val)
                out.append(val)
            mid_processed.append(out)
        h = mid_processed

        # 解码器路径
        for idx, block in enumerate(self._up_blocks):
            out_ch = block["out_ch"]
            # 跳跃连接
            skip_idx = len(self._up_blocks) - 1 - idx
            if skip_idx < len(skips):
                skip = skips[skip_idx]
                h = [_tensor_add(hi, si[:len(hi)]) if len(si) >= len(hi)
                     else _tensor_add(hi, si + [0.0] * (len(hi) - len(si)))
                     for hi, si in zip(h, skip)]

            processed = []
            for row in h:
                out = []
                for j in range(out_ch):
                    in_val = row[j % len(row)] if j < len(row) else 0.0
                    t_val = t_emb[j % len(t_emb)] * 0.1
                    c_val = cond_mean[j] * 0.05 if cond_mean and j < len(cond_mean) else 0.0
                    val = _gelu(in_val + t_val + c_val)
                    out.append(val)
                processed.append(out)
            h = processed

        # 确保输出维度正确
        result = []
        for row in h:
            if len(row) >= self._out_channels:
                result.append(row[:self._out_channels])
            else:
                result.append(row + [0.0] * (self._out_channels - len(row)))

        return result


# ===========================================================================
# 4. VAEDecoder - VAE解码器
# ===========================================================================

class VAEDecoder:
    """
    VAE解码器的纯Python实现。
    将潜空间表示解码为RGB图像像素值。
    """

    def __init__(self, latent_channels: int = 4, out_channels: int = 3, seed: Optional[int] = 42):
        self._latent_channels = latent_channels
        self._out_channels = out_channels
        self._seed = seed
        rng = random.Random(seed)

        # 解码器卷积核权重
        self._decoder_weights = []
        channels_list = [latent_channels, 64, 128, 256, out_channels]
        for i in range(len(channels_list) - 1):
            kernel_size = 3
            w = [[[
                rng.gauss(0, 0.02) for _ in range(channels_list[i + 1])
            ] for _ in range(kernel_size)] for _ in range(kernel_size)]
            self._decoder_weights.append({
                "weight": w,
                "bias": [0.0] * channels_list[i + 1],
                "in_ch": channels_list[i],
                "out_ch": channels_list[i + 1],
            })

        # 编码器卷积核权重
        self._encoder_weights = []
        enc_channels = [out_channels, 64, 128, 256, latent_channels]
        for i in range(len(enc_channels) - 1):
            kernel_size = 3
            w = [[[
                rng.gauss(0, 0.02) for _ in range(enc_channels[i + 1])
            ] for _ in range(kernel_size)] for _ in range(kernel_size)]
            self._encoder_weights.append({
                "weight": w,
                "bias": [0.0] * enc_channels[i + 1],
                "in_ch": enc_channels[i],
                "out_ch": enc_channels[i + 1],
            })

    def _conv2d(self, x: List[List[List[float]]], kernel: List[List[List[float]]],
                stride: int = 1, padding: int = 0) -> List[List[List[float]]]:
        """
        2D卷积操作（纯Python实现）。
        x: [height, width, channels]
        kernel: [kH, kW, in_channels, out_channels]
        """
        h_in = len(x)
        w_in = len(x[0]) if h_in > 0 else 0
        c_in = len(x[0][0]) if w_in > 0 else 0

        k_h = len(kernel)
        k_w = len(kernel[0]) if k_h > 0 else 0
        c_out = len(kernel[0][0]) if k_w > 0 else 0

        # 填充
        if padding > 0:
            padded = [[[0.0] * c_in for _ in range(w_in + 2 * padding)]
                       for _ in range(h_in + 2 * padding)]
            for i in range(h_in):
                for j in range(w_in):
                    padded[i + padding][j + padding] = x[i][j]
            x = padded
            h_in += 2 * padding
            w_in += 2 * padding

        h_out = (h_in - k_h) // stride + 1
        w_out = (w_in - k_w) // stride + 1

        output = [[[0.0] * c_out for _ in range(w_out)] for _ in range(h_out)]

        for oh in range(h_out):
            for ow in range(w_out):
                for oc in range(c_out):
                    val = 0.0
                    for kh in range(k_h):
                        for kw in range(k_w):
                            ih = oh * stride + kh
                            iw = ow * stride + kw
                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                for ic in range(c_in):
                                    val += x[ih][iw][ic] * kernel[kh][kw][ic][oc]
                    output[oh][ow][oc] = val

        return output

    def _upsample_nearest(self, x: List[List[List[float]]], scale: int = 2) -> List[List[List[float]]]:
        """
        最近邻上采样。
        将每个像素复制为scale x scale的块。
        """
        h = len(x)
        w = len(x[0]) if h > 0 else 0
        c = len(x[0][0]) if w > 0 else 0

        new_h = h * scale
        new_w = w * scale

        output = [[[0.0] * c for _ in range(new_w)] for _ in range(new_h)]
        for i in range(new_h):
            for j in range(new_w):
                output[i][j] = x[i // scale][j // scale][:]
        return output

    def _relu(self, x: List) -> List:
        """ReLU激活函数（张量版本）"""
        if isinstance(x[0], list):
            return [self._relu(xi) for xi in x]
        return [_relu(v) for v in x]

    def _sigmoid(self, x: List) -> List:
        """Sigmoid激活函数（张量版本）"""
        if isinstance(x[0], list):
            return [self._sigmoid(xi) for xi in x]
        return [_sigmoid(v) for v in x]

    def decode(self, latent: List[List[float]]) -> List[List[List[int]]]:
        """
        解码潜空间到RGB图像。

        Args:
            latent: 潜空间表示 [batch, latent_channels]

        Returns:
            RGB图像 [height, width, 3]，像素值范围[0, 255]
        """
        # 简化解码：将潜变量映射到图像尺寸
        batch_size = len(latent)
        img_size = 64  # 简化输出尺寸

        images = []
        for b in range(batch_size):
            lat = latent[b]
            # 使用潜变量生成确定性图像
            img = [[[0] * 3 for _ in range(img_size)] for _ in range(img_size)]

            for i in range(img_size):
                for j in range(img_size):
                    for c in range(3):
                        # 使用潜变量的不同通道和位置信息生成像素值
                        lat_idx = c % len(lat)
                        pos_factor = math.sin(i * 0.1 + lat[0]) * math.cos(j * 0.1 + lat[1])
                        val = 128 + int(lat[lat_idx] * 64 + pos_factor * 32)
                        val = max(0, min(255, val))
                        img[i][j][c] = val

            images.append(img)

        return images

    def encode(self, image: List[List[List[int]]]) -> List[List[float]]:
        """
        编码图像到潜空间。

        Args:
            image: RGB图像 [height, width, 3]

        Returns:
            潜空间表示 [1, latent_channels]
        """
        h = len(image)
        w = len(image[0]) if h > 0 else 0

        # 简化编码：计算图像统计特征
        latent = [0.0] * self._latent_channels

        for c in range(self._latent_channels):
            total = 0.0
            count = 0
            for i in range(h):
                for j in range(w):
                    img_c = c % 3
                    total += image[i][j][img_c]
                    count += 1
            if count > 0:
                latent[c] = (total / count - 128.0) / 128.0

        return [latent]


# ===========================================================================
# 5. StableDiffusionPipeline - SD管线
# ===========================================================================

class StableDiffusionPipeline:
    """
    Stable Diffusion管线主类。
    整合文本编码、UNet去噪、VAE解码等组件。
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._text_encoder = CLIPTextEncoder(seed=seed or 42)
        self._unet = UNetMini(seed=seed or 42)
        self._vae = VAEDecoder(seed=seed or 42)
        self._scheduler = DDIMScheduler()
        self._safety_checker = SafetyChecker()

    def _encode_prompt(self, prompt: str) -> List[List[float]]:
        """编码提示词为文本嵌入"""
        return self._text_encoder.encode(prompt)

    def _random_latents(self, shape: List[int], seed: Optional[int] = None) -> List[List[float]]:
        """生成随机初始潜变量"""
        actual_seed = seed if seed is not None else self._seed
        return _randn(shape, seed=actual_seed)

    def _classifier_free_guidance(self, noise_pred: List[List[float]],
                                   text_embeds: List[List[float]],
                                   neg_embeds: List[List[float]],
                                   guidance_scale: float) -> List[List[float]]:
        """
        无分类器引导(CFG)。
        noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        """
        # 简化：使用文本嵌入的统计信息调整噪声预测
        text_mean = []
        neg_mean = []
        dim = len(noise_pred[0])

        for j in range(dim):
            t_vals = [text_embeds[i][j] for i in range(min(len(text_embeds), dim))]
            n_vals = [neg_embeds[i][j] for i in range(min(len(neg_embeds), dim))]
            text_mean.append(sum(t_vals) / len(t_vals) if t_vals else 0.0)
            neg_mean.append(sum(n_vals) / len(n_vals) if n_vals else 0.0)

        result = []
        for i in range(len(noise_pred)):
            row = []
            for j in range(len(noise_pred[i])):
                guided = noise_pred[i][j] + guidance_scale * (text_mean[j] - neg_mean[j]) * 0.01
                row.append(guided)
            result.append(row)

        return result

    def _denoise_loop(self, latent: List[List[float]], text_embeds: List[List[float]],
                      neg_embeds: List[List[float]], steps: int,
                      guidance_scale: float) -> List[List[float]]:
        """
        去噪循环。
        在每个时间步使用UNet预测噪声并通过调度器更新潜变量。
        """
        self._scheduler.set_timesteps(steps)
        current_latent = [row[:] for row in latent]

        for step_idx, t in enumerate(self._scheduler._timesteps):
            # UNet预测噪声
            noise_pred = self._unet.forward(current_latent, t, text_embeds)

            # 应用CFG
            noise_pred = self._classifier_free_guidance(
                noise_pred, text_embeds, neg_embeds, guidance_scale
            )

            # 调度器步进
            current_latent = self._scheduler.step(noise_pred, t, current_latent)

        return current_latent

    def _latents_to_image(self, latents: List[List[float]]) -> List[List[List[int]]]:
        """将潜空间解码为图像"""
        return self._vae.decode(latents)

    def generate(self, prompt: str, negative_prompt: str = "",
                 num_steps: int = 50, guidance_scale: float = 7.5,
                 width: int = 512, height: int = 512,
                 seed: Optional[int] = None) -> GenerationResult:
        """
        生成图像主方法。

        Args:
            prompt: 正向提示词
            negative_prompt: 反向提示词
            num_steps: 去噪步数
            guidance_scale: 引导强度
            width: 图像宽度
            height: 图像高度
            seed: 随机种子

        Returns:
            GenerationResult: 生成结果
        """
        actual_seed = seed if seed is not None else self._seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2**32 - 1)

        # 编码提示词
        text_embeds = self._encode_prompt(prompt)
        neg_embeds = self._encode_prompt(negative_prompt) if negative_prompt else \
            self._encode_prompt("")

        # 生成初始潜变量 [1, 4]
        latent_shape = [1, 4]
        latent = self._random_latents(latent_shape, seed=actual_seed)

        # 去噪循环
        denoised_latent = self._denoise_loop(
            latent, text_embeds, neg_embeds, num_steps, guidance_scale
        )

        # 解码为图像
        images = self._latents_to_image(denoised_latent)

        return GenerationResult(
            images=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=actual_seed,
            steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )


# ===========================================================================
# 6. DDIMScheduler - DDIM噪声调度
# ===========================================================================

class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models) 噪声调度器。
    支持更高效的采样过程。
    """

    def __init__(self, num_train_timesteps: int = 1000):
        self._num_train_timesteps = num_train_timesteps
        self._timesteps: List[int] = []
        self._num_inference_steps = 50

        # 计算alpha相关系数
        self._betas = [
            0.00085 + (0.012 - 0.00085) * i / (num_train_timesteps - 1)
            for i in range(num_train_timesteps)
        ]
        self._alphas = [1.0 - b for b in self._betas]
        self._alpha_cumprod = []
        cumprod = 1.0
        for a in self._alphas:
            cumprod *= a
            self._alpha_cumprod.append(cumprod)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """
        设置推理步数。
        将训练时间步均匀划分为num_inference_steps个步。
        """
        self._num_inference_steps = num_inference_steps
        step_size = self._num_train_timesteps // num_inference_steps
        self._timesteps = [
            self._num_train_timesteps - 1 - i * step_size
            for i in range(num_inference_steps)
        ]

    def step(self, model_output: List[List[float]], timestep: int,
             sample: List[List[float]], eta: float = 0.0) -> List[List[float]]:
        """
        DDIM单步去噪。

        x_{t-1} = sqrt(alpha_prev) * pred_x0 + sqrt(1 - alpha_prev) * noise_direction

        Args:
            model_output: UNet预测的噪声
            timestep: 当前时间步
            sample: 当前样本
            eta: 随机性参数（0为完全确定性）

        Returns:
            去噪后的样本
        """
        # 获取当前和前一步的alpha
        alpha_t = self._alpha_cumprod[timestep]

        # 找到前一步
        prev_timestep = timestep - (self._num_train_timesteps // self._num_inference_steps)
        if prev_timestep < 0:
            alpha_prev = 1.0
        else:
            alpha_prev = self._alpha_cumprod[prev_timestep]

        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_alpha_prev = math.sqrt(alpha_prev)
        sqrt_one_minus_alpha_t = math.sqrt(1.0 - alpha_t)
        sqrt_one_minus_alpha_prev = math.sqrt(1.0 - alpha_prev)

        # 预测x_0
        pred_x0 = []
        for i in range(len(sample)):
            row = []
            for j in range(len(sample[i])):
                val = (sample[i][j] - sqrt_one_minus_alpha_t * model_output[i][j]) / sqrt_alpha_t
                row.append(val)
            pred_x0.append(row)

        # 计算噪声方向
        noise_direction = []
        for i in range(len(sample)):
            row = []
            for j in range(len(sample[i])):
                val = sqrt_one_minus_alpha_t * model_output[i][j]
                row.append(val)
            noise_direction.append(row)

        # DDIM更新
        result = []
        for i in range(len(sample)):
            row = []
            for j in range(len(sample[i])):
                val = sqrt_alpha_prev * pred_x0[i][j] + sqrt_one_minus_alpha_prev * noise_direction[i][j]
                row.append(val)
            result.append(row)

        return result

    def add_noise(self, original: List[List[float]], noise: List[List[float]],
                  timesteps: List[int]) -> List[List[float]]:
        """
        添加噪声到原始样本。
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        result = []
        for i in range(len(original)):
            t = timesteps[i] if i < len(timesteps) else timesteps[-1]
            sqrt_alpha = math.sqrt(self._alpha_cumprod[t])
            sqrt_one_minus_alpha = math.sqrt(1.0 - self._alpha_cumprod[t])
            row = []
            for j in range(len(original[i])):
                val = sqrt_alpha * original[i][j] + sqrt_one_minus_alpha * noise[i][j]
                row.append(val)
            result.append(row)
        return result


# ===========================================================================
# 7. ControlNetUnit - ControlNet条件控制
# ===========================================================================

class ControlNetUnit:
    """
    ControlNet条件控制单元。
    支持多种条件类型：pose、depth、canny、lineart、scribble。
    """

    def __init__(self, condition_type: str = "canny", seed: Optional[int] = 42):
        self._condition_type = condition_type
        self._seed = seed
        self._controlnet = UNetMini(seed=seed)
        self._condition_image: Optional[List] = None
        self._condition_strength = 1.0

    def set_condition(self, condition_image: List) -> None:
        """设置条件图像"""
        self._condition_image = condition_image

    def set_condition_strength(self, strength: float) -> None:
        """设置条件强度"""
        self._condition_strength = max(0.0, min(2.0, strength))

    def _preprocess_condition(self, image: List, cond_type: str) -> List:
        """
        预处理条件图像。
        根据条件类型调用不同的预处理方法。
        """
        if cond_type == "canny":
            return self._canny_edge_detection(image, low=50, high=150)
        elif cond_type == "depth":
            return self._depth_estimation(image)
        elif cond_type == "pose":
            return self._pose_detection(image)
        elif cond_type == "lineart":
            return self._lineart_extraction(image)
        elif cond_type == "scribble":
            return self._simplify_to_scribble(image)
        else:
            return image

    def _canny_edge_detection(self, image: List, low: float = 50,
                               high: float = 150) -> List:
        """
        Canny边缘检测算法。
        步骤: 高斯模糊 -> Sobel梯度 -> 非极大值抑制 -> 双阈值 -> 滞后边缘跟踪
        """
        if not image or not isinstance(image[0], list):
            return image

        h = len(image)
        w = len(image[0]) if h > 0 else 0

        # 转为灰度
        gray = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if isinstance(image[i][j], list):
                    gray[i][j] = sum(image[i][j]) // len(image[i][j])
                else:
                    gray[i][j] = image[i][j]

        # 高斯模糊 (3x3)
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel_sum = 16
        blurred = [[0] * w for _ in range(h)]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                val = 0
                for ki in range(3):
                    for kj in range(3):
                        val += gray[i + ki - 1][j + kj - 1] * kernel[ki][kj]
                blurred[i][j] = val // kernel_sum

        # Sobel梯度
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        grad_mag = [[0] * w for _ in range(h)]
        grad_dir = [[0.0] * w for _ in range(h)]

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                gx = 0
                gy = 0
                for ki in range(3):
                    for kj in range(3):
                        gx += blurred[i + ki - 1][j + kj - 1] * sobel_x[ki][kj]
                        gy += blurred[i + ki - 1][j + kj - 1] * sobel_y[ki][kj]
                grad_mag[i][j] = int(math.sqrt(gx * gx + gy * gy))
                grad_dir[i][j] = math.atan2(gy, gx)

        # 非极大值抑制
        nms = [[0] * w for _ in range(h)]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                angle = grad_dir[i][j] * 180.0 / math.pi
                if angle < 0:
                    angle += 180.0

                mag = grad_mag[i][j]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    n1, n2 = grad_mag[i][j + 1], grad_mag[i][j - 1]
                elif 22.5 <= angle < 67.5:
                    n1, n2 = grad_mag[i + 1][j - 1], grad_mag[i - 1][j + 1]
                elif 67.5 <= angle < 112.5:
                    n1, n2 = grad_mag[i + 1][j], grad_mag[i - 1][j]
                else:
                    n1, n2 = grad_mag[i - 1][j - 1], grad_mag[i + 1][j + 1]

                if mag >= n1 and mag >= n2:
                    nms[i][j] = mag

        # 双阈值 + 滞后边缘跟踪
        edges = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if nms[i][j] >= high:
                    edges[i][j] = 255
                elif nms[i][j] >= low:
                    # 检查8邻域是否有强边缘
                    is_edge = False
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and nms[ni][nj] >= high:
                                is_edge = True
                                break
                        if is_edge:
                            break
                    edges[i][j] = 255 if is_edge else 0

        return edges

    def _depth_estimation(self, image: List) -> List:
        """
        简化的深度估计。
        使用梯度幅值作为深度代理。
        """
        if not image or not isinstance(image[0], list):
            return image

        h = len(image)
        w = len(image[0]) if h > 0 else 0

        # 转为灰度
        gray = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if isinstance(image[i][j], list):
                    gray[i][j] = sum(image[i][j]) // len(image[i][j])
                else:
                    gray[i][j] = image[i][j]

        # 计算垂直梯度作为深度代理
        depth = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if i < h - 1:
                    depth[i][j] = abs(gray[i + 1][j] - gray[i][j])
                else:
                    depth[i][j] = 0

        # 归一化到[0, 255]
        max_depth = max(max(row) for row in depth) if depth else 1
        if max_depth == 0:
            max_depth = 1
        for i in range(h):
            for j in range(w):
                depth[i][j] = int(depth[i][j] / max_depth * 255)

        return depth

    def _pose_detection(self, image: List) -> List:
        """
        简化的姿态检测。
        检测图像中的关键边缘和角点作为姿态线索。
        """
        if not image or not isinstance(image[0], list):
            return image

        h = len(image)
        w = len(image[0]) if h > 0 else 0

        # 转为灰度
        gray = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if isinstance(image[i][j], list):
                    gray[i][j] = sum(image[i][j]) // len(image[i][j])
                else:
                    gray[i][j] = image[i][j]

        # Harris角点检测简化版
        # 计算x和y方向的梯度
        pose_map = [[0] * w for _ in range(h)]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                ix = gray[i][j + 1] - gray[i][j - 1]
                iy = gray[i + 1][j] - gray[i - 1][j]
                ixx = ix * ix
                iyy = iy * iy
                ixy = ix * iy
                # Harris响应
                det = ixx * iyy - ixy * ixy
                trace = ixx + iyy
                k = 0.04
                response = det - k * trace * trace
                pose_map[i][j] = max(0, min(255, int(response / 10000)))

        return pose_map

    def _lineart_extraction(self, image: List) -> List:
        """
        线稿提取。
        使用灰度反转 + 高斯模糊 + 颜色减淡混合。
        """
        if not image or not isinstance(image[0], list):
            return image

        h = len(image)
        w = len(image[0]) if h > 0 else 0

        # 转为灰度
        gray = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if isinstance(image[i][j], list):
                    gray[i][j] = sum(image[i][j]) // len(image[i][j])
                else:
                    gray[i][j] = image[i][j]

        # 灰度反转
        inverted = [[255 - gray[i][j] for j in range(w)] for i in range(h)]

        # 高斯模糊反转图
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel_sum = 16
        blurred = [[0] * w for _ in range(h)]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                val = 0
                for ki in range(3):
                    for kj in range(3):
                        val += inverted[i + ki - 1][j + kj - 1] * kernel[ki][kj]
                blurred[i][j] = val // kernel_sum

        # 颜色减淡混合: result = base / (1 - blend)
        lineart = [[0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                base = gray[i][j]
                blend = blurred[i][j]
                if blend >= 255:
                    lineart[i][j] = 255
                else:
                    denom = 255 - blend
                    if denom == 0:
                        lineart[i][j] = 255
                    else:
                        lineart[i][j] = min(255, int(base * 255 / denom))

        return lineart

    def _simplify_to_scribble(self, image: List) -> List:
        """将图像简化为涂鸦风格"""
        return self._canny_edge_detection(image, low=100, high=200)

    def apply_control(self, latent: List[List[float]], timestep: int,
                      context: Optional[List[List[float]]] = None) -> List[List[float]]:
        """
        应用ControlNet条件。
        将条件图像的影响注入到潜变量中。
        """
        if self._condition_image is None:
            return latent

        # 使用ControlNet UNet处理条件
        condition_latent = []
        for row in latent:
            condition_latent.append(row[:])

        # 通过ControlNet获取条件特征
        control_output = self._controlnet.forward(condition_latent, timestep, context)

        # 将条件特征与原始潜变量融合
        result = []
        for i in range(len(latent)):
            row = []
            for j in range(len(latent[i])):
                val = latent[i][j] + self._condition_strength * control_output[i][j] * 0.1
                row.append(val)
            result.append(row)

        return result


# ===========================================================================
# 8. SafetyChecker - 安全检查器
# ===========================================================================

class SafetyChecker:
    """
    内容安全检查器。
    检测生成图像中可能的不安全内容。
    """

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        # 不安全概念关键词
        self._unsafe_concepts = [
            "violence", "gore", "blood", "weapon", "nsfw",
            "hate", "illegal", "harmful", "dangerous",
        ]

    def check(self, image: List, prompt: str = "") -> Tuple[bool, float]:
        """
        检查图像安全性。

        Returns:
            (is_safe, confidence): 是否安全及置信度
        """
        # 基于提示词的简单检查
        prompt_lower = prompt.lower()
        max_score = 0.0

        for concept in self._unsafe_concepts:
            if concept in prompt_lower:
                max_score = max(max_score, 0.8)

        # 基于图像统计的检查（简化）
        if image and isinstance(image[0], list):
            h = len(image)
            w = len(image[0]) if h > 0 else 0
            if w > 0:
                # 检查极端像素值分布
                total_pixels = h * w
                extreme_count = 0
                for i in range(h):
                    for j in range(w):
                        if isinstance(image[i][j], list):
                            for c in image[i][j]:
                                if c < 10 or c > 245:
                                    extreme_count += 1
                        else:
                            if image[i][j] < 10 or image[i][j] > 245:
                                extreme_count += 1
                extreme_ratio = extreme_count / max(total_pixels, 1)
                score = extreme_ratio * 0.5
                max_score = max(max_score, score)

        is_safe = max_score < self._threshold
        return is_safe, max_score


# ===========================================================================
# 9. ImageInpainter - 图像局部重绘
# ===========================================================================

class ImageInpainter:
    """
    图像局部重绘（Inpainting）。
    对图像中被遮罩覆盖的区域进行重新生成。
    """

    def __init__(self, seed: Optional[int] = None):
        self._pipeline = StableDiffusionPipeline(seed=seed)
        self._seed = seed

    def _prepare_mask(self, mask: List[List[int]], latent_shape: List[int]) -> List[List[float]]:
        """
        准备掩码。
        将图像空间的掩码下采样到潜空间尺寸。
        """
        if not mask:
            return [[1.0] * latent_shape[1] for _ in range(latent_shape[0])]

        h_mask = len(mask)
        w_mask = len(mask[0]) if h_mask > 0 else 0

        # 计算缩放因子 (通常8x下采样)
        scale_h = max(1, h_mask // latent_shape[0])
        scale_w = max(1, w_mask // latent_shape[1])

        latent_mask = []
        for i in range(latent_shape[0]):
            row = []
            for j in range(latent_shape[1]):
                # 取对应区域的平均值
                total = 0
                count = 0
                for si in range(scale_h):
                    for sj in range(scale_w):
                        mi = i * scale_h + si
                        mj = j * scale_w + sj
                        if mi < h_mask and mj < w_mask:
                            total += mask[mi][mj]
                            count += 1
                avg = total / max(count, 1)
                row.append(avg / 255.0)
            latent_mask.append(row)

        return latent_mask

    def _blend(self, original: List[List[List[int]]], generated: List[List[List[int]]],
               mask: List[List[int]]) -> List[List[List[int]]]:
        """
        混合原始图像和生成图像。
        使用掩码决定哪些区域使用生成结果。
        """
        if not original or not generated:
            return generated or original

        h = min(len(original), len(generated))
        w = min(len(original[0]), len(generated[0])) if h > 0 else 0

        result = [[[0] * 3 for _ in range(w)] for _ in range(h)]
        for i in range(h):
            for j in range(w):
                mask_val = mask[i][j] / 255.0 if i < len(mask) and j < len(mask[i]) else 0.0
                for c in range(3):
                    orig_val = original[i][j][c] if c < len(original[i][j]) else 0
                    gen_val = generated[i][j][c] if c < len(generated[i][j]) else 0
                    blended = int(orig_val * (1 - mask_val) + gen_val * mask_val)
                    result[i][j][c] = max(0, min(255, blended))

        return result

    def inpaint(self, image: List[List[List[int]]], mask: List[List[int]],
                prompt: str, negative_prompt: str = "",
                num_steps: int = 50, guidance_scale: float = 7.5,
                seed: Optional[int] = None, **kwargs) -> GenerationResult:
        """
        执行图像局部重绘。

        Args:
            image: 原始图像 [H, W, 3]
            mask: 遮罩图像 [H, W]，255表示需要重绘的区域
            prompt: 提示词
            negative_prompt: 反向提示词
            num_steps: 去噪步数
            guidance_scale: 引导强度
            seed: 随机种子

        Returns:
            GenerationResult: 包含重绘结果
        """
        actual_seed = seed if seed is not None else self._seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2**32 - 1)

        # 编码原始图像到潜空间
        latent = self._pipeline._vae.encode(image)

        # 生成完整图像
        result = self._pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
        )

        # 混合原始图像和生成图像
        if result.images:
            blended = self._blend(image, result.images[0], mask)
            result.images = [blended]

        return result


# ===========================================================================
# 10. ImageUpscaler - 超分辨率
# ===========================================================================

class ImageUpscaler:
    """
    图像超分辨率放大器。
    使用Real-ESRGAN风格的残差网络进行图像放大。
    """

    def __init__(self, scale_factor: int = 4, seed: Optional[int] = 42):
        self._scale_factor = scale_factor
        self._seed = seed
        rng = random.Random(seed)

        # ESRGAN风格残差块权重
        self._num_residual_blocks = 8
        self._num_features = 32
        self._residual_weights = []
        for _ in range(self._num_residual_blocks):
            w = {
                "conv1": [[rng.gauss(0, 0.02) for _ in range(self._num_features)]
                           for _ in range(self._num_features)],
                "conv1_b": [0.0] * self._num_features,
                "conv2": [[rng.gauss(0, 0.02) for _ in range(self._num_features)]
                           for _ in range(self._num_features)],
                "conv2_b": [0.0] * self._num_features,
            }
            self._residual_weights.append(w)

        # 上采样权重
        self._upsample_weights = [
            [[rng.gauss(0, 0.02) for _ in range(self._num_features)]
             for _ in range(self._num_features * 4)]
        ]

    def _residual_block(self, x: List[float]) -> List[float]:
        """
        ESRGAN风格残差块。
        使用LeakyReLU激活和跳跃连接。
        """
        w = self._residual_weights[0]
        dim = len(x)

        # 调整维度
        if dim != self._num_features:
            # 使用前num_features个或填充
            x_adj = x[:self._num_features] if dim > self._num_features else \
                x + [0.0] * (self._num_features - dim)
        else:
            x_adj = x

        # 第一个卷积 + LeakyReLU
        hidden = _linear(x_adj, w["conv1"], w["conv1_b"])
        hidden = [max(0.2 * h, h) for h in hidden]  # LeakyReLU(0.2)

        # 第二个卷积
        out = _linear(hidden, w["conv2"], w["conv2_b"])

        # 残差连接
        result = [xi + oi * 0.1 for xi, oi in zip(x_adj, out)]
        return result

    def _pixel_shuffle(self, x: List[float], scale: int) -> List[float]:
        """
        像素重组（Pixel Shuffle）。
        将通道维度的特征重新排列到空间维度。
        [C*r^2] -> [C, r, r] 展平后实现上采样
        """
        c = len(x) // (scale * scale)
        result = []
        for i in range(c):
            for ry in range(scale):
                for rx in range(scale):
                    idx = i * scale * scale + ry * scale + rx
                    if idx < len(x):
                        result.append(x[idx])
        return result

    def _esrgan_upscale(self, image: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        Real-ESRGAN风格上采样。
        通过残差网络提取特征，然后使用像素重组进行上采样。
        """
        if not image or not isinstance(image[0], list):
            return image

        h = len(image)
        w = len(image[0]) if h > 0 else 0
        c = len(image[0][0]) if w > 0 else 0

        new_h = h * self._scale_factor
        new_w = w * self._scale_factor

        # 简化ESRGAN：双线性插值 + 残差增强
        # 先做双线性插值上采样
        upsampled = [[[0] * c for _ in range(new_w)] for _ in range(new_h)]

        for i in range(new_h):
            for j in range(new_w):
                # 映射回原图坐标
                src_y = i / self._scale_factor
                src_x = j / self._scale_factor

                y0 = int(src_y)
                x0 = int(src_x)
                y1 = min(y0 + 1, h - 1)
                x1 = min(x0 + 1, w - 1)

                fy = src_y - y0
                fx = src_x - x0

                for ch in range(c):
                    val = (image[y0][x0][ch] * (1 - fy) * (1 - fx) +
                           image[y0][x1][ch] * (1 - fy) * fx +
                           image[y1][x0][ch] * fy * (1 - fx) +
                           image[y1][x1][ch] * fy * fx)
                    upsampled[i][j][ch] = int(val)

        # 应用残差增强（简化版）
        for i in range(new_h):
            for j in range(new_w):
                # 使用周围像素做局部增强
                for ch in range(c):
                    val = upsampled[i][j][ch]
                    # 简单锐化
                    if 0 < i < new_h - 1 and 0 < j < new_w - 1:
                        center = val
                        neighbors = (upsampled[i-1][j][ch] + upsampled[i+1][j][ch] +
                                    upsampled[i][j-1][ch] + upsampled[i][j+1][ch]) / 4.0
                        sharpened = int(center * 1.5 - neighbors * 0.5)
                        upsampled[i][j][ch] = max(0, min(255, sharpened))

        return upsampled

    def upscale(self, image: List[List[List[int]]], prompt: str = "",
                seed: Optional[int] = None) -> GenerationResult:
        """
        执行超分辨率放大。

        Args:
            image: 输入图像 [H, W, 3]
            prompt: 可选的提示词（用于引导放大）
            seed: 随机种子

        Returns:
            GenerationResult: 包含放大后的图像
        """
        actual_seed = seed if seed is not None else self._seed

        # 执行ESRGAN风格上采样
        upscaled = self._esrgan_upscale(image)

        h = len(image)
        w = len(image[0]) if h > 0 else 0
        new_h = h * self._scale_factor
        new_w = w * self._scale_factor

        return GenerationResult(
            images=[upscaled],
            prompt=prompt,
            seed=actual_seed,
            width=new_w,
            height=new_h,
        )


# ===========================================================================
# 11. IPAdapter - 图像提示适配器
# ===========================================================================

class IPAdapter:
    """
    图像提示适配器(Image Prompt Adapter)。
    将参考图像的视觉特征注入到UNet中，实现图像引导生成。
    """

    def __init__(self, seed: Optional[int] = 42):
        self._seed = seed
        self._image_encoder = VAEDecoder(seed=seed)
        self._ip_layers = [
            {"dim": 256, "num_tokens": 4},
            {"dim": 512, "num_tokens": 4},
            {"dim": 512, "num_tokens": 4},
        ]
        self._reference_features: Optional[List[float]] = None
        self._reference_image: Optional[List] = None
        self._ip_scale = 1.0

        # IP-Adapter投影层权重
        rng = random.Random(seed)
        self._proj_weights = []
        for layer in self._ip_layers:
            w = {
                "image_proj": [[rng.gauss(0, 0.02) for _ in range(layer["dim"])]
                               for _ in range(layer["dim"])],
                "ip_proj": [[rng.gauss(0, 0.02) for _ in range(layer["dim"])]
                            for _ in range(layer["dim"])],
            }
            self._proj_weights.append(w)

    def set_reference_image(self, image: List) -> None:
        """设置参考图像"""
        self._reference_image = image
        self._reference_features = self._encode_image(image)

    def set_ip_scale(self, scale: float) -> None:
        """设置IP特征注入强度"""
        self._ip_scale = max(0.0, min(2.0, scale))

    def _encode_image(self, image: List) -> List[float]:
        """
        编码参考图像为特征向量。
        使用简化的CLIP图像编码逻辑。
        """
        if not image or not isinstance(image[0], list):
            return [0.0] * 256

        h = len(image)
        w = len(image[0]) if h > 0 else 0

        # 计算图像的全局特征
        features = []
        num_bins = 16  # 颜色直方图bin数

        for c in range(3):
            # 计算每个通道的颜色直方图
            hist = [0] * num_bins
            for i in range(h):
                for j in range(w):
                    if c < len(image[i][j]):
                        bin_idx = min(num_bins - 1, image[i][j][c] * num_bins // 256)
                        hist[bin_idx] += 1

            # 归一化
            total = sum(hist)
            if total > 0:
                hist = [h_val / total for h_val in hist]
            features.extend(hist)

        # 添加空间统计特征
        for c in range(3):
            mean_val = 0
            for i in range(h):
                for j in range(w):
                    if c < len(image[i][j]):
                        mean_val += image[i][j][c]
            mean_val = mean_val / max(h * w, 1)
            features.append(mean_val / 255.0)

        # 填充或截断到256维
        target_dim = 256
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]

        return features

    def _inject_ip_features(self, unet_features: List[List[float]],
                            ip_features: List[float]) -> List[List[float]]:
        """
        将IP特征注入到UNet特征中。
        使用交叉注意力机制将图像特征与文本特征融合。
        """
        if ip_features is None:
            return unet_features

        dim = len(ip_features)
        result = []

        for row in unet_features:
            # 使用门控机制融合
            gate = _sigmoid(sum(r * f for r, f in zip(row[:dim], ip_features)) / math.sqrt(dim))
            new_row = [r * (1 - self._ip_scale * gate) + f * self._ip_scale * gate
                       for r, f in zip(row[:dim], ip_features)]
            # 保持原始维度
            if len(row) > dim:
                new_row.extend(row[dim:])
            result.append(new_row)

        return result

    def generate(self, prompt: str, reference_image: Optional[List] = None,
                 negative_prompt: str = "", num_steps: int = 50,
                 guidance_scale: float = 7.5, width: int = 512,
                 height: int = 512, seed: Optional[int] = None,
                 **kwargs) -> GenerationResult:
        """
        使用图像提示生成图像。

        Args:
            prompt: 文本提示词
            reference_image: 参考图像
            negative_prompt: 反向提示词
            num_steps: 去噪步数
            guidance_scale: 引导强度
            width: 图像宽度
            height: 图像高度
            seed: 随机种子

        Returns:
            GenerationResult: 生成结果
        """
        actual_seed = seed if seed is not None else self._seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2**32 - 1)

        # 设置参考图像
        if reference_image is not None:
            self.set_reference_image(reference_image)

        # 使用SD管线生成
        pipeline = StableDiffusionPipeline(seed=actual_seed)
        text_embeds = pipeline._encode_prompt(prompt)
        neg_embeds = pipeline._encode_prompt(negative_prompt) if negative_prompt else \
            pipeline._encode_prompt("")

        # 生成初始潜变量
        latent_shape = [1, 4]
        latent = pipeline._random_latents(latent_shape, seed=actual_seed)

        # 去噪循环（带IP特征注入）
        pipeline._scheduler.set_timesteps(num_steps)
        current_latent = [row[:] for row in latent]

        for t in pipeline._scheduler._timesteps:
            # UNet预测噪声
            noise_pred = pipeline._unet.forward(current_latent, t, text_embeds)

            # 注入IP特征
            if self._reference_features is not None:
                noise_pred = self._inject_ip_features(noise_pred, self._reference_features)

            # CFG
            noise_pred = pipeline._classifier_free_guidance(
                noise_pred, text_embeds, neg_embeds, guidance_scale
            )

            # 调度器步进
            current_latent = pipeline._scheduler.step(noise_pred, t, current_latent)

        # 解码为图像
        images = pipeline._latents_to_image(current_latent)

        return GenerationResult(
            images=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=actual_seed,
            steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
