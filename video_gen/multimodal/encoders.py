"""
多模态编码器模块
实现文本、图像、音频、视频等多模态数据的编码和融合
"""

from typing import List, Dict, Tuple, Optional, Any
import math


class TextEncoder:
    """
    文本编码器
    基于词嵌入和位置编码的文本编码
    """
    
    def __init__(self, 
                 vocab_size: int = 49408,
                 embed_dim: int = 768,
                 max_length: int = 77):
        """
        初始化文本编码器
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
            max_length: 最大序列长度
        """
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._max_length = max_length
        
        # 初始化词嵌入表
        self._token_embedding: List[List[float]] = []
        self._init_token_embedding()
        
        # 初始化位置嵌入
        self._position_embedding: List[List[float]] = []
        self._init_position_embedding()
    
    def _init_token_embedding(self) -> None:
        """
        初始化词嵌入表
        使用Xavier/Glorot均匀初始化，使嵌入值在合理范围内分布
        Xavier初始化：limit = sqrt(6 / (fan_in + fan_out))
        这里fan_in=vocab_size, fan_out=embed_dim
        """
        import random
        rng = random.Random(42)  # 固定种子保证可复现性
        # Xavier均匀分布的边界值
        limit = math.sqrt(6.0 / (self._vocab_size + self._embed_dim))
        for i in range(self._vocab_size):
            # 每个词的嵌入向量使用Xavier初始化
            embedding = [rng.uniform(-limit, limit) for _ in range(self._embed_dim)]
            self._token_embedding.append(embedding)
    
    def _init_position_embedding(self) -> None:
        """初始化位置嵌入（正弦位置编码）"""
        for pos in range(self._max_length):
            embedding = []
            for i in range(self._embed_dim):
                if i % 2 == 0:
                    embedding.append(math.sin(pos / (10000 ** (i / self._embed_dim))))
                else:
                    embedding.append(math.cos(pos / (10000 ** ((i - 1) / self._embed_dim))))
            self._position_embedding.append(embedding)
    
    def encode(self, text: str) -> List[List[float]]:
        """
        编码文本（带Transformer编码层）
        
        流程：分词 -> 词嵌入+位置嵌入 -> Transformer编码(self-attention+FFN+LayerNorm) -> 输出
        
        Args:
            text: 输入文本
            
        Returns:
            文本嵌入 [seq_len, embed_dim]
        """
        # 分词
        tokens = self.tokenize(text)
        
        # 构建注意力掩码
        attention_mask = self._build_attention_mask(tokens)
        
        # 词嵌入 + 位置嵌入
        embeddings = []
        for i, token_id in enumerate(tokens):
            if token_id < self._vocab_size:
                token_emb = self._token_embedding[token_id]
            else:
                token_emb = [0.0] * self._embed_dim
            
            pos_emb = self._position_embedding[i] if i < self._max_length else [0.0] * self._embed_dim
            
            # 相加
            combined = [token_emb[j] + pos_emb[j] for j in range(self._embed_dim)]
            embeddings.append(combined)
        
        # ---- Transformer编码层 ----
        # 单层Transformer：self-attention + FFN + LayerNorm
        encoded = self._transformer_encode(embeddings, attention_mask)
        
        return encoded
    
    def _transformer_encode(self, embeddings: List[List[float]], 
                            attention_mask: List[int],
                            num_heads: int = 8) -> List[List[float]]:
        """
        Transformer编码层
        包含：多头自注意力 -> 残差连接 + LayerNorm -> FFN -> 残差连接 + LayerNorm
        
        Args:
            embeddings: 输入嵌入 [seq_len, embed_dim]
            attention_mask: 注意力掩码 [seq_len]
            num_heads: 注意力头数
            
        Returns:
            编码后的嵌入 [seq_len, embed_dim]
        """
        seq_len = len(embeddings)
        if seq_len == 0:
            return embeddings
        
        dim = self._embed_dim
        head_dim = dim // num_heads
        
        # ---- 多头自注意力（Multi-Head Self-Attention）----
        # 将嵌入投影到Q、K、V空间（使用确定性初始化的投影权重）
        Q = self._linear_projection(embeddings, dim, dim)
        K = self._linear_projection(embeddings, dim, dim)
        V = self._linear_projection(embeddings, dim, dim)
        
        # 分割为多头
        attn_output = self._multi_head_attention(Q, K, V, num_heads, head_dim, attention_mask)
        
        # 残差连接 + LayerNorm
        residual = embeddings
        embeddings = self._layer_norm(
            [[embeddings[i][j] + attn_output[i][j] for j in range(dim)] for i in range(seq_len)]
        )
        
        # ---- 前馈网络（FFN）----
        # FFN(x) = GELU(xW1 + b1)W2 + b2
        ffn_output = self._feed_forward(embeddings, dim, dim * 4)
        
        # 残差连接 + LayerNorm
        embeddings = self._layer_norm(
            [[embeddings[i][j] + ffn_output[i][j] for j in range(dim)] for i in range(seq_len)]
        )
        
        return embeddings
    
    def _linear_projection(self, x: List[List[float]], in_dim: int, out_dim: int) -> List[List[float]]:
        """
        线性投影层（确定性权重）
        
        Args:
            x: 输入 [seq_len, in_dim]
            in_dim: 输入维度
            out_dim: 输出维度
            
        Returns:
            投影结果 [seq_len, out_dim]
        """
        import random
        rng = random.Random(123)  # 固定种子保证确定性
        # Xavier初始化投影权重
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        # 生成投影权重矩阵 [in_dim, out_dim]
        weights = [[rng.uniform(-limit, limit) for _ in range(out_dim)] for _ in range(in_dim)]
        bias = [0.0] * out_dim
        
        result = []
        for seq in x:
            projected = bias[:]
            for i in range(min(len(seq), in_dim)):
                for j in range(out_dim):
                    projected[j] += seq[i] * weights[i][j]
            result.append(projected)
        return result
    
    def _multi_head_attention(self, Q: List[List[float]], K: List[List[float]], 
                              V: List[List[float]], num_heads: int, head_dim: int,
                              mask: List[int]) -> List[List[float]]:
        """
        多头自注意力计算
        
        Args:
            Q: 查询矩阵 [seq_len, dim]
            K: 键矩阵 [seq_len, dim]
            V: 值矩阵 [seq_len, dim]
            num_heads: 注意力头数
            head_dim: 每个头的维度
            mask: 注意力掩码
            
        Returns:
            注意力输出 [seq_len, dim]
        """
        seq_len = len(Q)
        dim = self._embed_dim
        output = [[0.0] * dim for _ in range(seq_len)]
        
        for h in range(num_heads):
            # 每个头处理对应的维度切片
            offset = h * head_dim
            
            # 计算该头的注意力分数
            for i in range(seq_len):
                if mask[i] == 0:
                    continue  # 跳过padding位置
                
                # 计算注意力权重
                scores = []
                for j in range(seq_len):
                    if mask[j] == 0:
                        scores.append(-1e9)  # padding位置给极小值
                        continue
                    # 点积注意力分数
                    score = sum(Q[i][offset + k] * K[j][offset + k] for k in range(head_dim))
                    score /= math.sqrt(head_dim)  # 缩放
                    scores.append(score)
                
                # Softmax归一化
                max_score = max(scores) if scores else 0
                exp_scores = [math.exp(s - max_score) for s in scores]
                sum_exp = sum(exp_scores)
                if sum_exp > 0:
                    attn_weights = [e / sum_exp for e in exp_scores]
                else:
                    attn_weights = [0.0] * len(scores)
                
                # 加权求和得到注意力输出
                for j in range(seq_len):
                    for k in range(head_dim):
                        output[i][offset + k] += attn_weights[j] * V[j][offset + k]
        
        return output
    
    def _feed_forward(self, x: List[List[float]], in_dim: int, hidden_dim: int) -> List[List[float]]:
        """
        前馈网络（FFN）
        两层线性变换 + GELU激活：FFN(x) = GELU(xW1 + b1)W2 + b2
        
        Args:
            x: 输入 [seq_len, in_dim]
            in_dim: 输入维度
            hidden_dim: 隐藏层维度
            
        Returns:
            FFN输出 [seq_len, in_dim]
        """
        import random
        rng = random.Random(456)
        
        # 第一层权重 [in_dim, hidden_dim]
        limit1 = math.sqrt(6.0 / (in_dim + hidden_dim))
        W1 = [[rng.uniform(-limit1, limit1) for _ in range(hidden_dim)] for _ in range(in_dim)]
        b1 = [0.0] * hidden_dim
        
        # 第二层权重 [hidden_dim, in_dim]
        limit2 = math.sqrt(6.0 / (hidden_dim + in_dim))
        W2 = [[rng.uniform(-limit2, limit2) for _ in range(in_dim)] for _ in range(hidden_dim)]
        b2 = [0.0] * in_dim
        
        result = []
        for seq in x:
            # 第一层线性变换
            hidden = b1[:]
            for i in range(min(len(seq), in_dim)):
                for j in range(hidden_dim):
                    hidden[j] += seq[i] * W1[i][j]
            
            # GELU激活函数：GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            activated = []
            for val in hidden:
                gelu_val = 0.5 * val * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (val + 0.044715 * val ** 3)))
                activated.append(gelu_val)
            
            # 第二层线性变换
            output = b2[:]
            for i in range(hidden_dim):
                for j in range(in_dim):
                    output[j] += activated[i] * W2[i][j]
            
            result.append(output)
        
        return result
    
    def _layer_norm(self, x: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
        """
        LayerNorm层归一化
        对每个向量的所有维度进行归一化：y = (x - mean) / sqrt(var + eps) * gamma + beta
        
        Args:
            x: 输入 [seq_len, dim]
            eps: 防止除零的小常数
            
        Returns:
            归一化后的输出 [seq_len, dim]
        """
        dim = len(x[0]) if x else 0
        # gamma和beta初始化为1和0（可学习参数的初始值）
        gamma = [1.0] * dim
        beta = [0.0] * dim
        
        result = []
        for seq in x:
            # 计算均值和方差
            mean = sum(seq) / dim
            variance = sum((v - mean) ** 2 for v in seq) / dim
            
            # 归一化
            normalized = []
            for j in range(dim):
                norm_val = (seq[j] - mean) / math.sqrt(variance + eps)
                normalized.append(gamma[j] * norm_val + beta[j])
            result.append(normalized)
        
        return result
    
    def tokenize(self, text: str) -> List[int]:
        """
        BPE风格的子词分词
        基于词频的合并算法：先将文本拆分为字符级token，
        然后迭代合并最高频的相邻token对，直到达到目标词表大小
        
        Args:
            text: 输入文本
            
        Returns:
            token ID列表
        """
        # 添加开始标记
        tokens = [49406]  # <start>
        
        # 截取有效文本长度（保留位置给start和end）
        effective_text = text[:self._max_length - 2]
        
        if not effective_text:
            # 空文本，直接添加结束标记
            tokens.append(49407)  # <end>
            while len(tokens) < self._max_length:
                tokens.append(0)  # <pad>
            return tokens[:self._max_length]
        
        # ---- 第一步：字符级分词 ----
        # 将文本拆分为字符序列，每个字符映射到一个基础token ID
        char_tokens = []
        for char in effective_text:
            # 使用字符的Unicode值映射到token空间
            # 保留特殊token区域 [0, 49406] 和 [49407, vocab_size)
            token_id = ord(char) % (self._vocab_size - 49407) + 49407
            char_tokens.append(token_id)
        
        # ---- 第二步：BPE合并 ----
        # 统计相邻token对的频率，迭代合并最高频的对
        # 合并后的新token ID通过哈希函数生成（模拟BPE词表查找）
        num_merge_iterations = 3  # 合并轮数，控制子词粒度
        
        for _ in range(num_merge_iterations):
            if len(char_tokens) < 2:
                break
            
            # 统计相邻token对的出现频率
            pair_counts = {}
            for k in range(len(char_tokens) - 1):
                pair = (char_tokens[k], char_tokens[k + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            if not pair_counts:
                break
            
            # 找到频率最高的token对
            best_pair = max(pair_counts, key=pair_counts.get)
            
            # 为合并后的token生成新ID（使用确定性哈希）
            merged_id = (best_pair[0] * 31 + best_pair[1] * 17) % (self._vocab_size - 49407) + 49407
            
            # 在token序列中执行合并
            new_tokens = []
            k = 0
            while k < len(char_tokens):
                if k < len(char_tokens) - 1 and char_tokens[k] == best_pair[0] and char_tokens[k + 1] == best_pair[1]:
                    new_tokens.append(merged_id)
                    k += 2  # 跳过已合并的两个token
                else:
                    new_tokens.append(char_tokens[k])
                    k += 1
            char_tokens = new_tokens
        
        # 将分词结果添加到输出
        tokens.extend(char_tokens)
        
        # 添加结束标记
        tokens.append(49407)  # <end>
        
        # 填充到最大长度
        while len(tokens) < self._max_length:
            tokens.append(0)  # <pad>
        
        return tokens[:self._max_length]
    
    def _build_attention_mask(self, tokens: List[int]) -> List[int]:
        """
        构建注意力掩码
        
        Args:
            tokens: token列表
            
        Returns:
            注意力掩码
        """
        mask = []
        for token in tokens:
            if token == 0:  # padding token
                mask.append(0)
            else:
                mask.append(1)
        return mask
    
    def batch_encode(self, texts: List[str]) -> List[List[List[float]]]:
        """批量编码文本"""
        return [self.encode(text) for text in texts]
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim
    
    @property
    def max_length(self) -> int:
        return self._max_length


class ImageEncoder:
    """
    图像编码器
    基于ViT风格的图像编码
    """
    
    def __init__(self, 
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 image_size: int = 224):
        """
        初始化图像编码器
        
        Args:
            patch_size: patch大小
            embed_dim: 嵌入维度
            image_size: 图像大小
        """
        self._patch_size = patch_size
        self._embed_dim = embed_dim
        self._image_size = image_size
        
        # patch数量
        self._num_patches = (image_size // patch_size) ** 2
        
        # 投影权重
        self._proj_weights: List[List[float]] = []
        self._init_projection()
        
        # CLS token
        self._cls_token = [math.sin(i * 0.1) for i in range(embed_dim)]
        
        # 位置嵌入
        self._pos_embedding: List[List[float]] = []
        self._init_position_embedding()
    
    def _init_projection(self) -> None:
        """初始化投影层"""
        patch_dim = self._patch_size * self._patch_size * 3
        for i in range(patch_dim):
            row = [math.sin((i + 1) * (j + 1) * 0.01) for j in range(self._embed_dim)]
            self._proj_weights.append(row)
    
    def _init_position_embedding(self) -> None:
        """初始化位置嵌入"""
        num_positions = self._num_patches + 1  # +1 for CLS token
        for pos in range(num_positions):
            embedding = []
            for i in range(self._embed_dim):
                if i % 2 == 0:
                    embedding.append(math.sin(pos / (10000 ** (i / self._embed_dim))))
                else:
                    embedding.append(math.cos(pos / (10000 ** ((i - 1) / self._embed_dim))))
            self._pos_embedding.append(embedding)
    
    def encode(self, image: List[List[List[float]]]) -> List[float]:
        """
        编码图像
        
        Args:
            image: 输入图像 [H, W, C]
            
        Returns:
            图像嵌入 [embed_dim]
        """
        # 分块
        patches = self._patchify(image)
        
        # 添加CLS token
        all_tokens = [self._cls_token] + patches
        
        # 添加位置嵌入
        embedded = []
        for i, token in enumerate(all_tokens):
            if i < len(self._pos_embedding):
                pos_emb = self._pos_embedding[i]
                combined = [token[j] + pos_emb[j] for j in range(self._embed_dim)]
                embedded.append(combined)
            else:
                embedded.append(token)
        
        # 简化的Transformer处理（平均池化）
        pooled = [0.0] * self._embed_dim
        for emb in embedded:
            for j in range(self._embed_dim):
                pooled[j] += emb[j]
        
        num_tokens = len(embedded)
        if num_tokens > 0:
            pooled = [pooled[j] / num_tokens for j in range(self._embed_dim)]
        
        return pooled
    
    def _patchify(self, image: List[List[List[float]]]) -> List[List[float]]:
        """
        图像分块
        
        Args:
            image: 输入图像 [H, W, C]
            
        Returns:
            patches列表 [num_patches, embed_dim]
        """
        h = len(image)
        w = len(image[0])
        c = len(image[0][0])
        
        patches = []
        
        for i in range(0, h - self._patch_size + 1, self._patch_size):
            for j in range(0, w - self._patch_size + 1, self._patch_size):
                # 提取patch
                patch = []
                for pi in range(self._patch_size):
                    for pj in range(self._patch_size):
                        for ch in range(c):
                            if i + pi < h and j + pj < w:
                                patch.append(image[i + pi][j + pj][ch])
                            else:
                                patch.append(0.0)
                
                # 投影到嵌入维度
                projected = self._positional_embedding([patch])
                patches.append(projected[0])
        
        return patches
    
    def _positional_embedding(self, patches: List[List[float]]) -> List[List[float]]:
        """
        位置嵌入（投影）
        
        Args:
            patches: patch列表
            
        Returns:
            嵌入后的patches
        """
        result = []
        patch_dim = len(patches[0]) if patches else 0
        
        for patch in patches:
            embedded = [0.0] * self._embed_dim
            
            # 简化的线性投影
            for i, val in enumerate(patch):
                if i < len(self._proj_weights):
                    for j in range(self._embed_dim):
                        embedded[j] += val * self._proj_weights[i][j]
            
            # 归一化
            norm = math.sqrt(sum(x * x for x in embedded) + 1e-6)
            embedded = [x / norm for x in embedded]
            
            result.append(embedded)
        
        return result
    
    @property
    def patch_size(self) -> int:
        return self._patch_size
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim


class AudioEncoder:
    """
    音频编码器
    基于MFCC和频谱特征的音频编码
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 feature_dim: int = 768,
                 n_mfcc: int = 13):
        """
        初始化音频编码器
        
        Args:
            sample_rate: 采样率
            feature_dim: 特征维度
            n_mfcc: MFCC系数数量
        """
        self._sample_rate = sample_rate
        self._feature_dim = feature_dim
        self._n_mfcc = n_mfcc
        
        # 投影权重
        self._proj_weights: List[List[float]] = []
        self._init_projection()
    
    def _init_projection(self) -> None:
        """初始化投影层"""
        for i in range(self._n_mfcc * 100):  # 假设最大100帧
            row = [math.sin((i + 1) * (j + 1) * 0.01) for j in range(self._feature_dim)]
            self._proj_weights.append(row)
    
    def encode(self, audio: List[float]) -> List[float]:
        """
        编码音频
        
        Args:
            audio: 音频波形 [num_samples]
            
        Returns:
            音频嵌入 [feature_dim]
        """
        # 计算MFCC特征
        mfcc = self._compute_mfcc(audio)
        
        # 计算频谱图
        spectrogram = self._compute_spectrogram(audio)
        
        # 融合特征
        combined = []
        for frame in mfcc:
            combined.extend(frame)
        
        # 投影到目标维度
        embedded = [0.0] * self._feature_dim
        for i, val in enumerate(combined):
            if i < len(self._proj_weights):
                for j in range(self._feature_dim):
                    embedded[j] += val * self._proj_weights[i][j]
        
        # 归一化
        norm = math.sqrt(sum(x * x for x in embedded) + 1e-6)
        embedded = [x / norm for x in embedded]
        
        return embedded
    
    def _compute_mfcc(self, audio: List[float]) -> List[List[float]]:
        """
        计算MFCC特征
        
        Args:
            audio: 音频波形
            
        Returns:
            MFCC特征 [num_frames, n_mfcc]
        """
        # 简化的MFCC计算
        frame_size = 400  # 25ms at 16kHz
        hop_size = 160    # 10ms hop
        
        mfcc_frames = []
        
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            
            # 预加重
            pre_emphasized = [frame[0]]
            for i in range(1, len(frame)):
                pre_emphasized.append(frame[i] - 0.97 * frame[i - 1])
            
            # 汉明窗
            windowed = []
            for i, val in enumerate(pre_emphasized):
                window = 0.54 - 0.46 * math.cos(2 * math.pi * i / (len(pre_emphasized) - 1))
                windowed.append(val * window)
            
            # 简化的DCT作为MFCC近似
            mfcc = []
            for k in range(self._n_mfcc):
                coeff = 0.0
                for n, val in enumerate(windowed):
                    coeff += val * math.cos(math.pi * k * (n + 0.5) / len(windowed))
                mfcc.append(coeff / len(windowed))
            
            mfcc_frames.append(mfcc)
        
        return mfcc_frames if mfcc_frames else [[0.0] * self._n_mfcc]
    
    def _compute_spectrogram(self, audio: List[float]) -> List[List[float]]:
        """
        计算频谱图
        
        Args:
            audio: 音频波形
            
        Returns:
            频谱图 [num_frames, num_freq_bins]
        """
        frame_size = 400
        hop_size = 160
        n_fft = 512
        
        spectrogram = []
        
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            
            # 补零
            padded = frame + [0.0] * (n_fft - len(frame))
            
            # 简化的DFT
            spectrum = []
            for k in range(n_fft // 2 + 1):
                real, imag = 0.0, 0.0
                for n, val in enumerate(padded):
                    angle = 2 * math.pi * k * n / n_fft
                    real += val * math.cos(angle)
                    imag -= val * math.sin(angle)
                magnitude = math.sqrt(real * real + imag * imag)
                spectrum.append(math.log(magnitude + 1e-6))
            
            spectrogram.append(spectrum)
        
        return spectrogram if spectrogram else [[0.0] * 257]
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim


class TemporalEncoder:
    """时序编码器"""
    
    def __init__(self, feature_dim: int = 768, num_layers: int = 2):
        self._feature_dim = feature_dim
        self._num_layers = num_layers
    
    def encode(self, frame_features: List[List[float]]) -> List[List[float]]:
        """编码时序信息"""
        seq_len = len(frame_features)
        if seq_len == 0:
            return []
        
        # 添加时序位置编码
        result = []
        for t, feat in enumerate(frame_features):
            pos_enc = []
            for i in range(self._feature_dim):
                if i % 2 == 0:
                    pos_enc.append(math.sin(t / (10000 ** (i / self._feature_dim))))
                else:
                    pos_enc.append(math.cos(t / (10000 ** ((i - 1) / self._feature_dim))))
            
            combined = [feat[j] + pos_enc[j] for j in range(min(len(feat), self._feature_dim))]
            result.append(combined)
        
        return result


class VideoEncoder:
    """
    视频编码器
    基于帧编码和时序建模的视频编码
    """
    
    def __init__(self, 
                 feature_dim: int = 768,
                 patch_size: int = 16):
        """
        初始化视频编码器
        
        Args:
            feature_dim: 特征维度
            patch_size: patch大小
        """
        self._feature_dim = feature_dim
        
        self._frame_encoder = ImageEncoder(patch_size=patch_size, embed_dim=feature_dim)
        self._temporal_encoder = TemporalEncoder(feature_dim=feature_dim)
    
    def encode(self, video_frames: List[List[List[List[float]]]]) -> List[float]:
        """
        编码视频
        
        Args:
            video_frames: 视频帧列表 [num_frames, H, W, C]
            
        Returns:
            视频嵌入 [feature_dim]
        """
        if not video_frames:
            return [0.0] * self._feature_dim
        
        # 编码每一帧
        frame_features = []
        for frame in video_frames:
            feat = self._frame_encoder.encode(frame)
            frame_features.append(feat)
        
        # 时序编码
        temporal_features = self._temporal_encoder.encode(frame_features)
        
        # 时序池化
        pooled = self._temporal_pooling(temporal_features)
        
        return pooled
    
    def _temporal_pooling(self, frame_features: List[List[float]]) -> List[float]:
        """
        时序池化
        
        Args:
            frame_features: 帧特征列表
            
        Returns:
            池化后的特征
        """
        if not frame_features:
            return [0.0] * self._feature_dim
        
        # 平均池化
        pooled = [0.0] * self._feature_dim
        num_frames = len(frame_features)
        
        for feat in frame_features:
            for j in range(min(len(feat), self._feature_dim)):
                pooled[j] += feat[j]
        
        if num_frames > 0:
            pooled = [pooled[j] / num_frames for j in range(self._feature_dim)]
        
        return pooled
    
    def encode_with_temporal(self, 
                             video_frames: List[List[List[List[float]]]]) -> List[List[float]]:
        """编码视频并保留时序信息"""
        if not video_frames:
            return []
        
        frame_features = []
        for frame in video_frames:
            feat = self._frame_encoder.encode(frame)
            frame_features.append(feat)
        
        return self._temporal_encoder.encode(frame_features)
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim


class LensController:
    """
    镜头控制器
    用于编码镜头脚本和生成镜头序列
    """
    
    def __init__(self):
        """初始化镜头控制器"""
        self._lens_types = [
            "远景", "全景", "中景", "近景", "特写", "大特写"
        ]
        
        self._motion_types = [
            "推", "拉", "摇", "移", "跟", "升降", "旋转", "固定"
        ]
        
        # 镜头类型编码
        self._lens_embeddings: Dict[str, List[float]] = {}
        self._init_lens_embeddings()
        
        # 运动类型编码
        self._motion_embeddings: Dict[str, List[float]] = {}
        self._init_motion_embeddings()
    
    def _init_lens_embeddings(self) -> None:
        """初始化镜头类型嵌入"""
        dim = 64
        for i, lens_type in enumerate(self._lens_types):
            embedding = [0.0] * dim
            embedding[i * 10 % dim] = 1.0
            embedding[(i * 10 + 5) % dim] = 0.5
            self._lens_embeddings[lens_type] = embedding
    
    def _init_motion_embeddings(self) -> None:
        """初始化运动类型嵌入"""
        dim = 64
        for i, motion_type in enumerate(self._motion_types):
            embedding = [0.0] * dim
            embedding[i * 8 % dim] = 1.0
            self._motion_embeddings[motion_type] = embedding
    
    def encode_lens_script(self, script: str) -> List[float]:
        """
        编码镜头脚本
        
        Args:
            script: 镜头脚本文本
            
        Returns:
            镜头嵌入
        """
        # 解析脚本
        lens_type = None
        motion_type = None
        params = {}
        
        # 简单解析
        for lt in self._lens_types:
            if lt in script:
                lens_type = lt
                break
        
        for mt in self._motion_types:
            if mt in script:
                motion_type = mt
                break
        
        # 编码
        embedding = [0.0] * 128
        
        if lens_type and lens_type in self._lens_embeddings:
            lens_emb = self._lens_embeddings[lens_type]
            for i, val in enumerate(lens_emb):
                embedding[i] = val
        
        if motion_type and motion_type in self._motion_embeddings:
            motion_emb = self._motion_embeddings[motion_type]
            for i, val in enumerate(motion_emb):
                embedding[64 + i] = val
        
        return embedding
    
    def _encode_lens_type(self, lens_type: str) -> List[float]:
        """
        编码镜头类型
        
        Args:
            lens_type: 镜头类型
            
        Returns:
            镜头类型嵌入
        """
        return self._lens_embeddings.get(lens_type, [0.0] * 64)
    
    def _encode_motion(self, motion: str, params: Dict[str, float]) -> List[float]:
        """
        编码运动参数
        
        Args:
            motion: 运动类型
            params: 运动参数
            
        Returns:
            运动嵌入
        """
        base_emb = self._motion_embeddings.get(motion, [0.0] * 64)
        
        # 添加参数编码
        result = base_emb.copy()
        
        # 编码速度、持续时间等参数
        if 'speed' in params:
            result[0] = params['speed']
        if 'duration' in params:
            result[1] = params['duration'] / 10.0  # 归一化
        if 'angle' in params:
            result[2] = params['angle'] / 360.0  # 归一化
        
        return result
    
    def generate_lens_sequence(self, 
                               story: str, 
                               num_shots: int) -> List[Dict[str, Any]]:
        """
        生成镜头序列
        
        Args:
            story: 故事描述
            num_shots: 镜头数量
            
        Returns:
            镜头序列列表
        """
        shots = []
        
        # 简化的镜头生成逻辑
        lens_progression = [
            ("远景", "固定"),
            ("全景", "推"),
            ("中景", "摇"),
            ("近景", "跟"),
            ("特写", "固定"),
        ]
        
        for i in range(num_shots):
            # 选择镜头类型和运动
            idx = i % len(lens_progression)
            lens_type, motion = lens_progression[idx]
            
            # 根据故事调整
            if "动作" in story:
                motion = self._motion_types[(i + 2) % len(self._motion_types)]
            elif "情感" in story or "情绪" in story:
                lens_type = "特写" if i % 2 == 0 else "近景"
            
            shot = {
                'shot_id': i,
                'lens_type': lens_type,
                'motion': motion,
                'duration': 3.0 + (i % 3),  # 3-5秒
                'lens_embedding': self._encode_lens_type(lens_type),
                'motion_embedding': self._encode_motion(motion, {}),
                'description': f"镜头{i+1}: {lens_type}, {motion}"
            }
            
            shots.append(shot)
        
        return shots
    
    @property
    def lens_types(self) -> List[str]:
        return self._lens_types.copy()
    
    @property
    def motion_types(self) -> List[str]:
        return self._motion_types.copy()


class TrajectoryProjector:
    """
    轨迹投影器
    将轨迹点投影到条件空间
    """
    
    def __init__(self, input_dim: int = 3, output_dim: int = 768):
        """
        初始化轨迹投影器
        
        Args:
            input_dim: 输入维度 (x, y, z)
            output_dim: 输出维度
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        
        # 投影权重
        self._weights: List[List[float]] = []
        self._bias: List[float] = []
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化投影权重"""
        for i in range(self._input_dim):
            row = []
            for j in range(self._output_dim):
                # 使用正弦初始化
                row.append(math.sin((i + 1) * (j + 1) * 0.01) * 0.1)
            self._weights.append(row)
        
        self._bias = [0.0] * self._output_dim
    
    def project(self, trajectory: List[List[float]]) -> List[List[float]]:
        """
        投影轨迹到条件空间
        
        Args:
            trajectory: 轨迹点列表 [[x, y, z], ...]
            
        Returns:
            投影后的特征 [num_points, output_dim]
        """
        result = []
        
        for point in trajectory:
            # 确保维度正确
            padded_point = point + [0.0] * max(0, self._input_dim - len(point))
            padded_point = padded_point[:self._input_dim]
            
            # 线性投影
            projected = self._bias.copy()
            for i, val in enumerate(padded_point):
                for j in range(self._output_dim):
                    projected[j] += val * self._weights[i][j]
            
            # 激活函数 (tanh)
            projected = [math.tanh(x) for x in projected]
            
            result.append(projected)
        
        return result
    
    def _interpolate_trajectory(self, 
                                points: List[List[float]], 
                                num_points: int) -> List[List[float]]:
        """
        轨迹插值
        
        Args:
            points: 原始轨迹点
            num_points: 目标点数
            
        Returns:
            插值后的轨迹
        """
        if len(points) < 2:
            return points
        
        if num_points <= len(points):
            return points[:num_points]
        
        result = []
        dim = len(points[0])
        
        # 计算累积距离
        distances = [0.0]
        for i in range(1, len(points)):
            dist = math.sqrt(sum((points[i][j] - points[i-1][j])**2 for j in range(dim)))
            distances.append(distances[-1] + dist)
        
        total_dist = distances[-1]
        if total_dist == 0:
            return points
        
        # 等距采样
        for i in range(num_points):
            target_dist = i * total_dist / (num_points - 1)
            
            # 找到对应的区间
            for j in range(len(distances) - 1):
                if distances[j] <= target_dist <= distances[j + 1]:
                    segment_dist = distances[j + 1] - distances[j]
                    if segment_dist > 0:
                        t = (target_dist - distances[j]) / segment_dist
                    else:
                        t = 0
                    
                    # 线性插值
                    point = []
                    for d in range(dim):
                        val = points[j][d] * (1 - t) + points[j + 1][d] * t
                        point.append(val)
                    result.append(point)
                    break
        
        return result
    
    def project_with_interpolation(self, 
                                   trajectory: List[List[float]], 
                                   num_points: int) -> List[List[float]]:
        """投影并插值轨迹"""
        interpolated = self._interpolate_trajectory(trajectory, num_points)
        return self.project(interpolated)
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class MultimodalFusion:
    """
    多模态融合
    融合文本、图像、音频、视频等多模态特征
    """
    
    def __init__(self, output_dim: int = 768):
        """
        初始化多模态融合
        
        Args:
            output_dim: 输出维度
        """
        self._output_dim = output_dim
        
        # 各模态投影层
        self._text_proj = self._create_projection(768, output_dim)
        self._image_proj = self._create_projection(768, output_dim)
        self._audio_proj = self._create_projection(768, output_dim)
        self._video_proj = self._create_projection(768, output_dim)
        
        # 融合权重
        self._fusion_weights = {
            'text': 0.4,
            'image': 0.3,
            'audio': 0.15,
            'video': 0.15
        }
    
    def _create_projection(self, input_dim: int, output_dim: int) -> List[List[float]]:
        """创建投影矩阵"""
        proj = []
        for i in range(input_dim):
            row = [math.sin((i + 1) * (j + 1) * 0.01) * 0.1 for j in range(output_dim)]
            proj.append(row)
        return proj
    
    def _apply_projection(self, features: List[float], proj: List[List[float]]) -> List[float]:
        """应用投影"""
        output_dim = len(proj[0]) if proj else self._output_dim
        result = [0.0] * output_dim
        
        for i, val in enumerate(features):
            if i < len(proj):
                for j in range(output_dim):
                    result[j] += val * proj[i][j]
        
        return result
    
    def fuse(self, 
             text_feat: Optional[List[float]] = None,
             image_feat: Optional[List[float]] = None,
             audio_feat: Optional[List[float]] = None,
             video_feat: Optional[List[float]] = None) -> List[float]:
        """
        融合多模态特征
        
        Args:
            text_feat: 文本特征
            image_feat: 图像特征
            audio_feat: 音频特征
            video_feat: 视频特征
            
        Returns:
            融合后的特征
        """
        # 投影各模态特征
        projected = {}
        
        if text_feat:
            projected['text'] = self._apply_projection(text_feat, self._text_proj)
        if image_feat:
            projected['image'] = self._apply_projection(image_feat, self._image_proj)
        if audio_feat:
            projected['audio'] = self._apply_projection(audio_feat, self._audio_proj)
        if video_feat:
            projected['video'] = self._apply_projection(video_feat, self._video_proj)
        
        if not projected:
            return [0.0] * self._output_dim
        
        # 加权融合
        result = [0.0] * self._output_dim
        total_weight = 0.0
        
        for modality, feat in projected.items():
            weight = self._fusion_weights.get(modality, 0.25)
            for j in range(min(len(feat), self._output_dim)):
                result[j] += weight * feat[j]
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            result = [r / total_weight for r in result]
        
        return result
    
    def _cross_modal_attention(self, 
                               query: List[float], 
                               contexts: List[List[float]]) -> List[float]:
        """
        跨模态注意力
        
        Args:
            query: 查询向量
            contexts: 上下文向量列表
            
        Returns:
            注意力加权的结果
        """
        if not contexts:
            return query
        
        dim = len(query)
        
        # 计算注意力分数
        scores = []
        for ctx in contexts:
            score = sum(q * c for q, c in zip(query, ctx[:dim]))
            scores.append(score)
        
        # Softmax
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        attention = [e / sum_exp for e in exp_scores]
        
        # 加权求和
        result = [0.0] * dim
        for i, ctx in enumerate(contexts):
            for j in range(min(len(ctx), dim)):
                result[j] += attention[i] * ctx[j]
        
        return result
    
    def fuse_with_attention(self, 
                           text_feat: Optional[List[float]] = None,
                           image_feat: Optional[List[float]] = None,
                           audio_feat: Optional[List[float]] = None,
                           video_feat: Optional[List[float]] = None) -> List[float]:
        """使用跨模态注意力融合"""
        # 收集所有特征
        contexts = []
        
        if text_feat:
            contexts.append(self._apply_projection(text_feat, self._text_proj))
        if image_feat:
            contexts.append(self._apply_projection(image_feat, self._image_proj))
        if audio_feat:
            contexts.append(self._apply_projection(audio_feat, self._audio_proj))
        if video_feat:
            contexts.append(self._apply_projection(video_feat, self._video_proj))
        
        if not contexts:
            return [0.0] * self._output_dim
        
        # 使用平均作为查询
        query = [0.0] * self._output_dim
        for ctx in contexts:
            for j in range(min(len(ctx), self._output_dim)):
                query[j] += ctx[j]
        query = [q / len(contexts) for q in query]
        
        # 跨模态注意力
        return self._cross_modal_attention(query, contexts)
    
    def set_fusion_weights(self, 
                          text: float = 0.4,
                          image: float = 0.3,
                          audio: float = 0.15,
                          video: float = 0.15) -> None:
        """设置融合权重"""
        total = text + image + audio + video
        if total > 0:
            self._fusion_weights = {
                'text': text / total,
                'image': image / total,
                'audio': audio / total,
                'video': video / total
            }
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


# 辅助函数
def create_multimodal_encoder(config: Dict[str, Any]) -> MultimodalFusion:
    """
    创建多模态编码器
    
    Args:
        config: 配置字典
        
    Returns:
        MultimodalFusion实例
    """
    output_dim = config.get('output_dim', 768)
    return MultimodalFusion(output_dim=output_dim)


def encode_prompt_with_modality(text: str, 
                                image: Optional[List[List[List[float]]]] = None,
                                encoder: Optional[MultimodalFusion] = None) -> List[float]:
    """
    编码带有多模态信息的提示
    
    Args:
        text: 文本提示
        image: 可选的图像
        encoder: 多模态编码器
        
    Returns:
        融合后的嵌入
    """
    if encoder is None:
        encoder = MultimodalFusion()
    
    text_encoder = TextEncoder()
    text_feat = text_encoder.encode(text)
    
    # 平均池化文本特征
    if text_feat:
        text_pooled = [0.0] * len(text_feat[0])
        for feat in text_feat:
            for j, val in enumerate(feat):
                text_pooled[j] += val
        text_pooled = [v / len(text_feat) for v in text_pooled]
    else:
        text_pooled = [0.0] * 768
    
    # 编码图像
    image_feat = None
    if image:
        image_encoder = ImageEncoder()
        image_feat = image_encoder.encode(image)
    
    return encoder.fuse(text_feat=text_pooled, image_feat=image_feat)
