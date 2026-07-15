"""
AnimateDiff Video Generation - Pure Python Implementation
=========================================================
纯Python实现的AnimateDiff视频生成管线，包含运动模块、AnimateDiff管线和SVD管线。
仅使用标准库，不依赖外部库。
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Any


# ===========================================================================
# 工具函数（与sd_pipeline共享逻辑，独立定义以保持文件独立性）
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
    max_val = max(x) if x else 0.0
    exps = [math.exp(v - max_val) for v in x]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(x)] * len(x)
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


def _layer_norm_1d(x: List[float], eps: float = 1e-5) -> List[float]:
    """一维层归一化"""
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


def _linear(x: List[float], weight: List[List[float]], bias: List[float]) -> List[float]:
    """线性变换 y = xW^T + b"""
    w_t = _transpose(weight)
    result = _matmul([x], w_t)[0]
    return [r + b[i] for i, r in enumerate(result)]


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


def _tensor_add(a: List, b: List) -> List:
    """张量逐元素加法"""
    if isinstance(a[0], list):
        return [_tensor_add(ai, bi) for ai, bi in zip(a, b)]
    return [ai + bi for ai, bi in zip(a, b)]


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


# ===========================================================================
# 1. MotionModule - 运动模块
# ===========================================================================

class MotionModule:
    """
    运动模块(Motion Module)。
    在UNet中注入时序注意力层，使静态图像扩散模型具备视频生成能力。
    核心思想：在UNet的每个分辨率层级插入时序注意力，建模帧间运动。
    """

    def __init__(self, num_frames: int = 16, temporal_attention: bool = True,
                 num_attention_heads: int = 8, motion_module_dim: int = 320,
                 seed: Optional[int] = 42):
        self._num_frames = num_frames
        self._temporal_attention = temporal_attention
        self._num_heads = num_attention_heads
        self._motion_module_dim = motion_module_dim
        self._seed = seed
        rng = random.Random(seed)

        # 时序注意力权重
        head_dim = motion_module_dim // num_attention_heads
        self._temporal_q = [[rng.gauss(0, 0.02) for _ in range(motion_module_dim)]
                            for _ in range(motion_module_dim)]
        self._temporal_k = [[rng.gauss(0, 0.02) for _ in range(motion_module_dim)]
                            for _ in range(motion_module_dim)]
        self._temporal_v = [[rng.gauss(0, 0.02) for _ in range(motion_module_dim)]
                            for _ in range(motion_module_dim)]
        self._temporal_out = [[rng.gauss(0, 0.02) for _ in range(motion_module_dim)]
                              for _ in range(motion_module_dim)]

        # 层归一化参数
        self._ln_w = [1.0] * motion_module_dim
        self._ln_b = [0.0] * motion_module_dim

        # 前馈网络权重
        ff_dim = motion_module_dim * 4
        self._ff1 = [[rng.gauss(0, 0.02) for _ in range(ff_dim)]
                     for _ in range(motion_module_dim)]
        self._ff2 = [[rng.gauss(0, 0.02) for _ in range(motion_module_dim)]
                     for _ in range(ff_dim)]

        # 运动插值权重（用于帧间平滑）
        self._motion_scale = 1.0
        self._motion_bias = [0.0] * motion_module_dim

    def _temporal_attention_block(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        时序注意力块。
        对视频帧序列在时间维度上执行自注意力。
        输入形状: [num_frames, spatial_dim, feature_dim]
        输出形状: [num_frames, spatial_dim, feature_dim]

        算法:
        1. 对每一帧的每个空间位置，收集所有帧同一位置的特征
        2. 计算Q, K, V
        3. 执行缩放点积注意力
        4. 加权聚合帧间信息
        """
        num_frames = len(x)
        if num_frames <= 1:
            return x

        spatial_dim = len(x[0])
        feat_dim = len(x[0][0])
        scale = math.sqrt(feat_dim)

        # 对每个空间位置执行时序注意力
        output = [[[] for _ in range(spatial_dim)] for _ in range(num_frames)]

        for s in range(spatial_dim):
            # 收集所有帧在位置s的特征
            frame_features = [x[f][s] for f in range(num_frames)]

            # 计算Q, K, V
            queries = [_linear(feat, self._temporal_q, [0.0] * feat_dim) for feat in frame_features]
            keys = [_linear(feat, self._temporal_k, [0.0] * feat_dim) for feat in frame_features]
            values = [_linear(feat, self._temporal_v, [0.0] * feat_dim) for feat in frame_features]

            # 计算注意力分数: QK^T / sqrt(d)
            attn_scores = []
            for f in range(num_frames):
                row = []
                for f2 in range(num_frames):
                    dot = sum(queries[f][d] * keys[f2][d] for d in range(feat_dim)) / scale
                    row.append(dot)
                attn_scores.append(row)

            # Softmax归一化
            attn_weights = [_softmax(row) for row in attn_scores]

            # 加权求和
            for f in range(num_frames):
                attended = [0.0] * feat_dim
                for f2 in range(num_frames):
                    for d in range(feat_dim):
                        attended[d] += attn_weights[f][f2] * values[f2][d]

                # 输出投影
                projected = _linear(attended, self._temporal_out, [0.0] * feat_dim)

                # 残差连接
                residual = [x[f][s][d] + projected[d] * self._motion_scale
                            for d in range(feat_dim)]
                output[f][s] = residual

        return output

    def _temporal_feed_forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        时序前馈网络。
        对每帧特征独立应用FFN，但权重在帧间共享。
        """
        num_frames = len(x)
        spatial_dim = len(x[0]) if num_frames > 0 else 0
        feat_dim = len(x[0][0]) if spatial_dim > 0 else 0

        output = [[[] for _ in range(spatial_dim)] for _ in range(num_frames)]

        for f in range(num_frames):
            for s in range(spatial_dim):
                feat = x[f][s]
                # FFN: GELU(xW1)W2
                hidden = _linear(feat, self._ff1, [0.0] * (feat_dim * 4))
                hidden = [_gelu(h) for h in hidden]
                out = _linear(hidden, self._ff2, [0.0] * feat_dim)
                # 残差连接
                output[f][s] = [feat[d] + out[d] for d in range(feat_dim)]

        return output

    def _layer_norm_temporal(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """对时序特征进行层归一化"""
        eps = 1e-5
        num_frames = len(x)
        spatial_dim = len(x[0]) if num_frames > 0 else 0
        feat_dim = len(x[0][0]) if spatial_dim > 0 else 0

        output = [[[] for _ in range(spatial_dim)] for _ in range(num_frames)]

        for f in range(num_frames):
            for s in range(spatial_dim):
                feat = x[f][s]
                mean = sum(feat) / len(feat)
                var = sum((v - mean) ** 2 for v in feat) / len(feat)
                std = math.sqrt(var + eps)
                normalized = [(feat[d] - mean) / std * self._ln_w[d] + self._ln_b[d]
                              for d in range(feat_dim)]
                output[f][s] = normalized

        return output

    def _motion_modules_insert(self, unet_blocks: List[Dict]) -> List[Dict]:
        """
        在UNet块中插入运动模块。
        返回修改后的UNet块配置列表。
        在每个下采样和上采样块之间插入时序注意力。
        """
        augmented_blocks = []
        for i, block in enumerate(unet_blocks):
            augmented_blocks.append(block)
            # 在特定位置插入运动模块
            if block.get("type") in ("down", "up", "mid"):
                augmented_blocks.append({
                    "type": "temporal",
                    "motion_module": self,
                    "inserted_after": i,
                    "num_frames": self._num_frames,
                })
        return augmented_blocks

    def forward(self, x: List[List[List[float]]],
                context: Optional[List[List[float]]] = None) -> List[List[List[float]]]:
        """
        运动模块前向传播。

        Args:
            x: 输入特征 [num_frames, spatial_dim, feature_dim]
            context: 可选的文本条件 [seq_len, embed_dim]

        Returns:
            输出特征 [num_frames, spatial_dim, feature_dim]
        """
        if not self._temporal_attention:
            return x

        # 层归一化
        x = self._layer_norm_temporal(x)

        # 时序自注意力
        x = self._temporal_attention_block(x)

        # 层归一化
        x = self._layer_norm_temporal(x)

        # 前馈网络
        x = self._temporal_feed_forward(x)

        return x

    def interpolate_motion(self, frames: List[List[List[List[int]]]],
                           target_fps: int = 8) -> List[List[List[List[int]]]]:
        """
        运动插值：在帧间生成中间帧以实现平滑过渡。

        使用线性插值在相邻帧之间生成新帧。

        Args:
            frames: 原始帧序列 [num_frames, H, W, C]
            target_fps: 目标帧率

        Returns:
            插值后的帧序列
        """
        if len(frames) <= 1:
            return frames

        h = len(frames[0])
        w = len(frames[0][0]) if h > 0 else 0
        c = len(frames[0][0][0]) if w > 0 else 0

        # 在每对相邻帧之间插入一帧
        interpolated = []
        for i in range(len(frames) - 1):
            interpolated.append(frames[i])

            # 线性插值生成中间帧
            mid_frame = [[[0] * c for _ in range(w)] for _ in range(h)]
            for py in range(h):
                for px in range(w):
                    for ch in range(c):
                        val = (frames[i][py][px][ch] + frames[i + 1][py][px][ch]) // 2
                        mid_frame[py][px][ch] = val
            interpolated.append(mid_frame)

        interpolated.append(frames[-1])
        return interpolated

    def compute_motion_flow(self, frame1: List[List[List[int]]],
                            frame2: List[List[List[int]]]) -> List[List[Tuple[float, float]]]:
        """
        计算两帧之间的光流（简化版）。
        使用块匹配算法估计运动向量。

        Args:
            frame1: 第一帧 [H, W, C]
            frame2: 第二帧 [H, W, C]

        Returns:
            光流场 [H//block_size, W//block_size, (dx, dy)]
        """
        h = len(frame1)
        w = len(frame1[0]) if h > 0 else 0
        block_size = 8
        search_range = 4

        flow_h = h // block_size
        flow_w = w // block_size
        flow = [[(0.0, 0.0) for _ in range(flow_w)] for _ in range(flow_h)]

        for by in range(flow_h):
            for bx in range(flow_w):
                # 提取当前块
                best_dx, best_dy = 0, 0
                best_sad = float('inf')

                y0 = by * block_size
                x0 = bx * block_size

                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        sad = 0
                        for py in range(block_size):
                            for px in range(block_size):
                                fy1 = y0 + py
                                fx1 = x0 + px
                                fy2 = fy1 + dy
                                fx2 = fx1 + dx

                                if (0 <= fy1 < h and 0 <= fx1 < w and
                                        0 <= fy2 < h and 0 <= fx2 < w):
                                    # 灰度值比较
                                    g1 = sum(frame1[fy1][fx1]) // 3 if isinstance(frame1[fy1][fx1], list) else frame1[fy1][fx1]
                                    g2 = sum(frame2[fy2][fx2]) // 3 if isinstance(frame2[fy2][fx2], list) else frame2[fy2][fx2]
                                    sad += abs(g1 - g2)

                        if sad < best_sad:
                            best_sad = sad
                            best_dx = dx
                            best_dy = dy

                flow[by][bx] = (float(best_dx), float(best_dy))

        return flow


# ===========================================================================
# 2. AnimateDiffPipeline - AnimateDiff管线
# ===========================================================================

class AnimateDiffPipeline:
    """
    AnimateDiff视频生成管线。
    在预训练的Stable Diffusion模型基础上注入运动模块，实现文本到视频生成。
    核心思路：冻结SD权重，仅训练运动模块的时序注意力层。
    """

    def __init__(self, num_frames: int = 16, fps: int = 8, seed: Optional[int] = None):
        self._num_frames = num_frames
        self._fps = fps
        self._seed = seed
        self._motion_module = MotionModule(
            num_frames=num_frames,
            temporal_attention=True,
            seed=seed or 42
        )

        # 延迟导入SD管线组件（避免循环依赖）
        self._text_encoder = None
        self._unet = None
        self._vae = None
        self._scheduler = None
        self._initialized = False

    def _init_temporal_layers(self) -> None:
        """
        初始化时序层和SD管线组件。
        使用简化版本以保持纯Python实现。
        """
        if self._initialized:
            return

        # 初始化简化的文本编码器
        self._text_encoder = self._create_simple_text_encoder()

        # 初始化简化的UNet
        self._unet = self._create_simple_unet()

        # 初始化简化的VAE
        self._vae = self._create_simple_vae()

        # 初始化DDIM调度器
        self._scheduler = self._create_simple_scheduler()

        self._initialized = True

    def _create_simple_text_encoder(self) -> Dict:
        """创建简化的文本编码器配置"""
        rng = random.Random(self._seed or 42)
        embed_dim = 768
        return {
            "embed_dim": embed_dim,
            "weights": [[rng.gauss(0, 0.02) for _ in range(embed_dim)]
                        for _ in range(embed_dim)],
        }

    def _create_simple_unet(self) -> Dict:
        """创建简化的UNet配置"""
        rng = random.Random(self._seed or 42)
        return {
            "in_channels": 4,
            "out_channels": 4,
            "time_embed_dim": 256,
            "weights": [[rng.gauss(0, 0.02) for _ in range(4)] for _ in range(4)],
        }

    def _create_simple_vae(self) -> Dict:
        """创建简化的VAE配置"""
        return {
            "latent_channels": 4,
            "out_channels": 3,
        }

    def _create_simple_scheduler(self) -> Dict:
        """创建简化的DDIM调度器"""
        num_train_timesteps = 1000
        betas = [
            0.00085 + (0.012 - 0.00085) * i / (num_train_timesteps - 1)
            for i in range(num_train_timesteps)
        ]
        alphas = [1.0 - b for b in betas]
        alpha_cumprod = []
        cumprod = 1.0
        for a in alphas:
            cumprod *= a
            alpha_cumprod.append(cumprod)
        return {
            "num_train_timesteps": num_train_timesteps,
            "alpha_cumprod": alpha_cumprod,
            "timesteps": [],
        }

    def _encode_video_prompt(self, prompt: str) -> List[List[float]]:
        """
        编码视频提示词。
        在文本编码基础上添加运动相关的位置编码。

        Args:
            prompt: 文本提示词

        Returns:
            文本嵌入 [seq_len, embed_dim]
        """
        self._init_temporal_layers()

        # 简化的文本编码
        embed_dim = self._text_encoder["embed_dim"]
        seq_len = 16  # 简化的序列长度

        # 使用哈希生成确定性的词嵌入
        prompt_hash = hash(prompt)
        rng = random.Random(prompt_hash)

        embeddings = []
        for i in range(seq_len):
            row = [rng.gauss(0, 0.1) for _ in range(embed_dim)]
            # 添加位置编码
            for d in range(embed_dim):
                if d % 2 == 0:
                    row[d] += math.sin(i / (10000 ** (d / embed_dim)))
                else:
                    row[d] += math.cos(i / (10000 ** ((d - 1) / embed_dim)))
            embeddings.append(row)

        return embeddings

    def _random_video_latents(self, shape: List[int],
                              seed: Optional[int] = None) -> List[List[List[float]]]:
        """
        生成随机视频潜变量。
        形状: [num_frames, batch_size, channels]
        """
        actual_seed = seed if seed is not None else self._seed
        return _randn(shape, seed=actual_seed)

    def _unet_forward_simple(self, x: List[float], t: int,
                              text_embeds: List[List[float]]) -> List[float]:
        """
        简化的UNet前向传播（单帧）。
        """
        rng = random.Random(t + hash(str(x[:4])))

        # 时间嵌入
        half_dim = 128
        time_emb = []
        for i in range(half_dim):
            freq = math.exp(i * math.log(10000) / (half_dim - 1))
            time_emb.append(math.sin(t * freq))
        for i in range(half_dim):
            freq = math.exp(i * math.log(10000) / (half_dim - 1))
            time_emb.append(math.cos(t * freq))

        # 简化的噪声预测
        text_mean = []
        dim = min(len(x), len(time_emb), len(text_embeds[0]) if text_embeds else len(time_emb))
        for d in range(dim):
            t_val = time_emb[d] if d < len(time_emb) else 0.0
            x_val = x[d] if d < len(x) else 0.0
            text_val = sum(text_embeds[k][d] for k in range(min(len(text_embeds), 4))) / max(len(text_embeds), 1) if text_embeds else 0.0
            text_mean.append(t_val * 0.1 + x_val * 0.8 + text_val * 0.1)

        # 输出维度对齐
        while len(text_mean) < len(x):
            text_mean.append(rng.gauss(0, 0.01))
        return text_mean[:len(x)]

    def _apply_temporal_attention(self, latents: List[List[List[float]]],
                                   text_embeds: List[List[float]]) -> List[List[List[float]]]:
        """
        对视频潜变量应用时序注意力。
        """
        num_frames = len(latents)
        if num_frames <= 1:
            return latents

        # 将每帧的潜变量展平为特征序列
        batch_size = len(latents[0])
        channels = len(latents[0][0]) if batch_size > 0 else 0

        # 构建时序注意力输入 [num_frames, batch_size * channels]
        temporal_input = []
        for f in range(num_frames):
            flat = []
            for b in range(batch_size):
                flat.extend(latents[f][b])
            temporal_input.append(flat)

        # 通过运动模块
        # 调整为 [num_frames, 1, feature_dim] 格式
        feat_dim = len(temporal_input[0])
        motion_input = [[temporal_input[f]] for f in range(num_frames)]
        motion_output = self._motion_module.forward(motion_input, text_embeds)

        # 解包回 [num_frames, batch_size, channels]
        result = []
        for f in range(num_frames):
            flat = motion_output[f][0]
            frame = []
            idx = 0
            for b in range(batch_size):
                ch = flat[idx:idx + channels]
                if len(ch) < channels:
                    ch = ch + [0.0] * (channels - len(ch))
                frame.append(ch[:channels])
                idx += channels
            result.append(frame)

        return result

    def _denoise_video(self, latents: List[List[List[float]]],
                       text_embeds: List[List[float]],
                       steps: int) -> List[List[List[float]]]:
        """
        视频去噪循环。
        在每个时间步对所有帧同时去噪，并应用时序注意力。

        Args:
            latents: 视频潜变量 [num_frames, batch, channels]
            text_embeds: 文本嵌入
            steps: 去噪步数

        Returns:
            去噪后的视频潜变量
        """
        scheduler = self._scheduler
        alpha_cumprod = scheduler["alpha_cumprod"]
        num_train = scheduler["num_train_timesteps"]

        # 设置推理时间步
        step_size = num_train // steps
        timesteps = [num_train - 1 - i * step_size for i in range(steps)]

        current = [[[v for v in row] for row in frame] for frame in latents]

        for step_idx, t in enumerate(timesteps):
            alpha_t = alpha_cumprod[t]
            sqrt_alpha_t = math.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = math.sqrt(1.0 - alpha_t)

            # 前一步
            prev_t = t - step_size
            if prev_t < 0:
                alpha_prev = 1.0
            else:
                alpha_prev = alpha_cumprod[prev_t]
            sqrt_alpha_prev = math.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = math.sqrt(1.0 - alpha_prev)

            # 对每一帧预测噪声
            noise_preds = []
            for f in range(len(current)):
                for b in range(len(current[f])):
                    noise_pred = self._unet_forward_simple(current[f][b], t, text_embeds)
                    noise_preds.append((f, b, noise_pred))

            # 应用时序注意力（每几步一次以节省计算）
            if step_idx % 5 == 0:
                current = self._apply_temporal_attention(current, text_embeds)

            # DDIM更新每帧
            for f, b, noise_pred in noise_preds:
                sample = current[f][b]
                # 预测x_0
                pred_x0 = [(sample[j] - sqrt_one_minus_alpha_t * noise_pred[j]) / sqrt_alpha_t
                           for j in range(len(sample))]
                # 更新
                new_sample = [sqrt_alpha_prev * pred_x0[j] +
                              sqrt_one_minus_alpha_prev * sqrt_one_minus_alpha_t * noise_pred[j]
                              for j in range(len(sample))]
                current[f][b] = new_sample

        return current

    def _latent_to_frame(self, latent: List[float], height: int = 64,
                         width: int = 64) -> List[List[List[int]]]:
        """
        将单帧潜变量解码为RGB图像。
        """
        img = [[[0] * 3 for _ in range(width)] for _ in range(height)]

        for i in range(height):
            for j in range(width):
                for c in range(3):
                    lat_idx = c % len(latent) if latent else 0
                    val = latent[lat_idx]
                    # 使用位置信息增加变化
                    pos_factor = math.sin(i * 0.1 + val) * math.cos(j * 0.1 + val * 0.5)
                    pixel = int(128 + val * 64 + pos_factor * 32)
                    pixel = max(0, min(255, pixel))
                    img[i][j][c] = pixel

        return img

    def _frames_to_video(self, latents: List[List[List[float]]],
                         height: int = 64, width: int = 64) -> List[List[List[List[int]]]]:
        """
        将潜变量帧序列解码为视频帧序列。

        Args:
            latents: 视频潜变量 [num_frames, batch, channels]
            height: 帧高度
            width: 帧宽度

        Returns:
            视频帧序列 [num_frames, H, W, 3]
        """
        frames = []
        for f in range(len(latents)):
            for b in range(len(latents[f])):
                frame = self._latent_to_frame(latents[f][b], height, width)
                frames.append(frame)
        return frames

    def generate(self, prompt: str, num_frames: int = 16, fps: int = 8,
                 num_steps: int = 25, guidance_scale: float = 7.5,
                 width: int = 512, height: int = 512,
                 seed: Optional[int] = None, **kwargs) -> 'GenerationResult':
        """
        AnimateDiff视频生成主方法。

        流程:
        1. 编码文本提示词
        2. 生成初始随机视频潜变量
        3. 迭代去噪（每步应用时序注意力）
        4. 将潜变量解码为视频帧

        Args:
            prompt: 文本提示词
            num_frames: 生成帧数
            fps: 帧率
            num_steps: 去噪步数
            guidance_scale: 引导强度
            width: 视频宽度
            height: 视频高度
            seed: 随机种子

        Returns:
            GenerationResult: 包含视频帧序列
        """
        self._init_temporal_layers()

        actual_seed = seed if seed is not None else self._seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2**32 - 1)

        self._num_frames = num_frames
        self._fps = fps

        # 更新运动模块帧数
        self._motion_module._num_frames = num_frames

        # 编码提示词
        text_embeds = self._encode_video_prompt(prompt)

        # 生成初始视频潜变量 [num_frames, 1, 4]
        latent_shape = [num_frames, 1, 4]
        latents = self._random_video_latents(latent_shape, seed=actual_seed)

        # 去噪循环
        denoised = self._denoise_video(latents, text_embeds, num_steps)

        # 解码为视频帧
        frame_height = height // 8  # 潜空间到图像空间的缩放
        frame_width = width // 8
        frames = self._frames_to_video(denoised, frame_height, frame_width)

        # 运动插值（如果需要更多帧）
        if len(frames) < num_frames:
            frames = self._motion_module.interpolate_motion(frames, fps)

        # 创建结果
        result = GenerationResult(
            images=frames,
            prompt=prompt,
            seed=actual_seed,
            steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
        result.fps = fps
        result.num_frames = len(frames)

        return result


# ===========================================================================
# 3. SVDPipeline - Stable Video Diffusion
# ===========================================================================

class SVDPipeline:
    """
    Stable Video Diffusion (SVD) 管线。
    从单张图像生成短视频片段。
    核心思想：将图像编码为潜变量，添加运动噪声，然后通过去噪生成视频。

    与AnimateDiff的区别:
    - SVD是图像到视频(I2V)，AnimateDiff是文本到视频(T2V)
    - SVD使用首帧作为强条件
    - SVD的运动噪声由motion_bucket_id控制
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._num_train_timesteps = 1000
        self._min_guidance_scale = 1.0
        self._max_guidance_scale = 3.0

        # 初始化组件
        self._unet_config = self._init_unet_config()
        self._vae_config = self._init_vae_config()
        self._scheduler_config = self._init_scheduler_config()

    def _init_unet_config(self) -> Dict:
        """初始化UNet配置"""
        rng = random.Random(self._seed or 42)
        return {
            "in_channels": 4,
            "out_channels": 4,
            "time_embed_dim": 256,
            "weights": [[rng.gauss(0, 0.02) for _ in range(4)] for _ in range(4)],
        }

    def _init_vae_config(self) -> Dict:
        """初始化VAE配置"""
        return {
            "latent_channels": 4,
            "out_channels": 3,
        }

    def _init_scheduler_config(self) -> Dict:
        """初始化调度器配置"""
        betas = [
            0.00085 + (0.012 - 0.00085) * i / (self._num_train_timesteps - 1)
            for i in range(self._num_train_timesteps)
        ]
        alphas = [1.0 - b for b in betas]
        alpha_cumprod = []
        cumprod = 1.0
        for a in alphas:
            cumprod *= a
            alpha_cumprod.append(cumprod)
        return {
            "num_train_timesteps": self._num_train_timesteps,
            "alpha_cumprod": alpha_cumprod,
        }

    def _encode_first_frame(self, image: List[List[List[int]]]) -> List[List[float]]:
        """
        编码首帧为潜变量。

        将输入图像通过VAE编码器映射到潜空间。
        同时生成帧间潜变量的初始值。

        Args:
            image: 首帧图像 [H, W, 3]

        Returns:
            首帧潜变量 [1, 4]
        """
        h = len(image)
        w = len(image[0]) if h > 0 else 0

        # 计算图像的全局统计特征作为潜变量
        latent = [0.0] * 4

        for c in range(3):
            total = 0.0
            count = 0
            for i in range(h):
                for j in range(w):
                    if c < len(image[i][j]):
                        total += image[i][j][c]
                        count += 1
            if count > 0:
                latent[c] = (total / count - 128.0) / 128.0

        # 第4个通道：边缘强度
        edge_sum = 0.0
        edge_count = 0
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                gx = abs(image[i][j][0] - image[i][j + 1][0])
                gy = abs(image[i][j][0] - image[i + 1][j][0])
                edge_sum += math.sqrt(gx * gx + gy * gy)
                edge_count += 1
        if edge_count > 0:
            latent[3] = edge_sum / edge_count / 255.0

        return [latent]

    def _add_motion_noise(self, latent: List[float],
                          motion_bucket_id: int,
                          num_frames: int,
                          seed: Optional[int] = None) -> List[List[List[float]]]:
        """
        添加运动噪声到首帧潜变量。

        motion_bucket_id控制运动幅度:
        - 1: 最小运动（几乎静止）
        - 127: 中等运动
        - 255: 最大运动

        算法:
        1. 将首帧潜变量复制num_frames份
        2. 根据motion_bucket_id计算噪声强度
        3. 添加与时间相关的运动噪声
        4. 确保首帧保持原始内容

        Args:
            latent: 首帧潜变量 [channels]
            motion_bucket_id: 运动幅度ID [1, 255]
            num_frames: 目标帧数
            seed: 随机种子

        Returns:
            带噪声的视频潜变量 [num_frames, 1, channels]
        """
        actual_seed = seed if seed is not None else self._seed

        # 将motion_bucket_id映射到噪声强度
        # 使用sigmoid-like映射使中间值有更大的动态范围
        normalized_motion = motion_bucket_id / 255.0
        noise_strength = normalized_motion * 0.5  # 最大噪声强度

        # 计算时间步t（对应噪声水平）
        alpha_cumprod = self._scheduler_config["alpha_cumprod"]
        # motion_bucket_id=1对应t=0（无噪声），255对应t=999（最大噪声）
        t = int(normalized_motion * (self._num_train_timesteps - 1))
        alpha_t = alpha_cumprod[min(t, len(alpha_cumprod) - 1)]
        sqrt_alpha = math.sqrt(alpha_t)
        sqrt_one_minus_alpha = math.sqrt(1.0 - alpha_t)

        # 生成视频潜变量
        rng = random.Random(actual_seed)
        video_latents = []

        for f in range(num_frames):
            frame_latent = []
            for b in range(1):  # batch_size = 1
                if f == 0:
                    # 首帧保持原始内容，添加少量噪声
                    frame = [latent[c] + rng.gauss(0, 0.01) for c in range(len(latent))]
                else:
                    # 后续帧添加运动噪声
                    # 时间相关的噪声：越远的帧噪声越大
                    frame_noise_ratio = f / max(num_frames - 1, 1)
                    frame = []
                    for c in range(len(latent)):
                        # 基础内容（随时间衰减）
                        content = latent[c] * (1.0 - frame_noise_ratio * 0.5)
                        # 运动噪声
                        noise = rng.gauss(0, noise_strength * frame_noise_ratio)
                        # 时间一致性噪声（相邻帧共享部分噪声）
                        temporal_noise = rng.gauss(0, noise_strength * 0.3)
                        frame.append(content + noise + temporal_noise)
                frame_latent.append(frame)
            video_latents.append(frame_latent)

        return video_latents

    def _unet_forward_video(self, x: List[float], t: int,
                             first_frame_latent: List[float],
                             frame_idx: int, num_frames: int) -> List[float]:
        """
        视频UNet前向传播（简化版）。
        包含对首帧条件的交叉注意力。

        Args:
            x: 当前帧潜变量
            t: 时间步
            first_frame_latent: 首帧潜变量（条件）
            frame_idx: 当前帧索引
            num_frames: 总帧数

        Returns:
            预测的噪声
        """
        rng = random.Random(t + frame_idx * 1000)

        # 时间嵌入
        half_dim = 128
        time_emb = []
        for i in range(half_dim):
            freq = math.exp(i * math.log(10000) / (half_dim - 1))
            time_emb.append(math.sin(t * freq))
        for i in range(half_dim):
            freq = math.exp(i * math.log(10000) / (half_dim - 1))
            time_emb.append(math.cos(t * freq))

        # 帧位置嵌入
        frame_pos = frame_idx / max(num_frames - 1, 1)
        pos_emb = [math.sin(frame_pos * math.pi) * 0.1 for _ in range(len(x))]
        pos_emb = pos_emb[:len(x)]

        # 首帧条件的影响
        condition_influence = []
        for c in range(len(x)):
            if c < len(first_frame_latent):
                # 首帧越接近，条件影响越强
                proximity = 1.0 / (1.0 + frame_idx * 0.5)
                condition_influence.append(first_frame_latent[c] * proximity * 0.3)
            else:
                condition_influence.append(0.0)

        # 组合预测
        dim = len(x)
        noise_pred = []
        for d in range(dim):
            t_val = time_emb[d % len(time_emb)] * 0.05
            x_val = x[d] * 0.8
            p_val = pos_emb[d] if d < len(pos_emb) else 0.0
            c_val = condition_influence[d] if d < len(condition_influence) else 0.0
            noise_pred.append(x_val + t_val + p_val + c_val)

        return noise_pred

    def _denoise_to_video(self, latent: List[List[List[float]]],
                          first_frame_latent: List[float],
                          steps: int,
                          noise_aug_strength: float = 0.02) -> List[List[List[float]]]:
        """
        去噪到视频。

        对视频潜变量进行迭代去噪，同时保持首帧一致性。

        Args:
            latent: 视频潜变量 [num_frames, 1, channels]
            first_frame_latent: 首帧潜变量
            steps: 去噪步数
            noise_aug_strength: 噪声增强强度

        Returns:
            去噪后的视频潜变量
        """
        alpha_cumprod = self._scheduler_config["alpha_cumprod"]
        num_train = self._num_train_timesteps
        num_frames = len(latent)

        # 设置推理时间步
        step_size = num_train // steps
        timesteps = [num_train - 1 - i * step_size for i in range(steps)]

        current = [[[v for v in row] for row in frame] for frame in latent]

        for step_idx, t in enumerate(timesteps):
            alpha_t = alpha_cumprod[t]
            sqrt_alpha_t = math.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = math.sqrt(1.0 - alpha_t)

            prev_t = t - step_size
            if prev_t < 0:
                alpha_prev = 1.0
            else:
                alpha_prev = alpha_cumprod[prev_t]
            sqrt_alpha_prev = math.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = math.sqrt(1.0 - alpha_prev)

            # 对每帧预测噪声
            for f in range(num_frames):
                for b in range(len(current[f])):
                    noise_pred = self._unet_forward_video(
                        current[f][b], t, first_frame_latent, f, num_frames
                    )

                    sample = current[f][b]

                    # DDIM更新
                    pred_x0 = [(sample[j] - sqrt_one_minus_alpha_t * noise_pred[j]) / sqrt_alpha_t
                               for j in range(len(sample))]

                    new_sample = [sqrt_alpha_prev * pred_x0[j] +
                                  sqrt_one_minus_alpha_prev * sqrt_one_minus_alpha_t * noise_pred[j]
                                  for j in range(len(sample))]

                    current[f][b] = new_sample

            # 首帧一致性约束：每步后将首帧拉回原始值
            if current and current[0]:
                for b in range(len(current[0])):
                    for c in range(len(first_frame_latent)):
                        # 渐进式约束：去噪初期约束弱，后期约束强
                        progress = step_idx / max(steps - 1, 1)
                        constraint_strength = progress * 0.5
                        current[0][b][c] = (current[0][b][c] * (1 - constraint_strength) +
                                            first_frame_latent[c] * constraint_strength)

        return current

    def _latent_to_frame(self, latent: List[float], height: int = 64,
                         width: int = 64, frame_idx: int = 0,
                         first_frame: Optional[List[List[List[int]]]] = None) -> List[List[List[int]]]:
        """
        将潜变量解码为视频帧。
        如果提供了首帧，使用首帧信息增强解码质量。
        """
        img = [[[0] * 3 for _ in range(width)] for _ in range(height)]

        for i in range(height):
            for j in range(width):
                for c in range(3):
                    lat_idx = c % len(latent) if latent else 0
                    val = latent[lat_idx]

                    # 使用帧索引创建时间变化
                    temporal_factor = math.sin(frame_idx * 0.5 + i * 0.05) * math.cos(j * 0.05)
                    pixel = int(128 + val * 64 + temporal_factor * 20)
                    pixel = max(0, min(255, pixel))

                    # 如果有首帧参考，混合首帧信息
                    if first_frame and frame_idx == 0:
                        if i < len(first_frame) and j < len(first_frame[0]):
                            pixel = first_frame[i][j][c]

                    img[i][j][c] = pixel

        return img

    def _latents_to_video(self, latents: List[List[List[float]]],
                          first_frame: Optional[List[List[List[int]]]] = None,
                          height: int = 64, width: int = 64) -> List[List[List[List[int]]]]:
        """
        将视频潜变量序列解码为视频帧序列。
        """
        frames = []
        for f in range(len(latents)):
            for b in range(len(latents[f])):
                frame = self._latent_to_frame(
                    latents[f][b], height, width,
                    frame_idx=f, first_frame=first_frame
                )
                frames.append(frame)
        return frames

    def _compute_motion_score(self, frames: List[List[List[List[int]]]]) -> float:
        """
        计算视频的运动分数。
        通过比较相邻帧的差异来衡量运动幅度。
        """
        if len(frames) <= 1:
            return 0.0

        total_diff = 0.0
        count = 0

        for f in range(len(frames) - 1):
            h = min(len(frames[f]), len(frames[f + 1]))
            w = min(len(frames[f][0]), len(frames[f + 1][0])) if h > 0 else 0

            for i in range(h):
                for j in range(w):
                    for c in range(min(3, len(frames[f][i][j]))):
                        diff = abs(frames[f][i][j][c] - frames[f + 1][i][j][c])
                        total_diff += diff
                        count += 1

        return total_diff / max(count, 1)

    def generate(self, image: List[List[List[int]]], num_frames: int = 14,
                 motion_bucket_id: int = 127, num_steps: int = 25,
                 noise_aug_strength: float = 0.02,
                 width: int = 512, height: int = 512,
                 seed: Optional[int] = None, **kwargs) -> 'GenerationResult':
        """
        SVD视频生成主方法。

        流程:
        1. 编码首帧为潜变量
        2. 根据motion_bucket_id添加运动噪声，生成初始视频潜变量
        3. 迭代去噪（保持首帧一致性）
        4. 解码为视频帧序列

        Args:
            image: 输入图像 [H, W, 3]
            num_frames: 生成帧数（通常14或25）
            motion_bucket_id: 运动幅度 [1, 255]
            num_steps: 去噪步数
            noise_aug_strength: 噪声增强强度
            width: 视频宽度
            height: 视频高度
            seed: 随机种子

        Returns:
            GenerationResult: 包含视频帧序列
        """
        actual_seed = seed if seed is not None else self._seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2**32 - 1)

        # 限制motion_bucket_id范围
        motion_bucket_id = max(1, min(255, motion_bucket_id))

        # 编码首帧
        first_frame_latent = self._encode_first_frame(image)[0]

        # 添加运动噪声
        video_latents = self._add_motion_noise(
            first_frame_latent, motion_bucket_id, num_frames, seed=actual_seed
        )

        # 去噪到视频
        denoised = self._denoise_to_video(
            video_latents, first_frame_latent, num_steps, noise_aug_strength
        )

        # 解码为视频帧
        frame_height = height // 8
        frame_width = width // 8
        frames = self._latents_to_video(denoised, first_frame=image,
                                         height=frame_height, width=frame_width)

        # 确保首帧与输入一致
        if frames:
            frames[0] = image

        # 计算运动分数
        motion_score = self._compute_motion_score(frames)

        result = GenerationResult(
            images=frames,
            prompt="",
            seed=actual_seed,
            steps=num_steps,
            width=width,
            height=height,
        )
        result.fps = 6
        result.num_frames = len(frames)
        result.motion_bucket_id = motion_bucket_id
        result.motion_score = motion_score

        return result


# ===========================================================================
# 4. GenerationResult - 视频生成结果（扩展）
# ===========================================================================

class GenerationResult:
    """
    生成结果封装类（视频扩展版）。
    支持图像和视频生成结果。
    """

    def __init__(self, images: List = None, prompt: str = "",
                 negative_prompt: str = "", seed: Optional[int] = None,
                 steps: int = 50, guidance_scale: float = 7.5,
                 width: int = 512, height: int = 512):
        self.images = images or []
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        # 视频特有属性
        self.fps: int = 8
        self.num_frames: int = 0
        self.motion_bucket_id: int = 127
        self.motion_score: float = 0.0

    def is_video(self) -> bool:
        """判断是否为视频结果"""
        return self.num_frames > 1 or self.fps > 0

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "num_images": len(self.images),
        }
        if self.is_video():
            result.update({
                "type": "video",
                "fps": self.fps,
                "num_frames": self.num_frames,
                "motion_bucket_id": self.motion_bucket_id,
                "motion_score": round(self.motion_score, 4),
            })
        else:
            result["type"] = "image"
        return result

    def get_frame(self, index: int) -> Optional[List[List[List[int]]]]:
        """获取指定帧"""
        if 0 <= index < len(self.images):
            return self.images[index]
        return None

    def get_duration_seconds(self) -> float:
        """获取视频时长（秒）"""
        if self.fps > 0:
            return self.num_frames / self.fps
        return 0.0
