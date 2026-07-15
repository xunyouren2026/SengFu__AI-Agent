"""
EfficientNet架构 - 完整实现
包含: EfficientNet-B0~B7, EfficientNet-Lite0~Lite4, Compound Scaling
基于论文: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import Union, List, Tuple, Optional, Callable

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



# ============================================================
# 基础组件
# ============================================================

def _swish(x: float) -> float:
    """Swish激活函数: x * sigmoid(x)"""
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return x * ex / (1.0 + ex)


def _sigmoid(x: float) -> float:
    """Sigmoid激活函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def _relu6(x: float) -> float:
    """ReLU6激活函数"""
    return min(max(0.0, x), 6.0)


def _hard_swish(x: float) -> float:
    """Hard Swish激活函数"""
    return x * max(0.0, min(1.0, x * 0.166667 + 0.5))


def _drop_connect(x: float, survival_prob: float, training: bool = True) -> float:
    """
    DropConnect正则化
    随机将权重置零（训练时），推理时不做任何操作
    论文: "Regularized Evolution for Image Classifier Architecture Search"
    
    Args:
        x: 输入值
        survival_prob: 保留概率
        training: 是否为训练模式
    Returns:
        处理后的值
    """
    if not training or survival_prob <= 0.0:
        return x
    if random.random() < survival_prob:
        return x
    return 0.0


def _conv2d_output_size(input_size: int, kernel: int, stride: int, padding: int) -> int:
    """计算2D卷积输出尺寸"""
    return (input_size + 2 * padding - kernel) // stride + 1


def _same_padding(input_size: int, kernel: int, stride: int) -> int:
    """计算same padding大小"""
    if stride == 1:
        return (kernel - 1) // 2
    else:
        out = math.ceil(input_size / stride)
        return max(0, (out - 1) * stride + kernel - input_size)


def _round_filters(filters: int, width_coefficient: float, divisor: int = 8) -> int:
    """
    按宽度系数调整卷积核数量并取整
    确保通道数是divisor的整数倍
    
    Args:
        filters: 原始通道数
        width_coefficient: 宽度缩放系数
        divisor: 取整整除数
    Returns:
        调整后的通道数
    """
    filters *= width_coefficient
    min_depth = divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats: int, depth_coefficient: float) -> int:
    """
    按深度系数调整网络层数
    
    Args:
        repeats: 原始重复次数
        depth_coefficient: 深度缩放系数
    Returns:
        调整后的重复次数
    """
    return int(math.ceil(depth_coefficient * repeats))


# ============================================================
# 2D卷积 + 批归一化 + 激活 (融合模块)
# ============================================================


# =============================================================================
# PyTorch Compatibility Utilities
# =============================================================================

def _to_tensor(x, device: str = None, dtype=None, requires_grad: bool = False):
    """
    Convert input to torch.Tensor.
    
    Supports:
    - torch.Tensor: returned as-is (with optional device/dtype cast)
    - list/tuple: converted to torch.Tensor
    - numpy.ndarray: converted to torch.Tensor
    - scalar: wrapped in torch.Tensor
    
    Args:
        x: Input data (tensor, list, tuple, numpy array, or scalar)
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Target dtype (torch.float32, torch.float64, etc.)
        requires_grad: Whether to track gradients
    
    Returns:
        torch.Tensor or original type if torch is not available
    """
    if not _HAS_TORCH:
        return x
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != torch.device(device):
            x = x.to(device=device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        if requires_grad and not x.requires_grad:
            x = x.requires_grad_(requires_grad=True)
        return x
    # Convert from list/tuple/numpy
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def _to_numpy(x):
    """Convert torch.Tensor to numpy array."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


def _to_list(x):
    """Convert torch.Tensor to nested Python list."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().tolist()


def _get_device(x):
    """Get device of tensor, default to 'cpu'."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.device
    return None


def _batch_dim(x):
    """Ensure input has batch dimension. If 2D, add batch dim to make 3D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 2:
            return x.unsqueeze(0)
    return x


def _unbatch(x):
    """Remove batch dimension if it's 1. If 3D with batch=1, squeeze to 2D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 3 and x.size(0) == 1:
            return x.squeeze(0)
    return x


class Conv2dBNAct:
    """
    融合的2D卷积 + 批归一化 + 激活函数模块
    EfficientNet的基础构建块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        groups: 分组数 (1=标准卷积, in_channels=深度卷积)
        activation: 激活函数名称 ('swish', 'relu6', 'none')
        bn_momentum: 批归一化动量
        bn_epsilon: 批归一化epsilon
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, groups: int = 1,
                 activation: str = 'swish', bn_momentum: float = 0.99,
                 bn_epsilon: float = 1e-3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation_name = activation
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        
        # 卷积核权重: [out_channels, in_channels/groups, kH, kW]
        fan_in = in_channels * kernel_size * kernel_size // groups
        fan_out = out_channels * kernel_size * kernel_size // groups
        std = math.sqrt(2.0 / (fan_in + fan_out))
        self.weight = [random.gauss(0, std) for _ in range(
            out_channels * (in_channels // groups) * kernel_size * kernel_size)]
        self.bias = [0.0] * out_channels
        
        # 批归一化参数
        self.bn_gamma = [1.0] * out_channels  # 缩放参数
        self.bn_beta = [0.0] * out_channels   # 偏移参数
        self.bn_running_mean = [0.0] * out_channels
        self.bn_running_var = [1.0] * out_channels
        
        # 缓存 (用于反向传播)
        self._cache = None
    
    def _activate(self, x: float) -> float:
        """应用激活函数"""
        if self.activation_name == 'swish':
            return _swish(x)
        elif self.activation_name == 'relu6':
            return _relu6(x)
        elif self.activation_name == 'hard_swish':
            return _hard_swish(x)
        elif self.activation_name == 'none' or self.activation_name == 'identity':
            return x
        else:
            return _swish(x)
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        前向传播
        
        Args:
            x: 输入张量 [C_in, H, W]
        Returns:
            输出张量 [C_out, H_out, W_out]
        """
        C_in = len(x)
        H_in = len(x[0])
        W_in = len(x[0][0])
        
        H_out = _conv2d_output_size(H_in, self.kernel_size, self.stride, self.padding)
        W_out = _conv2d_output_size(W_in, self.kernel_size, self.stride, self.padding)
        
        C_g = self.in_channels // self.groups  # 每组输入通道
        k = self.kernel_size
        
        output = []
        cache = {'x': x, 'conv_out': [], 'bn_out': []}
        
        for co in range(self.out_channels):
            channel_out = []
            g_start = (co // (self.out_channels // self.groups)) * C_g
            g_end = g_start + C_g
            
            for oh in range(H_out):
                row_out = []
                for ow in range(W_out):
                    val = self.bias[co]
                    for ci_idx, ci in enumerate(range(g_start, g_end)):
                        for kh in range(k):
                            for kw in range(k):
                                ih = oh * self.stride - self.padding + kh
                                iw = ow * self.stride - self.padding + kw
                                if 0 <= ih < H_in and 0 <= iw < W_in:
                                    w_idx = co * C_g * k * k + ci_idx * k * k + kh * k + kw
                                    val += self.weight[w_idx] * x[ci][ih][iw]
                    row_out.append(val)
                channel_out.append(row_out)
            
            cache['conv_out'].append(channel_out)
            output.append(channel_out)
        
        # 批归一化
        bn_output = []
        for co in range(self.out_channels):
            channel = output[co]
            flat = [channel[h][w] for h in range(H_out) for w in range(W_out)]
            mean = sum(flat) / len(flat)
            var = sum((v - mean) ** 2 for v in flat) / len(flat)
            
            # 更新运行统计量
            self.bn_running_mean[co] = (self.bn_momentum * self.bn_running_mean[co] +
                                         (1 - self.bn_momentum) * mean)
            self.bn_running_var[co] = (self.bn_momentum * self.bn_running_var[co] +
                                        (1 - self.bn_momentum) * var)
            
            bn_channel = []
            for h in range(H_out):
                bn_row = []
                for w in range(W_out):
                    normalized = (channel[h][w] - mean) / math.sqrt(var + self.bn_epsilon)
                    bn_val = self.bn_gamma[co] * normalized + self.bn_beta[co]
                    activated = self._activate(bn_val)
                    bn_row.append(activated)
                bn_channel.append(bn_row)
            bn_output.append(bn_channel)
            cache['bn_out'].append(bn_channel)
        
        self._cache = cache
        return bn_output
    
    def get_output_shape(self, input_h: int, input_w: int) -> Tuple[int, int, int]:
        """获取输出形状"""
        h = _conv2d_output_size(input_h, self.kernel_size, self.stride, self.padding)
        w = _conv2d_output_size(input_w, self.kernel_size, self.stride, self.padding)
        return (self.out_channels, h, w)


# ============================================================
# Squeeze-and-Excitation (SE) 模块
# ============================================================

class SqueezeExcitation:
    """
    Squeeze-and-Excitation模块
    通过全局平均池化 + 两层全连接实现通道注意力
    
    结构: GlobalAvgPool -> FC1(reduction) -> activation -> FC2(channels) -> sigmoid -> scale
    
    参数:
        channels: 输入通道数
        reduction: 通道缩减比例 (默认4)
        activation: 中间层激活函数
    """
    
    def __init__(self, channels: int, reduction: int = 4, activation: str = 'swish'):
        self.channels = channels
        self.reduced_channels = max(1, channels // reduction)
        self.activation_name = activation
        
        # FC1: channels -> reduced_channels
        self.fc1_weight = [random.gauss(0, math.sqrt(2.0 / channels))
                           for _ in range(self.reduced_channels * channels)]
        self.fc1_bias = [0.0] * self.reduced_channels
        
        # FC2: reduced_channels -> channels
        self.fc2_weight = [random.gauss(0, math.sqrt(2.0 / self.reduced_channels))
                           for _ in range(channels * self.reduced_channels)]
        self.fc2_bias = [0.0] * channels
    
    def _activate(self, x: float) -> float:
        if self.activation_name == 'swish':
            return _swish(x)
        elif self.activation_name == 'relu':
            return max(0.0, x)
        elif self.activation_name == 'relu6':
            return _relu6(x)
        return _swish(x)
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        前向传播
        
        Args:
            x: 输入张量 [C, H, W]
        Returns:
            通道注意力加权后的张量 [C, H, W]
        """
        C = len(x)
        H = len(x[0])
        W = len(x[0][0])
        
        # 全局平均池化: [C, H, W] -> [C]
        gap = []
        for c in range(C):
            s = sum(x[c][h][w] for h in range(H) for w in range(W))
            gap.append(s / (H * W))
        
        # FC1: [C] -> [reduced]
        fc1_out = []
        for r in range(self.reduced_channels):
            val = self.fc1_bias[r]
            for c in range(C):
                val += self.fc1_weight[r * C + c] * gap[c]
            fc1_out.append(self._activate(val))
        
        # FC2: [reduced] -> [C]
        fc2_out = []
        for c in range(C):
            val = self.fc2_bias[c]
            for r in range(self.reduced_channels):
                val += self.fc2_weight[c * self.reduced_channels + r] * fc1_out[r]
            fc2_out.append(_sigmoid(val))
        
        # 通道缩放
        output = []
        for c in range(C):
            scale = fc2_out[c]
            channel = []
            for h in range(H):
                row = [x[c][h][w] * scale for w in range(W)]
                channel.append(row)
            output.append(channel)
        
        return output


# ============================================================
# MBConv (Mobile Inverted Bottleneck Convolution)
# ============================================================

class MBConvBlock:
    """
    MBConv模块 - EfficientNet的核心构建块
    
    结构 (有SE):
        input -> expand_conv(1x1) -> BN -> Swish -> depthwise_conv(3x3) -> BN -> Swish
              -> SE -> project_conv(1x1) -> BN -> drop_connect -> output
              (+ residual connection when stride==1 and in==out)
    
    结构 (无SE):
        input -> expand_conv(1x1) -> BN -> Swish -> depthwise_conv(3x3) -> BN -> Swish
              -> project_conv(1x1) -> BN -> drop_connect -> output
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 深度卷积核大小 (3或5)
        stride: 步长 (1或2)
        expand_ratio: 扩展比率 (0表示不扩展，即无expand conv)
        se_ratio: SE模块缩减比率 (0表示不用SE)
        drop_connect_rate: DropConnect丢弃率
        activation: 激活函数
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expand_ratio: float = 1.0, se_ratio: float = 0.25,
                 drop_connect_rate: float = 0.0, activation: str = 'swish'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        self.activation = activation
        self.training = True
        
        # 是否使用残差连接
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # 扩展卷积
        expanded_channels = int(in_channels * expand_ratio)
        self.has_expand = expand_ratio != 1.0
        
        if self.has_expand:
            self.expand_conv = Conv2dBNAct(
                in_channels, expanded_channels, kernel_size=1,
                stride=1, padding=0, activation=activation
            )
        
        # 深度卷积
        dw_padding = _same_padding(0, kernel_size, stride)  # 近似same padding
        self.depthwise_conv = Conv2dBNAct(
            expanded_channels, expanded_channels, kernel_size=kernel_size,
            stride=stride, padding=dw_padding, groups=expanded_channels,
            activation=activation
        )
        
        # SE模块
        self.has_se = se_ratio > 0
        if self.has_se:
            self.se = SqueezeExcitation(expanded_channels,
                                         reduction=int(1.0 / se_ratio),
                                         activation=activation)
        
        # 投影卷积 (1x1, 无激活)
        self.project_conv = Conv2dBNAct(
            expanded_channels, out_channels, kernel_size=1,
            stride=1, padding=0, activation='none'
        )
        
        self.expanded_channels = expanded_channels
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        前向传播
        
        Args:
            x: 输入张量 [C_in, H, W]
        Returns:
            输出张量 [C_out, H_out, W_out]
        """
        identity = x
        
        # 扩展阶段
        if self.has_expand:
            x = self.expand_conv.forward(x)
        
        # 深度卷积阶段
        x = self.depthwise_conv.forward(x)
        
        # SE注意力阶段
        if self.has_se:
            x = self.se.forward(x)
        
        # 投影阶段
        x = self.project_conv.forward(x)
        
        # DropConnect
        if self.drop_connect_rate > 0:
            C = len(x)
            H = len(x[0])
            W = len(x[0][0])
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        x[c][h][w] = _drop_connect(
                            x[c][h][w], 1.0 - self.drop_connect_rate, self.training
                        )
        
        # 残差连接
        if self.use_residual:
            C = len(x)
            H = len(x[0])
            W = len(x[0][0])
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        x[c][h][w] += identity[c][h][w]
        
        return x
    
    def get_output_shape(self, input_h: int, input_w: int) -> Tuple[int, int, int]:
        """获取输出形状"""
        h, w = input_h, input_w
        if self.has_expand:
            _, h, w = self.expand_conv.get_output_shape(h, w)
        _, h, w = self.depthwise_conv.get_output_shape(h, w)
        _, h, w = self.project_conv.get_output_shape(h, w)
        if self.stride == 1 and self.in_channels == self.out_channels:
            return (self.out_channels, h, w)
        return (self.out_channels, h, w)


# ============================================================
# Stem (初始卷积层)
# ============================================================

class EfficientNetStem:
    """
    EfficientNet的Stem层
    Conv2d(3, channels, 3x3, stride=2) -> BN -> Swish
    
    参数:
        in_channels: 输入通道数 (通常为3)
        out_channels: 输出通道数
        activation: 激活函数
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 32,
                 activation: str = 'swish'):
        padding = _same_padding(0, 3, 2)
        self.conv = Conv2dBNAct(
            in_channels, out_channels, kernel_size=3,
            stride=2, padding=padding, activation=activation
        )
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        return self.conv.forward(x)
    
    def get_output_shape(self, h: int, w: int) -> Tuple[int, int, int]:
        return self.conv.get_output_shape(h, w)


# ============================================================
# Head (最终分类头)
# ============================================================

class EfficientNetHead:
    """
    EfficientNet的Head层
    Conv2d(last_channels, final_channels, 1x1) -> BN -> Swish
    -> GlobalAvgPool -> Dropout -> FC(num_classes)
    
    参数:
        in_channels: 输入通道数
        final_channels: 1x1卷积输出通道数
        num_classes: 分类数
        dropout_rate: Dropout率
        activation: 激活函数
    """
    
    def __init__(self, in_channels: int, final_channels: int = 1280,
                 num_classes: int = 1000, dropout_rate: float = 0.2,
                 activation: str = 'swish'):
        self.final_conv = Conv2dBNAct(
            in_channels, final_channels, kernel_size=1,
            stride=1, padding=0, activation=activation
        )
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.final_channels = final_channels
        
        # 分类全连接层
        self.fc_weight = [random.gauss(0, math.sqrt(2.0 / final_channels))
                          for _ in range(num_classes * final_channels)]
        self.fc_bias = [0.0] * num_classes
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[float], 'torch.Tensor']:
        """
        前向传播
        
        Args:
            x: 输入特征图 [C, H, W]
        Returns:
            分类logits [num_classes]
        """
        # 1x1卷积
        x = self.final_conv.forward(x)
        
        C = len(x)
        H = len(x[0])
        W = len(x[0][0])
        
        # 全局平均池化
        gap = []
        for c in range(C):
            s = sum(x[c][h][w] for h in range(H) for w in range(W))
            gap.append(s / (H * W))
        
        # Dropout
        if self.training and self.dropout_rate > 0:
            gap = [v if random.random() > self.dropout_rate else 0.0 for v in gap]
        
        # 全连接
        logits = []
        for cls in range(self.num_classes):
            val = self.fc_bias[cls]
            for c in range(C):
                val += self.fc_weight[cls * C + c] * gap[c]
            logits.append(val)
        
        return logits


# ============================================================
# EfficientNet 主网络
# ============================================================

class EfficientNet:
    """
    EfficientNet完整网络
    
    基于论文 "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
    使用复合缩放 (Compound Scaling) 统一缩放深度、宽度和分辨率
    
    网络结构:
        Stem(3x3, stride=2) -> [MBConvBlock x N] -> Head(1x1 + GAP + FC)
    
    参数:
        width_coefficient: 宽度缩放系数
        depth_coefficient: 深度缩放系数
        dropout_rate: 最终Dropout率
        num_classes: 分类数
        drop_connect_rate: DropConnect基础率 (实际率 = base * i / total_blocks)
    """
    
    # EfficientNet-B0的基准配置
    # (in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio)
    _BLOCK_CONFIGS = [
        (32, 16, 3, 1, 1, 0.25),     # Stage 1
        (16, 24, 3, 2, 6, 0.25),     # Stage 2
        (24, 40, 5, 2, 6, 0.25),     # Stage 3
        (40, 80, 3, 2, 6, 0.25),     # Stage 4
        (80, 112, 5, 1, 6, 0.25),    # Stage 5
        (112, 192, 5, 2, 6, 0.25),   # Stage 6
        (192, 320, 3, 1, 6, 0.25),   # Stage 7
    ]
    
    # 每个stage的重复次数 (B0基准)
    _BLOCK_REPEATS = [1, 2, 2, 3, 3, 4, 1]
    
    def __init__(self, width_coefficient: float = 1.0, depth_coefficient: float = 1.0,
                 dropout_rate: float = 0.2, num_classes: int = 1000,
                 drop_connect_rate: float = 0.2, activation: str = 'swish'):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.drop_connect_rate = drop_connect_rate
        self.activation = activation
        self.training = True
        
        # 计算总块数 (用于DropConnect率线性增长)
        total_blocks = sum(_round_repeats(r, depth_coefficient) for r in self._BLOCK_REPEATS)
        
        # Stem
        stem_channels = _round_filters(32, width_coefficient)
        self.stem = EfficientNetStem(
            in_channels=3, out_channels=stem_channels, activation=activation
        )
        
        # 构建MBConv块
        self.blocks = []
        block_idx = 0
        current_channels = stem_channels
        
        for stage_idx, (in_ch, out_ch, k, s, expand, se) in enumerate(self._BLOCK_CONFIGS):
            in_c = _round_filters(in_ch, width_coefficient)
            out_c = _round_filters(out_ch, width_coefficient)
            repeats = _round_repeats(self._BLOCK_REPEATS[stage_idx], depth_coefficient)
            
            for i in range(repeats):
                # 除第一块外，stride=1
                stride = s if i == 0 else 1
                # 除第一块外，输入通道=上一块输出
                input_c = in_c if i == 0 else out_c
                
                dc_rate = drop_connect_rate * block_idx / max(1, total_blocks - 1)
                
                block = MBConvBlock(
                    in_channels=input_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=stride,
                    expand_ratio=expand,
                    se_ratio=se,
                    drop_connect_rate=dc_rate,
                    activation=activation
                )
                self.blocks.append(block)
                block_idx += 1
            
            current_channels = out_c
        
        # Head
        last_channels = _round_filters(320, width_coefficient)
        final_channels = _round_filters(1280, width_coefficient)
        self.head = EfficientNetHead(
            in_channels=last_channels,
            final_channels=final_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation
        )
        self.head.training = self.training
        
        # 设置所有块的training标志
        for block in self.blocks:
            block.training = self.training
        
        self._total_blocks = total_blocks
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[float], 'torch.Tensor']:
        """
        前向传播
        
        Args:
            x: 输入图像 [3, H, W]
        Returns:
            分类logits [num_classes]
        """
        # Stem
        x = self.stem.forward(x)
        
        # MBConv blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Head
        logits = self.head.forward(x)
        return logits
    
    def extract_features(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        提取中间特征 (用于迁移学习)
        
        Args:
            x: 输入图像 [3, H, W]
        Returns:
            最后一个MBConv块的输出特征图 [C, H, W]
        """
        x = self.stem.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        return x
    
    def predict(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> int:
        """
        预测类别
        
        Args:
            x: 输入图像 [3, H, W]
        Returns:
            预测的类别索引
        """
        logits = self.forward(x)
        return max(range(len(logits)), key=lambda i: logits[i])
    
    def set_training(self, mode: bool):
        """设置训练/评估模式"""
        self.training = mode
        self.head.training = mode
        for block in self.blocks:
            block.training = mode
    
    def count_parameters(self) -> int:
        """计算总参数量"""
        count = 0
        
        # Stem参数
        stem_c = _round_filters(32, self.width_coefficient)
        count += stem_c * 3 * 3 * 3 + stem_c  # weight + bias
        count += stem_c * 2  # BN gamma + beta
        
        # Block参数
        for block in self.blocks:
            exp_c = block.expanded_channels
            in_c = block.in_channels
            out_c = block.out_channels
            k = block.kernel_size
            
            if block.has_expand:
                count += exp_c * in_c * 1 * 1 + exp_c  # expand conv
                count += exp_c * 2  # expand BN
            
            count += exp_c * k * k + exp_c  # depthwise conv
            count += exp_c * 2  # depthwise BN
            
            if block.has_se:
                se_reduced = max(1, exp_c // int(1.0 / block.se_ratio))
                count += se_reduced * exp_c + se_reduced  # FC1
                count += exp_c * se_reduced + exp_c  # FC2
            
            count += out_c * exp_c * 1 * 1 + out_c  # project conv
            count += out_c * 2  # project BN
        
        # Head参数
        last_c = _round_filters(320, self.width_coefficient)
        final_c = _round_filters(1280, self.width_coefficient)
        count += final_c * last_c * 1 * 1 + final_c  # 1x1 conv
        count += final_c * 2  # BN
        count += self.num_classes * final_c + self.num_classes  # FC
        
        return count
    
    def get_architecture_summary(self) -> str:
        """获取网络架构摘要"""
        lines = []
        lines.append(f"EfficientNet Architecture Summary")
        lines.append(f"{'='*60}")
        lines.append(f"Width coefficient:  {self.width_coefficient}")
        lines.append(f"Depth coefficient:  {self.depth_coefficient}")
        lines.append(f"Dropout rate:       {self.dropout_rate}")
        lines.append(f"Drop connect rate:  {self.drop_connect_rate}")
        lines.append(f"Num classes:        {self.num_classes}")
        lines.append(f"Total blocks:       {self._total_blocks}")
        lines.append(f"Total parameters:   {self.count_parameters():,}")
        lines.append(f"")
        
        stem_c = _round_filters(32, self.width_coefficient)
        lines.append(f"Stem: Conv2d(3, {stem_c}, 3x3, stride=2)")
        lines.append(f"")
        
        stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4',
                       'Stage 5', 'Stage 6', 'Stage 7']
        block_idx = 0
        for si, (in_ch, out_ch, k, s, exp, se) in enumerate(self._BLOCK_CONFIGS):
            in_c = _round_filters(in_ch, self.width_coefficient)
            out_c = _round_filters(out_ch, self.width_coefficient)
            repeats = _round_repeats(self._BLOCK_REPEATS[si], self.depth_coefficient)
            lines.append(f"{stage_names[si]}: {repeats}x MBConv{k}(in={in_c}, out={out_c}, "
                        f"expand={exp}, se={se}, stride={s})")
            block_idx += repeats
        
        final_c = _round_filters(1280, self.width_coefficient)
        lines.append(f"")
        lines.append(f"Head: Conv2d(1x1, {final_c}) -> GAP -> Dropout({self.dropout_rate}) "
                    f"-> FC({self.num_classes})")
        
        return '\n'.join(lines)


# ============================================================
# EfficientNet-B0 ~ B7 预定义模型
# ============================================================

# 复合缩放系数 (来自论文Table 1)
# (width_coefficient, depth_coefficient, resolution, dropout_rate)
_EFFICIENTNET_COEFFICIENTS = {
    'b0': (1.0, 1.0, 224, 0.2),
    'b1': (1.0, 1.1, 240, 0.2),
    'b2': (1.1, 1.2, 260, 0.3),
    'b3': (1.2, 1.4, 300, 0.3),
    'b4': (1.4, 1.8, 380, 0.4),
    'b5': (1.6, 2.2, 456, 0.4),
    'b6': (1.8, 2.6, 528, 0.5),
    'b7': (2.0, 3.1, 600, 0.5),
}


def efficientnet_b0(num_classes: int = 1000, dropout_rate: float = 0.2) -> EfficientNet:
    """EfficientNet-B0: 5.3M参数, 224x224输入"""
    return EfficientNet(
        width_coefficient=1.0, depth_coefficient=1.0,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


def efficientnet_b1(num_classes: int = 1000, dropout_rate: float = 0.2) -> EfficientNet:
    """EfficientNet-B1: 7.8M参数, 240x240输入"""
    return EfficientNet(
        width_coefficient=1.0, depth_coefficient=1.1,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


def efficientnet_b2(num_classes: int = 1000, dropout_rate: float = 0.3) -> EfficientNet:
    """EfficientNet-B2: 9.2M参数, 260x260输入"""
    return EfficientNet(
        width_coefficient=1.1, depth_coefficient=1.2,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


def efficientnet_b3(num_classes: int = 1000, dropout_rate: float = 0.3) -> EfficientNet:
    """EfficientNet-B3: 12M参数, 300x300输入"""
    return EfficientNet(
        width_coefficient=1.2, depth_coefficient=1.4,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


def efficientnet_b4(num_classes: int = 1000, dropout_rate: float = 0.4) -> EfficientNet:
    """EfficientNet-B4: 19M参数, 380x380输入"""
    return EfficientNet(
        width_coefficient=1.4, depth_coefficient=1.8,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


def efficientnet_b5(num_classes: int = 1000, dropout_rate: float = 0.4) -> EfficientNet:
    """EfficientNet-B5: 30M参数, 456x456输入"""
    return EfficientNet(
        width_coefficient=1.6, depth_coefficient=2.2,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


def efficientnet_b6(num_classes: int = 1000, dropout_rate: float = 0.5) -> EfficientNet:
    """EfficientNet-B6: 43M参数, 528x528输入"""
    return EfficientNet(
        width_coefficient=1.8, depth_coefficient=2.6,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


def efficientnet_b7(num_classes: int = 1000, dropout_rate: float = 0.5) -> EfficientNet:
    """EfficientNet-B7: 66M参数, 600x600输入"""
    return EfficientNet(
        width_coefficient=2.0, depth_coefficient=3.1,
        dropout_rate=dropout_rate, num_classes=num_classes,
        drop_connect_rate=0.2
    )


# ============================================================
# EfficientNet-Lite (轻量级版本)
# ============================================================

class EfficientNetLite:
    """
    EfficientNet-Lite: 面向边缘设备的轻量版本
    
    与标准EfficientNet的区别:
    1. 移除所有Squeeze-and-Excitation模块
    2. 使用ReLU6替代Swish激活函数
    3. 使用depthwise卷积的same padding
    
    参数:
        width_coefficient: 宽度缩放系数
        depth_coefficient: 深度缩放系数
        num_classes: 分类数
        dropout_rate: Dropout率
    """
    
    # Lite版本的块配置 (无SE)
    _LITE_BLOCK_CONFIGS = [
        (32, 16, 3, 1, 1),   # Stage 1
        (16, 24, 3, 2, 6),   # Stage 2
        (24, 40, 5, 2, 6),   # Stage 3
        (40, 80, 3, 2, 6),   # Stage 4
        (80, 112, 5, 1, 6),  # Stage 5
        (112, 192, 5, 2, 6), # Stage 6
        (192, 320, 3, 1, 6), # Stage 7
    ]
    
    _LITE_BLOCK_REPEATS = [1, 2, 2, 3, 3, 4, 1]
    
    def __init__(self, width_coefficient: float = 1.0, depth_coefficient: float = 1.0,
                 num_classes: int = 1000, dropout_rate: float = 0.2):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.training = True
        
        total_blocks = sum(_round_repeats(r, depth_coefficient)
                          for r in self._LITE_BLOCK_REPEATS)
        
        # Stem (使用ReLU6)
        stem_channels = _round_filters(32, width_coefficient)
        self.stem = EfficientNetStem(
            in_channels=3, out_channels=stem_channels, activation='relu6'
        )
        
        # 构建MBConv块 (无SE, 使用ReLU6)
        self.blocks = []
        block_idx = 0
        
        for stage_idx, (in_ch, out_ch, k, s, expand) in enumerate(self._LITE_BLOCK_CONFIGS):
            in_c = _round_filters(in_ch, width_coefficient)
            out_c = _round_filters(out_ch, width_coefficient)
            repeats = _round_repeats(self._LITE_BLOCK_REPEATS[stage_idx], depth_coefficient)
            
            for i in range(repeats):
                stride = s if i == 0 else 1
                input_c = in_c if i == 0 else out_c
                
                dc_rate = 0.2 * block_idx / max(1, total_blocks - 1)
                
                block = MBConvBlock(
                    in_channels=input_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=stride,
                    expand_ratio=expand,
                    se_ratio=0.0,  # 无SE
                    drop_connect_rate=dc_rate,
                    activation='relu6'  # 使用ReLU6
                )
                self.blocks.append(block)
                block_idx += 1
        
        # Head
        last_channels = _round_filters(320, width_coefficient)
        final_channels = _round_filters(1280, width_coefficient)
        self.head = EfficientNetHead(
            in_channels=last_channels,
            final_channels=final_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation='relu6'
        )
        self.head.training = self.training
        
        for block in self.blocks:
            block.training = self.training
        
        self._total_blocks = total_blocks
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[float], 'torch.Tensor']:
        """前向传播"""
        x = self.stem.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        logits = self.head.forward(x)
        return logits
    
    def extract_features(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """提取特征"""
        x = self.stem.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        return x
    
    def predict(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> int:
        """预测类别"""
        logits = self.forward(x)
        return max(range(len(logits)), key=lambda i: logits[i])
    
    def set_training(self, mode: bool):
        """设置训练/评估模式"""
        self.training = mode
        self.head.training = mode
        for block in self.blocks:
            block.training = mode
    
    def count_parameters(self) -> int:
        """计算总参数量"""
        count = 0
        stem_c = _round_filters(32, self.width_coefficient)
        count += stem_c * 3 * 3 * 3 + stem_c + stem_c * 2
        
        for block in self.blocks:
            exp_c = block.expanded_channels
            in_c = block.in_channels
            out_c = block.out_channels
            k = block.kernel_size
            
            if block.has_expand:
                count += exp_c * in_c + exp_c + exp_c * 2
            count += exp_c * k * k + exp_c + exp_c * 2
            count += out_c * exp_c + out_c + out_c * 2
        
        last_c = _round_filters(320, self.width_coefficient)
        final_c = _round_filters(1280, self.width_coefficient)
        count += final_c * last_c + final_c + final_c * 2
        count += self.num_classes * final_c + self.num_classes
        
        return count


# EfficientNet-Lite 预定义模型
_EFFICIENTNET_LITE_COEFFICIENTS = {
    'lite0': (1.0, 1.0, 224, 0.2),
    'lite1': (1.0, 1.1, 240, 0.2),
    'lite2': (1.1, 1.2, 260, 0.3),
    'lite3': (1.2, 1.4, 300, 0.3),
    'lite4': (1.4, 1.8, 380, 0.4),
}


def efficientnet_lite0(num_classes: int = 1000, dropout_rate: float = 0.2) -> EfficientNetLite:
    """EfficientNet-Lite0: ~4.7M参数"""
    return EfficientNetLite(
        width_coefficient=1.0, depth_coefficient=1.0,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


def efficientnet_lite1(num_classes: int = 1000, dropout_rate: float = 0.2) -> EfficientNetLite:
    """EfficientNet-Lite1: ~5.4M参数"""
    return EfficientNetLite(
        width_coefficient=1.0, depth_coefficient=1.1,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


def efficientnet_lite2(num_classes: int = 1000, dropout_rate: float = 0.3) -> EfficientNetLite:
    """EfficientNet-Lite2: ~6.5M参数"""
    return EfficientNetLite(
        width_coefficient=1.1, depth_coefficient=1.2,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


def efficientnet_lite3(num_classes: int = 1000, dropout_rate: float = 0.3) -> EfficientNetLite:
    """EfficientNet-Lite3: ~8.2M参数"""
    return EfficientNetLite(
        width_coefficient=1.2, depth_coefficient=1.4,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


def efficientnet_lite4(num_classes: int = 1000, dropout_rate: float = 0.4) -> EfficientNetLite:
    """EfficientNet-Lite4: ~11M参数"""
    return EfficientNetLite(
        width_coefficient=1.4, depth_coefficient=1.8,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


# ============================================================
# 复合缩放 (Compound Scaling) 工具
# ============================================================

class CompoundScaler:
    """
    复合缩放器 - EfficientNet的核心创新
    
    统一缩放网络的三个维度:
    - 深度 (depth): 网络层数
    - 宽度 (width): 通道数
    - 分辨率 (resolution): 输入图像大小
    
    缩放公式:
        depth = alpha ^ phi
        width = beta ^ phi
        resolution = gamma ^ phi
        约束: alpha * beta^2 * gamma^2 ≈ 2
    
    参数:
        alpha: 深度缩放因子
        beta: 宽度缩放因子
        gamma: 分辨率缩放因子
        phi: 复合系数 (控制整体模型大小)
    """
    
    def __init__(self, alpha: float = 1.2, beta: float = 1.1,
                 gamma: float = 1.15, phi: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
        
        # 验证约束
        constraint = alpha * (beta ** 2) * (gamma ** 2)
        self.constraint_value = constraint
    
    def get_depth_coefficient(self) -> float:
        """获取深度缩放系数"""
        return self.alpha ** self.phi
    
    def get_width_coefficient(self) -> float:
        """获取宽度缩放系数"""
        return self.beta ** self.phi
    
    def get_resolution(self, base_resolution: int = 224) -> int:
        """获取缩放后的分辨率"""
        return int(base_resolution * (self.gamma ** self.phi))
    
    def get_scaling_summary(self) -> dict:
        """获取缩放摘要"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'phi': self.phi,
            'depth_coefficient': self.get_depth_coefficient(),
            'width_coefficient': self.get_width_coefficient(),
            'resolution': self.get_resolution(),
            'constraint_alpha_beta2_gamma2': self.constraint_value
        }
    
    @staticmethod
    def from_model_size(target_flops: float, base_flops: float = 0.39) -> 'CompoundScaler':
        """
        根据目标FLOPs自动计算复合缩放系数
        
        使用网格搜索找到满足约束的最优 (alpha, beta, gamma, phi)
        
        参数:
            target_flops: 目标计算量 (BFLOPs)
            base_flops: 基线模型 (B0) 的计算量
        Returns:
            CompoundScaler实例
        """
        best_scaler = None
        best_diff = float('inf')
        
        # 网格搜索
        for phi in [i * 0.1 for i in range(5, 30)]:
            for alpha in [1.0, 1.1, 1.2, 1.3]:
                for beta in [1.0, 1.05, 1.1, 1.15]:
                    for gamma in [1.0, 1.05, 1.1, 1.15]:
                        constraint = alpha * (beta ** 2) * (gamma ** 2)
                        if abs(constraint - 2.0) > 0.3:
                            continue
                        
                        # 近似FLOPs缩放
                        depth = alpha ** phi
                        width = beta ** phi
                        resolution = gamma ** phi
                        approx_flops = base_flops * (depth * width ** 2 * resolution ** 2)
                        
                        diff = abs(approx_flops - target_flops)
                        if diff < best_diff:
                            best_diff = diff
                            best_scaler = CompoundScaler(alpha, beta, gamma, phi)
        
        if best_scaler is None:
            # 回退到默认
            phi = math.log2(target_flops / base_flops) / 2.0
            best_scaler = CompoundScaler(1.2, 1.1, 1.15, phi)
        
        return best_scaler


# ============================================================
# EfficientNetV2 (改进版)
# ============================================================

class FusedMBConvBlock:
    """
    Fused-MBConv块 - EfficientNetV2的改进
    
    与标准MBConv的区别:
    - 将expand和depthwise融合为一个3x3卷积
    - 减少内存访问开销
    - 适合早期层 (大通道数时更高效)
    
    结构:
        input -> fused_conv(3x3, expand) -> BN -> Swish -> SE -> project(1x1) -> BN
              -> drop_connect -> output (+ residual)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expand_ratio: float = 1.0, se_ratio: float = 0.25,
                 drop_connect_rate: float = 0.0, activation: str = 'swish'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        self.training = True
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        expanded_channels = int(in_channels * expand_ratio)
        self.expanded_channels = expanded_channels
        
        # 融合卷积 (expand + depthwise 合并为一个卷积)
        fused_padding = _same_padding(0, kernel_size, stride)
        self.fused_conv = Conv2dBNAct(
            in_channels, expanded_channels, kernel_size=kernel_size,
            stride=stride, padding=fused_padding, activation=activation
        )
        
        # SE模块
        self.has_se = se_ratio > 0
        if self.has_se:
            self.se = SqueezeExcitation(expanded_channels,
                                         reduction=int(1.0 / se_ratio),
                                         activation=activation)
        
        # 投影卷积
        self.project_conv = Conv2dBNAct(
            expanded_channels, out_channels, kernel_size=1,
            stride=1, padding=0, activation='none'
        )
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """前向传播"""
        identity = x
        
        # 融合卷积
        x = self.fused_conv.forward(x)
        
        # SE
        if self.has_se:
            x = self.se.forward(x)
        
        # 投影
        x = self.project_conv.forward(x)
        
        # DropConnect
        if self.drop_connect_rate > 0:
            C = len(x)
            H = len(x[0])
            W = len(x[0][0])
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        x[c][h][w] = _drop_connect(
                            x[c][h][w], 1.0 - self.drop_connect_rate, self.training
                        )
        
        # 残差
        if self.use_residual:
            C = len(x)
            H = len(x[0])
            W = len(x[0][0])
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        x[c][h][w] += identity[c][h][w]
        
        return x


class EfficientNetV2:
    """
    EfficientNetV2 - 改进版EfficientNet
    
    改进点:
    1. Fused-MBConv: 在早期层使用融合卷积，减少内存开销
    2. 渐进式学习率: 训练初期使用小学习率warm-up
    3. 预正则化: 在深度卷积后添加BN层
    
    网络结构:
        Stem -> [FusedMBConv x N1] -> [MBConv x N2] -> Head
    
    参数:
        width_coefficient: 宽度缩放系数
        depth_coefficient: 深度缩放系数
        num_classes: 分类数
        dropout_rate: Dropout率
    """
    
    # V2-S的基准配置
    # (type, in_ch, out_ch, kernel, stride, expand, se_ratio)
    # type: 'fused' = FusedMBConv, 'mbconv' = MBConv
    _V2_BLOCK_CONFIGS = [
        ('fused', 24, 24, 3, 1, 1, 0.0),     # Stage 1: Fused
        ('fused', 24, 48, 3, 2, 4, 0.0),     # Stage 2: Fused
        ('fused', 48, 64, 3, 2, 4, 0.0),     # Stage 3: Fused
        ('mbconv', 64, 128, 3, 2, 4, 0.25),  # Stage 4: MBConv
        ('mbconv', 128, 160, 3, 2, 6, 0.25), # Stage 5: MBConv
        ('mbconv', 160, 256, 3, 2, 6, 0.25), # Stage 6: MBConv
        ('mbconv', 256, 512, 3, 1, 6, 0.25), # Stage 7: MBConv
    ]
    
    _V2_BLOCK_REPEATS = [2, 4, 4, 6, 9, 15, 1]  # V2-S基准
    
    def __init__(self, width_coefficient: float = 1.0, depth_coefficient: float = 1.0,
                 num_classes: int = 1000, dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.training = True
        
        total_blocks = sum(_round_repeats(r, depth_coefficient)
                          for r in self._V2_BLOCK_REPEATS)
        
        # Stem (V2使用更大的初始卷积)
        stem_channels = _round_filters(24, width_coefficient)
        self.stem = EfficientNetStem(
            in_channels=3, out_channels=stem_channels, activation='swish'
        )
        
        # 构建块
        self.blocks = []
        block_idx = 0
        
        for stage_idx, (block_type, in_ch, out_ch, k, s, expand, se) in enumerate(
                self._V2_BLOCK_CONFIGS):
            in_c = _round_filters(in_ch, width_coefficient)
            out_c = _round_filters(out_ch, width_coefficient)
            repeats = _round_repeats(self._V2_BLOCK_REPEATS[stage_idx], depth_coefficient)
            
            for i in range(repeats):
                stride = s if i == 0 else 1
                input_c = in_c if i == 0 else out_c
                
                dc_rate = drop_connect_rate * block_idx / max(1, total_blocks - 1)
                
                if block_type == 'fused':
                    block = FusedMBConvBlock(
                        in_channels=input_c, out_channels=out_c,
                        kernel_size=k, stride=stride, expand_ratio=expand,
                        se_ratio=se, drop_connect_rate=dc_rate, activation='swish'
                    )
                else:
                    block = MBConvBlock(
                        in_channels=input_c, out_channels=out_c,
                        kernel_size=k, stride=stride, expand_ratio=expand,
                        se_ratio=se, drop_connect_rate=dc_rate, activation='swish'
                    )
                
                self.blocks.append(block)
                block.training = self.training
                block_idx += 1
        
        # Head
        last_c = _round_filters(512, width_coefficient)
        final_c = _round_filters(1280, width_coefficient)
        self.head = EfficientNetHead(
            in_channels=last_c, final_channels=final_c,
            num_classes=num_classes, dropout_rate=dropout_rate,
            activation='swish'
        )
        self.head.training = self.training
        
        self._total_blocks = total_blocks
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> Union[List[float], 'torch.Tensor']:
        """前向传播"""
        x = self.stem.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        return self.head.forward(x)
    
    def extract_features(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """提取特征"""
        x = self.stem.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        return x
    
    def predict(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> int:
        """预测类别"""
        logits = self.forward(x)
        return max(range(len(logits)), key=lambda i: logits[i])
    
    def set_training(self, mode: bool):
        """设置训练/评估模式"""
        self.training = mode
        self.head.training = mode
        for block in self.blocks:
            block.training = mode


def efficientnet_v2_s(num_classes: int = 1000, dropout_rate: float = 0.2) -> EfficientNetV2:
    """EfficientNetV2-S: ~21M参数"""
    return EfficientNetV2(
        width_coefficient=1.0, depth_coefficient=1.0,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


def efficientnet_v2_m(num_classes: int = 1000, dropout_rate: float = 0.3) -> EfficientNetV2:
    """EfficientNetV2-M: ~54M参数"""
    return EfficientNetV2(
        width_coefficient=1.4, depth_coefficient=1.0,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


def efficientnet_v2_l(num_classes: int = 1000, dropout_rate: float = 0.4) -> EfficientNetV2:
    """EfficientNetV2-L: ~119M参数"""
    return EfficientNetV2(
        width_coefficient=2.0, depth_coefficient=1.0,
        num_classes=num_classes, dropout_rate=dropout_rate
    )


# ============================================================
# 渐进式学习率调度 (EfficientNetV2使用)
# ============================================================

class ProgressiveLearningRateScheduler:
    """
    渐进式学习率调度器
    
    EfficientNetV2的训练策略:
    1. 初始阶段: 使用较小的图像分辨率和正则化强度
    2. 渐进增大: 逐步增大分辨率、dropout、mixup、cutmix等
    3. 最终阶段: 使用完整配置训练
    
    参数:
        base_lr: 基础学习率
        warmup_epochs: warm-up轮数
        total_epochs: 总训练轮数
        initial_resolution: 初始分辨率
        final_resolution: 最终分辨率
        initial_drop_rate: 初始dropout率
        final_drop_rate: 最终dropout率
        initial_mixup: 初始mixup强度
        final_mixup: 最终mixup强度
        initial_cutmix: 初始cutmix强度
        final_cutmix: 最终cutmix强度
    """
    
    def __init__(self, base_lr: float = 0.001, warmup_epochs: int = 5,
                 total_epochs: int = 300, initial_resolution: int = 128,
                 final_resolution: int = 384, initial_drop_rate: float = 0.1,
                 final_drop_rate: float = 0.4, initial_mixup: float = 0.0,
                 final_mixup: float = 0.8, initial_cutmix: float = 0.0,
                 final_cutmix: float = 1.0):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_resolution = initial_resolution
        self.final_resolution = final_resolution
        self.initial_drop_rate = initial_drop_rate
        self.final_drop_rate = final_drop_rate
        self.initial_mixup = initial_mixup
        self.final_mixup = final_mixup
        self.initial_cutmix = initial_cutmix
        self.final_cutmix = final_cutmix
    
    def get_lr(self, epoch: int, step: int = 0, total_steps: int = 1000) -> float:
        """
        获取当前学习率
        
        使用余弦退火 + 线性warm-up
        
        参数:
            epoch: 当前轮次
            step: 当前步数
            total_steps: 每轮总步数
        Returns:
            当前学习率
        """
        if epoch < self.warmup_epochs:
            # 线性warm-up
            progress = (epoch * total_steps + step) / (self.warmup_epochs * total_steps)
            return self.base_lr * progress
        else:
            # 余弦退火
            adjusted_epoch = epoch - self.warmup_epochs
            total_adjusted = self.total_epochs - self.warmup_epochs
            progress = adjusted_epoch / max(1, total_adjusted)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.base_lr * cosine_decay
    
    def get_resolution(self, epoch: int) -> int:
        """获取当前分辨率"""
        if epoch < self.warmup_epochs:
            return self.initial_resolution
        
        progress = min(1.0, (epoch - self.warmup_epochs) /
                       max(1, self.total_epochs - self.warmup_epochs))
        return int(self.initial_resolution +
                   (self.final_resolution - self.initial_resolution) * progress)
    
    def get_dropout_rate(self, epoch: int) -> float:
        """获取当前dropout率"""
        progress = min(1.0, epoch / max(1, self.total_epochs))
        return self.initial_drop_rate + (self.final_drop_rate - self.initial_drop_rate) * progress
    
    def get_mixup_alpha(self, epoch: int) -> float:
        """获取当前mixup alpha"""
        progress = min(1.0, epoch / max(1, self.total_epochs))
        return self.initial_mixup + (self.final_mixup - self.initial_mixup) * progress
    
    def get_cutmix_alpha(self, epoch: int) -> float:
        """获取当前cutmix alpha"""
        progress = min(1.0, epoch / max(1, self.total_epochs))
        return self.initial_cutmix + (self.final_cutmix - self.initial_cutmix) * progress
    
    def get_training_config(self, epoch: int, step: int = 0,
                            total_steps: int = 1000) -> dict:
        """
        获取当前epoch的完整训练配置
        
        Returns:
            包含lr, resolution, dropout, mixup, cutmix的配置字典
        """
        return {
            'learning_rate': self.get_lr(epoch, step, total_steps),
            'resolution': self.get_resolution(epoch),
            'dropout_rate': self.get_dropout_rate(epoch),
            'mixup_alpha': self.get_mixup_alpha(epoch),
            'cutmix_alpha': self.get_cutmix_alpha(epoch),
            'epoch': epoch,
            'phase': 'warmup' if epoch < self.warmup_epochs else 'training'
        }


# ============================================================
# Mixup和Cutmix数据增强
# ============================================================

class MixupAugmentation:
    """
    Mixup数据增强
    论文: "mixup: Beyond Empirical Risk Minimization"
    
    公式: x_mixed = lambda * x_i + (1 - lambda) * x_j
          y_mixed = lambda * y_i + (1 - lambda) * y_j
    
    参数:
        alpha: Beta分布参数 (0表示不使用)
        num_classes: 分类数
    """
    
    def __init__(self, alpha: float = 0.8, num_classes: int = 1000):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def _sample_lambda(self) -> float:
        """从Beta分布采样混合系数"""
        if self.alpha <= 0:
            return 1.0
        # 使用Gamma分布模拟Beta分布
        x = random.gammavariate(self.alpha, 1.0)
        y = random.gammavariate(self.alpha, 1.0)
        return x / (x + y)
    
    def __call__(self, x1: List[Union[List[List[float]], 'torch.Tensor']], x2: List[Union[List[List[float]], 'torch.Tensor']],
                 label1: int, label2: int) -> Tuple[List[Union[List[List[float]], 'torch.Tensor']], Union[List[float], 'torch.Tensor']]:
        """
        应用Mixup
        
        Args:
            x1: 第一张图像 [C, H, W]
            x2: 第二张图像 [C, H, W]
            label1: 第一张标签
            label2: 第二张标签
        Returns:
            (mixed_image, mixed_label)
        """
        lam = self._sample_lambda()
        
        C = len(x1)
        H = len(x1[0])
        W = len(x1[0][0])
        
        mixed = []
        for c in range(C):
            channel = []
            for h in range(H):
                row = []
                for w in range(W):
                    val = lam * x1[c][h][w] + (1.0 - lam) * x2[c][h][w]
                    row.append(val)
                channel.append(row)
            mixed.append(channel)
        
        # 混合标签 (one-hot)
        label = [0.0] * self.num_classes
        label[label1] = lam
        label[label2] = 1.0 - lam
        
        return mixed, label


class CutMixAugmentation:
    """
    CutMix数据增强
    论文: "CutMix: Regularization Strategy to Train Strong Classifiers"
    
    在一张图上裁剪一个矩形区域，用另一张图对应区域填充
    
    参数:
        alpha: Beta分布参数
        num_classes: 分类数
    """
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 1000):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def _sample_lambda(self) -> float:
        if self.alpha <= 0:
            return 1.0
        x = random.gammavariate(self.alpha, 1.0)
        y = random.gammavariate(self.alpha, 1.0)
        return x / (x + y)
    
    def _rand_bbox(self, W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
        """
        随机生成裁剪框
        
        Args:
            W: 图像宽度
            H: 图像高度
            lam: 面积比例
        Returns:
            (x1, y1, x2, y2) 裁剪框坐标
        """
        cut_ratio = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)
        
        return x1, y1, x2, y2
    
    def __call__(self, x1: List[Union[List[List[float]], 'torch.Tensor']], x2: List[Union[List[List[float]], 'torch.Tensor']],
                 label1: int, label2: int) -> Tuple[List[Union[List[List[float]], 'torch.Tensor']], Union[List[float], 'torch.Tensor']]:
        """
        应用CutMix
        
        Args:
            x1: 第一张图像 (背景)
            x2: 第二张图像 (裁剪源)
            label1: 第一张标签
            label2: 第二张标签
        Returns:
            (mixed_image, mixed_label)
        """
        lam = self._sample_lambda()
        
        C = len(x1)
        H = len(x1[0])
        W = len(x1[0][0])
        
        # 复制x1
        mixed = [[row[:] for row in channel] for channel in x1]
        
        # 获取裁剪框
        x1_bb, y1_bb, x2_bb, y2_bb = self._rand_bbox(W, H, lam)
        
        # 计算实际面积比例
        cut_area = (x2_bb - x1_bb) * (y2_bb - y1_bb)
        total_area = W * H
        adjusted_lam = 1.0 - cut_area / total_area
        
        # 填充裁剪区域
        for c in range(C):
            for h in range(y1_bb, y2_bb):
                for w in range(x1_bb, x2_bb):
                    if 0 <= h < H and 0 <= w < W:
                        mixed[c][h][w] = x2[c][h][w]
        
        # 混合标签
        label = [0.0] * self.num_classes
        label[label1] = adjusted_lam
        label[label2] = 1.0 - adjusted_lam
        
        return mixed, label


# ============================================================
# 模型工厂函数
# ============================================================

def create_efficientnet(variant: str = 'b0', num_classes: int = 1000,
                        pretrained: bool = False) -> object:
    """
    创建EfficientNet模型的工厂函数
    
    参数:
        variant: 模型变体 ('b0'~'b7', 'lite0'~'lite4', 'v2_s', 'v2_m', 'v2_l')
        num_classes: 分类数
        pretrained: 是否加载预训练权重 (本实现中为False, 仅保留接口)
    Returns:
        EfficientNet模型实例
    """
    variant = variant.lower()
    
    # EfficientNet-B系列
    b_models = {
        'b0': (efficientnet_b0, 0.2),
        'b1': (efficientnet_b1, 0.2),
        'b2': (efficientnet_b2, 0.3),
        'b3': (efficientnet_b3, 0.3),
        'b4': (efficientnet_b4, 0.4),
        'b5': (efficientnet_b5, 0.4),
        'b6': (efficientnet_b6, 0.5),
        'b7': (efficientnet_b7, 0.5),
    }
    
    # EfficientNet-Lite系列
    lite_models = {
        'lite0': (efficientnet_lite0, 0.2),
        'lite1': (efficientnet_lite1, 0.2),
        'lite2': (efficientnet_lite2, 0.3),
        'lite3': (efficientnet_lite3, 0.3),
        'lite4': (efficientnet_lite4, 0.4),
    }
    
    # EfficientNetV2系列
    v2_models = {
        'v2_s': (efficientnet_v2_s, 0.2),
        'v2_m': (efficientnet_v2_m, 0.3),
        'v2_l': (efficientnet_v2_l, 0.4),
    }
    
    if variant in b_models:
        create_fn, dropout = b_models[variant]
        return create_fn(num_classes=num_classes, dropout_rate=dropout)
    elif variant in lite_models:
        create_fn, dropout = lite_models[variant]
        return create_fn(num_classes=num_classes, dropout_rate=dropout)
    elif variant in v2_models:
        create_fn, dropout = v2_models[variant]
        return create_fn(num_classes=num_classes, dropout_rate=dropout)
    else:
        available = sorted(list(b_models.keys()) + list(lite_models.keys()) +
                          list(v2_models.keys()))
        raise ValueError(f"Unknown variant '{variant}'. Available: {available}")


# ============================================================
# 模型信息查询
# ============================================================

def get_efficientnet_info() -> dict:
    """
    获取所有EfficientNet变体的信息
    
    Returns:
        包含各变体参数量和推荐输入分辨率的字典
    """
    info = {}
    
    # B系列
    b_variants = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
    for v in b_variants:
        w, d, res, drop = _EFFICIENTNET_COEFFICIENTS[v]
        model = create_efficientnet(v)
        info[v] = {
            'width_coeff': w,
            'depth_coeff': d,
            'resolution': res,
            'dropout': drop,
            'parameters': model.count_parameters(),
            'type': 'EfficientNet-B'
        }
    
    # Lite系列
    lite_variants = ['lite0', 'lite1', 'lite2', 'lite3', 'lite4']
    for v in lite_variants:
        w, d, res, drop = _EFFICIENTNET_LITE_COEFFICIENTS[v]
        model = create_efficientnet(v)
        info[v] = {
            'width_coeff': w,
            'depth_coeff': d,
            'resolution': res,
            'dropout': drop,
            'parameters': model.count_parameters(),
            'type': 'EfficientNet-Lite'
        }
    
    # V2系列
    v2_info = {
        'v2_s': {'width_coeff': 1.0, 'depth_coeff': 1.0, 'resolution': 384,
                 'dropout': 0.2, 'type': 'EfficientNetV2'},
        'v2_m': {'width_coeff': 1.4, 'depth_coeff': 1.0, 'resolution': 384,
                 'dropout': 0.3, 'type': 'EfficientNetV2'},
        'v2_l': {'width_coeff': 2.0, 'depth_coeff': 1.0, 'resolution': 384,
                 'dropout': 0.4, 'type': 'EfficientNetV2'},
    }
    for v, vi in v2_info.items():
        model = create_efficientnet(v)
        vi['parameters'] = model.count_parameters()
        info[v] = vi
    
    return info


def print_efficientnet_comparison():
    """打印EfficientNet各变体对比表"""
    info = get_efficientnet_info()
    
    print(f"{'Variant':<12} {'Type':<20} {'Width':<8} {'Depth':<8} "
          f"{'Resolution':<12} {'Dropout':<8} {'Parameters':>12}")
    print("-" * 90)
    
    for variant in sorted(info.keys()):
        vi = info[variant]
        print(f"{variant:<12} {vi['type']:<20} {vi['width_coeff']:<8.1f} "
              f"{vi['depth_coeff']:<8.1f} {vi['resolution']:<12} "
              f"{vi['dropout']:<8.1f} {vi['parameters']:>12,}")
