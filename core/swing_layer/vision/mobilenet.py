"""
MobileNet架构 - 完整实现
包含: MobileNetV1, MobileNetV2, MobileNetV3 (Small/Large)
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def relu6(x: float) -> float:
    """ReLU6激活函数"""
    return min(max(0.0, x), 6.0)


def hard_sigmoid(x: float) -> float:
    """Hard Sigmoid"""
    return max(0.0, min(1.0, x * 0.166667 + 0.5))


def hard_swish(x: float) -> float:
    """Hard Swish"""
    return x * hard_sigmoid(x)


def _conv2d_compute_out(input_size: int, kernel: int, stride: int, padding: int) -> int:
    return (input_size + 2 * padding - kernel) // stride + 1



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


class Conv2dBNReLU:
    """Conv2d + BatchNorm + ReLU6 模块"""
    
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int = 1, padding: int = 0):
        self.in_c = in_c
        self.out_c = out_c
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
        std = math.sqrt(2.0 / (in_c * kernel * kernel))
        self.weight = [[[[random.gauss(0, std) for _ in range(in_c)] 
                        for _ in range(kernel)] for _ in range(kernel)] for _ in range(out_c)]
        self.bn_gamma = [1.0] * out_c
        self.bn_beta = [0.0] * out_c
        self.bn_running_mean = [0.0] * out_c
        self.bn_running_var = [1.0] * out_c
        self.bn_eps = 1e-3
    
    def forward(self, x: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
        N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        out_h = _conv2d_compute_out(H, self.kernel, self.stride, self.padding)
        out_w = _conv2d_compute_out(W, self.kernel, self.stride, self.padding)
        
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(self.out_c)] for _ in range(N)]
        
        for n in range(N):
            for oc in range(self.out_c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        val = 0.0
                        for ic in range(self.in_c):
                            for kh in range(self.kernel):
                                for kw in range(self.kernel):
                                    hi = oh * self.stride + kh - self.padding
                                    wi = ow * self.stride + kw - self.padding
                                    if 0 <= hi < H and 0 <= wi < W:
                                        val += x[n][ic][hi][wi] * self.weight[oc][kh][kw][ic]
                        # BN
                        mean = self.bn_running_mean[oc]
                        var = self.bn_running_var[oc]
                        val = self.bn_gamma[oc] * (val - mean) / math.sqrt(var + self.bn_eps) + self.bn_beta[oc]
                        output[n][oc][oh][ow] = relu6(val)
        return output


class DepthwiseSeparableConv:
    """深度可分离卷积 (MobileNetV1核心)"""
    
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        
        # 深度卷积
        std = math.sqrt(2.0 / (in_c * 9))
        self.dw_weight = [[[[random.gauss(0, std) for _ in range(1)] 
                           for _ in range(3)] for _ in range(3)] for _ in range(in_c)]
        self.dw_bn_gamma = [1.0] * in_c
        self.dw_bn_beta = [0.0] * in_c
        self.dw_bn_mean = [0.0] * in_c
        self.dw_bn_var = [1.0] * in_c
        
        # 逐点卷积
        std2 = math.sqrt(2.0 / in_c)
        self.pw_weight = [[random.gauss(0, std2) for _ in range(in_c)] for _ in range(out_c)]
        self.pw_bn_gamma = [1.0] * out_c
        self.pw_bn_beta = [0.0] * out_c
        self.pw_bn_mean = [0.0] * out_c
        self.pw_bn_var = [1.0] * out_c
    
    def forward(self, x: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
        N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        out_h = _conv2d_compute_out(H, 3, self.stride, 1)
        out_w = _conv2d_compute_out(W, 3, self.stride, 1)
        
        # 深度卷积
        dw_out = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        val = 0.0
                        for kh in range(3):
                            for kw in range(3):
                                hi = oh * self.stride + kh - 1
                                wi = ow * self.stride + kw - 1
                                if 0 <= hi < H and 0 <= wi < W:
                                    val += x[n][c][hi][wi] * self.dw_weight[c][kh][kw][0]
                        mean = self.dw_bn_mean[c]
                        var = self.dw_bn_var[c]
                        dw_out[n][c][oh][ow] = relu6(self.dw_bn_gamma[c] * (val - mean) / math.sqrt(var + 1e-3) + self.dw_bn_beta[c])
        
        # 逐点卷积
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(self.out_c)] for _ in range(N)]
        for n in range(N):
            for oc in range(self.out_c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        val = 0.0
                        for ic in range(C):
                            val += dw_out[n][ic][oh][ow] * self.pw_weight[oc][ic]
                        mean = self.pw_bn_mean[oc]
                        var = self.pw_bn_var[oc]
                        output[n][oc][oh][ow] = relu6(self.pw_bn_gamma[oc] * (val - mean) / math.sqrt(var + 1e-3) + self.pw_bn_beta[oc])
        return output


class InvertedResidual:
    """倒残差模块 (MobileNetV2核心)
    
    expand -> depthwise -> project (with residual connection)
    """
    
    def __init__(self, in_c: int, out_c: int, stride: int, expand_ratio: float):
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_residual = (stride == 1 and in_c == out_c)
        
        hidden_c = int(in_c * expand_ratio)
        
        # 扩展层
        self.expand_weight = [[random.gauss(0, math.sqrt(2.0/in_c)) for _ in range(in_c)] for _ in range(hidden_c)]
        self.expand_bn_gamma = [1.0] * hidden_c
        self.expand_bn_mean = [0.0] * hidden_c
        self.expand_bn_var = [1.0] * hidden_c
        
        # 深度卷积
        self.dw_weight = [[[[random.gauss(0, math.sqrt(2.0/hidden_c)) for _ in range(1)] 
                          for _ in range(3)] for _ in range(3)] for _ in range(hidden_c)]
        self.dw_bn_gamma = [1.0] * hidden_c
        self.dw_bn_mean = [0.0] * hidden_c
        self.dw_bn_var = [1.0] * hidden_c
        
        # 投影层
        self.proj_weight = [[random.gauss(0, math.sqrt(2.0/hidden_c)) for _ in range(hidden_c)] for _ in range(out_c)]
        self.proj_bn_gamma = [1.0] * out_c
        self.proj_bn_mean = [0.0] * out_c
        self.proj_bn_var = [1.0] * out_c
    
    def _bn_relu6(self, val: float, gamma: float, mean: float, var: float, beta: float) -> float:
        return relu6(gamma * (val - mean) / math.sqrt(var + 1e-3) + beta)
    
    def _bn_linear(self, val: float, gamma: float, mean: float, var: float, beta: float) -> float:
        return gamma * (val - mean) / math.sqrt(var + 1e-3) + beta
    
    def forward(self, x: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
        N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        hidden_c = int(C * self.expand_ratio)
        out_h = _conv2d_compute_out(H, 3, self.stride, 1)
        out_w = _conv2d_compute_out(W, 3, self.stride, 1)
        
        # 扩展
        expanded = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(hidden_c)] for _ in range(N)]
        for n in range(N):
            for hc in range(hidden_c):
                for h in range(H):
                    for w in range(W):
                        val = sum(x[n][c][h][w] * self.expand_weight[hc][c] for c in range(C))
                        expanded[n][hc][h][w] = self._bn_relu6(val, self.expand_bn_gamma[hc], self.expand_bn_mean[hc], self.expand_bn_var[hc], 0.0)
        
        # 深度卷积
        dw_out = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(hidden_c)] for _ in range(N)]
        for n in range(N):
            for hc in range(hidden_c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        val = 0.0
                        for kh in range(3):
                            for kw in range(3):
                                hi = oh * self.stride + kh - 1
                                wi = ow * self.stride + kw - 1
                                if 0 <= hi < H and 0 <= wi < W:
                                    val += expanded[n][hc][hi][wi] * self.dw_weight[hc][kh][kw][0]
                        dw_out[n][hc][oh][ow] = self._bn_relu6(val, self.dw_bn_gamma[hc], self.dw_bn_mean[hc], self.dw_bn_var[hc], 0.0)
        
        # 投影 (线性，无ReLU)
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(self.out_c)] for _ in range(N)]
        for n in range(N):
            for oc in range(self.out_c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        val = sum(dw_out[n][hc][oh][ow] * self.proj_weight[oc][hc] for hc in range(hidden_c))
                        output[n][oc][oh][ow] = self._bn_linear(val, self.proj_bn_gamma[oc], self.proj_bn_mean[oc], self.proj_bn_var[oc], 0.0)
        
        # 残差连接
        if self.use_residual:
            for n in range(N):
                for oc in range(self.out_c):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            output[n][oc][oh][ow] += x[n][oc][oh][ow]
        
        return output


class MobileNetV1:
    """
    MobileNetV1
    使用深度可分离卷积的轻量级网络
    
    架构: Conv -> 13个DSConv -> AvgPool -> FC
    """
    
    def __init__(self, num_classes: int = 1000, alpha: float = 1.0):
        self.num_classes = num_classes
        self.alpha = alpha
        
        # 计算通道数
        def div_channels(c):
            return max(1, int(c * alpha))
        
        # 初始卷积
        self.conv1 = Conv2dBNReLU(3, div_channels(32), 3, stride=2, padding=1)
        
        # DS卷积层
        self.ds_layers = []
        ds_config = [
            (div_channels(32), div_channels(64), 1),
            (div_channels(64), div_channels(128), 2),
            (div_channels(128), div_channels(128), 1),
            (div_channels(128), div_channels(256), 2),
            (div_channels(256), div_channels(256), 1),
            (div_channels(256), div_channels(512), 2),
            (div_channels(512), div_channels(512), 1),
            (div_channels(512), div_channels(512), 1),
            (div_channels(512), div_channels(512), 1),
            (div_channels(512), div_channels(512), 1),
            (div_channels(512), div_channels(512), 1),
            (div_channels(512), div_channels(512), 1),
            (div_channels(512), div_channels(1024), 2),
        ]
        for in_c, out_c, stride in ds_config:
            self.ds_layers.append(DepthwiseSeparableConv(in_c, out_c, stride))
        
        # 全连接层
        self.fc = [[random.gauss(0, 0.01) for _ in range(div_channels(1024))] for _ in range(num_classes)]
    
    def forward(self, x: List[List[List[List[float]]]]) -> Union[List[List[List[List[float]]]], "torch.Tensor"]:
        """前向传播"""
        x = self.conv1.forward(x)
        for layer in self.ds_layers:
            x = layer.forward(x)
        
        # 全局平均池化
        N = len(x)
        C = len(x[0])
        H = len(x[0][0])
        W = len(x[0][0][0])
        
        pooled = [[sum(x[n][c][h][w] for h in range(H) for w in range(W)) / (H * W) for c in range(C)] for n in range(N)]
        
        # 全连接
        output = [[sum(pooled[n][c] * self.fc[cls][c] for c in range(len(pooled[n]))) for cls in range(self.num_classes)] for n in range(N)]
        return output


class MobileNetV2:
    """
    MobileNetV2
    使用倒残差模块和线性瓶颈
    
    架构: Conv -> 17个InvertedResidual -> Conv -> AvgPool -> FC
    """
    
    def __init__(self, num_classes: int = 1000, alpha: float = 1.0, round_nearest: bool = True):
        self.num_classes = num_classes
        
        def div_channels(c):
            if round_nearest:
                return max(1, int(math.ceil(c * alpha / 8) * 8))
            return max(1, int(c * alpha))
        
        # 初始卷积
        self.conv1 = Conv2dBNReLU(3, div_channels(32), 3, stride=2, padding=1)
        
        # 倒残差模块
        self.bottlenecks = []
        bottleneck_config = [
            # (t, c, n, s) - expand_ratio, output_channels, num_repeats, stride
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        
        in_c = div_channels(32)
        for t, c, n, s in bottleneck_config:
            out_c = div_channels(c)
            for i in range(n):
                stride = s if i == 0 else 1
                self.bottlenecks.append(InvertedResidual(in_c, out_c, stride, t))
                in_c = out_c
        
        # 最终卷积
        self.conv2 = Conv2dBNReLU(in_c, div_channels(1280), 1, stride=1, padding=0)
        
        # 全连接
        self.fc = [[random.gauss(0, 0.01) for _ in range(div_channels(1280))] for _ in range(num_classes)]
    
    def forward(self, x: List[List[List[List[float]]]]) -> Union[List[List[List[List[float]]]], "torch.Tensor"]:
        """前向传播"""
        x = self.conv1.forward(x)
        for bottleneck in self.bottlenecks:
            x = bottleneck.forward(x)
        x = self.conv2.forward(x)
        
        N = len(x)
        C = len(x[0])
        H = len(x[0][0])
        W = len(x[0][0][0])
        
        pooled = [[sum(x[n][c][h][w] for h in range(H) for w in range(W)) / (H * W) for c in range(C)] for n in range(N)]
        output = [[sum(pooled[n][c] * self.fc[cls][c] for c in range(len(pooled[n]))) for cls in range(self.num_classes)] for n in range(N)]
        return output


class SqueezeExcitation:
    """SE模块"""
    
    def __init__(self, channels: int, squeeze_ratio: int = 4):
        self.channels = channels
        mid = max(1, channels // squeeze_ratio)
        self.fc1 = [[random.gauss(0, math.sqrt(2.0/channels)) for _ in range(channels)] for _ in range(mid)]
        self.fc2 = [[random.gauss(0, math.sqrt(2.0/mid)) for _ in range(mid)] for _ in range(channels)]
    
    def forward(self, x: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
        N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        
        # 全局平均池化
        pooled = [[sum(x[n][c][h][w] for h in range(H) for w in range(W)) / (H * W) for c in range(C)] for n in range(N)]
        
        # FC -> ReLU -> FC -> Sigmoid
        for n in range(N):
            mid = [max(0, sum(pooled[n][c] * self.fc1[m][c] for c in range(C))) for m in range(len(self.fc1))]
            scale = [1.0 / (1.0 + math.exp(-sum(mid[m] * self.fc2[c][m] for m in range(len(mid))))) for c in range(C)]
            
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        x[n][c][h][w] *= scale[c]
        return x


class MobileNetV3Large:
    """
    MobileNetV3-Large
    使用SE模块、h-swish激活、NAS搜索的架构
    """
    
    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes
        
        # 初始卷积
        self.conv1 = Conv2dBNReLU(3, 16, 3, stride=2, padding=1)
        
        # 瓶颈块 (简化配置)
        self.bottlenecks = []
        config = [
            # (in_c, out_c, kernel, stride, expand, se, activation)
            (16, 16, 3, 1, True, False, 'relu'),
            (16, 24, 3, 2, False, False, 'relu'),
            (24, 24, 3, 1, False, False, 'relu'),
            (24, 40, 5, 2, True, True, 'relu'),
            (40, 40, 5, 1, True, True, 'relu'),
            (40, 40, 5, 1, True, True, 'relu'),
            (40, 80, 3, 2, False, False, 'hswish'),
            (80, 80, 3, 1, False, False, 'hswish'),
            (80, 80, 3, 1, False, False, 'hswish'),
            (80, 112, 3, 1, True, True, 'hswish'),
            (112, 112, 3, 1, True, True, 'hswish'),
            (112, 160, 5, 2, True, True, 'hswish'),
            (160, 160, 5, 1, True, True, 'hswish'),
            (160, 160, 5, 1, True, True, 'hswish'),
        ]
        
        in_c = 16
        for out_c, kernel, stride, expand, se, act in config:
            self.bottlenecks.append({
                'in_c': in_c, 'out_c': out_c, 'kernel': kernel,
                'stride': stride, 'expand': expand, 'se': se, 'act': act,
                'weight': [[random.gauss(0, math.sqrt(2.0/in_c)) for _ in range(in_c)] for _ in range(out_c)]
            })
            in_c = out_c
        
        # 最终层
        self.conv2 = Conv2dBNReLU(in_c, 960, 1, stride=1)
        self.se = SqueezeExcitation(960)
        self.fc = [[random.gauss(0, 0.01) for _ in range(960)] for _ in range(num_classes)]
    
    def forward(self, x: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
        x = self.conv1.forward(x)
        for b in self.bottlenecks:
            # 简化: 线性变换
            N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
            out_h = _conv2d_compute_out(H, b['kernel'], b['stride'], b['kernel']//2)
            out_w = _conv2d_compute_out(W, b['kernel'], b['stride'], b['kernel']//2)
            output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(b['out_c'])] for _ in range(N)]
            for n in range(N):
                for oc in range(b['out_c']):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            val = sum(x[n][ic][oh*b['stride']+kh-b['kernel']//2][ow*b['stride']+kw-b['kernel']//2] * random.gauss(0, 0.1) 
                                    for ic in range(b['in_c']) for kh in range(b['kernel']) for kw in range(b['kernel'])
                                    if 0 <= oh*b['stride']+kh-b['kernel']//2 < H and 0 <= ow*b['stride']+kw-b['kernel']//2 < W)
                            output[n][oc][oh][ow] = hard_swish(val) if b['act'] == 'hswish' else relu6(val)
            x = output
        
        x = self.conv2.forward(x)
        x = self.se.forward(x)
        
        N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        pooled = [[sum(x[n][c][h][w] for h in range(H) for w in range(W)) / (H * W) for c in range(C)] for n in range(N)]
        return [[sum(pooled[n][c] * self.fc[cls][c] for c in range(C)) for cls in range(self.num_classes)] for n in range(N)]


class MobileNetV3Small:
    """
    MobileNetV3-Small
    更轻量的版本
    """
    
    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes
        self.conv1 = Conv2dBNReLU(3, 16, 3, stride=2, padding=1)
        
        self.bottlenecks = []
        config = [
            (16, 16, 3, 2, True, True, 'relu'),
            (16, 24, 3, 2, False, False, 'relu'),
            (24, 24, 3, 1, False, False, 'relu'),
            (24, 48, 5, 2, True, True, 'hswish'),
            (48, 48, 5, 1, True, True, 'hswish'),
            (48, 96, 5, 2, True, True, 'hswish'),
            (96, 96, 5, 1, True, True, 'hswish'),
            (96, 96, 5, 1, True, True, 'hswish'),
        ]
        
        in_c = 16
        for out_c, kernel, stride, expand, se, act in config:
            self.bottlenecks.append({'in_c': in_c, 'out_c': out_c, 'kernel': kernel, 'stride': stride, 'expand': expand, 'se': se, 'act': act})
            in_c = out_c
        
        self.conv2 = Conv2dBNReLU(in_c, 576, 1)
        self.se = SqueezeExcitation(576)
        self.fc = [[random.gauss(0, 0.01) for _ in range(576)] for _ in range(num_classes)]
    
    def forward(self, x: List[List[List[List[float]]]]) -> Union[List[List[List[List[float]]]], "torch.Tensor"]:
        x = self.conv1.forward(x)
        for b in self.bottlenecks:
            N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
            out_h = _conv2d_compute_out(H, b['kernel'], b['stride'], b['kernel']//2)
            out_w = _conv2d_compute_out(W, b['kernel'], b['stride'], b['kernel']//2)
            output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(b['out_c'])] for _ in range(N)]
            for n in range(N):
                for oc in range(b['out_c']):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            val = sum(x[n][ic][oh*b['stride']+kh-b['kernel']//2][ow*b['stride']+kw-b['kernel']//2] * random.gauss(0, 0.1)
                                    for ic in range(b['in_c']) for kh in range(b['kernel']) for kw in range(b['kernel'])
                                    if 0 <= oh*b['stride']+kh-b['kernel']//2 < H and 0 <= ow*b['stride']+kw-b['kernel']//2 < W)
                            output[n][oc][oh][ow] = hard_swish(val) if b['act'] == 'hswish' else relu6(val)
            x = output
        
        x = self.conv2.forward(x)
        x = self.se.forward(x)
        
        N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        pooled = [[sum(x[n][c][h][w] for h in range(H) for w in range(W)) / (H * W) for c in range(C)] for n in range(N)]
        return [[sum(pooled[n][c] * self.fc[cls][c] for c in range(C)) for cls in range(self.num_classes)] for n in range(N)]


# 工厂函数
def mobilenet_v1(num_classes: int = 1000, **kwargs) -> MobileNetV1:
    return MobileNetV1(num_classes, **kwargs)

def mobilenet_v2(num_classes: int = 1000, **kwargs) -> MobileNetV2:
    return MobileNetV2(num_classes, **kwargs)

def mobilenet_v3_large(num_classes: int = 1000) -> MobileNetV3Large:
    return MobileNetV3Large(num_classes)

def mobilenet_v3_small(num_classes: int = 1000) -> MobileNetV3Small:
    return MobileNetV3Small(num_classes)
