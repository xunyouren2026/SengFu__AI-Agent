"""
卷积层模块 - 完整实现
包含: Conv1d, Conv2d, Conv3d, TransposedConv1d, TransposedConv2d, 
      DepthwiseConv2d, SeparableConv2d, GroupedConv2d, DilatedConv2d等
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def im2col(input_data: List[List[Union[List[List[float]], 'torch.Tensor']]], 
           kernel_h: int, kernel_w: int,
           stride_h: int = 1, stride_w: int = 1,
           pad_h: int = 0, pad_w: int = 0,
           dilation_h: int = 1, dilation_w: int = 1) -> Tuple[Union[List[List[float]], 'torch.Tensor'], Tuple[int, int, int, int]]:
    """
    将4D输入转换为列矩阵用于卷积操作
    输入形状: (N, C, H, W)
    输出: (col_matrix, (N, out_h, out_w, C*kernel_h*kernel_w))
    """
    N = len(input_data)
    C = len(input_data[0]) if N > 0 else 0
    H = len(input_data[0][0]) if C > 0 else 0
    W = len(input_data[0][0][0]) if H > 0 else 0
    
    # 计算输出尺寸
    out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    col = []
    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = []
                for c in range(C):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_idx = oh * stride_h + kh * dilation_h - pad_h
                            w_idx = ow * stride_w + kw * dilation_w - pad_w
                            
                            if 0 <= h_idx < H and 0 <= w_idx < W:
                                patch.append(input_data[n][c][h_idx][w_idx])
                            else:
                                patch.append(0.0)
                col.append(patch)
    
    return col, (N, out_h, out_w, C * kernel_h * kernel_w)


def col2im(col: Union[List[List[float]], 'torch.Tensor'], input_shape: Tuple[int, int, int, int],
           kernel_h: int, kernel_w: int,
           stride_h: int = 1, stride_w: int = 1,
           pad_h: int = 0, pad_w: int = 0,
           dilation_h: int = 1, dilation_w: int = 1) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
    """将列矩阵转换回4D张量"""
    N, C, H, W = input_shape
    
    out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # 初始化输出
    output = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
    
    idx = 0
    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = col[idx]
                idx += 1
                p_idx = 0
                for c in range(C):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_idx = oh * stride_h + kh * dilation_h - pad_h
                            w_idx = ow * stride_w + kw * dilation_w - pad_w
                            
                            if 0 <= h_idx < H and 0 <= w_idx < W:
                                output[n][c][h_idx][w_idx] += patch[p_idx]
                            p_idx += 1
    
    return output



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


class ConvBase(ABC):
    """卷积层基类"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding_mode = padding_mode
        
        # 验证groups
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels和out_channels必须能被groups整除")
        
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
    
    @abstractmethod
    def forward(self, x) -> List:
        """前向传播"""
        pass
    
    @abstractmethod
    def backward(self, grad_output) -> Tuple:
        """反向传播，返回(input_grad, weight_grad, bias_grad)"""
        pass
    
    def _xavier_init(self, fan_in: int, fan_out: int) -> Union[List[List[float]], 'torch.Tensor']:
        """Xavier初始化"""
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
    
    def _kaiming_init(self, fan_in: int, fan_out: int, mode: str = 'fan_in') -> Union[List[List[float]], 'torch.Tensor']:
        """Kaiming初始化"""
        if mode == 'fan_in':
            std = math.sqrt(2.0 / fan_in)
        else:
            std = math.sqrt(2.0 / fan_out)
        return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


class Conv2d(ConvBase):
    """
    2D卷积层
    完整实现，包含前向和反向传播
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                        padding, dilation, groups, bias, padding_mode)
        
        # 处理kernel_size
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        # 处理stride
        if isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        
        # 处理padding
        if isinstance(padding, int):
            self.pad_h = padding
            self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
        
        # 处理dilation
        if isinstance(dilation, int):
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_h, self.dilation_w = dilation
        
        # 初始化权重
        # 权重形状: (out_channels, in_channels/groups, kernel_h, kernel_w)
        fan_in = self.in_channels_per_group * self.kernel_h * self.kernel_w
        fan_out = self.out_channels_per_group * self.kernel_h * self.kernel_w
        
        self.weight = []
        for g in range(self.groups):
            group_weight = self._kaiming_init(fan_in, self.out_channels_per_group)
            self.weight.append(group_weight)
        
        # 初始化偏置
        if bias:
            self.bias_param = [0.0 for _ in range(out_channels)]
        else:
            self.bias_param = None
        
        # 缓存
        self._input_cache = None
        self._col_cache = None
        self._input_shape_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """
        前向传播
        x: (N, C_in, H, W)
        返回: (N, C_out, H_out, W_out)
        """
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        self._input_shape_cache = (N, C, H, W)
        
        # 计算输出尺寸
        out_h = (H + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride_h + 1
        out_w = (W + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride_w + 1
        
        # 初始化输出
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] 
                   for _ in range(self.out_channels)] for _ in range(N)]
        
        # 对每个group进行卷积
        for n in range(N):
            for g in range(self.groups):
                # 获取当前group的输入通道
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                
                # 获取当前group的输出通道
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                # 获取当前group的权重
                group_weight = self.weight[g]
                
                for oc_idx, w_row in enumerate(group_weight):
                    oc = oc_start + oc_idx
                    
                    for oh in range(out_h):
                        for ow in range(out_w):
                            val = 0.0
                            w_idx = 0
                            
                            for c in range(c_start, c_end):
                                for kh in range(self.kernel_h):
                                    for kw in range(self.kernel_w):
                                        h_idx = oh * self.stride_h + kh * self.dilation_h - self.pad_h
                                        w_idx_input = ow * self.stride_w + kw * self.dilation_w - self.pad_w
                                        
                                        if 0 <= h_idx < H and 0 <= w_idx_input < W:
                                            val += x[n][c][h_idx][w_idx_input] * w_row[w_idx]
                                        w_idx += 1
                            
                            output[n][oc][oh][ow] = val
                            
                            # 添加偏置
                            if self.bias_param is not None:
                                output[n][oc][oh][ow] += self.bias_param[oc]
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple[List[List[Union[List[List[float]], 'torch.Tensor']]], 
                                                                            List[List[Union[List[List[float]], 'torch.Tensor']]], 
                                                                            Union[List[float], 'torch.Tensor']]:
        """
        反向传播
        返回: (grad_input, grad_weight, grad_bias)
        """
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N, C, H, W = self._input_shape_cache
        
        out_h = len(grad_output[0][0])
        out_w = len(grad_output[0][0][0])
        
        # 计算输入梯度
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        # 计算权重梯度
        grad_weight = [[[[0.0 for _ in range(len(self.weight[g][0]))]
                        for _ in range(self.out_channels_per_group)]
                       for g in range(self.groups)]]

        # 计算偏置梯度
        if self.bias_param is not None:
            grad_bias = [0.0 for _ in range(self.out_channels)]
        else:
            grad_bias = None
        
        for n in range(N):
            for g in range(self.groups):
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                group_weight = self.weight[g]
                
                for oc_idx in range(self.out_channels_per_group):
                    oc = oc_start + oc_idx
                    w_row = group_weight[oc_idx]
                    
                    for oh in range(out_h):
                        for ow in range(out_w):
                            grad_out = grad_output[n][oc][oh][ow]
                            
                            # 累加偏置梯度
                            if grad_bias is not None:
                                grad_bias[oc] += grad_out
                            
                            # 计算输入梯度和权重梯度
                            w_idx = 0
                            for c in range(c_start, c_end):
                                for kh in range(self.kernel_h):
                                    for kw in range(self.kernel_w):
                                        h_idx = oh * self.stride_h + kh * self.dilation_h - self.pad_h
                                        w_idx_input = ow * self.stride_w + kw * self.dilation_w - self.pad_w
                                        
                                        if 0 <= h_idx < H and 0 <= w_idx_input < W:
                                            # 输入梯度
                                            grad_input[n][c][h_idx][w_idx_input] += grad_out * w_row[w_idx]
                                            # 权重梯度
                                            grad_weight[g][oc_idx][w_idx] += grad_out * x[n][c][h_idx][w_idx_input]
                                        
                                        w_idx += 1
        
        return grad_input, grad_weight, grad_bias
    
    def get_weights(self) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """获取权重 (out_channels, in_channels/groups, kernel_h, kernel_w)"""
        weights = []
        for g in range(self.groups):
            for oc_idx in range(self.out_channels_per_group):
                w_row = self.weight[g][oc_idx]
                # 重塑为 (in_channels_per_group, kernel_h, kernel_w)
                reshaped = []
                idx = 0
                for c in range(self.in_channels_per_group):
                    channel_w = []
                    for kh in range(self.kernel_h):
                        row_w = []
                        for kw in range(self.kernel_w):
                            row_w.append(w_row[idx])
                            idx += 1
                        channel_w.append(row_w)
                    reshaped.append(channel_w)
                weights.append(reshaped)
        return weights


class Conv1d(ConvBase):
    """
    1D卷积层
    完整实现
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                        padding, dilation, groups, bias)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # 初始化权重
        fan_in = self.in_channels_per_group * kernel_size
        fan_out = self.out_channels_per_group * kernel_size
        
        self.weight = []
        for g in range(self.groups):
            group_weight = self._kaiming_init(fan_in, self.out_channels_per_group)
            self.weight.append(group_weight)
        
        if bias:
            self.bias_param = [0.0 for _ in range(out_channels)]
        else:
            self.bias_param = None
        
        self._input_cache = None
        self._input_shape_cache = None
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """
        前向传播
        x: (N, C_in, L)
        返回: (N, C_out, L_out)
        """
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        L = len(x[0][0]) if C > 0 else 0
        
        self._input_shape_cache = (N, C, L)
        
        # 计算输出长度
        out_l = (L + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # 初始化输出
        output = [[[0.0 for _ in range(out_l)] for _ in range(self.out_channels)] for _ in range(N)]
        
        for n in range(N):
            for g in range(self.groups):
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                group_weight = self.weight[g]
                
                for oc_idx, w_row in enumerate(group_weight):
                    oc = oc_start + oc_idx
                    
                    for ol in range(out_l):
                        val = 0.0
                        w_idx = 0
                        
                        for c in range(c_start, c_end):
                            for k in range(self.kernel_size):
                                l_idx = ol * self.stride + k * self.dilation - self.padding
                                
                                if 0 <= l_idx < L:
                                    val += x[n][c][l_idx] * w_row[w_idx]
                                w_idx += 1
                        
                        output[n][oc][ol] = val
                        if self.bias_param is not None:
                            output[n][oc][ol] += self.bias_param[oc]
        
        return output
    
    def backward(self, grad_output: List[Union[List[List[float]], 'torch.Tensor']]) -> Tuple[List[Union[List[List[float]], 'torch.Tensor']], 
                                                                       List[Union[List[List[float]], 'torch.Tensor']], 
                                                                       Union[List[float], 'torch.Tensor']]:
        """反向传播"""
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N, C, L = self._input_shape_cache
        
        out_l = len(grad_output[0][0])
        
        grad_input = [[[0.0 for _ in range(L)] for _ in range(C)] for _ in range(N)]
        grad_weight = [[[[0.0 for _ in range(len(self.weight[g][0]))]
                        for _ in range(self.out_channels_per_group)]
                       for g in range(self.groups)]]

        if self.bias_param is not None:
            grad_bias = [0.0 for _ in range(self.out_channels)]
        else:
            grad_bias = None
        
        for n in range(N):
            for g in range(self.groups):
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                group_weight = self.weight[g]
                
                for oc_idx in range(self.out_channels_per_group):
                    oc = oc_start + oc_idx
                    w_row = group_weight[oc_idx]
                    
                    for ol in range(out_l):
                        grad_out = grad_output[n][oc][ol]
                        
                        if grad_bias is not None:
                            grad_bias[oc] += grad_out
                        
                        w_idx = 0
                        for c in range(c_start, c_end):
                            for k in range(self.kernel_size):
                                l_idx = ol * self.stride + k * self.dilation - self.padding
                                
                                if 0 <= l_idx < L:
                                    grad_input[n][c][l_idx] += grad_out * w_row[w_idx]
                                    grad_weight[g][oc_idx][w_idx] += grad_out * x[n][c][l_idx]
                                
                                w_idx += 1
        
        return grad_input, grad_weight, grad_bias


class Conv3d(ConvBase):
    """
    3D卷积层
    完整实现
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                        padding, dilation, groups, bias)
        
        if isinstance(kernel_size, int):
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size, kernel_size, kernel_size
        else:
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size
        
        if isinstance(stride, int):
            self.stride_d, self.stride_h, self.stride_w = stride, stride, stride
        else:
            self.stride_d, self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_d, self.pad_h, self.pad_w = padding, padding, padding
        else:
            self.pad_d, self.pad_h, self.pad_w = padding
        
        if isinstance(dilation, int):
            self.dilation_d, self.dilation_h, self.dilation_w = dilation, dilation, dilation
        else:
            self.dilation_d, self.dilation_h, self.dilation_w = dilation
        
        # 初始化权重
        fan_in = self.in_channels_per_group * self.kernel_d * self.kernel_h * self.kernel_w
        fan_out = self.out_channels_per_group * self.kernel_d * self.kernel_h * self.kernel_w
        
        self.weight = []
        for g in range(self.groups):
            group_weight = self._kaiming_init(fan_in, self.out_channels_per_group)
            self.weight.append(group_weight)
        
        if bias:
            self.bias_param = [0.0 for _ in range(out_channels)]
        else:
            self.bias_param = None
        
        self._input_cache = None
        self._input_shape_cache = None
    
    def forward(self, x: List[List[List[Union[List[List[float]], 'torch.Tensor']]]]) -> List[List[List[Union[List[List[float]], 'torch.Tensor']]]]:
        """
        前向传播
        x: (N, C_in, D, H, W)
        返回: (N, C_out, D_out, H_out, W_out)
        """
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        D = len(x[0][0]) if C > 0 else 0
        H = len(x[0][0][0]) if D > 0 else 0
        W = len(x[0][0][0][0]) if H > 0 else 0
        
        self._input_shape_cache = (N, C, D, H, W)
        
        # 计算输出尺寸
        out_d = (D + 2 * self.pad_d - self.dilation_d * (self.kernel_d - 1) - 1) // self.stride_d + 1
        out_h = (H + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride_h + 1
        out_w = (W + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride_w + 1
        
        output = [[[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(out_d)]
                   for _ in range(self.out_channels)] for _ in range(N)]
        
        for n in range(N):
            for g in range(self.groups):
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                group_weight = self.weight[g]
                
                for oc_idx, w_row in enumerate(group_weight):
                    oc = oc_start + oc_idx
                    
                    for od in range(out_d):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                val = 0.0
                                w_idx = 0
                                
                                for c in range(c_start, c_end):
                                    for kd in range(self.kernel_d):
                                        for kh in range(self.kernel_h):
                                            for kw in range(self.kernel_w):
                                                d_idx = od * self.stride_d + kd * self.dilation_d - self.pad_d
                                                h_idx = oh * self.stride_h + kh * self.dilation_h - self.pad_h
                                                w_idx_input = ow * self.stride_w + kw * self.dilation_w - self.pad_w
                                                
                                                if 0 <= d_idx < D and 0 <= h_idx < H and 0 <= w_idx_input < W:
                                                    val += x[n][c][d_idx][h_idx][w_idx_input] * w_row[w_idx]
                                                w_idx += 1
                                
                                output[n][oc][od][oh][ow] = val
                                if self.bias_param is not None:
                                    output[n][oc][od][oh][ow] += self.bias_param[oc]
        
        return output
    
    def backward(self, grad_output: List[List[List[Union[List[List[float]], 'torch.Tensor']]]]) -> Tuple:
        """反向传播"""
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N, C, D, H, W = self._input_shape_cache
        
        out_d = len(grad_output[0][0])
        out_h = len(grad_output[0][0][0])
        out_w = len(grad_output[0][0][0][0])
        
        grad_input = [[[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(D)] 
                      for _ in range(C)] for _ in range(N)]
        grad_weight = [[[[0.0 for _ in range(len(self.weight[g][0]))]
                        for _ in range(self.out_channels_per_group)]
                       for g in range(self.groups)]]

        if self.bias_param is not None:
            grad_bias = [0.0 for _ in range(self.out_channels)]
        else:
            grad_bias = None
        
        for n in range(N):
            for g in range(self.groups):
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                group_weight = self.weight[g]
                
                for oc_idx in range(self.out_channels_per_group):
                    oc = oc_start + oc_idx
                    w_row = group_weight[oc_idx]
                    
                    for od in range(out_d):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                grad_out = grad_output[n][oc][od][oh][ow]
                                
                                if grad_bias is not None:
                                    grad_bias[oc] += grad_out
                                
                                w_idx = 0
                                for c in range(c_start, c_end):
                                    for kd in range(self.kernel_d):
                                        for kh in range(self.kernel_h):
                                            for kw in range(self.kernel_w):
                                                d_idx = od * self.stride_d + kd * self.dilation_d - self.pad_d
                                                h_idx = oh * self.stride_h + kh * self.dilation_h - self.pad_h
                                                w_idx_input = ow * self.stride_w + kw * self.dilation_w - self.pad_w
                                                
                                                if 0 <= d_idx < D and 0 <= h_idx < H and 0 <= w_idx_input < W:
                                                    grad_input[n][c][d_idx][h_idx][w_idx_input] += grad_out * w_row[w_idx]
                                                    grad_weight[g][oc_idx][w_idx] += grad_out * x[n][c][d_idx][h_idx][w_idx_input]
                                                
                                                w_idx += 1
        
        return grad_input, grad_weight, grad_bias


class TransposedConv2d:
    """
    2D转置卷积层 (反卷积)
    用于上采样
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels和out_channels必须能被groups整除")
        
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_h = padding
            self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
        
        if isinstance(output_padding, int):
            self.output_pad_h = output_padding
            self.output_pad_w = output_padding
        else:
            self.output_pad_h, self.output_pad_w = output_padding
        
        if isinstance(dilation, int):
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_h, self.dilation_w = dilation
        
        # 初始化权重
        # 注意：转置卷积的权重形状与普通卷积相反
        fan_in = self.out_channels_per_group * self.kernel_h * self.kernel_w
        fan_out = self.in_channels_per_group * self.kernel_h * self.kernel_w
        
        std = math.sqrt(2.0 / fan_in)
        self.weight = []
        for g in range(self.groups):
            group_weight = [[random.gauss(0, std) for _ in range(fan_in)] 
                           for _ in range(self.in_channels_per_group)]
            self.weight.append(group_weight)
        
        if bias:
            self.bias_param = [0.0 for _ in range(out_channels)]
        else:
            self.bias_param = None
        
        self._input_cache = None
        self._input_shape_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """
        前向传播
        x: (N, C_in, H, W)
        返回: (N, C_out, H_out, W_out)
        """
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        self._input_shape_cache = (N, C, H, W)
        
        # 计算输出尺寸
        out_h = (H - 1) * self.stride_h - 2 * self.pad_h + self.dilation_h * (self.kernel_h - 1) + self.output_pad_h + 1
        out_w = (W - 1) * self.stride_w - 2 * self.pad_w + self.dilation_w * (self.kernel_w - 1) + self.output_pad_w + 1
        
        # 初始化输出
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] 
                   for _ in range(self.out_channels)] for _ in range(N)]
        
        # 转置卷积：将输入值散射到输出
        for n in range(N):
            for g in range(self.groups):
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                group_weight = self.weight[g]
                
                for ic_idx in range(self.in_channels_per_group):
                    ic = c_start + ic_idx
                    w_row = group_weight[ic_idx]
                    
                    for ih in range(H):
                        for iw in range(W):
                            input_val = x[n][ic][ih][iw]
                            
                            # 计算输出位置
                            oh_base = ih * self.stride_h - self.pad_h
                            ow_base = iw * self.stride_w - self.pad_w
                            
                            w_idx = 0
                            for oc_idx in range(self.out_channels_per_group):
                                oc = oc_start + oc_idx
                                for kh in range(self.kernel_h):
                                    for kw in range(self.kernel_w):
                                        oh = oh_base + kh * self.dilation_h
                                        ow = ow_base + kw * self.dilation_w
                                        
                                        if 0 <= oh < out_h and 0 <= ow < out_w:
                                            output[n][oc][oh][ow] += input_val * w_row[w_idx]
                                        w_idx += 1
        
        # 添加偏置
        if self.bias_param is not None:
            for n in range(N):
                for oc in range(self.out_channels):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            output[n][oc][oh][ow] += self.bias_param[oc]
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple:
        """反向传播"""
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N, C, H, W = self._input_shape_cache
        
        out_h = len(grad_output[0][0])
        out_w = len(grad_output[0][0][0])
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        grad_weight = [[[[0.0 for _ in range(len(self.weight[g][0]))]
                        for _ in range(self.in_channels_per_group)]
                       for g in range(self.groups)]]
        
        if self.bias_param is not None:
            grad_bias = [0.0 for _ in range(self.out_channels)]
        else:
            grad_bias = None
        
        # 计算梯度
        for n in range(N):
            for g in range(self.groups):
                c_start = g * self.in_channels_per_group
                c_end = (g + 1) * self.in_channels_per_group
                oc_start = g * self.out_channels_per_group
                oc_end = (g + 1) * self.out_channels_per_group
                
                group_weight = self.weight[g]
                
                for ic_idx in range(self.in_channels_per_group):
                    ic = c_start + ic_idx
                    w_row = group_weight[ic_idx]
                    
                    for ih in range(H):
                        for iw in range(W):
                            oh_base = ih * self.stride_h - self.pad_h
                            ow_base = iw * self.stride_w - self.pad_w
                            
                            w_idx = 0
                            for oc_idx in range(self.out_channels_per_group):
                                oc = oc_start + oc_idx
                                for kh in range(self.kernel_h):
                                    for kw in range(self.kernel_w):
                                        oh = oh_base + kh * self.dilation_h
                                        ow = ow_base + kw * self.dilation_w
                                        
                                        if 0 <= oh < out_h and 0 <= ow < out_w:
                                            grad_out = grad_output[n][oc][oh][ow]
                                            grad_input[n][ic][ih][iw] += grad_out * w_row[w_idx]
                                            grad_weight[g][ic_idx][w_idx] += grad_out * x[n][ic][ih][iw]
                                        
                                        w_idx += 1
        
        # 偏置梯度
        if grad_bias is not None:
            for n in range(N):
                for oc in range(self.out_channels):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            grad_bias[oc] += grad_output[n][oc][oh][ow]
        
        return grad_input, grad_weight, grad_bias


class DepthwiseConv2d:
    """
    深度可分离卷积 - 深度卷积部分
    每个输入通道使用独立的卷积核
    """
    
    def __init__(self, channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True):
        self.channels = channels
        
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_h = padding
            self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
        
        if isinstance(dilation, int):
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_h, self.dilation_w = dilation
        
        # 每个通道一个卷积核
        fan_in = self.kernel_h * self.kernel_w
        std = math.sqrt(2.0 / fan_in)
        
        self.weight = [[[random.gauss(0, std) for _ in range(self.kernel_w)] 
                       for _ in range(self.kernel_h)] 
                      for _ in range(channels)]
        
        if bias:
            self.bias_param = [0.0 for _ in range(channels)]
        else:
            self.bias_param = None
        
        self._input_cache = None
        self._input_shape_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        self._input_shape_cache = (N, C, H, W)
        
        out_h = (H + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride_h + 1
        out_w = (W + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride_w + 1
        
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] 
                   for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                kernel = self.weight[c]
                
                for oh in range(out_h):
                    for ow in range(out_w):
                        val = 0.0
                        
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh * self.dilation_h - self.pad_h
                                w_idx = ow * self.stride_w + kw * self.dilation_w - self.pad_w
                                
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    val += x[n][c][h_idx][w_idx] * kernel[kh][kw]
                        
                        output[n][c][oh][ow] = val
                        if self.bias_param is not None:
                            output[n][c][oh][ow] += self.bias_param[c]
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple:
        """反向传播"""
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N, C, H, W = self._input_shape_cache
        
        out_h = len(grad_output[0][0])
        out_w = len(grad_output[0][0][0])
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        grad_weight = [[[0.0 for _ in range(self.kernel_w)] for _ in range(self.kernel_h)] for _ in range(len(self.weight))]
        
        if self.bias_param is not None:
            grad_bias = [0.0 for _ in range(C)]
        else:
            grad_bias = None
        
        for n in range(N):
            for c in range(C):
                kernel = self.weight[c]
                
                for oh in range(out_h):
                    for ow in range(out_w):
                        grad_out = grad_output[n][c][oh][ow]
                        
                        if grad_bias is not None:
                            grad_bias[c] += grad_out
                        
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh * self.dilation_h - self.pad_h
                                w_idx = ow * self.stride_w + kw * self.dilation_w - self.pad_w
                                
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    grad_input[n][c][h_idx][w_idx] += grad_out * kernel[kh][kw]
                                    grad_weight[c][kh][kw] += grad_out * x[n][c][h_idx][w_idx]
        
        return grad_input, grad_weight, grad_bias


class PointwiseConv2d:
    """
    逐点卷积 (1x1卷积)
    用于通道混合
    """
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1x1卷积权重
        std = math.sqrt(2.0 / in_channels)
        self.weight = [[random.gauss(0, std) for _ in range(in_channels)] 
                      for _ in range(out_channels)]
        
        if bias:
            self.bias_param = [0.0 for _ in range(out_channels)]
        else:
            self.bias_param = None
        
        self._input_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        output = [[[[0.0 for _ in range(W)] for _ in range(H)] 
                   for _ in range(self.out_channels)] for _ in range(N)]
        
        for n in range(N):
            for oc in range(self.out_channels):
                for oh in range(H):
                    for ow in range(W):
                        val = 0.0
                        for ic in range(C):
                            val += x[n][ic][oh][ow] * self.weight[oc][ic]
                        
                        output[n][oc][oh][ow] = val
                        if self.bias_param is not None:
                            output[n][oc][oh][ow] += self.bias_param[oc]
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple:
        """反向传播"""
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        grad_weight = [[0.0 for _ in range(C)] for _ in range(self.out_channels)]
        
        if self.bias_param is not None:
            grad_bias = [0.0 for _ in range(self.out_channels)]
        else:
            grad_bias = None
        
        for n in range(N):
            for oc in range(self.out_channels):
                for oh in range(H):
                    for ow in range(W):
                        grad_out = grad_output[n][oc][oh][ow]
                        
                        if grad_bias is not None:
                            grad_bias[oc] += grad_out
                        
                        for ic in range(C):
                            grad_input[n][ic][oh][ow] += grad_out * self.weight[oc][ic]
                            grad_weight[oc][ic] += grad_out * x[n][ic][oh][ow]
        
        return grad_input, grad_weight, grad_bias


class SeparableConv2d:
    """
    深度可分离卷积
    = DepthwiseConv + PointwiseConv
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True):
        self.depthwise = DepthwiseConv2d(in_channels, kernel_size, stride, padding, dilation, bias=False)
        self.pointwise = PointwiseConv2d(in_channels, out_channels, bias=bias)
        
        self._intermediate_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        # 深度卷积
        depth_out = self.depthwise.forward(x)
        self._intermediate_cache = depth_out
        
        # 逐点卷积
        output = self.pointwise.forward(depth_out)
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple:
        """反向传播"""
        # 逐点卷积梯度
        grad_depth, grad_pw_weight, grad_pw_bias = self.pointwise.backward(grad_output)
        
        # 深度卷积梯度
        grad_input, grad_dw_weight, grad_dw_bias = self.depthwise.backward(grad_depth)
        
        return grad_input, (grad_dw_weight, grad_pw_weight), (grad_dw_bias, grad_pw_bias)


class GroupedConv2d:
    """
    分组卷积
    将输入通道分成多个组，每组独立卷积
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 groups: int = 1,
                 bias: bool = True):
        self.conv = Conv2d(in_channels, out_channels, kernel_size, 
                          stride, padding, groups=groups, bias=bias)
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        return self.conv.forward(x)
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple:
        """反向传播"""
        return self.conv.backward(grad_output)


class DilatedConv2d:
    """
    空洞卷积 (膨胀卷积)
    扩大感受野而不增加参数
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 2,
                 bias: bool = True):
        self.conv = Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, dilation=dilation, bias=bias)
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        return self.conv.forward(x)
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple:
        """反向传播"""
        return self.conv.backward(grad_output)


class Conv2dWithActivation:
    """
    带激活函数的卷积层
    用于提高计算效率
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 activation: str = 'relu',
                 bias: bool = True):
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.activation = activation
        self._pre_activation_cache = None
    
    def _apply_activation(self, x: float) -> float:
        """应用激活函数"""
        if self.activation == 'relu':
            return max(0, x)
        elif self.activation == 'leaky_relu':
            return x if x > 0 else 0.01 * x
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + math.exp(-x)) if x > -500 else 0.0
        elif self.activation == 'tanh':
            return math.tanh(x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        else:
            return x
    
    def _activation_derivative(self, x: float) -> float:
        """激活函数导数"""
        if self.activation == 'relu':
            return 1.0 if x > 0 else 0.0
        elif self.activation == 'leaky_relu':
            return 1.0 if x > 0 else 0.01
        elif self.activation == 'sigmoid':
            sig = self._apply_activation(x)
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - math.tanh(x)**2
        elif self.activation == 'gelu':
            # GELU导数的近似
            cdf = 0.5 * (1 + math.erf(x / math.sqrt(2)))
            pdf = math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            return cdf + x * pdf
        else:
            return 1.0
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        # 先卷积
        conv_out = self.conv.forward(x)
        self._pre_activation_cache = conv_out
        
        # 再激活
        N = len(conv_out)
        C = len(conv_out[0]) if N > 0 else 0
        H = len(conv_out[0][0]) if C > 0 else 0
        W = len(conv_out[0][0][0]) if H > 0 else 0
        
        output = [[[[self._apply_activation(conv_out[n][c][h][w]) 
                    for w in range(W)] for h in range(H)] 
                  for c in range(C)] for n in range(N)]
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Tuple:
        """反向传播"""
        pre_act = self._pre_activation_cache
        N = len(pre_act)
        C = len(pre_act[0]) if N > 0 else 0
        H = len(pre_act[0][0]) if C > 0 else 0
        W = len(pre_act[0][0][0]) if H > 0 else 0
        
        # 计算激活函数的梯度
        grad_pre_act = [[[[grad_output[n][c][h][w] * self._activation_derivative(pre_act[n][c][h][w])
                         for w in range(W)] for h in range(H)]
                        for c in range(C)] for n in range(N)]
        
        # 卷积的梯度
        return self.conv.backward(grad_pre_act)


# 工厂函数
def conv1d(in_channels: int, out_channels: int, kernel_size: int,
           stride: int = 1, padding: int = 0, **kwargs) -> Conv1d:
    """创建1D卷积层"""
    return Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)


def conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
           **kwargs) -> Conv2d:
    """创建2D卷积层"""
    return Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)


def conv3d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]],
           stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0,
           **kwargs) -> Conv3d:
    """创建3D卷积层"""
    return Conv3d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)


def conv_transpose2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                     stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                     **kwargs) -> TransposedConv2d:
    """创建2D转置卷积层"""
    return TransposedConv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)


def depthwise_conv2d(channels: int, kernel_size: Union[int, Tuple[int, int]],
                     stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                     **kwargs) -> DepthwiseConv2d:
    """创建深度卷积层"""
    return DepthwiseConv2d(channels, kernel_size, stride, padding, **kwargs)


def pointwise_conv2d(in_channels: int, out_channels: int, **kwargs) -> PointwiseConv2d:
    """创建逐点卷积层"""
    return PointwiseConv2d(in_channels, out_channels, **kwargs)


def separable_conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                     stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                     **kwargs) -> SeparableConv2d:
    """创建深度可分离卷积层"""
    return SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)


def grouped_conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                   groups: int, stride: Union[int, Tuple[int, int]] = 1,
                   padding: Union[int, Tuple[int, int]] = 0, **kwargs) -> GroupedConv2d:
    """创建分组卷积层"""
    return GroupedConv2d(in_channels, out_channels, kernel_size, stride, padding, groups, **kwargs)


def dilated_conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                   dilation: Union[int, Tuple[int, int]] = 2, stride: Union[int, Tuple[int, int]] = 1,
                   padding: Union[int, Tuple[int, int]] = 0, **kwargs) -> DilatedConv2d:
    """创建空洞卷积层"""
    return DilatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, **kwargs)
