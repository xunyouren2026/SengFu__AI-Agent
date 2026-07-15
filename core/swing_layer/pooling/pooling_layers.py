"""
池化层模块 - 完整实现
包含: MaxPool, AvgPool, AdaptiveMaxPool, AdaptiveAvgPool, GlobalPool, 
      FractionalPool, LPPool, MixedPool, StochasticPool等
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def im2col_2d(input_data: List[List[Union[List[List[float]], 'torch.Tensor']]], 
              kernel_h: int, kernel_w: int,
              stride_h: int = 1, stride_w: int = 1,
              pad_h: int = 0, pad_w: int = 0) -> Tuple[Union[List[List[float]], 'torch.Tensor'], Tuple[int, int, int, int]]:
    """
    将4D输入转换为列矩阵用于池化操作
    输入形状: (N, C, H, W)
    输出: (col, (N, C, out_h, out_w))
    """
    N = len(input_data)
    C = len(input_data[0]) if N > 0 else 0
    H = len(input_data[0][0]) if C > 0 else 0
    W = len(input_data[0][0][0]) if H > 0 else 0
    
    # 添加padding
    if pad_h > 0 or pad_w > 0:
        padded = []
        for n in range(N):
            padded_n = []
            for c in range(C):
                padded_c = []
                for i in range(-pad_h, H + pad_h):
                    row = []
                    for j in range(-pad_w, W + pad_w):
                        if 0 <= i < H and 0 <= j < W:
                            row.append(input_data[n][c][i][j])
                        else:
                            row.append(0.0)
                    padded_c.append(row)
                padded_n.append(padded_c)
            padded.append(padded_n)
        input_data = padded
        H = H + 2 * pad_h
        W = W + 2 * pad_w
    
    out_h = (H - kernel_h) // stride_h + 1
    out_w = (W - kernel_w) // stride_w + 1
    
    col = []
    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    patch = []
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_idx = oh * stride_h + kh
                            w_idx = ow * stride_w + kw
                            if 0 <= h_idx < H and 0 <= w_idx < W:
                                patch.append(input_data[n][c][h_idx][w_idx])
                            else:
                                patch.append(0.0)
                    col.append(patch)
    
    return col, (N, C, out_h, out_w)


def col2im_2d(col: Union[List[List[float]], 'torch.Tensor'], shape: Tuple[int, int, int, int],
              kernel_h: int, kernel_w: int,
              stride_h: int = 1, stride_w: int = 1,
              pad_h: int = 0, pad_w: int = 0,
              H: int = 0, W: int = 0) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
    """将列矩阵转换回4D张量"""
    N, C, out_h, out_w = shape
    
    # 计算原始尺寸
    if H == 0:
        H = (out_h - 1) * stride_h + kernel_h - 2 * pad_h
    if W == 0:
        W = (out_w - 1) * stride_w + kernel_w - 2 * pad_w
    
    # 初始化输出
    output = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
    
    idx = 0
    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    patch = col[idx]
                    idx += 1
                    p_idx = 0
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_idx = oh * stride_h + kh - pad_h
                            w_idx = ow * stride_w + kw - pad_w
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


class PoolingBase(ABC):
    """池化层基类"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True):
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_h = padding
            self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
        
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        
        # 用于反向传播的缓存
        self._input_cache = None
        self._output_shape = None
        self._indices_cache = None
    
    def _compute_output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """计算输出形状"""
        if self.ceil_mode:
            out_h = math.ceil((H + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1)
            out_w = math.ceil((W + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1)
        else:
            out_h = (H + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
            out_w = (W + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        return max(1, out_h), max(1, out_w)
    
    @abstractmethod
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        pass
    
    @abstractmethod
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        pass


class MaxPool2d(PoolingBase):
    """
    2D最大池化层
    完整实现，包含前向和反向传播
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 ceil_mode: bool = False,
                 return_indices: bool = False):
        super().__init__(kernel_size, stride, padding, ceil_mode)
        if isinstance(dilation, int):
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_h, self.dilation_w = dilation
        self.return_indices = return_indices
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Union[List[List[Union[List[List[float]], 'torch.Tensor']]], 
                                                                   Tuple[List[List[Union[List[List[float]], 'torch.Tensor']]], 
                                                                         List[List[List[List[Tuple[int, int]]]]]]]:
        """
        前向传播
        x: (N, C, H, W)
        返回: (N, C, out_h, out_w) 或 (output, indices)
        """
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        # 添加padding
        if self.pad_h > 0 or self.pad_w > 0:
            padded = self._pad_input(x)
        else:
            padded = x
            self.pad_h = 0
            self.pad_w = 0
        
        padded_H = len(padded[0][0]) if C > 0 else 0
        padded_W = len(padded[0][0][0]) if padded_H > 0 else 0
        
        out_h, out_w = self._compute_output_shape(H, W)
        self._output_shape = (N, C, out_h, out_w)
        
        output = [[[[float('-inf') for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        indices = [[[[(-1, -1) for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        max_val = float('-inf')
                        max_idx = (-1, -1)
                        
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh * self.dilation_h
                                w_idx = ow * self.stride_w + kw * self.dilation_w
                                
                                if 0 <= h_idx < padded_H and 0 <= w_idx < padded_W:
                                    val = padded[n][c][h_idx][w_idx]
                                    if val > max_val:
                                        max_val = val
                                        max_idx = (h_idx, w_idx)
                        
                        output[n][c][oh][ow] = max_val
                        indices[n][c][oh][ow] = max_idx
        
        self._indices_cache = indices
        
        if self.return_indices:
            return output, indices
        return output
    
    def _pad_input(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """对输入进行padding"""
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        padded = []
        for n in range(N):
            padded_n = []
            for c in range(C):
                padded_c = []
                for i in range(-self.pad_h, H + self.pad_h):
                    row = []
                    for j in range(-self.pad_w, W + self.pad_w):
                        if 0 <= i < H and 0 <= j < W:
                            row.append(x[n][c][i][j])
                        else:
                            row.append(float('-inf'))
                    padded_c.append(row)
                padded_n.append(padded_c)
            padded.append(padded_n)
        return padded
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """
        反向传播
        梯度只传递到最大值的位置
        """
        if self._input_cache is None or self._indices_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        # 初始化梯度
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        indices = self._indices_cache
        _, _, out_h, out_w = self._output_shape
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_idx, w_idx = indices[n][c][oh][ow]
                        if h_idx >= 0 and w_idx >= 0:
                            # 调整索引以去除padding
                            adj_h = h_idx - self.pad_h
                            adj_w = w_idx - self.pad_w
                            if 0 <= adj_h < H and 0 <= adj_w < W:
                                grad_input[n][c][adj_h][adj_w] += grad_output[n][c][oh][ow]
        
        return grad_input


class AvgPool2d(PoolingBase):
    """
    2D平均池化层
    完整实现，包含前向和反向传播
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 divisor_override: Optional[int] = None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.divisor_override = divisor_override
        self._count_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """
        前向传播
        x: (N, C, H, W)
        返回: (N, C, out_h, out_w)
        """
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        out_h, out_w = self._compute_output_shape(H, W)
        self._output_shape = (N, C, out_h, out_w)
        
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        self._count_cache = [[[[0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        total = 0.0
                        count = 0
                        
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh - self.pad_h
                                w_idx = ow * self.stride_w + kw - self.pad_w
                                
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    total += x[n][c][h_idx][w_idx]
                                    count += 1
                                elif self.count_include_pad:
                                    count += 1
                        
                        if self.divisor_override is not None:
                            count = self.divisor_override
                        
                        if count > 0:
                            output[n][c][oh][ow] = total / count
                        self._count_cache[n][c][oh][ow] = count if count > 0 else 1
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None or self._count_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        _, _, out_h, out_w = self._output_shape
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        grad_val = grad_output[n][c][oh][ow]
                        count = self._count_cache[n][c][oh][ow]
                        
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh - self.pad_h
                                w_idx = ow * self.stride_w + kw - self.pad_w
                                
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    grad_input[n][c][h_idx][w_idx] += grad_val / count
        
        return grad_input


class AdaptiveMaxPool2d:
    """
    自适应最大池化层
    输出尺寸固定，自动计算kernel_size和stride
    """
    
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        if isinstance(output_size, int):
            self.output_h = output_size
            self.output_w = output_size
        else:
            self.output_h, self.output_w = output_size
        
        self._input_cache = None
        self._indices_cache = None
    
    def _compute_kernel_params(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """计算kernel_size和stride"""
        if self.output_h == 1 and self.output_w == 1:
            return H, W, 1, 1
        
        stride_h = H // self.output_h
        stride_w = W // self.output_w
        kernel_h = H - (self.output_h - 1) * stride_h
        kernel_w = W - (self.output_w - 1) * stride_w
        
        return kernel_h, kernel_w, stride_h, stride_w
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        kernel_h, kernel_w, stride_h, stride_w = self._compute_kernel_params(H, W)
        
        output = [[[[float('-inf') for _ in range(self.output_w)] for _ in range(self.output_h)] for _ in range(C)] for _ in range(N)]
        indices = [[[[(-1, -1) for _ in range(self.output_w)] for _ in range(self.output_h)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(self.output_h):
                    for ow in range(self.output_w):
                        max_val = float('-inf')
                        max_idx = (-1, -1)
                        
                        start_h = oh * stride_h
                        start_w = ow * stride_w
                        end_h = min(start_h + kernel_h, H)
                        end_w = min(start_w + kernel_w, W)
                        
                        for h in range(start_h, end_h):
                            for w in range(start_w, end_w):
                                val = x[n][c][h][w]
                                if val > max_val:
                                    max_val = val
                                    max_idx = (h, w)
                        
                        output[n][c][oh][ow] = max_val
                        indices[n][c][oh][ow] = max_idx
        
        self._indices_cache = indices
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None or self._indices_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(self.output_h):
                    for ow in range(self.output_w):
                        h_idx, w_idx = self._indices_cache[n][c][oh][ow]
                        if h_idx >= 0 and w_idx >= 0:
                            grad_input[n][c][h_idx][w_idx] += grad_output[n][c][oh][ow]
        
        return grad_input


class AdaptiveAvgPool2d:
    """
    自适应平均池化层
    输出尺寸固定，自动计算kernel_size和stride
    """
    
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        if isinstance(output_size, int):
            self.output_h = output_size
            self.output_w = output_size
        else:
            self.output_h, self.output_w = output_size
        
        self._input_cache = None
        self._count_cache = None
    
    def _compute_kernel_params(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """计算kernel_size和stride"""
        if self.output_h == 1 and self.output_w == 1:
            return H, W, 1, 1
        
        stride_h = H // self.output_h
        stride_w = W // self.output_w
        kernel_h = H - (self.output_h - 1) * stride_h
        kernel_w = W - (self.output_w - 1) * stride_w
        
        return kernel_h, kernel_w, stride_h, stride_w
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        kernel_h, kernel_w, stride_h, stride_w = self._compute_kernel_params(H, W)
        
        output = [[[[0.0 for _ in range(self.output_w)] for _ in range(self.output_h)] for _ in range(C)] for _ in range(N)]
        self._count_cache = [[[[0 for _ in range(self.output_w)] for _ in range(self.output_h)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(self.output_h):
                    for ow in range(self.output_w):
                        total = 0.0
                        count = 0
                        
                        start_h = oh * stride_h
                        start_w = ow * stride_w
                        end_h = min(start_h + kernel_h, H)
                        end_w = min(start_w + kernel_w, W)
                        
                        for h in range(start_h, end_h):
                            for w in range(start_w, end_w):
                                total += x[n][c][h][w]
                                count += 1
                        
                        if count > 0:
                            output[n][c][oh][ow] = total / count
                        self._count_cache[n][c][oh][ow] = count if count > 0 else 1
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None or self._count_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        kernel_h, kernel_w, stride_h, stride_w = self._compute_kernel_params(H, W)
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(self.output_h):
                    for ow in range(self.output_w):
                        grad_val = grad_output[n][c][oh][ow]
                        count = self._count_cache[n][c][oh][ow]
                        
                        start_h = oh * stride_h
                        start_w = ow * stride_w
                        end_h = min(start_h + kernel_h, H)
                        end_w = min(start_w + kernel_w, W)
                        
                        for h in range(start_h, end_h):
                            for w in range(start_w, end_w):
                                grad_input[n][c][h][w] += grad_val / count
        
        return grad_input


class GlobalAvgPool2d:
    """
    全局平均池化层
    将H x W池化为1 x 1
    """
    
    def __init__(self, keepdim: bool = True):
        self.keepdim = keepdim
        self._input_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Union[List[List[Union[List[List[float]], 'torch.Tensor']]], Union[List[List[float]], 'torch.Tensor']]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        total_elements = H * W
        
        if self.keepdim:
            output = [[[[0.0] for _ in range(1)] for _ in range(C)] for _ in range(N)]
            for n in range(N):
                for c in range(C):
                    total = sum(x[n][c][h][w] for h in range(H) for w in range(W))
                    output[n][c][0][0] = total / total_elements if total_elements > 0 else 0.0
        else:
            output = [[0.0 for _ in range(C)] for _ in range(N)]
            for n in range(N):
                for c in range(C):
                    total = sum(x[n][c][h][w] for h in range(H) for w in range(W))
                    output[n][c] = total / total_elements if total_elements > 0 else 0.0
        
        return output
    
    def backward(self, grad_output: Union[List[List[Union[List[List[float]], 'torch.Tensor']]], Union[List[List[float]], 'torch.Tensor']]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        total_elements = H * W
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                if self.keepdim:
                    grad_val = grad_output[n][c][0][0]
                else:
                    grad_val = grad_output[n][c]
                
                for h in range(H):
                    for w in range(W):
                        grad_input[n][c][h][w] = grad_val / total_elements
        
        return grad_input


class GlobalMaxPool2d:
    """
    全局最大池化层
    """
    
    def __init__(self, keepdim: bool = True):
        self.keepdim = keepdim
        self._input_cache = None
        self._indices_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> Union[List[List[Union[List[List[float]], 'torch.Tensor']]], Union[List[List[float]], 'torch.Tensor']]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        if self.keepdim:
            output = [[[[0.0] for _ in range(1)] for _ in range(C)] for _ in range(N)]
        else:
            output = [[0.0 for _ in range(C)] for _ in range(N)]
        
        indices = [[(-1, -1) for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                max_val = float('-inf')
                max_idx = (-1, -1)
                for h in range(H):
                    for w in range(W):
                        val = x[n][c][h][w]
                        if val > max_val:
                            max_val = val
                            max_idx = (h, w)
                
                if self.keepdim:
                    output[n][c][0][0] = max_val
                else:
                    output[n][c] = max_val
                indices[n][c] = max_idx
        
        self._indices_cache = indices
        return output
    
    def backward(self, grad_output: Union[List[List[Union[List[List[float]], 'torch.Tensor']]], Union[List[List[float]], 'torch.Tensor']]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None or self._indices_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                h_idx, w_idx = self._indices_cache[n][c]
                if h_idx >= 0 and w_idx >= 0:
                    if self.keepdim:
                        grad_val = grad_output[n][c][0][0]
                    else:
                        grad_val = grad_output[n][c]
                    grad_input[n][c][h_idx][w_idx] = grad_val
        
        return grad_input


class LPPool2d:
    """
    Lp范数池化层
    output = (sum(|x|^p))^(1/p)
    """
    
    def __init__(self, norm_type: float,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 ceil_mode: bool = False):
        self.norm_type = norm_type
        
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        
        self.ceil_mode = ceil_mode
        self._input_cache = None
        self._output_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        if self.ceil_mode:
            out_h = math.ceil((H - self.kernel_h) / self.stride_h + 1)
            out_w = math.ceil((W - self.kernel_w) / self.stride_w + 1)
        else:
            out_h = (H - self.kernel_h) // self.stride_h + 1
            out_w = (W - self.kernel_w) // self.stride_w + 1
        
        out_h = max(1, out_h)
        out_w = max(1, out_w)
        
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        
        p = self.norm_type
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        sum_pow = 0.0
                        
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh
                                w_idx = ow * self.stride_w + kw
                                
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    val = abs(x[n][c][h_idx][w_idx])
                                    sum_pow += val ** p
                        
                        if sum_pow > 0:
                            output[n][c][oh][ow] = sum_pow ** (1.0 / p)
        
        self._output_cache = output
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None or self._output_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        output = self._output_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        out_h = len(output[0][0])
        out_w = len(output[0][0][0])
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        p = self.norm_type
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        out_val = output[n][c][oh][ow]
                        grad_out = grad_output[n][c][oh][ow]
                        
                        if out_val > 0:
                            for kh in range(self.kernel_h):
                                for kw in range(self.kernel_w):
                                    h_idx = oh * self.stride_h + kh
                                    w_idx = ow * self.stride_w + kw
                                    
                                    if 0 <= h_idx < H and 0 <= w_idx < W:
                                        val = x[n][c][h_idx][w_idx]
                                        # d/dx (sum |x|^p)^(1/p) = |x|^(p-1) * sign(x) / (sum |x|^p)^((p-1)/p)
                                        grad_input[n][c][h_idx][w_idx] += (
                                            grad_out * 
                                            (abs(val) ** (p - 1)) * 
                                            (1 if val >= 0 else -1) / 
                                            (out_val ** (p - 1)) if out_val > 0 else 0
                                        )
        
        return grad_input


class FractionalMaxPool2d:
    """
    分数最大池化
    使用随机采样实现非整数步长的池化
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 output_ratio: Optional[float] = None,
                 output_size: Optional[Union[int, Tuple[int, int]]] = None,
                 return_indices: bool = False):
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        self.output_ratio = output_ratio
        self.output_size = output_size
        self.return_indices = return_indices
        
        self._input_cache = None
        self._indices_cache = None
        self._sample_indices = None
    
    def _generate_random_samples(self, H: int, W: int, out_h: int, out_w: int) -> Tuple[List[int], List[int]]:
        """生成随机采样位置"""
        # 生成行采样位置
        row_samples = sorted(random.sample(range(H), min(out_h * self.kernel_h, H)))
        col_samples = sorted(random.sample(range(W), min(out_w * self.kernel_w, W)))
        return row_samples, col_samples
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        # 计算输出尺寸
        if self.output_size is not None:
            if isinstance(self.output_size, int):
                out_h, out_w = self.output_size, self.output_size
            else:
                out_h, out_w = self.output_size
        elif self.output_ratio is not None:
            out_h = int(H * self.output_ratio)
            out_w = int(W * self.output_ratio)
        else:
            raise ValueError("需要指定output_size或output_ratio")
        
        # 生成随机采样位置
        row_samples, col_samples = self._generate_random_samples(H, W, out_h, out_w)
        self._sample_indices = (row_samples, col_samples)
        
        output = [[[[float('-inf') for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        indices = [[[[(-1, -1) for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        # 计算池化窗口
                        if oh < len(row_samples) - 1:
                            h_start = row_samples[oh]
                            h_end = min(row_samples[oh + 1], h_start + self.kernel_h, H)
                        else:
                            h_start = row_samples[oh] if oh < len(row_samples) else H - 1
                            h_end = min(h_start + self.kernel_h, H)
                        
                        if ow < len(col_samples) - 1:
                            w_start = col_samples[ow]
                            w_end = min(col_samples[ow + 1], w_start + self.kernel_w, W)
                        else:
                            w_start = col_samples[ow] if ow < len(col_samples) else W - 1
                            w_end = min(w_start + self.kernel_w, W)
                        
                        max_val = float('-inf')
                        max_idx = (-1, -1)
                        
                        for h in range(h_start, h_end):
                            for w in range(w_start, w_end):
                                val = x[n][c][h][w]
                                if val > max_val:
                                    max_val = val
                                    max_idx = (h, w)
                        
                        output[n][c][oh][ow] = max_val
                        indices[n][c][oh][ow] = max_idx
        
        self._indices_cache = indices
        
        if self.return_indices:
            return output, indices
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None or self._indices_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        output = self.forward(x) if not self.return_indices else self.forward(x)[0]
        out_h = len(output[0][0])
        out_w = len(output[0][0][0])
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_idx, w_idx = self._indices_cache[n][c][oh][ow]
                        if h_idx >= 0 and w_idx >= 0:
                            grad_input[n][c][h_idx][w_idx] += grad_output[n][c][oh][ow]
        
        return grad_input


class StochasticPool2d:
    """
    随机池化层
    使用概率分布进行池化，值越大被选中概率越高
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 training: bool = True):
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_h = padding
            self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
        
        self.training = training
        self._input_cache = None
        self._sampled_indices = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        # 添加padding
        if self.pad_h > 0 or self.pad_w > 0:
            padded = []
            for n in range(N):
                padded_n = []
                for c in range(C):
                    padded_c = []
                    for i in range(-self.pad_h, H + self.pad_h):
                        row = []
                        for j in range(-self.pad_w, W + self.pad_w):
                            if 0 <= i < H and 0 <= j < W:
                                row.append(x[n][c][i][j])
                            else:
                                row.append(0.0)
                        padded_c.append(row)
                    padded_n.append(padded_c)
                padded.append(padded_n)
            x = padded
            H = H + 2 * self.pad_h
            W = W + 2 * self.pad_w
        
        out_h = (H - self.kernel_h) // self.stride_h + 1
        out_w = (W - self.kernel_w) // self.stride_w + 1
        
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        self._sampled_indices = [[[[(-1, -1) for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        # 收集窗口内的值
                        values = []
                        positions = []
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh
                                w_idx = ow * self.stride_w + kw
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    values.append(x[n][c][h_idx][w_idx])
                                    positions.append((h_idx, w_idx))
                        
                        if not values:
                            continue
                        
                        if self.training:
                            # 训练时：随机采样
                            # 使用softmax概率
                            max_val = max(values)
                            exp_vals = [math.exp(v - max_val) for v in values]
                            sum_exp = sum(exp_vals)
                            probs = [e / sum_exp for e in exp_vals]
                            
                            # 随机采样
                            r = random.random()
                            cum_prob = 0.0
                            selected_idx = 0
                            for i, p in enumerate(probs):
                                cum_prob += p
                                if r <= cum_prob:
                                    selected_idx = i
                                    break
                            
                            output[n][c][oh][ow] = values[selected_idx]
                            self._sampled_indices[n][c][oh][ow] = positions[selected_idx]
                        else:
                            # 推理时：使用期望值
                            max_val = max(values)
                            exp_vals = [math.exp(v - max_val) for v in values]
                            sum_exp = sum(exp_vals)
                            output[n][c][oh][ow] = sum(v * e / sum_exp for v, e in zip(values, exp_vals))
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        output = self.forward(x)
        out_h = len(output[0][0])
        out_w = len(output[0][0][0])
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        if self.training:
            # 训练时：梯度只传递到采样的位置
            for n in range(N):
                for c in range(C):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            h_idx, w_idx = self._sampled_indices[n][c][oh][ow]
                            if h_idx >= 0 and w_idx >= 0:
                                adj_h = h_idx - self.pad_h
                                adj_w = w_idx - self.pad_w
                                if 0 <= adj_h < H and 0 <= adj_w < W:
                                    grad_input[n][c][adj_h][adj_w] += grad_output[n][c][oh][ow]
        else:
            # 推理时：梯度按概率分布传递
            for n in range(N):
                for c in range(C):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            # 计算概率分布
                            values = []
                            positions = []
                            for kh in range(self.kernel_h):
                                for kw in range(self.kernel_w):
                                    h_idx = oh * self.stride_h + kh
                                    w_idx = ow * self.stride_w + kw
                                    adj_h = h_idx - self.pad_h
                                    adj_w = w_idx - self.pad_w
                                    if 0 <= adj_h < H and 0 <= adj_w < W:
                                        values.append(x[n][c][adj_h][adj_w])
                                        positions.append((adj_h, adj_w))
                            
                            if values:
                                max_val = max(values)
                                exp_vals = [math.exp(v - max_val) for v in values]
                                sum_exp = sum(exp_vals)
                                
                                for val, pos, exp_val in zip(values, positions, exp_vals):
                                    prob = exp_val / sum_exp
                                    grad_input[n][c][pos[0]][pos[1]] += grad_output[n][c][oh][ow] * prob
        
        return grad_input


class MixedPool2d:
    """
    混合池化层
    组合最大池化和平均池化: output = alpha * maxpool + (1-alpha) * avgpool
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 alpha: float = 0.5):
        self.maxpool = MaxPool2d(kernel_size, stride, padding)
        self.avgpool = AvgPool2d(kernel_size, stride, padding)
        self.alpha = alpha
        self._input_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        max_out = self.maxpool.forward(x)
        avg_out = self.avgpool.forward(x)
        
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        out_h = len(max_out[0][0])
        out_w = len(max_out[0][0][0])
        
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        output[n][c][oh][ow] = (
                            self.alpha * max_out[n][c][oh][ow] + 
                            (1 - self.alpha) * avg_out[n][c][oh][ow]
                        )
        
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        N = len(grad_output)
        C = len(grad_output[0]) if N > 0 else 0
        out_h = len(grad_output[0][0])
        out_w = len(grad_output[0][0][0])
        
        # 分别计算两个池化的梯度
        max_grad = self.maxpool.backward(grad_output)
        avg_grad = self.avgpool.backward(grad_output)
        
        H = len(max_grad[0][0])
        W = len(max_grad[0][0][0])
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        grad_input[n][c][h][w] = (
                            self.alpha * max_grad[n][c][h][w] + 
                            (1 - self.alpha) * avg_grad[n][c][h][w]
                        )
        
        return grad_input


class PowerAveragePool2d:
    """
    幂平均池化
    output = (mean(|x|^p))^(1/p)
    当p->inf时趋近于最大池化，p=1时为平均池化
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 power: float = 2.0):
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
        
        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_h = padding
            self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding
        
        self.power = power
        self._input_cache = None
        self._output_cache = None
    
    def forward(self, x: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """前向传播"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        out_h = (H + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (W + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        
        output = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(C)] for _ in range(N)]
        
        p = self.power
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        sum_pow = 0.0
                        count = 0
                        
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh - self.pad_h
                                w_idx = ow * self.stride_w + kw - self.pad_w
                                
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    val = abs(x[n][c][h_idx][w_idx])
                                    sum_pow += val ** p
                                    count += 1
                        
                        if count > 0:
                            mean_pow = sum_pow / count
                            output[n][c][oh][ow] = mean_pow ** (1.0 / p)
        
        self._output_cache = output
        return output
    
    def backward(self, grad_output: List[List[Union[List[List[float]], 'torch.Tensor']]]) -> List[List[Union[List[List[float]], 'torch.Tensor']]]:
        """反向传播"""
        if self._input_cache is None or self._output_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        output = self._output_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        
        out_h = len(output[0][0])
        out_w = len(output[0][0][0])
        
        grad_input = [[[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)] for _ in range(N)]
        
        p = self.power
        
        for n in range(N):
            for c in range(C):
                for oh in range(out_h):
                    for ow in range(out_w):
                        out_val = output[n][c][oh][ow]
                        grad_out = grad_output[n][c][oh][ow]
                        
                        count = 0
                        for kh in range(self.kernel_h):
                            for kw in range(self.kernel_w):
                                h_idx = oh * self.stride_h + kh - self.pad_h
                                w_idx = ow * self.stride_w + kw - self.pad_w
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    count += 1
                        
                        if count > 0 and out_val > 0:
                            for kh in range(self.kernel_h):
                                for kw in range(self.kernel_w):
                                    h_idx = oh * self.stride_h + kh - self.pad_h
                                    w_idx = ow * self.stride_w + kw - self.pad_w
                                    
                                    if 0 <= h_idx < H and 0 <= w_idx < W:
                                        val = x[n][c][h_idx][w_idx]
                                        # d/dx (mean(|x|^p))^(1/p)
                                        grad_input[n][c][h_idx][w_idx] += (
                                            grad_out * 
                                            (abs(val) ** (p - 1)) * 
                                            (1 if val >= 0 else -1) / 
                                            (count * out_val ** (p - 1))
                                        )
        
        return grad_input


class MaxPool1d:
    """1D最大池化层"""
    
    def __init__(self, kernel_size: int, stride: Optional[int] = None, 
                 padding: int = 0, dilation: int = 1, ceil_mode: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self._input_cache = None
        self._indices_cache = None
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """前向传播 x: (N, C, L)"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        L = len(x[0][0]) if C > 0 else 0
        
        # 添加padding
        if self.padding > 0:
            padded = []
            for n in range(N):
                padded_n = []
                for c in range(C):
                    padded_c = [float('-inf')] * self.padding + x[n][c] + [float('-inf')] * self.padding
                    padded_n.append(padded_c)
                padded.append(padded_n)
            x = padded
            L = L + 2 * self.padding
        
        if self.ceil_mode:
            out_l = math.ceil((L - self.kernel_size) / self.stride + 1)
        else:
            out_l = (L - self.kernel_size) // self.stride + 1
        
        out_l = max(1, out_l)
        
        output = [[[float('-inf') for _ in range(out_l)] for _ in range(C)] for _ in range(N)]
        indices = [[[-1 for _ in range(out_l)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for ol in range(out_l):
                    max_val = float('-inf')
                    max_idx = -1
                    for k in range(self.kernel_size):
                        l_idx = ol * self.stride + k * self.dilation
                        if 0 <= l_idx < L:
                            val = x[n][c][l_idx]
                            if val > max_val:
                                max_val = val
                                max_idx = l_idx
                    output[n][c][ol] = max_val
                    indices[n][c][ol] = max_idx
        
        self._indices_cache = indices
        return output
    
    def backward(self, grad_output: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """反向传播"""
        if self._input_cache is None or self._indices_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        L = len(x[0][0]) if C > 0 else 0
        
        grad_input = [[[0.0 for _ in range(L)] for _ in range(C)] for _ in range(N)]
        
        out_l = len(grad_output[0][0])
        
        for n in range(N):
            for c in range(C):
                for ol in range(out_l):
                    l_idx = self._indices_cache[n][c][ol]
                    if l_idx >= 0:
                        adj_l = l_idx - self.padding
                        if 0 <= adj_l < L:
                            grad_input[n][c][adj_l] += grad_output[n][c][ol]
        
        return grad_input


class AvgPool1d:
    """1D平均池化层"""
    
    def __init__(self, kernel_size: int, stride: Optional[int] = None,
                 padding: int = 0, ceil_mode: bool = False,
                 count_include_pad: bool = True):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self._input_cache = None
        self._count_cache = None
    
    def forward(self, x: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """前向传播 x: (N, C, L)"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        L = len(x[0][0]) if C > 0 else 0
        
        if self.ceil_mode:
            out_l = math.ceil((L + 2 * self.padding - self.kernel_size) / self.stride + 1)
        else:
            out_l = (L + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        out_l = max(1, out_l)
        
        output = [[[0.0 for _ in range(out_l)] for _ in range(C)] for _ in range(N)]
        self._count_cache = [[[0 for _ in range(out_l)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for ol in range(out_l):
                    total = 0.0
                    count = 0
                    for k in range(self.kernel_size):
                        l_idx = ol * self.stride + k - self.padding
                        if 0 <= l_idx < L:
                            total += x[n][c][l_idx]
                            count += 1
                        elif self.count_include_pad:
                            count += 1
                    
                    if count > 0:
                        output[n][c][ol] = total / count
                    self._count_cache[n][c][ol] = count if count > 0 else 1
        
        return output
    
    def backward(self, grad_output: List[Union[List[List[float]], 'torch.Tensor']]) -> List[Union[List[List[float]], 'torch.Tensor']]:
        """反向传播"""
        if self._input_cache is None or self._count_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        L = len(x[0][0]) if C > 0 else 0
        
        grad_input = [[[0.0 for _ in range(L)] for _ in range(C)] for _ in range(N)]
        
        out_l = len(grad_output[0][0])
        
        for n in range(N):
            for c in range(C):
                for ol in range(out_l):
                    grad_val = grad_output[n][c][ol]
                    count = self._count_cache[n][c][ol]
                    
                    for k in range(self.kernel_size):
                        l_idx = ol * self.stride + k - self.padding
                        if 0 <= l_idx < L:
                            grad_input[n][c][l_idx] += grad_val / count
        
        return grad_input


class MaxPool3d:
    """3D最大池化层"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0):
        if isinstance(kernel_size, int):
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size, kernel_size, kernel_size
        else:
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size
        
        if stride is None:
            self.stride_d, self.stride_h, self.stride_w = self.kernel_d, self.kernel_h, self.kernel_w
        elif isinstance(stride, int):
            self.stride_d, self.stride_h, self.stride_w = stride, stride, stride
        else:
            self.stride_d, self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_d, self.pad_h, self.pad_w = padding, padding, padding
        else:
            self.pad_d, self.pad_h, self.pad_w = padding
        
        self._input_cache = None
        self._indices_cache = None
    
    def forward(self, x: List[List[List[Union[List[List[float]], 'torch.Tensor']]]]) -> List[List[List[Union[List[List[float]], 'torch.Tensor']]]]:
        """前向传播 x: (N, C, D, H, W)"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        D = len(x[0][0]) if C > 0 else 0
        H = len(x[0][0][0]) if D > 0 else 0
        W = len(x[0][0][0][0]) if H > 0 else 0
        
        out_d = (D + 2 * self.pad_d - self.kernel_d) // self.stride_d + 1
        out_h = (H + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (W + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        
        output = [[[[[float('-inf') for _ in range(out_w)] for _ in range(out_h)] 
                    for _ in range(out_d)] for _ in range(C)] for _ in range(N)]
        indices = [[[[[(-1, -1, -1) for _ in range(out_w)] for _ in range(out_h)] 
                    for _ in range(out_d)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for od in range(out_d):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            max_val = float('-inf')
                            max_idx = (-1, -1, -1)
                            
                            for kd in range(self.kernel_d):
                                for kh in range(self.kernel_h):
                                    for kw in range(self.kernel_w):
                                        d_idx = od * self.stride_d + kd - self.pad_d
                                        h_idx = oh * self.stride_h + kh - self.pad_h
                                        w_idx = ow * self.stride_w + kw - self.pad_w
                                        
                                        if 0 <= d_idx < D and 0 <= h_idx < H and 0 <= w_idx < W:
                                            val = x[n][c][d_idx][h_idx][w_idx]
                                            if val > max_val:
                                                max_val = val
                                                max_idx = (d_idx, h_idx, w_idx)
                            
                            output[n][c][od][oh][ow] = max_val
                            indices[n][c][od][oh][ow] = max_idx
        
        self._indices_cache = indices
        return output
    
    def backward(self, grad_output: List[List[List[Union[List[List[float]], 'torch.Tensor']]]]) -> List[List[List[Union[List[List[float]], 'torch.Tensor']]]]:
        """反向传播"""
        if self._input_cache is None or self._indices_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        D = len(x[0][0]) if C > 0 else 0
        H = len(x[0][0][0]) if D > 0 else 0
        W = len(x[0][0][0][0]) if H > 0 else 0
        
        grad_input = [[[[[0.0 for _ in range(W)] for _ in range(H)] 
                       for _ in range(D)] for _ in range(C)] for _ in range(N)]
        
        out_d = len(grad_output[0][0])
        out_h = len(grad_output[0][0][0])
        out_w = len(grad_output[0][0][0][0])
        
        for n in range(N):
            for c in range(C):
                for od in range(out_d):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            d_idx, h_idx, w_idx = self._indices_cache[n][c][od][oh][ow]
                            if d_idx >= 0 and h_idx >= 0 and w_idx >= 0:
                                grad_input[n][c][d_idx][h_idx][w_idx] += grad_output[n][c][od][oh][ow]
        
        return grad_input


class AvgPool3d:
    """3D平均池化层"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0):
        if isinstance(kernel_size, int):
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size, kernel_size, kernel_size
        else:
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size
        
        if stride is None:
            self.stride_d, self.stride_h, self.stride_w = self.kernel_d, self.kernel_h, self.kernel_w
        elif isinstance(stride, int):
            self.stride_d, self.stride_h, self.stride_w = stride, stride, stride
        else:
            self.stride_d, self.stride_h, self.stride_w = stride
        
        if isinstance(padding, int):
            self.pad_d, self.pad_h, self.pad_w = padding, padding, padding
        else:
            self.pad_d, self.pad_h, self.pad_w = padding
        
        self._input_cache = None
        self._count_cache = None
    
    def forward(self, x: List[List[List[Union[List[List[float]], 'torch.Tensor']]]]) -> List[List[List[Union[List[List[float]], 'torch.Tensor']]]]:
        """前向传播 x: (N, C, D, H, W)"""
        self._input_cache = x
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        D = len(x[0][0]) if C > 0 else 0
        H = len(x[0][0][0]) if D > 0 else 0
        W = len(x[0][0][0][0]) if H > 0 else 0
        
        out_d = (D + 2 * self.pad_d - self.kernel_d) // self.stride_d + 1
        out_h = (H + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (W + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        
        output = [[[[[0.0 for _ in range(out_w)] for _ in range(out_h)] 
                    for _ in range(out_d)] for _ in range(C)] for _ in range(N)]
        self._count_cache = [[[[[0 for _ in range(out_w)] for _ in range(out_h)] 
                              for _ in range(out_d)] for _ in range(C)] for _ in range(N)]
        
        for n in range(N):
            for c in range(C):
                for od in range(out_d):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            total = 0.0
                            count = 0
                            
                            for kd in range(self.kernel_d):
                                for kh in range(self.kernel_h):
                                    for kw in range(self.kernel_w):
                                        d_idx = od * self.stride_d + kd - self.pad_d
                                        h_idx = oh * self.stride_h + kh - self.pad_h
                                        w_idx = ow * self.stride_w + kw - self.pad_w
                                        
                                        if 0 <= d_idx < D and 0 <= h_idx < H and 0 <= w_idx < W:
                                            total += x[n][c][d_idx][h_idx][w_idx]
                                            count += 1
                            
                            if count > 0:
                                output[n][c][od][oh][ow] = total / count
                            self._count_cache[n][c][od][oh][ow] = count if count > 0 else 1
        
        return output
    
    def backward(self, grad_output: List[List[List[Union[List[List[float]], 'torch.Tensor']]]]) -> List[List[List[Union[List[List[float]], 'torch.Tensor']]]]:
        """反向传播"""
        if self._input_cache is None or self._count_cache is None:
            raise ValueError("需要先调用forward")
        
        x = self._input_cache
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        D = len(x[0][0]) if C > 0 else 0
        H = len(x[0][0][0]) if D > 0 else 0
        W = len(x[0][0][0][0]) if H > 0 else 0
        
        grad_input = [[[[[0.0 for _ in range(W)] for _ in range(H)] 
                       for _ in range(D)] for _ in range(C)] for _ in range(N)]
        
        out_d = len(grad_output[0][0])
        out_h = len(grad_output[0][0][0])
        out_w = len(grad_output[0][0][0][0])
        
        for n in range(N):
            for c in range(C):
                for od in range(out_d):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            grad_val = grad_output[n][c][od][oh][ow]
                            count = self._count_cache[n][c][od][oh][ow]
                            
                            for kd in range(self.kernel_d):
                                for kh in range(self.kernel_h):
                                    for kw in range(self.kernel_w):
                                        d_idx = od * self.stride_d + kd - self.pad_d
                                        h_idx = oh * self.stride_h + kh - self.pad_h
                                        w_idx = ow * self.stride_w + kw - self.pad_w
                                        
                                        if 0 <= d_idx < D and 0 <= h_idx < H and 0 <= w_idx < W:
                                            grad_input[n][c][d_idx][h_idx][w_idx] += grad_val / count
        
        return grad_input


# 工厂函数
def max_pool1d(kernel_size: int, stride: Optional[int] = None, 
               padding: int = 0, **kwargs) -> MaxPool1d:
    """创建1D最大池化层"""
    return MaxPool1d(kernel_size, stride, padding, **kwargs)


def max_pool2d(kernel_size: Union[int, Tuple[int, int]], 
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0, **kwargs) -> MaxPool2d:
    """创建2D最大池化层"""
    return MaxPool2d(kernel_size, stride, padding, **kwargs)


def max_pool3d(kernel_size: Union[int, Tuple[int, int, int]],
               stride: Optional[Union[int, Tuple[int, int, int]]] = None,
               padding: Union[int, Tuple[int, int, int]] = 0, **kwargs) -> MaxPool3d:
    """创建3D最大池化层"""
    return MaxPool3d(kernel_size, stride, padding, **kwargs)


def avg_pool1d(kernel_size: int, stride: Optional[int] = None,
               padding: int = 0, **kwargs) -> AvgPool1d:
    """创建1D平均池化层"""
    return AvgPool1d(kernel_size, stride, padding, **kwargs)


def avg_pool2d(kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0, **kwargs) -> AvgPool2d:
    """创建2D平均池化层"""
    return AvgPool2d(kernel_size, stride, padding, **kwargs)


def avg_pool3d(kernel_size: Union[int, Tuple[int, int, int]],
               stride: Optional[Union[int, Tuple[int, int, int]]] = None,
               padding: Union[int, Tuple[int, int, int]] = 0, **kwargs) -> AvgPool3d:
    """创建3D平均池化层"""
    return AvgPool3d(kernel_size, stride, padding, **kwargs)


def adaptive_max_pool2d(output_size: Union[int, Tuple[int, int]]) -> AdaptiveMaxPool2d:
    """创建自适应最大池化层"""
    return AdaptiveMaxPool2d(output_size)


def adaptive_avg_pool2d(output_size: Union[int, Tuple[int, int]]) -> AdaptiveAvgPool2d:
    """创建自适应平均池化层"""
    return AdaptiveAvgPool2d(output_size)


def global_avg_pool2d(keepdim: bool = True) -> GlobalAvgPool2d:
    """创建全局平均池化层"""
    return GlobalAvgPool2d(keepdim)


def global_max_pool2d(keepdim: bool = True) -> GlobalMaxPool2d:
    """创建全局最大池化层"""
    return GlobalMaxPool2d(keepdim)


def lp_pool2d(norm_type: float, kernel_size: Union[int, Tuple[int, int]],
              stride: Optional[Union[int, Tuple[int, int]]] = None, **kwargs) -> LPPool2d:
    """创建Lp范数池化层"""
    return LPPool2d(norm_type, kernel_size, stride, **kwargs)


def fractional_max_pool2d(kernel_size: Union[int, Tuple[int, int]],
                          output_ratio: Optional[float] = None,
                          output_size: Optional[Union[int, Tuple[int, int]]] = None,
                          **kwargs) -> FractionalMaxPool2d:
    """创建分数最大池化层"""
    return FractionalMaxPool2d(kernel_size, output_ratio, output_size, **kwargs)


def stochastic_pool2d(kernel_size: Union[int, Tuple[int, int]],
                      stride: Optional[Union[int, Tuple[int, int]]] = None,
                      padding: Union[int, Tuple[int, int]] = 0, **kwargs) -> StochasticPool2d:
    """创建随机池化层"""
    return StochasticPool2d(kernel_size, stride, padding, **kwargs)


def mixed_pool2d(kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 alpha: float = 0.5) -> MixedPool2d:
    """创建混合池化层"""
    return MixedPool2d(kernel_size, stride, padding, alpha)


def power_average_pool2d(kernel_size: Union[int, Tuple[int, int]],
                         stride: Optional[Union[int, Tuple[int, int]]] = None,
                         padding: Union[int, Tuple[int, int]] = 0,
                         power: float = 2.0) -> PowerAveragePool2d:
    """创建幂平均池化层"""
    return PowerAveragePool2d(kernel_size, stride, padding, power)
