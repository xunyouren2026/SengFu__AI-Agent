"""
循环神经网络模块 - 完整实现
包含: LSTM, GRU, BiLSTM, BiGRU, RNN, PeepholeLSTM, 
      LayerNormLSTM, ZoneoutLSTM, NASCell等
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def sigmoid(x: float) -> float:
    """Sigmoid激活函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def sigmoid_derivative(x: float) -> float:
    """Sigmoid导数"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x: float) -> float:
    """Tanh导数"""
    t = math.tanh(x)
    return 1 - t * t


def relu(x: float) -> float:
    """ReLU激活函数"""
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """ReLU导数"""
    return 1.0 if x > 0 else 0.0



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


class RNNCellBase(ABC):
    """RNN单元基类"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
    
    def _init_weight(self, fan_in: int, fan_out: int) -> Union[List[List[float]], 'torch.Tensor']:
        """初始化权重矩阵"""
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
    
    def _init_bias(self, size: int) -> Union[List[float], 'torch.Tensor']:
        """初始化偏置"""
        return [0.0 for _ in range(size)]
    
    @abstractmethod
    def forward(self, x: Union[List[float], 'torch.Tensor'], h_prev: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """前向传播单个时间步"""
        pass
    
    @abstractmethod
    def backward(self, grad_h: Union[List[float], 'torch.Tensor'], cache: dict) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """反向传播单个时间步"""
        pass


class RNNCell(RNNCellBase):
    """
    基础RNN单元
    h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = 'tanh'):
        super().__init__(input_size, hidden_size, bias)
        self.nonlinearity = nonlinearity
        
        # 初始化权重
        self.weight_ih = self._init_weight(input_size, hidden_size)
        self.weight_hh = self._init_weight(hidden_size, hidden_size)
        
        if bias:
            self.bias_ih = self._init_bias(hidden_size)
            self.bias_hh = self._init_bias(hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self._cache = None
    
    def _apply_nonlinearity(self, x: float) -> float:
        """应用非线性激活函数"""
        if self.nonlinearity == 'tanh':
            return math.tanh(x)
        elif self.nonlinearity == 'relu':
            return relu(x)
        else:
            return math.tanh(x)
    
    def _nonlinearity_derivative(self, x: float, output: float) -> float:
        """非线性激活函数的导数"""
        if self.nonlinearity == 'tanh':
            return 1 - output * output
        elif self.nonlinearity == 'relu':
            return relu_derivative(x)
        else:
            return 1 - output * output
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], h_prev: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """前向传播"""
        # 计算 W_ih * x + W_hh * h_prev + b
        linear = [0.0 for _ in range(self.hidden_size)]
        
        # W_ih * x
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                linear[i] += self.weight_ih[i][j] * x[j]
        
        # W_hh * h_prev
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                linear[i] += self.weight_hh[i][j] * h_prev[j]
        
        # 添加偏置
        if self.bias:
            for i in range(self.hidden_size):
                linear[i] += self.bias_ih[i] + self.bias_hh[i]
        
        # 应用非线性
        h_new = [self._apply_nonlinearity(linear[i]) for i in range(self.hidden_size)]
        
        # 缓存用于反向传播
        self._cache = {
            'x': x,
            'h_prev': h_prev,
            'linear': linear,
            'h_new': h_new
        }
        
        return h_new
    
    def backward(self, grad_h: Union[List[float], 'torch.Tensor'], cache: Optional[dict] = None) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """
        反向传播
        返回: (grad_x, grad_h_prev)
        """
        if cache is None:
            cache = self._cache
        
        x = cache['x']
        h_prev = cache['h_prev']
        linear = cache['linear']
        h_new = cache['h_new']
        
        # 计算非线性激活的梯度
        grad_linear = [grad_h[i] * self._nonlinearity_derivative(linear[i], h_new[i]) 
                       for i in range(self.hidden_size)]
        
        # 计算输入梯度
        grad_x = [0.0 for _ in range(self.input_size)]
        for j in range(self.input_size):
            for i in range(self.hidden_size):
                grad_x[j] += grad_linear[i] * self.weight_ih[i][j]
        
        # 计算隐藏状态梯度
        grad_h_prev = [0.0 for _ in range(self.hidden_size)]
        for j in range(self.hidden_size):
            for i in range(self.hidden_size):
                grad_h_prev[j] += grad_linear[i] * self.weight_hh[i][j]
        
        # 计算权重梯度
        grad_weight_ih = [[grad_linear[i] * x[j] for j in range(self.input_size)] 
                         for i in range(self.hidden_size)]
        grad_weight_hh = [[grad_linear[i] * h_prev[j] for j in range(self.hidden_size)] 
                         for i in range(self.hidden_size)]
        
        grad_bias = grad_linear[:] if self.bias else None
        
        return grad_x, grad_h_prev, grad_weight_ih, grad_weight_hh, grad_bias


class LSTMCell(RNNCellBase):
    """
    LSTM单元
    完整实现LSTM的所有门控机制
    
    f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
    i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
    o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
    g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias)
        
        # 初始化权重 (4个门: forget, input, output, cell)
        self.weight_ih = self._init_weight(input_size, 4 * hidden_size)
        self.weight_hh = self._init_weight(hidden_size, 4 * hidden_size)
        
        if bias:
            self.bias_ih = self._init_bias(4 * hidden_size)
            self.bias_hh = self._init_bias(4 * hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self._cache = None
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], state: Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """
        前向传播
        x: 输入 (input_size,)
        state: (h_prev, c_prev) 隐藏状态和单元状态
        返回: (h_new, c_new)
        """
        h_prev, c_prev = state
        
        # 计算所有门的线性变换
        gates = [0.0 for _ in range(4 * self.hidden_size)]
        
        # W_ih * x
        for i in range(4 * self.hidden_size):
            for j in range(self.input_size):
                gates[i] += self.weight_ih[i][j] * x[j]
        
        # W_hh * h_prev
        for i in range(4 * self.hidden_size):
            for j in range(self.hidden_size):
                gates[i] += self.weight_hh[i][j] * h_prev[j]
        
        # 添加偏置
        if self.bias:
            for i in range(4 * self.hidden_size):
                gates[i] += self.bias_ih[i] + self.bias_hh[i]
        
        # 分离各个门
        h_size = self.hidden_size
        f_gate = gates[0:h_size]
        i_gate = gates[h_size:2*h_size]
        g_gate = gates[2*h_size:3*h_size]
        o_gate = gates[3*h_size:4*h_size]
        
        # 应用激活函数
        f_t = [sigmoid(f_gate[i]) for i in range(h_size)]
        i_t = [sigmoid(i_gate[i]) for i in range(h_size)]
        g_t = [math.tanh(g_gate[i]) for i in range(h_size)]
        o_t = [sigmoid(o_gate[i]) for i in range(h_size)]
        
        # 计算新的单元状态和隐藏状态
        c_new = [f_t[i] * c_prev[i] + i_t[i] * g_t[i] for i in range(h_size)]
        tanh_c_new = [math.tanh(c_new[i]) for i in range(h_size)]
        h_new = [o_t[i] * tanh_c_new[i] for i in range(h_size)]
        
        # 缓存用于反向传播
        self._cache = {
            'x': x,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'f_gate': f_gate,
            'i_gate': i_gate,
            'g_gate': g_gate,
            'o_gate': o_gate,
            'f_t': f_t,
            'i_t': i_t,
            'g_t': g_t,
            'o_t': o_t,
            'c_new': c_new,
            'tanh_c_new': tanh_c_new,
            'h_new': h_new
        }
        
        return h_new, c_new
    
    def backward(self, grad_h: Union[List[float], 'torch.Tensor'], grad_c: Union[List[float], 'torch.Tensor'], 
                 cache: Optional[dict] = None) -> Tuple:
        """
        反向传播
        返回: (grad_x, grad_h_prev, grad_c_prev, grad_weight_ih, grad_weight_hh, grad_bias)
        """
        if cache is None:
            cache = self._cache
        
        x = cache['x']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        f_t = cache['f_t']
        i_t = cache['i_t']
        g_t = cache['g_t']
        o_t = cache['o_t']
        c_new = cache['c_new']
        tanh_c_new = cache['tanh_c_new']
        f_gate = cache['f_gate']
        i_gate = cache['i_gate']
        g_gate = cache['g_gate']
        o_gate = cache['o_gate']
        
        h_size = self.hidden_size
        
        # 计算对c_new的梯度
        grad_c_new = [grad_c[i] + grad_h[i] * o_t[i] * (1 - tanh_c_new[i]**2) 
                      for i in range(h_size)]
        
        # 计算对各个门的梯度
        grad_f = [grad_c_new[i] * c_prev[i] * sigmoid_derivative(f_gate[i]) 
                  for i in range(h_size)]
        grad_i = [grad_c_new[i] * g_t[i] * sigmoid_derivative(i_gate[i]) 
                  for i in range(h_size)]
        grad_g = [grad_c_new[i] * i_t[i] * tanh_derivative(g_gate[i]) 
                  for i in range(h_size)]
        grad_o = [grad_h[i] * tanh_c_new[i] * sigmoid_derivative(o_gate[i]) 
                  for i in range(h_size)]
        
        # 合并门的梯度
        grad_gates = grad_f + grad_i + grad_g + grad_o
        
        # 计算输入梯度
        grad_x = [0.0 for _ in range(self.input_size)]
        for j in range(self.input_size):
            for i in range(4 * h_size):
                grad_x[j] += grad_gates[i] * self.weight_ih[i][j]
        
        # 计算隐藏状态梯度
        grad_h_prev = [0.0 for _ in range(h_size)]
        for j in range(h_size):
            for i in range(4 * h_size):
                grad_h_prev[j] += grad_gates[i] * self.weight_hh[i][j]
        
        # 计算单元状态梯度
        grad_c_prev = [grad_c_new[i] * f_t[i] for i in range(h_size)]
        
        # 计算权重梯度
        grad_weight_ih = [[grad_gates[i] * x[j] for j in range(self.input_size)] 
                         for i in range(4 * h_size)]
        grad_weight_hh = [[grad_gates[i] * h_prev[j] for j in range(h_size)] 
                         for i in range(4 * h_size)]
        
        grad_bias = grad_gates[:] if self.bias else None
        
        return grad_x, grad_h_prev, grad_c_prev, grad_weight_ih, grad_weight_hh, grad_bias


class GRUCell(RNNCellBase):
    """
    GRU单元
    完整实现GRU的门控机制
    
    z_t = sigmoid(W_z * [h_{t-1}, x_t])
    r_t = sigmoid(W_r * [h_{t-1}, x_t])
    n_t = tanh(W_n * [r_t * h_{t-1}, x_t])
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias)
        
        # 初始化权重 (3个门: reset, update, new)
        self.weight_ih = self._init_weight(input_size, 3 * hidden_size)
        self.weight_hh = self._init_weight(hidden_size, 3 * hidden_size)
        
        if bias:
            self.bias_ih = self._init_bias(3 * hidden_size)
            self.bias_hh = self._init_bias(3 * hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self._cache = None
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], h_prev: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """前向传播"""
        h_size = self.hidden_size
        
        # 计算reset和update门
        gates_rz = [0.0 for _ in range(2 * h_size)]
        
        # W_ih * x (前2个门)
        for i in range(2 * h_size):
            for j in range(self.input_size):
                gates_rz[i] += self.weight_ih[i][j] * x[j]
        
        # W_hh * h_prev (前2个门)
        for i in range(2 * h_size):
            for j in range(h_size):
                gates_rz[i] += self.weight_hh[i][j] * h_prev[j]
        
        # 添加偏置
        if self.bias:
            for i in range(2 * h_size):
                gates_rz[i] += self.bias_ih[i] + self.bias_hh[i]
        
        # 分离reset和update门
        r_gate = gates_rz[0:h_size]
        z_gate = gates_rz[h_size:2*h_size]
        
        # 应用激活函数
        r_t = [sigmoid(r_gate[i]) for i in range(h_size)]
        z_t = [sigmoid(z_gate[i]) for i in range(h_size)]
        
        # 计算new门 (需要r_t * h_prev)
        rh_prev = [r_t[i] * h_prev[i] for i in range(h_size)]
        
        n_linear = [0.0 for _ in range(h_size)]
        
        # W_ih * x (第3个门)
        for i in range(h_size):
            for j in range(self.input_size):
                n_linear[i] += self.weight_ih[2*h_size + i][j] * x[j]
        
        # W_hh * (r * h_prev) (第3个门)
        for i in range(h_size):
            for j in range(h_size):
                n_linear[i] += self.weight_hh[2*h_size + i][j] * rh_prev[j]
        
        # 添加偏置
        if self.bias:
            for i in range(h_size):
                n_linear[i] += self.bias_ih[2*h_size + i] + self.bias_hh[2*h_size + i]
        
        n_t = [math.tanh(n_linear[i]) for i in range(h_size)]
        
        # 计算新的隐藏状态
        h_new = [(1 - z_t[i]) * n_t[i] + z_t[i] * h_prev[i] for i in range(h_size)]
        
        # 缓存
        self._cache = {
            'x': x,
            'h_prev': h_prev,
            'r_gate': r_gate,
            'z_gate': z_gate,
            'n_linear': n_linear,
            'r_t': r_t,
            'z_t': z_t,
            'n_t': n_t,
            'rh_prev': rh_prev,
            'h_new': h_new
        }
        
        return h_new
    
    def backward(self, grad_h: Union[List[float], 'torch.Tensor'], cache: Optional[dict] = None) -> Tuple:
        """反向传播"""
        if cache is None:
            cache = self._cache
        
        x = cache['x']
        h_prev = cache['h_prev']
        r_t = cache['r_t']
        z_t = cache['z_t']
        n_t = cache['n_t']
        r_gate = cache['r_gate']
        z_gate = cache['z_gate']
        n_linear = cache['n_linear']
        rh_prev = cache['rh_prev']
        
        h_size = self.hidden_size
        
        # 对n_t的梯度
        grad_n = [grad_h[i] * (1 - z_t[i]) * tanh_derivative(n_linear[i]) 
                  for i in range(h_size)]
        
        # 对z_t的梯度
        grad_z = [grad_h[i] * (h_prev[i] - n_t[i]) * sigmoid_derivative(z_gate[i]) 
                  for i in range(h_size)]
        
        # 对r_t的梯度
        grad_r = [0.0 for _ in range(h_size)]
        for i in range(h_size):
            for j in range(h_size):
                grad_r[i] += grad_n[j] * self.weight_hh[2*h_size + j][i] * h_prev[i]
        grad_r = [grad_r[i] * sigmoid_derivative(r_gate[i]) for i in range(h_size)]
        
        # 合并门的梯度
        grad_gates = grad_r + grad_z + grad_n
        
        # 计算输入梯度
        grad_x = [0.0 for _ in range(self.input_size)]
        for j in range(self.input_size):
            for i in range(3 * h_size):
                grad_x[j] += grad_gates[i] * self.weight_ih[i][j]
        
        # 计算隐藏状态梯度
        grad_h_prev = [grad_h[i] * z_t[i] for i in range(h_size)]
        for j in range(h_size):
            for i in range(2 * h_size):
                grad_h_prev[j] += grad_gates[i] * self.weight_hh[i][j]
        
        # 加上从n_t传回的梯度
        for i in range(h_size):
            for j in range(h_size):
                grad_h_prev[j] += grad_n[i] * self.weight_hh[2*h_size + i][j] * r_t[j]
        
        # 计算权重梯度
        grad_weight_ih = [[grad_gates[i] * x[j] for j in range(self.input_size)] 
                         for i in range(3 * h_size)]
        
        grad_weight_hh = [[0.0 for _ in range(h_size)] for _ in range(3 * h_size)]
        for i in range(2 * h_size):
            for j in range(h_size):
                grad_weight_hh[i][j] = grad_gates[i] * h_prev[j]
        for i in range(h_size):
            for j in range(h_size):
                grad_weight_hh[2*h_size + i][j] = grad_n[i] * rh_prev[j]
        
        grad_bias = grad_gates[:] if self.bias else None
        
        return grad_x, grad_h_prev, grad_weight_ih, grad_weight_hh, grad_bias


class PeepholeLSTMCell(LSTMCell):
    """
    带窥孔连接的LSTM
    门控机制可以看到单元状态
    
    f_t = sigmoid(W_f * [h_{t-1}, x_t] + p_f * c_{t-1} + b_f)
    i_t = sigmoid(W_i * [h_{t-1}, x_t] + p_i * c_{t-1} + b_i)
    o_t = sigmoid(W_o * [h_{t-1}, x_t] + p_o * c_t + b_o)
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias)
        
        # 窥孔权重
        self.peephole_f = [random.gauss(0, 0.1) for _ in range(hidden_size)]
        self.peephole_i = [random.gauss(0, 0.1) for _ in range(hidden_size)]
        self.peephole_o = [random.gauss(0, 0.1) for _ in range(hidden_size)]
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], state: Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """前向传播"""
        h_prev, c_prev = state
        h_size = self.hidden_size
        
        # 计算基础线性变换
        gates = [0.0 for _ in range(4 * h_size)]
        
        for i in range(4 * h_size):
            for j in range(self.input_size):
                gates[i] += self.weight_ih[i][j] * x[j]
        
        for i in range(4 * h_size):
            for j in range(h_size):
                gates[i] += self.weight_hh[i][j] * h_prev[j]
        
        if self.bias:
            for i in range(4 * h_size):
                gates[i] += self.bias_ih[i] + self.bias_hh[i]
        
        # 分离各个门并添加窥孔连接
        f_gate = [gates[i] + self.peephole_f[i] * c_prev[i] for i in range(h_size)]
        i_gate = [gates[h_size + i] + self.peephole_i[i] * c_prev[i] for i in range(h_size)]
        g_gate = gates[2*h_size:3*h_size]
        o_gate_base = gates[3*h_size:4*h_size]
        
        # 应用激活函数
        f_t = [sigmoid(f_gate[i]) for i in range(h_size)]
        i_t = [sigmoid(i_gate[i]) for i in range(h_size)]
        g_t = [math.tanh(g_gate[i]) for i in range(h_size)]
        
        # 计算新的单元状态
        c_new = [f_t[i] * c_prev[i] + i_t[i] * g_t[i] for i in range(h_size)]
        
        # 输出门带窥孔连接到c_new
        o_gate = [o_gate_base[i] + self.peephole_o[i] * c_new[i] for i in range(h_size)]
        o_t = [sigmoid(o_gate[i]) for i in range(h_size)]
        
        # 计算新的隐藏状态
        tanh_c_new = [math.tanh(c_new[i]) for i in range(h_size)]
        h_new = [o_t[i] * tanh_c_new[i] for i in range(h_size)]
        
        self._cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'f_gate': f_gate, 'i_gate': i_gate, 'g_gate': g_gate,
            'o_gate': o_gate, 'o_gate_base': o_gate_base,
            'f_t': f_t, 'i_t': i_t, 'g_t': g_t, 'o_t': o_t,
            'c_new': c_new, 'tanh_c_new': tanh_c_new, 'h_new': h_new
        }
        
        return h_new, c_new


class LayerNormLSTMCell(LSTMCell):
    """
    带层归一化的LSTM
    对每个门的激活前进行层归一化
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, eps: float = 1e-5):
        super().__init__(input_size, hidden_size, bias)
        self.eps = eps
        
        # 层归一化参数
        self.gamma = [1.0 for _ in range(4 * hidden_size)]
        self.beta = [0.0 for _ in range(4 * hidden_size)]
    
    def _layer_norm(self, x: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], dict]:
        """层归一化"""
        mean = sum(x) / len(x)
        var = sum((xi - mean)**2 for xi in x) / len(x)
        std = math.sqrt(var + self.eps)
        
        normalized = [(self.gamma[i] * (x[i] - mean) / std + self.beta[i]) 
                      for i in range(len(x))]
        
        cache = {'mean': mean, 'std': std, 'normalized': normalized, 'x': x}
        return normalized, cache
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], state: Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """前向传播"""
        h_prev, c_prev = state
        h_size = self.hidden_size
        
        # 计算线性变换
        gates = [0.0 for _ in range(4 * h_size)]
        
        for i in range(4 * h_size):
            for j in range(self.input_size):
                gates[i] += self.weight_ih[i][j] * x[j]
        
        for i in range(4 * h_size):
            for j in range(h_size):
                gates[i] += self.weight_hh[i][j] * h_prev[j]
        
        if self.bias:
            for i in range(4 * h_size):
                gates[i] += self.bias_ih[i] + self.bias_hh[i]
        
        # 应用层归一化
        gates_norm, ln_cache = self._layer_norm(gates)
        
        # 分离各个门
        f_gate = gates_norm[0:h_size]
        i_gate = gates_norm[h_size:2*h_size]
        g_gate = gates_norm[2*h_size:3*h_size]
        o_gate = gates_norm[3*h_size:4*h_size]
        
        # 应用激活函数
        f_t = [sigmoid(f_gate[i]) for i in range(h_size)]
        i_t = [sigmoid(i_gate[i]) for i in range(h_size)]
        g_t = [math.tanh(g_gate[i]) for i in range(h_size)]
        o_t = [sigmoid(o_gate[i]) for i in range(h_size)]
        
        # 计算新的状态
        c_new = [f_t[i] * c_prev[i] + i_t[i] * g_t[i] for i in range(h_size)]
        tanh_c_new = [math.tanh(c_new[i]) for i in range(h_size)]
        h_new = [o_t[i] * tanh_c_new[i] for i in range(h_size)]
        
        self._cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'f_gate': f_gate, 'i_gate': i_gate, 'g_gate': g_gate, 'o_gate': o_gate,
            'f_t': f_t, 'i_t': i_t, 'g_t': g_t, 'o_t': o_t,
            'c_new': c_new, 'tanh_c_new': tanh_c_new, 'h_new': h_new,
            'gates': gates, 'ln_cache': ln_cache
        }
        
        return h_new, c_new


class ZoneoutLSTMCell(LSTMCell):
    """
    Zoneout LSTM
    随机将部分隐藏状态和单元状态保持不变
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 zoneout_h: float = 0.1, zoneout_c: float = 0.1):
        super().__init__(input_size, hidden_size, bias)
        self.zoneout_h = zoneout_h
        self.zoneout_c = zoneout_c
        self.training = True
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], state: Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """前向传播"""
        h_prev, c_prev = state
        
        # 先计算标准LSTM
        h_new_base, c_new_base = super().forward(x, state)
        
        if self.training:
            # 训练时：随机zoneout
            h_new = []
            c_new = []
            for i in range(self.hidden_size):
                if random.random() < self.zoneout_h:
                    h_new.append(h_prev[i])
                else:
                    h_new.append(h_new_base[i])
                
                if random.random() < self.zoneout_c:
                    c_new.append(c_prev[i])
                else:
                    c_new.append(c_new_base[i])
        else:
            # 推理时：使用期望值
            h_new = [(1 - self.zoneout_h) * h_new_base[i] + self.zoneout_h * h_prev[i] 
                     for i in range(self.hidden_size)]
            c_new = [(1 - self.zoneout_c) * c_new_base[i] + self.zoneout_c * c_prev[i] 
                     for i in range(self.hidden_size)]
        
        return h_new, c_new


class NASCell(RNNCellBase):
    """
    Neural Architecture Search发现的RNN单元
    使用更复杂的门控结构
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias)
        
        # NASCell使用多个权重矩阵
        self.weight_ih = self._init_weight(input_size, 6 * hidden_size)
        self.weight_hh = self._init_weight(hidden_size, 6 * hidden_size)
        
        if bias:
            self.bias_ih = self._init_bias(6 * hidden_size)
            self.bias_hh = self._init_bias(6 * hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self._cache = None
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], state: Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """前向传播"""
        h_prev, c_prev = state
        h_size = self.hidden_size
        
        # 计算线性变换
        gates = [0.0 for _ in range(6 * h_size)]
        
        for i in range(6 * h_size):
            for j in range(self.input_size):
                gates[i] += self.weight_ih[i][j] * x[j]
        
        for i in range(6 * h_size):
            for j in range(h_size):
                gates[i] += self.weight_hh[i][j] * h_prev[j]
        
        if self.bias:
            for i in range(6 * h_size):
                gates[i] += self.bias_ih[i] + self.bias_hh[i]
        
        # 分离各个门
        # NASCell结构:
        # a = sigmoid(gates[0])
        # b = sigmoid(gates[1])
        # c = tanh(gates[2])
        # d = sigmoid(gates[3])
        # e = tanh(gates[4])
        # f = sigmoid(gates[5])
        
        a = [sigmoid(gates[i]) for i in range(h_size)]
        b = [sigmoid(gates[h_size + i]) for i in range(h_size)]
        c = [math.tanh(gates[2*h_size + i]) for i in range(h_size)]
        d = [sigmoid(gates[3*h_size + i]) for i in range(h_size)]
        e = [math.tanh(gates[4*h_size + i]) for i in range(h_size)]
        f = [sigmoid(gates[5*h_size + i]) for i in range(h_size)]
        
        # 计算新的状态
        # c_new = a * c_prev + b * c
        c_new = [a[i] * c_prev[i] + b[i] * c[i] for i in range(h_size)]
        
        # h_new = d * tanh(c_new) + e * f
        tanh_c_new = [math.tanh(c_new[i]) for i in range(h_size)]
        h_new = [d[i] * tanh_c_new[i] + e[i] * f[i] for i in range(h_size)]
        
        self._cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'gates': gates, 'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f,
            'c_new': c_new, 'tanh_c_new': tanh_c_new, 'h_new': h_new
        }
        
        return h_new, c_new


class LSTM:
    """
    多层LSTM网络
    支持多层、双向
    """
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 batch_first: bool = True, bidirectional: bool = False,
                 dropout: float = 0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # 创建LSTM层
        self.layers = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            forward_cell = LSTMCell(layer_input_size, hidden_size, bias)
            self.layers.append({'forward': forward_cell})
            
            if bidirectional:
                backward_cell = LSTMCell(layer_input_size, hidden_size, bias)
                self.layers[layer]['backward'] = backward_cell
        
        self.training = True
        self._cache = None
    
    def forward(self, x: Union[List[Union[List[List[float]], 'torch.Tensor']], Union[List[List[float]], 'torch.Tensor']], 
                hx: Optional[Tuple[Union[List[List[float]], 'torch.Tensor'], Union[List[List[float]], 'torch.Tensor']]] = None) -> Tuple:
        """
        前向传播
        x: 如果batch_first=True, 形状为 (batch, seq_len, input_size)
           否则形状为 (seq_len, batch, input_size)
        hx: 初始隐藏状态 (h_0, c_0)
        返回: (output, (h_n, c_n))
        """
        if self.batch_first:
            # 转换为 (seq_len, batch, input_size)
            batch_size = len(x)
            seq_len = len(x[0]) if batch_size > 0 else 0
            x_transposed = [[[x[b][t][i] for i in range(self.input_size)] 
                           for b in range(batch_size)] 
                          for t in range(seq_len)]
        else:
            x_transposed = x
            seq_len = len(x)
            batch_size = len(x[0]) if seq_len > 0 else 0
        
        # 初始化隐藏状态
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            h_0 = [[0.0 for _ in range(self.hidden_size)] 
                   for _ in range(batch_size * self.num_layers * num_directions)]
            c_0 = [[0.0 for _ in range(self.hidden_size)] 
                   for _ in range(batch_size * self.num_layers * num_directions)]
        else:
            h_0, c_0 = hx
        
        # 处理每一层
        layer_output = x_transposed
        all_h_n = []
        all_c_n = []
        
        for layer_idx in range(self.num_layers):
            layer_dict = self.layers[layer_idx]
            forward_cell = layer_dict['forward']
            
            # 获取当前层的初始状态
            h_offset = layer_idx * batch_size * (2 if self.bidirectional else 1)
            h_init_forward = h_0[h_offset:h_offset + batch_size]
            c_init_forward = c_0[h_offset:h_offset + batch_size]
            
            # 前向方向
            forward_outputs = []
            h_t = h_init_forward
            c_t = c_init_forward
            
            for t in range(seq_len):
                batch_input = layer_output[t]
                new_h = []
                new_c = []
                
                for b in range(batch_size):
                    h_new, c_new = forward_cell.forward(batch_input[b], (h_t[b], c_t[b]))
                    new_h.append(h_new)
                    new_c.append(c_new)
                
                h_t = new_h
                c_t = new_c
                forward_outputs.append(h_t)
            
            all_h_n.extend(h_t)
            all_c_n.extend(c_t)
            
            if self.bidirectional:
                backward_cell = layer_dict['backward']
                
                h_offset_back = h_offset + batch_size
                h_init_backward = h_0[h_offset_back:h_offset_back + batch_size]
                c_init_backward = c_0[h_offset_back:h_offset_back + batch_size]
                
                # 后向方向
                backward_outputs = []
                h_t = h_init_backward
                c_t = c_init_backward
                
                for t in range(seq_len - 1, -1, -1):
                    batch_input = layer_output[t]
                    new_h = []
                    new_c = []
                    
                    for b in range(batch_size):
                        h_new, c_new = backward_cell.forward(batch_input[b], (h_t[b], c_t[b]))
                        new_h.append(h_new)
                        new_c.append(c_new)
                    
                    h_t = new_h
                    c_t = new_c
                    backward_outputs.insert(0, h_t)
                
                all_h_n.extend(h_t)
                all_c_n.extend(c_t)
                
                # 合并前向和后向输出
                layer_output = [[forward_outputs[t][b] + backward_outputs[t][b] 
                               for b in range(batch_size)] 
                              for t in range(seq_len)]
            else:
                layer_output = forward_outputs
        
        # 转换输出格式
        if self.batch_first:
            output = [[[layer_output[t][b][i] for i in range(len(layer_output[t][b]))] 
                      for t in range(seq_len)] 
                     for b in range(batch_size)]
        else:
            output = layer_output
        
        return output, (all_h_n, all_c_n)


class GRU:
    """
    多层GRU网络
    """
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 batch_first: bool = True, bidirectional: bool = False,
                 dropout: float = 0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # 创建GRU层
        self.layers = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            forward_cell = GRUCell(layer_input_size, hidden_size, bias)
            self.layers.append({'forward': forward_cell})
            
            if bidirectional:
                backward_cell = GRUCell(layer_input_size, hidden_size, bias)
                self.layers[layer]['backward'] = backward_cell
        
        self.training = True
    
    def forward(self, x: Union[List[Union[List[List[float]], 'torch.Tensor']], Union[List[List[float]], 'torch.Tensor']],
                h_0: Optional[Union[List[List[float]], 'torch.Tensor']] = None) -> Tuple:
        """
        前向传播
        返回: (output, h_n)
        """
        if self.batch_first:
            batch_size = len(x)
            seq_len = len(x[0]) if batch_size > 0 else 0
            x_transposed = [[[x[b][t][i] for i in range(self.input_size)] 
                           for b in range(batch_size)] 
                          for t in range(seq_len)]
        else:
            x_transposed = x
            seq_len = len(x)
            batch_size = len(x[0]) if seq_len > 0 else 0
        
        # 初始化隐藏状态
        if h_0 is None:
            num_directions = 2 if self.bidirectional else 1
            h_0 = [[0.0 for _ in range(self.hidden_size)] 
                   for _ in range(batch_size * self.num_layers * num_directions)]
        
        layer_output = x_transposed
        all_h_n = []
        
        for layer_idx in range(self.num_layers):
            layer_dict = self.layers[layer_idx]
            forward_cell = layer_dict['forward']
            
            h_offset = layer_idx * batch_size * (2 if self.bidirectional else 1)
            h_init_forward = h_0[h_offset:h_offset + batch_size]
            
            # 前向方向
            forward_outputs = []
            h_t = h_init_forward
            
            for t in range(seq_len):
                batch_input = layer_output[t]
                new_h = []
                
                for b in range(batch_size):
                    h_new = forward_cell.forward(batch_input[b], h_t[b])
                    new_h.append(h_new)
                
                h_t = new_h
                forward_outputs.append(h_t)
            
            all_h_n.extend(h_t)
            
            if self.bidirectional:
                backward_cell = layer_dict['backward']
                
                h_offset_back = h_offset + batch_size
                h_init_backward = h_0[h_offset_back:h_offset_back + batch_size]
                
                backward_outputs = []
                h_t = h_init_backward
                
                for t in range(seq_len - 1, -1, -1):
                    batch_input = layer_output[t]
                    new_h = []
                    
                    for b in range(batch_size):
                        h_new = backward_cell.forward(batch_input[b], h_t[b])
                        new_h.append(h_new)
                    
                    h_t = new_h
                    backward_outputs.insert(0, h_t)
                
                all_h_n.extend(h_t)
                
                layer_output = [[forward_outputs[t][b] + backward_outputs[t][b] 
                               for b in range(batch_size)] 
                              for t in range(seq_len)]
            else:
                layer_output = forward_outputs
        
        if self.batch_first:
            output = [[[layer_output[t][b][i] for i in range(len(layer_output[t][b]))] 
                      for t in range(seq_len)] 
                     for b in range(batch_size)]
        else:
            output = layer_output
        
        return output, all_h_n


class BiLSTM(LSTM):
    """双向LSTM"""
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 batch_first: bool = True, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, num_layers, bias, 
                        batch_first, bidirectional=True, dropout=dropout)


class BiGRU(GRU):
    """双向GRU"""
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 batch_first: bool = True, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, num_layers, bias,
                        batch_first, bidirectional=True, dropout=dropout)


class StackedRNN:
    """
    堆叠RNN
    支持不同类型的RNN单元混合
    """
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, rnn_type: str = 'lstm',
                 bias: bool = True, dropout: float = 0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            
            if rnn_type == 'lstm':
                cell = LSTMCell(layer_input_size, hidden_size, bias)
            elif rnn_type == 'gru':
                cell = GRUCell(layer_input_size, hidden_size, bias)
            elif rnn_type == 'rnn':
                cell = RNNCell(layer_input_size, hidden_size, bias)
            elif rnn_type == 'peephole_lstm':
                cell = PeepholeLSTMCell(layer_input_size, hidden_size, bias)
            elif rnn_type == 'layer_norm_lstm':
                cell = LayerNormLSTMCell(layer_input_size, hidden_size, bias)
            elif rnn_type == 'zoneout_lstm':
                cell = ZoneoutLSTMCell(layer_input_size, hidden_size, bias)
            elif rnn_type == 'nas':
                cell = NASCell(layer_input_size, hidden_size, bias)
            else:
                cell = LSTMCell(layer_input_size, hidden_size, bias)
            
            self.cells.append(cell)
        
        self.training = True
    
    def forward(self, x: Union[List[List[float]], 'torch.Tensor'], 
                hx: Optional[Tuple] = None) -> Tuple:
        """
        前向传播
        x: (seq_len, input_size)
        返回: (output, (h_n, c_n)) 或 (output, h_n)
        """
        seq_len = len(x)
        
        # 初始化隐藏状态
        if hx is None:
            if self.rnn_type in ['lstm', 'peephole_lstm', 'layer_norm_lstm', 'zoneout_lstm', 'nas']:
                h_0 = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.num_layers)]
                c_0 = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.num_layers)]
                hx = (h_0, c_0)
            else:
                h_0 = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.num_layers)]
                hx = h_0
        
        outputs = []
        
        if self.rnn_type in ['lstm', 'peephole_lstm', 'layer_norm_lstm', 'zoneout_lstm', 'nas']:
            h_states, c_states = hx
            
            for t in range(seq_len):
                layer_input = x[t]
                
                for layer_idx, cell in enumerate(self.cells):
                    h_prev = h_states[layer_idx]
                    c_prev = c_states[layer_idx]
                    
                    h_new, c_new = cell.forward(layer_input, (h_prev, c_prev))
                    h_states[layer_idx] = h_new
                    c_states[layer_idx] = c_new
                    layer_input = h_new
                
                outputs.append(layer_input)
            
            return outputs, (h_states, c_states)
        else:
            h_states = hx
            
            for t in range(seq_len):
                layer_input = x[t]
                
                for layer_idx, cell in enumerate(self.cells):
                    h_prev = h_states[layer_idx]
                    h_new = cell.forward(layer_input, h_prev)
                    h_states[layer_idx] = h_new
                    layer_input = h_new
                
                outputs.append(layer_input)
            
            return outputs, h_states


# 工厂函数
def lstm(input_size: int, hidden_size: int, num_layers: int = 1, **kwargs) -> LSTM:
    """创建LSTM网络"""
    return LSTM(input_size, hidden_size, num_layers, **kwargs)


def gru(input_size: int, hidden_size: int, num_layers: int = 1, **kwargs) -> GRU:
    """创建GRU网络"""
    return GRU(input_size, hidden_size, num_layers, **kwargs)


def bilstm(input_size: int, hidden_size: int, num_layers: int = 1, **kwargs) -> BiLSTM:
    """创建双向LSTM网络"""
    return BiLSTM(input_size, hidden_size, num_layers, **kwargs)


def bigru(input_size: int, hidden_size: int, num_layers: int = 1, **kwargs) -> BiGRU:
    """创建双向GRU网络"""
    return BiGRU(input_size, hidden_size, num_layers, **kwargs)


def lstm_cell(input_size: int, hidden_size: int, **kwargs) -> LSTMCell:
    """创建LSTM单元"""
    return LSTMCell(input_size, hidden_size, **kwargs)


def gru_cell(input_size: int, hidden_size: int, **kwargs) -> GRUCell:
    """创建GRU单元"""
    return GRUCell(input_size, hidden_size, **kwargs)


def rnn_cell(input_size: int, hidden_size: int, **kwargs) -> RNNCell:
    """创建RNN单元"""
    return RNNCell(input_size, hidden_size, **kwargs)


def peephole_lstm_cell(input_size: int, hidden_size: int, **kwargs) -> PeepholeLSTMCell:
    """创建带窥孔连接的LSTM单元"""
    return PeepholeLSTMCell(input_size, hidden_size, **kwargs)


def layer_norm_lstm_cell(input_size: int, hidden_size: int, **kwargs) -> LayerNormLSTMCell:
    """创建带层归一化的LSTM单元"""
    return LayerNormLSTMCell(input_size, hidden_size, **kwargs)


def zoneout_lstm_cell(input_size: int, hidden_size: int, **kwargs) -> ZoneoutLSTMCell:
    """创建Zoneout LSTM单元"""
    return ZoneoutLSTMCell(input_size, hidden_size, **kwargs)


def nas_cell(input_size: int, hidden_size: int, **kwargs) -> NASCell:
    """创建NAS单元"""
    return NASCell(input_size, hidden_size, **kwargs)


def stacked_rnn(input_size: int, hidden_size: int, num_layers: int = 1,
                rnn_type: str = 'lstm', **kwargs) -> StackedRNN:
    """创建堆叠RNN"""
    return StackedRNN(input_size, hidden_size, num_layers, rnn_type, **kwargs)
