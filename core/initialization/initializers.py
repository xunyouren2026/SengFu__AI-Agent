"""
权重初始化模块 - 包含各种权重初始化方法的真实实现
包括: Xavier/Glorot, He/Kaiming, Orthogonal, Truncated Normal,
      Sparse, LSUV, Delta-Orthogonal, etc.
"""

import math
import random
from typing import Optional, Tuple, List, Union, Callable
from enum import Enum


class InitType(Enum):
    """初始化类型"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    CONSTANT = "constant"
    ZEROS = "zeros"
    ONES = "ones"
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    ORTHOGONAL = "orthogonal"
    SPARSE = "sparse"
    TRUNCATED_NORMAL = "truncated_normal"


def zeros(shape: Tuple[int, ...]) -> List:
    """零初始化"""
    if len(shape) == 1:
        return [0.0] * shape[0]
    elif len(shape) == 2:
        return [[0.0] * shape[1] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[0.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[0.0] * shape[3] for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def ones(shape: Tuple[int, ...]) -> List:
    """一初始化"""
    if len(shape) == 1:
        return [1.0] * shape[0]
    elif len(shape) == 2:
        return [[1.0] * shape[1] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[1.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[1.0] * shape[3] for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def constant(shape: Tuple[int, ...], value: float) -> List:
    """常数初始化"""
    if len(shape) == 1:
        return [value] * shape[0]
    elif len(shape) == 2:
        return [[value] * shape[1] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[value] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[value] * shape[3] for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def uniform(shape: Tuple[int, ...], a: float = 0.0, b: float = 1.0) -> List:
    """均匀分布初始化"""
    if len(shape) == 1:
        return [random.uniform(a, b) for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[random.uniform(a, b) for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[random.uniform(a, b) for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[random.uniform(a, b) for _ in range(shape[3])] 
                  for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def normal(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> List:
    """正态分布初始化"""
    if len(shape) == 1:
        return [random.gauss(mean, std) for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[random.gauss(mean, std) for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[random.gauss(mean, std) for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[random.gauss(mean, std) for _ in range(shape[3])] 
                  for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def truncated_normal(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0,
                     a: float = -2.0, b: float = 2.0) -> List:
    """
    截断正态分布初始化
    
    只采样在[a, b]范围内的值
    """
    def sample():
        while True:
            x = random.gauss(mean, std)
            if a <= x <= b:
                return x
    
    if len(shape) == 1:
        return [sample() for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[sample() for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[sample() for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 4:
        return [[[[sample() for _ in range(shape[3])] 
                  for _ in range(shape[2])] 
                 for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError(f"Unsupported shape dimension: {len(shape)}")


def xavier_uniform(shape: Tuple[int, ...], gain: float = 1.0) -> List:
    """
    Xavier/Glorot均匀初始化
    
    适用于tanh和sigmoid激活函数
    
    std = gain * sqrt(2 / (fan_in + fan_out))
    bound = sqrt(3) * std
    """
    fan_in, fan_out = _calculate_fan(shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    return uniform(shape, -bound, bound)


def xavier_normal(shape: Tuple[int, ...], gain: float = 1.0) -> List:
    """
    Xavier/Glorot正态初始化
    
    std = gain * sqrt(2 / (fan_in + fan_out))
    """
    fan_in, fan_out = _calculate_fan(shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, 0.0, std)


def kaiming_uniform(shape: Tuple[int, ...], a: float = 0.0, 
                    mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> List:
    """
    Kaiming/He均匀初始化
    
    适用于ReLU及其变体
    
    std = gain / sqrt(fan)
    bound = sqrt(3) * std
    """
    fan_in, fan_out = _calculate_fan(shape)
    
    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    
    return uniform(shape, -bound, bound)


def kaiming_normal(shape: Tuple[int, ...], a: float = 0.0,
                   mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> List:
    """
    Kaiming/He正态初始化
    
    std = gain / sqrt(fan)
    """
    fan_in, fan_out = _calculate_fan(shape)
    
    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    
    return normal(shape, 0.0, std)


def orthogonal(shape: Tuple[int, ...], gain: float = 1.0) -> List:
    """
    正交初始化
    
    适用于RNN/LSTM
    """
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2D shape")
    
    flat_shape = (shape[0], int(math.prod(shape[1:])))
    
    # 生成随机矩阵
    rows, cols = flat_shape
    
    if rows > cols:
        # 生成正交列
        q = _random_orthogonal(rows, cols)
    else:
        # 生成正交行
        q = _random_orthogonal(cols, rows)
        q = [[q[j][i] for j in range(cols)] for i in range(rows)]
    
    # 应用gain
    q = [[q[i][j] * gain for j in range(cols)] for i in range(rows)]
    
    # 重塑为目标形状
    return _reshape(q, shape)


def _random_orthogonal(n: int, m: int) -> List[List[float]]:
    """生成随机正交矩阵 (QR分解)"""
    # 生成随机矩阵
    a = [[random.gauss(0, 1) for _ in range(m)] for _ in range(n)]
    
    # Gram-Schmidt正交化
    q = []
    for j in range(m):
        # 取第j列
        v = [a[i][j] for i in range(n)]
        
        # 减去在已有正交向量上的投影
        for k in range(len(q)):
            proj = sum(v[i] * q[k][i] for i in range(n))
            v = [v[i] - proj * q[k][i] for i in range(n)]
        
        # 归一化
        norm = math.sqrt(sum(v[i] ** 2 for i in range(n)))
        if norm > 1e-10:
            v = [v[i] / norm for i in range(n)]
            q.append(v)
    
    # 转置返回
    return [[q[j][i] for j in range(len(q))] for i in range(n)]


def sparse(shape: Tuple[int, ...], sparsity: float, std: float = 0.01) -> List:
    """
    稀疏初始化
    
    大部分元素为0，只有少数非零
    """
    total_elements = int(math.prod(shape))
    num_nonzero = int(total_elements * (1 - sparsity))
    
    # 初始化为零
    result = zeros(shape)
    
    # 随机选择非零位置
    indices = random.sample(range(total_elements), num_nonzero)
    
    for idx in indices:
        # 计算多维索引
        multi_idx = []
        remaining = idx
        for dim in reversed(shape):
            multi_idx.append(remaining % dim)
            remaining //= dim
        multi_idx = list(reversed(multi_idx))
        
        # 设置值
        _set_value(result, multi_idx, random.gauss(0, std))
    
    return result


def lsuv(shape: Tuple[int, ...], input_data: List, 
         target_mean: float = 0.0, target_std: float = 1.0,
         max_iter: int = 10) -> List:
    """
    LSUV (Layer-Sequential Unit-Variance) 初始化
    
    根据输入数据调整权重，使输出具有目标统计特性
    """
    # 初始正交初始化
    w = orthogonal(shape)
    
    for _ in range(max_iter):
        # 计算输出
        output = _forward(input_data, w)
        
        # 计算统计量
        output_flat = _flatten(output)
        current_mean = sum(output_flat) / len(output_flat)
        current_std = math.sqrt(sum((x - current_mean) ** 2 for x in output_flat) / len(output_flat))
        
        if current_std < 1e-10:
            break
        
        # 调整权重
        scale = target_std / current_std
        w = _scale(w, scale)
        
        # 检查收敛
        if abs(current_std - target_std) < 0.01:
            break
    
    return w


def delta_orthogonal(shape: Tuple[int, ...], gain: float = 1.0) -> List:
    """
    Delta正交初始化
    
    用于深度网络的恒等映射初始化
    """
    if len(shape) < 2:
        raise ValueError("Delta-orthogonal initialization requires at least 2D shape")
    
    rows, cols = shape[0], shape[1]
    
    if rows != cols:
        # 如果不是方阵，使用标准正交初始化
        return orthogonal(shape, gain)
    
    # 创建单位矩阵
    w = [[1.0 if i == j else 0.0 for j in range(cols)] for i in range(rows)]
    
    # 应用gain
    w = [[w[i][j] * gain for j in range(cols)] for i in range(rows)]
    
    # 如果有更多维度，扩展
    if len(shape) > 2:
        w = _reshape(w, shape)
    
    return w


def dirac(shape: Tuple[int, ...], groups: int = 1) -> List:
    """
    Dirac初始化
    
    用于卷积层的恒等映射初始化
    """
    if len(shape) != 4:
        raise ValueError("Dirac initialization is for 4D convolution weights")
    
    out_channels, in_channels, kh, kw = shape
    
    # 初始化为零
    w = zeros(shape)
    
    # 设置中心元素
    channels_per_group = in_channels // groups
    
    for g in range(groups):
        for c in range(channels_per_group):
            out_c = g * channels_per_group + c
            in_c = g * channels_per_group + c
            
            if out_c < out_channels and in_c < in_channels:
                # 设置中心位置为1
                center_h = kh // 2
                center_w = kw // 2
                w[out_c][in_c][center_h][center_w] = 1.0
    
    return w


def _calculate_fan(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """计算fan_in和fan_out"""
    if len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    elif len(shape) >= 3:
        # 卷积层
        fan_in = int(math.prod(shape[1:]))
        fan_out = shape[0] * int(math.prod(shape[2:]))
    else:
        fan_in = fan_out = 1
    
    return fan_in, fan_out


def _calculate_gain(nonlinearity: str, a: float = 0.0) -> float:
    """计算激活函数的gain"""
    if nonlinearity == 'linear' or nonlinearity == 'identity':
        return 1.0
    elif nonlinearity == 'sigmoid':
        return 1.0
    elif nonlinearity == 'tanh':
        return 5.0 / 3.0
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        return math.sqrt(2.0 / (1.0 + a ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4.0
    else:
        return 1.0


def _reshape(data: List, shape: Tuple[int, ...]) -> List:
    """重塑数据"""
    flat = _flatten(data)
    return _unflatten(flat, shape)


def _flatten(data: List) -> List[float]:
    """展平数据"""
    result = []
    for item in data:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _unflatten(flat: List[float], shape: Tuple[int, ...]) -> List:
    """从扁平列表重建多维数组"""
    if len(shape) == 1:
        return flat[:shape[0]]
    
    result = []
    chunk_size = int(math.prod(shape[1:]))
    for i in range(shape[0]):
        chunk = flat[i * chunk_size:(i + 1) * chunk_size]
        result.append(_unflatten(chunk, shape[1:]))
    return result


def _set_value(data: List, indices: List[int], value: float):
    """设置多维数组的值"""
    if len(indices) == 1:
        data[indices[0]] = value
    else:
        _set_value(data[indices[0]], indices[1:], value)


def _scale(data: List, factor: float) -> List:
    """缩放数据"""
    if isinstance(data[0], list):
        return [[_scale(item, factor) if isinstance(item, list) else item * factor 
                 for item in row] for row in data]
    return [item * factor for item in data]


def _forward(input_data: List, weight: List) -> List:
    """简化的前向传播"""
    if isinstance(input_data[0], list) and isinstance(weight[0], list):
        # 矩阵乘法
        return [[sum(input_data[i][k] * weight[k][j] for k in range(len(weight)))
                 for j in range(len(weight[0]))] for i in range(len(input_data))]
    return input_data


# 工厂函数
def get_initializer(name: str, **kwargs) -> Callable:
    """根据名称获取初始化函数"""
    initializers = {
        'zeros': zeros,
        'ones': ones,
        'constant': constant,
        'uniform': uniform,
        'normal': normal,
        'truncated_normal': truncated_normal,
        'xavier_uniform': xavier_uniform,
        'xavier_normal': xavier_normal,
        'kaiming_uniform': kaiming_uniform,
        'kaiming_normal': kaiming_normal,
        'orthogonal': orthogonal,
        'sparse': sparse,
        'delta_orthogonal': delta_orthogonal,
        'dirac': dirac
    }
    
    name_lower = name.lower()
    if name_lower not in initializers:
        raise ValueError(f"Unknown initializer: {name}. Available: {list(initializers.keys())}")
    
    init_fn = initializers[name_lower]
    
    def wrapper(shape):
        return init_fn(shape, **kwargs)
    
    return wrapper


def init_weights(module, init_type: str = 'kaiming_normal', **kwargs):
    """初始化模块权重"""
    init_fn = get_initializer(init_type, **kwargs)
    
    # 这里假设module有weight属性
    # 实际使用时需要根据具体框架调整
    if hasattr(module, 'weight') and module.weight is not None:
        shape = module.weight.shape if hasattr(module.weight, 'shape') else (len(module.weight), len(module.weight[0]))
        module.weight = init_fn(shape)
    
    if hasattr(module, 'bias') and module.bias is not None:
        if hasattr(module.bias, 'shape'):
            shape = module.bias.shape
        else:
            shape = (len(module.bias),)
        module.bias = zeros(shape)
    
    return module
