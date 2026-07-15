"""
归一化层模块 - 包含各种归一化技术的真实实现
包括: BatchNorm, LayerNorm, InstanceNorm, GroupNorm, 
      RMSNorm, WeightNorm, SpectralNorm, SyncBatchNorm
"""

import math
import random
from typing import Optional, Tuple, List, Union, Dict, Any
from collections import defaultdict


class BatchNorm:
    """
    Batch Normalization
    
    y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    
    训练时使用batch统计量，推理时使用running统计量
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = [1.0] * num_features
            self.bias = [0.0] * num_features
        else:
            self.weight = None
            self.bias = None
        
        if track_running_stats:
            self.running_mean = [0.0] * num_features
            self.running_var = [1.0] * num_features
        else:
            self.running_mean = None
            self.running_var = None
        
        self.training = True
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        x: [batch_size, num_features, ...] 或 [H, W, C]
        这里假设 [H, W, C] 格式
        """
        h = len(x)
        w = len(x[0])
        c = len(x[0][0])
        
        if self.training:
            # 计算batch统计量
            mean = [0.0] * c
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        mean[k] += x[i][j][k]
            
            total = h * w
            mean = [m / total for m in mean]
            
            var = [0.0] * c
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        var[k] += (x[i][j][k] - mean[k]) ** 2
            var = [v / total for v in var]
            
            # 更新running统计量
            if self.track_running_stats:
                for k in range(c):
                    self.running_mean[k] = (1 - self.momentum) * self.running_mean[k] + self.momentum * mean[k]
                    self.running_var[k] = (1 - self.momentum) * self.running_var[k] + self.momentum * var[k]
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        output = [[[0.0] * c for _ in range(w)] for _ in range(h)]
        
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    normalized = (x[i][j][k] - mean[k]) / math.sqrt(var[k] + self.eps)
                    
                    if self.affine:
                        output[i][j][k] = self.weight[k] * normalized + self.bias[k]
                    else:
                        output[i][j][k] = normalized
        
        return output
    
    def __call__(self, x):
        return self.forward(x)


class LayerNorm:
    """
    Layer Normalization
    
    对每个样本的所有特征进行归一化
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = [1.0] * normalized_shape
            self.bias = [0.0] * normalized_shape
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """
        x: [batch_size, features] 或 [seq_len, features]
        """
        output = []
        
        for row in x:
            # 计算均值和方差
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            
            # 归一化
            normalized = [(v - mean) / math.sqrt(var + self.eps) for v in row]
            
            # 应用affine变换
            if self.elementwise_affine:
                normalized = [self.weight[i] * normalized[i] + self.bias[i]
                             for i in range(len(normalized))]
            
            output.append(normalized)
        
        return output
    
    def __call__(self, x):
        return self.forward(x)


class InstanceNorm:
    """
    Instance Normalization
    
    对每个样本的每个通道独立归一化
    常用于风格迁移
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        track_running_stats: bool = False
    ):
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = [1.0] * num_features
            self.bias = [0.0] * num_features
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        x: [H, W, C]
        """
        h = len(x)
        w = len(x[0])
        c = len(x[0][0])
        
        output = [[[0.0] * c for _ in range(w)] for _ in range(h)]
        
        for k in range(c):
            # 对每个通道独立归一化
            mean = sum(x[i][j][k] for i in range(h) for j in range(w)) / (h * w)
            var = sum((x[i][j][k] - mean) ** 2 for i in range(h) for j in range(w)) / (h * w)
            
            for i in range(h):
                for j in range(w):
                    normalized = (x[i][j][k] - mean) / math.sqrt(var + self.eps)
                    
                    if self.affine:
                        output[i][j][k] = self.weight[k] * normalized + self.bias[k]
                    else:
                        output[i][j][k] = normalized
        
        return output
    
    def __call__(self, x):
        return self.forward(x)


class GroupNorm:
    """
    Group Normalization
    
    将通道分组，每组独立归一化
    是LayerNorm和InstanceNorm的泛化
    """
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = [1.0] * num_channels
            self.bias = [0.0] * num_channels
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        x: [H, W, C]
        """
        h = len(x)
        w = len(x[0])
        c = len(x[0][0])
        
        channels_per_group = c // self.num_groups
        output = [[[0.0] * c for _ in range(w)] for _ in range(h)]
        
        for g in range(self.num_groups):
            # 计算每组的均值和方差
            start_c = g * channels_per_group
            end_c = start_c + channels_per_group
            
            total = h * w * channels_per_group
            mean = sum(x[i][j][k] for i in range(h) for j in range(w) for k in range(start_c, end_c)) / total
            
            var = sum((x[i][j][k] - mean) ** 2 for i in range(h) for j in range(w) for k in range(start_c, end_c)) / total
            
            # 归一化
            for i in range(h):
                for j in range(w):
                    for k in range(start_c, end_c):
                        normalized = (x[i][j][k] - mean) / math.sqrt(var + self.eps)
                        
                        if self.affine:
                            output[i][j][k] = self.weight[k] * normalized + self.bias[k]
                        else:
                            output[i][j][k] = normalized
        
        return output
    
    def __call__(self, x):
        return self.forward(x)


class RMSNorm:
    """
    RMS Normalization (Root Mean Square Layer Normalization)
    
    不计算均值，只使用RMS进行归一化
    比LayerNorm更高效
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = [1.0] * normalized_shape
            self.bias = [0.0] * normalized_shape
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        output = []
        
        for row in x:
            # 计算RMS
            rms = math.sqrt(sum(v ** 2 for v in row) / len(row) + self.eps)
            
            # 归一化
            normalized = [v / rms for v in row]
            
            # 应用affine变换
            if self.elementwise_affine:
                normalized = [self.weight[i] * normalized[i] + self.bias[i]
                             for i in range(len(normalized))]
            
            output.append(normalized)
        
        return output
    
    def __call__(self, x):
        return self.forward(x)


class WeightNorm:
    """
    Weight Normalization
    
    将权重分解为方向和大小: w = g * v / ||v||
    """
    
    def __init__(self, weight: List[List[float]], dim: int = 0):
        self.dim = dim
        self.weight = weight
        
        # 计算方向和大小
        self._compute_params()
    
    def _compute_params(self):
        """计算方向向量v和大小g"""
        if self.dim == 0:
            # 对每行计算
            self.g = [math.sqrt(sum(w ** 2 for w in row)) for row in self.weight]
            self.v = [[w / g if g > 0 else 0.0 for w in row] 
                     for row, g in zip(self.weight, self.g)]
        else:
            # 对每列计算
            cols = len(self.weight[0])
            self.g = [math.sqrt(sum(self.weight[i][j] ** 2 for i in range(len(self.weight)))) 
                     for j in range(cols)]
            self.v = [[self.weight[i][j] / self.g[j] if self.g[j] > 0 else 0.0 
                      for j in range(cols)] for i in range(len(self.weight))]
    
    def forward(self) -> List[List[float]]:
        """重构权重"""
        if self.dim == 0:
            return [[self.g[i] * self.v[i][j] for j in range(len(self.v[0]))]
                   for i in range(len(self.v))]
        else:
            return [[self.g[j] * self.v[i][j] for j in range(len(self.v[0]))]
                   for i in range(len(self.v))]
    
    def __call__(self):
        return self.forward()


class SpectralNorm:
    """
    Spectral Normalization
    
    通过除以谱范数（最大奇异值）约束权重
    常用于GAN的判别器
    """
    
    def __init__(
        self,
        weight: List[List[float]],
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12
    ):
        self.weight = weight
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        
        # 初始化u和v向量
        if dim == 0:
            n = len(weight)
            m = len(weight[0])
        else:
            n = len(weight[0])
            m = len(weight)
        
        self.u = [1.0 / math.sqrt(n)] * n
        self.v = [1.0 / math.sqrt(m)] * m
    
    def forward(self) -> List[List[float]]:
        """应用谱归一化"""
        # 幂迭代估计最大奇异值
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            wtu = [sum(self.weight[i][j] * self.u[i] for i in range(len(self.weight)))
                  for j in range(len(self.weight[0]))]
            wtu_norm = math.sqrt(sum(x ** 2 for x in wtu)) + self.eps
            self.v = [x / wtu_norm for x in wtu]
            
            # u = W v / ||W v||
            wv = [sum(self.weight[i][j] * self.v[j] for j in range(len(self.weight[0])))
                 for i in range(len(self.weight))]
            wv_norm = math.sqrt(sum(x ** 2 for x in wv)) + self.eps
            self.u = [x / wv_norm for x in wv]
        
        # 计算谱范数
        sigma = sum(sum(self.weight[i][j] * self.v[j] for j in range(len(self.weight[0]))) * self.u[i]
                   for i in range(len(self.weight)))
        
        # 归一化权重
        return [[w / (sigma + self.eps) for w in row] for row in self.weight]
    
    def __call__(self):
        return self.forward()


class SyncBatchNorm:
    """
    Synchronized Batch Normalization
    
    用于多GPU训练，同步所有GPU的统计量
    这里是概念实现，实际需要分布式通信
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = [1.0] * num_features
            self.bias = [0.0] * num_features
        else:
            self.weight = None
            self.bias = None
        
        if track_running_stats:
            self.running_mean = [0.0] * num_features
            self.running_var = [1.0] * num_features
        else:
            self.running_mean = None
            self.running_var = None
        
        self.training = True
        
        # 用于同步的缓冲区
        self._mean_buffer = []
        self._var_buffer = []
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """前向传播（本地计算）"""
        h = len(x)
        w = len(x[0])
        c = len(x[0][0])
        
        if self.training:
            # 计算本地统计量
            mean = [0.0] * c
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        mean[k] += x[i][j][k]
            
            total = h * w
            mean = [m / total for m in mean]
            
            var = [0.0] * c
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        var[k] += (x[i][j][k] - mean[k]) ** 2
            var = [v / total for v in var]
            
            # 存储用于同步
            self._mean_buffer = mean
            self._var_buffer = var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        output = [[[0.0] * c for _ in range(w)] for _ in range(h)]
        
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    normalized = (x[i][j][k] - mean[k]) / math.sqrt(var[k] + self.eps)
                    
                    if self.affine:
                        output[i][j][k] = self.weight[k] * normalized + self.bias[k]
                    else:
                        output[i][j][k] = normalized
        
        return output
    
    def sync_statistics(self, other_means: List[List[float]], other_vars: List[List[float]]):
        """
        同步统计量（概念实现）
        
        other_means: 其他GPU的均值列表
        other_vars: 其他GPU的方差列表
        """
        if not self.training:
            return
        
        # 合并所有GPU的统计量
        all_means = [self._mean_buffer] + other_means
        all_vars = [self._var_buffer] + other_vars
        
        num_gpus = len(all_means)
        c = len(all_means[0])
        
        # 平均
        synced_mean = [sum(m[k] for m in all_means) / num_gpus for k in range(c)]
        synced_var = [sum(v[k] for v in all_vars) / num_gpus for k in range(c)]
        
        # 更新running统计量
        if self.track_running_stats:
            for k in range(c):
                self.running_mean[k] = (1 - self.momentum) * self.running_mean[k] + self.momentum * synced_mean[k]
                self.running_var[k] = (1 - self.momentum) * self.running_var[k] + self.momentum * synced_var[k]
        
        self._mean_buffer = synced_mean
        self._var_buffer = synced_var
    
    def __call__(self, x):
        return self.forward(x)


class FRN:
    """
    Feature Response Normalization
    
    一种不需要batch统计量的归一化方法
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        learnable_eps: bool = False
    ):
        self.num_features = num_features
        self.eps = eps
        self.learnable_eps = learnable_eps
        
        self.weight = [1.0] * num_features
        self.bias = [0.0] * num_features
        
        if learnable_eps:
            self.eps_param = math.log(eps)
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        h = len(x)
        w = len(x[0])
        c = len(x[0][0])
        
        eps = math.exp(self.eps_param) if self.learnable_eps else self.eps
        
        output = [[[0.0] * c for _ in range(w)] for _ in range(h)]
        
        for k in range(c):
            # 计算RMS
            rms_sq = sum(x[i][j][k] ** 2 for i in range(h) for j in range(w)) / (h * w)
            rms = math.sqrt(rms_sq + eps)
            
            for i in range(h):
                for j in range(w):
                    output[i][j][k] = self.weight[k] * x[i][j][k] / rms + self.bias[k]
        
        return output
    
    def __call__(self, x):
        return self.forward(x)


class EvoNorm:
    """
    EvoNorm - 进化归一化
    
    结合归一化和激活函数
    """
    
    def __init__(
        self,
        num_features: int,
        groups: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
        version: str = 'B0'
    ):
        self.num_features = num_features
        self.groups = groups
        self.eps = eps
        self.affine = affine
        self.version = version
        
        if affine:
            self.weight = [1.0] * num_features
            self.bias = [0.0] * num_features
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        h = len(x)
        w = len(x[0])
        c = len(x[0][0])
        
        output = [[[0.0] * c for _ in range(w)] for _ in range(h)]
        
        channels_per_group = c // self.groups
        
        for g in range(self.groups):
            start_c = g * channels_per_group
            end_c = start_c + channels_per_group
            
            # 计算组内统计量
            total = h * w * channels_per_group
            mean = sum(x[i][j][k] for i in range(h) for j in range(w) for k in range(start_c, end_c)) / total
            var = sum((x[i][j][k] - mean) ** 2 for i in range(h) for j in range(w) for k in range(start_c, end_c)) / total
            
            std = math.sqrt(var + self.eps)
            
            for i in range(h):
                for j in range(w):
                    for k in range(start_c, end_c):
                        normalized = (x[i][j][k] - mean) / std
                        
                        if self.version == 'B0':
                            # EvoNorm-B0: 使用sigmoid门控
                            v = x[i][j][k] / std
                            sig = 1.0 / (1.0 + math.exp(-v))
                            result = normalized * sig
                        else:
                            result = normalized
                        
                        if self.affine:
                            output[i][j][k] = self.weight[k] * result + self.bias[k]
                        else:
                            output[i][j][k] = result
        
        return output
    
    def __call__(self, x):
        return self.forward(x)


# 工厂函数
def get_normalization(name: str, **kwargs):
    """根据名称获取归一化层"""
    normalizations = {
        'batch': BatchNorm,
        'layer': LayerNorm,
        'instance': InstanceNorm,
        'group': GroupNorm,
        'rms': RMSNorm,
        'sync_batch': SyncBatchNorm,
        'frn': FRN,
        'evo': EvoNorm
    }
    
    name_lower = name.lower()
    if name_lower not in normalizations:
        raise ValueError(f"Unknown normalization: {name}. Available: {list(normalizations.keys())}")
    
    return normalizations[name_lower](**kwargs)
