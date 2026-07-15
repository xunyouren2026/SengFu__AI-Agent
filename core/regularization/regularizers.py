"""
正则化模块 - 包含各种正则化技术的真实实现
包括: Dropout, DropConnect, DropBlock, SpatialDropout, Zoneout, 
      Weight Decay, Label Smoothing, Mixup, CutMix, Stochastic Depth, etc.
"""

import math
import random
from typing import Optional, Tuple, List, Union, Callable, Dict, Any
from abc import ABC, abstractmethod


class Tensor:
    """简化的张量类"""
    def __init__(self, data, requires_grad=False):
        self.data = data if isinstance(data, list) else [data]
        self.requires_grad = requires_grad
        self.grad = None
    
    @property
    def shape(self):
        if not isinstance(self.data, list):
            return ()
        if not isinstance(self.data[0], list):
            return (len(self.data),)
        return (len(self.data), len(self.data[0]))


class Regularizer(ABC):
    """正则化器基类"""
    
    @abstractmethod
    def forward(self, x: Tensor) -> float:
        """计算正则化损失"""
        pass
    
    def __call__(self, x: Tensor) -> float:
        return self.forward(x)


class Dropout:
    """
    标准Dropout
    
    训练时以概率p随机将神经元置零，推理时不变
    为了保持期望不变，训练时需要缩放: x / (1-p)
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.inplace = inplace
        self.training = True
        self.mask = None
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        if not self.training or self.p == 0:
            return x
        
        # 生成mask
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        scale = 1.0 / (1.0 - self.p)
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                # 2D
                self.mask = [[random.random() > self.p for _ in row] for row in data]
                result = [[val * scale if m else 0.0 for val, m in zip(row, mask_row)]
                         for row, mask_row in zip(data, self.mask)]
            else:
                # 1D
                self.mask = [random.random() > self.p for _ in data]
                result = [val * scale if m else 0.0 for val, m in zip(data, self.mask)]
        else:
            # 标量
            self.mask = random.random() > self.p
            result = data * scale if self.mask else 0.0
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class DropConnect:
    """
    DropConnect - 随机丢弃权重连接
    
    与Dropout不同，DropConnect随机将权重矩阵的元素置零
    常用于全连接层
    """
    
    def __init__(self, p: float = 0.5):
        if p < 0 or p > 1:
            raise ValueError(f"drop probability must be in [0, 1], got {p}")
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, weight: Union[Tensor, List]) -> Union[Tensor, List]:
        if not self.training or self.p == 0:
            return weight
        
        if isinstance(weight, Tensor):
            data = weight.data
        else:
            data = weight
        
        scale = 1.0 / (1.0 - self.p)
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                # 2D权重矩阵
                self.mask = [[random.random() > self.p for _ in row] for row in data]
                result = [[w * scale if m else 0.0 for w, m in zip(row, mask_row)]
                         for row, mask_row in zip(data, self.mask)]
            else:
                # 1D
                self.mask = [random.random() > self.p for _ in data]
                result = [w * scale if m else 0.0 for w, m in zip(data, self.mask)]
        else:
            self.mask = random.random() > self.p
            result = data * scale if self.mask else 0.0
        
        if isinstance(weight, Tensor):
            return Tensor(result, weight.requires_grad)
        return result
    
    def __call__(self, weight: Union[Tensor, List]) -> Union[Tensor, List]:
        return self.forward(weight)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class DropBlock:
    """
    DropBlock - 随机丢弃连续区域
    
    专门用于卷积网络，丢弃连续的特征区域而非单个像素
    比标准Dropout更有效
    """
    
    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0
    ):
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.gamma_scale = gamma_scale
        self.training = True
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        if not self.training or self.drop_prob == 0:
            return x
        
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        if not isinstance(data, list) or not isinstance(data[0], list):
            return x
        
        height = len(data)
        width = len(data[0])
        
        # 计算gamma (每个像素被丢弃的概率)
        gamma = self._compute_gamma(height, width)
        
        # 生成mask
        mask = self._generate_mask(height, width, gamma)
        
        # 应用mask
        if isinstance(data[0][0], list):
            # 多通道
            result = [[[val * mask[h][w] for val in data[h][w]] 
                      for w in range(width)] for h in range(height)]
        else:
            # 单通道
            result = [[data[h][w] * mask[h][w] 
                      for w in range(width)] for h in range(height)]
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def _compute_gamma(self, height: int, width: int) -> float:
        """计算gamma值"""
        return (self.drop_prob * height * width) / (
            (self.block_size ** 2) * 
            (height - self.block_size + 1) * 
            (width - self.block_size + 1)
        ) * self.gamma_scale
    
    def _generate_mask(self, height: int, width: int, gamma: float) -> List[List[float]]:
        """生成DropBlock mask"""
        # 创建基础mask
        mask = [[1.0 for _ in range(width)] for _ in range(height)]
        
        # 生成随机mask
        random_mask = [[random.random() < gamma for _ in range(width - self.block_size + 1)]
                       for _ in range(height - self.block_size + 1)]
        
        # 扩展mask到block大小
        for i in range(len(random_mask)):
            for j in range(len(random_mask[0])):
                if random_mask[i][j]:
                    # 将block区域置零
                    for bi in range(self.block_size):
                        for bj in range(self.block_size):
                            mask[i + bi][j + bj] = 0.0
        
        # 归一化
        total = sum(sum(row) for row in mask)
        if total > 0:
            scale = height * width / total
            mask = [[m * scale for m in row] for row in mask]
        
        return mask
    
    def __call__(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class SpatialDropout:
    """
    Spatial Dropout - 随机丢弃整个特征图
    
    随机丢弃整个通道，而不是单个像素
    适用于卷积网络
    """
    
    def __init__(self, p: float = 0.5):
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        if not self.training or self.p == 0:
            return x
        
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        # 假设输入是 [C, H, W] 或 [H, W, C]
        # 这里假设是 [H, W, C] 格式
        if not isinstance(data, list) or not isinstance(data[0], list):
            return x
        
        height = len(data)
        width = len(data[0])
        
        if isinstance(data[0][0], list):
            channels = len(data[0][0])
        else:
            # 不是多通道，回退到标准dropout
            return Dropout(self.p).forward(x)
        
        # 为每个通道生成mask
        self.mask = [random.random() > self.p for _ in range(channels)]
        scale = 1.0 / (1.0 - self.p)
        
        result = [[[val * scale if self.mask[c] else 0.0 
                   for c, val in enumerate(data[h][w])]
                  for w in range(width)] for h in range(height)]
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class Zoneout:
    """
    Zoneout - 用于RNN的正则化
    
    随机保持前一个状态而不是更新到新状态
    h_t = z * h_{t-1} + (1 - z) * h_t_new
    其中z是随机mask
    """
    
    def __init__(self, p: float = 0.1):
        if p < 0 or p > 1:
            raise ValueError(f"zoneout probability must be in [0, 1], got {p}")
        self.p = p
        self.training = True
    
    def forward(
        self,
        new_state: Union[Tensor, List],
        old_state: Union[Tensor, List]
    ) -> Union[Tensor, List]:
        """
        new_state: 当前计算的新状态
        old_state: 前一个状态
        """
        if not self.training or self.p == 0:
            return new_state
        
        if isinstance(new_state, Tensor):
            new_data = new_state.data
            old_data = old_state.data if isinstance(old_state, Tensor) else old_state
        else:
            new_data = new_state
            old_data = old_state
        
        # 生成随机mask
        if isinstance(new_data, list):
            if isinstance(new_data[0], list):
                mask = [[random.random() < self.p for _ in row] for row in new_data]
                result = [[m * old + (1 - m) * new for m, old, new in zip(mask_row, old_row, new_row)]
                         for mask_row, old_row, new_row in zip(mask, old_data, new_data)]
            else:
                mask = [random.random() < self.p for _ in new_data]
                result = [m * old + (1 - m) * new for m, old, new in zip(mask, old_data, new_data)]
        else:
            mask = random.random() < self.p
            result = mask * old_data + (1 - mask) * new_data
        
        if isinstance(new_state, Tensor):
            return Tensor(result, new_state.requires_grad)
        return result
    
    def __call__(self, new_state, old_state):
        return self.forward(new_state, old_state)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class StochasticDepth:
    """
    Stochastic Depth - 随机深度
    
    训练时随机跳过整个残差块
    推理时使用所有层
    
    p_l = 1 - l/L * (1 - p_L)
    其中l是当前层，L是总层数，p_L是最后一层的存活概率
    """
    
    def __init__(self, p: float = 0.5, mode: str = 'row'):
        """
        p: 最后一层的存活概率
        mode: 'row' 或 'batch'
        """
        if p < 0 or p > 1:
            raise ValueError(f"survival probability must be in [0, 1], got {p}")
        self.p = p
        self.mode = mode
        self.training = True
    
    def forward(
        self,
        x: Union[Tensor, List],
        residual: Union[Tensor, List],
        layer_idx: int,
        total_layers: int
    ) -> Union[Tensor, List]:
        """
        x: 输入
        residual: 残差分支的输出
        layer_idx: 当前层索引 (从1开始)
        total_layers: 总层数
        """
        if not self.training:
            # 推理时使用所有层
            return self._add(x, residual)
        
        # 计算当前层的存活概率
        survival_prob = 1 - layer_idx / total_layers * (1 - self.p)
        
        if random.random() < survival_prob:
            return self._add(x, residual)
        else:
            return x
    
    def _add(self, a, b):
        if isinstance(a, Tensor):
            a_data = a.data
            b_data = b.data if isinstance(b, Tensor) else b
        else:
            a_data = a
            b_data = b
        
        if isinstance(a_data, list):
            if isinstance(a_data[0], list):
                return [[x + y for x, y in zip(row_a, row_b)] 
                       for row_a, row_b in zip(a_data, b_data)]
            else:
                return [x + y for x, y in zip(a_data, b_data)]
        else:
            return a_data + b_data
    
    def __call__(self, x, residual, layer_idx, total_layers):
        return self.forward(x, residual, layer_idx, total_layers)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class DropPath:
    """
    Drop Path (Stochastic Depth) - 随机丢弃路径
    
    用于Vision Transformer等架构
    以概率drop_prob将整个路径置零
    """
    
    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob
        self.training = True
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        if not self.training or self.drop_prob == 0:
            return x
        
        if random.random() < self.drop_prob:
            # 将整个路径置零
            if isinstance(x, Tensor):
                data = x.data
            else:
                data = x
            
            if isinstance(data, list):
                if isinstance(data[0], list):
                    result = [[0.0 for _ in row] for row in data]
                else:
                    result = [0.0 for _ in data]
            else:
                result = 0.0
            
            if isinstance(x, Tensor):
                return Tensor(result, x.requires_grad)
            return result
        
        # 缩放以保持期望
        scale = 1.0 / (1.0 - self.drop_prob)
        return self._scale(x, scale)
    
    def _scale(self, x, factor):
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                result = [[val * factor for val in row] for row in data]
            else:
                result = [val * factor for val in data]
        else:
            result = data * factor
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class WeightDecay(Regularizer):
    """
    权重衰减 (L2正则化)
    
    L = lambda * sum(w^2)
    """
    
    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_
    
    def forward(self, weight: Tensor) -> float:
        if isinstance(weight, Tensor):
            data = weight.data
        else:
            data = weight
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                l2_sum = sum(w ** 2 for row in data for w in row)
            else:
                l2_sum = sum(w ** 2 for w in data)
        else:
            l2_sum = data ** 2
        
        return self.lambda_ * l2_sum


class L1Regularization(Regularizer):
    """
    L1正则化
    
    L = lambda * sum(|w|)
    """
    
    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_
    
    def forward(self, weight: Tensor) -> float:
        if isinstance(weight, Tensor):
            data = weight.data
        else:
            data = weight
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                l1_sum = sum(abs(w) for row in data for w in row)
            else:
                l1_sum = sum(abs(w) for w in data)
        else:
            l1_sum = abs(data)
        
        return self.lambda_ * l1_sum


class ElasticNetRegularization(Regularizer):
    """
    Elastic Net正则化 (L1 + L2)
    
    L = lambda1 * sum(|w|) + lambda2 * sum(w^2)
    """
    
    def __init__(self, lambda1: float = 0.01, lambda2: float = 0.01):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def forward(self, weight: Tensor) -> float:
        if isinstance(weight, Tensor):
            data = weight.data
        else:
            data = weight
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                l1_sum = sum(abs(w) for row in data for w in row)
                l2_sum = sum(w ** 2 for row in data for w in row)
            else:
                l1_sum = sum(abs(w) for w in data)
                l2_sum = sum(w ** 2 for w in data)
        else:
            l1_sum = abs(data)
            l2_sum = data ** 2
        
        return self.lambda1 * l1_sum + self.lambda2 * l2_sum


class MaxNormRegularization(Regularizer):
    """
    最大范数约束
    
    如果 ||w|| > max_norm, 则 w = w * max_norm / ||w||
    """
    
    def __init__(self, max_norm: float = 3.0):
        self.max_norm = max_norm
    
    def forward(self, weight: Tensor) -> float:
        # 最大范数约束不产生损失，而是直接修改权重
        return 0.0
    
    def constrain(self, weight: Union[Tensor, List]) -> Union[Tensor, List]:
        """应用约束"""
        if isinstance(weight, Tensor):
            data = weight.data
        else:
            data = weight
        
        # 计算L2范数
        if isinstance(data, list):
            if isinstance(data[0], list):
                norm = math.sqrt(sum(w ** 2 for row in data for w in row))
            else:
                norm = math.sqrt(sum(w ** 2 for w in data))
        else:
            norm = abs(data)
        
        if norm > self.max_norm:
            scale = self.max_norm / norm
            if isinstance(data, list):
                if isinstance(data[0], list):
                    result = [[w * scale for w in row] for row in data]
                else:
                    result = [w * scale for w in data]
            else:
                result = data * scale
            
            if isinstance(weight, Tensor):
                return Tensor(result, weight.requires_grad)
            return result
        
        return weight


class OrthogonalRegularization(Regularizer):
    """
    正交正则化
    
    鼓励权重矩阵正交化
    L = lambda * ||W^T W - I||^2
    """
    
    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_
    
    def forward(self, weight: Tensor) -> float:
        if isinstance(weight, Tensor):
            data = weight.data
        else:
            data = weight
        
        if not isinstance(data, list) or not isinstance(data[0], list):
            return 0.0
        
        # 计算 W^T W
        n = len(data[0])  # 列数
        m = len(data)     # 行数
        
        # W^T W
        wtw = [[sum(data[k][i] * data[k][j] for k in range(m))
                for j in range(n)] for i in range(n)]
        
        # ||W^T W - I||^2
        loss = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    loss += (wtw[i][j] - 1) ** 2
                else:
                    loss += wtw[i][j] ** 2
        
        return self.lambda_ * loss


class SpectralNormalization:
    """
    谱归一化
    
    通过除以权重矩阵的谱范数（最大奇异值）来约束权重
    常用于GAN的判别器
    """
    
    def __init__(self, n_power_iterations: int = 1, eps: float = 1e-12):
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.u = None
        self.v = None
    
    def forward(self, weight: Union[Tensor, List]) -> Union[Tensor, List]:
        if isinstance(weight, Tensor):
            data = weight.data
        else:
            data = weight
        
        if not isinstance(data, list) or not isinstance(data[0], list):
            return weight
        
        m = len(data)     # 行数
        n = len(data[0])  # 列数
        
        # 初始化u和v
        if self.u is None:
            self.u = [1.0 / math.sqrt(m)] * m
        if self.v is None:
            self.v = [1.0 / math.sqrt(n)] * n
        
        # 幂迭代估计最大奇异值
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            wtu = [sum(data[i][j] * self.u[i] for i in range(m)) for j in range(n)]
            wtu_norm = math.sqrt(sum(x ** 2 for x in wtu)) + self.eps
            self.v = [x / wtu_norm for x in wtu]
            
            # u = W v / ||W v||
            wv = [sum(data[i][j] * self.v[j] for j in range(n)) for i in range(m)]
            wv_norm = math.sqrt(sum(x ** 2 for x in wv)) + self.eps
            self.u = [x / wv_norm for x in wv]
        
        # 计算谱范数 (最大奇异值)
        sigma = sum(sum(data[i][j] * self.v[j] for j in range(n)) * self.u[i] for i in range(m))
        
        # 归一化权重
        result = [[w / (sigma + self.eps) for w in row] for row in data]
        
        if isinstance(weight, Tensor):
            return Tensor(result, weight.requires_grad)
        return result
    
    def __call__(self, weight):
        return self.forward(weight)


class VariationalDropout:
    """
    变分Dropout - 用于变分推断
    
    使用高斯分布的变分推断
    可以学习dropout率
    """
    
    def __init__(self, alpha: float = 1.0, threshold: float = 3.0):
        self.alpha = alpha
        self.threshold = threshold
        self.training = True
        self.log_alpha = None
    
    def forward(self, x: Union[Tensor, List], log_alpha: Optional[float] = None) -> Union[Tensor, List]:
        if log_alpha is not None:
            self.log_alpha = log_alpha
        else:
            self.log_alpha = math.log(self.alpha)
        
        if not self.training:
            return x
        
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        # 计算标准差
        sigma = math.sqrt(math.exp(self.log_alpha))
        
        # 添加噪声
        if isinstance(data, list):
            if isinstance(data[0], list):
                result = [[val + random.gauss(0, sigma * abs(val)) for val in row] for row in data]
            else:
                result = [val + random.gauss(0, sigma * abs(val)) for val in data]
        else:
            result = data + random.gauss(0, sigma * abs(data))
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x, log_alpha=None):
        return self.forward(x, log_alpha)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def get_kl_divergence(self) -> float:
        """计算KL散度用于正则化"""
        if self.log_alpha is None:
            return 0.0
        
        # KL散度的近似
        k1, k2, k3 = 0.63576, 1.8732, 1.48675
        log_alpha = self.log_alpha
        
        kl = -k1 * math.tanh(k2 * log_alpha) + 0.5 * math.log(1 + math.exp(-log_alpha)) + k3
        
        return kl


class GaussianDropout:
    """
    高斯Dropout
    
    乘以高斯噪声而不是二值mask
    x * (1 + N(0, sigma^2))
    """
    
    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma
        self.training = True
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        if not self.training or self.sigma == 0:
            return x
        
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                result = [[val * (1 + random.gauss(0, self.sigma)) for val in row] for row in data]
            else:
                result = [val * (1 + random.gauss(0, self.sigma)) for val in data]
        else:
            result = data * (1 + random.gauss(0, self.sigma))
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class AlphaDropout:
    """
    Alpha Dropout - 用于SELU激活函数
    
    保持自归一化属性的dropout变体
    """
    
    def __init__(self, p: float = 0.5):
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.training = True
        
        # SELU参数
        self.alpha = 1.67326324235437728481704
        self.scale = 1.05070098735548049341933
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        if not self.training or self.p == 0:
            return x
        
        # 计算alpha dropout参数
        q = 1.0 - self.p
        a = self.alpha * self.scale
        mean = a * (self.p - q)
        std = math.sqrt(self.p * (a ** 2 + q) - (mean ** 2))
        
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        if isinstance(data, list):
            if isinstance(data[0], list):
                result = []
                for row in data:
                    new_row = []
                    for val in row:
                        if random.random() < self.p:
                            new_row.append(mean + std * random.gauss(0, 1))
                        else:
                            new_row.append(val)
                    result.append(new_row)
            else:
                result = [mean + std * random.gauss(0, 1) if random.random() < self.p else val
                         for val in data]
        else:
            if random.random() < self.p:
                result = mean + std * random.gauss(0, 1)
            else:
                result = data
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class LockedDropout:
    """
    锁定Dropout - 在整个序列上使用相同的mask
    
    用于RNN/LSTM，确保时间步之间的一致性
    """
    
    def __init__(self, p: float = 0.5):
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        """
        x: [seq_len, batch_size, hidden_size]
        """
        if not self.training or self.p == 0:
            return x
        
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        if not isinstance(data, list):
            return x
        
        # 假设是 [seq_len, batch_size, hidden_size]
        seq_len = len(data)
        if seq_len == 0:
            return x
        
        batch_size = len(data[0]) if isinstance(data[0], list) else 1
        if batch_size > 0 and isinstance(data[0][0], list):
            hidden_size = len(data[0][0])
        else:
            hidden_size = 1
        
        # 生成一个共享的mask
        scale = 1.0 / (1.0 - self.p)
        self.mask = [[random.random() > self.p for _ in range(hidden_size)]
                     for _ in range(batch_size)]
        
        # 应用mask
        result = []
        for t in range(seq_len):
            timestep = []
            for b in range(batch_size):
                if isinstance(data[t][b], list):
                    hidden = [val * scale if self.mask[b][h] else 0.0
                             for h, val in enumerate(data[t][b])]
                else:
                    hidden = data[t][b] * scale if self.mask[b][0] else 0.0
                timestep.append(hidden)
            result.append(timestep)
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class WordDropout:
    """
    词Dropout - 用于NLP
    
    随机将整个词嵌入置零
    """
    
    def __init__(self, p: float = 0.1):
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.training = True
    
    def forward(self, x: Union[Tensor, List]) -> Union[Tensor, List]:
        """
        x: [seq_len, embedding_size] 或 [batch_size, seq_len, embedding_size]
        """
        if not self.training or self.p == 0:
            return x
        
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
        
        if not isinstance(data, list):
            return x
        
        # 检测维度
        if isinstance(data[0][0], list):
            # [batch, seq, embed]
            result = []
            for batch in data:
                new_batch = []
                for word in batch:
                    if random.random() < self.p:
                        new_batch.append([0.0] * len(word))
                    else:
                        new_batch.append(word)
                result.append(new_batch)
        else:
            # [seq, embed]
            result = []
            for word in data:
                if random.random() < self.p:
                    result.append([0.0] * len(word))
                else:
                    result.append(word)
        
        if isinstance(x, Tensor):
            return Tensor(result, x.requires_grad)
        return result
    
    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class EmbeddingDropout:
    """
    嵌入层Dropout
    
    随机将整个嵌入行置零
    """
    
    def __init__(self, p: float = 0.1):
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p
        self.training = True
    
    def forward(
        self,
        embedding: Union[Tensor, List],
        indices: List[int]
    ) -> Union[Tensor, List]:
        """
        embedding: 嵌入矩阵 [vocab_size, embed_size]
        indices: 当前使用的词索引
        """
        if not self.training or self.p == 0:
            return embedding
        
        if isinstance(embedding, Tensor):
            data = embedding.data
        else:
            data = embedding
        
        if not isinstance(data, list):
            return embedding
        
        # 为每个索引生成mask
        mask = {idx: random.random() > self.p for idx in indices}
        scale = 1.0 / (1.0 - self.p)
        
        # 应用mask
        result = []
        for i, row in enumerate(data):
            if i in mask:
                if mask[i]:
                    result.append([val * scale for val in row])
                else:
                    result.append([0.0] * len(row))
            else:
                result.append(row)
        
        if isinstance(embedding, Tensor):
            return Tensor(result, embedding.requires_grad)
        return result
    
    def __call__(self, embedding, indices):
        return self.forward(embedding, indices)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# 组合正则化器
class CombinedRegularizer(Regularizer):
    """组合多个正则化器"""
    
    def __init__(self, regularizers: List[Tuple[Regularizer, float]]):
        """
        regularizers: [(regularizer, weight), ...]
        """
        self.regularizers = regularizers
    
    def forward(self, x: Tensor) -> float:
        total = 0.0
        for reg, weight in self.regularizers:
            total += weight * reg(x)
        return total


# 工厂函数
def get_dropout(name: str, **kwargs):
    """根据名称获取Dropout变体"""
    dropouts = {
        'standard': Dropout,
        'dropconnect': DropConnect,
        'dropblock': DropBlock,
        'spatial': SpatialDropout,
        'zoneout': Zoneout,
        'stochastic_depth': StochasticDepth,
        'drop_path': DropPath,
        'variational': VariationalDropout,
        'gaussian': GaussianDropout,
        'alpha': AlphaDropout,
        'locked': LockedDropout,
        'word': WordDropout,
        'embedding': EmbeddingDropout
    }
    
    name_lower = name.lower()
    if name_lower not in dropouts:
        raise ValueError(f"Unknown dropout: {name}. Available: {list(dropouts.keys())}")
    
    return dropouts[name_lower](**kwargs)


def get_regularizer(name: str, **kwargs) -> Regularizer:
    """根据名称获取正则化器"""
    regularizers = {
        'weight_decay': WeightDecay,
        'l1': L1Regularization,
        'elastic_net': ElasticNetRegularization,
        'max_norm': MaxNormRegularization,
        'orthogonal': OrthogonalRegularization
    }
    
    name_lower = name.lower()
    if name_lower not in regularizers:
        raise ValueError(f"Unknown regularizer: {name}. Available: {list(regularizers.keys())}")
    
    return regularizers[name_lower](**kwargs)
