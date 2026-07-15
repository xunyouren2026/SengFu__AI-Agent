"""
激活函数模块 - 包含各种激活函数的真实实现
包括: ReLU, LeakyReLU, PReLU, ELU, SELU, GELU, Swish, Mish, 
      HardSwish, HardSigmoid, Softmax, LogSoftmax, etc.
"""

import math
from typing import Optional, List, Union, Callable
from abc import ABC, abstractmethod


class Activation(ABC):
    """激活函数基类"""
    
    @abstractmethod
    def forward(self, x: float) -> float:
        """前向传播"""
        pass
    
    @abstractmethod
    def backward(self, x: float) -> float:
        """计算导数"""
        pass
    
    def __call__(self, x: Union[float, List]) -> Union[float, List]:
        if isinstance(x, list):
            if isinstance(x[0], list):
                return [[self.forward(val) for val in row] for row in x]
            return [self.forward(val) for val in x]
        return self.forward(x)


class ReLU(Activation):
    """
    ReLU激活函数
    
    f(x) = max(0, x)
    f'(x) = 1 if x > 0 else 0
    """
    
    def forward(self, x: float) -> float:
        return max(0.0, x)
    
    def backward(self, x: float) -> float:
        return 1.0 if x > 0 else 0.0


class LeakyReLU(Activation):
    """
    Leaky ReLU激活函数
    
    f(x) = x if x > 0 else alpha * x
    f'(x) = 1 if x > 0 else alpha
    """
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: float) -> float:
        return x if x > 0 else self.alpha * x
    
    def backward(self, x: float) -> float:
        return 1.0 if x > 0 else self.alpha


class PReLU(Activation):
    """
    Parametric ReLU激活函数
    
    f(x) = x if x > 0 else alpha * x
    alpha是可学习参数
    """
    
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
    
    def forward(self, x: float) -> float:
        return x if x > 0 else self.alpha * x
    
    def backward(self, x: float) -> float:
        return 1.0 if x > 0 else self.alpha
    
    def backward_alpha(self, x: float) -> float:
        """对alpha的梯度"""
        return 0.0 if x > 0 else x


class ELU(Activation):
    """
    ELU激活函数
    
    f(x) = x if x > 0 else alpha * (exp(x) - 1)
    f'(x) = 1 if x > 0 else alpha * exp(x)
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x: float) -> float:
        if x > 0:
            return x
        return self.alpha * (math.exp(x) - 1)
    
    def backward(self, x: float) -> float:
        if x > 0:
            return 1.0
        return self.alpha * math.exp(x)


class SELU(Activation):
    """
    SELU激活函数 (Self-Normalizing)
    
    f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    
    其中 alpha ≈ 1.673263, scale ≈ 1.050701
    """
    
    def __init__(self):
        self.alpha = 1.67326324235437728481704
        self.scale = 1.05070098735548049341933
    
    def forward(self, x: float) -> float:
        if x > 0:
            return self.scale * x
        return self.scale * self.alpha * (math.exp(x) - 1)
    
    def backward(self, x: float) -> float:
        if x > 0:
            return self.scale
        return self.scale * self.alpha * math.exp(x)


class GELU(Activation):
    """
    GELU激活函数 (Gaussian Error Linear Unit)
    
    f(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    
    近似: f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    
    def __init__(self, approximate: bool = True):
        self.approximate = approximate
    
    def forward(self, x: float) -> float:
        if self.approximate:
            # 快速近似
            return 0.5 * x * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            ))
        else:
            # 精确计算
            return x * 0.5 * (1.0 + self._erf(x / math.sqrt(2.0)))
    
    def backward(self, x: float) -> float:
        if self.approximate:
            # 近似导数
            cdf = 0.5 * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            ))
            pdf = math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
            return cdf + x * pdf
        else:
            cdf = 0.5 * (1.0 + self._erf(x / math.sqrt(2.0)))
            pdf = math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
            return cdf + x * pdf
    
    @staticmethod
    def _erf(x: float) -> float:
        """误差函数的近似计算"""
        # 使用Horner方法的多项式近似
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return sign * y


class Swish(Activation):
    """
    Swish激活函数 (SiLU)
    
    f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    f'(x) = f(x) + sigmoid(x) * (1 - f(x))
    """
    
    def forward(self, x: float) -> float:
        return x / (1.0 + math.exp(-x)) if x >= 0 else x * math.exp(x) / (1.0 + math.exp(x))
    
    def backward(self, x: float) -> float:
        sig = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        swish = x * sig
        return swish + sig * (1.0 - sig)


class HardSwish(Activation):
    """
    Hard Swish激活函数 (MobileNetV3)
    
    f(x) = x * (relu6(x + 3) / 6)
    """
    
    def forward(self, x: float) -> float:
        if x <= -3:
            return 0.0
        elif x >= 3:
            return x
        else:
            return x * (x + 3) / 6.0
    
    def backward(self, x: float) -> float:
        if x <= -3:
            return 0.0
        elif x >= 3:
            return 1.0
        else:
            return (2 * x + 3) / 6.0


class Mish(Activation):
    """
    Mish激活函数
    
    f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    
    def forward(self, x: float) -> float:
        # 数值稳定的softplus
        if x > 20:
            softplus = x
        elif x < -20:
            softplus = math.exp(x)
        else:
            softplus = math.log(1.0 + math.exp(x))
        
        return x * math.tanh(softplus)
    
    def backward(self, x: float) -> float:
        # 数值稳定的softplus
        if x > 20:
            softplus = x
        elif x < -20:
            softplus = math.exp(x)
        else:
            softplus = math.log(1.0 + math.exp(x))
        
        tanh_softplus = math.tanh(softplus)
        sech_sq = 1.0 - tanh_softplus ** 2
        
        # sigmoid
        if x > 20:
            sig = 1.0
        elif x < -20:
            sig = 0.0
        else:
            sig = 1.0 / (1.0 + math.exp(-x))
        
        return tanh_softplus + x * sech_sq * sig


class Softplus(Activation):
    """
    Softplus激活函数
    
    f(x) = ln(1 + exp(x))
    f'(x) = sigmoid(x)
    """
    
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x: float) -> float:
        beta_x = self.beta * x
        if beta_x > self.threshold:
            return x
        return math.log(1.0 + math.exp(beta_x)) / self.beta
    
    def backward(self, x: float) -> float:
        beta_x = self.beta * x
        if beta_x > self.threshold:
            return 1.0
        return 1.0 / (1.0 + math.exp(-beta_x))


class Softsign(Activation):
    """
    Softsign激活函数
    
    f(x) = x / (1 + |x|)
    f'(x) = 1 / (1 + |x|)^2
    """
    
    def forward(self, x: float) -> float:
        return x / (1.0 + abs(x))
    
    def backward(self, x: float) -> float:
        denom = 1.0 + abs(x)
        return 1.0 / (denom * denom)


class Tanh(Activation):
    """
    Tanh激活函数
    
    f(x) = tanh(x)
    f'(x) = 1 - tanh(x)^2
    """
    
    def forward(self, x: float) -> float:
        return math.tanh(x)
    
    def backward(self, x: float) -> float:
        t = math.tanh(x)
        return 1.0 - t * t


class Sigmoid(Activation):
    """
    Sigmoid激活函数
    
    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    
    def forward(self, x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            return math.exp(x) / (1.0 + math.exp(x))
    
    def backward(self, x: float) -> float:
        sig = self.forward(x)
        return sig * (1.0 - sig)


class HardSigmoid(Activation):
    """
    Hard Sigmoid激活函数
    
    f(x) = max(0, min(1, 0.2 * x + 0.5))
    """
    
    def forward(self, x: float) -> float:
        return max(0.0, min(1.0, 0.2 * x + 0.5))
    
    def backward(self, x: float) -> float:
        if -2.5 < x < 2.5:
            return 0.2
        return 0.0


class HardTanh(Activation):
    """
    Hard Tanh激活函数
    
    f(x) = max(-1, min(1, x))
    """
    
    def forward(self, x: float) -> float:
        return max(-1.0, min(1.0, x))
    
    def backward(self, x: float) -> float:
        if -1.0 < x < 1.0:
            return 1.0
        return 0.0


class ReLU6(Activation):
    """
    ReLU6激活函数
    
    f(x) = min(max(0, x), 6)
    """
    
    def forward(self, x: float) -> float:
        return min(max(0.0, x), 6.0)
    
    def backward(self, x: float) -> float:
        if 0.0 < x < 6.0:
            return 1.0
        return 0.0


class Softmax:
    """
    Softmax函数
    
    softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
    """
    
    def __init__(self, dim: int = -1):
        self.dim = dim
    
    def forward(self, x: List[float]) -> List[float]:
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]
    
    def backward(self, x: List[float]) -> List[List[float]]:
        """计算Jacobian矩阵"""
        s = self.forward(x)
        n = len(s)
        jacobian = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian[i][j] = s[i] * (1.0 - s[i])
                else:
                    jacobian[i][j] = -s[i] * s[j]
        
        return jacobian
    
    def __call__(self, x: List[float]) -> List[float]:
        return self.forward(x)


class LogSoftmax:
    """
    Log Softmax函数
    
    log_softmax(x_i) = x_i - log(sum_j(exp(x_j)))
    """
    
    def __init__(self, dim: int = -1):
        self.dim = dim
    
    def forward(self, x: List[float]) -> List[float]:
        max_x = max(x)
        log_sum_exp = max_x + math.log(sum(math.exp(xi - max_x) for xi in x))
        return [xi - log_sum_exp for xi in x]
    
    def backward(self, x: List[float]) -> List[List[float]]:
        """计算Jacobian矩阵"""
        s = Softmax()(x)
        n = len(s)
        jacobian = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian[i][j] = 1.0 - s[j]
                else:
                    jacobian[i][j] = -s[j]
        
        return jacobian
    
    def __call__(self, x: List[float]) -> List[float]:
        return self.forward(x)


class Softmin:
    """
    Softmin函数
    
    softmin(x_i) = exp(-x_i) / sum_j(exp(-x_j))
    """
    
    def forward(self, x: List[float]) -> List[float]:
        neg_x = [-xi for xi in x]
        max_neg = max(neg_x)
        exp_neg = [math.exp(neg_xi - max_neg) for neg_xi in neg_x]
        sum_exp = sum(exp_neg)
        return [e / sum_exp for e in exp_neg]
    
    def __call__(self, x: List[float]) -> List[float]:
        return self.forward(x)


class TanhShrink(Activation):
    """
    TanhShrink激活函数
    
    f(x) = x - tanh(x)
    f'(x) = tanh(x)^2
    """
    
    def forward(self, x: float) -> float:
        return x - math.tanh(x)
    
    def backward(self, x: float) -> float:
        t = math.tanh(x)
        return t * t


class ThresholdReLU(Activation):
    """
    Threshold ReLU激活函数
    
    f(x) = x if x > threshold else 0
    """
    
    def __init__(self, threshold: float = 0.0, value: float = 0.0):
        self.threshold = threshold
        self.value = value
    
    def forward(self, x: float) -> float:
        return x if x > self.threshold else self.value
    
    def backward(self, x: float) -> float:
        return 1.0 if x > self.threshold else 0.0


class RReLU(Activation):
    """
    Randomized Leaky ReLU
    
    训练时alpha随机，推理时使用固定值
    """
    
    def __init__(self, lower: float = 0.125, upper: float = 0.333):
        self.lower = lower
        self.upper = upper
        self.training = True
        self._alpha = None
    
    def forward(self, x: float) -> float:
        if x > 0:
            return x
        
        if self.training:
            self._alpha = self.lower + random.random() * (self.upper - self.lower)
        else:
            self._alpha = (self.lower + self.upper) / 2.0
        
        return self._alpha * x
    
    def backward(self, x: float) -> float:
        if x > 0:
            return 1.0
        return self._alpha if self._alpha is not None else (self.lower + self.upper) / 2.0


class GLU:
    """
    Gated Linear Unit
    
    GLU(a, b) = a * sigmoid(b)
    其中输入被分成两半
    """
    
    def forward(self, x: List[float]) -> List[float]:
        n = len(x) // 2
        a = x[:n]
        b = x[n:]
        
        result = []
        for ai, bi in zip(a, b):
            sig = 1.0 / (1.0 + math.exp(-bi))
            result.append(ai * sig)
        
        return result
    
    def __call__(self, x: List[float]) -> List[float]:
        return self.forward(x)


class SwiGLU:
    """
    SwiGLU激活函数 (PaLM使用)
    
    SwiGLU(x) = Swish(xW) * (xV)
    """
    
    def forward(self, x: List[float]) -> List[float]:
        n = len(x) // 2
        a = x[:n]
        b = x[n:]
        
        swish = Swish()
        result = []
        for ai, bi in zip(a, b):
            result.append(swish.forward(ai) * bi)
        
        return result
    
    def __call__(self, x: List[float]) -> List[float]:
        return self.forward(x)


class GeGLU:
    """
    GeGLU激活函数
    
    GeGLU(x) = GELU(xW) * (xV)
    """
    
    def forward(self, x: List[float]) -> List[float]:
        n = len(x) // 2
        a = x[:n]
        b = x[n:]
        
        gelu = GELU()
        result = []
        for ai, bi in zip(a, b):
            result.append(gelu.forward(ai) * bi)
        
        return result
    
    def __call__(self, x: List[float]) -> List[float]:
        return self.forward(x)


# 工厂函数
def get_activation(name: str, **kwargs) -> Union[Activation, Callable]:
    """根据名称获取激活函数"""
    activations = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'prelu': PReLU,
        'elu': ELU,
        'selu': SELU,
        'gelu': GELU,
        'swish': Swish,
        'hard_swish': HardSwish,
        'mish': Mish,
        'softplus': Softplus,
        'softsign': Softsign,
        'tanh': Tanh,
        'sigmoid': Sigmoid,
        'hard_sigmoid': HardSigmoid,
        'hard_tanh': HardTanh,
        'relu6': ReLU6,
        'tanhshrink': TanhShrink,
        'threshold_relu': ThresholdReLU,
        'rrelu': RReLU,
        'softmax': Softmax,
        'log_softmax': LogSoftmax,
        'softmin': Softmin,
        'glu': GLU,
        'swiglu': SwiGLU,
        'geglu': GeGLU
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name_lower](**kwargs)


import random  # RReLU需要
