"""
SuperFastMath CUDA - GPU加速数学运算

CUDA kernel实现，提供GPU上的极致性能数学运算。
支持批量exp/ln运算，适用于大规模并行计算场景。
"""

import numpy as np
from typing import Union, Optional, Tuple
import math


class GPUMathBackend:
    """
    GPU数学运算后端管理
    
    自动检测GPU可用性，管理CUDA上下文
    """
    
    def __init__(self):
        self.cuda_available = self._check_cuda()
        self.device_count = 0
        self.current_device = 0
        
        if self.cuda_available:
            try:
                import cupy as cp
                self.cp = cp
                self.device_count = cp.cuda.runtime.getDeviceCount()
            except ImportError:
                self.cuda_available = False
                
    def _check_cuda(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            return True
        except:
            return False
    
    def get_device_info(self) -> dict:
        """获取GPU设备信息"""
        if not self.cuda_available:
            return {'available': False}
        
        try:
            device = self.cp.cuda.Device(self.current_device)
            return {
                'available': True,
                'device_count': self.device_count,
                'current_device': self.current_device,
                'device_name': device.attributes.get('name', 'Unknown'),
                'total_memory': device.mem_info[1] if hasattr(device, 'mem_info') else 0,
            }
        except:
            return {'available': False}
    
    def to_gpu(self, arr: np.ndarray) -> 'cp.ndarray':
        """将数组转移到GPU"""
        if not self.cuda_available:
            raise RuntimeError("CUDA not available")
        return self.cp.asarray(arr)
    
    def to_cpu(self, arr: 'cp.ndarray') -> np.ndarray:
        """将数组转移回CPU"""
        if not self.cuda_available:
            raise RuntimeError("CUDA not available")
        return self.cp.asnumpy(arr)
    
    def sync(self):
        """同步CUDA流"""
        if self.cuda_available:
            self.cp.cuda.Device(self.current_device).synchronize()


class CUDAExp:
    """
    CUDA加速的指数运算
    
    使用CuPy实现GPU并行exp计算
    """
    
    def __init__(self, backend: Optional[GPUMathBackend] = None):
        self.backend = backend or GPUMathBackend()
        self._init_constants()
        
    def _init_constants(self):
        """初始化常量"""
        self.ln2 = 0.6931471805599453
        self.ln2_inv = 1.4426950408889634
        
    def exp(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """
        CUDA加速的exp运算
        
        Args:
            x: 输入数组(CPU或GPU)
            
        Returns:
            GPU上的exp(x)结果
        """
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available, use CPU implementation")
        
        # 转移到GPU
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
        
        # 使用CuPy的exp (已优化)
        result = self.backend.cp.exp(x_gpu)
        
        return result
    
    def exp2(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """计算2^x"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        return self.backend.cp.exp2(x_gpu)
    
    def exp10(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """计算10^x"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        return self.backend.cp.power(10.0, x_gpu)
    
    def expm1(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """计算exp(x) - 1"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        return self.backend.cp.expm1(x_gpu)


class CUDALog:
    """
    CUDA加速的对数运算
    """
    
    def __init__(self, backend: Optional[GPUMathBackend] = None):
        self.backend = backend or GPUMathBackend()
        
    def log(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """CUDA加速的ln运算"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        return self.backend.cp.log(x_gpu)
    
    def log2(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """计算log2(x)"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        return self.backend.cp.log2(x_gpu)
    
    def log10(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """计算log10(x)"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        return self.backend.cp.log10(x_gpu)
    
    def log1p(self, x: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
        """计算log(1+x)"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        return self.backend.cp.log1p(x_gpu)


class CUDAMath:
    """
    CUDA数学运算统一接口
    
    提供完整的GPU加速数学运算功能
    """
    
    def __init__(self):
        self.backend = GPUMathBackend()
        self.cuda_exp = CUDAExp(self.backend)
        self.cuda_log = CUDALog(self.backend)
        
    def is_available(self) -> bool:
        """检查CUDA是否可用"""
        return self.backend.cuda_available
    
    def exp(self, x: Union[np.ndarray, 'cp.ndarray'], 
            return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU exp运算"""
        result = self.cuda_exp.exp(x)
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def log(self, x: Union[np.ndarray, 'cp.ndarray'],
            return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU log运算"""
        result = self.cuda_log.log(x)
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def pow(self, base: Union[np.ndarray, 'cp.ndarray'], 
            exp: Union[float, np.ndarray, 'cp.ndarray'],
            return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU pow运算"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(base, np.ndarray):
            base_gpu = self.backend.to_gpu(base)
        else:
            base_gpu = base
            
        if isinstance(exp, np.ndarray):
            exp_gpu = self.backend.to_gpu(exp)
        else:
            exp_gpu = exp
            
        result = self.backend.cp.power(base_gpu, exp_gpu)
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def sqrt(self, x: Union[np.ndarray, 'cp.ndarray'],
             return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU sqrt运算"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        result = self.backend.cp.sqrt(x_gpu)
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def sigmoid(self, x: Union[np.ndarray, 'cp.ndarray'],
                return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU sigmoid"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        result = 1.0 / (1.0 + self.backend.cp.exp(-x_gpu))
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def tanh(self, x: Union[np.ndarray, 'cp.ndarray'],
             return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU tanh"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        result = self.backend.cp.tanh(x_gpu)
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def softmax(self, x: Union[np.ndarray, 'cp.ndarray'], 
                axis: int = -1,
                return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU softmax"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        exp_x = self.backend.cp.exp(x_gpu - self.backend.cp.max(x_gpu, axis=axis, keepdims=True))
        result = exp_x / self.backend.cp.sum(exp_x, axis=axis, keepdims=True)
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def layer_norm(self, x: Union[np.ndarray, 'cp.ndarray'],
                   eps: float = 1e-5,
                   return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU层归一化"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        mean = self.backend.cp.mean(x_gpu, axis=-1, keepdims=True)
        var = self.backend.cp.var(x_gpu, axis=-1, keepdims=True)
        result = (x_gpu - mean) / self.backend.cp.sqrt(var + eps)
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def rms_norm(self, x: Union[np.ndarray, 'cp.ndarray'],
                 eps: float = 1e-5,
                 return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU RMS归一化"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        rms = self.backend.cp.sqrt(self.backend.cp.mean(x_gpu * x_gpu, axis=-1, keepdims=True) + eps)
        result = x_gpu / rms
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def gelu(self, x: Union[np.ndarray, 'cp.ndarray'],
             return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU GELU激活"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        sqrt_2_over_pi = 0.7978845608028654
        result = 0.5 * x_gpu * (1.0 + self.backend.cp.tanh(
            sqrt_2_over_pi * (x_gpu + 0.044715 * x_gpu * x_gpu * x_gpu)))
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def silu(self, x: Union[np.ndarray, 'cp.ndarray'],
             return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU SiLU激活"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(x, np.ndarray):
            x_gpu = self.backend.to_gpu(x)
        else:
            x_gpu = x
            
        result = x_gpu * self.sigmoid(x_gpu)
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def matmul(self, a: Union[np.ndarray, 'cp.ndarray'], 
               b: Union[np.ndarray, 'cp.ndarray'],
               return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU矩阵乘法"""
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if isinstance(a, np.ndarray):
            a_gpu = self.backend.to_gpu(a)
        else:
            a_gpu = a
            
        if isinstance(b, np.ndarray):
            b_gpu = self.backend.to_gpu(b)
        else:
            b_gpu = b
            
        result = self.backend.cp.matmul(a_gpu, b_gpu)
        
        if return_cpu:
            return self.backend.to_cpu(result)
        return result
    
    def batch_matmul(self, a: Union[np.ndarray, 'cp.ndarray'],
                     b: Union[np.ndarray, 'cp.ndarray'],
                     return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """GPU批量矩阵乘法"""
        return self.matmul(a, b, return_cpu)
    
    def attention(self, q: Union[np.ndarray, 'cp.ndarray'],
                  k: Union[np.ndarray, 'cp.ndarray'],
                  v: Union[np.ndarray, 'cp.ndarray'],
                  mask: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
                  scale: Optional[float] = None,
                  return_cpu: bool = False) -> Union['cp.ndarray', np.ndarray]:
        """
        GPU自注意力计算
        
        Args:
            q: Query [batch, heads, seq, dim]
            k: Key [batch, heads, seq, dim]
            v: Value [batch, heads, seq, dim]
            mask: 可选掩码
            scale: 缩放因子
            
        Returns:
            注意力输出
        """
        if not self.backend.cuda_available:
            raise RuntimeError("CUDA not available")
        
        # 转移到GPU
        if isinstance(q, np.ndarray):
            q_gpu = self.backend.to_gpu(q)
            k_gpu = self.backend.to_gpu(k)
            v_gpu = self.backend.to_gpu(v)
        else:
            q_gpu, k_gpu, v_gpu = q, k, v
        
        # 计算注意力分数
        scores = self.backend.cp.matmul(q_gpu, k_gpu.transpose(-2, -1))
        
        if scale is None:
            dim = q_gpu.shape[-1]
            scale = 1.0 / math.sqrt(dim)
        
        scores = scores * scale
        
        # 应用掩码
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_gpu = self.backend.to_gpu(mask)
            else:
                mask_gpu = mask
            scores = scores + mask_gpu
        
        # Softmax
        exp_scores = self.backend.cp.exp(scores - self.backend.cp.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / self.backend.cp.sum(exp_scores, axis=-1, keepdims=True)
        
        # 应用到Value
        output = self.backend.cp.matmul(weights, v_gpu)
        
        if return_cpu:
            return self.backend.to_cpu(output)
        return output
    
    def synchronize(self):
        """同步GPU"""
        self.backend.sync()


# 便捷函数
def cuda_exp(x: np.ndarray, return_cpu: bool = True) -> np.ndarray:
    """CUDA exp"""
    cuda_math = CUDAMath()
    result = cuda_math.exp(x, return_cpu=return_cpu)
    return result if return_cpu else cuda_math.backend.to_cpu(result)

def cuda_log(x: np.ndarray, return_cpu: bool = True) -> np.ndarray:
    """CUDA log"""
    cuda_math = CUDAMath()
    result = cuda_math.log(x, return_cpu=return_cpu)
    return result if return_cpu else cuda_math.backend.to_cpu(result)

def cuda_pow(base: np.ndarray, exp: Union[float, np.ndarray], 
             return_cpu: bool = True) -> np.ndarray:
    """CUDA pow"""
    cuda_math = CUDAMath()
    result = cuda_math.pow(base, exp, return_cpu=return_cpu)
    return result if return_cpu else cuda_math.backend.to_cpu(result)

def cuda_sigmoid(x: np.ndarray, return_cpu: bool = True) -> np.ndarray:
    """CUDA sigmoid"""
    cuda_math = CUDAMath()
    result = cuda_math.sigmoid(x, return_cpu=return_cpu)
    return result if return_cpu else cuda_math.backend.to_cpu(result)

def cuda_softmax(x: np.ndarray, axis: int = -1, return_cpu: bool = True) -> np.ndarray:
    """CUDA softmax"""
    cuda_math = CUDAMath()
    result = cuda_math.softmax(x, axis=axis, return_cpu=return_cpu)
    return result if return_cpu else cuda_math.backend.to_cpu(result)
