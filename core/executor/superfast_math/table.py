"""
SuperFastMath Table - 查表法加速

预计算查找表实现，提供O(1)复杂度的数学运算。
适用于需要极致速度且可接受一定精度的场景。
"""

import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
import math


@dataclass
class TableConfig:
    """查找表配置"""
    table_size: int = 65536  # 表大小
    input_min: float = -10.0  # 最小输入值
    input_max: float = 10.0   # 最大输入值
    interpolation: str = "linear"  # 插值方法: linear, cubic, lanczos
    use_cache: bool = True    # 是否缓存结果


class Interpolation:
    """插值方法"""
    
    @staticmethod
    def linear(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
        """线性插值"""
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    
    @staticmethod
    def cubic(x: float, x0: float, x1: float, 
              y0: float, y1: float, 
              m0: float, m1: float) -> float:
        """三次Hermite插值"""
        t = (x - x0) / (x1 - x0)
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        
        return h00*y0 + h10*m0*(x1-x0) + h01*y1 + h11*m1*(x1-x0)
    
    @staticmethod
    def lanczos(x: float, x0: float, x1: float,
                y_values: np.ndarray, idx: int, a: int = 3) -> float:
        """Lanczos插值"""
        t = (x - x0) / (x1 - x0)
        result = 0.0
        
        for i in range(-a + 1, a):
            idx_i = idx + i
            if 0 <= idx_i < len(y_values):
                # Lanczos核
                if i == 0:
                    L = 1.0
                else:
                    pi_t = math.pi * (t - i)
                    L = a * math.sin(pi_t) * math.sin(pi_t / a) / (pi_t * pi_t)
                result += y_values[idx_i] * L
        
        return result


class LookupTable:
    """
    通用查找表
    
    预计算任意函数的查找表，支持多种插值方法
    """
    
    def __init__(self, func: Callable[[float], float], 
                 config: Optional[TableConfig] = None):
        self.func = func
        self.config = config or TableConfig()
        self.table = None
        self._build_table()
        
    def _build_table(self):
        """构建查找表"""
        n = self.config.table_size
        self.table = np.zeros(n, dtype=np.float64)
        
        # 在输入范围内均匀采样
        self.step = (self.config.input_max - self.config.input_min) / (n - 1)
        
        for i in range(n):
            x = self.config.input_min + i * self.step
            self.table[i] = self.func(x)
        
        # 预计算导数（用于三次插值）
        self.derivatives = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            self.derivatives[i] = (self.table[i + 1] - self.table[i - 1]) / (2 * self.step)
        self.derivatives[0] = (self.table[1] - self.table[0]) / self.step
        self.derivatives[-1] = (self.table[-1] - self.table[-2]) / self.step
        
    def lookup(self, x: float) -> float:
        """
        查表获取函数值
        
        Args:
            x: 输入值
            
        Returns:
            插值后的函数值
        """
        # 范围检查
        if x < self.config.input_min:
            return self.table[0]
        if x > self.config.input_max:
            return self.table[-1]
        
        # 计算索引
        idx = int((x - self.config.input_min) / self.step)
        idx = max(0, min(idx, len(self.table) - 2))
        
        # 获取相邻点
        x0 = self.config.input_min + idx * self.step
        x1 = x0 + self.step
        y0 = self.table[idx]
        y1 = self.table[idx + 1]
        
        # 插值
        if self.config.interpolation == "linear":
            return Interpolation.linear(x, x0, x1, y0, y1)
        elif self.config.interpolation == "cubic":
            m0 = self.derivatives[idx]
            m1 = self.derivatives[idx + 1]
            return Interpolation.cubic(x, x0, x1, y0, y1, m0, m1)
        elif self.config.interpolation == "lanczos":
            return Interpolation.lanczos(x, x0, x1, self.table, idx)
        else:
            return Interpolation.linear(x, x0, x1, y0, y1)
    
    def lookup_batch(self, x: np.ndarray) -> np.ndarray:
        """批量查表"""
        return np.array([self.lookup(xi) for xi in x])
    
    def get_max_error(self, n_test: int = 10000) -> float:
        """估计最大误差"""
        test_points = np.linspace(
            self.config.input_min, 
            self.config.input_max, 
            n_test
        )
        
        max_error = 0.0
        for x in test_points:
            approx = self.lookup(x)
            exact = self.func(x)
            error = abs(approx - exact)
            max_error = max(max_error, error)
        
        return max_error
    
    def get_rmse(self, n_test: int = 10000) -> float:
        """估计RMSE"""
        test_points = np.linspace(
            self.config.input_min,
            self.config.input_max,
            n_test
        )
        
        errors = []
        for x in test_points:
            approx = self.lookup(x)
            exact = self.func(x)
            errors.append((approx - exact) ** 2)
        
        return math.sqrt(np.mean(errors))


class TableGenerator:
    """
    查找表生成器
    
    为常用数学函数生成优化的查找表
    """
    
    @staticmethod
    def exp_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成exp查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-10.0,
            input_max=10.0,
            interpolation="cubic"
        )
        return LookupTable(math.exp, cfg)
    
    @staticmethod
    def log_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成log查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=0.001,
            input_max=100.0,
            interpolation="cubic"
        )
        return LookupTable(math.log, cfg)
    
    @staticmethod
    def sigmoid_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成sigmoid查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-10.0,
            input_max=10.0,
            interpolation="linear"
        )
        return LookupTable(lambda x: 1.0 / (1.0 + math.exp(-x)), cfg)
    
    @staticmethod
    def tanh_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成tanh查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-5.0,
            input_max=5.0,
            interpolation="linear"
        )
        return LookupTable(math.tanh, cfg)
    
    @staticmethod
    def sin_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成sin查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=0.0,
            input_max=2 * math.pi,
            interpolation="cubic"
        )
        return LookupTable(math.sin, cfg)
    
    @staticmethod
    def cos_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成cos查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=0.0,
            input_max=2 * math.pi,
            interpolation="cubic"
        )
        return LookupTable(math.cos, cfg)
    
    @staticmethod
    def atan_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成atan查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-10.0,
            input_max=10.0,
            interpolation="cubic"
        )
        return LookupTable(math.atan, cfg)
    
    @staticmethod
    def sqrt_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成sqrt查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=0.0,
            input_max=1000.0,
            interpolation="cubic"
        )
        return LookupTable(math.sqrt, cfg)
    
    @staticmethod
    def rsqrt_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成rsqrt查找表"""
        cfg = config or TableConfig(
            table_size=65536,
            input_min=0.001,
            input_max=1000.0,
            interpolation="cubic"
        )
        return LookupTable(lambda x: 1.0 / math.sqrt(x), cfg)
    
    @staticmethod
    def gelu_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成GELU查找表"""
        def gelu(x):
            sqrt_2_over_pi = 0.7978845608028654
            return 0.5 * x * (1.0 + math.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
        
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-5.0,
            input_max=5.0,
            interpolation="cubic"
        )
        return LookupTable(gelu, cfg)
    
    @staticmethod
    def silu_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成SiLU查找表"""
        def silu(x):
            return x / (1.0 + math.exp(-x))
        
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-5.0,
            input_max=5.0,
            interpolation="linear"
        )
        return LookupTable(silu, cfg)
    
    @staticmethod
    def mish_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成Mish查找表"""
        def mish(x):
            return x * math.tanh(math.log(1.0 + math.exp(x)))
        
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-5.0,
            input_max=5.0,
            interpolation="linear"
        )
        return LookupTable(mish, cfg)
    
    @staticmethod
    def swish_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成Swish查找表 (SiLU别名)"""
        return TableGenerator.silu_table(config)
    
    @staticmethod
    def elu_table(alpha: float = 1.0, config: Optional[TableConfig] = None) -> LookupTable:
        """生成ELU查找表"""
        def elu(x):
            return x if x > 0 else alpha * (math.exp(x) - 1.0)
        
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-5.0,
            input_max=5.0,
            interpolation="linear"
        )
        return LookupTable(elu, cfg)
    
    @staticmethod
    def selu_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成SELU查找表"""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        def selu(x):
            return scale * (x if x > 0 else alpha * (math.exp(x) - 1.0))
        
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-5.0,
            input_max=5.0,
            interpolation="linear"
        )
        return LookupTable(selu, cfg)
    
    @staticmethod
    def softplus_table(config: Optional[TableConfig] = None) -> LookupTable:
        """生成Softplus查找表"""
        def softplus(x):
            return math.log(1.0 + math.exp(x))
        
        cfg = config or TableConfig(
            table_size=65536,
            input_min=-10.0,
            input_max=10.0,
            interpolation="linear"
        )
        return LookupTable(softplus, cfg)
    
    @staticmethod
    def generate_all_tables(output_dir: str = "tables"):
        """生成所有常用查找表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        tables = {
            'exp': TableGenerator.exp_table(),
            'log': TableGenerator.log_table(),
            'sigmoid': TableGenerator.sigmoid_table(),
            'tanh': TableGenerator.tanh_table(),
            'sin': TableGenerator.sin_table(),
            'cos': TableGenerator.cos_table(),
            'atan': TableGenerator.atan_table(),
            'sqrt': TableGenerator.sqrt_table(),
            'rsqrt': TableGenerator.rsqrt_table(),
            'gelu': TableGenerator.gelu_table(),
            'silu': TableGenerator.silu_table(),
            'mish': TableGenerator.mish_table(),
            'selu': TableGenerator.selu_table(),
            'softplus': TableGenerator.softplus_table(),
        }
        
        # 保存表到文件
        for name, table in tables.items():
            filepath = os.path.join(output_dir, f"{name}_table.npy")
            np.save(filepath, table.table)
            print(f"Saved {name} table to {filepath}")
            print(f"  Max error: {table.get_max_error():.2e}")
            print(f"  RMSE: {table.get_rmse():.2e}")
        
        return tables


class FastTableMath:
    """
    基于查找表的快速数学运算
    
    预加载所有常用函数的查找表，提供O(1)访问
    """
    
    def __init__(self, table_dir: Optional[str] = None):
        self.tables = {}
        
        if table_dir:
            self._load_tables(table_dir)
        else:
            self._init_default_tables()
    
    def _init_default_tables(self):
        """初始化默认查找表"""
        self.tables['exp'] = TableGenerator.exp_table()
        self.tables['log'] = TableGenerator.log_table()
        self.tables['sigmoid'] = TableGenerator.sigmoid_table()
        self.tables['tanh'] = TableGenerator.tanh_table()
        self.tables['sin'] = TableGenerator.sin_table()
        self.tables['cos'] = TableGenerator.cos_table()
        self.tables['sqrt'] = TableGenerator.sqrt_table()
        self.tables['rsqrt'] = TableGenerator.rsqrt_table()
        self.tables['gelu'] = TableGenerator.gelu_table()
        self.tables['silu'] = TableGenerator.silu_table()
    
    def _load_tables(self, table_dir: str):
        """从文件加载查找表"""
        import os
        
        table_files = {
            'exp': 'exp_table.npy',
            'log': 'log_table.npy',
            'sigmoid': 'sigmoid_table.npy',
            'tanh': 'tanh_table.npy',
            'sin': 'sin_table.npy',
            'cos': 'cos_table.npy',
            'sqrt': 'sqrt_table.npy',
            'rsqrt': 'rsqrt_table.npy',
            'gelu': 'gelu_table.npy',
            'silu': 'silu_table.npy',
        }
        
        for name, filename in table_files.items():
            filepath = os.path.join(table_dir, filename)
            if os.path.exists(filepath):
                # 加载表数据
                table_data = np.load(filepath)
                # 创建配置
                config = TableConfig(table_size=len(table_data))
                # 创建查找表对象
                self.tables[name] = LookupTable(lambda x: x, config)
                self.tables[name].table = table_data
    
    def exp(self, x: float) -> float:
        """快速exp"""
        return self.tables['exp'].lookup(x)
    
    def log(self, x: float) -> float:
        """快速log"""
        return self.tables['log'].lookup(x)
    
    def sigmoid(self, x: float) -> float:
        """快速sigmoid"""
        return self.tables['sigmoid'].lookup(x)
    
    def tanh(self, x: float) -> float:
        """快速tanh"""
        return self.tables['tanh'].lookup(x)
    
    def sin(self, x: float) -> float:
        """快速sin"""
        # 归一化到[0, 2pi]
        x = x % (2 * math.pi)
        return self.tables['sin'].lookup(x)
    
    def cos(self, x: float) -> float:
        """快速cos"""
        x = x % (2 * math.pi)
        return self.tables['cos'].lookup(x)
    
    def sqrt(self, x: float) -> float:
        """快速sqrt"""
        return self.tables['sqrt'].lookup(x)
    
    def rsqrt(self, x: float) -> float:
        """快速rsqrt"""
        return self.tables['rsqrt'].lookup(x)
    
    def gelu(self, x: float) -> float:
        """快速GELU"""
        return self.tables['gelu'].lookup(x)
    
    def silu(self, x: float) -> float:
        """快速SiLU"""
        return self.tables['silu'].lookup(x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """快速softmax"""
        exp_x = np.array([self.exp(xi) for xi in x])
        return exp_x / np.sum(exp_x)
    
    def layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """快速层归一化"""
        mean = np.mean(x)
        var = np.var(x)
        return (x - mean) / self.sqrt(var + eps)
    
    def rms_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """快速RMS归一化"""
        rms = self.sqrt(np.mean(x * x) + eps)
        return x / rms


# 便捷函数
_table_math = None

def get_fast_table_math(table_dir: Optional[str] = None) -> FastTableMath:
    """获取全局FastTableMath实例"""
    global _table_math
    if _table_math is None:
        _table_math = FastTableMath(table_dir)
    return _table_math
