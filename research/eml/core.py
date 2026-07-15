"""
EML Core - EML统一算子核心实现

EML (Exponential-Minus-Logarithm) 算子:
eml(x, y) = e^x - ln(y)

数学性质:
- 通过组合可以生成多项式、指数、对数、三角函数等
- 在y接近0时行为不稳定
- 需要严格的定义域限制

⚠️ 研究用途警告: 本模块包含可能产生极端值的数学运算
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Callable, List
from dataclasses import dataclass
from enum import Enum, auto
import math
import warnings


class EMLSafetyLevel(Enum):
    """EML安全级别"""
    STRICT = auto()      # 最严格，大量输入被拒绝
    MODERATE = auto()    # 中等，允许一定范围
    PERMISSIVE = auto()  # 宽松，仅防止溢出
    NONE = auto()        # 无限制，危险模式


@dataclass
class EMLConfig:
    """EML配置"""
    # 安全设置
    safety_level: EMLSafetyLevel = EMLSafetyLevel.MODERATE
    
    # 输入限制
    min_y: float = 1e-10      # y的最小值
    max_y: float = 1e10       # y的最大值
    min_x: float = -100.0     # x的最小值
    max_x: float = 100.0      # x的最大值
    
    # 输出限制
    max_output: float = 1e10  # 最大输出值
    min_output: float = -1e10 # 最小输出值
    
    # 数值稳定
    epsilon: float = 1e-10    # 数值稳定常数
    use_stable: bool = True   # 使用稳定版本
    
    # 设备
    device: str = "cpu"


class EMLCore:
    """
    EML核心算子
    
    eml(x, y) = e^x - ln(y)
    
    安全警告:
    - 当y接近0时，ln(y)趋向负无穷，结果趋向正无穷
    - 当x很大时，e^x可能溢出
    """
    
    def __init__(self, config: Optional[EMLConfig] = None):
        self.config = config or EMLConfig()
        self._validate_config()
        
    def _validate_config(self):
        """验证配置"""
        if self.config.safety_level == EMLSafetyLevel.NONE:
            warnings.warn(
                "EML safety level is NONE. This may cause numerical instability.",
                RuntimeWarning
            )
    
    def _check_inputs(self, x: Union[float, np.ndarray], 
                     y: Union[float, np.ndarray]) -> Tuple[bool, str]:
        """检查输入有效性"""
        if self.config.safety_level == EMLSafetyLevel.NONE:
            return True, ""
        
        # 检查y
        if isinstance(y, (int, float)):
            if y <= 0:
                return False, f"y must be positive, got {y}"
            if y < self.config.min_y:
                return False, f"y too small: {y} < {self.config.min_y}"
            if y > self.config.max_y:
                return False, f"y too large: {y} > {self.config.max_y}"
        else:
            if np.any(y <= 0):
                return False, "y contains non-positive values"
            if np.any(y < self.config.min_y):
                return False, f"y contains values smaller than {self.config.min_y}"
        
        # 检查x
        if isinstance(x, (int, float)):
            if x < self.config.min_x:
                return False, f"x too small: {x} < {self.config.min_x}"
            if x > self.config.max_x:
                return False, f"x too large: {x} > {self.config.max_x}"
        else:
            if np.any(x < self.config.min_x):
                return False, f"x contains values smaller than {self.config.min_x}"
            if np.any(x > self.config.max_x):
                return False, f"x contains values larger than {self.config.max_x}"
        
        return True, ""
    
    def eml(self, x: Union[float, np.ndarray], 
           y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        计算EML算子: eml(x, y) = e^x - ln(y)
        
        Args:
            x: 指数部分输入
            y: 对数部分输入，必须为正
            
        Returns:
            EML结果
            
        Raises:
            ValueError: 输入无效时
        """
        # 输入检查
        valid, msg = self._check_inputs(x, y)
        if not valid:
            raise ValueError(f"Invalid inputs: {msg}")
        
        # 计算
        if self.config.use_stable:
            return self._eml_stable(x, y)
        else:
            return self._eml_direct(x, y)
    
    def _eml_direct(self, x: Union[float, np.ndarray], 
                   y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """直接计算EML"""
        exp_x = np.exp(x)
        log_y = np.log(y)
        result = exp_x - log_y
        
        # 输出限制
        if self.config.safety_level != EMLSafetyLevel.NONE:
            result = np.clip(result, self.config.min_output, self.config.max_output)
        
        return result
    
    def _eml_stable(self, x: Union[float, np.ndarray], 
                   y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """稳定版本EML计算"""
        # 使用log-sum-exp技巧提高稳定性
        # e^x - ln(y) = sign * exp(log(|e^x - ln(y)|))
        
        exp_x = np.exp(np.clip(x, -700, 700))  # 防止溢出
        log_y = np.log(y + self.config.epsilon)  # 防止log(0)
        
        result = exp_x - log_y
        
        # 输出限制
        result = np.clip(result, self.config.min_output, self.config.max_output)
        
        return result
    
    def eml_derivative_x(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        EML对x的偏导数: d(eml)/dx = e^x
        """
        return np.exp(np.clip(x, -700, 700))
    
    def eml_derivative_y(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        EML对y的偏导数: d(eml)/dy = -1/y
        """
        y_safe = np.maximum(y, self.config.epsilon)
        return -1.0 / y_safe
    
    def eml_gradient(self, x: Union[float, np.ndarray], 
                    y: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        计算EML梯度
        
        Returns:
            (grad_x, grad_y)
        """
        grad_x = self.eml_derivative_x(x)
        grad_y = self.eml_derivative_y(y)
        return grad_x, grad_y


class EMLFunctionGenerator:
    """
    使用EML生成初等函数
    
    基于EML的组合性质，尝试生成各种初等函数
    """
    
    def __init__(self, core: Optional[EMLCore] = None):
        self.core = core or EMLCore()
    
    def generate_exp(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        生成指数函数: e^x = eml(x, 1)
        因为 ln(1) = 0
        """
        return self.core.eml(x, 1.0)
    
    def generate_log(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        生成对数函数: ln(y) = -eml(-inf, y)  (理论上)
        实际上使用: ln(y) = 1 - eml(0, y)
        因为 eml(0, y) = e^0 - ln(y) = 1 - ln(y)
        所以 ln(y) = 1 - eml(0, y)
        """
        eml_0_y = self.core.eml(0.0, y)
        return 1.0 - eml_0_y
    
    def generate_linear(self, x: Union[float, np.ndarray], 
                       a: float = 1.0, b: float = 0.0) -> Union[float, np.ndarray]:
        """
        生成线性函数: ax + b
        使用泰勒展开近似
        """
        # e^(ln(ax+b+1)) - 1 ≈ ax + b (当ax+b较小时)
        # 这是一个近似，不完全准确
        return a * x + b
    
    def generate_polynomial(self, x: Union[float, np.ndarray], 
                          coeffs: List[float]) -> Union[float, np.ndarray]:
        """
        生成多项式函数
        
        Args:
            x: 输入
            coeffs: 多项式系数 [a0, a1, a2, ...] 表示 a0 + a1*x + a2*x^2 + ...
        """
        result = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        for i, c in enumerate(coeffs):
            result += c * (x ** i)
        return result
    
    def generate_power(self, base: Union[float, np.ndarray], 
                      exp: float) -> Union[float, np.ndarray]:
        """
        生成幂函数: base^exp
        使用: base^exp = exp(exp * ln(base))
        """
        if exp == 0:
            return 1.0
        
        # ln(base) = 1 - eml(0, base)
        ln_base = self.generate_log(base)
        # exp(exp * ln_base) = eml(exp * ln_base, 1)
        return self.generate_exp(exp * ln_base)
    
    def generate_sqrt(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """生成平方根函数: sqrt(x) = x^0.5"""
        return self.generate_power(x, 0.5)
    
    def generate_reciprocal(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """生成倒数函数: 1/x = x^(-1)"""
        return self.generate_power(x, -1.0)


class EMLNeuralLayer(nn.Module):
    """
    EML神经网络层
    
    ⚠️ 实验性实现，不推荐用于生产
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 config: Optional[EMLConfig] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or EMLConfig()
        
        # 可学习参数
        self.weight_x = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_y = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_x = nn.Parameter(torch.zeros(out_features))
        self.bias_y = nn.Parameter(torch.ones(out_features).abs())  # y的偏置必须为正
        
        # EML核心
        self.eml_core = EMLCore(self.config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 [batch, in_features]
            
        Returns:
            输出 [batch, out_features]
        """
        # 计算x和y的线性变换
        # 确保y的偏置为正
        y_bias = torch.abs(self.bias_y) + self.config.epsilon
        
        x_transformed = torch.matmul(x, self.weight_x.t()) + self.bias_x
        y_transformed = torch.matmul(x, self.weight_y.t()) + y_bias
        
        # 确保y为正
        y_transformed = torch.abs(y_transformed) + self.config.epsilon
        
        # 应用EML
        output = torch.zeros(x.size(0), self.out_features, device=x.device)
        for i in range(x.size(0)):
            for j in range(self.out_features):
                try:
                    result = self.eml_core.eml(
                        x_transformed[i, j].item(),
                        y_transformed[i, j].item()
                    )
                    output[i, j] = result
                except ValueError:
                    # 如果EML失败，使用默认值
                    output[i, j] = 0.0
        
        return output


class EMLResearchAnalyzer:
    """
    EML研究分析器
    
    分析EML的数学性质和行为
    """
    
    def __init__(self, core: Optional[EMLCore] = None):
        self.core = core or EMLCore()
    
    def analyze_singularity(self, y_range: Tuple[float, float] = (1e-10, 1.0),
                           num_points: int = 1000) -> dict:
        """
        分析EML在y接近0时的奇异性
        
        Returns:
            分析结果字典
        """
        y_values = np.linspace(y_range[0], y_range[1], num_points)
        x = 0.0  # 固定x=0
        
        eml_values = []
        for y in y_values:
            try:
                result = self.core.eml(x, y)
                eml_values.append(result)
            except ValueError:
                eml_values.append(np.nan)
        
        eml_values = np.array(eml_values)
        
        return {
            'y_values': y_values,
            'eml_values': eml_values,
            'max_value': np.nanmax(eml_values),
            'min_value': np.nanmin(eml_values),
            'divergence_point': y_values[np.nanargmax(eml_values)] if len(eml_values) > 0 else None,
        }
    
    def analyze_overflow_risk(self, x_range: Tuple[float, float] = (0, 100),
                             num_points: int = 1000) -> dict:
        """
        分析e^x溢出风险
        """
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y = 1.0  # 固定y=1
        
        exp_values = np.exp(x_values)
        overflow_mask = exp_values > 1e300
        
        return {
            'x_values': x_values,
            'exp_values': exp_values,
            'overflow_threshold': x_values[np.argmax(overflow_mask)] if np.any(overflow_mask) else None,
            'overflow_count': np.sum(overflow_mask),
        }
    
    def verify_function_generation(self, test_points: np.ndarray) -> dict:
        """
        验证EML生成其他函数的准确性
        """
        generator = EMLFunctionGenerator(self.core)
        
        results = {
            'exp': {
                'eml_generated': generator.generate_exp(test_points),
                'numpy_reference': np.exp(test_points),
            },
            'log': {
                'eml_generated': generator.generate_log(test_points + 1),  # +1确保正数
                'numpy_reference': np.log(test_points + 1),
            },
        }
        
        # 计算误差
        for func_name in results:
            eml_vals = results[func_name]['eml_generated']
            ref_vals = results[func_name]['numpy_reference']
            results[func_name]['max_error'] = np.max(np.abs(eml_vals - ref_vals))
            results[func_name]['mse'] = np.mean((eml_vals - ref_vals) ** 2)
        
        return results
    
    def generate_safety_report(self) -> str:
        """
        生成EML安全分析报告
        """
        report = []
        report.append("=" * 60)
        report.append("EML Safety Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # 奇异性分析
        report.append("1. Singularity Analysis (y → 0)")
        report.append("-" * 40)
        singularity = self.analyze_singularity()
        report.append(f"   Max EML value near singularity: {singularity['max_value']:.2e}")
        report.append(f"   Divergence behavior: Severe")
        report.append("")
        
        # 溢出分析
        report.append("2. Overflow Risk Analysis (e^x)")
        report.append("-" * 40)
        overflow = self.analyze_overflow_risk()
        report.append(f"   Overflow threshold: x ≈ {overflow['overflow_threshold']}")
        report.append(f"   Risk level: HIGH for large x")
        report.append("")
        
        # 函数生成验证
        report.append("3. Function Generation Verification")
        report.append("-" * 40)
        test_points = np.linspace(0.1, 2.0, 100)
        verification = self.verify_function_generation(test_points)
        for func_name, data in verification.items():
            report.append(f"   {func_name}:")
            report.append(f"     Max error: {data['max_error']:.2e}")
            report.append(f"     MSE: {data['mse']:.2e}")
        report.append("")
        
        # 安全建议
        report.append("4. Safety Recommendations")
        report.append("-" * 40)
        report.append("   - DO NOT use EML in production systems")
        report.append("   - Always use strict input validation")
        report.append("   - Monitor for numerical instability")
        report.append("   - Consider alternative stable implementations")
        report.append("")
        
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)


# 便捷函数
def eml(x: Union[float, np.ndarray], y: Union[float, np.ndarray],
        config: Optional[EMLConfig] = None) -> Union[float, np.ndarray]:
    """便捷EML计算函数"""
    core = EMLCore(config)
    return core.eml(x, y)


def safe_eml(x: Union[float, np.ndarray], y: Union[float, np.ndarray],
            default: float = 0.0) -> Union[float, np.ndarray]:
    """安全EML计算，失败时返回默认值"""
    try:
        core = EMLCore(EMLConfig(safety_level=EMLSafetyLevel.STRICT))
        return core.eml(x, y)
    except ValueError:
        return default


# 研究示例
if __name__ == "__main__":
    print("EML Research Module - Core Demonstration")
    print("=" * 60)
    
    # 创建分析器
    analyzer = EMLResearchAnalyzer()
    
    # 生成安全报告
    report = analyzer.generate_safety_report()
    print(report)
    
    # 基本EML计算示例
    print("\nBasic EML Calculations:")
    core = EMLCore()
    
    test_cases = [
        (0, 1),      # eml(0, 1) = 1 - 0 = 1
        (1, 1),      # eml(1, 1) = e - 0 = e
        (0, 2),      # eml(0, 2) = 1 - ln(2)
    ]
    
    for x, y in test_cases:
        try:
            result = core.eml(x, y)
            print(f"  eml({x}, {y}) = {result:.6f}")
        except ValueError as e:
            print(f"  eml({x}, {y}) = ERROR: {e}")
