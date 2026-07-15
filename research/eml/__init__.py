"""
EML Research Module - EML统一数学基础研究模块

EML (Exponential-Minus-Logarithm) 统一算子研究
eml(x,y) = e^x - ln(y)

⚠️ 警告: 本模块仅供研究用途，不应用于生产环境
⚠️ Warning: This module is for research purposes only, not for production use

研究目标:
1. 探索EML算子的数学性质
2. 验证EML是否能生成所有初等函数
3. 评估EML在神经网络中的适用性
4. 分析EML的安全性和潜在风险

安全注意事项:
- EML算子在y接近0时会产生极端值
- 需要严格的输入验证和边界检查
- 不建议在关键系统中使用
"""

__version__ = "0.1.0-research"
__status__ = "experimental"

from .core import EMLCore, EMLConfig, EMLSafetyLevel
from .genetic import EMLGeneticProgramming
from .experiments import EMLExperiments
from .sandbox import EMLSandbox

__all__ = [
    "EMLCore",
    "EMLConfig", 
    "EMLSafetyLevel",
    "EMLGeneticProgramming",
    "EMLExperiments",
    "EMLSandbox",
]

# 研究模块警告
import warnings
warnings.warn(
    "EML模块仅供研究用途，不应用于生产环境。 "
    "EML operator may produce extreme values and should not be used in production.",
    category=UserWarning,
    stacklevel=2
)
