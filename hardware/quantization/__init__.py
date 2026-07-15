"""
AGI Unified Framework - Quantization Module

模块路径: hardware/quantization/__init__.py

提供多种量化算法的统一接口，包括:
- AWQ (Activation-aware Weight Quantization)
- GPTQ (General-purpose Post-Training Quantization)
- INT8/INT4 对称/非对称量化
- FP8 浮点量化
- BitsAndBytes 8-bit/4-bit量化
- ONNX 量化
"""

from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import warnings

# 版本信息
__version__ = "1.0.0"
__author__ = "AGI Framework Team"


class QuantizationType(Enum):
    """量化类型枚举"""
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    FP16 = "fp16"
    AWQ = "awq"
    GPTQ = "gptq"
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT = "bnb_4bit"
    ONNX = "onnx"


class QuantizationScheme(Enum):
    """量化方案枚举"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    DYNAMIC = "dynamic"
    STATIC = "static"


@dataclass
class QuantizationConfig:
    """
    量化配置类

    属性:
        quant_type: 量化类型
        scheme: 量化方案
        bits: 量化位数
        group_size: 分组大小 (用于AWQ/GPTQ)
        symmetric: 是否对称量化
        per_channel: 是否按通道量化
        calibration_samples: 校准样本数
        calibration_method: 校准方法 (minmax, entropy, percentile)
        percentile: 百分位数 (用于percentile校准)
        round_method: 舍入方法 (round, floor, ceil)
        observer_method: 观察方法
        enable_bias_correction: 是否启用偏置校正
        device: 计算设备
        dtype: 数据类型
    """
    quant_type: QuantizationType = QuantizationType.INT8
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    bits: int = 8
    group_size: int = 128
    symmetric: bool = True
    per_channel: bool = True
    calibration_samples: int = 128
    calibration_method: str = "minmax"
    percentile: float = 99.99
    round_method: str = "round"
    observer_method: str = "minmax"
    enable_bias_correction: bool = True
    device: str = "cuda"
    dtype: str = "float16"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['quant_type'] = self.quant_type.value
        result['scheme'] = self.scheme.value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """从字典创建配置"""
        if 'quant_type' in config_dict:
            config_dict['quant_type'] = QuantizationType(config_dict['quant_type'])
        if 'scheme' in config_dict:
            config_dict['scheme'] = QuantizationScheme(config_dict['scheme'])
        return cls(**config_dict)

    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QuantizationConfig":
        """从文件加载配置"""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class CalibrationDatasetConfig:
    """
    校准数据集配置

    属性:
        dataset_name: 数据集名称
        num_samples: 样本数量
        sequence_length: 序列长度
        batch_size: 批大小
        shuffle: 是否打乱
        seed: 随机种子
    """
    dataset_name: str = "wikitext"
    num_samples: int = 128
    sequence_length: int = 2048
    batch_size: int = 1
    shuffle: bool = True
    seed: int = 42


class QuantizationRegistry:
    """量化器注册表"""

    _quantizers: Dict[QuantizationType, Type] = {}

    @classmethod
    def register(cls, quant_type: QuantizationType, quantizer_class: Type):
        """注册量化器"""
        cls._quantizers[quant_type] = quantizer_class

    @classmethod
    def get_quantizer(cls, quant_type: QuantizationType) -> Optional[Type]:
        """获取量化器类"""
        return cls._quantizers.get(quant_type)

    @classmethod
    def list_quantizers(cls) -> List[QuantizationType]:
        """列出所有可用的量化器"""
        return list(cls._quantizers.keys())

    @classmethod
    def create_quantizer(cls, quant_type: QuantizationType, config: Optional[QuantizationConfig] = None):
        """创建量化器实例"""
        quantizer_class = cls.get_quantizer(quant_type)
        if quantizer_class is None:
            raise ValueError(f"Quantizer {quant_type} not registered")
        return quantizer_class(config or QuantizationConfig(quant_type=quant_type))


def get_quantization_config(
    quant_type: Union[str, QuantizationType],
    **kwargs
) -> QuantizationConfig:
    """
    获取量化配置的便捷函数

    Args:
        quant_type: 量化类型
        **kwargs: 其他配置参数

    Returns:
        QuantizationConfig实例
    """
    if isinstance(quant_type, str):
        quant_type = QuantizationType(quant_type)

    # 预设配置
    presets = {
        QuantizationType.INT8: {
            'bits': 8,
            'scheme': QuantizationScheme.SYMMETRIC,
            'per_channel': True
        },
        QuantizationType.INT4: {
            'bits': 4,
            'scheme': QuantizationScheme.SYMMETRIC,
            'group_size': 128
        },
        QuantizationType.FP8: {
            'bits': 8,
            'scheme': QuantizationScheme.DYNAMIC
        },
        QuantizationType.AWQ: {
            'bits': 4,
            'group_size': 128,
            'scheme': QuantizationScheme.SYMMETRIC
        },
        QuantizationType.GPTQ: {
            'bits': 4,
            'group_size': 128,
            'scheme': QuantizationScheme.SYMMETRIC
        },
        QuantizationType.BNB_8BIT: {
            'bits': 8,
            'scheme': QuantizationScheme.DYNAMIC
        },
        QuantizationType.BNB_4BIT: {
            'bits': 4,
            'scheme': QuantizationScheme.DYNAMIC
        },
    }

    preset = presets.get(quant_type, {})
    preset.update(kwargs)
    preset['quant_type'] = quant_type

    return QuantizationConfig(**preset)


def quantize_model(
    model: Any,
    quant_type: Union[str, QuantizationType],
    calibration_data: Optional[Any] = None,
    config: Optional[QuantizationConfig] = None,
    **kwargs
) -> Any:
    """
    模型量化便捷函数

    Args:
        model: 待量化模型
        quant_type: 量化类型
        calibration_data: 校准数据
        config: 量化配置
        **kwargs: 额外配置参数

    Returns:
        量化后的模型
    """
    if config is None:
        config = get_quantization_config(quant_type, **kwargs)
    else:
        if isinstance(quant_type, str):
            quant_type = QuantizationType(quant_type)
        config.quant_type = quant_type

    quantizer = QuantizationRegistry.create_quantizer(config.quant_type, config)
    return quantizer.quantize(model, calibration_data)


# 导出主要组件
__all__ = [
    # 枚举
    'QuantizationType',
    'QuantizationScheme',

    # 配置类
    'QuantizationConfig',
    'CalibrationDatasetConfig',

    # 注册表
    'QuantizationRegistry',

    # 便捷函数
    'get_quantization_config',
    'quantize_model',

    # 版本信息
    '__version__',
]


# 延迟导入量化器类，避免循环导入
def _register_quantizers():
    """注册所有量化器"""
    try:
        from .int8_quantizer import Int8Quantizer
        QuantizationRegistry.register(QuantizationType.INT8, Int8Quantizer)
    except ImportError as e:
        warnings.warn(f"Failed to import Int8Quantizer: {e}")

    try:
        from .int4_quantizer import Int4Quantizer
        QuantizationRegistry.register(QuantizationType.INT4, Int4Quantizer)
    except ImportError as e:
        warnings.warn(f"Failed to import Int4Quantizer: {e}")

    try:
        from .fp8_quantizer import FP8Quantizer
        QuantizationRegistry.register(QuantizationType.FP8, FP8Quantizer)
    except ImportError as e:
        warnings.warn(f"Failed to import FP8Quantizer: {e}")

    try:
        from .awq import AWQQuantizer
        QuantizationRegistry.register(QuantizationType.AWQ, AWQQuantizer)
    except ImportError as e:
        warnings.warn(f"Failed to import AWQQuantizer: {e}")

    try:
        from .gptq import GPTQQuantizer
        QuantizationRegistry.register(QuantizationType.GPTQ, GPTQQuantizer)
    except ImportError as e:
        warnings.warn(f"Failed to import GPTQQuantizer: {e}")

    try:
        from .bitsandbytes import BitsAndBytesQuantizer
        QuantizationRegistry.register(QuantizationType.BNB_8BIT, BitsAndBytesQuantizer)
        QuantizationRegistry.register(QuantizationType.BNB_4BIT, BitsAndBytesQuantizer)
    except ImportError as e:
        warnings.warn(f"Failed to import BitsAndBytesQuantizer: {e}")

    try:
        from .onnx_quantizer import ONNXQuantizer
        QuantizationRegistry.register(QuantizationType.ONNX, ONNXQuantizer)
    except ImportError as e:
        warnings.warn(f"Failed to import ONNXQuantizer: {e}")


# 模块加载时注册量化器
_register_quantizers()
