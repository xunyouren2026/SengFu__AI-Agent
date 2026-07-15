"""
量化压缩 - INT8/FP16
"""
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import struct
import math


class QuantizationMethod(Enum):
    """量化方法"""
    INT8 = "int8"  # 8位整数量化
    INT4 = "int4"  # 4位整数量化
    FP16 = "fp16"  # 16位浮点
    DYNAMIC = "dynamic"  # 动态量化
    SYMMETRIC = "symmetric"  # 对称量化
    ASYMMETRIC = "asymmetric"  # 非对称量化


class QuantizationConfig:
    """量化配置"""
    
    def __init__(
        self,
        method: QuantizationMethod = QuantizationMethod.INT8,
        per_channel: bool = True,
        symmetric: bool = True
    ):
        self.method = method
        self.per_channel = per_channel
        self.symmetric = symmetric


class QuantizationRange:
    """量化范围"""
    
    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 0.0
    ):
        self.min_val = min_val
        self.max_val = max_val
    
    @property
    def scale(self) -> float:
        """计算缩放因子"""
        if self.max_val == self.min_val:
            return 1.0
        return (self.max_val - self.min_val) / 255.0  # INT8
    
    @property
    def zero_point(self) -> int:
        """计算零点"""
        if self.scale == 0:
            return 0
        return round(-self.min_val / self.scale)
    
    def to_dict(self) -> Dict[str, float]:
        return {'min': self.min_val, 'max': self.max_val}


class INT8Quantizer:
    """
    INT8量化器
    
    将浮点数量化为8位整数
    """
    
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        self._ranges: Dict[str, QuantizationRange] = {}
    
    def compute_range(
        self,
        values: List[float],
        name: Optional[str] = None
    ) -> QuantizationRange:
        """计算量化范围"""
        if not values:
            return QuantizationRange()
        
        min_val = min(values)
        max_val = max(values)
        
        if self.symmetric:
            # 对称量化：范围是 [-max_abs, max_abs]
            max_abs = max(abs(min_val), abs(max_val))
            qrange = QuantizationRange(-max_abs, max_abs)
        else:
            qrange = QuantizationRange(min_val, max_val)
        
        if name:
            self._ranges[name] = qrange
        
        return qrange
    
    def quantize(
        self,
        values: List[float],
        qrange: Optional[QuantizationRange] = None
    ) -> Tuple[List[int], QuantizationRange]:
        """
        量化
        
        Args:
            values: 浮点数列表
            qrange: 量化范围，None则自动计算
        
        Returns:
            (量化后的整数列表, 使用的量化范围)
        """
        if not values:
            return [], QuantizationRange()
        
        if qrange is None:
            qrange = self.compute_range(values)
        
        scale = qrange.scale
        if scale == 0:
            return [0] * len(values), qrange
        
        zero_point = qrange.zero_point
        
        quantized = []
        for v in values:
            q = round(v / scale + zero_point)
            # 裁剪到INT8范围
            q = max(-128, min(127, q))
            quantized.append(q)
        
        return quantized, qrange
    
    def dequantize(
        self,
        quantized: List[int],
        qrange: QuantizationRange
    ) -> List[float]:
        """反量化"""
        scale = qrange.scale
        zero_point = qrange.zero_point
        
        return [(q - zero_point) * scale for q in quantized]
    
    def quantize_params(
        self,
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, QuantizationRange]]:
        """
        量化参数字典
        
        Returns:
            (量化后的参数, 量化范围字典)
        """
        quantized_params: Dict[str, Any] = {}
        ranges: Dict[str, QuantizationRange] = {}
        
        for key, value in params.items():
            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                float_values = [float(v) for v in value]
                q_values, qrange = self.quantize(float_values)
                quantized_params[key] = q_values
                ranges[key] = qrange
            elif isinstance(value, (int, float)):
                # 标量保持不变或简单量化
                quantized_params[key] = value
            else:
                quantized_params[key] = value
        
        return quantized_params, ranges
    
    def dequantize_params(
        self,
        quantized_params: Dict[str, Any],
        ranges: Dict[str, QuantizationRange]
    ) -> Dict[str, Any]:
        """反量化参数字典"""
        params: Dict[str, Any] = {}
        
        for key, value in quantized_params.items():
            if key in ranges and isinstance(value, list):
                params[key] = self.dequantize(value, ranges[key])
            else:
                params[key] = value
        
        return params


class FP16Quantizer:
    """
    FP16量化器
    
    将FP32转换为FP16
    """
    
    def __init__(self):
        self._fp16_max = 65504.0
        self._fp16_min = -65504.0
    
    def quantize(self, value: float) -> float:
        """量化单个值到FP16范围"""
        # 简化实现：裁剪到FP16范围
        return max(self._fp16_min, min(self._fp16_max, value))
    
    def quantize_list(self, values: List[float]) -> List[float]:
        """量化列表"""
        return [self.quantize(v) for v in values]
    
    def quantize_params(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """量化参数字典"""
        quantized: Dict[str, Any] = {}
        
        for key, value in params.items():
            if isinstance(value, list):
                quantized[key] = self.quantize_list([
                    float(v) for v in value if isinstance(v, (int, float))
                ])
            elif isinstance(value, (int, float)):
                quantized[key] = self.quantize(float(value))
            else:
                quantized[key] = value
        
        return quantized
    
    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        return 2.0  # FP32 -> FP16 压缩比为2


class DynamicQuantizer:
    """
    动态量化器
    
    根据数据分布动态选择量化参数
    """
    
    def __init__(
        self,
        bits: int = 8,
        percentile: float = 99.9
    ):
        self.bits = bits
        self.percentile = percentile
        self._int8_quantizer = INT8Quantizer()
    
    def compute_percentile_range(
        self,
        values: List[float]
    ) -> QuantizationRange:
        """计算百分位范围"""
        if not values:
            return QuantizationRange()
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        lower_idx = int(n * (100 - self.percentile) / 100)
        upper_idx = int(n * self.percentile / 100)
        
        lower_idx = max(0, min(lower_idx, n - 1))
        upper_idx = max(0, min(upper_idx, n - 1))
        
        return QuantizationRange(
            sorted_values[lower_idx],
            sorted_values[upper_idx]
        )
    
    def quantize(
        self,
        values: List[float]
    ) -> Tuple[List[int], QuantizationRange]:
        """动态量化"""
        qrange = self.compute_percentile_range(values)
        return self._int8_quantizer.quantize(values, qrange)
    
    def dequantize(
        self,
        quantized: List[int],
        qrange: QuantizationRange
    ) -> List[float]:
        """反量化"""
        return self._int8_quantizer.dequantize(quantized, qrange)


class Quantizer:
    """
    统一量化接口
    
    支持多种量化方法
    """
    
    def __init__(
        self,
        method: QuantizationMethod = QuantizationMethod.INT8,
        symmetric: bool = True
    ):
        self.method = method
        self.symmetric = symmetric
        
        # 创建具体量化器
        self._quantizer = self._create_quantizer()
        
        # 统计信息
        self._total_quantized = 0
        self._total_original_bytes = 0
        self._total_quantized_bytes = 0
    
    def _create_quantizer(self) -> Any:
        """创建量化器实例"""
        if self.method == QuantizationMethod.INT8:
            return INT8Quantizer(self.symmetric)
        elif self.method == QuantizationMethod.FP16:
            return FP16Quantizer()
        elif self.method == QuantizationMethod.DYNAMIC:
            return DynamicQuantizer()
        else:
            return INT8Quantizer(self.symmetric)
    
    def quantize(
        self,
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        量化参数
        
        Args:
            params: 参数字典
        
        Returns:
            (量化后的参数, 量化元数据)
        """
        if self.method == QuantizationMethod.INT8:
            quantized, ranges = self._quantizer.quantize_params(params)
            metadata = {
                'ranges': {k: v.to_dict() for k, v in ranges.items()},
                'method': 'int8'
            }
        elif self.method == QuantizationMethod.FP16:
            quantized = self._quantizer.quantize_params(params)
            metadata = {'method': 'fp16'}
        else:
            quantized, ranges = self._quantizer.quantize_params(params)
            metadata = {
                'ranges': {k: v.to_dict() for k, v in ranges.items()},
                'method': 'dynamic'
            }
        
        # 更新统计
        self._total_quantized += 1
        self._total_original_bytes += self._estimate_size(params)
        self._total_quantized_bytes += self._estimate_size(quantized)
        
        return quantized, metadata
    
    def dequantize(
        self,
        quantized_params: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """反量化参数"""
        if metadata.get('method') == 'fp16':
            return quantized_params  # FP16不需要反量化
        
        if 'ranges' in metadata:
            ranges = {
                k: QuantizationRange(v['min'], v['max'])
                for k, v in metadata['ranges'].items()
            }
            return self._quantizer.dequantize_params(quantized_params, ranges)
        
        return quantized_params
    
    def _estimate_size(self, params: Dict[str, Any]) -> int:
        """估算参数大小"""
        size = 0
        for value in params.values():
            if isinstance(value, list):
                size += len(value) * 4  # 假设FP32
            elif isinstance(value, (int, float)):
                size += 4
        return size
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        if self._total_original_bytes == 0:
            ratio = 1.0
        else:
            ratio = self._total_original_bytes / self._total_quantized_bytes
        
        return {
            'method': self.method.value,
            'total_quantized': self._total_quantized,
            'total_original_bytes': self._total_original_bytes,
            'total_quantized_bytes': self._total_quantized_bytes,
            'compression_ratio': ratio
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._total_quantized = 0
        self._total_original_bytes = 0
        self._total_quantized_bytes = 0


class QuantizationAwareTraining:
    """
    量化感知训练
    
    在训练过程中模拟量化效果
    """
    
    def __init__(
        self,
        quantizer: Quantizer,
        start_epoch: int = 0,
        end_epoch: int = 10
    ):
        self.quantizer = quantizer
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self._current_epoch = 0
    
    def should_quantize(self) -> bool:
        """是否应该应用量化"""
        return self.start_epoch <= self._current_epoch <= self.end_epoch
    
    def quantize_forward(
        self,
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """前向传播时的量化"""
        if not self.should_quantize():
            return params, {}
        
        return self.quantizer.quantize(params)
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch"""
        self._current_epoch = epoch
    
    def advance_epoch(self) -> int:
        """推进epoch"""
        self._current_epoch += 1
        return self._current_epoch
