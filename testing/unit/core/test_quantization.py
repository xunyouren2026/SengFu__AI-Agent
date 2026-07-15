"""
TestQuantization - 核心算法单元测试：量化模块

模块路径: testing/unit/core/test_quantization.py
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


@dataclass
class QuantizationConfig:
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    scale_factor: Optional[float] = None
    zero_point: Optional[int] = None


class MockQuantizer:
    """模拟量化器"""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.calibration_stats: Dict[str, Any] = {}

    def calibrate(self, data: np.ndarray) -> Dict[str, float]:
        self.calibration_stats = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "abs_max": float(np.max(np.abs(data))),
        }
        qmin = 0 if not self.config.symmetric else -(2 ** (self.config.bits - 1))
        qmax = 2 ** (self.config.bits - 1) - 1 if self.config.symmetric else 2 ** self.config.bits - 1
        self.calibration_stats["qmin"] = qmin
        self.calibration_stats["qmax"] = qmax
        abs_max = self.calibration_stats["abs_max"]
        if abs_max > 0:
            self.config.scale_factor = abs_max / (2 ** (self.config.bits - 1))
        else:
            self.config.scale_factor = 1.0
        return self.calibration_stats

    def quantize(self, data: np.ndarray) -> np.ndarray:
        if self.config.scale_factor is None:
            self.calibrate(data)
        if self.config.symmetric:
            qmin = -(2 ** (self.config.bits - 1))
            qmax = 2 ** (self.config.bits - 1) - 1
            quantized = np.round(data / self.config.scale_factor).astype(np.float32)
            quantized = np.clip(quantized, qmin, qmax)
        else:
            qmin = 0
            qmax = 2 ** self.config.bits - 1
            quantized = np.round(data / self.config.scale_factor + qmin).astype(np.float32)
            quantized = np.clip(quantized, qmin, qmax)
        return quantized

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        if self.config.symmetric:
            return quantized * self.config.scale_factor
        else:
            return (quantized - 0) * self.config.scale_factor

    def compute_error(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        diff = original - reconstructed
        return {
            "mse": float(np.mean(diff ** 2)),
            "mae": float(np.mean(np.abs(diff))),
            "max_error": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "relative_error": float(np.mean(np.abs(diff) / (np.abs(original) + 1e-8))),
        }

    def compute_compression_ratio(self, original_bits: int = 32) -> float:
        return original_bits / self.config.bits

    def per_channel_quantize(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        scales = np.max(np.abs(data), axis=axis, keepdims=True)
        scales = np.where(scales == 0, 1.0, scales)
        scales = scales / (2 ** (self.config.bits - 1))
        quantized = np.round(data / scales).astype(np.float32)
        qmin = -(2 ** (self.config.bits - 1))
        qmax = 2 ** (self.config.bits - 1) - 1
        return np.clip(quantized, qmin, qmax)

    def per_channel_dequantize(self, quantized: np.ndarray,
                                original: np.ndarray, axis: int = 0) -> np.ndarray:
        scales = np.max(np.abs(original), axis=axis, keepdims=True)
        scales = np.where(scales == 0, 1.0, scales)
        scales = scales / (2 ** (self.config.bits - 1))
        return quantized * scales


class TestQuantizationBasic:
    """基础量化测试"""

    def setup_method(self):
        self.quantizer = MockQuantizer(QuantizationConfig(bits=8, symmetric=True))

    def test_quantize_returns_integers(self):
        data = np.random.randn(100).astype(np.float32)
        q = self.quantizer.quantize(data)
        assert np.all(q == np.round(q))

    def test_quantize_within_range(self):
        data = np.random.randn(100).astype(np.float32)
        q = self.quantizer.quantize(data)
        qmin = -(2 ** 7)
        qmax = 2 ** 7 - 1
        assert np.all(q >= qmin) and np.all(q <= qmax)

    def test_roundtrip(self):
        data = np.random.randn(100).astype(np.float32)
        q = self.quantizer.quantize(data)
        dq = self.quantizer.dequantize(q)
        error = self.quantizer.compute_error(data, dq)
        assert error["rmse"] < 1.0

    def test_zero_input(self):
        data = np.zeros(100, dtype=np.float32)
        q = self.quantizer.quantize(data)
        dq = self.quantizer.dequantize(q)
        assert np.allclose(dq, 0, atol=1e-6)

    def test_constant_input(self):
        data = np.full(100, 5.0, dtype=np.float32)
        q = self.quantizer.quantize(data)
        dq = self.quantizer.dequantize(q)
        assert np.allclose(dq, data, atol=self.quantizer.config.scale_factor)


class TestQuantizationConfig:
    """量化配置测试"""

    def test_8bit_quantization(self):
        q = MockQuantizer(QuantizationConfig(bits=8))
        assert q.config.bits == 8
        assert q.compute_compression_ratio() == 4.0

    def test_4bit_quantization(self):
        q = MockQuantizer(QuantizationConfig(bits=4))
        assert q.compute_compression_ratio() == 8.0

    def test_16bit_quantization(self):
        q = MockQuantizer(QuantizationConfig(bits=16))
        assert q.compute_compression_ratio() == 2.0

    def test_2bit_quantization(self):
        q = MockQuantizer(QuantizationConfig(bits=2))
        data = np.random.randn(100).astype(np.float32)
        q_vals = q.quantize(data)
        qmin = -(2 ** 1)
        qmax = 2 ** 1 - 1
        assert np.all(q_vals >= qmin) and np.all(q_vals <= qmax)


class TestCalibration:
    """校准测试"""

    def setup_method(self):
        self.quantizer = MockQuantizer(QuantizationConfig(bits=8))

    def test_calibration_stats(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        stats = self.quantizer.calibrate(data)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0

    def test_calibration_sets_scale(self):
        data = np.random.randn(100).astype(np.float32) * 10
        self.quantizer.calibrate(data)
        assert self.quantizer.config.scale_factor is not None
        assert self.quantizer.config.scale_factor > 0

    def test_calibration_zero_data(self):
        data = np.zeros(100, dtype=np.float32)
        stats = self.quantizer.calibrate(data)
        assert stats["abs_max"] == 0.0
        assert self.quantizer.config.scale_factor == 1.0


class TestErrorMetrics:
    """误差指标测试"""

    def setup_method(self):
        self.quantizer = MockQuantizer(QuantizationConfig(bits=8))

    def test_error_metrics_keys(self):
        data = np.random.randn(100).astype(np.float32)
        q = self.quantizer.quantize(data)
        dq = self.quantizer.dequantize(q)
        errors = self.quantizer.compute_error(data, dq)
        assert "mse" in errors
        assert "mae" in errors
        assert "max_error" in errors
        assert "rmse" in errors
        assert "relative_error" in errors

    def test_higher_bits_lower_error(self):
        data = np.random.randn(1000).astype(np.float32) * 5
        q8 = MockQuantizer(QuantizationConfig(bits=8))
        q4 = MockQuantizer(QuantizationConfig(bits=4))
        e8 = q8.compute_error(data, q8.dequantize(q8.quantize(data)))
        e4 = q4.compute_error(data, q4.dequantize(q4.quantize(data)))
        assert e8["mse"] < e4["mse"]

    def test_perfect_reconstruction_small_range(self):
        data = np.array([-1, 0, 1], dtype=np.float32)
        q = self.quantizer.quantize(data)
        dq = self.quantizer.dequantize(q)
        errors = self.quantizer.compute_error(data, dq)
        assert errors["max_error"] <= self.quantizer.config.scale_factor


class TestPerChannelQuantization:
    """逐通道量化测试"""

    def setup_method(self):
        self.quantizer = MockQuantizer(QuantizationConfig(bits=8))

    def test_per_channel_shape_preserved(self):
        data = np.random.randn(10, 5).astype(np.float32)
        q = self.quantizer.per_channel_quantize(data, axis=0)
        assert q.shape == data.shape

    def test_per_channel_roundtrip(self):
        data = np.random.randn(10, 5).astype(np.float32) * 10
        q = self.quantizer.per_channel_quantize(data, axis=0)
        dq = self.quantizer.per_channel_dequantize(q, data, axis=0)
        error = np.mean(np.abs(data - dq))
        assert error < 1.0

    def test_per_channel_vs_global(self):
        data = np.random.randn(10, 5).astype(np.float32)
        data[:, 0] *= 100
        data[:, 4] *= 0.01
        q_global = self.quantizer.quantize(data)
        dq_global = self.quantizer.dequantize(q_global)
        q_pc = self.quantizer.per_channel_quantize(data, axis=0)
        dq_pc = self.quantizer.per_channel_dequantize(q_pc, data, axis=0)
        e_global = np.mean((data - dq_global) ** 2)
        e_pc = np.mean((data - dq_pc) ** 2)
        assert e_pc < e_global * 2
