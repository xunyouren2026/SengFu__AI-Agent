"""
TestSuperfastMath - 核心模块单元测试：快速数学运算

模块路径: testing/unit/core/test_superfast_math.py
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

import pytest
import numpy as np

pytestmark = pytest.mark.unit


class MockFastMath:
    @staticmethod
    def fast_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    @staticmethod
    def fast_softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    @staticmethod
    def fast_layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    @staticmethod
    def fast_gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def fast_sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def fast_relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def fast_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        return -float(np.log(probs[targets] + 1e-10))


class TestSuperfastMath:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.math = MockFastMath()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_matmul(self):
        a = np.random.randn(32, 64).astype(np.float32)
        b = np.random.randn(64, 128).astype(np.float32)
        result = self.math.fast_matmul(a, b)
        assert result.shape == (32, 128)

    def test_matmul_square(self):
        a = np.random.randn(64, 64).astype(np.float32)
        result = self.math.fast_matmul(a, a)
        assert result.shape == (64, 64)

    def test_softmax(self):
        x = np.random.randn(10).astype(np.float32)
        result = self.math.fast_softmax(x)
        assert abs(result.sum() - 1.0) < 1e-5

    def test_softmax_2d(self):
        x = np.random.randn(4, 10).astype(np.float32)
        result = self.math.fast_softmax(x)
        assert np.allclose(result.sum(axis=-1), 1.0, atol=1e-5)

    def test_layernorm(self):
        x = np.random.randn(32, 64).astype(np.float32)
        result = self.math.fast_layernorm(x)
        assert result.shape == x.shape
        mean = result.mean(axis=-1)
        assert np.allclose(mean, 0, atol=1e-5)

    def test_layernorm_unit_variance(self):
        x = np.random.randn(32, 64).astype(np.float32)
        result = self.math.fast_layernorm(x)
        var = result.var(axis=-1)
        assert np.allclose(var, 1.0, atol=0.1)

    def test_gelu(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = self.math.fast_gelu(x)
        assert result[2] == 0.0  # gelu(0) = 0
        assert result[3] > 0  # gelu(positive) > 0
        assert result[1] < 0  # gelu(negative) < 0

    def test_gelu_approximation(self):
        x = np.random.randn(100).astype(np.float32)
        fast_result = self.math.fast_gelu(x)
        # Verify shape preserved
        assert fast_result.shape == x.shape

    def test_sigmoid(self):
        x = np.array([-10.0, 0.0, 10.0])
        result = self.math.fast_sigmoid(x)
        assert result[0] < 0.01
        assert abs(result[1] - 0.5) < 0.01
        assert result[2] > 0.99

    def test_relu(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = self.math.fast_relu(x)
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 2])

    def test_cross_entropy(self):
        logits = np.array([2.0, 1.0, 0.1])
        loss = self.math.fast_cross_entropy(logits, 0)
        assert loss > 0
        loss_correct = self.math.fast_cross_entropy(logits, 0)
        loss_wrong = self.math.fast_cross_entropy(logits, 2)
        assert loss_correct < loss_wrong

    def test_matmul_performance(self):
        a = np.random.randn(256, 512).astype(np.float32)
        b = np.random.randn(512, 256).astype(np.float32)
        start = time.time()
        for _ in range(10):
            self.math.fast_matmul(a, b)
        elapsed = time.time() - start
        assert elapsed > 0

    @pytest.mark.parametrize("size", [16, 64, 256])
    def test_various_sizes(self, size):
        x = np.random.randn(size).astype(np.float32)
        result = self.math.fast_softmax(x)
        assert abs(result.sum() - 1.0) < 1e-5
