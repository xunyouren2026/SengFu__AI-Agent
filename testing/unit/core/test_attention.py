"""
TestAttention - 核心算法单元测试：注意力机制模块

模块路径: testing/unit/core/test_attention.py
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


class MockAttention:
    """模拟注意力机制"""

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.scale = np.sqrt(self.d_head)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray,
                                      V: np.ndarray,
                                      mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores + mask
        weights = self.softmax(scores)
        output = np.matmul(weights, V)
        return output, weights

    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        return mask

    def multi_head_attention(self, query: np.ndarray, key: np.ndarray,
                              value: np.ndarray,
                              mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        Q_heads = self._split_heads(query, batch_size)
        K_heads = self._split_heads(key, batch_size)
        V_heads = self._split_heads(value, batch_size)

        attn_output, attn_weights = self.scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask)

        output = self._combine_heads(attn_output, batch_size, seq_len)
        return output, attn_weights

    def _split_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        x = x.reshape(batch_size, -1, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray, batch_size: int,
                       seq_len: int) -> np.ndarray:
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return x

    def linear_projection(self, x: np.ndarray, weight: np.ndarray,
                          bias: Optional[np.ndarray] = None) -> np.ndarray:
        out = np.matmul(x, weight.T)
        if bias is not None:
            out = out + bias
        return out

    def compute_attention_stats(self, weights: np.ndarray) -> Dict[str, float]:
        return {
            "mean_attention": float(np.mean(weights)),
            "max_attention": float(np.max(weights)),
            "min_attention": float(np.min(weights)),
            "entropy": float(-np.sum(weights * np.log(weights + 1e-10))),
            "sparsity": float(np.mean(weights < 0.01)),
        }


class TestSoftmax:
    """Softmax函数测试"""

    def setup_method(self):
        self.attn = MockAttention()

    def test_softmax_sums_to_one(self):
        x = np.random.randn(10)
        s = self.attn.softmax(x)
        assert np.isclose(np.sum(s), 1.0)

    def test_softmax_positive(self):
        x = np.random.randn(10)
        s = self.attn.softmax(x)
        assert np.all(s >= 0)

    def test_softmax_uniform_input(self):
        x = np.ones(5)
        s = self.attn.softmax(x)
        assert np.allclose(s, 0.2, atol=1e-6)

    def test_softmax_extreme_values(self):
        x = np.array([1000, 0, 0, 0])
        s = self.attn.softmax(x)
        assert np.isclose(s[0], 1.0, atol=1e-6)
        assert np.allclose(s[1:], 0.0, atol=1e-6)

    def test_softmax_2d(self):
        x = np.random.randn(3, 5)
        s = self.attn.softmax(x, axis=-1)
        assert np.allclose(np.sum(s, axis=-1), 1.0)


class TestScaledDotProductAttention:
    """缩放点积注意力测试"""

    def setup_method(self):
        self.attn = MockAttention(d_model=16, n_heads=1)

    def test_output_shape(self):
        Q = np.random.randn(2, 5, 16).astype(np.float32)
        K = np.random.randn(2, 5, 16).astype(np.float32)
        V = np.random.randn(2, 5, 16).astype(np.float32)
        output, weights = self.attn.scaled_dot_product_attention(Q, K, V)
        assert output.shape == (2, 5, 16)
        assert weights.shape == (2, 5, 5)

    def test_attention_weights_sum_to_one(self):
        Q = np.random.randn(1, 4, 16).astype(np.float32)
        K = np.random.randn(1, 4, 16).astype(np.float32)
        V = np.random.randn(1, 4, 16).astype(np.float32)
        _, weights = self.attn.scaled_dot_product_attention(Q, K, V)
        assert np.allclose(np.sum(weights[0], axis=-1), 1.0, atol=1e-5)

    def test_scaled_by_sqrt_d(self):
        d_head = 16
        attn = MockAttention(d_model=16, n_heads=1)
        Q = np.ones((1, 1, 16), dtype=np.float32)
        K = np.ones((1, 1, 16), dtype=np.float32)
        V = np.ones((1, 1, 16), dtype=np.float32)
        _, weights = attn.scaled_dot_product_attention(Q, K, V)
        expected = attn.softmax(np.array([[16.0 / np.sqrt(d_head)]]))[0, 0]
        assert np.isclose(weights[0, 0, 0], expected, atol=1e-5)

    def test_masked_attention(self):
        Q = np.random.randn(1, 4, 16).astype(np.float32)
        K = np.random.randn(1, 4, 16).astype(np.float32)
        V = np.random.randn(1, 4, 16).astype(np.float32)
        mask = np.triu(np.full((4, 4), -np.inf), k=1)
        _, weights = self.attn.scaled_dot_product_attention(Q, K, V, mask)
        assert np.allclose(weights[0, 0, 1:], 0.0, atol=1e-5)
        assert weights[0, 0, 0] > 0.99


class TestCausalMask:
    """因果掩码测试"""

    def setup_method(self):
        self.attn = MockAttention()

    def test_causal_mask_shape(self):
        mask = self.attn.create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_causal_mask_upper_triangle(self):
        mask = self.attn.create_causal_mask(4)
        assert np.isfinite(mask[0, 0])
        assert np.isfinite(mask[1, 0])
        assert np.isinf(mask[0, 1])
        assert np.isinf(mask[0, 3])

    def test_causal_mask_diagonal(self):
        mask = self.attn.create_causal_mask(3)
        for i in range(3):
            assert np.isfinite(mask[i, i])


class TestMultiHeadAttention:
    """多头注意力测试"""

    def setup_method(self):
        self.attn = MockAttention(d_model=64, n_heads=4)

    def test_multi_head_output_shape(self):
        Q = np.random.randn(2, 5, 64).astype(np.float32)
        K = np.random.randn(2, 5, 64).astype(np.float32)
        V = np.random.randn(2, 5, 64).astype(np.float32)
        output, weights = self.attn.multi_head_attention(Q, K, V)
        assert output.shape == (2, 5, 64)
        assert weights.shape == (2, 4, 5, 5)

    def test_multi_head_weights_sum_to_one(self):
        Q = np.random.randn(1, 3, 64).astype(np.float32)
        K = np.random.randn(1, 3, 64).astype(np.float32)
        V = np.random.randn(1, 3, 64).astype(np.float32)
        _, weights = self.attn.multi_head_attention(Q, K, V)
        for h in range(4):
            assert np.allclose(np.sum(weights[0, h], axis=-1), 1.0, atol=1e-5)

    def test_causal_multi_head(self):
        Q = np.random.randn(1, 5, 64).astype(np.float32)
        K = np.random.randn(1, 5, 64).astype(np.float32)
        V = np.random.randn(1, 5, 64).astype(np.float32)
        mask = self.attn.create_causal_mask(5)
        _, weights = self.attn.multi_head_attention(Q, K, V, mask)
        for h in range(4):
            assert np.allclose(weights[0, h, 0, 1:], 0.0, atol=1e-5)


class TestLinearProjection:
    """线性投影测试"""

    def setup_method(self):
        self.attn = MockAttention(d_model=64, n_heads=4)

    def test_linear_output_shape(self):
        x = np.random.randn(2, 5, 64).astype(np.float32)
        w = np.random.randn(64, 64).astype(np.float32)
        out = self.attn.linear_projection(x, w)
        assert out.shape == (2, 5, 64)

    def test_linear_with_bias(self):
        x = np.random.randn(2, 3, 64).astype(np.float32)
        w = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64).astype(np.float32)
        out = self.attn.linear_projection(x, w, b)
        assert out.shape == (2, 3, 64)

    def test_linear_without_bias(self):
        x = np.random.randn(1, 1, 64).astype(np.float32)
        w = np.random.randn(64, 64).astype(np.float32)
        out = self.attn.linear_projection(x, w)
        expected = np.matmul(x, w.T)
        assert np.allclose(out, expected)


class TestAttentionStats:
    """注意力统计测试"""

    def setup_method(self):
        self.attn = MockAttention(d_model=16, n_heads=1)

    def test_stats_keys(self):
        weights = self.attn.softmax(np.random.randn(4, 4))
        stats = self.attn.compute_attention_stats(weights)
        assert "mean_attention" in stats
        assert "max_attention" in stats
        assert "entropy" in stats
        assert "sparsity" in stats

    def test_focused_attention_high_entropy(self):
        weights = np.zeros((1, 4, 4))
        weights[0, :, 0] = 1.0
        stats = self.attn.compute_attention_stats(weights)
        assert stats["entropy"] < 0.01

    def test_uniform_attention_max_entropy(self):
        weights = np.full((1, 4, 4), 0.25)
        stats = self.attn.compute_attention_stats(weights)
        assert stats["entropy"] > 1.0
