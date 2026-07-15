"""
TestMoE - 核心算法单元测试：混合专家模型模块

模块路径: testing/unit/core/test_moe.py
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


class MockExpert:
    """模拟专家网络"""

    def __init__(self, expert_id: int, d_model: int = 64):
        self.expert_id = expert_id
        self.d_model = d_model
        self.call_count = 0
        self.weight = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return np.matmul(x, self.weight)

    def reset_stats(self):
        self.call_count = 0


class MockRouter:
    """模拟路由器"""

    def __init__(self, n_experts: int, d_model: int = 64, top_k: int = 2):
        self.n_experts = n_experts
        self.d_model = d_model
        self.top_k = top_k
        self.gate_weight = np.random.randn(d_model, n_experts).astype(np.float32) * 0.02

    def route(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits = np.matmul(x, self.gate_weight)
        weights = self._softmax(logits)
        top_k_weights, top_k_indices = self._top_k(weights, self.top_k)
        top_k_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)
        return top_k_weights, top_k_indices

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _top_k(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.argsort(x, axis=-1)[..., ::-1][..., :k]
        weights = np.take_along_axis(x, indices, axis=-1)
        return weights, indices

    def compute_load_balance(self, routing_indices: np.ndarray,
                              n_experts: int) -> Dict[str, float]:
        total_tokens = routing_indices.size
        expert_counts = np.zeros(n_experts)
        for idx in range(n_experts):
            expert_counts[idx] = np.sum(routing_indices == idx)
        freq = expert_counts / total_tokens
        ideal = 1.0 / n_experts
        imbalance = float(np.sum(np.abs(freq - ideal)))
        max_ratio = float(np.max(expert_counts) / (np.min(expert_counts) + 1e-8))
        return {
            "expert_frequencies": freq.tolist(),
            "load_imbalance": imbalance,
            "max_load_ratio": max_ratio,
            "coefficient_of_variation": float(np.std(freq) / (np.mean(freq) + 1e-8)),
        }


class MockMoE:
    """模拟混合专家模型"""

    def __init__(self, n_experts: int = 4, d_model: int = 64, top_k: int = 2):
        self.n_experts = n_experts
        self.d_model = d_model
        self.top_k = top_k
        self.experts = [MockExpert(i, d_model) for i in range(n_experts)]
        self.router = MockRouter(n_experts, d_model, top_k)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        weights, indices = self.router.route(x)
        output = np.zeros_like(x)
        aux_loss = 0.0

        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = indices[..., k] == e
                if np.any(mask):
                    expert_out = self.experts[e].forward(x)
                    w = weights[..., k:k+1]
                    output += mask[..., np.newaxis] * expert_out * w[..., np.newaxis]

        n_experts = self.n_experts
        total = indices.size
        expert_counts = np.zeros(n_experts)
        for idx in range(n_experts):
            expert_counts[idx] = np.sum(indices == idx)
        freq = expert_counts / total
        aux_loss = float(n_experts * np.sum(freq ** 2))

        return output, {"aux_loss": aux_loss, "routing_weights": weights}


class TestMockExpert:
    """专家网络测试"""

    def setup_method(self):
        self.expert = MockExpert(0, d_model=32)

    def test_forward_shape(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        out = self.expert.forward(x)
        assert out.shape == x.shape

    def test_call_count_increments(self):
        x = np.random.randn(1, 1, 32).astype(np.float32)
        assert self.expert.call_count == 0
        self.expert.forward(x)
        assert self.expert.call_count == 1
        self.expert.forward(x)
        assert self.expert.call_count == 2

    def test_reset_stats(self):
        x = np.random.randn(1, 1, 32).astype(np.float32)
        self.expert.forward(x)
        self.expert.reset_stats()
        assert self.expert.call_count == 0

    def test_deterministic_output(self):
        x = np.random.randn(1, 1, 32).astype(np.float32)
        out1 = self.expert.forward(x)
        out2 = self.expert.forward(x)
        assert np.allclose(out1, out2)


class TestMockRouter:
    """路由器测试"""

    def setup_method(self):
        self.router = MockRouter(n_experts=4, d_model=32, top_k=2)

    def test_route_returns_weights_and_indices(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        weights, indices = self.router.route(x)
        assert weights.shape == (2, 5, 2)
        assert indices.shape == (2, 5, 2)

    def test_routing_weights_sum_to_one(self):
        x = np.random.randn(1, 4, 32).astype(np.float32)
        weights, _ = self.router.route(x)
        assert np.allclose(np.sum(weights[0], axis=-1), 1.0, atol=1e-5)

    def test_routing_weights_positive(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        weights, _ = self.router.route(x)
        assert np.all(weights >= 0)

    def test_top_k_indices_valid(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        _, indices = self.router.route(x)
        assert np.all(indices >= 0)
        assert np.all(indices < self.router.n_experts)


class TestLoadBalancing:
    """负载均衡测试"""

    def setup_method(self):
        self.router = MockRouter(n_experts=4, d_model=32, top_k=2)

    def test_balanced_routing(self):
        indices = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        stats = self.router.compute_load_balance(indices, 4)
        assert stats["load_imbalance"] < 1.0

    def test_imbalanced_routing(self):
        indices = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        stats = self.router.compute_load_balance(indices, 4)
        assert stats["load_imbalance"] > 1.0
        assert stats["max_load_ratio"] > 1.0

    def test_load_balance_stats_keys(self):
        indices = np.random.randint(0, 4, size=(10, 2))
        stats = self.router.compute_load_balance(indices, 4)
        assert "expert_frequencies" in stats
        assert "load_imbalance" in stats
        assert "max_load_ratio" in stats
        assert "coefficient_of_variation" in stats


class TestMoEForward:
    """MoE前向传播测试"""

    def setup_method(self):
        self.moe = MockMoE(n_experts=4, d_model=32, top_k=2)

    def test_forward_output_shape(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        output, info = self.moe.forward(x)
        assert output.shape == x.shape

    def test_forward_returns_aux_loss(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        _, info = self.moe.forward(x)
        assert "aux_loss" in info
        assert info["aux_loss"] >= 0

    def test_forward_returns_routing_weights(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        _, info = self.moe.forward(x)
        assert "routing_weights" in info

    def test_different_experts_used(self):
        x = np.random.randn(10, 5, 32).astype(np.float32)
        self.moe.forward(x)
        counts = [e.call_count for e in self.moe.experts]
        active = sum(1 for c in counts if c > 0)
        assert active >= 2

    def test_aux_loss_range(self):
        x = np.random.randn(2, 5, 32).astype(np.float32)
        _, info = self.moe.forward(x)
        assert 0 <= info["aux_loss"] <= self.moe.n_experts

    def test_varying_top_k(self):
        moe1 = MockMoE(n_experts=4, d_model=32, top_k=1)
        moe2 = MockMoE(n_experts=4, d_model=32, top_k=4)
        x = np.random.randn(2, 5, 32).astype(np.float32)
        _, info1 = moe1.forward(x)
        _, info2 = moe2.forward(x)
        assert info1["routing_weights"].shape[-1] == 1
        assert info2["routing_weights"].shape[-1] == 4
