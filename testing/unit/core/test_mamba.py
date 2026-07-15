"""
TestMamba - 核心算法单元测试：Mamba（状态空间模型）模块

模块路径: testing/unit/core/test_mamba.py
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


@dataclass
class SSMConfig:
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 16


class MockSSM:
    """模拟状态空间模型 (Mamba-style)"""

    def __init__(self, config: Optional[SSMConfig] = None):
        self.config = config or SSMConfig()
        self.d_model = self.config.d_model
        self.d_state = self.config.d_state
        self.d_inner = self.config.d_model * self.config.expand_factor

        self.A = np.random.randn(self.d_state, self.d_state).astype(np.float32) * 0.01
        self.B = np.random.randn(self.d_state, self.d_inner).astype(np.float32) * 0.01
        self.C = np.random.randn(self.d_model, self.d_state).astype(np.float32) * 0.01
        self.D = np.random.randn(self.d_model, self.d_inner).astype(np.float32) * 0.01

        self.dt_bias = np.random.randn(self.d_inner).astype(np.float32)
        self.A_log = np.log(np.abs(self.A) + 1e-4)

    def discretize(self, dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dA = np.exp(self.A[np.newaxis, :, :] * dt[:, np.newaxis, np.newaxis])
        dB = (self.B[np.newaxis, :, :] *
              ((np.exp(self.A[np.newaxis, :, :] * dt[:, np.newaxis, np.newaxis]) - 1) /
               (self.A[np.newaxis, :, :] + 1e-8)))
        return dA, dB, self.C

    def ssm_step(self, x_t: np.ndarray, state: np.ndarray,
                 dt: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        A_discrete = np.exp(self.A * dt)
        B_discrete = ((np.exp(self.A * dt) - 1) / (self.A + 1e-8))

        new_state = A_discrete @ state + B_discrete @ x_t
        y_t = self.C @ new_state + self.D @ x_t
        return y_t, new_state

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, seq_len, _ = x.shape
        states = np.zeros((batch_size, self.d_state), dtype=np.float32)
        outputs = np.zeros((batch_size, seq_len, self.d_model), dtype=np.float32)

        for t in range(seq_len):
            x_t = x[:, t, :]
            for b in range(batch_size):
                y_t, new_state = self.ssm_step(x_t[b], states[b])
                outputs[b, t, :] = y_t
                states[b] = new_state

        return outputs, states

    def selective_scan(self, x: np.ndarray, dt: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_inner = x.shape
        dA, dB, C = self.discretize(dt)

        states = np.zeros((batch_size, self.d_state), dtype=np.float32)
        outputs = np.zeros((batch_size, seq_len, self.d_model), dtype=np.float32)

        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = dA[t] @ states.T + dB[t] @ x_t.T
            y_t = C @ new_states
            outputs[:, t, :] = y_t.T
            states = new_states.T

        return outputs

    def compute_state_norm(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(state))

    def state_decay_rate(self, dt: float = 0.001) -> float:
        A_discrete = np.exp(self.A * dt)
        eigenvalues = np.linalg.eigvals(A_discrete)
        return float(np.mean(np.abs(eigenvalues)))


class TestSSMStep:
    """SSM单步测试"""

    def setup_method(self):
        self.ssm = MockSSM(SSMConfig(d_model=16, d_state=8, expand_factor=2))

    def test_step_output_shape(self):
        x_t = np.random.randn(32).astype(np.float32)
        state = np.zeros(8, dtype=np.float32)
        y_t, new_state = self.ssm.ssm_step(x_t, state)
        assert y_t.shape == (16,)
        assert new_state.shape == (8,)

    def test_step_state_evolution(self):
        x_t = np.random.randn(32).astype(np.float32)
        state = np.zeros(8, dtype=np.float32)
        _, new_state = self.ssm.ssm_step(x_t, state)
        assert not np.allclose(new_state, 0)

    def test_step_deterministic(self):
        x_t = np.random.randn(32).astype(np.float32)
        state = np.zeros(8, dtype=np.float32)
        y1, s1 = self.ssm.ssm_step(x_t, state)
        y2, s2 = self.ssm.ssm_step(x_t, state.copy())
        assert np.allclose(y1, y2)
        assert np.allclose(s1, s2)

    def test_step_zero_input(self):
        state = np.random.randn(8).astype(np.float32)
        y_t, new_state = self.ssm.ssm_step(np.zeros(32, dtype=np.float32), state)
        assert new_state.shape == (8,)


class TestSSMForward:
    """SSM前向传播测试"""

    def setup_method(self):
        self.ssm = MockSSM(SSMConfig(d_model=16, d_state=8, expand_factor=2))

    def test_forward_output_shape(self):
        x = np.random.randn(2, 10, 32).astype(np.float32)
        output, states = self.ssm.forward(x)
        assert output.shape == (2, 10, 16)
        assert states.shape == (2, 8)

    def test_forward_sequential_consistency(self):
        x = np.random.randn(1, 5, 32).astype(np.float32)
        output, _ = self.ssm.forward(x)
        for t in range(5):
            single_x = x[:, t:t+1, :]
            single_out, _ = self.ssm.forward(single_x)
            assert np.allclose(output[:, t, :], single_out[:, 0, :], atol=1e-4)

    def test_forward_batch_consistency(self):
        x = np.random.randn(3, 5, 32).astype(np.float32)
        output, _ = self.ssm.forward(x)
        for b in range(3):
            single = x[b:b+1]
            single_out, _ = self.ssm.forward(single)
            assert np.allclose(output[b], single_out[0], atol=1e-4)


class TestSelectiveScan:
    """选择性扫描测试"""

    def setup_method(self):
        self.ssm = MockSSM(SSMConfig(d_model=16, d_state=8, expand_factor=2))

    def test_selective_scan_shape(self):
        x = np.random.randn(2, 10, 32).astype(np.float32)
        dt = np.random.uniform(0.001, 0.01, (10,)).astype(np.float32)
        output = self.ssm.selective_scan(x, dt)
        assert output.shape == (2, 10, 16)

    def test_selective_scan_dt_dependent(self):
        x = np.random.randn(1, 5, 32).astype(np.float32)
        dt1 = np.full(5, 0.001, dtype=np.float32)
        dt2 = np.full(5, 0.01, dtype=np.float32)
        out1 = self.ssm.selective_scan(x, dt1)
        out2 = self.ssm.selective_scan(x, dt2)
        assert not np.allclose(out1, out2)


class TestDiscretization:
    """离散化测试"""

    def setup_method(self):
        self.ssm = MockSSM(SSMConfig(d_model=16, d_state=8, expand_factor=2))

    def test_discretize_shapes(self):
        dt = np.random.uniform(0.001, 0.01, (5,)).astype(np.float32)
        dA, dB, C = self.ssm.discretize(dt)
        assert dA.shape == (5, 8, 8)
        assert dB.shape == (5, 8, 32)
        assert C.shape == (16, 8)

    def test_discretize_dA_stable(self):
        dt = np.full(5, 0.001, dtype=np.float32)
        dA, _, _ = self.ssm.discretize(dt)
        eigenvalues = np.linalg.eigvals(dA)
        assert np.all(np.abs(eigenvalues) <= 1.0 + 1e-5)


class TestStateAnalysis:
    """状态分析测试"""

    def setup_method(self):
        self.ssm = MockSSM(SSMConfig(d_model=16, d_state=8, expand_factor=2))

    def test_state_norm(self):
        state = np.random.randn(8).astype(np.float32)
        norm = self.ssm.compute_state_norm(state)
        expected = float(np.linalg.norm(state))
        assert np.isclose(norm, expected)

    def test_state_norm_zero(self):
        norm = self.ssm.compute_state_norm(np.zeros(8))
        assert norm == 0.0

    def test_decay_rate(self):
        rate = self.ssm.state_decay_rate(dt=0.001)
        assert 0 <= rate <= 1.0

    def test_decay_rate_smaller_dt_slower_decay(self):
        rate_small = self.ssm.state_decay_rate(dt=0.0001)
        rate_large = self.ssm.state_decay_rate(dt=0.01)
        assert rate_small <= rate_large

    def test_long_sequence_state_growth(self):
        x = np.random.randn(1, 100, 32).astype(np.float32) * 0.1
        _, final_state = self.ssm.forward(x)
        norm = self.ssm.compute_state_norm(final_state[0])
        assert norm > 0
