"""
TestEWC - 核心模块单元测试：EWC (Elastic Weight Consolidation)

模块路径: testing/unit/core/test_ewc.py
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


class MockEWC:
    def __init__(self, num_params: int = 100, lambda_ewc: float = 5000.0):
        self.num_params = num_params
        self.lambda_ewc = lambda_ewc
        self.fisher_matrix = np.zeros(num_params, dtype=np.float32)
        self.optimal_params = np.zeros(num_params, dtype=np.float32)
        self.task_count = 0

    def compute_fisher(self, model_params: np.ndarray, data_samples: List[np.ndarray]) -> np.ndarray:
        fisher = np.zeros(self.num_params, dtype=np.float32)
        for sample in data_samples:
            grad = np.random.randn(self.num_params).astype(np.float32)
            fisher += grad ** 2
        fisher /= len(data_samples)
        return fisher

    def consolidate(self, model_params: np.ndarray, fisher: np.ndarray):
        self.optimal_params = model_params.copy()
        self.fisher_matrix += fisher
        self.task_count += 1

    def compute_ewc_loss(self, current_params: np.ndarray) -> float:
        param_diff = current_params - self.optimal_params
        loss = 0.5 * self.lambda_ewc * np.sum(self.fisher_matrix * param_diff ** 2)
        return float(loss)

    def compute_total_loss(self, task_loss: float, current_params: np.ndarray) -> float:
        ewc_loss = self.compute_ewc_loss(current_params)
        return task_loss + ewc_loss

    def get_importance_weights(self) -> np.ndarray:
        return self.fisher_matrix / (self.fisher_matrix.sum() + 1e-8)

    def prune_low_importance(self, threshold: float = 0.01) -> np.ndarray:
        weights = self.get_importance_weights()
        mask = weights > threshold
        return mask


class TestEWC:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.ewc = MockEWC(num_params=100)
        self.test_data = []
        yield
        self.test_data.clear()

    def test_ewc_init(self):
        assert self.ewc.lambda_ewc == 5000.0
        assert self.ewc.task_count == 0
        assert np.allclose(self.ewc.fisher_matrix, 0)

    def test_compute_fisher(self):
        params = np.random.randn(100).astype(np.float32)
        samples = [np.random.randn(10).astype(np.float32) for _ in range(5)]
        fisher = self.ewc.compute_fisher(params, samples)
        assert fisher.shape == (100,)
        assert np.all(fisher >= 0)

    def test_consolidate(self):
        params = np.random.randn(100).astype(np.float32)
        fisher = np.random.rand(100).astype(np.float32)
        self.ewc.consolidate(params, fisher)
        assert self.ewc.task_count == 1
        np.testing.assert_array_equal(self.ewc.optimal_params, params)

    def test_consolidate_multiple_tasks(self):
        for i in range(3):
            params = np.random.randn(100).astype(np.float32)
            fisher = np.random.rand(100).astype(np.float32)
            self.ewc.consolidate(params, fisher)
        assert self.ewc.task_count == 3

    def test_ewc_loss_zero_at_optimal(self):
        params = np.random.randn(100).astype(np.float32)
        self.ewc.consolidate(params, np.ones(100).astype(np.float32))
        loss = self.ewc.compute_ewc_loss(params)
        assert abs(loss) < 1e-5

    def test_ewc_loss_positive_when_diverged(self):
        params = np.random.randn(100).astype(np.float32)
        self.ewc.consolidate(params, np.ones(100).astype(np.float32))
        diverged = params + 10.0
        loss = self.ewc.compute_ewc_loss(diverged)
        assert loss > 0

    def test_total_loss(self):
        params = np.random.randn(100).astype(np.float32)
        self.ewc.consolidate(params, np.ones(100).astype(np.float32))
        total = self.ewc.compute_total_loss(0.5, params)
        assert total >= 0.5

    def test_importance_weights(self):
        fisher = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.ewc.fisher_matrix = fisher
        weights = self.ewc.get_importance_weights()
        assert abs(weights.sum() - 1.0) < 1e-5
        assert weights[-1] > weights[0]

    def test_prune_low_importance(self):
        fisher = np.array([0.001, 0.1, 0.001, 0.5, 0.001])
        self.ewc.fisher_matrix = fisher
        mask = self.ewc.prune_low_importance(threshold=0.01)
        assert mask.sum() == 2

    @pytest.mark.parametrize("lambda_val", [100, 1000, 5000, 10000])
    def test_various_lambda_values(self, lambda_val):
        ewc = MockEWC(lambda_ewc=lambda_val)
        params = np.random.randn(100).astype(np.float32)
        ewc.consolidate(params, np.ones(100).astype(np.float32))
        loss = ewc.compute_ewc_loss(params + 1.0)
        assert loss > 0
