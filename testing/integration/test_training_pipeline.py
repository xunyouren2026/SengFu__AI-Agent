"""
TestTrainingPipeline - 集成测试：训练管道
模块路径: testing/integration/test_training_pipeline.py
"""
import os, sys, json, time, random, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import pytest
import numpy as np

pytestmark = pytest.mark.integration

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adam"
    loss_fn: str = "cross_entropy"

@dataclass
class DataBatch:
    inputs: np.ndarray
    labels: np.ndarray
    batch_id: int = 0

class MockDataLoader:
    def __init__(self, num_samples=100, feature_dim=10, num_classes=5):
        self.num_samples = num_samples
        self.data = np.random.randn(num_samples, feature_dim).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=num_samples)

    def get_batch(self, batch_size, batch_idx):
        start = batch_idx * batch_size
        end = min(start + batch_size, self.num_samples)
        return DataBatch(inputs=self.data[start:end], labels=self.labels[start:end], batch_id=batch_idx)

    def num_batches(self, batch_size):
        return (self.num_samples + batch_size - 1) // batch_size

    def split(self, val_ratio=0.2):
        idx = int(self.num_samples * (1 - val_ratio))
        return (self.data[:idx], self.labels[:idx]), (self.data[idx:], self.labels[idx:])

class MockTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics_history = []

    def train_step(self, batch: DataBatch) -> Dict[str, float]:
        return {"loss": random.uniform(0.5, 2.0), "accuracy": random.uniform(0.6, 0.95)}

    def validate(self, val_data, val_labels) -> Dict[str, float]:
        return {"val_loss": random.uniform(0.5, 1.5), "val_accuracy": random.uniform(0.7, 0.95)}

    def get_lr_schedule(self, epoch: int) -> float:
        return self.config.learning_rate * (0.95 ** epoch)

class TestTrainingPipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.config = TrainingConfig()
        self.dataloader = MockDataLoader()
        self.trainer = MockTrainer(self.config)
        self.test_data = []
        yield
        self.test_data.clear()

    def test_get_batch(self):
        batch = self.dataloader.get_batch(32, 0)
        assert isinstance(batch, DataBatch) and batch.inputs.shape[0] <= 32

    def test_num_batches(self):
        assert self.dataloader.num_batches(32) == 4

    def test_data_split(self):
        (tx, ty), (vx, vy) = self.dataloader.split(0.2)
        assert len(tx) == 80 and len(vx) == 20

    def test_train_step(self):
        batch = self.dataloader.get_batch(32, 0)
        metrics = self.trainer.train_step(batch)
        assert "loss" in metrics and metrics["loss"] > 0

    def test_lr_schedule(self):
        assert self.trainer.get_lr_schedule(5) < self.trainer.get_lr_schedule(0)

    def test_full_training_epoch(self):
        num_batches = self.dataloader.num_batches(self.config.batch_size)
        total_loss = 0
        for i in range(num_batches):
            batch = self.dataloader.get_batch(self.config.batch_size, i)
            total_loss += self.trainer.train_step(batch)["loss"]
        assert total_loss / num_batches > 0

    def test_multi_epoch_training(self):
        for epoch in range(3):
            for i in range(self.dataloader.num_batches(self.config.batch_size)):
                self.trainer.train_step(self.dataloader.get_batch(self.config.batch_size, i))
            self.trainer.metrics_history.append({"epoch": epoch, "lr": self.trainer.get_lr_schedule(epoch)})
        assert len(self.trainer.metrics_history) == 3

    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64, 128])
    def test_various_batch_sizes(self, batch_size):
        n = self.dataloader.num_batches(batch_size)
        assert n > 0

    def test_gradient_clipping_simulation(self):
        gradients = np.random.randn(100) * 10
        max_norm = 1.0
        total_norm = np.linalg.norm(gradients)
        if total_norm > max_norm:
            gradients = gradients * (max_norm / total_norm)
        assert np.linalg.norm(gradients) <= max_norm + 1e-9

    def test_early_stopping_check(self):
        val_losses = [2.0, 1.5, 1.2, 1.1, 1.15]
        patience = 2
        best_loss, wait, stopped = float("inf"), 0, False
        for loss in val_losses:
            if loss < best_loss:
                best_loss, wait = loss, 0
            else:
                wait += 1
                if wait >= patience:
                    stopped = True
                    break
        assert stopped and best_loss == 1.1

    @pytest.mark.asyncio
    async def test_async_data_loading(self):
        async def load_batch(idx):
            await asyncio.sleep(0.001)
            return self.dataloader.get_batch(32, idx)
        batches = await asyncio.gather(*[load_batch(i) for i in range(4)])
        assert len(batches) == 4
