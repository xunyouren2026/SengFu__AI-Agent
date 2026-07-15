"""
BenchmarkDataset - 测试数据库：数据集基准测试

模块路径: testing/database/benchmark_dataset.py
"""

import os, sys, json, time, random
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

import pytest
import numpy as np

pytestmark = pytest.mark.unit


@dataclass
class DatasetStats:
    name: str
    num_samples: int
    num_features: int
    num_classes: int
    size_mb: float
    load_time_s: float


class MockDataset:
    def __init__(self, name: str, num_samples: int = 1000, num_features: int = 10, num_classes: int = 5):
        self.name = name
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, num_features).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=num_samples)

    def get_stats(self) -> DatasetStats:
        size_bytes = self.data.nbytes + self.labels.nbytes
        return DatasetStats(
            name=self.name, num_samples=self.num_samples, num_features=self.num_features,
            num_classes=self.num_classes, size_mb=size_bytes / (1024*1024),
            load_time_s=0.0
        )

    def get_sample(self, idx: int) -> tuple:
        return self.data[idx], self.labels[idx]

    def get_batch(self, indices: List[int]) -> tuple:
        return self.data[indices], self.labels[indices]

    def class_distribution(self) -> Dict[int, int]:
        dist = {}
        for label in self.labels:
            dist[int(label)] = dist.get(int(label), 0) + 1
        return dist

    def shuffle(self):
        idx = np.random.permutation(self.num_samples)
        self.data = self.data[idx]
        self.labels = self.labels[idx]

    def split(self, ratio: float = 0.8):
        idx = int(self.num_samples * ratio)
        return (self.data[:idx], self.labels[:idx]), (self.data[idx:], self.labels[idx:])


class TestBenchmarkDataset:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.dataset = MockDataset("test_dataset")
        self.test_data = []
        yield
        self.test_data.clear()

    def test_dataset_creation(self):
        assert self.dataset.num_samples == 1000
        assert self.dataset.data.shape == (1000, 10)

    def test_get_stats(self):
        stats = self.dataset.get_stats()
        assert stats.name == "test_dataset"
        assert stats.num_samples == 1000

    def test_get_sample(self):
        data, label = self.dataset.get_sample(0)
        assert data.shape == (10,)
        assert isinstance(label, (int, np.integer))

    def test_get_batch(self):
        data, labels = self.dataset.get_batch([0, 1, 2, 3])
        assert data.shape == (4, 10)
        assert len(labels) == 4

    def test_class_distribution(self):
        dist = self.dataset.class_distribution()
        assert sum(dist.values()) == 1000
        assert len(dist) <= self.dataset.num_classes

    def test_shuffle(self):
        original_first = self.dataset.data[0].copy()
        self.dataset.shuffle()
        assert not np.array_equal(original_first, self.dataset.data[0])

    def test_split(self):
        (train_x, train_y), (val_x, val_y) = self.dataset.split(0.8)
        assert len(train_x) == 800 and len(val_x) == 200

    def test_various_dataset_sizes(self):
        for size in [100, 1000, 10000]:
            ds = MockDataset(f"ds_{size}", num_samples=size)
            assert ds.data.shape[0] == size

    def test_dataset_serialization(self):
        stats = self.dataset.get_stats()
        data = json.dumps({"name": stats.name, "samples": stats.num_samples, "size_mb": stats.size_mb})
        parsed = json.loads(data)
        assert parsed["samples"] == 1000

    def test_persistence(self):
        data_path = self.temp_dir / "dataset.npz"
        np.savez(data_path, data=self.dataset.data, labels=self.dataset.labels)
        loaded = np.load(data_path)
        assert loaded["data"].shape == self.dataset.data.shape
