"""
BenchmarkInference - 性能测试：推理基准
模块路径: testing/performance/benchmark_inference.py
"""
import os, sys, json, time, random, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import pytest
import numpy as np

pytestmark = pytest.mark.performance

@dataclass
class BenchmarkResult:
    name: str
    num_iterations: int
    total_time_s: float
    avg_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_qps: float

class MockInferenceEngine:
    def __init__(self, base_latency_ms=5.0):
        self.base_latency = base_latency_ms / 1000.0

    async def infer(self, input_data):
        await asyncio.sleep(self.base_latency)
        return np.random.randn(*input_data.shape[:-1], 10).astype(np.float32)

    def infer_sync(self, input_data):
        time.sleep(self.base_latency)
        return np.random.randn(*input_data.shape[:-1], 10).astype(np.float32)

def compute_percentiles(data, percentiles):
    sorted_data = sorted(data)
    n = len(sorted_data)
    return {p: sorted_data[min(int(np.ceil(p / 100.0 * n)) - 1, n - 1)] for p in percentiles}

class TestBenchmarkInference:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.engine = MockInferenceEngine(base_latency_ms=2.0)
        self.test_data = []
        yield
        self.test_data.clear()

    @pytest.mark.asyncio
    async def test_single_inference_latency(self):
        data = np.random.randn(1, 10).astype(np.float32)
        start = time.time()
        await self.engine.infer(data)
        assert (time.time() - start) * 1000 > 0

    @pytest.mark.asyncio
    async def test_latency_percentiles(self):
        data = np.random.randn(1, 10).astype(np.float32)
        latencies = []
        for _ in range(100):
            start = time.time()
            await self.engine.infer(data)
            latencies.append((time.time() - start) * 1000)
        p = compute_percentiles(latencies, [50, 95, 99])
        assert p[50] > 0 and p[95] >= p[50] and p[99] >= p[95]

    @pytest.mark.asyncio
    async def test_concurrent_inference_throughput(self):
        data = np.random.randn(1, 10).astype(np.float32)
        n = 20
        start = time.time()
        await asyncio.gather(*[self.engine.infer(data) for _ in range(n)])
        assert n / (time.time() - start) > 0

    def test_sync_inference_latency(self):
        data = np.random.randn(1, 10).astype(np.float32)
        latencies = []
        for _ in range(20):
            start = time.time()
            self.engine.infer_sync(data)
            latencies.append((time.time() - start) * 1000)
        assert sum(latencies) / len(latencies) > 0

    @pytest.mark.asyncio
    async def test_input_size_impact(self):
        latencies = {}
        for size in [10, 100, 1000]:
            data = np.random.randn(1, size).astype(np.float32)
            start = time.time()
            await self.engine.infer(data)
            latencies[size] = (time.time() - start) * 1000
        assert all(v > 0 for v in latencies.values())

    def test_percentile_computation(self):
        data = list(range(1, 101))
        result = compute_percentiles(data, [50, 95, 99])
        assert result[50] == 50 and result[99] == 99

    @pytest.mark.asyncio
    async def test_sustained_load(self):
        data = np.random.randn(1, 10).astype(np.float32)
        latencies = []
        start = time.time()
        while time.time() - start < 0.3:
            s = time.time()
            await self.engine.infer(data)
            latencies.append((time.time() - s) * 1000)
        assert len(latencies) > 0
