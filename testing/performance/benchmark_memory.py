"""
BenchmarkMemory - 性能测试：内存基准
模块路径: testing/performance/benchmark_memory.py
"""
import os, sys, json, time, random, tempfile, shutil, gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import pytest
import numpy as np

pytestmark = pytest.mark.performance

@dataclass
class MemoryBenchmark:
    name: str
    allocation_size_mb: float
    peak_memory_mb: float
    allocation_time_ms: float
    deallocation_time_ms: float

class MockMemoryManager:
    def __init__(self):
        self.allocations = []
        self.total_allocated_bytes = 0

    def allocate(self, shape, dtype=np.float32):
        arr = np.zeros(shape, dtype=dtype)
        self.allocations.append(arr)
        self.total_allocated_bytes += arr.nbytes
        return arr

    def allocate_random(self, shape, dtype=np.float32):
        arr = np.random.randn(*shape).astype(dtype)
        self.allocations.append(arr)
        self.total_allocated_bytes += arr.nbytes
        return arr

    def deallocate_all(self):
        self.total_allocated_bytes = 0
        self.allocations.clear()
        gc.collect()

    def get_total_mb(self):
        return self.total_allocated_bytes / (1024 * 1024)

class TestBenchmarkMemory:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.manager = MockMemoryManager()
        self.test_data = []
        yield
        self.manager.deallocate_all()
        self.test_data.clear()

    def test_small_allocation(self):
        arr = self.manager.allocate((100, 100))
        assert arr.shape == (100, 100) and len(self.manager.allocations) == 1

    def test_large_allocation(self):
        arr = self.manager.allocate((10000, 1000))
        assert arr.nbytes == 10000 * 1000 * 4

    def test_allocation_speed(self):
        start = time.time()
        for _ in range(100):
            self.manager.allocate((100, 100))
        assert (time.time() - start) * 1000 > 0

    def test_deallocation(self):
        for _ in range(10):
            self.manager.allocate((1000, 1000))
        self.manager.deallocate_all()
        assert len(self.manager.allocations) == 0

    def test_total_memory_tracking(self):
        self.manager.allocate((1000, 1000))
        self.manager.allocate((2000, 2000))
        assert abs(self.manager.get_total_mb() - 20.0) < 0.01

    def test_dtype_impact(self):
        f32 = self.manager.allocate((1000, 1000), np.float32)
        self.manager.deallocate_all()
        f64 = self.manager.allocate((1000, 1000), np.float64)
        assert f64.nbytes == 2 * f32.nbytes

    def test_random_allocation(self):
        arr = self.manager.allocate_random((100, 50))
        assert arr.std() > 0

    def test_sequential_allocation_growth(self):
        sizes = []
        for i in range(1, 6):
            self.manager.allocate((i * 100, i * 100))
            sizes.append(self.manager.get_total_mb())
        assert sizes == sorted(sizes)

    def test_gc_after_deallocation(self):
        self.manager.allocate((10000, 10000))
        self.manager.deallocate_all()
        gc.collect()
        assert len(self.manager.allocations) == 0

    @pytest.mark.asyncio
    async def test_async_allocation(self):
        async def alloc():
            await asyncio.sleep(0.001)
            return self.manager.allocate((100, 100))
        results = await asyncio.gather(*[alloc() for _ in range(10)])
        assert len(results) == 10

    @pytest.mark.parametrize("size_mb,expected_elements", [(0.004, 1000), (0.04, 10000), (0.4, 100000)])
    def test_size_to_elements(self, size_mb, expected_elements):
        assert int(size_mb * 1024 * 1024 / 4) == expected_elements
