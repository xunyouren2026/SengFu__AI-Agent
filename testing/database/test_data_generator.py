"""
TestDataGenerator - 测试数据库：数据生成器

模块路径: testing/database/test_data_generator.py
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
class GeneratedData:
    data_type: str
    num_samples: int
    schema: Dict[str, str]
    records: List[Dict[str, Any]]


class MockDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_numeric(self, num_samples: int, low: float = 0.0, high: float = 100.0) -> List[float]:
        return [self.rng.uniform(low, high) for _ in range(num_samples)]

    def generate_categorical(self, num_samples: int, categories: List[str]) -> List[str]:
        return [self.rng.choice(categories) for _ in range(num_samples)]

    def generate_text(self, num_samples: int, min_len: int = 5, max_len: int = 50) -> List[str]:
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "AI", "machine", "learning", "data", "model", "training"]
        results = []
        for _ in range(num_samples):
            length = self.rng.randint(min_len, max_len)
            text = " ".join(self.rng.choice(words) for _ in range(length))
            results.append(text)
        return results

    def generate_structured(self, num_samples: int, schema: Dict[str, str]) -> List[Dict]:
        records = []
        for _ in range(num_samples):
            record = {}
            for field_name, field_type in schema.items():
                if field_type == "int":
                    record[field_name] = self.rng.randint(0, 1000)
                elif field_type == "float":
                    record[field_name] = round(self.rng.uniform(0, 100), 2)
                elif field_type == "str":
                    record[field_name] = f"value_{self.rng.randint(0, 10000)}"
                elif field_type == "bool":
                    record[field_name] = self.rng.choice([True, False])
            records.append(record)
        return records

    def generate_time_series(self, num_points: int, trend: float = 0.1, noise: float = 1.0) -> List[float]:
        values = []
        current = 0.0
        for _ in range(num_points):
            current += trend + self.np_rng.normal(0, noise)
            values.append(round(current, 4))
        return values

    def generate_with_constraints(self, num_samples: int, constraints: Dict) -> List[Dict]:
        records = []
        for _ in range(num_samples):
            record = {}
            for field_name, constraint in constraints.items():
                if constraint["type"] == "range":
                    record[field_name] = round(self.rng.uniform(constraint["min"], constraint["max"]), 2)
                elif constraint["type"] == "enum":
                    record[field_name] = self.rng.choice(constraint["values"])
            records.append(record)
        return records


class TestDataGenerator:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.generator = MockDataGenerator(seed=42)
        self.test_data = []
        yield
        self.test_data.clear()

    def test_generate_numeric(self):
        data = self.generator.generate_numeric(100)
        assert len(data) == 100
        assert all(isinstance(v, float) for v in data)
        assert all(0 <= v <= 100 for v in data)

    def test_generate_categorical(self):
        data = self.generator.generate_categorical(50, ["cat", "dog", "bird"])
        assert len(data) == 50
        assert all(v in ["cat", "dog", "bird"] for v in data)

    def test_generate_text(self):
        data = self.generator.generate_text(20)
        assert len(data) == 20
        assert all(isinstance(v, str) and len(v) > 0 for v in data)

    def test_generate_structured(self):
        schema = {"id": "int", "name": "str", "score": "float", "active": "bool"}
        records = self.generator.generate_structured(10, schema)
        assert len(records) == 10
        assert all("id" in r and "name" in r and "score" in r and "active" in r for r in records)

    def test_generate_time_series(self):
        ts = self.generator.generate_time_series(100, trend=0.1, noise=0.5)
        assert len(ts) == 100
        assert all(isinstance(v, float) for v in ts)

    def test_time_series_trend(self):
        ts = self.generator.generate_time_series(1000, trend=1.0, noise=0.1)
        assert ts[-1] > ts[0]  # upward trend

    def test_generate_with_constraints(self):
        constraints = {
            "age": {"type": "range", "min": 18, "max": 65},
            "department": {"type": "enum", "values": ["engineering", "sales", "hr"]}
        }
        records = self.generator.generate_with_constraints(50, constraints)
        assert len(records) == 50
        assert all(18 <= r["age"] <= 65 for r in records)
        assert all(r["department"] in ["engineering", "sales", "hr"] for r in records)

    def test_deterministic_with_seed(self):
        gen1 = MockDataGenerator(seed=42)
        gen2 = MockDataGenerator(seed=42)
        d1 = gen1.generate_numeric(10)
        d2 = gen2.generate_numeric(10)
        assert d1 == d2

    def test_different_seeds(self):
        gen1 = MockDataGenerator(seed=1)
        gen2 = MockDataGenerator(seed=2)
        d1 = gen1.generate_numeric(10)
        d2 = gen2.generate_numeric(10)
        assert d1 != d2

    def test_large_data_generation(self):
        data = self.generator.generate_numeric(100000)
        assert len(data) == 100000

    def test_save_generated_data(self):
        records = self.generator.generate_structured(10, {"id": "int", "value": "float"})
        file_path = self.temp_dir / "generated.json"
        with open(file_path, "w") as f:
            json.dump(records, f)
        assert file_path.exists()
        with open(file_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 10

    @pytest.mark.parametrize("num_samples", [10, 100, 1000])
    def test_various_sizes(self, num_samples):
        data = self.generator.generate_numeric(num_samples)
        assert len(data) == num_samples
