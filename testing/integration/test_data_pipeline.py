"""
TestDataPipeline - 集成测试：数据管道
模块路径: testing/integration/test_data_pipeline.py
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
class DataRecord:
    id: str
    features: Dict[str, Any]
    label: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class MockDataSource:
    def __init__(self, num_records=100):
        self.records = [DataRecord(id=f"rec_{i}", features={"value": random.random(), "category": random.choice(["A", "B", "C"])},
                                    label=random.choice(["pos", "neg"])) for i in range(num_records)]

    def fetch(self, limit=None):
        return self.records[:limit] if limit else self.records

    def fetch_batch(self, batch_size, offset=0):
        return self.records[offset:offset + batch_size]

class MockTransformer:
    def transform(self, records):
        for r in records:
            r.features["value_normalized"] = r.features["value"] / 100.0
            r.features["category_encoded"] = {"A": 0, "B": 1, "C": 2}.get(r.features["category"], -1)
        return records

    def filter(self, records, predicate):
        return [r for r in records if predicate(r)]

    def aggregate(self, records, key):
        groups: Dict[str, List] = {}
        for r in records:
            groups.setdefault(r.features.get(key, "unknown"), []).append(r)
        return groups

class MockDataSink:
    def __init__(self):
        self.stored = []

    def store(self, records):
        self.stored.extend(records)
        return len(records)

class TestDataPipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.source = MockDataSource()
        self.transformer = MockTransformer()
        self.sink = MockDataSink()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_fetch_all_records(self):
        assert len(self.source.fetch()) == 100

    def test_fetch_limited_records(self):
        assert len(self.source.fetch(limit=10)) == 10

    def test_fetch_batch_with_offset(self):
        batch = self.source.fetch_batch(10, 50)
        assert len(batch) == 10 and batch[0].id == "rec_50"

    def test_transform_records(self):
        records = self.source.fetch(limit=5)
        transformed = self.transformer.transform(records)
        for r in transformed:
            assert "value_normalized" in r.features

    def test_filter_records(self):
        records = self.source.fetch(limit=20)
        filtered = self.transformer.filter(records, lambda r: r.features["category"] == "A")
        assert all(r.features["category"] == "A" for r in filtered)

    def test_aggregate_records(self):
        records = self.source.fetch(limit=30)
        groups = self.transformer.aggregate(records, "category")
        total = sum(len(v) for v in groups.values())
        assert total == 30

    def test_full_pipeline(self):
        records = self.source.fetch(limit=50)
        transformed = self.transformer.transform(records)
        filtered = self.transformer.filter(transformed, lambda r: r.features["value"] > 0.3)
        count = self.sink.store(filtered)
        assert count > 0

    @pytest.mark.asyncio
    async def test_async_pipeline(self):
        async def process_batch(idx):
            await asyncio.sleep(0.01)
            return self.transformer.transform(self.source.fetch_batch(10, idx * 10))
        batches = await asyncio.gather(*[process_batch(i) for i in range(5)])
        assert sum(len(b) for b in batches) == 50

    @pytest.mark.parametrize("category", ["A", "B", "C"])
    def test_category_filtering(self, category):
        records = self.source.fetch(limit=50)
        filtered = self.transformer.filter(records, lambda r: r.features["category"] == category)
        assert all(r.features["category"] == category for r in filtered)

    def test_empty_pipeline_handling(self):
        empty = MockDataSource(num_records=0)
        assert len(empty.fetch()) == 0
