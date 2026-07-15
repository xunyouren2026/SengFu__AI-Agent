"""
TestInferencePipeline - 集成测试：推理管道
模块路径: testing/integration/test_inference_pipeline.py
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
class InferenceRequest:
    request_id: str
    input_data: np.ndarray
    model_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceResponse:
    request_id: str
    output: np.ndarray
    model_name: str
    latency_ms: float
    confidence: float = 1.0

class MockModelServer:
    def __init__(self):
        self.request_count = 0
        self.total_latency = 0.0

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        start = time.time()
        self.request_count += 1
        await asyncio.sleep(0.01)
        output = np.random.randn(*request.input_data.shape[:-1], 10).astype(np.float32)
        latency = (time.time() - start) * 1000
        self.total_latency += latency
        return InferenceResponse(request_id=request.request_id, output=output,
                                 model_name=request.model_name, latency_ms=latency,
                                 confidence=random.uniform(0.7, 0.99))

    def get_stats(self):
        return {"total_requests": self.request_count,
                "avg_latency_ms": self.total_latency / max(self.request_count, 1)}

class MockPreprocessor:
    def preprocess(self, data):
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8
        return (data - mean) / std

class MockPostprocessor:
    def postprocess(self, output):
        probs = self._softmax(output)
        predicted = int(np.argmax(probs))
        return {"predicted_class": predicted, "confidence": float(probs[predicted])}

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class TestInferencePipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.server = MockModelServer()
        self.preprocessor = MockPreprocessor()
        self.postprocessor = MockPostprocessor()
        self.test_data = []
        yield
        self.test_data.clear()

    @pytest.mark.asyncio
    async def test_single_inference(self):
        data = np.random.randn(1, 10).astype(np.float32)
        req = InferenceRequest("r1", data, "model1")
        resp = await self.server.predict(req)
        assert resp.latency_ms > 0

    @pytest.mark.asyncio
    async def test_batch_inference(self):
        reqs = [InferenceRequest(f"br{i}", np.random.randn(1, 10).astype(np.float32), "model1") for i in range(10)]
        responses = await asyncio.gather(*[self.server.predict(r) for r in reqs])
        assert len(responses) == 10 and all(r.latency_ms > 0 for r in responses)

    def test_preprocessing(self):
        data = np.random.randn(100, 10).astype(np.float32)
        processed = self.preprocessor.preprocess(data)
        assert abs(processed.mean()) < 1e-5

    def test_postprocessing(self):
        output = np.random.randn(10).astype(np.float32)
        result = self.postprocessor.postprocess(output)
        assert "predicted_class" in result and 0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_full_inference_pipeline(self):
        raw = np.random.randn(1, 10).astype(np.float32)
        preprocessed = self.preprocessor.preprocess(raw)
        resp = await self.server.predict(InferenceRequest("full1", preprocessed, "model1"))
        result = self.postprocessor.postprocess(resp.output.flatten())
        assert "predicted_class" in result

    @pytest.mark.asyncio
    async def test_server_stats_after_requests(self):
        for i in range(5):
            await self.server.predict(InferenceRequest(f"stat_{i}", np.random.randn(1, 10).astype(np.float32), "model1"))
        assert self.server.get_stats()["total_requests"] == 5

    @pytest.mark.parametrize("shape", [(1, 10), (1, 50), (1, 100)])
    @pytest.mark.asyncio
    async def test_various_input_shapes(self, shape):
        resp = await self.server.predict(InferenceRequest("shape", np.random.randn(*shape).astype(np.float32), "model1"))
        assert resp.output is not None
