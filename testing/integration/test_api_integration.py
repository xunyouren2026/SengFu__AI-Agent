"""
TestApiIntegration - 集成测试：API集成
模块路径: testing/integration/test_api_integration.py
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
class APIRequest:
    method: str
    endpoint: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict] = None
    params: Dict[str, str] = field(default_factory=dict)

@dataclass
class APIResponse:
    status_code: int
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)
    response_time: float = 0.0

class MockAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.request_history: List[APIRequest] = []
        self._requests_count = 0

    async def request(self, req: APIRequest) -> APIResponse:
        self.request_history.append(req)
        self._requests_count += 1
        await asyncio.sleep(0.01)
        return APIResponse(status_code=200, body={"message": "ok", "data": req.body},
                           headers={"content-type": "application/json"}, response_time=0.05)

    async def get(self, endpoint: str, params: Dict = None) -> APIResponse:
        return await self.request(APIRequest(method="GET", endpoint=endpoint, params=params or {}))

    async def post(self, endpoint: str, body: Dict = None) -> APIResponse:
        return await self.request(APIRequest(method="POST", endpoint=endpoint, body=body))

    async def put(self, endpoint: str, body: Dict = None) -> APIResponse:
        return await self.request(APIRequest(method="PUT", endpoint=endpoint, body=body))

    async def delete(self, endpoint: str) -> APIResponse:
        return await self.request(APIRequest(method="DELETE", endpoint=endpoint))

class MockRateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.window_seconds]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

class TestApiIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.client = MockAPIClient()
        self.rate_limiter = MockRateLimiter()
        self.test_data = []
        yield
        self.test_data.clear()

    @pytest.mark.asyncio
    async def test_get_request(self):
        resp = await self.client.get("/api/users")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_post_request(self):
        resp = await self.client.post("/api/users", body={"name": "test"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_put_request(self):
        resp = await self.client.put("/api/users/1", body={"name": "updated"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_request(self):
        resp = await self.client.delete("/api/users/1")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_request_history_tracking(self):
        await self.client.get("/api/a")
        await self.client.post("/api/b")
        assert len(self.client.request_history) == 2
        assert self.client.request_history[0].method == "GET"

    def test_rate_limiter_blocks_excess(self):
        limiter = MockRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.allow()
        assert not limiter.allow()

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        broken = MockAPIClient()
        broken.request = AsyncMock(return_value=APIResponse(status_code=500, body={"error": "Server Error"}))
        resp = await broken.get("/api/error")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self):
        tasks = [self.client.get(f"/api/item/{i}") for i in range(10)]
        responses = await asyncio.gather(*tasks)
        assert len(responses) == 10 and all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_batch_operations(self):
        batch = [self.client.post("/api/items", body={"id": i}) for i in range(5)]
        results = await asyncio.gather(*batch)
        assert len(results) == 5

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
    @pytest.mark.asyncio
    async def test_all_http_methods(self, method):
        if method == "GET": resp = await self.client.get("/test")
        elif method == "POST": resp = await self.client.post("/test")
        elif method == "PUT": resp = await self.client.put("/test")
        else: resp = await self.client.delete("/test")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        attempt = 0
        async def flaky():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                return APIResponse(status_code=503, body={"error": "unavailable"})
            return APIResponse(status_code=200, body={"ok": True})
        for _ in range(3):
            resp = await flaky()
            if resp.status_code == 200: break
        assert resp.status_code == 200 and attempt == 3
