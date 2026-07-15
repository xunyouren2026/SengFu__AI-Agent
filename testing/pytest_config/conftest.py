"""
Conftest - Pytest配置与夹具

模块路径: testing/pytest_config/conftest.py

提供pytest全局fixtures、配置钩子和测试环境管理。
"""

import os
import sys
import json
import time
import random
import tempfile
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Generator
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

import pytest


# ============================================================
# 日志配置
# ============================================================

def configure_test_logging(level: str = "DEBUG", log_file: Optional[str] = None) -> logging.Logger:
    """配置测试日志系统

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 可选的日志文件路径

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger("agi_test")
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# ============================================================
# 测试环境管理
# ============================================================

class TestEnvironment:
    """测试环境管理器，负责创建和清理测试所需的临时资源"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="agi_test_")
        self._created_dirs: List[str] = []
        self._created_files: Dict[str, str] = {}
        self._cleanup_callbacks: List[Callable] = []

    def create_temp_dir(self, prefix: str = "test_") -> str:
        """创建临时目录

        Args:
            prefix: 目录名前缀

        Returns:
            临时目录的绝对路径
        """
        dir_path = tempfile.mkdtemp(prefix=prefix, dir=self.base_dir)
        self._created_dirs.append(dir_path)
        return dir_path

    def create_temp_file(self, filename: str, content: str = "",
                          directory: Optional[str] = None) -> str:
        """创建临时文件

        Args:
            filename: 文件名
            content: 文件内容
            directory: 所在目录，默认为base_dir

        Returns:
            临时文件的绝对路径
        """
        dir_path = directory or self.base_dir
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        self._created_files[file_path] = content
        return file_path

    def create_json_file(self, filename: str, data: Any,
                          directory: Optional[str] = None) -> str:
        """创建JSON临时文件

        Args:
            filename: 文件名
            data: 要序列化的数据
            directory: 所在目录

        Returns:
            文件路径
        """
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return self.create_temp_file(filename, content, directory)

    def register_cleanup(self, callback: Callable) -> None:
        """注册清理回调函数

        Args:
            callback: 清理时调用的函数
        """
        self._cleanup_callbacks.append(callback)

    def cleanup(self) -> None:
        """清理所有创建的临时资源"""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception:
                pass
        for file_path in self._created_files:
            try:
                os.remove(file_path)
            except OSError:
                pass
        for dir_path in reversed(self._created_dirs):
            try:
                shutil.rmtree(dir_path, ignore_errors=True)
            except OSError:
                pass
        self._created_dirs.clear()
        self._created_files.clear()
        self._cleanup_callbacks.clear()

    @property
    def base_path(self) -> str:
        """获取基础目录路径"""
        return self.base_dir


# ============================================================
# Pytest Fixtures
# ============================================================

@pytest.fixture(scope="session")
def test_logger():
    """会话级别的测试日志器"""
    return configure_test_logging(level="INFO")


@pytest.fixture(scope="session")
def test_env():
    """会话级别的测试环境管理器"""
    env = TestEnvironment()
    yield env
    env.cleanup()


@pytest.fixture(scope="function")
def temp_dir(test_env):
    """函数级别的临时目录，每个测试用例独立"""
    dir_path = test_env.create_temp_dir()
    yield dir_path
    try:
        shutil.rmtree(dir_path, ignore_errors=True)
    except OSError:
        pass


@pytest.fixture(scope="function")
def temp_file(test_env):
    """函数级别的临时文件工厂"""
    def _create(filename: str = "test.txt", content: str = "") -> str:
        return test_env.create_temp_file(filename, content)
    return _create


@pytest.fixture(scope="function")
def mock_config():
    """提供模拟的配置字典"""
    return {
        "debug": False,
        "log_level": "INFO",
        "max_retries": 3,
        "timeout": 30.0,
        "api_base_url": "http://localhost:8080",
        "database_url": "sqlite:///:memory:",
        "cache_enabled": True,
        "cache_ttl": 300,
    }


@pytest.fixture(scope="function")
def sample_data():
    """提供测试用的样本数据"""
    return {
        "users": [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"},
            {"id": 3, "name": "Charlie", "role": "user"},
        ],
        "products": [
            {"id": 101, "name": "Widget", "price": 9.99},
            {"id": 102, "name": "Gadget", "price": 24.99},
        ],
        "metadata": {
            "version": "1.0.0",
            "total_items": 5,
        },
    }


@pytest.fixture(scope="session")
def event_loop():
    """会话级别的事件循环，用于异步测试"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def async_client(event_loop):
    """异步测试客户端"""
    class AsyncTestClient:
        def __init__(self, loop: asyncio.AbstractEventLoop):
            self._loop = loop
            self._requests: List[Dict[str, Any]] = []
            self._responses: List[Any] = []

        def add_response(self, response: Any) -> None:
            self._responses.append(response)

        async def request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
            entry = {"method": method, "url": url, "kwargs": kwargs}
            self._requests.append(entry)
            if self._responses:
                return self._responses.pop(0)
            return {"status": 200, "data": None}

        @property
        def request_history(self) -> List[Dict[str, Any]]:
            return list(self._requests)

        def clear(self) -> None:
            self._requests.clear()
            self._responses.clear()

    return AsyncTestClient(event_loop)


@pytest.fixture(scope="function")
def mock_response():
    """模拟HTTP响应工厂"""
    def _create(status_code: int = 200, data: Any = None,
                headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "status_code": status_code,
            "data": data,
            "headers": headers or {"Content-Type": "application/json"},
            "elapsed": random.uniform(0.01, 0.1),
        }
    return _create


# ============================================================
# Pytest Hooks
# ============================================================

def pytest_configure(config):
    """pytest配置钩子：注册自定义标记"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "async_test: marks tests as async tests")


def pytest_collection_modifyitems(config, items):
    """修改测试收集：自动为测试添加标记"""
    for item in items:
        if "async" in item.name or "async" in getattr(item, "fixturenames", []):
            item.add_marker(pytest.mark.async_test)
        if item.fspath and "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif item.fspath and "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


def pytest_runtest_setup(item):
    """测试运行前钩子：检查标记条件"""
    if item.get_closest_marker("slow") and not item.config.getoption("--run-slow", False):
        pytest.skip("Skipping slow test (use --run-slow to include)")


def pytest_sessionstart(session):
    """会话开始钩子"""
    session.config._test_start_time = time.time()


def pytest_sessionfinish(session, exitstatus):
    """会话结束钩子：打印测试摘要"""
    start_time = getattr(session.config, "_test_start_time", time.time())
    duration = time.time() - start_time
    print(f"\n[Conftest] Test session duration: {duration:.2f}s")


# ============================================================
# 测试辅助工具
# ============================================================

class AssertionHelper:
    """测试断言辅助工具"""

    @staticmethod
    def assert_dict_contains(actual: Dict, expected: Dict) -> None:
        """断言字典包含预期的键值对

        Args:
            actual: 实际字典
            expected: 预期包含的键值对
        """
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in {list(actual.keys())}"
            assert actual[key] == value, f"Key '{key}': expected {value}, got {actual[key]}"

    @staticmethod
    def assert_list_length(lst: List, expected_length: int) -> None:
        """断言列表长度"""
        assert len(lst) == expected_length, f"Expected length {expected_length}, got {len(lst)}"

    @staticmethod
    def assert_in_range(value: float, min_val: float, max_val: float) -> None:
        """断言数值在范围内"""
        assert min_val <= value <= max_val, f"Value {value} not in range [{min_val}, {max_val}]"

    @staticmethod
    def assert_valid_json(data: str) -> Dict:
        """断言字符串是有效的JSON并返回解析结果"""
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSON: {e}")


class TestDataBuilder:
    """测试数据构建器，用于快速生成各种测试数据"""

    @staticmethod
    def build_user(user_id: Optional[int] = None, **overrides) -> Dict[str, Any]:
        """构建用户数据"""
        data = {
            "id": user_id or random.randint(1, 10000),
            "name": f"user_{random.randint(1000, 9999)}",
            "email": f"user{random.randint(1000, 9999)}@test.com",
            "role": "user",
            "created_at": time.time(),
            "is_active": True,
        }
        data.update(overrides)
        return data

    @staticmethod
    def build_request(method: str = "GET", path: str = "/api/test",
                       headers: Optional[Dict] = None, body: Any = None) -> Dict[str, Any]:
        """构建HTTP请求数据"""
        return {
            "method": method,
            "path": path,
            "headers": headers or {"Content-Type": "application/json"},
            "body": body,
            "timestamp": time.time(),
            "request_id": hashlib.md5(f"{method}{path}{time.time()}".encode()).hexdigest()[:16],
        }

    @staticmethod
    def build_response(status_code: int = 200, data: Any = None,
                       error: Optional[str] = None) -> Dict[str, Any]:
        """构建HTTP响应数据"""
        response = {
            "status_code": status_code,
            "data": data,
            "error": error,
            "timestamp": time.time(),
        }
        if error:
            response["success"] = False
        else:
            response["success"] = status_code < 400
        return response

    @staticmethod
    def build_batch(items: int = 10, builder: Optional[Callable] = None) -> List[Any]:
        """批量构建测试数据"""
        if builder is None:
            builder = TestDataBuilder.build_user
        return [builder() for _ in range(items)]


class Timer:
    """精确计时器，用于测试性能测量"""

    def __init__(self):
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._laps: List[float] = []

    def start(self) -> "Timer":
        """开始计时"""
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        """停止计时"""
        self._end_time = time.perf_counter()
        return self

    def lap(self) -> float:
        """记录一个分段时间"""
        lap_time = time.perf_counter()
        if self._laps:
            elapsed = lap_time - self._laps[-1]
        else:
            elapsed = lap_time - self._start_time
        self._laps.append(lap_time)
        return elapsed

    @property
    def elapsed(self) -> float:
        """获取总耗时"""
        if self._end_time > 0:
            return self._end_time - self._start_time
        return time.perf_counter() - self._start_time

    @property
    def lap_times(self) -> List[float]:
        """获取所有分段时间"""
        times = []
        for i, lap_time in enumerate(self._laps):
            if i == 0:
                times.append(lap_time - self._start_time)
            else:
                times.append(lap_time - self._laps[i - 1])
        return times

    def reset(self) -> None:
        """重置计时器"""
        self._start_time = 0.0
        self._end_time = 0.0
        self._laps.clear()


# ============================================================
# 参数化测试数据
# ============================================================

def generate_id_combinations() -> List[pytest.param]:
    """生成各种ID格式的参数化测试数据"""
    return [
        pytest.param(1, id="positive_int"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative_int"),
        pytest.param("abc-123", id="string_id"),
        pytest.param("00000000-0000-0000-0000-000000000000", id="uuid"),
        pytest.param(None, id="none"),
    ]


def generate_edge_case_inputs() -> List[pytest.param]:
    """生成边界情况测试输入"""
    return [
        pytest.param("", id="empty_string"),
        pytest.param(" ", id="whitespace"),
        pytest.param(None, id="null"),
        pytest.param([], id="empty_list"),
        pytest.param({}, id="empty_dict"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(True, id="boolean_true"),
        pytest.param(False, id="boolean_false"),
    ]


def generate_payload_sizes() -> List[pytest.param]:
    """生成不同大小的测试负载"""
    return [
        pytest.param(1, id="1_byte"),
        pytest.param(1024, id="1kb"),
        pytest.param(1024 * 1024, id="1mb"),
        pytest.param(10 * 1024 * 1024, id="10mb"),
    ]
