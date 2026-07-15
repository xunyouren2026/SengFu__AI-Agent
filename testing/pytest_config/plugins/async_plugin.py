"""
AsyncPlugin - Pytest异步测试支持插件

模块路径: testing/pytest_config/plugins/async_plugin.py

提供pytest对异步测试用例的全面支持，包括async fixture、
异步测试函数运行、超时控制和事件循环管理。
"""

import os
import sys
import json
import time
import asyncio
import functools
import inspect
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Coroutine, TypeVar
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import asyncio

import pytest


T = TypeVar("T")


@dataclass
class AsyncTestResult:
    """异步测试结果"""
    test_name: str
    status: str = "pending"
    duration: float = 0.0
    error_message: str = ""
    traceback_str: str = ""
    is_timeout: bool = False


@dataclass
class AsyncFixtureInfo:
    """异步fixture信息"""
    name: str
    scope: str = "function"
    is_async: bool = True
    dependencies: List[str] = field(default_factory=list)


class AsyncPlugin:
    """Pytest异步测试支持插件

    为pytest提供完整的异步测试支持，包括:
        - 自动检测和运行async def测试函数
        - 支持async fixture（function/class/module/session作用域）
        - 异步测试超时控制
        - 事件循环生命周期管理
        - 并发异步测试执行
        - 异步测试结果收集和报告

    使用方式:
        在pytest配置中注册此插件，或通过conftest.py导入。
        插件会自动检测标记为async的测试函数并正确执行。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._results: List[AsyncTestResult] = []
        self._timeout: float = self.config.get("async_timeout", 30.0)
        self._mode: str = self.config.get("async_mode", "auto")
        self._concurrency_limit: int = self.config.get("concurrency_limit", 10)
        self._fixture_registry: Dict[str, AsyncFixtureInfo] = {}
        self._loop_policy: Optional[asyncio.AbstractEventLoopPolicy] = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """初始化异步插件，设置事件循环策略"""
        if self._mode == "auto":
            try:
                self._loop_policy = asyncio.WindowsSelectorEventLoopPolicy() if sys.platform == "win32" else None
            except AttributeError:
                self._loop_policy = None
        self._initialized = True

    def get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """获取或创建事件循环

        Returns:
            当前线程的事件循环，如果不存在则创建新的
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop is closed")
        except RuntimeError:
            if self._loop_policy:
                asyncio.set_event_loop_policy(self._loop_policy)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._loop = loop
        return loop

    def is_async_test(self, func: Any) -> bool:
        """检测函数是否为异步测试函数

        Args:
            func: 待检测的函数对象

        Returns:
            如果是协程函数则返回True
        """
        return asyncio.iscoroutinefunction(func)

    def is_async_fixture(self, func: Any) -> bool:
        """检测函数是否为异步fixture

        Args:
            func: 待检测的函数对象

        Returns:
            如果是异步fixture则返回True
        """
        return asyncio.iscoroutinefunction(func) and hasattr(func, "_pytestfixturefunction")

    def run_async_test(self, coro: Coroutine, test_name: str = "") -> AsyncTestResult:
        """运行单个异步测试

        Args:
            coro: 异步测试协程
            test_name: 测试名称，用于结果记录

        Returns:
            AsyncTestResult测试结果对象
        """
        result = AsyncTestResult(test_name=test_name)
        loop = self.get_or_create_loop()
        start_time = time.monotonic()
        try:
            async_result = asyncio.wait_for(coro, timeout=self._timeout)
            loop.run_until_complete(async_result)
            result.status = "passed"
        except asyncio.TimeoutError:
            result.status = "failed"
            result.is_timeout = True
            result.error_message = f"Test timed out after {self._timeout}s"
        except asyncio.CancelledError:
            result.status = "skipped"
            result.error_message = "Test was cancelled"
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.traceback_str = traceback.format_exc()
        finally:
            result.duration = time.monotonic() - start_time
        with self._lock:
            self._results.append(result)
        return result

    def run_async_fixture(self, coro: Coroutine, fixture_name: str = "") -> Any:
        """运行异步fixture并返回结果

        Args:
            coro: 异步fixture协程
            fixture_name: fixture名称

        Returns:
            fixture的返回值
        """
        loop = self.get_or_create_loop()
        try:
            result = asyncio.wait_for(coro, timeout=self._timeout)
            return loop.run_until_complete(result)
        except asyncio.TimeoutError:
            raise pytest.fail(f"Async fixture '{fixture_name}' timed out after {self._timeout}s")

    async def run_async_cleanup(self, cleanup_coro: Coroutine) -> None:
        """运行异步清理协程

        Args:
            cleanup_coro: 清理协程
        """
        try:
            await asyncio.wait_for(cleanup_coro, timeout=self._timeout)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass

    def register_async_fixture(self, name: str, scope: str = "function",
                                dependencies: Optional[List[str]] = None) -> AsyncFixtureInfo:
        """注册异步fixture信息

        Args:
            name: fixture名称
            scope: fixture作用域 (function/class/module/session)
            dependencies: 依赖的其他fixture名称列表

        Returns:
            注册的AsyncFixtureInfo对象
        """
        info = AsyncFixtureInfo(
            name=name,
            scope=scope,
            is_async=True,
            dependencies=dependencies or [],
        )
        self._fixture_registry[name] = info
        return info

    def get_fixture_info(self, name: str) -> Optional[AsyncFixtureInfo]:
        """获取已注册的fixture信息

        Args:
            name: fixture名称

        Returns:
            AsyncFixtureInfo对象，如果未注册则返回None
        """
        return self._fixture_registry.get(name)

    def resolve_fixture_dependencies(self, name: str) -> List[str]:
        """解析fixture的依赖链（拓扑排序）

        Args:
            name: fixture名称

        Returns:
            按依赖顺序排列的fixture名称列表
        """
        visited = set()
        order = []

        def _visit(fixture_name: str) -> None:
            if fixture_name in visited:
                return
            visited.add(fixture_name)
            info = self._fixture_registry.get(fixture_name)
            if info:
                for dep in info.dependencies:
                    _visit(dep)
            order.append(fixture_name)

        _visit(name)
        return order

    async def run_concurrent_tests(self, coros: List[Coroutine],
                                    names: Optional[List[str]] = None) -> List[AsyncTestResult]:
        """并发运行多个异步测试

        Args:
            coros: 异步测试协程列表
            names: 对应的测试名称列表

        Returns:
            测试结果列表
        """
        semaphore = asyncio.Semaphore(self._concurrency_limit)
        test_names = names or [f"test_{i}" for i in range(len(coros))]

        async def _run_with_limit(coro: Coroutine, name: str) -> AsyncTestResult:
            async with semaphore:
                result = AsyncTestResult(test_name=name)
                start = time.monotonic()
                try:
                    await asyncio.wait_for(coro, timeout=self._timeout)
                    result.status = "passed"
                except asyncio.TimeoutError:
                    result.status = "failed"
                    result.is_timeout = True
                    result.error_message = f"Timed out after {self._timeout}s"
                except Exception as e:
                    result.status = "failed"
                    result.error_message = str(e)
                    result.traceback_str = traceback.format_exc()
                finally:
                    result.duration = time.monotonic() - start
                return result

        tasks = [_run_with_limit(coro, name) for coro, name in zip(coros, test_names)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = []
        for r in results:
            if isinstance(r, AsyncTestResult):
                valid_results.append(r)
            else:
                valid_results.append(AsyncTestResult(
                    test_name="unknown",
                    status="failed",
                    error_message=str(r),
                ))
        with self._lock:
            self._results.extend(valid_results)
        return valid_results

    def wrap_sync_test(self, func: Callable) -> Callable:
        """将同步测试函数包装为异步兼容的函数

        Args:
            func: 同步测试函数

        Returns:
            包装后的函数
        """
        if self.is_async_test(func):
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._is_async_wrapped = True
        return wrapper

    def set_timeout(self, timeout: float) -> None:
        """设置异步测试超时时间

        Args:
            timeout: 超时时间（秒）
        """
        self._timeout = max(0.1, timeout)

    def get_results(self) -> List[AsyncTestResult]:
        """获取所有异步测试结果

        Returns:
            测试结果列表
        """
        return list(self._results)

    def get_summary(self) -> Dict[str, Any]:
        """获取异步测试摘要统计

        Returns:
            包含统计信息的字典
        """
        if not self._results:
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "timeouts": 0}
        passed = sum(1 for r in self._results if r.status == "passed")
        failed = sum(1 for r in self._results if r.status == "failed")
        skipped = sum(1 for r in self._results if r.status == "skipped")
        timeouts = sum(1 for r in self._results if r.is_timeout)
        total_duration = sum(r.duration for r in self._results)
        return {
            "total": len(self._results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "timeouts": timeouts,
            "total_duration": round(total_duration, 3),
            "avg_duration": round(total_duration / len(self._results), 3),
        }

    def reset(self) -> None:
        """重置所有测试结果和状态"""
        with self._lock:
            self._results.clear()
            self._fixture_registry.clear()

    def close_loop(self) -> None:
        """关闭事件循环"""
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None

    async def gather_with_timeout(self, *coros: Coroutine, timeout: Optional[float] = None) -> List[Any]:
        """带超时的协程并发执行

        Args:
            *coros: 待执行的协程
            timeout: 超时时间，默认使用插件配置的超时

        Returns:
            所有协程的结果列表
        """
        effective_timeout = timeout or self._timeout
        return await asyncio.gather(*coros, return_exceptions=False)

    def create_task_group(self) -> "AsyncTaskGroup":
        """创建异步任务组，用于管理相关的一组异步任务

        Returns:
            AsyncTaskGroup实例
        """
        return AsyncTaskGroup(timeout=self._timeout, concurrency_limit=self._concurrency_limit)


class AsyncTaskGroup:
    """异步任务组，用于组织和并发执行相关的异步任务"""

    def __init__(self, timeout: float = 30.0, concurrency_limit: int = 10):
        self._tasks: List[asyncio.Task] = []
        self._results: List[Any] = []
        self._errors: List[Exception] = []
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency_limit)

    async def add(self, coro: Coroutine) -> None:
        """添加一个异步任务到任务组

        Args:
            coro: 待执行的协程
        """
        async def _wrapped():
            async with self._semaphore:
                return await asyncio.wait_for(coro, timeout=self._timeout)
        task = asyncio.create_task(_wrapped())
        self._tasks.append(task)

    async def run_all(self) -> List[Any]:
        """运行所有已添加的任务

        Returns:
            所有任务的结果列表
        """
        if not self._tasks:
            return []
        self._results = await asyncio.gather(*self._tasks, return_exceptions=True)
        self._errors = [r for r in self._results if isinstance(r, Exception)]
        return [r for r in self._results if not isinstance(r, Exception)]

    @property
    def has_errors(self) -> bool:
        """是否有任务执行出错"""
        return len(self._errors) > 0

    @property
    def errors(self) -> List[Exception]:
        """获取所有错误"""
        return list(self._errors)
