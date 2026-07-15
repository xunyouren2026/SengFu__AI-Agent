"""
BenchmarkPlugin - Pytest基准测试插件

模块路径: testing/pytest_config/plugins/benchmark_plugin.py

提供pytest基准测试功能，支持性能测量、统计分析和回归检测。
"""

import os
import sys
import json
import time
import math
import statistics
import functools
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

import pytest


@dataclass
class BenchmarkResult:
    """单个基准测试结果"""
    name: str
    iterations: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    mean_time: float = 0.0
    median_time: float = 0.0
    stdev_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0
    ops_per_sec: float = 0.0
    times: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time": round(self.total_time, 6),
            "min_time": round(self.min_time, 6),
            "max_time": round(self.max_time, 6),
            "mean_time": round(self.mean_time, 6),
            "median_time": round(self.median_time, 6),
            "stdev_time": round(self.stdev_time, 6),
            "p95_time": round(self.p95_time, 6),
            "p99_time": round(self.p99_time, 6),
            "ops_per_sec": round(self.ops_per_sec, 2),
        }


@dataclass
class RegressionCheck:
    """回归检测结果"""
    benchmark_name: str
    current_mean: float
    baseline_mean: float
    threshold_pct: float
    regression_detected: bool = False
    change_pct: float = 0.0
    message: str = ""


class BenchmarkPlugin:
    """Pytest基准测试插件

    提供全面的基准测试功能:
        - 精确的执行时间测量
        - 统计分析（均值、中位数、标准差、百分位数）
        - 每秒操作数（ops/sec）计算
        - 性能回归检测
        - 基线管理和比较
        - JSON/终端格式的报告输出
        - 支持setup/warmup阶段
        - 支持异步函数基准测试
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._results: Dict[str, BenchmarkResult] = {}
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._regression_threshold: float = self.config.get("regression_threshold", 10.0)
        self._default_iterations: int = self.config.get("default_iterations", 100)
        self._default_warmup: int = self.config.get("default_warmup", 5)
        self._baseline_file: Optional[str] = self.config.get("baseline_file", None)
        self._lock = threading.Lock()
        self._timer_func = time.perf_counter

    def initialize(self) -> None:
        """初始化基准测试插件，加载基线数据"""
        if self._baseline_file and os.path.exists(self._baseline_file):
            self.load_baselines(self._baseline_file)
        self._initialized = True

    def benchmark(self, func: Optional[Callable] = None, name: str = "",
                  iterations: Optional[int] = None, warmup: Optional[int] = None,
                  setup: Optional[Callable] = None) -> Any:
        """基准测试装饰器或直接调用

        可以作为装饰器使用:
            @plugin.benchmark(iterations=1000)
            def test_func():
                ...

        也可以直接调用:
            result = plugin.benchmark(my_func, name="my_bench", iterations=100)

        Args:
            func: 被测函数
            name: 基准测试名称
            iterations: 迭代次数
            warmup: 预热次数
            setup: 每次迭代前的setup函数

        Returns:
            装饰器或BenchmarkResult
        """
        iters = iterations or self._default_iterations
        warm = warmup or self._default_warmup

        def decorator(fn: Callable) -> Callable:
            bench_name = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return self.run_benchmark(fn, bench_name, iters, warm, setup, *args, **kwargs)

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                return await self.run_async_benchmark(fn, bench_name, iters, warm, setup, *args, **kwargs)

            if asyncio.iscoroutinefunction(fn):
                return async_wrapper
            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def run_benchmark(self, func: Callable, name: str = "",
                      iterations: Optional[int] = None,
                      warmup: Optional[int] = None,
                      setup: Optional[Callable] = None,
                      *args, **kwargs) -> BenchmarkResult:
        """运行同步基准测试

        Args:
            func: 被测函数
            name: 基准测试名称
            iterations: 迭代次数
            warmup: 预热次数
            setup: 每次迭代前的setup函数
            *args, **kwargs: 传递给被测函数的参数

        Returns:
            BenchmarkResult基准测试结果
        """
        iters = iterations or self._default_iterations
        warm = warmup or self._default_warmup
        bench_name = name or func.__name__

        times = []
        for i in range(warm):
            if setup:
                setup()
            func(*args, **kwargs)

        for i in range(iters):
            if setup:
                setup()
            start = self._timer_func()
            func(*args, **kwargs)
            elapsed = self._timer_func() - start
            times.append(elapsed)

        result = self._compute_stats(bench_name, times)
        with self._lock:
            self._results[bench_name] = result
        return result

    async def run_async_benchmark(self, func: Callable, name: str = "",
                                   iterations: Optional[int] = None,
                                   warmup: Optional[int] = None,
                                   setup: Optional[Callable] = None,
                                   *args, **kwargs) -> BenchmarkResult:
        """运行异步基准测试

        Args:
            func: 异步被测函数
            name: 基准测试名称
            iterations: 迭代次数
            warmup: 预热次数
            setup: 每次迭代前的setup函数

        Returns:
            BenchmarkResult基准测试结果
        """
        iters = iterations or self._default_iterations
        warm = warmup or self._default_warmup
        bench_name = name or func.__name__

        times = []
        for i in range(warm):
            if setup:
                setup()
            await func(*args, **kwargs)

        for i in range(iters):
            if setup:
                setup()
            start = self._timer_func()
            await func(*args, **kwargs)
            elapsed = self._timer_func() - start
            times.append(elapsed)

        result = self._compute_stats(bench_name, times)
        with self._lock:
            self._results[bench_name] = result
        return result

    def _compute_stats(self, name: str, times: List[float]) -> BenchmarkResult:
        """计算基准测试统计数据

        Args:
            name: 基准测试名称
            times: 每次迭代的时间列表

        Returns:
            BenchmarkResult包含完整统计信息
        """
        if not times:
            return BenchmarkResult(name=name)
        sorted_times = sorted(times)
        n = len(sorted_times)
        total = sum(sorted_times)
        mean = total / n
        median = sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        stdev = statistics.stdev(sorted_times) if n > 1 else 0.0
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        p95 = sorted_times[min(p95_idx, n - 1)]
        p99 = sorted_times[min(p99_idx, n - 1)]
        ops = 1.0 / mean if mean > 0 else 0.0

        return BenchmarkResult(
            name=name,
            iterations=n,
            total_time=total,
            min_time=sorted_times[0],
            max_time=sorted_times[-1],
            mean_time=mean,
            median_time=median,
            stdev_time=stdev,
            p95_time=p95,
            p99_time=p99,
            ops_per_sec=ops,
            times=sorted_times,
        )

    def compare(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """比较两个基准测试的结果

        Args:
            name_a: 第一个基准测试名称
            name_b: 第二个基准测试名称

        Returns:
            包含比较结果的字典
        """
        a = self._results.get(name_a)
        b = self._results.get(name_b)
        if not a or not b:
            missing = name_a if not a else name_b
            raise ValueError(f"Benchmark '{missing}' not found")
        if a.mean_time == 0:
            speedup = float("inf")
        else:
            speedup = b.mean_time / a.mean_time
        return {
            "name_a": name_a,
            "name_b": name_b,
            "mean_a": round(a.mean_time, 6),
            "mean_b": round(b.mean_time, 6),
            "speedup": round(speedup, 2),
            "winner": name_a if speedup > 1 else name_b,
        }

    def check_regression(self, name: str, threshold_pct: Optional[float] = None) -> RegressionCheck:
        """检查基准测试是否存在性能回归

        Args:
            name: 基准测试名称
            threshold_pct: 回归阈值百分比，默认使用插件配置

        Returns:
            RegressionCheck回归检测结果
        """
        result = self._results.get(name)
        if not result:
            raise ValueError(f"Benchmark '{name}' not found in results")
        baseline = self._baselines.get(name)
        if not baseline:
            return RegressionCheck(
                benchmark_name=name,
                current_mean=result.mean_time,
                baseline_mean=0.0,
                threshold_pct=threshold_pct or self._regression_threshold,
                regression_detected=False,
                message="No baseline available for comparison",
            )
        baseline_mean = baseline.get("mean_time", 0.0)
        threshold = threshold_pct or self._regression_threshold
        if baseline_mean == 0:
            change_pct = 0.0
        else:
            change_pct = ((result.mean_time - baseline_mean) / baseline_mean) * 100.0
        is_regression = change_pct > threshold
        return RegressionCheck(
            benchmark_name=name,
            current_mean=result.mean_time,
            baseline_mean=baseline_mean,
            threshold_pct=threshold,
            regression_detected=is_regression,
            change_pct=change_pct,
            message=f"Performance {'regression' if is_regression else 'improvement'}: {change_pct:+.1f}%",
        )

    def save_baselines(self, filepath: str) -> None:
        """保存当前结果作为基线数据

        Args:
            filepath: 基线文件路径
        """
        data = {}
        for name, result in self._results.items():
            data[name] = result.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_baselines(self, filepath: str) -> None:
        """从文件加载基线数据

        Args:
            filepath: 基线文件路径
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self._baselines = json.load(f)
        except (IOError, json.JSONDecodeError):
            self._baselines = {}

    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """获取指定基准测试的结果

        Args:
            name: 基准测试名称

        Returns:
            BenchmarkResult或None
        """
        return self._results.get(name)

    def get_all_results(self) -> Dict[str, BenchmarkResult]:
        """获取所有基准测试结果

        Returns:
            基准测试名称到结果的映射字典
        """
        return dict(self._results)

    def format_report(self, sort_by: str = "mean_time") -> str:
        """格式化终端输出的基准测试报告

        Args:
            sort_by: 排序字段 (mean_time/median_time/ops_per_sec/name)

        Returns:
            格式化的报告字符串
        """
        if not self._results:
            return "No benchmark results available."
        reverse = sort_by in ("ops_per_sec",)
        sorted_results = sorted(
            self._results.values(),
            key=lambda r: getattr(r, sort_by, r.mean_time),
            reverse=reverse,
        )
        lines = [
            "=" * 80,
            "BENCHMARK REPORT",
            "=" * 80,
            f"{'Name':30s} {'Iterations':>10s} {'Mean (s)':>12s} {'Median (s)':>12s} {'Ops/s':>12s}",
            "-" * 80,
        ]
        for r in sorted_results:
            lines.append(
                f"{r.name:30s} {r.iterations:10d} {r.mean_time:12.6f} "
                f"{r.median_time:12.6f} {r.ops_per_sec:12.2f}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)

    def save_json_report(self, filepath: str) -> None:
        """保存JSON格式的基准测试报告

        Args:
            filepath: 输出文件路径
        """
        data = {
            "timestamp": time.time(),
            "benchmarks": {},
        }
        for name, result in self._results.items():
            data["benchmarks"][name] = result.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def reset(self) -> None:
        """重置所有基准测试结果"""
        with self._lock:
            self._results.clear()

    def set_iterations(self, iterations: int) -> None:
        """设置默认迭代次数

        Args:
            iterations: 迭代次数
        """
        self._default_iterations = max(1, iterations)

    def set_regression_threshold(self, threshold_pct: float) -> None:
        """设置回归检测阈值

        Args:
            threshold_pct: 阈值百分比
        """
        self._regression_threshold = max(0.0, threshold_pct)

    def get_summary(self) -> Dict[str, Any]:
        """获取基准测试摘要

        Returns:
            包含摘要统计信息的字典
        """
        if not self._results:
            return {"total_benchmarks": 0}
        total_time = sum(r.total_time for r in self._results.values())
        fastest = min(self._results.values(), key=lambda r: r.mean_time)
        slowest = max(self._results.values(), key=lambda r: r.mean_time)
        return {
            "total_benchmarks": len(self._results),
            "total_time": round(total_time, 3),
            "fastest": {"name": fastest.name, "mean_time": round(fastest.mean_time, 6)},
            "slowest": {"name": slowest.name, "mean_time": round(slowest.mean_time, 6)},
        }
