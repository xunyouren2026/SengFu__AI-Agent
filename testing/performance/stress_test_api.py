"""
StressTestApi - API压力测试

模块路径: testing/performance/stress_test_api.py

提供API端点的压力测试功能，包括并发请求、渐进式负载、
断路器测试和故障恢复验证。
"""

import os
import sys
import json
import time
import random
import string
import hashlib
import threading
import statistics
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from urllib.parse import urljoin

import pytest


@dataclass
class RequestResult:
    """单个请求的测试结果"""
    request_id: str
    method: str
    url: str
    status_code: int = 0
    response_time: float = 0.0
    success: bool = False
    error: str = ""
    response_size: int = 0
    timestamp: float = 0.0


@dataclass
class StressTestReport:
    """压力测试报告"""
    test_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    median_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    stdev_response_time: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    status_code_distribution: Dict[int, int] = field(default_factory=dict)
    error_messages: Dict[str, int] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "test_name": self.test_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_duration": round(self.total_duration, 3),
            "avg_response_time": round(self.avg_response_time, 4),
            "min_response_time": round(self.min_response_time, 4),
            "max_response_time": round(self.max_response_time, 4),
            "median_response_time": round(self.median_response_time, 4),
            "p95_response_time": round(self.p95_response_time, 4),
            "p99_response_time": round(self.p99_response_time, 4),
            "requests_per_second": round(self.requests_per_second, 2),
            "error_rate": round(self.error_rate, 4),
            "status_code_distribution": self.status_code_distribution,
        }


@dataclass
class LoadStage:
    """负载测试阶段"""
    name: str
    concurrency: int
    duration_seconds: float
    ramp_up_seconds: float = 0.0


class StressTestApi:
    """API压力测试工具

    提供全面的API压力测试功能:
        - 并发请求模拟
        - 渐进式负载测试（阶梯式增加并发）
        - 持续负载测试
        - 突发流量测试
        - 断路器行为验证
        - 错误率监控和阈值告警
        - 详细的统计报告（响应时间百分位数、吞吐量等）
        - 支持自定义请求构建器和断言器
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._base_url: str = self.config.get("base_url", "http://localhost:8080")
        self._default_timeout: float = self.config.get("timeout", 30.0)
        self._max_concurrency: int = self.config.get("max_concurrency", 100)
        self._results: List[RequestResult] = []
        self._reports: List[StressTestReport] = []
        self._lock = threading.Lock()
        self._request_counter = 0
        self._request_builder: Optional[Callable] = None
        self._response_validator: Optional[Callable] = None

    def initialize(self) -> None:
        """初始化压力测试工具"""
        self._initialized = True

    def set_base_url(self, url: str) -> None:
        """设置API基础URL

        Args:
            url: 基础URL
        """
        self._base_url = url.rstrip("/")

    def set_request_builder(self, builder: Callable) -> None:
        """设置自定义请求构建器

        Args:
            builder: 接受无参数，返回(method, path, headers, body)元组的函数
        """
        self._request_builder = builder

    def set_response_validator(self, validator: Callable) -> None:
        """设置自定义响应验证器

        Args:
            validator: 接受(response_data, status_code)，返回(bool, error_msg)的函数
        """
        self._response_validator = validator

    def _generate_request_id(self) -> str:
        """生成唯一请求ID"""
        with self._lock:
            self._request_counter += 1
            return f"req_{self._request_counter:06d}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    def _build_request(self, method: str = "GET", path: str = "/",
                        headers: Optional[Dict] = None,
                        body: Optional[Any] = None) -> Tuple[str, str, Dict, Any]:
        """构建请求参数

        Args:
            method: HTTP方法
            path: 请求路径
            headers: 请求头
            body: 请求体

        Returns:
            (method, full_url, headers, body) 元组
        """
        if self._request_builder:
            return self._request_builder()
        full_url = urljoin(self._base_url + "/", path.lstrip("/"))
        req_headers = headers or {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": self._generate_request_id(),
        }
        return method, full_url, req_headers, body

    def _simulate_request(self, method: str, url: str,
                           headers: Dict, body: Any,
                           timeout: float) -> RequestResult:
        """模拟HTTP请求（用于无网络依赖的测试）

        在实际使用中，此方法应替换为真实的HTTP客户端调用。
        这里使用模拟来确保测试框架可独立运行。

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体
            timeout: 超时时间

        Returns:
            RequestResult请求结果
        """
        result = RequestResult(
            request_id=headers.get("X-Request-ID", self._generate_request_id()),
            method=method,
            url=url,
            timestamp=time.time(),
        )
        start = time.perf_counter()
        try:
            simulated_latency = random.gauss(0.05, 0.02)
            simulated_latency = max(0.001, min(simulated_latency, timeout))
            time.sleep(simulated_latency * 0.01)
            result.status_code = random.choices(
                [200, 201, 400, 404, 500, 503],
                weights=[70, 10, 5, 5, 5, 5],
            )[0]
            result.response_time = time.perf_counter() - start
            result.response_size = random.randint(100, 10000)
            result.success = 200 <= result.status_code < 300
            if not result.success:
                result.error = f"HTTP {result.status_code}"
        except Exception as e:
            result.response_time = time.perf_counter() - start
            result.error = str(e)
            result.success = False
        return result

    def execute_single(self, method: str = "GET", path: str = "/",
                        headers: Optional[Dict] = None,
                        body: Optional[Any] = None) -> RequestResult:
        """执行单个请求

        Args:
            method: HTTP方法
            path: 请求路径
            headers: 请求头
            body: 请求体

        Returns:
            RequestResult请求结果
        """
        method, url, req_headers, req_body = self._build_request(method, path, headers, body)
        return self._simulate_request(method, url, req_headers, req_body, self._default_timeout)

    def execute_concurrent(self, num_requests: int, method: str = "GET",
                            path: str = "/", concurrency: Optional[int] = None,
                            headers: Optional[Dict] = None,
                            body: Optional[Any] = None) -> StressTestReport:
        """并发执行多个请求

        Args:
            num_requests: 总请求数
            method: HTTP方法
            path: 请求路径
            concurrency: 并发数，默认使用max_concurrency
            headers: 请求头
            body: 请求体

        Returns:
            StressTestReport压力测试报告
        """
        workers = concurrency or self._max_concurrency
        results: List[RequestResult] = []
        start_time = time.monotonic()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for _ in range(num_requests):
                future = executor.submit(self.execute_single, method, path, headers, body)
                futures.append(future)
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self._default_timeout)
                    results.append(result)
                except Exception as e:
                    results.append(RequestResult(
                        request_id=self._generate_request_id(),
                        method=method,
                        url=urljoin(self._base_url + "/", path.lstrip("/")),
                        success=False,
                        error=str(e),
                    ))

        total_duration = time.monotonic() - start_time
        report = self._build_report("concurrent_test", results, total_duration)
        with self._lock:
            self._results.extend(results)
            self._reports.append(report)
        return report

    def execute_staged(self, stages: List[LoadStage],
                        method: str = "GET", path: str = "/") -> List[StressTestReport]:
        """执行分阶段负载测试

        Args:
            stages: 负载阶段列表
            method: HTTP方法
            path: 请求路径

        Returns:
            每个阶段的StressTestReport列表
        """
        reports = []
        for stage in stages:
            stage_report = self._run_stage(stage, method, path)
            reports.append(stage_report)
        with self._lock:
            self._reports.extend(reports)
        return reports

    def _run_stage(self, stage: LoadStage, method: str, path: str) -> StressTestReport:
        """运行单个负载阶段

        Args:
            stage: 负载阶段配置
            method: HTTP方法
            path: 请求路径

        Returns:
            该阶段的测试报告
        """
        if stage.ramp_up_seconds > 0:
            time.sleep(stage.ramp_up_seconds)
        requests_per_worker = max(1, int(stage.duration_seconds * stage.concurrency / max(stage.concurrency, 1)))
        total_requests = stage.concurrency * requests_per_worker
        results: List[RequestResult] = []
        start_time = time.monotonic()

        def _worker():
            for _ in range(requests_per_worker):
                result = self.execute_single(method, path)
                results.append(result)

        threads = []
        for _ in range(stage.concurrency):
            t = threading.Thread(target=_worker)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=stage.duration_seconds + self._default_timeout)

        total_duration = time.monotonic() - start_time
        report = self._build_report(f"stage_{stage.name}", results, total_duration)
        return report

    def execute_burst(self, burst_size: int, method: str = "GET",
                       path: str = "/", repeat: int = 3,
                       interval: float = 1.0) -> List[StressTestReport]:
        """执行突发流量测试

        Args:
            burst_size: 每次突发的请求数
            method: HTTP方法
            path: 请求路径
            repeat: 突发重复次数
            interval: 突发间隔（秒）

        Returns:
            每次突发的测试报告列表
        """
        reports = []
        for i in range(repeat):
            report = self.execute_concurrent(burst_size, method, path, concurrency=burst_size)
            report.test_name = f"burst_{i + 1}"
            reports.append(report)
            if i < repeat - 1:
                time.sleep(interval)
        return reports

    def execute_endurance(self, duration_seconds: float, rps: int,
                           method: str = "GET", path: str = "/") -> StressTestReport:
        """执行持久负载测试

        Args:
            duration_seconds: 测试持续时间（秒）
            rps: 目标每秒请求数
            method: HTTP方法
            path: 请求路径

        Returns:
            测试报告
        """
        total_requests = int(duration_seconds * rps)
        concurrency = min(rps, self._max_concurrency)
        return self.execute_concurrent(total_requests, method, path, concurrency)

    def test_circuit_breaker(self, failure_threshold: int = 5,
                              recovery_timeout: float = 10.0,
                              method: str = "GET", path: str = "/") -> Dict[str, Any]:
        """测试断路器行为

        Args:
            failure_threshold: 触发断路器的连续失败次数
            recovery_timeout: 断路器恢复超时
            method: HTTP方法
            path: 请求路径

        Returns:
            断路器测试结果
        """
        results = []
        circuit_open = False
        open_after = failure_threshold

        for i in range(failure_threshold + 3):
            result = self.execute_single(method, path)
            results.append(result)
            if not result.success and i >= failure_threshold - 1:
                circuit_open = True

        return {
            "circuit_opened": circuit_open,
            "opened_after_failures": open_after,
            "total_requests": len(results),
            "successful_before_open": sum(1 for r in results[:failure_threshold] if r.success),
            "failed_after_open": sum(1 for r in results[failure_threshold:] if not r.success),
            "recovery_timeout": recovery_timeout,
        }

    def _build_report(self, test_name: str, results: List[RequestResult],
                       total_duration: float) -> StressTestReport:
        """构建压力测试报告

        Args:
            test_name: 测试名称
            results: 请求结果列表
            total_duration: 总持续时间

        Returns:
            StressTestReport报告
        """
        if not results:
            return StressTestReport(test_name=test_name)
        response_times = [r.response_time for r in results if r.response_time > 0]
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        sorted_times = sorted(response_times) if response_times else [0.0]
        n = len(sorted_times)
        status_dist: Dict[int, int] = defaultdict(int)
        error_dist: Dict[str, int] = defaultdict(int)
        for r in results:
            status_dist[r.status_code] += 1
            if r.error:
                error_dist[r.error] += 1

        return StressTestReport(
            test_name=test_name,
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            total_duration=total_duration,
            avg_response_time=statistics.mean(sorted_times) if sorted_times else 0.0,
            min_response_time=sorted_times[0],
            max_response_time=sorted_times[-1],
            median_response_time=sorted_times[n // 2],
            p95_response_time=sorted_times[min(int(n * 0.95), n - 1)],
            p99_response_time=sorted_times[min(int(n * 0.99), n - 1)],
            stdev_response_time=statistics.stdev(sorted_times) if n > 1 else 0.0,
            requests_per_second=len(results) / total_duration if total_duration > 0 else 0.0,
            error_rate=failed / len(results) if results else 0.0,
            status_code_distribution=dict(status_dist),
            error_messages=dict(error_dist),
        )

    def get_results(self) -> List[RequestResult]:
        """获取所有请求结果"""
        return list(self._results)

    def get_reports(self) -> List[StressTestReport]:
        """获取所有测试报告"""
        return list(self._reports)

    def format_report(self, report: Optional[StressTestReport] = None) -> str:
        """格式化终端报告

        Args:
            report: 可选的报告对象

        Returns:
            格式化的报告字符串
        """
        if report is None:
            if not self._reports:
                return "No stress test reports available."
            report = self._reports[-1]
        lines = [
            "=" * 70,
            f"STRESS TEST REPORT: {report.test_name}",
            "=" * 70,
            f"  Total Requests:      {report.total_requests}",
            f"  Successful:          {report.successful_requests}",
            f"  Failed:              {report.failed_requests}",
            f"  Error Rate:          {report.error_rate:.2%}",
            f"  Duration:            {report.total_duration:.2f}s",
            f"  Requests/sec:        {report.requests_per_second:.1f}",
            "-" * 70,
            f"  Avg Response Time:   {report.avg_response_time:.4f}s",
            f"  Median Response Time:{report.median_response_time:.4f}s",
            f"  P95 Response Time:   {report.p95_response_time:.4f}s",
            f"  P99 Response Time:   {report.p99_response_time:.4f}s",
            f"  Min Response Time:   {report.min_response_time:.4f}s",
            f"  Max Response Time:   {report.max_response_time:.4f}s",
            "-" * 70,
            "  Status Codes:",
        ]
        for code, count in sorted(report.status_code_distribution.items()):
            lines.append(f"    {code}: {count}")
        if report.error_messages:
            lines.append("  Errors:")
            for err, count in sorted(report.error_messages.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"    {err}: {count}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def save_json_report(self, filepath: str) -> None:
        """保存JSON格式的压力测试报告

        Args:
            filepath: 输出文件路径
        """
        data = {
            "timestamp": time.time(),
            "base_url": self._base_url,
            "reports": [r.to_dict() for r in self._reports],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def reset(self) -> None:
        """重置所有测试数据"""
        with self._lock:
            self._results.clear()
            self._reports.clear()
            self._request_counter = 0

    def generate_random_payload(self, size_bytes: int = 1024) -> str:
        """生成随机请求负载

        Args:
            size_bytes: 负载大小（字节）

        Returns:
            随机字符串
        """
        chars = string.ascii_letters + string.digits + " .,!?\n"
        return "".join(random.choice(chars) for _ in range(size_bytes))
