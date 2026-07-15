"""
Edge Inference Benchmarking Module

Provides latency measurement (p50/p95/p99), throughput testing, memory profiling,
power consumption estimation, model comparison, and result visualization (ASCII charts).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import statistics
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

try:
    import resource as _resource
except ImportError:
    _resource = None  # type: ignore

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_BENCHMARK_ITERATIONS = 100
DEFAULT_BATCH_SIZE = 1
MEMORY_SAMPLE_INTERVAL = 0.1  # seconds
POWER_MODEL_COEFFICIENT = 0.15  # watts per GFLOP


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BenchmarkType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    POWER = "power"
    COMPREHENSIVE = "comprehensive"


class ModelFormat(Enum):
    TFLITE = "tflite"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    CUSTOM = "custom"


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    DSP = "dsp"
    TPU = "tpu"
    FPGA = "fpga"


class PrecisionType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class LatencySample:
    """A single latency measurement."""
    iteration: int = 0
    latency_ms: float = 0.0
    timestamp: float = 0.0
    batch_size: int = 1
    input_size: int = 0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""
    count: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p999_ms: float = 0.0
    total_time_s: float = 0.0
    throughput_fps: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "p999_ms": self.p999_ms,
            "total_time_s": self.total_time_s,
            "throughput_fps": self.throughput_fps,
        }


@dataclass
class MemorySample:
    """A single memory measurement."""
    timestamp: float = 0.0
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    heap_mb: float = 0.0
    gpu_mb: float = 0.0
    shared_mb: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class MemoryStats:
    """Memory profiling statistics."""
    peak_rss_mb: float = 0.0
    avg_rss_mb: float = 0.0
    min_rss_mb: float = 0.0
    max_rss_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    avg_gpu_mb: float = 0.0
    sample_count: int = 0
    allocation_rate_mb_per_s: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "peak_rss_mb": self.peak_rss_mb,
            "avg_rss_mb": self.avg_rss_mb,
            "min_rss_mb": self.min_rss_mb,
            "max_rss_mb": self.max_rss_mb,
            "peak_gpu_mb": self.peak_gpu_mb,
            "avg_gpu_mb": self.avg_gpu_mb,
            "sample_count": self.sample_count,
            "allocation_rate_mb_per_s": self.allocation_rate_mb_per_s,
        }


@dataclass
class PowerEstimate:
    """Estimated power consumption."""
    avg_watts: float = 0.0
    peak_watts: float = 0.0
    min_watts: float = 0.0
    energy_joules: float = 0.0
    samples: List[Tuple[float, float]] = field(default_factory=list)

    def efficiency_fps_per_watt(self, fps: float) -> float:
        return fps / self.avg_watts if self.avg_watts > 0 else 0.0


@dataclass
class ModelBenchmarkConfig:
    """Configuration for benchmarking a model."""
    model_name: str = ""
    model_format: ModelFormat = ModelFormat.TFLITE
    device: DeviceType = DeviceType.CPU
    precision: PrecisionType = PrecisionType.FP32
    batch_size: int = DEFAULT_BATCH_SIZE
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    benchmark_iterations: int = DEFAULT_BENCHMARK_ITERATIONS
    input_shape: List[int] = field(default_factory=lambda: [1, 224, 224, 3])
    input_type: str = "float32"
    num_threads: int = 4
    model_path: str = ""
    delegate: str = ""


@dataclass
class BenchmarkResult:
    """Complete result of a benchmark run."""
    model_name: str = ""
    model_format: str = "unknown"
    device: str = "cpu"
    precision: str = "fp32"
    batch_size: int = 1
    latency: Optional[LatencyStats] = None
    memory: Optional[MemoryStats] = None
    power: Optional[PowerEstimate] = None
    warmup_iterations: int = 0
    benchmark_iterations: int = 0
    total_time_s: float = 0.0
    timestamp: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_format": self.model_format,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "latency": self.latency.to_dict() if self.latency else None,
            "memory": self.memory.to_dict() if self.memory else None,
            "power": {
                "avg_watts": self.power.avg_watts,
                "peak_watts": self.power.peak_watts,
                "energy_joules": self.power.energy_joules,
            } if self.power else None,
            "total_time_s": self.total_time_s,
            "timestamp": self.timestamp,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Latency Profiler
# ---------------------------------------------------------------------------

class LatencyProfiler:
    """Measures inference latency with percentile statistics."""

    def __init__(self) -> None:
        self._samples: List[LatencySample] = []

    def record(self, latency_ms: float, iteration: int = 0) -> None:
        self._samples.append(LatencySample(
            iteration=iteration,
            latency_ms=latency_ms,
        ))

    def clear(self) -> None:
        self._samples.clear()

    def compute_stats(self) -> LatencyStats:
        if not self._samples:
            return LatencyStats()

        latencies = sorted(s.latency_ms for s in self._samples)
        n = len(latencies)
        total = sum(latencies)

        stats = LatencyStats(
            count=n,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            std_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            min_ms=latencies[0],
            max_ms=latencies[-1],
            total_time_s=total / 1000.0,
        )

        stats.p50_ms = self._percentile(latencies, 50)
        stats.p90_ms = self._percentile(latencies, 90)
        stats.p95_ms = self._percentile(latencies, 95)
        stats.p99_ms = self._percentile(latencies, 99)
        stats.p999_ms = self._percentile(latencies, 99.9)

        if stats.total_time_s > 0:
            stats.throughput_fps = n / stats.total_time_s

        return stats

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * percentile / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def get_samples(self) -> List[LatencySample]:
        return list(self._samples)


# ---------------------------------------------------------------------------
# Throughput Tester
# ---------------------------------------------------------------------------

class ThroughputTester:
    """Tests inference throughput under various conditions."""

    def __init__(self) -> None:
        self._results: Dict[int, float] = {}

    def measure_throughput(
        self,
        inference_fn: Callable[[], float],
        duration_s: float = 5.0,
        batch_size: int = 1,
        num_threads: int = 1,
    ) -> float:
        start = time.time()
        count = 0
        end_time = start + duration_s

        while time.time() < end_time:
            inference_fn()
            count += 1

        elapsed = time.time() - start
        fps = (count * batch_size) / elapsed
        self._results[batch_size] = fps
        return fps

    def measure_scaling(
        self,
        inference_fn: Callable[[], float],
        batch_sizes: Optional[List[int]] = None,
        duration_per_batch: float = 3.0,
    ) -> Dict[int, float]:
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        results: Dict[int, float] = {}
        for bs in batch_sizes:
            fps = self.measure_throughput(inference_fn, duration_per_batch, bs)
            results[bs] = fps
        self._results = results
        return results

    def get_results(self) -> Dict[int, float]:
        return dict(self._results)


# ---------------------------------------------------------------------------
# Memory Profiler
# ---------------------------------------------------------------------------

class MemoryProfiler:
    """Profiles memory usage during inference."""

    def __init__(self) -> None:
        self._samples: List[MemorySample] = []
        self._profiling = False
        self._thread: Optional[threading.Thread] = None
        self._baseline_rss: float = 0.0
        self._baseline_vms: float = 0.0

    def take_snapshot(self) -> MemorySample:
        rss, vms = self._get_memory_usage()
        return MemorySample(
            rss_mb=rss,
            vms_mb=vms,
            heap_mb=rss * 0.6,
        )

    def start_profiling(self, interval: float = MEMORY_SAMPLE_INTERVAL) -> None:
        baseline = self.take_snapshot()
        self._baseline_rss = baseline.rss_mb
        self._baseline_vms = baseline.vms_mb
        self._profiling = True
        self._thread = threading.Thread(
            target=self._profile_loop, daemon=True, name="mem-profiler",
            args=(interval,),
        )
        self._thread.start()

    def stop_profiling(self) -> None:
        self._profiling = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _profile_loop(self, interval: float) -> None:
        while self._profiling:
            sample = self.take_snapshot()
            self._samples.append(sample)
            time.sleep(interval)

    def compute_stats(self) -> MemoryStats:
        if not self._samples:
            return MemoryStats()

        rss_values = [s.rss_mb for s in self._samples]
        gpu_values = [s.gpu_mb for s in self._samples if s.gpu_mb > 0]

        duration = (self._samples[-1].timestamp - self._samples[0].timestamp) if len(self._samples) > 1 else 1.0
        peak_rss = max(rss_values) - self._baseline_rss if rss_values else 0

        stats = MemoryStats(
            peak_rss_mb=max(0, peak_rss),
            avg_rss_mb=statistics.mean(rss_values) - self._baseline_rss if rss_values else 0,
            min_rss_mb=min(rss_values) - self._baseline_rss if rss_values else 0,
            max_rss_mb=max(rss_values) - self._baseline_rss if rss_values else 0,
            sample_count=len(self._samples),
            allocation_rate_mb_per_s=max(0, peak_rss) / duration if duration > 0 else 0,
        )

        if gpu_values:
            stats.peak_gpu_mb = max(gpu_values)
            stats.avg_gpu_mb = statistics.mean(gpu_values)

        return stats

    def clear(self) -> None:
        self._samples.clear()

    def _get_memory_usage(self) -> Tuple[float, float]:
        if _resource is not None:
            usage = _resource.getrusage(_resource.RUSAGE_SELF)
            rss = usage.ru_maxrss / 1024.0  # KB to MB
            return rss, rss
        elif _psutil is not None:
            process = _psutil.Process()
            mem = process.memory_info()
            return mem.rss / (1024 * 1024), mem.vms / (1024 * 1024)
        else:
            return 0.0, 0.0

    def get_samples(self) -> List[MemorySample]:
        return list(self._samples)


# ---------------------------------------------------------------------------
# Power Estimator
# ---------------------------------------------------------------------------

class PowerEstimator:
    """Estimates power consumption during inference."""

    def __init__(self) -> None:
        self._samples: List[Tuple[float, float]] = []
        self._baseline_watts: float = 5.0  # idle power

    def record_sample(self, utilization: float, duration_s: float = 0.0) -> None:
        watts = self._baseline_watts + utilization * 15.0  # up to 20W under load
        self._samples.append((time.time(), watts))

    def estimate_from_latency(
        self,
        latency_ms: float,
        model_flops: float = 1.0,
        device_efficiency: float = 0.5,
    ) -> PowerEstimate:
        """Estimate power from model FLOPs and latency."""
        inference_time_s = latency_ms / 1000.0
        gflops = model_flops / (latency_ms / 1000.0) if latency_ms > 0 else 0
        watts = self._baseline_watts + gflops * POWER_MODEL_COEFFICIENT / device_efficiency
        energy = watts * inference_time_s

        return PowerEstimate(
            avg_watts=watts,
            peak_watts=watts * 1.3,
            min_watts=self._baseline_watts,
            energy_joules=energy,
        )

    def estimate_from_profile(
        self,
        total_time_s: float,
        avg_utilization: float = 0.5,
    ) -> PowerEstimate:
        avg_watts = self._baseline_watts + avg_utilization * 15.0
        return PowerEstimate(
            avg_watts=avg_watts,
            peak_watts=avg_watts * 1.5,
            min_watts=self._baseline_watts,
            energy_joules=avg_watts * total_time_s,
        )

    def compute_stats(self) -> PowerEstimate:
        if not self._samples:
            return PowerEstimate()
        watts = [w for _, w in self._samples]
        duration = self._samples[-1][0] - self._samples[0][0] if len(self._samples) > 1 else 1.0
        return PowerEstimate(
            avg_watts=statistics.mean(watts),
            peak_watts=max(watts),
            min_watts=min(watts),
            energy_joules=sum(watts) / len(watts) * duration,
            samples=list(self._samples),
        )

    def clear(self) -> None:
        self._samples.clear()


# ---------------------------------------------------------------------------
# Model Comparator
# ---------------------------------------------------------------------------

class ModelComparator:
    """Compares benchmark results across models and configurations."""

    def __init__(self) -> None:
        self._results: Dict[str, BenchmarkResult] = {}

    def add_result(self, result: BenchmarkResult) -> None:
        key = f"{result.model_name}_{result.device}_{result.precision}_bs{result.batch_size}"
        self._results[key] = result

    def compare(self, model_a: str, model_b: str) -> Dict[str, Any]:
        ra = self._results.get(model_a)
        rb = self._results.get(model_b)
        if ra is None or rb is None:
            return {"error": "Model not found"}

        comparison: Dict[str, Any] = {}
        if ra.latency and rb.latency:
            comparison["latency"] = {
                f"{model_a}_p50_ms": ra.latency.p50_ms,
                f"{model_b}_p50_ms": rb.latency.p50_ms,
                "speedup": rb.latency.p50_ms / ra.latency.p50_ms if ra.latency.p50_ms > 0 else 0,
                "winner": model_a if ra.latency.p50_ms < rb.latency.p50_ms else model_b,
            }
        if ra.latency and rb.latency:
            comparison["throughput"] = {
                f"{model_a}_fps": ra.latency.throughput_fps,
                f"{model_b}_fps": rb.latency.throughput_fps,
                "winner": model_a if ra.latency.throughput_fps > rb.latency.throughput_fps else model_b,
            }
        if ra.memory and rb.memory:
            comparison["memory"] = {
                f"{model_a}_peak_mb": ra.memory.peak_rss_mb,
                f"{model_b}_peak_mb": rb.memory.peak_rss_mb,
                "winner": model_a if ra.memory.peak_rss_mb < rb.memory.peak_rss_mb else model_b,
            }
        return comparison

    def rank_by_latency(self) -> List[Tuple[str, float]]:
        items = []
        for key, result in self._results.items():
            if result.latency:
                items.append((key, result.latency.p95_ms))
        items.sort(key=lambda x: x[1])
        return items

    def rank_by_throughput(self) -> List[Tuple[str, float]]:
        items = []
        for key, result in self._results.items():
            if result.latency:
                items.append((key, result.latency.throughput_fps))
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def rank_by_memory(self) -> List[Tuple[str, float]]:
        items = []
        for key, result in self._results.items():
            if result.memory:
                items.append((key, result.memory.peak_rss_mb))
        items.sort(key=lambda x: x[1])
        return items

    def get_all_results(self) -> Dict[str, BenchmarkResult]:
        return dict(self._results)


# ---------------------------------------------------------------------------
# Result Reporter
# ---------------------------------------------------------------------------

class ResultReporter:
    """Generates ASCII chart visualizations and reports."""

    def __init__(self) -> None:
        self._results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        self._results.append(result)

    def generate_latency_chart(self, result: BenchmarkResult) -> str:
        if not result.latency:
            return "No latency data available"
        stats = result.latency
        chart_width = 50
        max_val = stats.p999_ms if stats.p999_ms > 0 else stats.max_ms
        if max_val == 0:
            return "No data"

        lines: List[str] = []
        lines.append(f"Latency Distribution: {result.model_name}")
        lines.append(f"{'=' * 60}")
        lines.append(f"Iterations: {stats.count}")
        lines.append(f"Mean: {stats.mean_ms:.2f}ms | Median: {stats.median_ms:.2f}ms")
        lines.append(f"P50: {stats.p50_ms:.2f}ms | P95: {stats.p95_ms:.2f}ms | P99: {stats.p99_ms:.2f}ms")
        lines.append(f"Min: {stats.min_ms:.2f}ms | Max: {stats.max_ms:.2f}ms")
        lines.append(f"Std: {stats.std_ms:.2f}ms")
        lines.append(f"Throughput: {stats.throughput_fps:.1f} FPS")
        lines.append("")

        percentiles = [
            ("Min ", stats.min_ms),
            ("P50 ", stats.p50_ms),
            ("P90 ", stats.p90_ms),
            ("P95 ", stats.p95_ms),
            ("P99 ", stats.p99_ms),
            ("P999", stats.p999_ms),
            ("Max ", stats.max_ms),
        ]
        for label, value in percentiles:
            bar_len = int((value / max_val) * chart_width)
            bar = "#" * bar_len
            lines.append(f"{label}: {value:8.2f}ms |{bar:<{chart_width}}|")

        return "\n".join(lines)

    def generate_comparison_table(self) -> str:
        if not self._results:
            return "No results to compare"
        lines: List[str] = []
        lines.append(f"{'Model':<20} {'Device':<8} {'Prec':<6} {'BS':>3} {'P50(ms)':>8} {'P95(ms)':>8} {'FPS':>8} {'Mem(MB)':>8}")
        lines.append("-" * 80)
        for r in self._results:
            p50 = f"{r.latency.p50_ms:.2f}" if r.latency else "N/A"
            p95 = f"{r.latency.p95_ms:.2f}" if r.latency else "N/A"
            fps = f"{r.latency.throughput_fps:.1f}" if r.latency else "N/A"
            mem = f"{r.memory.peak_rss_mb:.1f}" if r.memory else "N/A"
            lines.append(
                f"{r.model_name:<20} {r.device:<8} {r.precision:<6} "
                f"{r.batch_size:>3} {p50:>8} {p95:>8} {fps:>8} {mem:>8}"
            )
        return "\n".join(lines)

    def generate_bar_chart(
        self,
        data: Dict[str, float],
        title: str = "Comparison",
        width: int = 40,
    ) -> str:
        lines: List[str] = []
        lines.append(title)
        lines.append("-" * (width + 20))
        max_val = max(data.values()) if data else 1
        if max_val == 0:
            max_val = 1
        max_label_len = max(len(k) for k in data.keys()) if data else 10
        for label, value in data.items():
            bar_len = int((value / max_val) * width)
            bar = "#" * bar_len
            lines.append(f"{label:<{max_label_len}} |{bar:<{width}}| {value:.2f}")
        return "\n".join(lines)

    def generate_summary(self) -> str:
        lines: List[str] = []
        lines.append(f"Benchmark Summary ({len(self._results)} models)")
        lines.append("=" * 60)
        lines.append(self.generate_comparison_table())
        lines.append("")

        if len(self._results) > 1:
            best_latency = min(
                (r for r in self._results if r.latency),
                key=lambda r: r.latency.p95_ms,
                default=None,
            )
            best_throughput = max(
                (r for r in self._results if r.latency),
                key=lambda r: r.latency.throughput_fps,
                default=None,
            )
            if best_latency:
                lines.append(f"Best latency: {best_latency.model_name} ({best_latency.latency.p95_ms:.2f}ms P95)")
            if best_throughput:
                lines.append(f"Best throughput: {best_throughput.model_name} ({best_throughput.latency.throughput_fps:.1f} FPS)")

        return "\n".join(lines)

    def export_json(self, path: str) -> None:
        data = [r.to_dict() for r in self._results]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Benchmark Runner (Main Facade)
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Main facade for running edge inference benchmarks."""

    def __init__(self) -> None:
        self.latency_profiler = LatencyProfiler()
        self.throughput_tester = ThroughputTester()
        self.memory_profiler = MemoryProfiler()
        self.power_estimator = PowerEstimator()
        self.model_comparator = ModelComparator()
        self.reporter = ResultReporter()
        self._lock = threading.Lock()

    def run_benchmark(
        self,
        config: ModelBenchmarkConfig,
        inference_fn: Optional[Callable[[], float]] = None,
    ) -> BenchmarkResult:
        if inference_fn is None:
            inference_fn = self._simulate_inference

        result = BenchmarkResult(
            model_name=config.model_name,
            model_format=config.model_format.value,
            device=config.device.value,
            precision=config.precision.value,
            batch_size=config.batch_size,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
            timestamp=datetime.utcnow().isoformat(),
        )

        self.latency_profiler.clear()
        self.memory_profiler.clear()
        self.power_estimator.clear()

        # Warmup
        logger.info("Warming up (%d iterations)...", config.warmup_iterations)
        for i in range(config.warmup_iterations):
            inference_fn()

        # Benchmark
        logger.info("Running benchmark (%d iterations)...", config.benchmark_iterations)
        self.memory_profiler.start_profiling()
        start_time = time.time()

        for i in range(config.benchmark_iterations):
            latency = inference_fn()
            self.latency_profiler.record(latency, i)
            utilization = min(1.0, latency / 100.0)
            self.power_estimator.record_sample(utilization)

        self.memory_profiler.stop_profiling()
        total_time = time.time() - start_time

        result.latency = self.latency_profiler.compute_stats()
        result.memory = self.memory_profiler.compute_stats()
        result.power = self.power_estimator.compute_stats()
        result.total_time_s = total_time

        self.model_comparator.add_result(result)
        self.reporter.add_result(result)

        logger.info(
            "Benchmark complete: %s P50=%.2fms P95=%.2fms FPS=%.1f",
            config.model_name,
            result.latency.p50_ms,
            result.latency.p95_ms,
            result.latency.throughput_fps,
        )
        return result

    def run_comparison(
        self,
        configs: List[ModelBenchmarkConfig],
        inference_fns: Optional[Dict[str, Callable[[], float]]] = None,
    ) -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        for config in configs:
            fn = None
            if inference_fns:
                fn = inference_fns.get(config.model_name)
            result = self.run_benchmark(config, fn)
            results.append(result)
        return results

    def generate_report(self) -> str:
        return self.reporter.generate_summary()

    def export_results(self, path: str) -> None:
        self.reporter.export_json(path)

    @staticmethod
    def _simulate_inference() -> float:
        """Simulate inference latency."""
        base_latency = 5.0
        noise = random.gauss(0, 1.0)
        jitter = random.uniform(-0.5, 0.5)
        return max(0.1, base_latency + noise + jitter)
