"""
Performance Profiler Module

性能分析器实现，提供延迟分析、吞吐量监控和瓶颈识别功能。
"""

from __future__ import annotations

import time
import statistics
import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..config import PerformanceConfig

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    """瓶颈类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    DATABASE = "database"
    LOCK_CONTENTION = "lock_contention"
    UNKNOWN = "unknown"


@dataclass
class LatencyStats:
    """
    延迟统计
    
    Attributes:
        count: 样本数
        min_ms: 最小延迟（毫秒）
        max_ms: 最大延迟（毫秒）
        mean_ms: 平均延迟（毫秒）
        median_ms: 中位数延迟（毫秒）
        p50_ms: 50分位数（毫秒）
        p90_ms: 90分位数（毫秒）
        p95_ms: 95分位数（毫秒）
        p99_ms: 99分位数（毫秒）
        stddev_ms: 标准差（毫秒）
    """
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    stddev_ms: float = 0.0
    
    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        """从样本计算统计信息"""
        if not samples:
            return cls()
        
        sorted_samples = sorted(samples)
        count = len(sorted_samples)
        
        def percentile(p: float) -> float:
            idx = int(count * p / 100)
            return sorted_samples[min(idx, count - 1)]
        
        return cls(
            count=count,
            min_ms=min(sorted_samples),
            max_ms=max(sorted_samples),
            mean_ms=statistics.mean(sorted_samples),
            median_ms=statistics.median(sorted_samples),
            p50_ms=percentile(50),
            p90_ms=percentile(90),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            stddev_ms=statistics.stdev(sorted_samples) if count > 1 else 0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "count": self.count,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "stddev_ms": round(self.stddev_ms, 3)
        }


@dataclass
class ThroughputStats:
    """
    吞吐量统计
    
    Attributes:
        requests_per_second: 每秒请求数
        requests_per_minute: 每分钟请求数
        bytes_per_second: 每秒字节数
        total_requests: 总请求数
        total_bytes: 总字节数
        window_seconds: 统计窗口（秒）
    """
    requests_per_second: float = 0.0
    requests_per_minute: float = 0.0
    bytes_per_second: float = 0.0
    total_requests: int = 0
    total_bytes: int = 0
    window_seconds: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "requests_per_second": round(self.requests_per_second, 2),
            "requests_per_minute": round(self.requests_per_minute, 2),
            "bytes_per_second": round(self.bytes_per_second, 2),
            "total_requests": self.total_requests,
            "total_bytes": self.total_bytes,
            "window_seconds": self.window_seconds
        }


@dataclass
class BottleneckInfo:
    """
    瓶颈信息
    
    Attributes:
        type: 瓶颈类型
        severity: 严重程度 (1-10)
        description: 描述
        affected_components: 受影响的组件
        suggested_actions: 建议操作
        timestamp: 时间戳
    """
    type: BottleneckType
    severity: int
    description: str
    affected_components: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type.value,
            "severity": self.severity,
            "description": self.description,
            "affected_components": self.affected_components,
            "suggested_actions": self.suggested_actions,
            "timestamp": self.timestamp
        }


@dataclass
class PerformanceMetrics:
    """
    性能指标
    
    Attributes:
        latency_stats: 延迟统计
        throughput_stats: 吞吐量统计
        cpu_percent: CPU使用率
        memory_percent: 内存使用率
        active_threads: 活动线程数
        timestamp: 时间戳
    """
    latency_stats: LatencyStats = field(default_factory=LatencyStats)
    throughput_stats: ThroughputStats = field(default_factory=ThroughputStats)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_threads: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "latency_stats": self.latency_stats.to_dict(),
            "throughput_stats": self.throughput_stats.to_dict(),
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_percent": round(self.memory_percent, 2),
            "active_threads": self.active_threads,
            "timestamp": self.timestamp
        }


@dataclass
class ProfileReport:
    """
    性能分析报告
    
    Attributes:
        start_time: 开始时间
        end_time: 结束时间
        duration_seconds: 持续时间
        metrics: 性能指标列表
        bottlenecks: 瓶颈列表
        summary: 摘要
    """
    start_time: float
    end_time: float
    duration_seconds: float
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    bottlenecks: List[BottleneckInfo] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metrics": [m.to_dict() for m in self.metrics],
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "summary": self.summary
        }


class LatencyProfiler:
    """
    延迟分析器
    
    测量和分析操作延迟。
    """
    
    def __init__(self, max_samples: int = 10000):
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._max_samples = max_samples
        self._lock = threading.Lock()
    
    def record(self, operation: str, latency_ms: float) -> None:
        """
        记录延迟
        
        Args:
            operation: 操作名称
            latency_ms: 延迟（毫秒）
        """
        with self._lock:
            samples = self._samples[operation]
            samples.append(latency_ms)
            
            # Limit sample size
            if len(samples) > self._max_samples:
                self._samples[operation] = samples[-self._max_samples:]
    
    def time_operation(self, operation: str) -> "LatencyTimer":
        """
        创建计时器
        
        Args:
            operation: 操作名称
            
        Returns:
            计时器上下文管理器
        """
        return LatencyTimer(self, operation)
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, LatencyStats]:
        """
        获取延迟统计
        
        Args:
            operation: 操作名称，如果为None则返回所有操作
            
        Returns:
            延迟统计字典
        """
        with self._lock:
            if operation:
                samples = self._samples.get(operation, [])
                return {operation: LatencyStats.from_samples(samples)}
            else:
                return {
                    op: LatencyStats.from_samples(samples)
                    for op, samples in self._samples.items()
                }
    
    def clear(self, operation: Optional[str] = None) -> None:
        """清除样本"""
        with self._lock:
            if operation:
                self._samples.pop(operation, None)
            else:
                self._samples.clear()


class LatencyTimer:
    """延迟计时器"""
    
    def __init__(self, profiler: LatencyProfiler, operation: str):
        self._profiler = profiler
        self._operation = operation
        self._start_time: Optional[float] = None
    
    def __enter__(self) -> "LatencyTimer":
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time is not None:
            latency_ms = (time.time() - self._start_time) * 1000
            self._profiler.record(self._operation, latency_ms)


class ThroughputMonitor:
    """
    吞吐量监控器
    
    监控请求和数据的吞吐量。
    """
    
    def __init__(self, window_seconds: float = 60.0):
        self._window_seconds = window_seconds
        self._requests: deque = deque()
        self._bytes: deque = deque()
        self._lock = threading.Lock()
    
    def record_request(self, byte_count: int = 0) -> None:
        """
        记录请求
        
        Args:
            byte_count: 字节数
        """
        now = time.time()
        with self._lock:
            self._requests.append(now)
            if byte_count > 0:
                self._bytes.append((now, byte_count))
            self._cleanup_old_data(now)
    
    def _cleanup_old_data(self, current_time: float) -> None:
        """清理旧数据"""
        cutoff = current_time - self._window_seconds
        
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
        
        while self._bytes and self._bytes[0][0] < cutoff:
            self._bytes.popleft()
    
    def get_stats(self) -> ThroughputStats:
        """获取吞吐量统计"""
        now = time.time()
        
        with self._lock:
            self._cleanup_old_data(now)
            
            request_count = len(self._requests)
            total_bytes = sum(b[1] for b in self._bytes)
            
            if self._window_seconds > 0:
                rps = request_count / self._window_seconds
                bps = total_bytes / self._window_seconds
            else:
                rps = 0.0
                bps = 0.0
            
            return ThroughputStats(
                requests_per_second=rps,
                requests_per_minute=rps * 60,
                bytes_per_second=bps,
                total_requests=request_count,
                total_bytes=total_bytes,
                window_seconds=self._window_seconds
            )
    
    def reset(self) -> None:
        """重置监控器"""
        with self._lock:
            self._requests.clear()
            self._bytes.clear()


class BottleneckDetector:
    """
    瓶颈检测器
    
    检测系统性能瓶颈。
    """
    
    def __init__(
        self,
        latency_threshold_ms: float = 1000.0,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0
    ):
        self._latency_threshold = latency_threshold_ms
        self._cpu_threshold = cpu_threshold
        self._memory_threshold = memory_threshold
        self._bottlenecks: List[BottleneckInfo] = []
        self._lock = threading.Lock()
    
    def analyze(
        self,
        latency_stats: LatencyStats,
        cpu_percent: float,
        memory_percent: float,
        component: str = "system"
    ) -> List[BottleneckInfo]:
        """
        分析性能数据
        
        Args:
            latency_stats: 延迟统计
            cpu_percent: CPU使用率
            memory_percent: 内存使用率
            component: 组件名称
            
        Returns:
            检测到的瓶颈列表
        """
        bottlenecks: List[BottleneckInfo] = []
        
        # Check latency
        if latency_stats.p95_ms > self._latency_threshold:
            severity = min(10, int((latency_stats.p95_ms / self._latency_threshold) * 5))
            bottlenecks.append(BottleneckInfo(
                type=BottleneckType.UNKNOWN,
                severity=severity,
                description=f"High latency detected: p95={latency_stats.p95_ms:.2f}ms",
                affected_components=[component],
                suggested_actions=[
                    "Check for slow database queries",
                    "Review external service dependencies",
                    "Consider implementing caching"
                ]
            ))
        
        # Check CPU
        if cpu_percent > self._cpu_threshold:
            severity = min(10, int((cpu_percent / self._cpu_threshold) * 5))
            bottlenecks.append(BottleneckInfo(
                type=BottleneckType.CPU,
                severity=severity,
                description=f"High CPU usage: {cpu_percent:.1f}%",
                affected_components=[component],
                suggested_actions=[
                    "Profile CPU usage",
                    "Optimize hot paths",
                    "Consider scaling horizontally"
                ]
            ))
        
        # Check memory
        if memory_percent > self._memory_threshold:
            severity = min(10, int((memory_percent / self._memory_threshold) * 5))
            bottlenecks.append(BottleneckInfo(
                type=BottleneckType.MEMORY,
                severity=severity,
                description=f"High memory usage: {memory_percent:.1f}%",
                affected_components=[component],
                suggested_actions=[
                    "Check for memory leaks",
                    "Review object retention",
                    "Consider increasing memory limit"
                ]
            ))
        
        with self._lock:
            self._bottlenecks.extend(bottlenecks)
        
        return bottlenecks
    
    def get_bottlenecks(
        self,
        since: Optional[float] = None,
        min_severity: int = 1
    ) -> List[BottleneckInfo]:
        """
        获取瓶颈列表
        
        Args:
            since: 起始时间
            min_severity: 最小严重程度
            
        Returns:
            瓶颈列表
        """
        with self._lock:
            bottlenecks = self._bottlenecks.copy()
        
        if since:
            bottlenecks = [b for b in bottlenecks if b.timestamp >= since]
        
        bottlenecks = [b for b in bottlenecks if b.severity >= min_severity]
        
        return bottlenecks
    
    def clear_bottlenecks(self) -> None:
        """清除瓶颈记录"""
        with self._lock:
            self._bottlenecks.clear()


class PerformanceProfiler:
    """
    性能分析器
    
    提供延迟分析、吞吐量监控和瓶颈识别功能。
    
    Example:
        >>> config = PerformanceConfig()
        >>> profiler = PerformanceProfiler(config)
        >>> 
        >>> # Profile an operation
        >>> with profiler.time_operation("database_query"):
        ...     result = db.query()
        >>> 
        >>> # Record throughput
        >>> profiler.record_throughput(bytes_sent=1024)
        >>> 
        >>> # Get report
        >>> report = profiler.generate_report()
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        初始化性能分析器
        
        Args:
            config: 性能配置
        """
        self._config = config or PerformanceConfig()
        self._latency_profiler = LatencyProfiler()
        self._throughput_monitor = ThroughputMonitor(
            self._config.throughput_window_ms / 1000.0
        )
        self._bottleneck_detector = BottleneckDetector(
            latency_threshold_ms=self._config.bottleneck_threshold_ms
        )
        self._metrics_history: List[PerformanceMetrics] = []
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._running = False
        self._monitoring_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """启动性能分析"""
        self._running = True
        
        if self._config.enable_memory_profiling or self._config.enable_cpu_profiling:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
        
        logger.info("PerformanceProfiler started")
    
    def stop(self) -> None:
        """停止性能分析"""
        self._running = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("PerformanceProfiler stopped")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        import psutil
        
        while self._running:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                metrics = PerformanceMetrics(
                    latency_stats=self.get_latency_stats(),
                    throughput_stats=self.get_throughput_stats(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    active_threads=threading.active_count()
                )
                
                with self._lock:
                    self._metrics_history.append(metrics)
                    
                    # Limit history size
                    if len(self._metrics_history) > 1000:
                        self._metrics_history = self._metrics_history[-1000:]
                
                # Detect bottlenecks
                self._bottleneck_detector.analyze(
                    metrics.latency_stats,
                    cpu_percent,
                    memory.percent
                )
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self._config.profiling_interval_ms / 1000.0)
    
    def time_operation(self, operation: str) -> LatencyTimer:
        """
        创建操作计时器
        
        Args:
            operation: 操作名称
            
        Returns:
            计时器
        """
        return self._latency_profiler.time_operation(operation)
    
    def record_latency(self, operation: str, latency_ms: float) -> None:
        """
        记录延迟
        
        Args:
            operation: 操作名称
            latency_ms: 延迟（毫秒）
        """
        self._latency_profiler.record(operation, latency_ms)
    
    def record_throughput(self, byte_count: int = 0) -> None:
        """
        记录吞吐量
        
        Args:
            byte_count: 字节数
        """
        self._throughput_monitor.record_request(byte_count)
    
    def get_latency_stats(self, operation: Optional[str] = None) -> LatencyStats:
        """
        获取延迟统计
        
        Args:
            operation: 操作名称
            
        Returns:
            延迟统计
        """
        stats_dict = self._latency_profiler.get_stats(operation)
        
        if operation:
            return stats_dict.get(operation, LatencyStats())
        else:
            # Combine all stats
            all_samples = []
            for samples in self._latency_profiler._samples.values():
                all_samples.extend(samples)
            return LatencyStats.from_samples(all_samples)
    
    def get_throughput_stats(self) -> ThroughputStats:
        """获取吞吐量统计"""
        return self._throughput_monitor.get_stats()
    
    def get_bottlenecks(
        self,
        since: Optional[float] = None,
        min_severity: int = 1
    ) -> List[BottleneckInfo]:
        """
        获取瓶颈列表
        
        Args:
            since: 起始时间
            min_severity: 最小严重程度
            
        Returns:
            瓶颈列表
        """
        return self._bottleneck_detector.get_bottlenecks(since, min_severity)
    
    def generate_report(self) -> ProfileReport:
        """
        生成性能报告
        
        Returns:
            性能报告
        """
        end_time = time.time()
        
        with self._lock:
            metrics = self._metrics_history.copy()
        
        bottlenecks = self._bottleneck_detector.get_bottlenecks(
            since=self._start_time
        )
        
        # Calculate summary
        if metrics:
            avg_cpu = statistics.mean(m.cpu_percent for m in metrics)
            avg_memory = statistics.mean(m.memory_percent for m in metrics)
            max_latency = max(
                (m.latency_stats.p99_ms for m in metrics if m.latency_stats.count > 0),
                default=0
            )
        else:
            avg_cpu = 0.0
            avg_memory = 0.0
            max_latency = 0.0
        
        summary = {
            "average_cpu_percent": round(avg_cpu, 2),
            "average_memory_percent": round(avg_memory, 2),
            "max_p99_latency_ms": round(max_latency, 2),
            "total_bottlenecks": len(bottlenecks),
            "high_severity_bottlenecks": len([b for b in bottlenecks if b.severity >= 7])
        }
        
        return ProfileReport(
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=end_time - self._start_time,
            metrics=metrics,
            bottlenecks=bottlenecks,
            summary=summary
        )
    
    def reset(self) -> None:
        """重置分析器"""
        self._latency_profiler.clear()
        self._throughput_monitor.reset()
        self._bottleneck_detector.clear_bottlenecks()
        
        with self._lock:
            self._metrics_history.clear()
        
        self._start_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "latency_operations": list(self._latency_profiler._samples.keys()),
            "throughput": self.get_throughput_stats().to_dict(),
            "bottleneck_count": len(self._bottleneck_detector.get_bottlenecks()),
            "metrics_history_size": len(self._metrics_history)
        }


from collections import defaultdict
