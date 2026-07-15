"""
指标收集器模块

提供纯Python实现的指标收集功能，模拟Prometheus的核心指标类型。
支持计数器、直方图、仪表盘、计时器等指标类型。
"""

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Counter: 计数器
# ---------------------------------------------------------------------------

class Counter:
    """单调递增计数器。

    计数器只能增加或重置，不能减少。
    适用于统计请求总数、错误总数等场景。

    使用示例::

        counter = Counter("request_count", "Total number of requests")
        counter.inc()
        counter.inc(5)
        print(counter.get())  # 6
    """

    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        """初始化计数器。

        Args:
            name: 指标名称
            description: 指标描述
            tags: 标签字典
        """
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value: float = 0.0
        self._lock = threading.Lock()
        self._created_at = time.time()

    def inc(self, amount: float = 1.0) -> None:
        """增加计数器值。

        Args:
            amount: 增加量，必须为正数

        Raises:
            ValueError: 如果amount为负数
        """
        if amount < 0:
            raise ValueError(f"Counter can only increase, got negative amount: {amount}")
        with self._lock:
            self._value += amount

    def reset(self) -> None:
        """重置计数器为0。"""
        with self._lock:
            self._value = 0.0

    def get(self) -> float:
        """获取当前值。

        Returns:
            当前计数值
        """
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "type": "counter",
            "name": self.name,
            "description": self.description,
            "tags": dict(self.tags),
            "value": self.get(),
        }


# ---------------------------------------------------------------------------
# Gauge: 仪表盘
# ---------------------------------------------------------------------------

class Gauge:
    """仪表盘，表示可以任意上下波动的值。

    适用于统计当前活跃请求数、CPU使用率、内存使用量等场景。

    使用示例::

        gauge = Gauge("active_requests", "Current active requests")
        gauge.inc()
        gauge.dec()
        gauge.set(42)
    """

    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        """初始化仪表盘。

        Args:
            name: 指标名称
            description: 指标描述
            tags: 标签字典
        """
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value: float = 0.0
        self._lock = threading.Lock()
        self._created_at = time.time()

    def set(self, value: float) -> None:
        """设置当前值。

        Args:
            value: 新值
        """
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """增加当前值。

        Args:
            amount: 增加量
        """
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """减少当前值。

        Args:
            amount: 减少量
        """
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        """获取当前值。

        Returns:
            当前值
        """
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "type": "gauge",
            "name": self.name,
            "description": self.description,
            "tags": dict(self.tags),
            "value": self.get(),
        }


# ---------------------------------------------------------------------------
# Histogram: 直方图
# ---------------------------------------------------------------------------

class Histogram:
    """直方图，用于观察值的分布。

    自动计算分位数（百分位数）、均值、最大值、最小值等统计信息。
    适用于统计请求延迟、响应大小等场景。

    使用示例::

        hist = Histogram("request_latency", "Request latency in seconds")
        for _ in range(100):
            hist.observe(0.1 + random.random() * 0.5)
        print(hist.get_percentile(50))  # 中位数
        print(hist.get_percentile(99))  # P99
        print(hist.get_mean())
    """

    # 默认分位数桶边界
    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'),
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """初始化直方图。

        Args:
            name: 指标名称
            description: 指标描述
            buckets: 分桶边界，默认使用 DEFAULT_BUCKETS
            tags: 标签字典
        """
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._buckets = tuple(sorted(buckets or self.DEFAULT_BUCKETS))
        # 每个桶的计数
        self._bucket_counts: List[float] = [0.0] * (len(self._buckets) + 1)
        self._sum: float = 0.0
        self._count: int = 0
        # 保存所有观察值用于精确分位数计算
        self._observations: List[float] = []
        self._lock = threading.Lock()
        self._max_value: float = float('-inf')
        self._min_value: float = float('inf')

    def observe(self, value: float) -> None:
        """记录一个观察值。

        Args:
            value: 观察值
        """
        if value < 0:
            raise ValueError(f"Histogram value must be non-negative, got: {value}")
        with self._lock:
            self._sum += value
            self._count += 1
            self._observations.append(value)
            if value > self._max_value:
                self._max_value = value
            if value < self._min_value:
                self._min_value = value
            # 更新桶计数
            for i, boundary in enumerate(self._buckets):
                if value <= boundary:
                    self._bucket_counts[i] += 1
                    break
            else:
                # 超过所有桶边界
                self._bucket_counts[-1] += 1

    def get_percentile(self, percentile: float) -> float:
        """获取指定百分位数。

        Args:
            percentile: 百分位数（0-100），如50表示中位数，99表示P99

        Returns:
            指定百分位数的值
        """
        if not (0 <= percentile <= 100):
            raise ValueError(f"Percentile must be between 0 and 100, got: {percentile}")
        with self._lock:
            if not self._observations:
                return 0.0
            sorted_obs = sorted(self._observations)
            idx = (percentile / 100.0) * (len(sorted_obs) - 1)
            lower = int(math.floor(idx))
            upper = int(math.ceil(idx))
            if lower == upper:
                return sorted_obs[lower]
            # 线性插值
            fraction = idx - lower
            return sorted_obs[lower] + fraction * (sorted_obs[upper] - sorted_obs[lower])

    def get_mean(self) -> float:
        """获取均值。

        Returns:
            所有观察值的平均值
        """
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._sum / self._count

    def get_max(self) -> float:
        """获取最大值。

        Returns:
            所有观察值中的最大值
        """
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._max_value

    def get_min(self) -> float:
        """获取最小值。

        Returns:
            所有观察值中的最小值
        """
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._min_value

    def get_sum(self) -> float:
        """获取总和。

        Returns:
            所有观察值的总和
        """
        with self._lock:
            return self._sum

    def get_count(self) -> int:
        """获取观察次数。

        Returns:
            观察值总数
        """
        with self._lock:
            return self._count

    def get_stddev(self) -> float:
        """获取标准差。

        Returns:
            所有观察值的标准差
        """
        with self._lock:
            if self._count < 2:
                return 0.0
            mean = self._sum / self._count
            variance = sum((x - mean) ** 2 for x in self._observations) / self._count
            return math.sqrt(variance)

    def get_bucket_counts(self) -> Dict[str, float]:
        """获取各桶的累计计数。

        Returns:
            桶边界到累计计数的映射
        """
        with self._lock:
            result = {}
            cumulative = 0.0
            for i, boundary in enumerate(self._buckets):
                cumulative += self._bucket_counts[i]
                if boundary == float('inf'):
                    key = "+Inf"
                else:
                    key = str(boundary)
                result[key] = cumulative
            return result

    def reset(self) -> None:
        """重置直方图。"""
        with self._lock:
            self._bucket_counts = [0.0] * (len(self._buckets) + 1)
            self._sum = 0.0
            self._count = 0
            self._observations.clear()
            self._max_value = float('-inf')
            self._min_value = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "type": "histogram",
            "name": self.name,
            "description": self.description,
            "tags": dict(self.tags),
            "count": self.get_count(),
            "sum": round(self.get_sum(), 6),
            "mean": round(self.get_mean(), 6),
            "max": round(self.get_max(), 6),
            "min": round(self.get_min(), 6),
            "stddev": round(self.get_stddev(), 6),
            "p50": round(self.get_percentile(50), 6),
            "p90": round(self.get_percentile(90), 6),
            "p95": round(self.get_percentile(95), 6),
            "p99": round(self.get_percentile(99), 6),
            "buckets": self.get_bucket_counts(),
        }


# ---------------------------------------------------------------------------
# Timer: 计时器
# ---------------------------------------------------------------------------

class Timer:
    """计时器，基于Histogram实现。

    用于方便地测量操作耗时。

    使用示例::

        timer = Timer("operation_duration", "Operation duration")
        with timer:
            time.sleep(0.1)
        print(timer.histogram.get_mean())  # ~0.1
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """初始化计时器。

        Args:
            name: 指标名称
            description: 指标描述
            buckets: 分桶边界
            tags: 标签字典
        """
        self.histogram = Histogram(
            name=name,
            description=description or f"Timer for {name}",
            buckets=buckets,
            tags=tags,
        )
        self._start_time: Optional[float] = None

    def start(self) -> "Timer":
        """开始计时。

        Returns:
            self（支持链式调用）
        """
        self._start_time = time.time()
        return self

    def stop(self) -> float:
        """停止计时并记录观察值。

        Returns:
            计时时间（秒）
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        elapsed = time.time() - self._start_time
        self.histogram.observe(elapsed)
        self._start_time = None
        return elapsed

    def __enter__(self) -> "Timer":
        """支持上下文管理器。"""
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出上下文时自动停止计时。"""
        self.stop()


# ---------------------------------------------------------------------------
# MetricsRegistry: 指标注册表
# ---------------------------------------------------------------------------

class MetricsRegistry:
    """指标注册表。

    管理所有已注册的指标，提供按名称查找和批量导出功能。
    """

    def __init__(self):
        """初始化指标注册表。"""
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, metric: Any) -> None:
        """注册一个指标。

        Args:
            metric: 指标对象（Counter, Gauge, Histogram, Timer等）

        Raises:
            ValueError: 如果同名指标已存在
        """
        with self._lock:
            name = metric.name
            if name in self._metrics:
                raise ValueError(f"Metric '{name}' is already registered")
            self._metrics[name] = metric

    def get(self, name: str) -> Optional[Any]:
        """获取指定名称的指标。

        Args:
            name: 指标名称

        Returns:
            指标对象，如果不存在则返回None
        """
        with self._lock:
            return self._metrics.get(name)

    def get_all(self) -> Dict[str, Any]:
        """获取所有已注册指标的副本。

        Returns:
            指标名称到指标对象的映射
        """
        with self._lock:
            return dict(self._metrics)

    def remove(self, name: str) -> bool:
        """移除指定名称的指标。

        Args:
            name: 指标名称

        Returns:
            是否成功移除
        """
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                return True
            return False

    def clear(self) -> None:
        """清空所有已注册的指标。"""
        with self._lock:
            self._metrics.clear()

    def names(self) -> List[str]:
        """获取所有已注册的指标名称。

        Returns:
            指标名称列表
        """
        with self._lock:
            return list(self._metrics.keys())

    def export_all(self) -> Dict[str, Dict[str, Any]]:
        """导出所有指标为字典。

        Returns:
            指标名称到指标字典表示的映射
        """
        with self._lock:
            result = {}
            for name, metric in self._metrics.items():
                if hasattr(metric, 'to_dict'):
                    result[name] = metric.to_dict()
                elif isinstance(metric, Timer):
                    result[name] = metric.histogram.to_dict()
                else:
                    result[name] = {"name": name, "value": str(metric)}
            return result


# ---------------------------------------------------------------------------
# MetricsCollector: 指标收集器
# ---------------------------------------------------------------------------

class MetricsCollector:
    """指标收集器。

    提供统一的指标管理接口，内置常用指标，支持注册自定义指标和批量导出。

    内置指标:
    - request_count: 请求总数（Counter）
    - request_latency: 请求延迟（Histogram）
    - error_count: 错误总数（Counter）
    - active_requests: 当前活跃请求数（Gauge）
    - token_usage: Token使用量（Counter）
    - gpu_utilization: GPU利用率（Gauge）

    使用示例::

        collector = MetricsCollector()
        collector.request_count.inc()
        collector.active_requests.inc()
        # ... 处理请求
        collector.active_requests.dec()
        collector.request_latency.observe(0.123)

        # 导出所有指标
        metrics = collector.export_metrics()
    """

    def __init__(self, service_name: str = "unknown"):
        """初始化指标收集器。

        Args:
            service_name: 服务名称，用于标签
        """
        self.service_name = service_name
        self._registry = MetricsRegistry()
        self._init_builtin_metrics()

    def _init_builtin_metrics(self) -> None:
        """初始化内置指标。"""
        service_tag = {"service": self.service_name}

        # 请求总数
        self._registry.register(Counter(
            name="request_count",
            description="Total number of requests",
            tags=service_tag,
        ))

        # 请求延迟
        self._registry.register(Histogram(
            name="request_latency",
            description="Request latency in seconds",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')),
            tags=service_tag,
        ))

        # 错误总数
        self._registry.register(Counter(
            name="error_count",
            description="Total number of errors",
            tags=service_tag,
        ))

        # 活跃请求数
        self._registry.register(Gauge(
            name="active_requests",
            description="Current number of active requests",
            tags=service_tag,
        ))

        # Token使用量
        self._registry.register(Counter(
            name="token_usage",
            description="Total token usage",
            tags=service_tag,
        ))

        # GPU利用率
        self._registry.register(Gauge(
            name="gpu_utilization",
            description="GPU utilization percentage (0-100)",
            tags=service_tag,
        ))

    # ------------------------------------------------------------------
    # 便捷属性访问内置指标
    # ------------------------------------------------------------------

    @property
    def request_count(self) -> Counter:
        """请求总数计数器。"""
        return self._registry.get("request_count")

    @property
    def request_latency(self) -> Histogram:
        """请求延迟直方图。"""
        return self._registry.get("request_latency")

    @property
    def error_count(self) -> Counter:
        """错误总数计数器。"""
        return self._registry.get("error_count")

    @property
    def active_requests(self) -> Gauge:
        """活跃请求数仪表盘。"""
        return self._registry.get("active_requests")

    @property
    def token_usage(self) -> Counter:
        """Token使用量计数器。"""
        return self._registry.get("token_usage")

    @property
    def gpu_utilization(self) -> Gauge:
        """GPU利用率仪表盘。"""
        return self._registry.get("gpu_utilization")

    # ------------------------------------------------------------------
    # 指标注册与查询
    # ------------------------------------------------------------------

    def register_metric(self, metric: Any) -> None:
        """注册自定义指标。

        Args:
            metric: 指标对象（Counter, Gauge, Histogram, Timer等）

        Raises:
            ValueError: 如果同名指标已存在
        """
        self._registry.register(metric)

    def get_metric(self, name: str) -> Optional[Any]:
        """获取指定名称的指标。

        Args:
            name: 指标名称

        Returns:
            指标对象，如果不存在则返回None
        """
        return self._registry.get(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有已注册指标的快照。

        Returns:
            指标名称到指标对象的映射
        """
        return self._registry.get_all()

    def export_metrics(self) -> Dict[str, Dict[str, Any]]:
        """导出所有指标为字典格式。

        Returns:
            指标名称到指标字典表示的映射
        """
        return self._registry.export_all()

    def create_counter(
        self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """创建并注册一个计数器。

        Args:
            name: 指标名称
            description: 指标描述
            tags: 标签字典

        Returns:
            创建的Counter实例

        Raises:
            ValueError: 如果同名指标已存在
        """
        counter = Counter(name=name, description=description, tags=tags)
        self._registry.register(counter)
        return counter

    def create_gauge(
        self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """创建并注册一个仪表盘。

        Args:
            name: 指标名称
            description: 指标描述
            tags: 标签字典

        Returns:
            创建的Gauge实例

        Raises:
            ValueError: 如果同名指标已存在
        """
        gauge = Gauge(name=name, description=description, tags=tags)
        self._registry.register(gauge)
        return gauge

    def create_histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Histogram:
        """创建并注册一个直方图。

        Args:
            name: 指标名称
            description: 指标描述
            buckets: 分桶边界
            tags: 标签字典

        Returns:
            创建的Histogram实例

        Raises:
            ValueError: 如果同名指标已存在
        """
        hist = Histogram(name=name, description=description, buckets=buckets, tags=tags)
        self._registry.register(hist)
        return hist

    def create_timer(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Timer:
        """创建并注册一个计时器。

        Args:
            name: 指标名称
            description: 指标描述
            buckets: 分桶边界
            tags: 标签字典

        Returns:
            创建的Timer实例

        Raises:
            ValueError: 如果同名指标已存在
        """
        timer = Timer(name=name, description=description, buckets=buckets, tags=tags)
        self._registry.register(timer)
        return timer

    def reset_all(self) -> None:
        """重置所有指标。"""
        for name, metric in self._registry.get_all().items():
            if hasattr(metric, 'reset'):
                metric.reset()
            elif isinstance(metric, Timer):
                metric.histogram.reset()

    def get_metric_names(self) -> List[str]:
        """获取所有已注册的指标名称。

        Returns:
            指标名称列表
        """
        return self._registry.names()
