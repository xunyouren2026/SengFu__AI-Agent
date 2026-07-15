"""
Metrics Collector Module

指标收集器实现，提供计数器、仪表盘、直方图等指标的收集和存储功能。
"""

from __future__ import annotations

import time
import threading
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..config import MetricsConfig

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


MetricValue = Union[int, float]


@dataclass
class MetricRecord:
    """
    指标记录
    
    Attributes:
        name: 指标名称
        value: 指标值
        labels: 标签字典
        timestamp: 时间戳
        metric_type: 指标类型
    """
    name: str
    value: MetricValue
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE
    
    def with_labels(self, **labels: str) -> "MetricRecord":
        """添加标签"""
        new_labels = {**self.labels, **labels}
        return MetricRecord(
            name=self.name,
            value=self.value,
            labels=new_labels,
            timestamp=self.timestamp,
            metric_type=self.metric_type
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "metric_type": self.metric_type.value
        }


class Metric(ABC):
    """指标抽象基类"""
    
    def __init__(self, name: str, description: str = "", unit: str = ""):
        self._name = name
        self._description = description
        self._unit = unit
        self._lock = threading.Lock()
    
    @property
    def name(self) -> str:
        """指标名称"""
        return self._name
    
    @property
    def description(self) -> str:
        """指标描述"""
        return self._description
    
    @property
    def unit(self) -> str:
        """指标单位"""
        return self._unit
    
    @abstractmethod
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """获取指标值"""
        pass
    
    @abstractmethod
    def to_records(self) -> List[MetricRecord]:
        """转换为记录列表"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清除指标数据"""
        pass


class Counter(Metric):
    """
    计数器
    
    单调递增的计数器，只能增加不能减少。
    适用于记录请求数、错误数等累计值。
    
    Example:
        >>> counter = Counter("requests_total", "Total requests")
        >>> counter.add(1, {"method": "GET", "status": "200"})
        >>> counter.add(1, {"method": "POST", "status": "201"})
    """
    
    def __init__(self, name: str, description: str = "", unit: str = "1"):
        super().__init__(name, description, unit)
        self._values: Dict[Tuple[Tuple[str, str], ...], int] = defaultdict(int)
    
    def add(self, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        增加计数
        
        Args:
            value: 增加值，必须为非负数
            labels: 标签字典
            
        Raises:
            ValueError: 如果value为负数
        """
        if value < 0:
            raise ValueError("Counter can only increase, value must be non-negative")
        
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_tuple] += value
    
    def inc(self, labels: Optional[Dict[str, str]] = None) -> None:
        """增加1"""
        self.add(1, labels)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """获取计数器值"""
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values[label_tuple]
    
    def get_all_values(self) -> Dict[Dict[str, str], int]:
        """获取所有标签组合的计数器值"""
        with self._lock:
            return {
                dict(k): v for k, v in self._values.items()
            }
    
    def to_records(self) -> List[MetricRecord]:
        """转换为记录列表"""
        with self._lock:
            return [
                MetricRecord(
                    name=self._name,
                    value=v,
                    labels=dict(k),
                    metric_type=MetricType.COUNTER
                )
                for k, v in self._values.items()
            ]
    
    def clear(self) -> None:
        """清除计数器数据"""
        with self._lock:
            self._values.clear()


class Gauge(Metric):
    """
    仪表盘
    
    可增可减的指标，表示瞬时值。
    适用于记录温度、内存使用量、队列长度等。
    
    Example:
        >>> gauge = Gauge("memory_usage_bytes", "Memory usage in bytes")
        >>> gauge.set(1024 * 1024 * 100)  # 100MB
        >>> gauge.inc(1024 * 1024)  # +1MB
        >>> gauge.dec(512 * 1024)   # -512KB
    """
    
    def __init__(self, name: str, description: str = "", unit: str = "1"):
        super().__init__(name, description, unit)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)
    
    def set(self, value: MetricValue, labels: Optional[Dict[str, str]] = None) -> None:
        """
        设置值
        
        Args:
            value: 要设置的值
            labels: 标签字典
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_tuple] = float(value)
    
    def inc(self, value: MetricValue = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        增加值
        
        Args:
            value: 增加值
            labels: 标签字典
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_tuple] += float(value)
    
    def dec(self, value: MetricValue = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        减少值
        
        Args:
            value: 减少值
            labels: 标签字典
        """
        self.inc(-float(value), labels)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """获取仪表盘值"""
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values[label_tuple]
    
    def get_all_values(self) -> Dict[Dict[str, str], float]:
        """获取所有标签组合的仪表盘值"""
        with self._lock:
            return {
                dict(k): v for k, v in self._values.items()
            }
    
    def to_records(self) -> List[MetricRecord]:
        """转换为记录列表"""
        with self._lock:
            return [
                MetricRecord(
                    name=self._name,
                    value=v,
                    labels=dict(k),
                    metric_type=MetricType.GAUGE
                )
                for k, v in self._values.items()
            ]
    
    def clear(self) -> None:
        """清除仪表盘数据"""
        with self._lock:
            self._values.clear()


class Histogram(Metric):
    """
    直方图
    
    采样观测值并计入配置的桶中，同时记录总和和计数。
    适用于记录请求延迟、响应大小等。
    
    Example:
        >>> histogram = Histogram(
        ...     "request_duration_seconds",
        ...     "Request duration",
        ...     buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ... )
        >>> histogram.observe(0.023, {"method": "GET"})
    """
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
        buckets: Optional[List[float]] = None
    ):
        super().__init__(name, description, unit)
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[Tuple[Tuple[str, str], ...], List[int]] = defaultdict(
            lambda: [0] * (len(self._buckets) + 1)
        )
        self._sums: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)
        self._total_counts: Dict[Tuple[Tuple[str, str], ...], int] = defaultdict(int)
    
    def observe(self, value: MetricValue, labels: Optional[Dict[str, str]] = None) -> None:
        """
        观测值
        
        Args:
            value: 观测值
            labels: 标签字典
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        
        with self._lock:
            # Find bucket
            bucket_idx = len(self._buckets)
            for i, bucket in enumerate(self._buckets):
                if value <= bucket:
                    bucket_idx = i
                    break
            
            self._counts[label_tuple][bucket_idx] += 1
            self._sums[label_tuple] += float(value)
            self._total_counts[label_tuple] += 1
    
    def time(self, labels: Optional[Dict[str, str]] = None) -> "HistogramTimer":
        """
        创建计时器上下文管理器
        
        Returns:
            HistogramTimer实例
        """
        return HistogramTimer(self, labels)
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """获取观测总数"""
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._total_counts[label_tuple]
    
    def get_bucket_counts(self, labels: Optional[Dict[str, str]] = None) -> List[int]:
        """获取桶计数"""
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._counts[label_tuple].copy()
    
    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取总和"""
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._sums[label_tuple]
    
    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """获取计数"""
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._total_counts[label_tuple]
    
    def get_buckets(self) -> List[float]:
        """获取桶边界"""
        return self._buckets.copy()
    
    def to_records(self) -> List[MetricRecord]:
        """转换为记录列表"""
        records = []
        with self._lock:
            for label_tuple in self._counts.keys():
                labels = dict(label_tuple)
                
                # Add sum record
                records.append(MetricRecord(
                    name=f"{self._name}_sum",
                    value=self._sums[label_tuple],
                    labels=labels,
                    metric_type=MetricType.HISTOGRAM
                ))
                
                # Add count record
                records.append(MetricRecord(
                    name=f"{self._name}_count",
                    value=self._total_counts[label_tuple],
                    labels=labels,
                    metric_type=MetricType.HISTOGRAM
                ))
                
                # Add bucket records
                cumulative_count = 0
                for i, bucket in enumerate(self._buckets):
                    cumulative_count += self._counts[label_tuple][i]
                    bucket_labels = {**labels, "le": str(bucket)}
                    records.append(MetricRecord(
                        name=f"{self._name}_bucket",
                        value=cumulative_count,
                        labels=bucket_labels,
                        metric_type=MetricType.HISTOGRAM
                    ))
                
                # Add +Inf bucket
                cumulative_count += self._counts[label_tuple][-1]
                inf_labels = {**labels, "le": "+Inf"}
                records.append(MetricRecord(
                    name=f"{self._name}_bucket",
                    value=cumulative_count,
                    labels=inf_labels,
                    metric_type=MetricType.HISTOGRAM
                ))
        
        return records
    
    def clear(self) -> None:
        """清除直方图数据"""
        with self._lock:
            self._counts.clear()
            self._sums.clear()
            self._total_counts.clear()


class HistogramTimer:
    """直方图计时器上下文管理器"""
    
    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        self._histogram = histogram
        self._labels = labels
        self._start_time: Optional[float] = None
    
    def __enter__(self) -> "HistogramTimer":
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time is not None:
            duration = time.time() - self._start_time
            self._histogram.observe(duration, self._labels)


class MetricsCollector:
    """
    指标收集器
    
    统一管理所有指标的收集、存储和导出。
    
    Example:
        >>> config = MetricsConfig()
        >>> collector = MetricsCollector(config)
        >>> 
        >>> # Create metrics
        >>> counter = collector.counter("requests_total", "Total requests")
        >>> gauge = collector.gauge("active_connections", "Active connections")
        >>> histogram = collector.histogram("request_duration", "Request duration")
        >>> 
        >>> # Record metrics
        >>> counter.inc({"method": "GET"})
        >>> gauge.set(100)
        >>> histogram.observe(0.023)
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        初始化指标收集器
        
        Args:
            config: 指标配置
        """
        self._config = config or MetricsConfig()
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        self._running = False
        self._collect_callbacks: List[Callable[[], None]] = []
        self._exporters: List[Any] = []
        self._collection_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """启动收集器"""
        self._running = True
        self._start_collection_loop()
        logger.info("MetricsCollector started")
    
    def shutdown(self) -> None:
        """关闭收集器"""
        self._running = False
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)
        self._export_metrics()
        logger.info("MetricsCollector shutdown")
    
    def _start_collection_loop(self) -> None:
        """启动收集循环"""
        def collection_loop():
            while self._running:
                time.sleep(self._config.collection_interval_ms / 1000)
                if self._running:
                    self._collect()
        
        self._collection_thread = threading.Thread(
            target=collection_loop,
            daemon=True
        )
        self._collection_thread.start()
    
    def _collect(self) -> None:
        """执行收集"""
        for callback in self._collect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Collection callback failed: {e}")
    
    def _export_metrics(self) -> None:
        """导出指标"""
        records = self.get_all_records()
        for exporter in self._exporters:
            try:
                exporter.export(records)
            except Exception as e:
                logger.error(f"Export failed: {e}")
    
    def register_metric(self, metric: Metric) -> Metric:
        """
        注册指标
        
        Args:
            metric: 指标实例
            
        Returns:
            注册的指标
        """
        with self._lock:
            if len(self._metrics) >= self._config.max_metrics:
                raise RuntimeError(f"Maximum number of metrics ({self._config.max_metrics}) reached")
            self._metrics[metric.name] = metric
        return metric
    
    def counter(
        self,
        name: str,
        description: str = "",
        unit: str = "1"
    ) -> Counter:
        """
        创建或获取计数器
        
        Args:
            name: 计数器名称
            description: 描述
            unit: 单位
            
        Returns:
            Counter实例
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Counter):
                    raise TypeError(f"Metric {name} exists but is not a Counter")
                return metric
        
        counter = Counter(name, description, unit)
        return self.register_metric(counter)
    
    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "1"
    ) -> Gauge:
        """
        创建或获取仪表盘
        
        Args:
            name: 仪表盘名称
            description: 描述
            unit: 单位
            
        Returns:
            Gauge实例
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Gauge):
                    raise TypeError(f"Metric {name} exists but is not a Gauge")
                return metric
        
        gauge = Gauge(name, description, unit)
        return self.register_metric(gauge)
    
    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """
        创建或获取直方图
        
        Args:
            name: 直方图名称
            description: 描述
            unit: 单位
            buckets: 桶边界
            
        Returns:
            Histogram实例
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Histogram):
                    raise TypeError(f"Metric {name} exists but is not a Histogram")
                return metric
        
        if buckets is None and self._config.enable_histogram:
            buckets = self._config.histogram_buckets
        
        histogram = Histogram(name, description, unit, buckets)
        return self.register_metric(histogram)
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        获取指标
        
        Args:
            name: 指标名称
            
        Returns:
            指标实例或None
        """
        with self._lock:
            return self._metrics.get(name)
    
    def remove_metric(self, name: str) -> bool:
        """
        移除指标
        
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
    
    def record(
        self,
        name: str,
        value: MetricValue,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        记录指标值（通用接口）
        
        Args:
            name: 指标名称
            value: 指标值
            labels: 标签
        """
        metric = self.get_metric(name)
        if metric:
            if isinstance(metric, Counter):
                metric.add(int(value), labels)
            elif isinstance(metric, Gauge):
                metric.set(value, labels)
            elif isinstance(metric, Histogram):
                metric.observe(value, labels)
        else:
            # Auto-create gauge if not exists
            gauge = self.gauge(name)
            gauge.set(value, labels)
    
    def get_all_records(self) -> List[MetricRecord]:
        """
        获取所有指标记录
        
        Returns:
            指标记录列表
        """
        records = []
        with self._lock:
            for metric in self._metrics.values():
                records.extend(metric.to_records())
        return records
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """
        获取所有指标
        
        Returns:
            指标字典
        """
        with self._lock:
            return self._metrics.copy()
    
    def clear(self) -> None:
        """清除所有指标数据"""
        with self._lock:
            for metric in self._metrics.values():
                metric.clear()
    
    def add_collection_callback(self, callback: Callable[[], None]) -> None:
        """
        添加收集回调
        
        Args:
            callback: 回调函数
        """
        self._collect_callbacks.append(callback)
    
    def remove_collection_callback(self, callback: Callable[[], None]) -> bool:
        """
        移除收集回调
        
        Args:
            callback: 回调函数
            
        Returns:
            是否成功移除
        """
        if callback in self._collect_callbacks:
            self._collect_callbacks.remove(callback)
            return True
        return False
    
    def add_exporter(self, exporter: Any) -> None:
        """
        添加导出器
        
        Args:
            exporter: 导出器实例
        """
        self._exporters.append(exporter)
    
    def remove_exporter(self, exporter: Any) -> bool:
        """
        移除导出器
        
        Args:
            exporter: 导出器实例
            
        Returns:
            是否成功移除
        """
        if exporter in self._exporters:
            self._exporters.remove(exporter)
            return True
        return False
