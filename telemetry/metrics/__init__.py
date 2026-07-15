"""
Metrics Module

指标收集和聚合模块，提供计数器、仪表盘、直方图等指标的
收集、存储和聚合功能。
"""

from .collector import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    MetricType,
    MetricValue,
    MetricRecord,
)

from .aggregator import (
    MetricsAggregator,
    AggregationType,
    TimeWindow,
    AggregatedMetric,
    AggregationResult,
)

__all__ = [
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricType",
    "MetricValue",
    "MetricRecord",
    "MetricsAggregator",
    "AggregationType",
    "TimeWindow",
    "AggregatedMetric",
    "AggregationResult",
]
