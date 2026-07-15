"""
Custom Metrics Registry Module

This module provides a custom metrics registry for the AGI Unified Framework,
supporting Prometheus-compatible metric types and export formats.

Features:
- Metric registration with type validation
- Gauge, Counter, and Histogram metric types
- Aggregation strategies (sum, count, quantile)
- Prometheus export format
- Metric family management
- Thread-safe operations
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod
import threading
import time
import math
import re
import copy
import logging


logger = logging.getLogger(__name__)

# Type aliases
Labels = Dict[str, str]
Value = Union[int, float]


class MetricType(Enum):
    """Supported metric types."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()
    UNKNOWN = auto()


class AggregationType(Enum):
    """Supported aggregation types."""
    SUM = auto()
    COUNT = auto()
    MIN = auto()
    MAX = auto()
    AVG = auto()
    QUANTILE = auto()


@dataclass
class MetricFamily:
    """
    Metric family definition.
    
    A metric family is a group of related metrics with the same name
    and different label combinations.
    """
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    help: str = ""
    
    def __post_init__(self) -> None:
        """Validate metric family after initialization."""
        if not self.name:
            raise ValueError("Metric name cannot be empty")
        
        # Validate name format (Prometheus naming convention)
        if not re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$', self.name):
            raise ValueError(f"Invalid metric name: {self.name}")
        
        # Validate label names
        for label in self.labels:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label):
                raise ValueError(f"Invalid label name: {label}")
    
    def to_prometheus(self) -> str:
        """Generate Prometheus exposition format."""
        lines = []
        
        # Build the header comment with HELP and TYPE
        if self.help:
            lines.append(f"# HELP {self.name} {self.help}")
        if self.unit:
            lines.append(f"# UNIT {self.name} {self.unit}")
        
        type_name = self.metric_type.name.lower()
        lines.append(f"# TYPE {self.name} {type_name}")
        
        return "\n".join(lines)
    
    def validate_labels(self, labels: Labels) -> bool:
        """Validate that labels match the family definition."""
        for label in self.labels:
            if label not in labels:
                return False
        return True


@dataclass
class Observation:
    """A single observation of a metric."""
    value: Value
    timestamp: float = field(default_factory=time.time)
    labels: Labels = field(default_factory=dict)


class AggregationStrategy(ABC):
    """
    Abstract base class for aggregation strategies.
    
    Defines how multiple observations are combined.
    """
    
    @abstractmethod
    def aggregate(self, values: List[Value]) -> Value:
        """Aggregate multiple values."""
        pass
    
    @abstractmethod
    def merge(
        self,
        existing: Optional[Value],
        new_value: Value,
    ) -> Value:
        """Merge a new value with existing aggregated value."""
        pass


class SumAggregation(AggregationStrategy):
    """Sum aggregation strategy."""
    
    def aggregate(self, values: List[Value]) -> Value:
        if not values:
            return 0
        return sum(values)
    
    def merge(self, existing: Optional[Value], new_value: Value) -> Value:
        return (existing or 0) + new_value


class CountAggregation(AggregationStrategy):
    """Count aggregation strategy."""
    
    def aggregate(self, values: List[Value]) -> Value:
        return len(values)
    
    def merge(self, existing: Optional[Value], new_value: Value) -> Value:
        return (existing or 0) + 1


class MinAggregation(AggregationStrategy):
    """Minimum aggregation strategy."""
    
    def aggregate(self, values: List[Value]) -> Value:
        if not values:
            return 0
        return min(values)
    
    def merge(self, existing: Optional[Value], new_value: Value) -> Value:
        if existing is None:
            return new_value
        return min(existing, new_value)


class MaxAggregation(AggregationStrategy):
    """Maximum aggregation strategy."""
    
    def aggregate(self, values: List[Value]) -> Value:
        if not values:
            return 0
        return max(values)
    
    def merge(self, existing: Optional[Value], new_value: Value) -> Value:
        if existing is None:
            return new_value
        return max(existing, new_value)


class AverageAggregation(AggregationStrategy):
    """Average aggregation strategy."""
    
    def aggregate(self, values: List[Value]) -> Value:
        if not values:
            return 0
        return sum(values) / len(values)
    
    def merge(
        self,
        existing: Optional[Value],
        new_value: Value,
    ) -> Value:
        # This is a simplified merge - for true average we'd need count and sum
        return (existing or 0) + new_value


class QuantileAggregation(AggregationStrategy):
    """Quantile aggregation strategy."""
    
    def __init__(self, quantile: float = 0.5) -> None:
        """Initialize with quantile (0.0 to 1.0)."""
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")
        self.quantile = quantile
    
    def aggregate(self, values: List[Value]) -> Value:
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * self.quantile)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def merge(self, existing: Optional[Value], new_value: Value) -> Value:
        # For quantile, we can't incrementally merge
        # This is a placeholder that just keeps the new value
        return new_value


def create_aggregation(
    agg_type: AggregationType,
    **kwargs: Any,
) -> AggregationStrategy:
    """
    Create an aggregation strategy.
    
    Args:
        agg_type: Type of aggregation
        **kwargs: Additional arguments for specific aggregations
        
    Returns:
        AggregationStrategy instance
    """
    if agg_type == AggregationType.SUM:
        return SumAggregation()
    elif agg_type == AggregationType.COUNT:
        return CountAggregation()
    elif agg_type == AggregationType.MIN:
        return MinAggregation()
    elif agg_type == AggregationType.MAX:
        return MaxAggregation()
    elif agg_type == AggregationType.AVG:
        return AverageAggregation()
    elif agg_type == AggregationType.QUANTILE:
        return QuantileAggregation(kwargs.get("quantile", 0.5))
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")


class Metric(ABC):
    """
    Abstract base class for metrics.
    
    All metric types (Counter, Gauge, Histogram) inherit from this class.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        unit: str = "",
    ) -> None:
        """
        Initialize metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            unit: Unit of measurement
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self.unit = unit
        self._created_at = time.time()
        self._last_updated = time.time()
        self._lock = threading.RLock()
        
        # Validate name
        if not re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$', name):
            raise ValueError(f"Invalid metric name: {name}")
    
    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Get the metric type."""
        pass
    
    @abstractmethod
    def get_value(self, labels: Labels) -> Value:
        """Get current value for given labels."""
        pass
    
    @abstractmethod
    def get_all_values(self) -> List[Tuple[Labels, Value]]:
        """Get all label combinations and their values."""
        pass
    
    @abstractmethod
    def reset(self, labels: Optional[Labels] = None) -> None:
        """Reset the metric."""
        pass
    
    def to_prometheus(
        self,
        value: Value,
        labels: Labels,
        timestamp: Optional[float] = None,
    ) -> str:
        """Convert metric to Prometheus exposition format."""
        label_str = self._format_labels(labels)
        
        suffix = ""
        if self.unit:
            suffix = f"_{self.unit}"
        
        name = f"{self.name}{suffix}"
        
        if timestamp:
            return f"{name}{label_str} {value} {int(timestamp * 1000)}"
        return f"{name}{label_str} {value}"
    
    def _format_labels(self, labels: Labels) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"
    
    def _validate_labels(self, labels: Labels) -> None:
        """Validate labels against metric definition."""
        for label in self.labels:
            if label not in labels:
                raise ValueError(f"Missing required label: {label}")
        
        # Check for unexpected labels
        for label in labels:
            if label not in self.labels:
                raise ValueError(f"Unexpected label: {label}")


class GaugeMetric(Metric):
    """
    Gauge metric type.
    
    A gauge represents a value that can go up or down.
    Examples: current memory usage, temperature, queue size.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        unit: str = "",
        initial_value: Value = 0,
    ) -> None:
        """Initialize gauge metric."""
        super().__init__(name, description, labels, unit)
        self._values: Dict[str, Value] = {}
        self._label_keys: Tuple[str, ...] = tuple(labels) if labels else ()
        
        if initial_value != 0:
            self._values[""] = initial_value
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE
    
    def _get_key(self, labels: Labels) -> str:
        """Generate a key for label combination."""
        if not labels:
            return ""
        return "|".join(f"{k}={labels.get(k, '')}" for k in self._label_keys)
    
    def set(self, value: Value, labels: Optional[Labels] = None) -> None:
        """
        Set the gauge to a specific value.
        
        Args:
            value: New value
            labels: Label combination
        """
        with self._lock:
            self._validate_labels(labels or {})
            key = self._get_key(labels or {})
            self._values[key] = value
            self._last_updated = time.time()
    
    def inc(self, amount: Value = 1, labels: Optional[Labels] = None) -> None:
        """
        Increment the gauge.
        
        Args:
            amount: Amount to increment by
            labels: Label combination
        """
        with self._lock:
            self._validate_labels(labels or {})
            key = self._get_key(labels or {})
            current = self._values.get(key, 0)
            self._values[key] = current + amount
            self._last_updated = time.time()
    
    def dec(self, amount: Value = 1, labels: Optional[Labels] = None) -> None:
        """
        Decrement the gauge.
        
        Args:
            amount: Amount to decrement by
            labels: Label combination
        """
        self.inc(-amount, labels)
    
    def get_value(self, labels: Labels) -> Value:
        """Get current value for given labels."""
        with self._lock:
            key = self._get_key(labels)
            return self._values.get(key, 0)
    
    def get_all_values(self) -> List[Tuple[Labels, Value]]:
        """Get all label combinations and their values."""
        with self._lock:
            results = []
            for key, value in self._values.items():
                if key == "":
                    results.append(({}, value))
                else:
                    labels = {}
                    for part in key.split("|"):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            labels[k] = v
                    results.append((labels, value))
            return results
    
    def reset(self, labels: Optional[Labels] = None) -> None:
        """Reset the gauge."""
        with self._lock:
            if labels is None:
                self._values.clear()
            else:
                key = self._get_key(labels)
                self._values.pop(key, None)
            self._last_updated = time.time()


class CounterMetric(Metric):
    """
    Counter metric type.
    
    A counter monotonically increases over time.
    It can only be incremented, and resets on process restart.
    Examples: total requests, total errors.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        unit: str = "",
        initial_value: Value = 0,
    ) -> None:
        """Initialize counter metric."""
        super().__init__(name, description, labels, unit)
        self._values: Dict[str, Value] = {}
        self._label_keys: Tuple[str, ...] = tuple(labels) if labels else ()
        
        if initial_value > 0:
            self._values[""] = initial_value
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER
    
    def _get_key(self, labels: Labels) -> str:
        """Generate a key for label combination."""
        if not labels:
            return ""
        return "|".join(f"{k}={labels.get(k, '')}" for k in self._label_keys)
    
    def inc(self, amount: Value = 1, labels: Optional[Labels] = None) -> None:
        """
        Increment the counter.
        
        Args:
            amount: Amount to increment by
            labels: Label combination
        """
        with self._lock:
            self._validate_labels(labels or {})
            key = self._get_key(labels or {})
            current = self._values.get(key, 0)
            
            if current < 0:
                logger.warning(f"Counter {self.name} has negative value, resetting to 0")
                current = 0
            
            self._values[key] = current + amount
            self._last_updated = time.time()
    
    def get_value(self, labels: Labels) -> Value:
        """Get current value for given labels."""
        with self._lock:
            key = self._get_key(labels)
            return max(self._values.get(key, 0), 0)
    
    def get_all_values(self) -> List[Tuple[Labels, Value]]:
        """Get all label combinations and their values."""
        with self._lock:
            results = []
            for key, value in self._values.items():
                if key == "":
                    results.append(({}, max(value, 0)))
                else:
                    labels = {}
                    for part in key.split("|"):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            labels[k] = v
                    results.append((labels, max(value, 0)))
            return results
    
    def reset(self, labels: Optional[Labels] = None) -> None:
        """Reset the counter."""
        with self._lock:
            if labels is None:
                self._values.clear()
            else:
                key = self._get_key(labels)
                self._values.pop(key, None)
            self._last_updated = time.time()


class HistogramMetric(Metric):
    """
    Histogram metric type.
    
    A histogram samples observations and counts them in configurable buckets.
    Examples: request latency, response size.
    """
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        unit: str = "",
        buckets: Tuple[float, ...] = DEFAULT_BUCKETS,
        max_value: Optional[float] = None,
    ) -> None:
        """
        Initialize histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Label names
            unit: Unit of measurement
            buckets: Bucket upper bounds
            max_value: Maximum possible value (for +Inf bucket)
        """
        super().__init__(name, description, labels, unit)
        self.buckets = tuple(sorted(buckets))
        self.max_value = max_value
        self._counts: Dict[str, Dict[str, int]] = {}
        self._sums: Dict[str, float] = {}
        self._totals: Dict[str, int] = {}
        self._label_keys: Tuple[str, ...] = tuple(labels) if labels else ()
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM
    
    def _get_key(self, labels: Labels) -> str:
        """Generate a key for label combination."""
        if not labels:
            return ""
        return "|".join(f"{k}={labels.get(k, '')}" for k in self._label_keys)
    
    def observe(self, value: float, labels: Optional[Labels] = None) -> None:
        """
        Record an observation.
        
        Args:
            value: Observed value
            labels: Label combination
        """
        with self._lock:
            self._validate_labels(labels or {})
            key = self._get_key(labels or {})
            
            # Initialize if needed
            if key not in self._counts:
                self._counts[key] = {f"+Inf": 0}
                for bucket in self.buckets:
                    self._counts[key][str(bucket)] = 0
                self._sums[key] = 0.0
                self._totals[key] = 0
            
            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][str(bucket)] += 1
                self._counts[key]["+Inf"] += 1
            
            # Update sum and count
            self._sums[key] += value
            self._totals[key] += 1
            self._last_updated = time.time()
    
    def get_value(self, labels: Labels) -> Value:
        """Get total count for given labels."""
        with self._lock:
            key = self._get_key(labels)
            return self._totals.get(key, 0)
    
    def get_bucket_counts(
        self,
        labels: Optional[Labels] = None,
    ) -> Dict[str, int]:
        """Get bucket counts for given labels."""
        with self._lock:
            key = self._get_key(labels or {})
            return dict(self._counts.get(key, {}))
    
    def get_sum(self, labels: Optional[Labels] = None) -> float:
        """Get sum of observations for given labels."""
        with self._lock:
            key = self._get_key(labels or {})
            return self._sums.get(key, 0.0)
    
    def get_all_values(self) -> List[Tuple[Labels, Value]]:
        """Get all label combinations and their total counts."""
        with self._lock:
            results = []
            for key, total in self._totals.items():
                if key == "":
                    results.append(({}, total))
                else:
                    labels = {}
                    for part in key.split("|"):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            labels[k] = v
                    results.append((labels, total))
            return results
    
    def to_prometheus_buckets(
        self,
        labels: Optional[Labels] = None,
        timestamp: Optional[float] = None,
    ) -> List[str]:
        """Generate Prometheus bucket lines."""
        with self._lock:
            key = self._get_key(labels or {})
            bucket_counts = self._counts.get(key, {})
            
            # Calculate cumulative counts
            cumulative = 0
            results = []
            
            suffix = "_bucket" if self.unit else "_bucket"
            
            for bucket in self.buckets:
                cumulative += bucket_counts.get(str(bucket), 0)
                bucket_labels = dict(labels or {})
                bucket_labels["le"] = str(bucket)
                label_str = self._format_labels(bucket_labels)
                results.append(f"{self.name}{suffix}{label_str} {cumulative}")
            
            # +Inf bucket
            inf_labels = dict(labels or {})
            inf_labels["le"] = "+Inf"
            inf_label_str = self._format_labels(inf_labels)
            results.append(f"{self.name}_bucket{inf_label_str} {cumulative}")
            
            # Sum and count
            if timestamp:
                results.append(f"{self.name}_sum{self._format_labels(labels or {})} {self._sums.get(key, 0)} {int(timestamp * 1000)}")
                results.append(f"{self.name}_count{self._format_labels(labels or {})} {self._totals.get(key, 0)} {int(timestamp * 1000)}")
            else:
                results.append(f"{self.name}_sum{self._format_labels(labels or {})} {self._sums.get(key, 0)}")
                results.append(f"{self.name}_count{self._format_labels(labels or {})} {self._totals.get(key, 0)}")
            
            return results
    
    def reset(self, labels: Optional[Labels] = None) -> None:
        """Reset the histogram."""
        with self._lock:
            if labels is None:
                self._counts.clear()
                self._sums.clear()
                self._totals.clear()
            else:
                key = self._get_key(labels)
                self._counts.pop(key, None)
                self._sums.pop(key, None)
                self._totals.pop(key, None)
            self._last_updated = time.time()


class MetricsRegistry:
    """
    Central metrics registry.
    
    Manages registration and retrieval of metrics, and provides
    Prometheus-compatible export functionality.
    """
    
    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()
    
    def __init__(self) -> None:
        """Initialize the metrics registry."""
        self._metrics: Dict[str, Metric] = {}
        self._metric_families: Dict[str, MetricFamily] = {}
        self._collectors: List[Callable[[], None]] = []
        self._export_lock = threading.RLock()
        self._created_at = time.time()
        self._total_metrics_registered = 0
    
    @classmethod
    def get_instance(cls) -> "MetricsRegistry":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance._metrics.clear()
                cls._instance._metric_families.clear()
                cls._instance._collectors.clear()
            cls._instance = None
    
    def register_metric(
        self,
        metric: Metric,
        family: Optional[MetricFamily] = None,
    ) -> None:
        """
        Register a metric with the registry.
        
        Args:
            metric: Metric to register
            family: Optional metric family definition
            
        Raises:
            ValueError: If metric name is invalid or already registered
        """
        with self._export_lock:
            if metric.name in self._metrics:
                existing = self._metrics[metric.name]
                if type(existing) != type(metric):
                    raise ValueError(
                        f"Metric {metric.name} already registered with different type: "
                        f"{type(existing).__name__} vs {type(metric).__name__}"
                    )
                return  # Already registered with same type
            
            self._metrics[metric.name] = metric
            self._total_metrics_registered += 1
            
            if family:
                self._metric_families[metric.name] = family
            else:
                # Create default family
                self._metric_families[metric.name] = MetricFamily(
                    name=metric.name,
                    metric_type=metric.metric_type,
                    description=metric.description,
                    unit=metric.unit,
                    labels=metric.labels,
                )
    
    def unregister_metric(self, name: str) -> bool:
        """
        Unregister a metric.
        
        Args:
            name: Metric name
            
        Returns:
            True if unregistered, False if not found
        """
        with self._export_lock:
            if name in self._metrics:
                del self._metrics[name]
                self._metric_families.pop(name, None)
                return True
            return False
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a registered metric by name."""
        with self._export_lock:
            return self._metrics.get(name)
    
    def get_all_metrics(self) -> List[Metric]:
        """Get all registered metrics."""
        with self._export_lock:
            return list(self._metrics.values())
    
    def get_metric_family(self, name: str) -> Optional[MetricFamily]:
        """Get metric family definition."""
        with self._export_lock:
            return self._metric_families.get(name)
    
    def get_all_families(self) -> List[MetricFamily]:
        """Get all metric family definitions."""
        with self._export_lock:
            return list(self._metric_families.values())
    
    def register_collector(self, collector: Callable[[], None]) -> None:
        """
        Register a collector function.
        
        Collector functions are called before exporting to allow
        updating metrics dynamically.
        
        Args:
            collector: Function that updates metrics
        """
        with self._export_lock:
            if collector not in self._collectors:
                self._collectors.append(collector)
    
    def unregister_collector(self, collector: Callable[[], None]) -> bool:
        """Unregister a collector function."""
        with self._export_lock:
            if collector in self._collectors:
                self._collectors.remove(collector)
                return True
            return False
    
    def collect(self) -> None:
        """Run all registered collectors."""
        with self._export_lock:
            for collector in self._collectors:
                try:
                    collector()
                except Exception as e:
                    logger.error(f"Collector failed: {e}")
    
    # Convenience methods for creating metrics
    
    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        unit: str = "",
    ) -> CounterMetric:
        """Register a counter metric."""
        metric = CounterMetric(
            name=name,
            description=description,
            labels=labels,
            unit=unit,
        )
        self.register_metric(metric)
        return metric
    
    def register_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        unit: str = "",
        initial_value: Value = 0,
    ) -> GaugeMetric:
        """Register a gauge metric."""
        metric = GaugeMetric(
            name=name,
            description=description,
            labels=labels,
            unit=unit,
            initial_value=initial_value,
        )
        self.register_metric(metric)
        return metric
    
    def register_histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        unit: str = "",
        buckets: Tuple[float, ...] = HistogramMetric.DEFAULT_BUCKETS,
    ) -> HistogramMetric:
        """Register a histogram metric."""
        metric = HistogramMetric(
            name=name,
            description=description,
            labels=labels,
            unit=unit,
            buckets=buckets,
        )
        self.register_metric(metric)
        return metric
    
    # Metric operations
    
    def inc_counter(
        self,
        name: str,
        amount: Value = 1,
        labels: Optional[Labels] = None,
    ) -> None:
        """Increment a counter metric."""
        metric = self.get_metric(name)
        if isinstance(metric, CounterMetric):
            metric.inc(amount, labels)
    
    def set_gauge(
        self,
        name: str,
        value: Value,
        labels: Optional[Labels] = None,
    ) -> None:
        """Set a gauge metric value."""
        metric = self.get_metric(name)
        if isinstance(metric, GaugeMetric):
            metric.set(value, labels)
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Labels] = None,
    ) -> None:
        """Record an observation in a histogram."""
        metric = self.get_metric(name)
        if isinstance(metric, HistogramMetric):
            metric.observe(value, labels)
    
    def reset_metric(
        self,
        name: str,
        labels: Optional[Labels] = None,
    ) -> None:
        """Reset a metric."""
        metric = self.get_metric(name)
        if metric:
            metric.reset(labels)
    
    # Export methods
    
    def to_prometheus(self) -> str:
        """
        Export all metrics in Prometheus text format.
        
        Returns:
            Metrics in Prometheus exposition format
        """
        self.collect()
        
        with self._export_lock:
            lines = []
            
            # Add registry info
            lines.append(f"# AGI Unified Framework Metrics Registry")
            lines.append(f"# Exported at {datetime.utcnow().isoformat()}")
            lines.append("")
            
            for family in self._metric_families.values():
                # Add HELP and TYPE
                if family.help:
                    lines.append(f"# HELP {family.name} {family.help}")
                if family.unit:
                    lines.append(f"# UNIT {family.name} {family.unit}")
                lines.append(f"# TYPE {family.name} {family.metric_type.name.lower()}")
                
                metric = self._metrics.get(family.name)
                if metric:
                    if isinstance(metric, HistogramMetric):
                        # Histograms need special handling
                        for labels, _ in metric.get_all_values():
                            lines.extend(metric.to_prometheus_buckets(labels))
                    else:
                        for labels, value in metric.get_all_values():
                            lines.append(metric.to_prometheus(value, labels))
                
                lines.append("")
            
            return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        self.collect()
        
        with self._export_lock:
            result: Dict[str, Any] = {}
            
            for name, metric in self._metrics.items():
                family = self._metric_families.get(name)
                
                metric_data: Dict[str, Any] = {
                    "name": name,
                    "type": metric.metric_type.name,
                    "values": [],
                }
                
                if family:
                    metric_data["description"] = family.description
                    metric_data["unit"] = family.unit
                    metric_data["labels"] = family.labels
                
                for labels, value in metric.get_all_values():
                    metric_data["values"].append({
                        "labels": labels,
                        "value": value,
                    })
                
                result[name] = metric_data
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._export_lock:
            return {
                "total_metrics": len(self._metrics),
                "total_families": len(self._metric_families),
                "total_collectors": len(self._collectors),
                "total_registered": self._total_metrics_registered,
                "uptime_seconds": time.time() - self._created_at,
                "metric_types": {
                    mt.name: sum(
                        1 for m in self._metrics.values()
                        if m.metric_type == mt
                    )
                    for mt in MetricType
                },
            }
    
    def clear(self) -> None:
        """Clear all metrics and families."""
        with self._export_lock:
            self._metrics.clear()
            self._metric_families.clear()
            self._collectors.clear()


class PrometheusExporter:
    """
    Prometheus metrics exporter.
    
    Provides HTTP server functionality for exposing metrics
    in Prometheus format.
    """
    
    def __init__(
        self,
        registry: Optional[MetricsRegistry] = None,
        host: str = "0.0.0.0",
        port: int = 9090,
        endpoint: str = "/metrics",
    ) -> None:
        """
        Initialize Prometheus exporter.
        
        Args:
            registry: Metrics registry to export
            host: Host to bind to
            port: Port to bind to
            endpoint: Metrics endpoint path
        """
        self.registry = registry or MetricsRegistry.get_instance()
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start the exporter server."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
            )
            self._server_thread.start()
            logger.info(f"Prometheus exporter started on {self.host}:{self.port}")
    
    def stop(self) -> None:
        """Stop the exporter server."""
        with self._lock:
            self._running = False
            if self._server_thread:
                self._server_thread.join(timeout=5)
                self._server_thread = None
            logger.info("Prometheus exporter stopped")
    
    def _run_server(self) -> None:
        """Run the HTTP server."""
        import http.server
        import socketserver
        
        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            registry_instance = self.registry
            endpoint = self.endpoint
            
            def do_GET(self) -> None:
                if self.path == self.endpoint or self.path == f"{self.endpoint}/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.end_headers()
                    
                    metrics_output = self.registry_instance.to_prometheus()
                    self.wfile.write(metrics_output.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format: str, *args: Any) -> None:
                logger.debug(format % args)
        
        try:
            with socketserver.TCPServer((self.host, self.port), MetricsHandler) as httpd:
                httpd.allow_reuse_address = True
                while self._running:
                    httpd.handle_request()
        except Exception as e:
            logger.error(f"Prometheus exporter server error: {e}")
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        return self.registry.to_prometheus()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return MetricsRegistry.get_instance()


# Pre-configured common metrics

def create_request_metrics(registry: Optional[MetricsRegistry] = None) -> Dict[str, Metric]:
    """
    Create common request-related metrics.
    
    Args:
        registry: Optional registry to use
        
    Returns:
        Dictionary of created metrics
    """
    reg = registry or MetricsRegistry.get_instance()
    
    metrics: Dict[str, Metric] = {}
    
    # Request counter
    metrics["http_requests_total"] = reg.register_counter(
        name="http_requests_total",
        description="Total number of HTTP requests",
        labels=["method", "endpoint", "status"],
        unit="requests",
    )
    
    # Request duration histogram
    metrics["http_request_duration_seconds"] = reg.register_histogram(
        name="http_request_duration_seconds",
        description="HTTP request duration in seconds",
        labels=["method", "endpoint"],
        unit="seconds",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    
    # Request size histogram
    metrics["http_request_size_bytes"] = reg.register_histogram(
        name="http_request_size_bytes",
        description="HTTP request size in bytes",
        labels=["method", "endpoint"],
        unit="bytes",
        buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
    )
    
    # Response size histogram
    metrics["http_response_size_bytes"] = reg.register_histogram(
        name="http_response_size_bytes",
        description="HTTP response size in bytes",
        labels=["method", "endpoint"],
        unit="bytes",
        buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
    )
    
    # Active requests gauge
    metrics["http_requests_in_flight"] = reg.register_gauge(
        name="http_requests_in_flight",
        description="Number of requests currently being processed",
        labels=["method", "endpoint"],
        unit="requests",
    )
    
    return metrics


def create_system_metrics(registry: Optional[MetricsRegistry] = None) -> Dict[str, Metric]:
    """
    Create common system metrics.
    
    Args:
        registry: Optional registry to use
        
    Returns:
        Dictionary of created metrics
    """
    reg = registry or MetricsRegistry.get_instance()
    
    metrics: Dict[str, Metric] = {}
    
    # Memory metrics
    metrics["process_memory_bytes"] = reg.register_gauge(
        name="process_memory_bytes",
        description="Process memory usage in bytes",
        labels=["type"],
        unit="bytes",
    )
    
    # CPU metrics
    metrics["process_cpu_seconds_total"] = reg.register_counter(
        name="process_cpu_seconds_total",
        description="Total CPU time consumed in seconds",
        unit="seconds",
    )
    
    # Thread metrics
    metrics["process_threads"] = reg.register_gauge(
        name="process_threads",
        description="Number of threads in the process",
        unit="threads",
    )
    
    # Uptime
    metrics["process_uptime_seconds"] = reg.register_gauge(
        name="process_uptime_seconds",
        description="Process uptime in seconds",
        unit="seconds",
    )
    
    return metrics


__all__ = [
    # Enums
    "MetricType",
    "AggregationType",
    # Core classes
    "MetricFamily",
    "Metric",
    "GaugeMetric",
    "CounterMetric",
    "HistogramMetric",
    # Aggregation
    "AggregationStrategy",
    "SumAggregation",
    "CountAggregation",
    "MinAggregation",
    "MaxAggregation",
    "AverageAggregation",
    "QuantileAggregation",
    "create_aggregation",
    # Registry
    "MetricsRegistry",
    "PrometheusExporter",
    "get_registry",
    # Helpers
    "create_request_metrics",
    "create_system_metrics",
    # Types
    "Labels",
    "Value",
    "Observation",
]
