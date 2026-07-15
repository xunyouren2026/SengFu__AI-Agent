"""
Metrics Aggregator Module

指标聚合器实现，提供多维度聚合、时间窗口和降采样功能。
"""

from __future__ import annotations

import time
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .collector import MetricRecord, MetricValue, MetricType

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """聚合类型枚举"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"
    VARIANCE = "variance"
    RATE = "rate"


@dataclass
class TimeWindow:
    """
    时间窗口
    
    Attributes:
        duration_ms: 窗口持续时间（毫秒）
        start_time: 窗口开始时间
        end_time: 窗口结束时间
    """
    duration_ms: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
        if self.end_time is None:
            self.end_time = self.start_time + (self.duration_ms / 1000)
    
    def contains(self, timestamp: float) -> bool:
        """检查时间戳是否在窗口内"""
        return self.start_time <= timestamp <= self.end_time
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """检查窗口是否已过期"""
        current = current_time or time.time()
        return current > self.end_time
    
    def slide(self, new_start_time: Optional[float] = None) -> "TimeWindow":
        """滑动窗口"""
        start = new_start_time or time.time()
        return TimeWindow(
            duration_ms=self.duration_ms,
            start_time=start,
            end_time=start + (self.duration_ms / 1000)
        )


@dataclass
class AggregatedMetric:
    """
    聚合指标
    
    Attributes:
        name: 指标名称
        value: 聚合值
        labels: 标签
        aggregation_type: 聚合类型
        window: 时间窗口
        sample_count: 样本数
        timestamp: 时间戳
    """
    name: str
    value: MetricValue
    labels: Dict[str, str] = field(default_factory=dict)
    aggregation_type: AggregationType = AggregationType.AVG
    window: Optional[TimeWindow] = None
    sample_count: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "aggregation_type": self.aggregation_type.value,
            "window": {
                "start": self.window.start_time,
                "end": self.window.end_time
            } if self.window else None,
            "sample_count": self.sample_count,
            "timestamp": self.timestamp
        }


@dataclass
class AggregationResult:
    """
    聚合结果
    
    Attributes:
        metrics: 聚合指标列表
        duration_ms: 聚合耗时
        record_count: 处理的记录数
        errors: 错误信息
    """
    metrics: List[AggregatedMetric] = field(default_factory=list)
    duration_ms: float = 0.0
    record_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "duration_ms": self.duration_ms,
            "record_count": self.record_count,
            "errors": self.errors
        }


class AggregationFunction(ABC):
    """聚合函数抽象基类"""
    
    @abstractmethod
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        """执行聚合"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取函数名称"""
        pass


class SumAggregation(AggregationFunction):
    """求和聚合"""
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        return sum(values)
    
    def get_name(self) -> str:
        return "sum"


class AvgAggregation(AggregationFunction):
    """平均值聚合"""
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def get_name(self) -> str:
        return "avg"


class MinAggregation(AggregationFunction):
    """最小值聚合"""
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        if not values:
            return 0.0
        return min(values)
    
    def get_name(self) -> str:
        return "min"


class MaxAggregation(AggregationFunction):
    """最大值聚合"""
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        if not values:
            return 0.0
        return max(values)
    
    def get_name(self) -> str:
        return "max"


class CountAggregation(AggregationFunction):
    """计数聚合"""
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        return len(values)
    
    def get_name(self) -> str:
        return "count"


class PercentileAggregation(AggregationFunction):
    """百分位数聚合"""
    
    def __init__(self, percentile: float):
        self._percentile = percentile
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * self._percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_name(self) -> str:
        return f"p{int(self._percentile)}"


class StdDevAggregation(AggregationFunction):
    """标准差聚合"""
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def get_name(self) -> str:
        return "stddev"


class RateAggregation(AggregationFunction):
    """速率聚合"""
    
    def __init__(self, window_seconds: float = 60.0):
        self._window_seconds = window_seconds
    
    def aggregate(self, values: List[MetricValue]) -> MetricValue:
        if not values or self._window_seconds <= 0:
            return 0.0
        return sum(values) / self._window_seconds
    
    def get_name(self) -> str:
        return f"rate({self._window_seconds}s)"


class AggregationRegistry:
    """聚合函数注册表"""
    
    _functions: Dict[AggregationType, Callable[[], AggregationFunction]] = {
        AggregationType.SUM: SumAggregation,
        AggregationType.AVG: AvgAggregation,
        AggregationType.MIN: MinAggregation,
        AggregationType.MAX: MaxAggregation,
        AggregationType.COUNT: CountAggregation,
        AggregationType.P50: lambda: PercentileAggregation(50),
        AggregationType.P90: lambda: PercentileAggregation(90),
        AggregationType.P95: lambda: PercentileAggregation(95),
        AggregationType.P99: lambda: PercentileAggregation(99),
        AggregationType.STDDEV: StdDevAggregation,
        AggregationType.RATE: RateAggregation,
    }
    
    @classmethod
    def get_function(cls, agg_type: AggregationType) -> AggregationFunction:
        """获取聚合函数"""
        factory = cls._functions.get(agg_type)
        if factory:
            return factory()
        raise ValueError(f"Unknown aggregation type: {agg_type}")
    
    @classmethod
    def register(
        cls,
        agg_type: AggregationType,
        factory: Callable[[], AggregationFunction]
    ) -> None:
        """注册聚合函数"""
        cls._functions[agg_type] = factory


@dataclass
class AggregationRule:
    """
    聚合规则
    
    Attributes:
        metric_pattern: 指标名称模式（支持通配符）
        aggregation_types: 聚合类型列表
        group_by: 分组标签
        window_ms: 时间窗口（毫秒）
        downsample_factor: 降采样因子
    """
    metric_pattern: str
    aggregation_types: List[AggregationType] = field(
        default_factory=lambda: [AggregationType.AVG]
    )
    group_by: List[str] = field(default_factory=list)
    window_ms: int = 60000
    downsample_factor: int = 1
    
    def matches(self, metric_name: str) -> bool:
        """检查指标名称是否匹配规则"""
        import fnmatch
        return fnmatch.fnmatch(metric_name, self.metric_pattern)


class MetricsAggregator:
    """
    指标聚合器
    
    提供多维度聚合、时间窗口和降采样功能。
    
    Example:
        >>> aggregator = MetricsAggregator()
        >>> 
        >>> # Add aggregation rules
        >>> rule = AggregationRule(
        ...     metric_pattern="request_duration_*",
        ...     aggregation_types=[AggregationType.AVG, AggregationType.P95, AggregationType.P99],
        ...     group_by=["method", "status"],
        ...     window_ms=60000
        ... )
        >>> aggregator.add_rule(rule)
        >>> 
        >>> # Add records and aggregate
        >>> aggregator.add_record(record)
        >>> result = aggregator.aggregate()
    """
    
    def __init__(self):
        """初始化聚合器"""
        self._rules: List[AggregationRule] = []
        self._records: List[MetricRecord] = []
        self._windows: Dict[str, TimeWindow] = {}
        self._aggregated_cache: Dict[str, List[AggregatedMetric]] = {}
        self._lock = threading.Lock()
        self._last_aggregation_time: Optional[float] = None
    
    def add_rule(self, rule: AggregationRule) -> None:
        """
        添加聚合规则
        
        Args:
            rule: 聚合规则
        """
        with self._lock:
            self._rules.append(rule)
    
    def remove_rule(self, rule: AggregationRule) -> bool:
        """
        移除聚合规则
        
        Args:
            rule: 聚合规则
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if rule in self._rules:
                self._rules.remove(rule)
                return True
            return False
    
    def get_rules(self) -> List[AggregationRule]:
        """获取所有规则"""
        with self._lock:
            return self._rules.copy()
    
    def add_record(self, record: MetricRecord) -> None:
        """
        添加指标记录
        
        Args:
            record: 指标记录
        """
        with self._lock:
            self._records.append(record)
    
    def add_records(self, records: List[MetricRecord]) -> None:
        """
        批量添加指标记录
        
        Args:
            records: 指标记录列表
        """
        with self._lock:
            self._records.extend(records)
    
    def aggregate(
        self,
        window: Optional[TimeWindow] = None,
        rules: Optional[List[AggregationRule]] = None
    ) -> AggregationResult:
        """
        执行聚合
        
        Args:
            window: 时间窗口，默认为None（使用所有记录）
            rules: 要应用的规则，默认为None（使用所有规则）
            
        Returns:
            聚合结果
        """
        start_time = time.time()
        result = AggregationResult()
        
        with self._lock:
            rules_to_apply = rules or self._rules
            
            if not rules_to_apply:
                result.errors.append("No aggregation rules defined")
                return result
            
            # Filter records by time window
            records_to_aggregate = self._records
            if window:
                records_to_aggregate = [
                    r for r in self._records 
                    if window.contains(r.timestamp)
                ]
            
            result.record_count = len(records_to_aggregate)
            
            # Group records by rule
            for rule in rules_to_apply:
                try:
                    aggregated = self._apply_rule(rule, records_to_aggregate)
                    result.metrics.extend(aggregated)
                except Exception as e:
                    error_msg = f"Rule {rule.metric_pattern} failed: {str(e)}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            
            # Clear processed records if window was specified
            if window:
                self._records = [
                    r for r in self._records 
                    if not window.contains(r.timestamp)
                ]
            
            self._last_aggregation_time = time.time()
        
        result.duration_ms = (time.time() - start_time) * 1000
        return result
    
    def _apply_rule(
        self,
        rule: AggregationRule,
        records: List[MetricRecord]
    ) -> List[AggregatedMetric]:
        """
        应用聚合规则
        
        Args:
            rule: 聚合规则
            records: 指标记录列表
            
        Returns:
            聚合指标列表
        """
        # Filter matching records
        matching_records = [r for r in records if rule.matches(r.name)]
        
        if not matching_records:
            return []
        
        # Group records
        groups: Dict[Tuple[Tuple[str, str], ...], List[MetricRecord]] = defaultdict(list)
        for record in matching_records:
            if rule.group_by:
                group_key = tuple(
                    (k, record.labels.get(k, "")) 
                    for k in rule.group_by
                )
            else:
                group_key = ()
            groups[group_key].append(record)
        
        # Aggregate each group
        aggregated: List[AggregatedMetric] = []
        for group_key, group_records in groups.items():
            labels = dict(group_key) if group_key else {}
            values = [r.value for r in group_records]
            
            for agg_type in rule.aggregation_types:
                try:
                    func = AggregationRegistry.get_function(agg_type)
                    value = func.aggregate(values)
                    
                    metric = AggregatedMetric(
                        name=f"{group_records[0].name}_{func.get_name()}",
                        value=value,
                        labels=labels,
                        aggregation_type=agg_type,
                        sample_count=len(values)
                    )
                    aggregated.append(metric)
                except Exception as e:
                    logger.error(f"Aggregation {agg_type} failed: {e}")
        
        return aggregated
    
    def downsample(
        self,
        records: List[MetricRecord],
        factor: int,
        aggregation_type: AggregationType = AggregationType.AVG
    ) -> List[MetricRecord]:
        """
        降采样
        
        Args:
            records: 指标记录列表
            factor: 降采样因子（每N个样本保留1个）
            aggregation_type: 聚合类型
            
        Returns:
            降采样后的记录
        """
        if factor <= 1:
            return records
        
        # Group by name and labels
        groups: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[MetricRecord]] = defaultdict(list)
        for record in records:
            key = (record.name, tuple(sorted(record.labels.items())))
            groups[key].append(record)
        
        downsampled: List[MetricRecord] = []
        func = AggregationRegistry.get_function(aggregation_type)
        
        for (name, label_tuple), group_records in groups.items():
            # Process in chunks
            for i in range(0, len(group_records), factor):
                chunk = group_records[i:i + factor]
                values = [r.value for r in chunk]
                
                downsampled.append(MetricRecord(
                    name=name,
                    value=func.aggregate(values),
                    labels=dict(label_tuple),
                    timestamp=chunk[-1].timestamp,
                    metric_type=chunk[0].metric_type
                ))
        
        return downsampled
    
    def rolling_aggregate(
        self,
        metric_name: str,
        window_ms: int,
        aggregation_type: AggregationType = AggregationType.AVG,
        step_ms: Optional[int] = None
    ) -> List[AggregatedMetric]:
        """
        滚动聚合
        
        Args:
            metric_name: 指标名称
            window_ms: 窗口大小（毫秒）
            aggregation_type: 聚合类型
            step_ms: 步长（毫秒），默认为窗口大小的一半
            
        Returns:
            聚合指标列表
        """
        step = step_ms or (window_ms // 2)
        
        with self._lock:
            # Filter records for this metric
            records = [r for r in self._records if r.name == metric_name]
            
            if not records:
                return []
            
            # Sort by timestamp
            records.sort(key=lambda r: r.timestamp)
            
            # Create rolling windows
            results: List[AggregatedMetric] = []
            start_time = records[0].timestamp
            end_time = records[-1].timestamp
            
            current_time = start_time
            while current_time <= end_time:
                window = TimeWindow(
                    duration_ms=window_ms,
                    start_time=current_time,
                    end_time=current_time + (window_ms / 1000)
                )
                
                # Get records in window
                window_records = [
                    r for r in records 
                    if window.contains(r.timestamp)
                ]
                
                if window_records:
                    values = [r.value for r in window_records]
                    func = AggregationRegistry.get_function(aggregation_type)
                    
                    results.append(AggregatedMetric(
                        name=f"{metric_name}_rolling_{func.get_name()}",
                        value=func.aggregate(values),
                        aggregation_type=aggregation_type,
                        window=window,
                        sample_count=len(values),
                        timestamp=current_time
                    ))
                
                current_time += (step / 1000)
            
            return results
    
    def get_cached_aggregation(self, cache_key: str) -> Optional[List[AggregatedMetric]]:
        """获取缓存的聚合结果"""
        with self._lock:
            return self._aggregated_cache.get(cache_key)
    
    def cache_aggregation(
        self,
        cache_key: str,
        metrics: List[AggregatedMetric]
    ) -> None:
        """缓存聚合结果"""
        with self._lock:
            self._aggregated_cache[cache_key] = metrics
    
    def clear_cache(self) -> None:
        """清除缓存"""
        with self._lock:
            self._aggregated_cache.clear()
    
    def clear_records(self) -> None:
        """清除所有记录"""
        with self._lock:
            self._records.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "record_count": len(self._records),
                "rule_count": len(self._rules),
                "cache_size": len(self._aggregated_cache),
                "last_aggregation_time": self._last_aggregation_time
            }


class DimensionReducer:
    """
    维度缩减器
    
    用于减少指标维度，降低存储和计算成本。
    """
    
    def __init__(self):
        self._dimensions_to_keep: Set[str] = set()
        self._dimensions_to_drop: Set[str] = set()
    
    def keep(self, *dimensions: str) -> "DimensionReducer":
        """指定要保留的维度"""
        self._dimensions_to_keep.update(dimensions)
        return self
    
    def drop(self, *dimensions: str) -> "DimensionReducer":
        """指定要丢弃的维度"""
        self._dimensions_to_drop.update(dimensions)
        return self
    
    def reduce(self, record: MetricRecord) -> MetricRecord:
        """
        缩减记录维度
        
        Args:
            record: 原始记录
            
        Returns:
            维度缩减后的记录
        """
        if self._dimensions_to_keep:
            new_labels = {
                k: v for k, v in record.labels.items() 
                if k in self._dimensions_to_keep
            }
        else:
            new_labels = {
                k: v for k, v in record.labels.items() 
                if k not in self._dimensions_to_drop
            }
        
        return MetricRecord(
            name=record.name,
            value=record.value,
            labels=new_labels,
            timestamp=record.timestamp,
            metric_type=record.metric_type
        )
    
    def reduce_batch(self, records: List[MetricRecord]) -> List[MetricRecord]:
        """批量缩减维度"""
        return [self.reduce(r) for r in records]
