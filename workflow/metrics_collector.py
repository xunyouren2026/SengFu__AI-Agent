"""
工作流指标收集模块 (Workflow Metrics Collector)

提供工作流指标收集和分析功能：
- 节点执行时间追踪
- 成功率/失败率统计
- 瓶颈检测
- 自定义指标

类:
    WorkflowMetrics: 工作流指标
    NodeMetrics: 节点指标
    MetricsAggregator: 指标聚合器
    BottleneckDetector: 瓶颈检测器
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Tuple
import threading
import statistics
import json

# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AggregationType(Enum):
    """聚合类型"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime = field(default_factory=datetime.now)
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """指标序列"""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    points: List[MetricPoint] = field(default_factory=list)
    labels: Set[str] = field(default_factory=set)
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None) -> MetricPoint:
        """添加数据点"""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {},
        )
        self.points.append(point)
        return point
    
    def get_latest(self) -> Optional[MetricPoint]:
        """获取最新数据点"""
        if self.points:
            return self.points[-1]
        return None
    
    def get_value_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[float]:
        """获取指定时间范围的值"""
        points = self._filter_by_time(start_time, end_time)
        return [p.value for p in points]
    
    def _filter_by_time(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """按时间过滤"""
        if not start_time and not end_time:
            return self.points
        
        result = []
        for point in self.points:
            if start_time and point.timestamp < start_time:
                continue
            if end_time and point.timestamp > end_time:
                continue
            result.append(point)
        
        return result
    
    def calculate_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """计算统计信息"""
        values = self.get_value_range(start_time, end_time)
        
        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        def percentile(p: float) -> float:
            """计算百分位数"""
            idx = int(n * p)
            if idx >= n:
                idx = n - 1
            return sorted_values[idx]
        
        return {
            "count": n,
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": percentile(0.5),
            "p90": percentile(0.9),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }


@dataclass
class NodeMetrics:
    """节点指标"""
    node_id: str
    node_name: str
    workflow_id: Optional[str] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    last_execution: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    error_messages: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    @property
    def failure_rate(self) -> float:
        """失败率"""
        if self.execution_count == 0:
            return 0.0
        return self.failure_count / self.execution_count
    
    @property
    def avg_duration_ms(self) -> float:
        """平均执行时长"""
        if self.execution_count == 0:
            return 0.0
        return self.total_duration_ms / self.execution_count
    
    def record_execution(
        self,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """记录执行"""
        self.execution_count += 1
        self.last_execution = datetime.now()
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        
        if success:
            self.success_count += 1
            self.last_success = datetime.now()
        else:
            self.failure_count += 1
            self.last_failure = datetime.now()
            if error:
                self.error_messages.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "workflow_id": self.workflow_id,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float('inf') else 0,
            "max_duration_ms": self.max_duration_ms,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "recent_errors": self.error_messages[-10:],
            "tags": list(self.tags),
        }


@dataclass
class WorkflowMetrics:
    """工作流指标"""
    workflow_id: str
    workflow_name: str
    workflow_type: str = "general"
    status: str = "pending"
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_execution: Optional[datetime] = None
    node_metrics: Dict[str, NodeMetrics] = field(default_factory=dict)
    custom_metrics: Dict[str, MetricSeries] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    @property
    def avg_duration_ms(self) -> float:
        """平均执行时长"""
        if self.execution_count == 0:
            return 0.0
        return self.total_duration_ms / self.execution_count
    
    def get_node_metrics(self, node_id: str) -> Optional[NodeMetrics]:
        """获取节点指标"""
        return self.node_metrics.get(node_id)
    
    def add_node_metrics(self, node_metrics: NodeMetrics) -> None:
        """添加节点指标"""
        self.node_metrics[node_metrics.node_id] = node_metrics
    
    def get_slowest_nodes(self, limit: int = 5) -> List[Tuple[str, float]]:
        """获取最慢的节点"""
        node_durations = [
            (node_id, metrics.avg_duration_ms)
            for node_id, metrics in self.node_metrics.items()
        ]
        return sorted(node_durations, key=lambda x: x[1], reverse=True)[:limit]
    
    def get_failed_nodes(self) -> List[NodeMetrics]:
        """获取失败的节点"""
        return [
            metrics for metrics in self.node_metrics.values()
            if metrics.failure_count > 0
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_type": self.workflow_type,
            "status": self.status,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "node_count": len(self.node_metrics),
            "nodes": {
                node_id: metrics.to_dict()
                for node_id, metrics in self.node_metrics.items()
            },
            "custom_metrics": {
                name: {
                    "type": series.metric_type.value,
                    "latest": series.get_latest().value if series.get_latest() else None,
                    "count": len(series.points),
                }
                for name, series in self.custom_metrics.items()
            },
        }


class MetricsAggregator:
    """指标聚合器"""
    
    def __init__(
        self,
        window_size: timedelta = timedelta(minutes=5),
        retention_period: timedelta = timedelta(hours=1),
    ):
        self.window_size = window_size
        self.retention_period = retention_period
        self._metrics: Dict[str, MetricSeries] = {}
        self._workflow_metrics: Dict[str, WorkflowMetrics] = {}
        self._node_metrics: Dict[str, NodeMetrics] = {}
        self._lock = threading.Lock()
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
        labels: Optional[Set[str]] = None,
    ) -> MetricSeries:
        """注册指标"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    description=description,
                    unit=unit,
                    labels=labels or set(),
                )
            return self._metrics[name]
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """记录指标"""
        with self._lock:
            if name not in self._metrics:
                self.register_metric(name, MetricType.GAUGE)
            
            self._metrics[name].add_point(value, labels)
            self._cleanup_old_metrics()
    
    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """记录计数器"""
        metric = self._get_or_create_metric(name, MetricType.COUNTER)
        metric.add_point(value, labels)
    
    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """记录仪表值"""
        metric = self._get_or_create_metric(name, MetricType.GAUGE)
        metric.add_point(value, labels)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """记录直方图"""
        metric = self._get_or_create_metric(name, MetricType.HISTOGRAM)
        metric.add_point(value, labels)
    
    def _get_or_create_metric(
        self,
        name: str,
        metric_type: MetricType,
    ) -> MetricSeries:
        """获取或创建指标"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                )
            return self._metrics[name]
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """获取指标"""
        return self._metrics.get(name)
    
    def get_metrics(
        self,
        prefix: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
    ) -> List[MetricSeries]:
        """获取指标列表"""
        results = []
        for metric in self._metrics.values():
            if prefix and not metric.name.startswith(prefix):
                continue
            if metric_type and metric.metric_type != metric_type:
                continue
            results.append(metric)
        return results
    
    def aggregate(
        self,
        metric_name: str,
        aggregation: AggregationType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        percentiles: Optional[List[float]] = None,
    ) -> Any:
        """聚合指标"""
        metric = self._metrics.get(metric_name)
        if not metric:
            return None
        
        values = metric.get_value_range(start_time, end_time)
        
        if not values:
            return 0.0 if aggregation != AggregationType.COUNT else 0
        
        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVG:
            return statistics.mean(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return len(values)
        elif aggregation == AggregationType.PERCENTILE:
            if not percentiles:
                percentiles = [0.5, 0.9, 0.95, 0.99]
            sorted_values = sorted(values)
            result = {}
            for p in percentiles:
                idx = int(len(sorted_values) * p)
                if idx >= len(sorted_values):
                    idx = len(sorted_values) - 1
                result[f"p{int(p * 100)}"] = sorted_values[idx]
            return result
        
        return None
    
    def aggregate_all(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, float]]:
        """聚合所有指标"""
        result = {}
        for name, metric in self._metrics.items():
            result[name] = metric.calculate_statistics(start_time, end_time)
        return result
    
    def _cleanup_old_metrics(self) -> None:
        """清理过期指标"""
        cutoff_time = datetime.now() - self.retention_period
        
        for metric in self._metrics.values():
            while metric.points and metric.points[0].timestamp < cutoff_time:
                metric.points.pop(0)
    
    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """获取工作流指标"""
        return self._workflow_metrics.get(workflow_id)
    
    def get_node_metrics(self, node_id: str) -> Optional[NodeMetrics]:
        """获取节点指标"""
        return self._node_metrics.get(node_id)
    
    def record_workflow_execution(
        self,
        workflow_id: str,
        workflow_name: str,
        workflow_type: str,
        status: str,
        duration_ms: float,
    ) -> WorkflowMetrics:
        """记录工作流执行"""
        with self._lock:
            if workflow_id not in self._workflow_metrics:
                self._workflow_metrics[workflow_id] = WorkflowMetrics(
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    workflow_type=workflow_type,
                )
            
            metrics = self._workflow_metrics[workflow_id]
            metrics.execution_count += 1
            metrics.total_duration_ms += duration_ms
            metrics.last_execution = datetime.now()
            
            if status == "success":
                metrics.success_count += 1
                metrics.status = "success"
            else:
                metrics.failure_count += 1
                metrics.status = "failed"
            
            # 记录内置指标
            self.record_histogram("workflow.duration_ms", duration_ms)
            self.record_counter("workflow.executions", 1.0)
            
            return metrics
    
    def record_node_execution(
        self,
        node_id: str,
        node_name: str,
        workflow_id: Optional[str],
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> NodeMetrics:
        """记录节点执行"""
        with self._lock:
            if node_id not in self._node_metrics:
                self._node_metrics[node_id] = NodeMetrics(
                    node_id=node_id,
                    node_name=node_name,
                    workflow_id=workflow_id,
                )
            
            metrics = self._node_metrics[node_id]
            metrics.record_execution(success, duration_ms, error)
            
            # 记录内置指标
            self.record_histogram(f"node.{node_id}.duration_ms", duration_ms)
            self.record_counter(f"node.{node_id}.executions", 1.0)
            if success:
                self.record_counter(f"node.{node_id}.successes", 1.0)
            else:
                self.record_counter(f"node.{node_id}.failures", 1.0)
            
            return metrics
    
    def export_metrics(self) -> Dict[str, Any]:
        """导出所有指标"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                name: {
                    "type": metric.metric_type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                    "latest": metric.get_latest().value if metric.get_latest() else None,
                    "count": len(metric.points),
                    "statistics": metric.calculate_statistics(),
                }
                for name, metric in self._metrics.items()
            },
            "workflows": {
                wf_id: wf.to_dict()
                for wf_id, wf in self._workflow_metrics.items()
            },
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self._node_metrics.items()
            },
        }


class BottleneckDetector:
    """瓶颈检测器"""
    
    def __init__(
        self,
        aggregator: MetricsAggregator,
        slow_threshold_ms: float = 1000.0,
        failure_threshold_rate: float = 0.1,
        min_sample_size: int = 10,
    ):
        self.aggregator = aggregator
        self.slow_threshold_ms = slow_threshold_ms
        self.failure_threshold_rate = failure_threshold_rate
        self.min_sample_size = min_sample_size
        self._bottleneck_history: List[Dict[str, Any]] = []
    
    def detect_bottlenecks(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """检测瓶颈"""
        bottlenecks: List[Dict[str, Any]] = []
        
        # 检测慢节点
        slow_nodes = self._detect_slow_nodes(start_time, end_time)
        bottlenecks.extend(slow_nodes)
        
        # 检测高失败率节点
        failed_nodes = self._detect_high_failure_nodes(start_time, end_time)
        bottlenecks.extend(failed_nodes)
        
        # 检测资源使用瓶颈
        resource_bottlenecks = self._detect_resource_bottlenecks(start_time, end_time)
        bottlenecks.extend(resource_bottlenecks)
        
        # 检测并发瓶颈
        concurrency_bottlenecks = self._detect_concurrency_bottlenecks(start_time, end_time)
        bottlenecks.extend(concurrency_bottlenecks)
        
        # 记录历史
        if bottlenecks:
            self._bottleneck_history.append({
                "timestamp": datetime.now().isoformat(),
                "bottlenecks": bottlenecks,
            })
        
        return bottlenecks
    
    def _detect_slow_nodes(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """检测慢节点"""
        bottlenecks = []
        
        for node_id, node_metrics in self.aggregator._node_metrics.items():
            if node_metrics.execution_count < self.min_sample_size:
                continue
            
            avg_duration = node_metrics.avg_duration_ms
            if avg_duration > self.slow_threshold_ms:
                bottlenecks.append({
                    "type": "slow_node",
                    "node_id": node_id,
                    "node_name": node_metrics.node_name,
                    "avg_duration_ms": avg_duration,
                    "max_duration_ms": node_metrics.max_duration_ms,
                    "threshold_ms": self.slow_threshold_ms,
                    "severity": self._calculate_severity(
                        avg_duration / self.slow_threshold_ms
                    ),
                    "recommendation": f"Consider optimizing node {node_metrics.node_name} or adding caching",
                })
        
        return bottlenecks
    
    def _detect_high_failure_nodes(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """检测高失败率节点"""
        bottlenecks = []
        
        for node_id, node_metrics in self.aggregator._node_metrics.items():
            if node_metrics.execution_count < self.min_sample_size:
                continue
            
            failure_rate = node_metrics.failure_rate
            if failure_rate > self.failure_threshold_rate:
                bottlenecks.append({
                    "type": "high_failure_rate",
                    "node_id": node_id,
                    "node_name": node_metrics.node_name,
                    "failure_rate": failure_rate,
                    "failure_count": node_metrics.failure_count,
                    "execution_count": node_metrics.execution_count,
                    "threshold_rate": self.failure_threshold_rate,
                    "severity": self._calculate_severity(
                        failure_rate / self.failure_threshold_rate
                    ),
                    "recent_errors": node_metrics.error_messages[-5:],
                    "recommendation": f"Investigate failures in node {node_metrics.node_name}",
                })
        
        return bottlenecks
    
    def _detect_resource_bottlenecks(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """检测资源使用瓶颈"""
        bottlenecks = []
        
        # 检查CPU使用率
        cpu_metric = self.aggregator.get_metric("system.cpu.usage")
        if cpu_metric:
            values = cpu_metric.get_value_range(start_time, end_time)
            if values:
                avg_cpu = statistics.mean(values)
                if avg_cpu > 80:
                    bottlenecks.append({
                        "type": "resource_cpu",
                        "avg_usage": avg_cpu,
                        "threshold": 80,
                        "severity": self._calculate_severity(avg_cpu / 80),
                        "recommendation": "Consider scaling horizontally or optimizing computation",
                    })
        
        # 检查内存使用率
        memory_metric = self.aggregator.get_metric("system.memory.usage")
        if memory_metric:
            values = memory_metric.get_value_range(start_time, end_time)
            if values:
                avg_memory = statistics.mean(values)
                if avg_memory > 85:
                    bottlenecks.append({
                        "type": "resource_memory",
                        "avg_usage": avg_memory,
                        "threshold": 85,
                        "severity": self._calculate_severity(avg_memory / 85),
                        "recommendation": "Consider increasing memory or optimizing memory usage",
                    })
        
        return bottlenecks
    
    def _detect_concurrency_bottlenecks(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """检测并发瓶颈"""
        bottlenecks = []
        
        # 检查队列长度
        queue_metric = self.aggregator.get_metric("queue.length")
        if queue_metric:
            values = queue_metric.get_value_range(start_time, end_time)
            if values:
                max_queue = max(values)
                if max_queue > 100:
                    bottlenecks.append({
                        "type": "concurrency_queue",
                        "max_queue_length": max_queue,
                        "threshold": 100,
                        "severity": self._calculate_severity(max_queue / 100),
                        "recommendation": "Consider increasing worker count or optimizing processing speed",
                    })
        
        # 检查等待时间
        wait_metric = self.aggregator.get_metric("queue.wait_time_ms")
        if wait_metric:
            stats = wait_metric.calculate_statistics(start_time, end_time)
            if stats["avg"] > 5000:  # 5秒
                bottlenecks.append({
                    "type": "concurrency_wait",
                    "avg_wait_time_ms": stats["avg"],
                    "p99_wait_time_ms": stats["p99"],
                    "threshold_ms": 5000,
                    "severity": self._calculate_severity(stats["avg"] / 5000),
                    "recommendation": "Tasks are waiting too long in queue, consider scaling",
                })
        
        return bottlenecks
    
    def _calculate_severity(self, ratio: float) -> str:
        """计算严重程度"""
        if ratio < 1.5:
            return "low"
        elif ratio < 3.0:
            return "medium"
        elif ratio < 5.0:
            return "high"
        else:
            return "critical"
    
    def get_bottleneck_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """获取瓶颈报告"""
        bottlenecks = self.detect_bottlenecks(start_time, end_time)
        
        # 按类型分组
        by_type: Dict[str, List] = defaultdict(list)
        for bottleneck in bottlenecks:
            by_type[bottleneck["type"]].append(bottleneck)
        
        # 按严重程度排序
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_bottlenecks = sorted(
            bottlenecks,
            key=lambda b: severity_order.get(b["severity"], 4)
        )
        
        return {
            "report_time": datetime.now().isoformat(),
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "summary": {
                "total_bottlenecks": len(bottlenecks),
                "critical": sum(1 for b in bottlenecks if b["severity"] == "critical"),
                "high": sum(1 for b in bottlenecks if b["severity"] == "high"),
                "medium": sum(1 for b in bottlenecks if b["severity"] == "medium"),
                "low": sum(1 for b in bottlenecks if b["severity"] == "low"),
            },
            "by_type": {
                k: len(v) for k, v in by_type.items()
            },
            "bottlenecks": sorted_bottlenecks,
            "recommendations": self._generate_recommendations(sorted_bottlenecks),
        }
    
    def _generate_recommendations(
        self,
        bottlenecks: List[Dict[str, Any]]
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 按类型收集建议
        types_with_bottlenecks = set(b["type"] for b in bottlenecks)
        
        if "slow_node" in types_with_bottlenecks:
            recommendations.append(
                "Multiple nodes are experiencing slow execution times. "
                "Consider implementing caching, optimizing algorithms, "
                "or adding parallelization."
            )
        
        if "high_failure_rate" in types_with_bottlenecks:
            recommendations.append(
                "Some nodes have high failure rates. "
                "Review error logs and implement proper error handling, "
                "retry logic, and fallback mechanisms."
            )
        
        if "resource_cpu" in types_with_bottlenecks or "resource_memory" in types_with_bottlenecks:
            recommendations.append(
                "Resource utilization is high. "
                "Consider scaling horizontally (adding more instances) "
                "or vertically (upgrading resources)."
            )
        
        if "concurrency_queue" in types_with_bottlenecks or "concurrency_wait" in types_with_bottlenecks:
            recommendations.append(
                "Concurrency bottlenecks detected. "
                "Increase worker pool size or optimize task processing efficiency."
            )
        
        return recommendations
    
    def get_bottleneck_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取瓶颈历史"""
        return self._bottleneck_history[-limit:]


class WorkflowMetricsCollector:
    """工作流指标收集器"""
    
    _instance: Optional[WorkflowMetricsCollector] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args: Any, **kwargs: Any) -> WorkflowMetricsCollector:
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        window_size: timedelta = timedelta(minutes=5),
        retention_period: timedelta = timedelta(hours=24),
    ):
        if self._initialized:
            return
        
        self.window_size = window_size
        self.retention_period = retention_period
        
        self.aggregator = MetricsAggregator(
            window_size=window_size,
            retention_period=retention_period,
        )
        self.bottleneck_detector = BottleneckDetector(self.aggregator)
        
        # 预注册内置指标
        self._register_builtin_metrics()
        
        self._initialized = True
        self._context: Dict[str, Any] = {}
    
    def _register_builtin_metrics(self) -> None:
        """注册内置指标"""
        # 工作流指标
        self.aggregator.register_metric(
            "workflow.executions",
            MetricType.COUNTER,
            "Total workflow executions",
        )
        self.aggregator.register_metric(
            "workflow.duration_ms",
            MetricType.HISTOGRAM,
            "Workflow execution duration in milliseconds",
            "ms",
        )
        self.aggregator.register_metric(
            "workflow.success_rate",
            MetricType.GAUGE,
            "Workflow success rate",
        )
        
        # 节点指标
        self.aggregator.register_metric(
            "node.executions",
            MetricType.COUNTER,
            "Total node executions",
        )
        self.aggregator.register_metric(
            "node.duration_ms",
            MetricType.HISTOGRAM,
            "Node execution duration in milliseconds",
            "ms",
        )
        
        # 系统指标
        self.aggregator.register_metric(
            "system.cpu.usage",
            MetricType.GAUGE,
            "CPU usage percentage",
            "%",
        )
        self.aggregator.register_metric(
            "system.memory.usage",
            MetricType.GAUGE,
            "Memory usage percentage",
            "%",
        )
        self.aggregator.register_metric(
            "queue.length",
            MetricType.GAUGE,
            "Task queue length",
        )
        self.aggregator.register_metric(
            "queue.wait_time_ms",
            MetricType.HISTOGRAM,
            "Task wait time in queue",
            "ms",
        )
    
    def set_context(
        self,
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """设置上下文"""
        if workflow_id is not None:
            self._context["workflow_id"] = workflow_id
        if node_id is not None:
            self._context["node_id"] = node_id
    
    def clear_context(self) -> None:
        """清除上下文"""
        self._context = {}
    
    def record_workflow_start(self, workflow_id: str, workflow_name: str) -> None:
        """记录工作流开始"""
        self.set_context(workflow_id=workflow_id)
        self.aggregator.record_counter("workflow.executions")
    
    def record_workflow_end(
        self,
        workflow_id: str,
        workflow_name: str,
        status: str,
        duration_ms: float,
    ) -> None:
        """记录工作流结束"""
        self.aggregator.record_workflow_execution(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            workflow_type=self._context.get("workflow_type", "general"),
            status=status,
            duration_ms=duration_ms,
        )
        self.clear_context()
    
    def record_node_start(self, node_id: str, node_name: str) -> None:
        """记录节点开始"""
        self.set_context(node_id=node_id)
        self.aggregator.record_counter(f"node.{node_id}.executions")
    
    def record_node_end(
        self,
        node_id: str,
        node_name: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """记录节点结束"""
        self.aggregator.record_node_execution(
            node_id=node_id,
            node_name=node_name,
            workflow_id=self._context.get("workflow_id"),
            success=success,
            duration_ms=duration_ms,
            error=error,
        )
    
    def record_custom_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
    ) -> None:
        """记录自定义指标"""
        self.aggregator.record_metric(name, value)
    
    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """获取工作流指标"""
        return self.aggregator.get_workflow_metrics(workflow_id)
    
    def get_node_metrics(self, node_id: str) -> Optional[NodeMetrics]:
        """获取节点指标"""
        return self.aggregator.get_node_metrics(node_id)
    
    def get_bottleneck_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """获取瓶颈报告"""
        return self.bottleneck_detector.get_bottleneck_report(start_time, end_time)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return self.aggregator.export_metrics()
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """获取统计信息"""
        aggregated = self.aggregator.aggregate_all(start_time, end_time)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "metrics": aggregated,
            "workflow_count": len(self.aggregator._workflow_metrics),
            "node_count": len(self.aggregator._node_metrics),
        }


# 全局收集器实例
_default_collector: Optional[WorkflowMetricsCollector] = None


def get_metrics_collector(**kwargs: Any) -> WorkflowMetricsCollector:
    """获取全局指标收集器"""
    global _default_collector
    
    if _default_collector is None:
        _default_collector = WorkflowMetricsCollector(**kwargs)
    
    return _default_collector


__all__ = [
    # 枚举类型
    "MetricType",
    "AggregationType",
    # 数据类
    "MetricPoint",
    "MetricSeries",
    "NodeMetrics",
    "WorkflowMetrics",
    # 核心类
    "MetricsAggregator",
    "BottleneckDetector",
    "WorkflowMetricsCollector",
    # 辅助函数
    "get_metrics_collector",
]
