"""
注册中心监控指标
记录注册数量、心跳延迟
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels
        }


@dataclass
class HistogramBucket:
    """直方图桶"""
    upper_bound: float
    count: int = 0

    def add(self) -> None:
        """增加计数"""
        self.count += 1


@dataclass
class Histogram:
    """直方图指标"""
    name: str
    buckets: List[HistogramBucket] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0

    def __post_init__(self):
        if not self.buckets:
            # 默认桶边界
            self.buckets = [
                HistogramBucket(0.005),
                HistogramBucket(0.01),
                HistogramBucket(0.025),
                HistogramBucket(0.05),
                HistogramBucket(0.1),
                HistogramBucket(0.25),
                HistogramBucket(0.5),
                HistogramBucket(1.0),
                HistogramBucket(2.5),
                HistogramBucket(5.0),
                HistogramBucket(10.0),
                HistogramBucket(float('inf'))
            ]

    def observe(self, value: float) -> None:
        """观察一个值"""
        self.sum_value += value
        self.count += 1
        
        for bucket in self.buckets:
            if value <= bucket.upper_bound:
                bucket.add()
                break

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum_value,
            "buckets": [
                {"le": b.upper_bound, "count": b.count}
                for b in self.buckets
            ]
        }


class MetricsCollector:
    """
    指标收集器
    
    收集注册中心的各类监控指标
    """

    def __init__(self, max_history: int = 10000):
        """
        初始化指标收集器
        
        Args:
            max_history: 历史数据最大保留数量
        """
        self._max_history = max_history
        
        # 计数器
        self._counters: Dict[str, int] = {}
        
        # 仪表盘（当前值）
        self._gauges: Dict[str, float] = {}
        
        # 直方图
        self._histograms: Dict[str, Histogram] = {}
        
        # 历史数据
        self._history: Dict[str, deque] = {}
        
        # 心跳延迟记录
        self._heartbeat_latencies: deque = deque(maxlen=max_history)
        
        # 锁
        self._lock = threading.RLock()

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        增加计数器
        
        Args:
            name: 指标名称
            value: 增加值
            labels: 标签
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] = self._counters.get(key, 0) + value
            self._record_history(key, self._counters[key])

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        设置仪表盘值
        
        Args:
            name: 指标名称
            value: 值
            labels: 标签
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            self._record_history(key, value)

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None
    ) -> None:
        """
        观察直方图值
        
        Args:
            name: 指标名称
            value: 观察值
            labels: 标签
            buckets: 自定义桶边界
        """
        with self._lock:
            key = self._make_key(name, labels)
            
            if key not in self._histograms:
                if buckets:
                    hist_buckets = [HistogramBucket(b) for b in buckets]
                    hist_buckets.append(HistogramBucket(float('inf')))
                    self._histograms[key] = Histogram(name, hist_buckets)
                else:
                    self._histograms[key] = Histogram(name)
            
            self._histograms[key].observe(value)

    def record_heartbeat_latency(self, latency_ms: float, agent_id: str = "") -> None:
        """
        记录心跳延迟
        
        Args:
            latency_ms: 延迟（毫秒）
            agent_id: Agent ID
        """
        with self._lock:
            self._heartbeat_latencies.append({
                "latency_ms": latency_ms,
                "agent_id": agent_id,
                "timestamp": time.time()
            })
        
        # 同时记录到直方图
        self.observe_histogram("heartbeat_latency_seconds", latency_ms / 1000.0)

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """获取计数器值"""
        with self._lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """获取仪表盘值"""
        with self._lock:
            key = self._make_key(name, labels)
            return self._gauges.get(key, 0.0)

    def get_histogram(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Histogram]:
        """获取直方图"""
        with self._lock:
            key = self._make_key(name, labels)
            return self._histograms.get(key)

    def get_heartbeat_stats(self) -> Dict[str, Any]:
        """获取心跳延迟统计"""
        with self._lock:
            if not self._heartbeat_latencies:
                return {
                    "count": 0,
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0
                }
            
            latencies = [r["latency_ms"] for r in self._heartbeat_latencies]
            latencies.sort()
            
            count = len(latencies)
            
            def percentile(p: float) -> float:
                idx = int(count * p / 100)
                return latencies[min(idx, count - 1)]
            
            return {
                "count": count,
                "avg_ms": sum(latencies) / count,
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p50_ms": percentile(50),
                "p95_ms": percentile(95),
                "p99_ms": percentile(99)
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: v.to_dict() for k, v in self._histograms.items()
                },
                "heartbeat_stats": self.get_heartbeat_stats()
            }

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """生成指标键"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record_history(self, key: str, value: float) -> None:
        """记录历史数据"""
        if key not in self._history:
            self._history[key] = deque(maxlen=self._max_history)
        
        self._history[key].append({
            "value": value,
            "timestamp": time.time()
        })

    def clear(self) -> None:
        """清空所有指标"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._history.clear()
            self._heartbeat_latencies.clear()


class RegistryMetrics:
    """
    注册中心专用指标
    
    预定义的注册中心监控指标
    """

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self._collector = collector or MetricsCollector()

    @property
    def collector(self) -> MetricsCollector:
        """获取指标收集器"""
        return self._collector

    def record_registration(self, success: bool = True) -> None:
        """记录注册事件"""
        self._collector.increment_counter("registry_registrations_total")
        if success:
            self._collector.increment_counter("registry_registrations_success_total")
        else:
            self._collector.increment_counter("registry_registrations_failed_total")

    def record_deregistration(self) -> None:
        """记录注销事件"""
        self._collector.increment_counter("registry_deregistrations_total")

    def record_heartbeat(self, latency_ms: float) -> None:
        """记录心跳"""
        self._collector.increment_counter("registry_heartbeats_total")
        self._collector.record_heartbeat_latency(latency_ms)

    def record_discovery(self, found_count: int) -> None:
        """记录发现请求"""
        self._collector.increment_counter("registry_discoveries_total")
        self._collector.set_gauge("registry_last_discovery_count", found_count)

    def set_active_agents(self, count: int) -> None:
        """设置活跃Agent数量"""
        self._collector.set_gauge("registry_active_agents", count)

    def set_healthy_agents(self, count: int) -> None:
        """设置健康Agent数量"""
        self._collector.set_gauge("registry_healthy_agents", count)

    def set_unhealthy_agents(self, count: int) -> None:
        """设置不健康Agent数量"""
        self._collector.set_gauge("registry_unhealthy_agents", count)

    def record_lease_expired(self) -> None:
        """记录租约过期"""
        self._collector.increment_counter("registry_lease_expired_total")

    def record_watch_event(self, event_type: str) -> None:
        """记录监听事件"""
        self._collector.increment_counter(
            "registry_watch_events_total",
            labels={"type": event_type}
        )

    def record_health_check(self, success: bool, latency_ms: float) -> None:
        """记录健康检查"""
        self._collector.increment_counter("registry_health_checks_total")
        if success:
            self._collector.increment_counter("registry_health_checks_success_total")
        else:
            self._collector.increment_counter("registry_health_checks_failed_total")
        
        self._collector.observe_histogram(
            "registry_health_check_latency_seconds",
            latency_ms / 1000.0
        )

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return {
            "registrations": self._collector.get_counter("registry_registrations_total"),
            "deregistrations": self._collector.get_counter("registry_deregistrations_total"),
            "heartbeats": self._collector.get_counter("registry_heartbeats_total"),
            "discoveries": self._collector.get_counter("registry_discoveries_total"),
            "active_agents": self._collector.get_gauge("registry_active_agents"),
            "healthy_agents": self._collector.get_gauge("registry_healthy_agents"),
            "heartbeat_stats": self._collector.get_heartbeat_stats()
        }


class MetricsExporter:
    """
    指标导出器
    
    将指标导出为不同格式
    """

    def __init__(self, collector: MetricsCollector):
        self._collector = collector

    def export_prometheus(self) -> str:
        """
        导出为Prometheus格式
        
        Returns:
            Prometheus格式的指标字符串
        """
        lines = []
        metrics = self._collector.get_all_metrics()
        
        # 计数器
        for name, value in metrics["counters"].items():
            lines.append(f"# TYPE {name.split('{')[0]} counter")
            lines.append(f"{name} {value}")
        
        # 仪表盘
        for name, value in metrics["gauges"].items():
            lines.append(f"# TYPE {name.split('{')[0]} gauge")
            lines.append(f"{name} {value}")
        
        # 直方图
        for name, hist in metrics["histograms"].items():
            base_name = name.split('{')[0]
            lines.append(f"# TYPE {base_name} histogram")
            
            for bucket in hist["buckets"]:
                lines.append(
                    f'{base_name}_bucket{{le="{bucket["le"]}"}} {bucket["count"]}'
                )
            lines.append(f'{base_name}_sum {hist["sum"]}')
            lines.append(f'{base_name}_count {hist["count"]}')
        
        return "\n".join(lines)

    def export_json(self) -> str:
        """
        导出为JSON格式
        
        Returns:
            JSON格式的指标字符串
        """
        import json
        return json.dumps(self._collector.get_all_metrics(), indent=2)


class MetricsReporter:
    """
    指标报告器
    
    定期报告指标
    """

    def __init__(
        self,
        collector: MetricsCollector,
        report_interval: float = 60.0,
        reporter: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self._collector = collector
        self._report_interval = report_interval
        self._reporter = reporter
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动报告器"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._report_loop,
            daemon=True,
            name="MetricsReporter"
        )
        self._thread.start()

    def stop(self) -> None:
        """停止报告器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._report_interval + 1)

    def _report_loop(self) -> None:
        """报告循环"""
        import time
        
        while self._running:
            time.sleep(self._report_interval)
            
            if not self._running:
                break
            
            metrics = self._collector.get_all_metrics()
            
            if self._reporter:
                try:
                    self._reporter(metrics)
                except Exception:
                    pass
            else:
                # 默认输出到日志
                print(f"[Metrics] {metrics}")
