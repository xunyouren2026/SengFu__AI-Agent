"""
SLA监控系统 - 检查响应时间、成功率是否达标
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict, deque
from statistics import mean, stdev


class SLAMetricType(Enum):
    RESPONSE_TIME = auto()
    THROUGHPUT = auto()
    AVAILABILITY = auto()
    ERROR_RATE = auto()
    SUCCESS_RATE = auto()
    LATENCY_P50 = auto()
    LATENCY_P95 = auto()
    LATENCY_P99 = auto()


class SLAStatus(Enum):
    COMPLIANT = auto()
    WARNING = auto()
    VIOLATED = auto()
    UNKNOWN = auto()


class AlertSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class SLAMetric:
    metric_type: SLAMetricType
    target_value: float
    warning_threshold: float
    violation_threshold: float
    unit: str
    measurement_window_seconds: int = 300


@dataclass
class MetricReading:
    timestamp: float
    value: float
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAReport:
    agent_id: str
    listing_id: str
    period_start: float
    period_end: float
    metrics: Dict[SLAMetricType, Dict[str, Any]]
    overall_status: SLAStatus
    violations: List[Dict[str, Any]]
    uptime_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id, "listing_id": self.listing_id,
            "period_start": self.period_start, "period_end": self.period_end,
            "metrics": {k.name: v for k, v in self.metrics.items()},
            "overall_status": self.overall_status.name,
            "violations_count": len(self.violations), "uptime_percentage": self.uptime_percentage
        }


@dataclass
class SLAAlert:
    alert_id: str
    agent_id: str
    listing_id: str
    metric_type: SLAMetricType
    severity: AlertSeverity
    message: str
    timestamp: float
    current_value: float
    threshold_value: float
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[float] = None


class SLAMonitor:
    def __init__(self):
        self._sla_definitions: Dict[str, Dict[SLAMetricType, SLAMetric]] = defaultdict(dict)
        self._metric_history: Dict[str, Dict[SLAMetricType, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10000)))
        self._alerts: Dict[str, SLAAlert] = {}
        self._active_alerts: Set[str] = set()
        self._alert_callbacks: List[Callable[[SLAAlert], None]] = []
        self._violation_callbacks: List[Callable[[str, SLAMetricType, float], None]] = []
        self._default_window = 300
        self._alert_cooldown_seconds = 300
        self._last_alert_time: Dict[str, float] = {}
    
    def define_sla(self, listing_id: str, metric_type: SLAMetricType, target_value: float,
                   warning_threshold: float, violation_threshold: float, unit: str, window_seconds: int = 300) -> SLAMetric:
        metric = SLAMetric(metric_type=metric_type, target_value=target_value, warning_threshold=warning_threshold,
                          violation_threshold=violation_threshold, unit=unit, measurement_window_seconds=window_seconds)
        self._sla_definitions[listing_id][metric_type] = metric
        return metric
    
    def record_metric(self, listing_id: str, metric_type: SLAMetricType, value: float,
                      task_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        reading = MetricReading(timestamp=time.time(), value=value, task_id=task_id, metadata=metadata or {})
        self._metric_history[listing_id][metric_type].append(reading)
        self._check_thresholds(listing_id, metric_type, value)
    
    def record_task_completion(self, listing_id: str, task_id: str, success: bool,
                               response_time_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.record_metric(listing_id, SLAMetricType.RESPONSE_TIME, response_time_ms, task_id, metadata)
        self.record_metric(listing_id, SLAMetricType.SUCCESS_RATE, 1.0 if success else 0.0, task_id, metadata)
        if not success:
            self.record_metric(listing_id, SLAMetricType.ERROR_RATE, 1.0, task_id, metadata)
    
    def _check_thresholds(self, listing_id: str, metric_type: SLAMetricType, value: float) -> None:
        sla = self._sla_definitions[listing_id].get(metric_type)
        if not sla:
            return
        alert_key = f"{listing_id}:{metric_type.name}"
        now = time.time()
        last_alert = self._last_alert_time.get(alert_key, 0)
        if now - last_alert < self._alert_cooldown_seconds:
            return
        if value > sla.violation_threshold:
            self._create_alert(listing_id, metric_type, AlertSeverity.CRITICAL, value, sla.violation_threshold)
            self._last_alert_time[alert_key] = now
            for callback in self._violation_callbacks:
                try:
                    callback(listing_id, metric_type, value)
                except Exception:
                    pass
        elif value > sla.warning_threshold:
            self._create_alert(listing_id, metric_type, AlertSeverity.WARNING, value, sla.warning_threshold)
            self._last_alert_time[alert_key] = now
    
    def _create_alert(self, listing_id: str, metric_type: SLAMetricType, severity: AlertSeverity,
                      current_value: float, threshold_value: float) -> SLAAlert:
        alert = SLAAlert(
            alert_id=str(time.time()), agent_id=listing_id, listing_id=listing_id, metric_type=metric_type,
            severity=severity, message=f"{metric_type.name} {'violated' if severity == AlertSeverity.CRITICAL else 'warning'}: {current_value:.2f} > {threshold_value:.2f}",
            timestamp=time.time(), current_value=current_value, threshold_value=threshold_value
        )
        self._alerts[alert.alert_id] = alert
        self._active_alerts.add(alert.alert_id)
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        alert = self._alerts.get(alert_id)
        if alert:
            alert.acknowledged = True
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        alert = self._alerts.get(alert_id)
        if not alert:
            return False
        alert.resolved = True
        alert.resolved_at = time.time()
        self._active_alerts.discard(alert_id)
        return True
    
    def get_current_metrics(self, listing_id: str, window_seconds: int = 300) -> Dict[SLAMetricType, Dict[str, float]]:
        now = time.time()
        cutoff = now - window_seconds
        result = {}
        for metric_type, readings in self._metric_history[listing_id].items():
            recent = [r.value for r in readings if r.timestamp > cutoff]
            if recent:
                result[metric_type] = {
                    "current": recent[-1] if recent else 0,
                    "mean": mean(recent), "min": min(recent), "max": max(recent),
                    "count": len(recent)
                }
                if len(recent) > 1:
                    result[metric_type]["stdev"] = stdev(recent)
        return result
    
    def calculate_percentile(self, listing_id: str, metric_type: SLAMetricType, percentile: float, window_seconds: int = 300) -> float:
        cutoff = time.time() - window_seconds
        readings = self._metric_history[listing_id].get(metric_type, deque())
        values = sorted([r.value for r in readings if r.timestamp > cutoff])
        if not values:
            return 0.0
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]
    
    def generate_report(self, listing_id: str, period_start: float, period_end: float) -> SLAReport:
        violations = []
        metrics_summary = {}
        overall_status = SLAStatus.COMPLIANT
        
        for metric_type, sla in self._sla_definitions[listing_id].items():
            readings = [r for r in self._metric_history[listing_id].get(metric_type, deque())
                       if period_start <= r.timestamp <= period_end]
            if not readings:
                continue
            values = [r.value for r in readings]
            avg_value = mean(values)
            violation_count = sum(1 for v in values if v > sla.violation_threshold)
            warning_count = sum(1 for v in values if sla.warning_threshold < v <= sla.violation_threshold)
            
            metrics_summary[metric_type] = {
                "average": avg_value, "min": min(values), "max": max(values),
                "count": len(values), "violations": violation_count, "warnings": warning_count,
                "target": sla.target_value, "unit": sla.unit
            }
            
            if violation_count > 0:
                overall_status = SLAStatus.VIOLATED
                violations.append({"metric": metric_type.name, "count": violation_count, "threshold": sla.violation_threshold})
            elif warning_count > 0 and overall_status != SLAStatus.VIOLATED:
                overall_status = SLAStatus.WARNING
        
        success_readings = [r for r in self._metric_history[listing_id].get(SLAMetricType.SUCCESS_RATE, deque())
                           if period_start <= r.timestamp <= period_end]
        uptime = mean([r.value for r in success_readings]) * 100 if success_readings else 100.0
        
        return SLAReport(
            agent_id=listing_id, listing_id=listing_id, period_start=period_start, period_end=period_end,
            metrics=metrics_summary, overall_status=overall_status, violations=violations, uptime_percentage=uptime
        )
    
    def check_sla_compliance(self, listing_id: str, window_seconds: int = 300) -> Dict[SLAMetricType, SLAStatus]:
        result = {}
        now = time.time()
        cutoff = now - window_seconds
        
        for metric_type, sla in self._sla_definitions[listing_id].items():
            readings = [r.value for r in self._metric_history[listing_id].get(metric_type, deque()) if r.timestamp > cutoff]
            if not readings:
                result[metric_type] = SLAStatus.UNKNOWN
                continue
            avg_value = mean(readings)
            if avg_value > sla.violation_threshold:
                result[metric_type] = SLAStatus.VIOLATED
            elif avg_value > sla.warning_threshold:
                result[metric_type] = SLAStatus.WARNING
            else:
                result[metric_type] = SLAStatus.COMPLIANT
        return result
    
    def get_active_alerts(self, listing_id: Optional[str] = None) -> List[SLAAlert]:
        alerts = [self._alerts[aid] for aid in self._active_alerts if aid in self._alerts]
        if listing_id:
            alerts = [a for a in alerts if a.listing_id == listing_id]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, listing_id: str, limit: int = 100) -> List[SLAAlert]:
        alerts = [a for a in self._alerts.values() if a.listing_id == listing_id]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
    
    def add_alert_callback(self, callback: Callable[[SLAAlert], None]) -> None:
        self._alert_callbacks.append(callback)
    
    def add_violation_callback(self, callback: Callable[[str, SLAMetricType, float], None]) -> None:
        self._violation_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        total_alerts = len(self._alerts)
        active_alerts = len(self._active_alerts)
        total_listings = len(self._sla_definitions)
        
        severity_counts = defaultdict(int)
        for alert in self._alerts.values():
            severity_counts[alert.severity.name] += 1
        
        return {
            "total_alerts": total_alerts, "active_alerts": active_alerts,
            "total_monitored_listings": total_listings,
            "severity_distribution": dict(severity_counts),
            "total_metric_readings": sum(sum(len(r) for r in m.values()) for m in self._metric_history.values())
        }
