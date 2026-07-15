"""
Alert Rules Module

告警规则引擎实现，提供阈值告警、异常检测和规则组合功能。
"""

from __future__ import annotations

import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..config import AlertConfig

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    
    @property
    def numeric_value(self) -> int:
        """数值级别"""
        values = {
            AlertSeverity.CRITICAL: 5,
            AlertSeverity.HIGH: 4,
            AlertSeverity.MEDIUM: 3,
            AlertSeverity.LOW: 2,
            AlertSeverity.INFO: 1
        }
        return values[self]


class AlertStatus(Enum):
    """告警状态"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class Alert:
    """
    告警
    
    Attributes:
        id: 告警ID
        name: 告警名称
        severity: 严重程度
        status: 状态
        message: 消息
        labels: 标签
        annotations: 注释
        starts_at: 开始时间
        ends_at: 结束时间
        value: 触发值
        threshold: 阈值
    """
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    starts_at: float = field(default_factory=time.time)
    ends_at: Optional[float] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "labels": self.labels,
            "annotations": self.annotations,
            "starts_at": self.starts_at,
            "ends_at": self.ends_at,
            "value": self.value,
            "threshold": self.threshold
        }
    
    def resolve(self) -> None:
        """解决告警"""
        self.status = AlertStatus.RESOLVED
        self.ends_at = time.time()


class AlertRule(ABC):
    """告警规则抽象基类"""
    
    def __init__(
        self,
        name: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        cooldown_seconds: float = 300.0
    ):
        self.name = name
        self.severity = severity
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: Optional[float] = None
        self._active_alerts: Dict[str, Alert] = {}
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """评估规则"""
        pass
    
    def is_in_cooldown(self) -> bool:
        """检查是否在冷却期"""
        if self._last_fired is None:
            return False
        return time.time() - self._last_fired < self.cooldown_seconds
    
    def _create_alert(
        self,
        message: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        extra_labels: Optional[Dict[str, str]] = None
    ) -> Alert:
        """创建告警"""
        import uuid
        
        labels = {**self.labels, **(extra_labels or {})}
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            id=alert_id,
            name=self.name,
            severity=self.severity,
            status=AlertStatus.FIRING,
            message=message,
            labels=labels,
            annotations=self.annotations,
            value=value,
            threshold=threshold
        )
        
        self._active_alerts[alert_id] = alert
        self._last_fired = time.time()
        
        return alert
    
    def resolve_alerts(self) -> List[Alert]:
        """解决所有活动告警"""
        resolved = []
        for alert in list(self._active_alerts.values()):
            alert.resolve()
            resolved.append(alert)
            del self._active_alerts[alert.id]
        return resolved


class ThresholdRule(AlertRule):
    """
    阈值告警规则
    
    当指标值超过或低于阈值时触发告警。
    """
    
    class Operator(Enum):
        """比较操作符"""
        GT = ">"
        GTE = ">="
        LT = "<"
        LTE = "<="
        EQ = "=="
        NEQ = "!="
    
    def __init__(
        self,
        name: str,
        metric: str,
        threshold: float,
        operator: Operator = Operator.GT,
        duration_seconds: float = 0.0,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.metric = metric
        self.threshold = threshold
        self.operator = operator
        self.duration_seconds = duration_seconds
        self._first_triggered: Optional[float] = None
    
    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """评估阈值规则"""
        alerts = []
        
        value = data.get(self.metric)
        if value is None:
            return alerts
        
        # Check condition
        triggered = self._check_condition(value)
        
        if triggered:
            if self._first_triggered is None:
                self._first_triggered = time.time()
            
            # Check duration
            if self.duration_seconds > 0:
                elapsed = time.time() - self._first_triggered
                if elapsed < self.duration_seconds:
                    return alerts
            
            if not self.is_in_cooldown():
                alert = self._create_alert(
                    message=f"{self.metric} {self.operator.value} {self.threshold}",
                    value=float(value),
                    threshold=self.threshold
                )
                alerts.append(alert)
                logger.warning(f"Threshold alert triggered: {self.name}")
        else:
            self._first_triggered = None
            # Resolve existing alerts
            resolved = self.resolve_alerts()
            alerts.extend(resolved)
        
        return alerts
    
    def _check_condition(self, value: float) -> bool:
        """检查条件"""
        ops = {
            self.Operator.GT: lambda x, y: x > y,
            self.Operator.GTE: lambda x, y: x >= y,
            self.Operator.LT: lambda x, y: x < y,
            self.Operator.LTE: lambda x, y: x <= y,
            self.Operator.EQ: lambda x, y: x == y,
            self.Operator.NEQ: lambda x, y: x != y
        }
        return ops[self.operator](value, self.threshold)


class AnomalyRule(AlertRule):
    """
    异常检测规则
    
    基于统计方法检测异常值。
    """
    
    def __init__(
        self,
        name: str,
        metric: str,
        method: str = "zscore",
        threshold: float = 3.0,
        window_size: int = 100,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.metric = metric
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self._history: List[float] = []
        self._lock = threading.Lock()
    
    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """评估异常规则"""
        alerts = []
        
        value = data.get(self.metric)
        if value is None:
            return alerts
        
        with self._lock:
            self._history.append(float(value))
            if len(self._history) > self.window_size:
                self._history = self._history[-self.window_size:]
            
            if len(self._history) < 10:
                return alerts
            
            is_anomaly = self._detect_anomaly(float(value))
        
        if is_anomaly and not self.is_in_cooldown():
            alert = self._create_alert(
                message=f"Anomaly detected in {self.metric}: {value}",
                value=float(value)
            )
            alerts.append(alert)
            logger.warning(f"Anomaly alert triggered: {self.name}")
        
        return alerts
    
    def _detect_anomaly(self, value: float) -> bool:
        """检测异常"""
        if self.method == "zscore":
            return self._zscore_detection(value)
        elif self.method == "iqr":
            return self._iqr_detection(value)
        elif self.method == "mad":
            return self._mad_detection(value)
        return False
    
    def _zscore_detection(self, value: float) -> bool:
        """Z-Score检测"""
        import statistics
        mean = statistics.mean(self._history)
        std = statistics.stdev(self._history) if len(self._history) > 1 else 0
        if std == 0:
            return False
        zscore = abs((value - mean) / std)
        return zscore > self.threshold
    
    def _iqr_detection(self, value: float) -> bool:
        """IQR检测"""
        sorted_history = sorted(self._history)
        q1_idx = len(sorted_history) // 4
        q3_idx = 3 * len(sorted_history) // 4
        q1 = sorted_history[q1_idx]
        q3 = sorted_history[q3_idx]
        iqr = q3 - q1
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        return value < lower or value > upper
    
    def _mad_detection(self, value: float) -> bool:
        """MAD检测"""
        import statistics
        median = statistics.median(self._history)
        mad = statistics.median([abs(x - median) for x in self._history])
        if mad == 0:
            return False
        modified_z = 0.6745 * (value - median) / mad
        return abs(modified_z) > self.threshold


class CompositeRule(AlertRule):
    """
    组合规则
    
    组合多个规则，支持AND/OR逻辑。
    """
    
    class Logic(Enum):
        AND = "and"
        OR = "or"
    
    def __init__(
        self,
        name: str,
        rules: List[AlertRule],
        logic: Logic = Logic.AND,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.rules = rules
        self.logic = logic
    
    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """评估组合规则"""
        all_alerts = []
        rule_triggered = []
        
        for rule in self.rules:
            alerts = rule.evaluate(data)
            all_alerts.extend(alerts)
            rule_triggered.append(len(alerts) > 0)
        
        if self.logic == self.Logic.AND:
            should_fire = all(rule_triggered)
        else:  # OR
            should_fire = any(rule_triggered)
        
        if should_fire and not self.is_in_cooldown():
            alert = self._create_alert(
                message=f"Composite rule triggered: {self.name}",
                extra_labels={"composite": "true"}
            )
            all_alerts.append(alert)
        
        return all_alerts


class AlertRuleEngine:
    """
    告警规则引擎
    
    管理和评估告警规则。
    
    Example:
        >>> engine = AlertRuleEngine(config)
        >>> 
        >>> # Add rules
        >>> rule = ThresholdRule(
        ...     name="high_latency",
        ...     metric="request_latency_ms",
        ...     threshold=1000,
        ...     severity=AlertSeverity.HIGH
        ... )
        >>> engine.add_rule(rule)
        >>> 
        >>> # Evaluate
        >>> alerts = engine.evaluate({"request_latency_ms": 1500})
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        初始化规则引擎
        
        Args:
            config: 告警配置
        """
        self._config = config or AlertConfig()
        self._rules: List[AlertRule] = []
        self._lock = threading.Lock()
        self._running = False
        self._evaluation_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable[[Alert], None]] = []
    
    def start(self) -> None:
        """启动引擎"""
        self._running = True
        logger.info("AlertRuleEngine started")
    
    def stop(self) -> None:
        """停止引擎"""
        self._running = False
        logger.info("AlertRuleEngine stopped")
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        添加规则
        
        Args:
            rule: 告警规则
        """
        with self._lock:
            self._rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        移除规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否成功移除
        """
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.name == rule_name:
                    self._rules.pop(i)
                    return True
            return False
    
    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """
        评估所有规则
        
        Args:
            data: 指标数据
            
        Returns:
            触发的告警列表
        """
        all_alerts = []
        
        with self._lock:
            rules = self._rules.copy()
        
        for rule in rules:
            try:
                alerts = rule.evaluate(data)
                all_alerts.extend(alerts)
                
                for alert in alerts:
                    if alert.status == AlertStatus.FIRING:
                        self._notify_alert(alert)
            except Exception as e:
                logger.error(f"Rule evaluation failed for {rule.name}: {e}")
        
        return all_alerts
    
    def _notify_alert(self, alert: Alert) -> None:
        """通知告警"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调"""
        self._alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活动告警"""
        alerts = []
        with self._lock:
            for rule in self._rules:
                alerts.extend(rule._active_alerts.values())
        return alerts
    
    def silence_alert(self, alert_id: str, duration_seconds: float) -> bool:
        """
        静默告警
        
        Args:
            alert_id: 告警ID
            duration_seconds: 静默时长
            
        Returns:
            是否成功
        """
        with self._lock:
            for rule in self._rules:
                if alert_id in rule._active_alerts:
                    alert = rule._active_alerts[alert_id]
                    alert.status = AlertStatus.SILENCED
                    return True
        return False
