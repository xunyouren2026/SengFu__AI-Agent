"""
Alert Engine Module

提供审计告警引擎，支持告警规则配置、评估和通知功能。
"""

import json
import os
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set


class AlertSeverity(Enum):
    """告警严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class NotificationChannel(Enum):
    """通知渠道枚举"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SYSLOG = "syslog"
    CONSOLE = "console"


class AlertStatus(Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertCondition:
    """
    告警条件类
    
    定义告警触发的条件规则。
    """
    
    # 支持的运算符
    OPERATORS = {
        "eq": lambda x, y: x == y,
        "ne": lambda x, y: x != y,
        "gt": lambda x, y: x > y,
        "gte": lambda x, y: x >= y,
        "lt": lambda x, y: x < y,
        "lte": lambda x, y: x <= y,
        "contains": lambda x, y: y in x if isinstance(x, (str, list)) else False,
        "in": lambda x, y: x in y if isinstance(y, (str, list, set)) else False,
    }
    
    def __init__(
        self,
        metric: str,
        operator: str,
        threshold: Any,
        time_window: Optional[int] = None
    ):
        """
        初始化告警条件
        
        Args:
            metric: 监控指标名称
            operator: 比较运算符（eq/ne/gt/gte/lt/lte/contains/in）
            threshold: 阈值
            time_window: 时间窗口（秒）
        """
        self.metric = metric
        self.operator = operator
        self.threshold = threshold
        self.time_window = time_window
    
    def evaluate(self, value: Any) -> bool:
        """
        评估条件
        
        Args:
            value: 实际值
            
        Returns:
            是否满足条件
        """
        if self.operator not in self.OPERATORS:
            return False
        return self.OPERATORS[self.operator](value, self.threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "time_window": self.time_window
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertCondition':
        """从字典创建实例"""
        return cls(
            metric=data.get("metric", ""),
            operator=data.get("operator", "eq"),
            threshold=data.get("threshold"),
            time_window=data.get("time_window")
        )


class AlertRule:
    """
    告警规则类
    
    定义完整的告警规则，包括条件和通知配置。
    """
    
    def __init__(
        self,
        name: str,
        condition: AlertCondition,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        notification_channels: Optional[List[NotificationChannel]] = None,
        cooldown: int = 300,
        description: str = "",
        enabled: bool = True
    ):
        """
        初始化告警规则
        
        Args:
            name: 规则名称
            condition: 告警条件
            severity: 严重级别
            notification_channels: 通知渠道列表
            cooldown: 冷却时间（秒）
            description: 规则描述
            enabled: 是否启用
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.notification_channels = notification_channels or [NotificationChannel.CONSOLE]
        self.cooldown = cooldown
        self.description = description
        self.enabled = enabled
        self._last_triggered: Optional[datetime] = None
    
    def can_trigger(self) -> bool:
        """检查是否可以触发（冷却时间已过）"""
        if self._last_triggered is None:
            return True
        return (datetime.utcnow() - self._last_triggered).total_seconds() >= self.cooldown
    
    def mark_triggered(self) -> None:
        """标记为已触发"""
        self._last_triggered = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "condition": self.condition.to_dict(),
            "severity": self.severity.value,
            "notification_channels": [c.value for c in self.notification_channels],
            "cooldown": self.cooldown,
            "description": self.description,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """从字典创建实例"""
        return cls(
            name=data.get("name", ""),
            condition=AlertCondition.from_dict(data.get("condition", {})),
            severity=AlertSeverity(data.get("severity", "medium")),
            notification_channels=[
                NotificationChannel(c) for c in data.get("notification_channels", ["console"])
            ],
            cooldown=data.get("cooldown", 300),
            description=data.get("description", ""),
            enabled=data.get("enabled", True)
        )


class Alert:
    """
    告警类
    
    表示一个具体的告警实例。
    """
    
    def __init__(
        self,
        rule: AlertRule,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        alert_id: Optional[str] = None
    ):
        self.id = alert_id or str(uuid.uuid4())
        self.rule = rule
        self.timestamp = datetime.utcnow()
        self.severity = rule.severity
        self.message = message
        self.context = context or {}
        self.status = AlertStatus.ACTIVE
        self.acknowledged_by: Optional[str] = None
        self.acknowledged_at: Optional[datetime] = None
        self.resolved_at: Optional[datetime] = None
    
    def acknowledge(self, user: str) -> None:
        """确认告警"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()
    
    def resolve(self) -> None:
        """解决告警"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
    
    def suppress(self) -> None:
        """抑制告警"""
        self.status = AlertStatus.SUPPRESSED
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "rule_name": self.rule.name,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "status": self.status.value,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertManager:
    """
    告警管理器
    
    管理告警的生命周期，包括抑制、聚合和升级。
    """
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._suppressed_rules: Set[str] = set()
        self._alert_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # 聚合配置
        self._aggregation_window = 300  # 5分钟
        self._aggregation_threshold = 3  # 同一规则3次触发后聚合
        self._aggregated_alerts: Dict[str, List[Alert]] = defaultdict(list)
    
    def add_alert(self, alert: Alert) -> bool:
        """
        添加告警
        
        Args:
            alert: 告警对象
            
        Returns:
            是否成功添加
        """
        with self._lock:
            # 检查规则是否被抑制
            if alert.rule.name in self._suppressed_rules:
                alert.suppress()
            
            self._alerts[alert.id] = alert
            
            # 聚合检查
            self._aggregated_alerts[alert.rule.name].append(alert)
            
            return True
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警"""
        with self._lock:
            return self._alerts.get(alert_id)
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        获取活动告警
        
        Args:
            severity: 按严重级别筛选
            
        Returns:
            活动告警列表
        """
        with self._lock:
            alerts = [
                alert for alert in self._alerts.values()
                if alert.status == AlertStatus.ACTIVE
            ]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.acknowledge(user)
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.resolve()
                return True
            return False
    
    def suppress_rule(self, rule_name: str, duration: Optional[int] = None) -> None:
        """
        抑制规则
        
        Args:
            rule_name: 规则名称
            duration: 抑制持续时间（秒），None表示永久抑制
        """
        with self._lock:
            self._suppressed_rules.add(rule_name)
            
            if duration:
                # 这里可以添加定时恢复逻辑
                pass
    
    def unsuppress_rule(self, rule_name: str) -> None:
        """取消规则抑制"""
        with self._lock:
            self._suppressed_rules.discard(rule_name)
    
    def get_aggregated_alerts(self, rule_name: str) -> List[Alert]:
        """获取聚合告警"""
        with self._lock:
            return list(self._aggregated_alerts.get(rule_name, []))
    
    def cleanup_old_alerts(self, max_age_hours: int = 72) -> int:
        """
        清理旧告警
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            清理的告警数量
        """
        cleaned = 0
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with self._lock:
            to_remove = [
                alert_id for alert_id, alert in self._alerts.items()
                if alert.timestamp < cutoff and alert.status in (
                    AlertStatus.RESOLVED, AlertStatus.SUPPRESSED
                )
            ]
            
            for alert_id in to_remove:
                alert = self._alerts.pop(alert_id)
                self._alert_history.append(alert.to_dict())
                cleaned += 1
        
        return cleaned


class AuditAlertEngine:
    """
    审计告警引擎
    
    核心告警引擎，负责规则管理、条件评估和告警发送。
    """
    
    def __init__(self, alert_manager: Optional[AlertManager] = None):
        self.alert_manager = alert_manager or AlertManager()
        self._rules: Dict[str, AlertRule] = {}
        self._metric_collectors: Dict[str, Callable[[], Any]] = {}
        self._notification_handlers: Dict[NotificationChannel, Callable[[Alert], None]] = {}
        self._lock = threading.RLock()
        
        # 注册默认通知处理器
        self._register_default_handlers()
        
        # 初始化内置规则
        self._init_builtin_rules()
    
    def _register_default_handlers(self) -> None:
        """注册默认通知处理器"""
        self._notification_handlers[NotificationChannel.CONSOLE] = self._console_notification
        self._notification_handlers[NotificationChannel.SYSLOG] = self._syslog_notification
    
    def _console_notification(self, alert: Alert) -> None:
        """控制台通知处理器"""
        print(f"[AUDIT ALERT] [{alert.severity.value.upper()}] {alert.message}")
    
    def _syslog_notification(self, alert: Alert) -> None:
        """系统日志通知处理器"""
        import syslog
        severity_map = {
            AlertSeverity.CRITICAL: syslog.LOG_CRIT,
            AlertSeverity.HIGH: syslog.LOG_ERR,
            AlertSeverity.MEDIUM: syslog.LOG_WARNING,
            AlertSeverity.LOW: syslog.LOG_NOTICE,
            AlertSeverity.INFO: syslog.LOG_INFO
        }
        syslog.syslog(
            severity_map.get(alert.severity, syslog.LOG_WARNING),
            f"Audit Alert: {alert.message}"
        )
    
    def _init_builtin_rules(self) -> None:
        """初始化内置告警规则"""
        # 多次失败登录
        self.configure_alert(
            name="multiple_failed_logins",
            condition=AlertCondition(
                metric="failed_login_count",
                operator="gte",
                threshold=5,
                time_window=300
            ),
            severity=AlertSeverity.HIGH,
            notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.SYSLOG],
            cooldown=600,
            description="检测到多次失败登录尝试，可能存在暴力破解攻击"
        )
        
        # 异常时间访问
        self.configure_alert(
            name="off_hours_access",
            condition=AlertCondition(
                metric="off_hours_access_count",
                operator="gte",
                threshold=10,
                time_window=3600
            ),
            severity=AlertSeverity.MEDIUM,
            notification_channels=[NotificationChannel.CONSOLE],
            cooldown=1800,
            description="检测到异常时间（非工作时间）的访问活动"
        )
        
        # 大量数据导出
        self.configure_alert(
            name="mass_data_export",
            condition=AlertCondition(
                metric="records_exported",
                operator="gte",
                threshold=10000,
                time_window=300
            ),
            severity=AlertSeverity.HIGH,
            notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.SYSLOG],
            cooldown=300,
            description="检测到大量数据导出操作，可能存在数据泄露风险"
        )
        
        # 权限变更
        self.configure_alert(
            name="privilege_change",
            condition=AlertCondition(
                metric="privilege_change_count",
                operator="gte",
                threshold=1,
                time_window=60
            ),
            severity=AlertSeverity.CRITICAL,
            notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.SYSLOG],
            cooldown=60,
            description="检测到权限变更操作"
        )
        
        # 配置变更
        self.configure_alert(
            name="configuration_change",
            condition=AlertCondition(
                metric="config_change_count",
                operator="gte",
                threshold=1,
                time_window=60
            ),
            severity=AlertSeverity.HIGH,
            notification_channels=[NotificationChannel.CONSOLE],
            cooldown=300,
            description="检测到系统配置变更"
        )
        
        # 敏感数据访问
        self.configure_alert(
            name="sensitive_data_access",
            condition=AlertCondition(
                metric="sensitive_access_count",
                operator="gte",
                threshold=50,
                time_window=300
            ),
            severity=AlertSeverity.MEDIUM,
            notification_channels=[NotificationChannel.CONSOLE],
            cooldown=600,
            description="检测到大量敏感数据访问"
        )
    
    def configure_alert(
        self,
        name: str,
        condition: AlertCondition,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        notification_channels: Optional[List[NotificationChannel]] = None,
        cooldown: int = 300,
        description: str = "",
        enabled: bool = True
    ) -> AlertRule:
        """
        配置告警规则
        
        Args:
            name: 规则名称
            condition: 告警条件
            severity: 严重级别
            notification_channels: 通知渠道
            cooldown: 冷却时间
            description: 规则描述
            enabled: 是否启用
            
        Returns:
            创建的告警规则
        """
        rule = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            notification_channels=notification_channels,
            cooldown=cooldown,
            description=description,
            enabled=enabled
        )
        
        with self._lock:
            self._rules[name] = rule
        
        return rule
    
    def register_metric_collector(
        self,
        metric_name: str,
        collector: Callable[[], Any]
    ) -> None:
        """
        注册指标收集器
        
        Args:
            metric_name: 指标名称
            collector: 收集器函数
        """
        with self._lock:
            self._metric_collectors[metric_name] = collector
    
    def evaluate_alerts(self) -> List[Alert]:
        """
        评估所有告警规则
        
        Returns:
            触发的告警列表
        """
        triggered_alerts = []
        
        with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue
                
                if not rule.can_trigger():
                    continue
                
                # 获取指标值
                collector = self._metric_collectors.get(rule.condition.metric)
                if collector is None:
                    continue
                
                try:
                    value = collector()
                except Exception as e:
                    print(f"Failed to collect metric {rule.condition.metric}: {e}")
                    continue
                
                # 评估条件
                if rule.condition.evaluate(value):
                    alert = self._create_alert(rule, value)
                    self.alert_manager.add_alert(alert)
                    self.send_alert(alert)
                    rule.mark_triggered()
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def _create_alert(self, rule: AlertRule, metric_value: Any) -> Alert:
        """创建告警"""
        message = f"{rule.description} (metric: {rule.condition.metric}={metric_value})"
        
        context = {
            "metric_value": metric_value,
            "threshold": rule.condition.threshold,
            "operator": rule.condition.operator,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return Alert(rule=rule, message=message, context=context)
    
    def send_alert(self, alert: Alert) -> bool:
        """
        发送告警
        
        Args:
            alert: 告警对象
            
        Returns:
            是否成功发送
        """
        success = True
        
        for channel in alert.rule.notification_channels:
            handler = self._notification_handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Failed to send alert via {channel.value}: {e}")
                    success = False
            else:
                print(f"No handler registered for channel {channel.value}")
                success = False
        
        return success
    
    def register_notification_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Alert], None]
    ) -> None:
        """
        注册通知处理器
        
        Args:
            channel: 通知渠道
            handler: 处理器函数
        """
        with self._lock:
            self._notification_handlers[channel] = handler
    
    def get_rule(self, name: str) -> Optional[AlertRule]:
        """获取告警规则"""
        with self._lock:
            return self._rules.get(name)
    
    def list_rules(self) -> List[AlertRule]:
        """列出所有规则"""
        with self._lock:
            return list(self._rules.values())
    
    def enable_rule(self, name: str) -> bool:
        """启用规则"""
        with self._lock:
            rule = self._rules.get(name)
            if rule:
                rule.enabled = True
                return True
            return False
    
    def disable_rule(self, name: str) -> bool:
        """禁用规则"""
        with self._lock:
            rule = self._rules.get(name)
            if rule:
                rule.enabled = False
                return True
            return False
    
    def delete_rule(self, name: str) -> bool:
        """删除规则"""
        with self._lock:
            if name in self._rules:
                del self._rules[name]
                return True
            return False
    
    def update_metric_value(self, metric_name: str, value: Any) -> None:
        """
        直接更新指标值（用于测试或手动触发）
        
        Args:
            metric_name: 指标名称
            value: 指标值
        """
        self.register_metric_collector(metric_name, lambda: value)
