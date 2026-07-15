"""
Alert Notifier Module

告警通知器实现，提供邮件、Webhook、PagerDuty等多渠道通知功能。
"""

from __future__ import annotations

import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .rules import Alert, AlertSeverity

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    SMS = "sms"


@dataclass
class NotificationMessage:
    """通知消息"""
    title: str
    body: str
    severity: AlertSeverity
    alert_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Notifier(ABC):
    """通知器抽象基类"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._lock = threading.Lock()
        self._stats = {
            "sent": 0,
            "failed": 0,
            "last_sent": None
        }
    
    @abstractmethod
    def send(self, message: NotificationMessage) -> bool:
        """发送通知"""
        pass
    
    def notify(self, alert: Alert) -> bool:
        """
        发送告警通知
        
        Args:
            alert: 告警对象
            
        Returns:
            是否成功
        """
        if not self.enabled:
            return False
        
        message = self._create_message(alert)
        
        try:
            success = self.send(message)
            with self._lock:
                if success:
                    self._stats["sent"] += 1
                    self._stats["last_sent"] = time.time()
                else:
                    self._stats["failed"] += 1
            return success
        except Exception as e:
            logger.error(f"Notification failed for {self.name}: {e}")
            with self._lock:
                self._stats["failed"] += 1
            return False
    
    def _create_message(self, alert: Alert) -> NotificationMessage:
        """从告警创建消息"""
        title = f"[{alert.severity.value.upper()}] {alert.name}"
        body = f"""
Alert: {alert.name}
Severity: {alert.severity.value}
Status: {alert.status.value}
Message: {alert.message}
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.starts_at))}

Labels:
{json.dumps(alert.labels, indent=2)}
"""
        
        return NotificationMessage(
            title=title,
            body=body,
            severity=alert.severity,
            alert_id=alert.id,
            metadata=alert.to_dict()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return self._stats.copy()


class EmailNotifier(Notifier):
    """邮件通知器"""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True,
        **kwargs
    ):
        super().__init__("email", **kwargs)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls
    
    def send(self, message: NotificationMessage) -> bool:
        """发送邮件"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg["Subject"] = message.title
            
            msg.attach(MIMEText(message.body, "plain"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent: {message.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class WebhookNotifier(Notifier):
    """Webhook通知器"""
    
    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 10.0,
        **kwargs
    ):
        super().__init__("webhook", **kwargs)
        self.url = url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout_seconds
    
    def send(self, message: NotificationMessage) -> bool:
        """发送Webhook请求"""
        try:
            import urllib.request
            import urllib.error
            
            payload = {
                "title": message.title,
                "body": message.body,
                "severity": message.severity.value,
                "alert_id": message.alert_id,
                "timestamp": message.timestamp,
                "metadata": message.metadata
            }
            
            data = json.dumps(payload).encode("utf-8")
            
            req = urllib.request.Request(
                self.url,
                data=data,
                headers=self.headers,
                method=self.method
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status < 400
                
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False


class PagerDutyNotifier(Notifier):
    """PagerDuty通知器"""
    
    PAGERDUTY_API_URL = "https://events.pagerduty.com/v2/enqueue"
    
    def __init__(
        self,
        routing_key: str,
        severity_mapping: Optional[Dict[AlertSeverity, str]] = None,
        **kwargs
    ):
        super().__init__("pagerduty", **kwargs)
        self.routing_key = routing_key
        self.severity_mapping = severity_mapping or {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.HIGH: "error",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "warning",
            AlertSeverity.INFO: "info"
        }
    
    def send(self, message: NotificationMessage) -> bool:
        """发送PagerDuty事件"""
        try:
            import urllib.request
            
            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "dedup_key": message.alert_id,
                "payload": {
                    "summary": message.title,
                    "severity": self.severity_mapping.get(message.severity, "warning"),
                    "source": "telemetry-alerts",
                    "custom_details": message.metadata
                }
            }
            
            data = json.dumps(payload).encode("utf-8")
            
            req = urllib.request.Request(
                self.PAGERDUTY_API_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status < 400
                
        except Exception as e:
            logger.error(f"Failed to send PagerDuty event: {e}")
            return False


class SlackNotifier(Notifier):
    """Slack通知器"""
    
    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "Telemetry Alerts",
        **kwargs
    ):
        super().__init__("slack", **kwargs)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
    
    def send(self, message: NotificationMessage) -> bool:
        """发送Slack消息"""
        try:
            import urllib.request
            
            color_map = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.HIGH: "#FF6600",
                AlertSeverity.MEDIUM: "#FFCC00",
                AlertSeverity.LOW: "#00CC00",
                AlertSeverity.INFO: "#0066CC"
            }
            
            payload = {
                "username": self.username,
                "attachments": [{
                    "color": color_map.get(message.severity, "#808080"),
                    "title": message.title,
                    "text": message.body,
                    "footer": "AGI Telemetry",
                    "ts": message.timestamp
                }]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            data = json.dumps(payload).encode("utf-8")
            
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status < 400
                
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False


class AlertNotifier:
    """
    告警通知管理器
    
    管理多个通知渠道。
    
    Example:
        >>> notifier = AlertNotifier(config)
        >>> 
        >>> # Add notifiers
        >>> notifier.add_notifier(EmailNotifier(...))
        >>> notifier.add_notifier(SlackNotifier(...))
        >>> 
        >>> # Send notification
        >>> notifier.notify(alert)
    """
    
    def __init__(self, config: Optional[Any] = None):
        self._notifiers: List[Notifier] = []
        self._lock = threading.Lock()
        self._severity_filter: Set[AlertSeverity] = set(AlertSeverity)
    
    def add_notifier(self, notifier: Notifier) -> None:
        """添加通知器"""
        with self._lock:
            self._notifiers.append(notifier)
    
    def remove_notifier(self, name: str) -> bool:
        """移除通知器"""
        with self._lock:
            for i, notifier in enumerate(self._notifiers):
                if notifier.name == name:
                    self._notifiers.pop(i)
                    return True
            return False
    
    def notify(self, alert: Alert) -> Dict[str, bool]:
        """
        发送告警通知到所有渠道
        
        Args:
            alert: 告警对象
            
        Returns:
            各渠道发送结果
        """
        if alert.severity not in self._severity_filter:
            return {}
        
        results = {}
        
        with self._lock:
            notifiers = self._notifiers.copy()
        
        for notifier in notifiers:
            success = notifier.notify(alert)
            results[notifier.name] = success
        
        return results
    
    def set_severity_filter(self, severities: List[AlertSeverity]) -> None:
        """设置严重级别过滤器"""
        self._severity_filter = set(severities)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有通知器统计"""
        with self._lock:
            return {
                notifier.name: notifier.get_stats()
                for notifier in self._notifiers
            }
