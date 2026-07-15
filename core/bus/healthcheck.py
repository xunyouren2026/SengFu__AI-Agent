"""Health Check module"""
import json, socket, threading, time, urllib.request, urllib.error, urllib.parse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import random, subprocess

class HealthStatusEnum(Enum):
    HEALTHY = "healthy"; DEGRADED = "degraded"; UNHEALTHY = "unhealthy"; UNKNOWN = "unknown"; MAINTENANCE = "maintenance"

class AlertSeverity(Enum):
    INFO = "info"; WARNING = "warning"; ERROR = "error"; CRITICAL = "critical"

@dataclass
class HealthCheckResult:
    backend_name: str; status: HealthStatusEnum; latency_ms: float = 0.0
    timestamp: Optional[datetime] = None; error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0; consecutive_successes: int = 0
    def __post_init__(self):
        if self.timestamp is None: self.timestamp = datetime.now(timezone.utc)
    def to_dict(self) -> dict:
        return {"backend_name": self.backend_name, "status": self.status.value, "latency_ms": self.latency_ms, "timestamp": self.timestamp.isoformat() if self.timestamp else None, "error_message": self.error_message, "details": self.details, "consecutive_failures": self.consecutive_failures, "consecutive_successes": self.consecutive_successes}

@dataclass
class HealthAlert:
    alert_id: str; severity: AlertSeverity; title: str; message: str; backend_name: str
    timestamp: Optional[datetime] = None; resolved: bool = False; resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if self.timestamp is None: self.timestamp = datetime.now(timezone.utc)
    def resolve(self) -> None:
        self.resolved = True; self.resolved_at = datetime.now(timezone.utc)
    def to_dict(self) -> dict:
        return {"alert_id": self.alert_id, "severity": self.severity.value, "title": self.title, "message": self.message, "backend_name": self.backend_name, "timestamp": self.timestamp.isoformat() if self.timestamp else None, "resolved": self.resolved, "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None, "metadata": self.metadata}

@dataclass
class BackendHealth:
    name: str; endpoint: str; status: HealthStatusEnum = HealthStatusEnum.UNKNOWN
    last_check: Optional[datetime] = None; last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None; consecutive_failures: int = 0; consecutive_successes: int = 0
    avg_latency_ms: float = 0.0; min_latency_ms: float = 0.0; max_latency_ms: float = 0.0
    total_checks: int = 0; total_failures: int = 0; uptime_percentage: float = 0.0
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))
    def update_from_result(self, result: "HealthCheckResult") -> None:
        self.last_check = result.timestamp; self.status = result.status
        if result.status == HealthStatusEnum.HEALTHY:
            self.last_success = result.timestamp; self.consecutive_successes += 1; self.consecutive_failures = 0
        else:
            self.last_failure = result.timestamp; self.consecutive_failures += 1; self.consecutive_successes = 0
        self.total_checks += 1
        if result.status != HealthStatusEnum.HEALTHY: self.total_failures += 1
        if result.latency_ms > 0:
            if self.total_checks == 1: self.avg_latency_ms = self.min_latency_ms = self.max_latency_ms = result.latency_ms
            else:
                self.avg_latency_ms = (self.avg_latency_ms * (self.total_checks - 1) + result.latency_ms) / self.total_checks
                self.min_latency_ms = min(self.min_latency_ms, result.latency_ms)
                self.max_latency_ms = max(self.max_latency_ms, result.latency_ms)
        if self.total_checks > 0: self.uptime_percentage = (self.total_checks - self.total_failures) / self.total_checks * 100
        self.recent_results.append(result)
    def to_dict(self) -> dict:
        return {"name": self.name, "endpoint": self.endpoint, "status": self.status.value, "last_check": self.last_check.isoformat() if self.last_check else None, "last_success": self.last_success.isoformat() if self.last_success else None, "last_failure": self.last_failure.isoformat() if self.last_failure else None, "consecutive_failures": self.consecutive_failures, "consecutive_successes": self.consecutive_successes, "avg_latency_ms": self.avg_latency_ms, "min_latency_ms": self.min_latency_ms, "max_latency_ms": self.max_latency_ms, "total_checks": self.total_checks, "total_failures": self.total_failures, "uptime_percentage": self.uptime_percentage}

class ReconnectionManager:
    def __init__(self, initial_delay: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0, jitter: float = 0.1):
        self._initial_delay = initial_delay; self._max_delay = max_delay; self._multiplier = multiplier; self._jitter = jitter
        self._retry_count: Dict[str, int] = defaultdict(int); self._next_retry_time: Dict[str, float] = {}
        self._last_success_time: Dict[str, float] = {}; self._lock = threading.Lock()
    def record_success(self, backend_name: str) -> None:
        with self._lock:
            self._retry_count[backend_name] = 0; self._last_success_time[backend_name] = time.time()
            if backend_name in self._next_retry_time: del self._next_retry_time[backend_name]
    def record_failure(self, backend_name: str) -> None:
        with self._lock:
            self._retry_count[backend_name] += 1
            self._next_retry_time[backend_name] = time.time() + self._calculate_delay_internal(backend_name)
    def should_retry(self, backend_name: str) -> bool:
        with self._lock:
            if backend_name not in self._next_retry_time: return True
            return time.time() >= self._next_retry_time[backend_name]
    def calculate_delay(self, backend_name: str) -> float:
        with self._lock: return self._calculate_delay_internal(backend_name)
    def _calculate_delay_internal(self, backend_name: str) -> float:
        rc = self._retry_count[backend_name]
        bd = min(self._initial_delay * (self._multiplier ** rc), self._max_delay)
        if self._jitter > 0: bd += random.uniform(-bd * self._jitter, bd * self._jitter)
        return max(0.1, bd)
    def get_retry_count(self, backend_name: str) -> int:
        with self._lock: return self._retry_count.get(backend_name, 0)
    def reset(self, backend_name: str) -> None:
        with self._lock:
            self._retry_count[backend_name] = 0
            if backend_name in self._next_retry_time: del self._next_retry_time[backend_name]
    def get_time_until_retry(self, backend_name: str) -> float:
        with self._lock:
            if backend_name not in self._next_retry_time: return 0.0
            return max(0.0, self._next_retry_time[backend_name] - time.time())
    def get_all_pending_retries(self) -> List[Tuple[str, float]]:
        with self._lock:
            ct = time.time()
            return [(n, max(0.0, t - ct)) for n, t in self._next_retry_time.items()]

class HealthChecker:
    def __init__(self, timeout: float = 5.0): self._timeout = timeout
    def check(self, backend_name: str, endpoint: str, check_type: str = "http") -> HealthCheckResult:
        if check_type == "http": return self._check_http(backend_name, endpoint)
        elif check_type == "tcp": return self._check_tcp(backend_name, endpoint)
        elif check_type == "ping": return self._check_ping(backend_name, endpoint)
        return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNKNOWN, error_message="unknown check type")
    def _check_http(self, backend_name: str, endpoint: str) -> HealthCheckResult:
        st = time.time()
        try:
            if not endpoint.startswith(("http://", "https://")): endpoint = "http://" + endpoint
            req = urllib.request.Request(endpoint)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                lat = (time.time() - st) * 1000
                if 200 <= resp.status < 300: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.HEALTHY, latency_ms=lat, details={"status_code": resp.status})
                if 300 <= resp.status < 400: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.DEGRADED, latency_ms=lat, error_message="redirect: " + str(resp.status))
                return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=lat, error_message="error: " + str(resp.status))
        except urllib.error.HTTPError as e: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="HTTP: " + str(e.code))
        except urllib.error.URLError as e: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="URL: " + str(e.reason))
        except socket.timeout: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="timeout")
        except Exception as e: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="fail: " + str(e))
    def _check_tcp(self, backend_name: str, endpoint: str) -> HealthCheckResult:
        st = time.time()
        try:
            if "://" in endpoint: p = urllib.parse.urlparse(endpoint); h = p.hostname or "localhost"; pt = p.port or 80
            else: parts = endpoint.split(":"); h = parts[0]; pt = int(parts[1]) if len(parts) > 1 else 80
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(self._timeout)
            r = s.connect_ex((h, pt)); s.close(); lat = (time.time() - st) * 1000
            if r == 0: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.HEALTHY, latency_ms=lat, details={"host": h, "port": pt})
            return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=lat, error_message="connect fail: " + str(r))
        except socket.gaierror as e: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="DNS: " + str(e))
        except socket.timeout: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="timeout")
        except Exception as e: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="fail: " + str(e))
    def _check_ping(self, backend_name: str, endpoint: str) -> HealthCheckResult:
        st = time.time()
        try:
            if "://" in endpoint: h = urllib.parse.urlparse(endpoint).hostname or "localhost"
            else: h = endpoint.split(":")[0]
            r = subprocess.run(["ping", "-c", "1", "-W", str(int(self._timeout)), h], capture_output=True, timeout=self._timeout+1)
            lat = (time.time() - st) * 1000
            if r.returncode == 0: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.HEALTHY, latency_ms=lat, details={"host": h})
            return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=lat, error_message="ping fail")
        except subprocess.TimeoutExpired: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="timeout")
        except Exception as e: return HealthCheckResult(backend_name=backend_name, status=HealthStatusEnum.UNHEALTHY, latency_ms=(time.time()-st)*1000, error_message="fail: " + str(e))

class BusHealthChecker:
    def __init__(self, check_interval: float = 30.0, health_threshold: int = 3, unhealthy_threshold: int = 5, timeout: float = 5.0):
        self._check_interval = check_interval; self._health_threshold = health_threshold
        self._unhealthy_threshold = unhealthy_threshold; self._timeout = timeout
        self._checker = HealthChecker(timeout); self._reconnection_manager = ReconnectionManager()
        self._backends: Dict[str, dict] = {}; self._alerts: Dict[str, HealthAlert] = {}
        self._alert_history: deque = deque(maxlen=1000); self._running = False; self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._on_health_change: Optional[Callable[[str, HealthStatusEnum, HealthStatusEnum], None]] = None
        self._on_alert

    def _generate_alert(self, result: HealthCheckResult) -> None:
        with self._lock:
            aid = "alert_" + result.backend_name
            if aid in self._alerts and not self._alerts[aid].resolved: return
            severity = AlertSeverity.CRITICAL if result.consecutive_failures >= 5 else AlertSeverity.ERROR
            alert = HealthAlert(alert_id=aid, severity=severity, title="Backend unhealthy: " + result.backend_name,
                              message=result.error_message or ("consecutive " + str(result.consecutive_failures) + " failures"),
                              backend_name=result.backend_name)
            self._alerts[aid] = alert; self._alert_history.append(alert)
            if self._on_alert: self._on_alert(alert)
    def _process_reconnections(self) -> None:
        for name in list(self._backends.keys()):
            if self._reconnection_manager.should_retry(name):
                result = self._checker.check(name, self._backends[name]["endpoint"], self._backends[name]["check_type"])
                if result.status == HealthStatusEnum.HEALTHY:
                    self.trigger_reconnect(name); self._backends[name]["status"] = HealthStatusEnum.HEALTHY
                    if name in self._alerts: self._alerts[name].resolve()
    def set_health_change_callback(self, callback: Callable[[str, HealthStatusEnum, HealthStatusEnum], None]) -> None:
        self._on_health_change = callback
    def set_alert_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        self._on_alert = callback
    def set_reconnect_callback(self, callback: Callable[[str], None]) -> None:
        self._on_reconnect = callback
    def set_report_callback(self, callback: Callable[[dict], None]) -> None:
        self._on_report = callback
    def force_check(self, backend_name: str) -> Optional[HealthCheckResult]:
        with self._lock:
            if backend_name not in self._backends: return None
            cfg = self._backends[backend_name]
            result = self._checker.check(backend_name, cfg["endpoint"], cfg["check_type"])
            cfg.update({"status": result.status, "last_result": result.to_dict(), "last_check": datetime.now(timezone.utc).isoformat()})
            if result.status == HealthStatusEnum.HEALTHY: self._reconnection_manager.record_success(backend_name)
            else: self._reconnection_manager.record_failure(backend_name)
            if self._should_generate_alert(result): self._generate_alert(result)
            return result
    def get_backend_stats(self, backend_name: str) -> dict:
        with self._lock:
            if backend_name not in self._backends: return {}
            cfg = self._backends[backend_name]
            return {"endpoint": cfg.get("endpoint"), "status": cfg.get("status", HealthStatusEnum.UNKNOWN).value,
                    "last_check": cfg.get("last_check"), "consecutive_failures": cfg.get("consecutive_failures", 0)}
    def clear_alerts(self, resolved_only: bool = True) -> int:
        with self._lock:
            if resolved_only: [self._alerts.pop(k) for k in list(self._alerts.keys()) if self._alerts[k].resolved]
            else: self._alerts.clear()
            return len(self._alerts)
    def export_health_report(self, filepath: str) -> bool:
        try:
            with open(filepath, 'w') as f: json.dump(self.generate_report(), f, indent=2)
            return True
        except Exception: return False


# Additional helper classes and functions

class HealthStatusReporter:
    """健康状态报告器"""
    
    def __init__(self, checker: "BusHealthChecker"):
        self._checker = checker
    
    def generate_json_report(self) -> str:
        """生成JSON格式的健康报告"""
        import json
        return json.dumps(self._checker.generate_report(), indent=2)
    
    def generate_text_report(self) -> str:
        """生成文本格式的健康报告"""
        report = self._checker.generate_report()
        lines = ["=" * 60, "Health Status Report", "=" * 60, ""]
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Overall Status: {report['overall_status']}")
        lines.append(f"Message: {report['message']}")
        lines.append(f"Active Alerts: {report['active_alerts']}")
        lines.append(f"Total Checks: {report['total_checks']}")
        lines.append("")
        lines.append("Backends:")
        for name, info in report['backends'].items():
            lines.append(f"  - {name}: {info['status']} ({info['endpoint']})")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def generate_html_report(self) -> str:
        """生成HTML格式的健康报告"""
        report = self._checker.generate_report()
        html = ['<!DOCTYPE html>', '<html>', '<head><title>Health Report</title></head>', '<body>']
        html.append(f"<h1>Health Status Report</h1>")
        html.append(f"<p>Timestamp: {report['timestamp']}</p>")
        html.append(f"<p>Overall Status: <strong>{report['overall_status']}</strong></p>")
        html.append(f"<p>Active Alerts: {report['active_alerts']}</p>")
        html.append("<h2>Backends</h2><ul>")
        for name, info in report['backends'].items():
            color = "green" if info['status'] == "healthy" else "red"
            html.append(f"<li><span style='color:{color}'>{name}</span>: {info['status']} ({info['endpoint']})</li>")
        html.append("</ul></body></html>")
        return "\n".join(html)


class HealthThresholdManager:
    """健康阈值管理器"""
    
    def __init__(self):
        self._thresholds: Dict[str, dict] = {}
        self._defaults = {
            "latency_warning_ms": 1000,
            "latency_critical_ms": 5000,
            "failure_rate_warning": 0.1,
            "failure_rate_critical": 0.5,
            "uptime_min_percentage": 99.0,
        }
    
    def set_threshold(self, backend_name: str, key: str, value: float) -> None:
        if backend_name not in self._thresholds:
            self._thresholds[backend_name] = dict(self._defaults)
        self._thresholds[backend_name][key] = value
    
    def get_threshold(self, backend_name: str, key: str) -> float:
        return self._thresholds.get(backend_name, {}).get(key, self._defaults.get(key, 0))
    
    def check_latency_threshold(self, backend_name: str, latency_ms: float) -> Tuple[str, bool]:
        """检查延迟阈值"""
        warning = self.get_threshold(backend_name, "latency_warning_ms")
        critical = self.get_threshold(backend_name, "latency_critical_ms")
        if latency_ms >= critical: return ("critical", True)
        if latency_ms >= warning: return ("warning", True)
        return ("ok", False)
    
    def check_failure_rate(self, backend_name: str, failures: int, total: int) -> Tuple[str, bool]:
        """检查失败率"""
        if total == 0: return ("ok", False)
        rate = failures / total
        warning = self.get_threshold(backend_name, "failure_rate_warning")
        critical = self.get_threshold(backend_name, "failure_rate_critical")
        if rate >= critical: return ("critical", True)
        if rate >= warning: return ("warning", True)
        return ("ok", False)


class HealthHistory:
    """健康历史记录"""
    
    def __init__(self, max_size: int = 1000):
        self._history: deque = deque(maxlen=max_size)
    
    def add_entry(self, result: "HealthCheckResult") -> None:
        self._history.append({
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "backend_name": result.backend_name,
            "status": result.status.value,
            "latency_ms": result.latency_ms,
            "error_message": result.error_message,
        })
    
    def get_history(self, backend_name: Optional[str] = None, limit: int = 100) -> List[dict]:
        entries = list(self._history)
        if backend_name:
            entries = [e for e in entries if e["backend_name"] == backend_name]
        return entries[-limit:]
    
    def get_uptime_percentage(self, backend_name: str, hours: int = 24) -> float:
        """计算指定时间段内的可用性"""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        entries = [e for e in self._history if e["backend_name"] == backend_name and 
                   e.get("timestamp") and datetime.fromisoformat(e["timestamp"]).timestamp() > cutoff]
        if not entries: return 100.0
        healthy = sum(1 for e in entries if e["status"] == "healthy")
        return (healthy / len(entries)) * 100


class HealthEventBus:
    """健康事件总线"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            return True
        return False
    
    def publish(self, event_type: str, data: Any) -> None:
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception:
                pass


class HealthAggregator:
    """健康状态聚合器"""
    
    def __init__(self, checker: "BusHealthChecker"):
        self._checker = checker
    
    def aggregate_by_region(self, region_map: Dict[str, str]) -> Dict[str, dict]:
        """按区域聚合健康状态"""
        region_health: Dict[str, Dict] = defaultdict(lambda: {"healthy": 0, "unhealthy": 0, "total": 0})
        for name, status in self._checker.get_all_backend_status().items():
            region = region_map.get(name, "unknown")
            region_health[region]["total"] += 1
            if status.get("status") == HealthStatusEnum.HEALTHY.value:
                region_health[region]["healthy"] += 1
            else:
                region_health[region]["unhealthy"] += 1
        return dict(region_health)
    
    def aggregate_by_type(self, type_map: Dict[str, str]) -> Dict[str, dict]:
        """按类型聚合健康状态"""
        type_health: Dict[str, Dict] = defaultdict(lambda: {"healthy": 0, "unhealthy": 0, "total": 0})
        for name, status in self._checker.get_all_backend_status().items():
            btype = type_map.get(name, "unknown")
            type_health[btype]["total"] += 1
            if status.get("status") == HealthStatusEnum.HEALTHY.value:
                type_health[btype]["healthy"] += 1
            else:
                type_health[btype]["unhealthy"] += 1
        return dict(type_health)
    
    def get_weighted_health_score(self, weights: Dict[str, float]) -> float:
        """计算加权健康分数"""
        total_weight = sum(weights.values())
        weighted_score = 0.0
        for name, weight in weights.items():
            status = self._checker.get_all_backend_status().get(name, {})
            if status.get("status") == HealthStatusEnum.HEALTHY.value:
                weighted_score += weight
        return (weighted_score / total_weight * 100) if total_weight > 0 else 0.0
