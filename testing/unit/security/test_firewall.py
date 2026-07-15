"""
TestFirewall - 安全单元测试：防火墙模块

模块路径: testing/unit/security/test_firewall.py
"""
import time
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


@dataclass
class FirewallRule:
    rule_id: str
    action: str  # "allow", "deny", "rate_limit"
    source_ip: Optional[str] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    path_pattern: Optional[str] = None
    priority: int = 100
    enabled: bool = True


@dataclass
class FirewallLog:
    timestamp: float
    source_ip: str
    action: str
    rule_id: Optional[str]
    path: str
    status_code: int


class MockFirewall:
    """模拟防火墙"""

    def __init__(self):
        self.rules: List[FirewallRule] = []
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = defaultdict(list)
        self.logs: List[FirewallLog] = []
        self.default_action = "allow"
        self.max_requests_per_minute = 60
        self.max_requests_per_hour = 1000

    def add_rule(self, rule: FirewallRule) -> bool:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        return True

    def remove_rule(self, rule_id: str) -> bool:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < before

    def block_ip(self, ip: str, reason: str = "") -> bool:
        self.blocked_ips.add(ip)
        return True

    def unblock_ip(self, ip: str) -> bool:
        if ip in self.blocked_ips:
            self.blocked_ips.discard(ip)
            return True
        return False

    def is_ip_blocked(self, ip: str) -> bool:
        return ip in self.blocked_ips

    def check_rate_limit(self, ip: str) -> Dict[str, Any]:
        now = time.time()
        self.rate_limits[ip] = [t for t in self.rate_limits[ip] if now - t < 3600]
        minute_ago = [t for t in self.rate_limits[ip] if now - t < 60]
        return {
            "requests_last_minute": len(minute_ago),
            "requests_last_hour": len(self.rate_limits[ip]),
            "minute_limit": self.max_requests_per_minute,
            "hour_limit": self.max_requests_per_hour,
            "allowed": len(minute_ago) < self.max_requests_per_minute
                     and len(self.rate_limits[ip]) < self.max_requests_per_hour,
        }

    def record_request(self, ip: str):
        self.rate_limits[ip].append(time.time())

    def evaluate_request(self, ip: str, path: str = "/",
                         port: int = 443, protocol: str = "https") -> Dict[str, Any]:
        if self.is_ip_blocked(ip):
            self._log(ip, "deny", None, path, 403)
            return {"allowed": False, "reason": "IP blocked", "status_code": 403}

        rate = self.check_rate_limit(ip)
        if not rate["allowed"]:
            self._log(ip, "deny", None, path, 429)
            return {"allowed": False, "reason": "Rate limit exceeded", "status_code": 429}

        for rule in self.rules:
            if not rule.enabled:
                continue
            if self._matches_rule(rule, ip, path, port, protocol):
                self._log(ip, rule.action, rule.rule_id, path,
                          200 if rule.action == "allow" else 403)
                return {
                    "allowed": rule.action == "allow",
                    "reason": f"Rule {rule.rule_id}",
                    "status_code": 200 if rule.action == "allow" else 403,
                }

        self._log(ip, self.default_action, None, path, 200)
        return {"allowed": self.default_action == "allow", "reason": "Default", "status_code": 200}

    def _matches_rule(self, rule: FirewallRule, ip: str, path: str,
                      port: int, protocol: str) -> bool:
        if rule.source_ip and rule.source_ip != ip:
            return False
        if rule.destination_port and rule.destination_port != port:
            return False
        if rule.protocol and rule.protocol != protocol:
            return False
        if rule.path_pattern and not re.search(rule.path_pattern, path):
            return False
        return True

    def _log(self, ip: str, action: str, rule_id: Optional[str],
             path: str, status_code: int):
        self.logs.append(FirewallLog(
            timestamp=time.time(), source_ip=ip, action=action,
            rule_id=rule_id, path=path, status_code=status_code,
        ))

    def get_logs(self, ip: Optional[str] = None, limit: int = 100) -> List[FirewallLog]:
        result = self.logs
        if ip:
            result = [l for l in result if l.source_ip == ip]
        return result[-limit:]

    def clear_logs(self):
        self.logs.clear()

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.logs)
        allowed = sum(1 for l in self.logs if l.action == "allow")
        denied = sum(1 for l in self.logs if l.action == "deny")
        return {
            "total_requests": total,
            "allowed": allowed,
            "denied": denied,
            "blocked_ips": len(self.blocked_ips),
            "active_rules": sum(1 for r in self.rules if r.enabled),
        }


class TestFirewallRules:
    """防火墙规则管理测试"""

    def setup_method(self):
        self.fw = MockFirewall()

    def test_add_rule(self):
        rule = FirewallRule(rule_id="r1", action="deny", source_ip="1.2.3.4")
        assert self.fw.add_rule(rule) is True
        assert len(self.fw.rules) == 1

    def test_remove_rule(self):
        rule = FirewallRule(rule_id="r1", action="deny", source_ip="1.2.3.4")
        self.fw.add_rule(rule)
        assert self.fw.remove_rule("r1") is True
        assert len(self.fw.rules) == 0

    def test_remove_nonexistent_rule(self):
        assert self.fw.remove_rule("nonexistent") is False

    def test_rule_priority_ordering(self):
        r1 = FirewallRule(rule_id="low", action="allow", priority=1)
        r2 = FirewallRule(rule_id="high", action="deny", priority=100)
        self.fw.add_rule(r1)
        self.fw.add_rule(r2)
        assert self.fw.rules[0].rule_id == "high"

    def test_disabled_rule_skipped(self):
        rule = FirewallRule(rule_id="r1", action="deny", source_ip="1.2.3.4", enabled=False)
        self.fw.add_rule(rule)
        result = self.fw.evaluate_request("1.2.3.4")
        assert result["allowed"] is True

    def test_rule_matching_ip(self):
        rule = FirewallRule(rule_id="r1", action="deny", source_ip="10.0.0.1")
        self.fw.add_rule(rule)
        result = self.fw.evaluate_request("10.0.0.1")
        assert result["allowed"] is False

    def test_rule_matching_port(self):
        rule = FirewallRule(rule_id="r1", action="deny", destination_port=22)
        self.fw.add_rule(rule)
        result = self.fw.evaluate_request("1.2.3.4", port=22)
        assert result["allowed"] is False

    def test_rule_matching_path(self):
        rule = FirewallRule(rule_id="r1", action="deny", path_pattern=r"/admin/.*")
        self.fw.add_rule(rule)
        result = self.fw.evaluate_request("1.2.3.4", path="/admin/secret")
        assert result["allowed"] is False


class TestIPBlocking:
    """IP封禁测试"""

    def setup_method(self):
        self.fw = MockFirewall()

    def test_block_ip(self):
        self.fw.block_ip("1.2.3.4")
        assert self.fw.is_ip_blocked("1.2.3.4") is True

    def test_unblock_ip(self):
        self.fw.block_ip("1.2.3.4")
        assert self.fw.unblock_ip("1.2.3.4") is True
        assert self.fw.is_ip_blocked("1.2.3.4") is False

    def test_unblock_non_blocked_ip(self):
        assert self.fw.unblock_ip("5.6.7.8") is False

    def test_blocked_ip_denied(self):
        self.fw.block_ip("1.2.3.4")
        result = self.fw.evaluate_request("1.2.3.4")
        assert result["allowed"] is False
        assert result["status_code"] == 403

    def test_multiple_blocked_ips(self):
        for ip in ["1.1.1.1", "2.2.2.2", "3.3.3.3"]:
            self.fw.block_ip(ip)
        assert len(self.fw.blocked_ips) == 3


class TestRateLimiting:
    """速率限制测试"""

    def setup_method(self):
        self.fw = MockFirewall()
        self.fw.max_requests_per_minute = 5

    def test_under_rate_limit(self):
        for _ in range(3):
            self.fw.record_request("1.2.3.4")
        rate = self.fw.check_rate_limit("1.2.3.4")
        assert rate["allowed"] is True
        assert rate["requests_last_minute"] == 3

    def test_over_rate_limit(self):
        for _ in range(10):
            self.fw.record_request("1.2.3.4")
        rate = self.fw.check_rate_limit("1.2.3.4")
        assert rate["allowed"] is False

    def test_rate_limit_returns_429(self):
        for _ in range(10):
            self.fw.record_request("1.2.3.4")
        result = self.fw.evaluate_request("1.2.3.4")
        assert result["status_code"] == 429

    def test_separate_ip_rates(self):
        for _ in range(10):
            self.fw.record_request("1.1.1.1")
        rate = self.fw.check_rate_limit("2.2.2.2")
        assert rate["allowed"] is True
        assert rate["requests_last_minute"] == 0


class TestFirewallLogging:
    """防火墙日志测试"""

    def setup_method(self):
        self.fw = MockFirewall()

    def test_log_created_on_request(self):
        self.fw.evaluate_request("1.2.3.4")
        assert len(self.fw.logs) == 1

    def test_log_contains_correct_fields(self):
        self.fw.evaluate_request("1.2.3.4", path="/api/test")
        log = self.fw.logs[0]
        assert log.source_ip == "1.2.3.4"
        assert log.path == "/api/test"

    def test_get_logs_by_ip(self):
        self.fw.evaluate_request("1.1.1.1")
        self.fw.evaluate_request("2.2.2.2")
        self.fw.evaluate_request("1.1.1.1")
        logs = self.fw.get_logs(ip="1.1.1.1")
        assert len(logs) == 2

    def test_get_logs_limit(self):
        for _ in range(10):
            self.fw.evaluate_request("1.2.3.4")
        logs = self.fw.get_logs(limit=3)
        assert len(logs) == 3

    def test_clear_logs(self):
        self.fw.evaluate_request("1.2.3.4")
        self.fw.clear_logs()
        assert len(self.fw.logs) == 0

    def test_stats(self):
        self.fw.evaluate_request("1.2.3.4")
        self.fw.block_ip("5.6.7.8")
        self.fw.evaluate_request("5.6.7.8")
        stats = self.fw.get_stats()
        assert stats["total_requests"] == 2
        assert stats["denied"] == 1
        assert stats["blocked_ips"] == 1
