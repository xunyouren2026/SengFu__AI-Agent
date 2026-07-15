"""
Policy Validation Engine Module

Path whitelist/blacklist, domain whitelist, command whitelist,
resource quota policies, time-based policies, and role-based access control.
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class PolicyAction(Enum):
    """Actions that can be taken by a policy."""
    ALLOW = "allow"
    DENY = "deny"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    LOG = "log"
    SANITIZE = "sanitize"
    TRANSFORM = "transform"


class PolicyEffect(Enum):
    """Effect of a policy decision."""
    PERMIT = "permit"
    DENY = "deny"
    PARTIAL = "partial"
    DEFER = "defer"


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""
    decision_id: str
    effect: PolicyEffect
    action: PolicyAction
    reason: str
    policy_id: str = ""
    priority: int = 0
    conditions_matched: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "effect": self.effect.value,
            "action": self.action.value,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "priority": self.priority,
            "conditions_matched": self.conditions_matched,
            "timestamp": self.timestamp,
        }

    @property
    def is_allowed(self) -> bool:
        return self.effect == PolicyEffect.PERMIT

    @property
    def is_denied(self) -> bool:
        return self.effect == PolicyEffect.DENY


@dataclass
class PolicyRule:
    """A single policy rule."""
    rule_id: str
    name: str
    description: str
    action: PolicyAction
    priority: int = 0
    enabled: bool = True
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "action": self.action.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "conditions": self.conditions,
        }


class PathPolicy:
    """Manages path-based whitelist and blacklist policies."""

    def __init__(self) -> None:
        self._whitelist: List[str] = []
        self._blacklist: List[str] = []
        self._whitelist_enabled: bool = False
        self._blacklist_enabled: bool = True
        self._allowed_extensions: Set[str] = set()
        self._denied_extensions: Set[str] = set()
        self._max_path_length: int = 4096
        self._compiled_whitelist: List[re.Pattern] = []
        self._compiled_blacklist: List[re.Pattern] = []

    def add_whitelist(self, pattern: str) -> None:
        self._whitelist.append(pattern)
        self._compiled_whitelist.append(re.compile(fnmatch.translate(pattern)))

    def add_blacklist(self, pattern: str) -> None:
        self._blacklist.append(pattern)
        self._compiled_blacklist.append(re.compile(fnmatch.translate(pattern)))

    def set_whitelist_enabled(self, enabled: bool) -> None:
        self._whitelist_enabled = enabled

    def set_blacklist_enabled(self, enabled: bool) -> None:
        self._blacklist_enabled = enabled

    def add_allowed_extension(self, ext: str) -> None:
        if not ext.startswith("."):
            ext = "." + ext
        self._allowed_extensions.add(ext.lower())

    def add_denied_extension(self, ext: str) -> None:
        if not ext.startswith("."):
            ext = "." + ext
        self._denied_extensions.add(ext.lower())

    def evaluate(self, path: str) -> PolicyDecision:
        if not path or len(path) > self._max_path_length:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"Path length {len(path) if path else 0} exceeds limit or is empty",
            )
        if self._blacklist_enabled:
            for pattern, compiled in zip(self._blacklist, self._compiled_blacklist):
                if compiled.match(path):
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.DENY,
                        action=PolicyAction.DENY,
                        reason=f"Path matches blacklist pattern: {pattern}",
                    )
        if self._denied_extensions:
            import os
            _, ext = os.path.splitext(path)
            if ext.lower() in self._denied_extensions:
                return PolicyDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    effect=PolicyEffect.DENY,
                    action=PolicyAction.DENY,
                    reason=f"File extension '{ext}' is denied",
                )
        if self._whitelist_enabled and self._whitelist:
            matched = False
            for pattern, compiled in zip(self._whitelist, self._compiled_whitelist):
                if compiled.match(path):
                    matched = True
                    break
            if not matched:
                return PolicyDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    effect=PolicyEffect.DENY,
                    action=PolicyAction.DENY,
                    reason="Path does not match any whitelist pattern",
                )
        if self._allowed_extensions:
            import os
            _, ext = os.path.splitext(path)
            if ext.lower() not in self._allowed_extensions:
                return PolicyDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    effect=PolicyEffect.DENY,
                    action=PolicyAction.DENY,
                    reason=f"File extension '{ext}' is not in allowed extensions",
                )
        return PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.PERMIT,
            action=PolicyAction.ALLOW,
            reason="Path policy check passed",
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "whitelist": self._whitelist,
            "blacklist": self._blacklist,
            "whitelist_enabled": self._whitelist_enabled,
            "blacklist_enabled": self._blacklist_enabled,
            "allowed_extensions": list(self._allowed_extensions),
            "denied_extensions": list(self._denied_extensions),
        }


class DomainPolicy:
    """Manages domain-based whitelist and blacklist policies."""

    def __init__(self) -> None:
        self._whitelist: Set[str] = set()
        self._blacklist: Set[str] = set()
        self._whitelist_enabled: bool = False
        self._blacklist_enabled: bool = True
        self._allowed_schemes: Set[str] = {"https", "http"}
        self._denied_schemes: Set[str] = {"file", "gopher", "dict"}
        self._allowed_ports: Set[int] = {80, 443, 8080, 8443}
        self._denied_ports: Set[int] = set()
        self._private_ip_blocked: bool = True

    def add_whitelist_domain(self, domain: str) -> None:
        self._whitelist.add(domain.lower())

    def add_blacklist_domain(self, domain: str) -> None:
        self._blacklist.add(domain.lower())

    def set_whitelist_enabled(self, enabled: bool) -> None:
        self._whitelist_enabled = enabled

    def set_blacklist_enabled(self, enabled: bool) -> None:
        self._blacklist_enabled = enabled

    def add_allowed_scheme(self, scheme: str) -> None:
        self._allowed_schemes.add(scheme.lower())

    def add_denied_scheme(self, scheme: str) -> None:
        self._denied_schemes.add(scheme.lower())

    def add_allowed_port(self, port: int) -> None:
        self._allowed_ports.add(port)

    def add_denied_port(self, port: int) -> None:
        self._denied_ports.add(port)

    def evaluate(self, url: str) -> PolicyDecision:
        parsed = self._parse_url(url)
        if parsed is None:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"Could not parse URL: {url[:100]}",
            )
        scheme = parsed.get("scheme", "").lower()
        host = parsed.get("host", "").lower()
        port = parsed.get("port", 0)
        if scheme in self._denied_schemes:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"URL scheme '{scheme}' is denied",
            )
        if self._allowed_schemes and scheme not in self._allowed_schemes:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"URL scheme '{scheme}' is not in allowed schemes",
            )
        if port in self._denied_ports:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"Port {port} is denied",
            )
        if self._allowed_ports and port and port not in self._allowed_ports:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"Port {port} is not in allowed ports",
            )
        if self._private_ip_blocked and self._is_private_ip(host):
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"Host '{host}' resolves to private IP range",
            )
        if self._blacklist_enabled:
            for blocked in self._blacklist:
                if host == blocked or host.endswith("." + blocked):
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.DENY,
                        action=PolicyAction.DENY,
                        reason=f"Domain '{host}' matches blacklist entry '{blocked}'",
                    )
        if self._whitelist_enabled and self._whitelist:
            matched = False
            for allowed in self._whitelist:
                if host == allowed or host.endswith("." + allowed):
                    matched = True
                    break
            if not matched:
                return PolicyDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    effect=PolicyEffect.DENY,
                    action=PolicyAction.DENY,
                    reason=f"Domain '{host}' does not match any whitelist entry",
                )
        return PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.PERMIT,
            action=PolicyAction.ALLOW,
            reason="Domain policy check passed",
        )

    def _parse_url(self, url: str) -> Optional[Dict[str, Any]]:
        pattern = re.compile(
            r'^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://'
            r'(?:(?P<userinfo>[^@]+)@)?'
            r'(?P<host>[^:/\?#]+)'
            r'(?::(?P<port>\d+))?'
            r'(?P<path>/[^\?#]*)?'
            r'(?:\?(?P<query>[^#]*))?'
            r'(?:#(?P<fragment>.*))?$'
        )
        match = pattern.match(url)
        if not match:
            return None
        port = int(match.group("port")) if match.group("port") else 0
        if port == 0:
            scheme = match.group("scheme", "").lower()
            port = 443 if scheme == "https" else 80
        return {
            "scheme": match.group("scheme"),
            "userinfo": match.group("userinfo"),
            "host": match.group("host"),
            "port": port,
            "path": match.group("path") or "/",
            "query": match.group("query"),
            "fragment": match.group("fragment"),
        }

    def _is_private_ip(self, host: str) -> bool:
        private_patterns = [
            r'^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}$',
            r'^192\.168\.\d{1,3}\.\d{1,3}$',
            r'^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^0\.0\.0\.0$',
            r'^localhost$',
            r'^169\.254\.169\.254$',
        ]
        for pattern in private_patterns:
            if re.match(pattern, host):
                return True
        return False


class CommandPolicy:
    """Manages command execution whitelist and blacklist."""

    def __init__(self) -> None:
        self._whitelist: List[str] = []
        self._blacklist: List[str] = []
        self._whitelist_enabled: bool = False
        self._blacklist_enabled: bool = True
        self._allowed_flags: Dict[str, Set[str]] = {}
        self._denied_flags: Set[str] = set()
        self._max_command_length: int = 4096
        self._compiled_whitelist: List[re.Pattern] = []
        self._compiled_blacklist: List[re.Pattern] = []

    def add_whitelist(self, pattern: str) -> None:
        self._whitelist.append(pattern)
        self._compiled_whitelist.append(re.compile(pattern))

    def add_blacklist(self, pattern: str) -> None:
        self._blacklist.append(pattern)
        self._compiled_blacklist.append(re.compile(pattern))

    def set_whitelist_enabled(self, enabled: bool) -> None:
        self._whitelist_enabled = enabled

    def set_blacklist_enabled(self, enabled: bool) -> None:
        self._blacklist_enabled = enabled

    def add_allowed_flag(self, command: str, flag: str) -> None:
        if command not in self._allowed_flags:
            self._allowed_flags[command] = set()
        self._allowed_flags[command].add(flag)

    def add_denied_flag(self, flag: str) -> None:
        self._denied_flags.add(flag)

    def evaluate(self, command: str) -> PolicyDecision:
        if not command or len(command) > self._max_command_length:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason="Command is empty or exceeds maximum length",
            )
        base_command = command.strip().split()[0] if command.strip() else ""
        if self._blacklist_enabled:
            for pattern, compiled in zip(self._blacklist, self._compiled_blacklist):
                if compiled.search(command):
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.DENY,
                        action=PolicyAction.DENY,
                        reason=f"Command matches blacklist pattern: {pattern}",
                    )
        if self._denied_flags:
            flags = re.findall(r'-+\w+', command)
            for flag in flags:
                if flag in self._denied_flags:
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.DENY,
                        action=PolicyAction.DENY,
                        reason=f"Denied flag '{flag}' detected in command",
                    )
        if self._whitelist_enabled and self._whitelist:
            matched = False
            for pattern, compiled in zip(self._whitelist, self._compiled_whitelist):
                if compiled.search(command):
                    matched = True
                    break
            if not matched:
                return PolicyDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    effect=PolicyEffect.DENY,
                    action=PolicyAction.DENY,
                    reason=f"Command '{base_command}' does not match any whitelist pattern",
                )
        if base_command in self._allowed_flags:
            allowed = self._allowed_flags[base_command]
            flags = re.findall(r'-+\w+', command)
            for flag in flags:
                if flag not in allowed:
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.DENY,
                        action=PolicyAction.DENY,
                        reason=f"Flag '{flag}' not allowed for command '{base_command}'",
                    )
        return PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.PERMIT,
            action=PolicyAction.ALLOW,
            reason="Command policy check passed",
        )


class ResourceQuota:
    """Manages resource quota policies."""

    def __init__(self) -> None:
        self._quotas: Dict[str, Dict[str, Any]] = {}
        self._usage: Dict[str, Dict[str, float]] = {}
        self._reset_interval: float = 3600.0
        self._last_reset: float = time.time()

    def set_quota(
        self,
        resource: str,
        limit: float,
        window_seconds: float = 3600.0,
        per_user: bool = False,
    ) -> None:
        self._quotas[resource] = {
            "limit": limit,
            "window_seconds": window_seconds,
            "per_user": per_user,
        }

    def record_usage(
        self, resource: str, amount: float, user_id: str = ""
    ) -> PolicyDecision:
        self._check_reset()
        key = f"{resource}:{user_id}" if user_id and self._quotas.get(resource, {}).get("per_user") else resource
        if key not in self._usage:
            self._usage[key] = {"total": 0.0, "last_updated": time.time()}
        quota = self._quotas.get(resource)
        if quota is None:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.PERMIT,
                action=PolicyAction.ALLOW,
                reason=f"No quota defined for resource '{resource}'",
            )
        self._usage[key]["total"] += amount
        self._usage[key]["last_updated"] = time.time()
        if self._usage[key]["total"] > quota["limit"]:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.QUOTA,
                reason=(
                    f"Resource '{resource}' quota exceeded: "
                    f"{self._usage[key]['total']:.2f}/{quota['limit']:.2f}"
                ),
                metadata={"usage": self._usage[key]["total"], "limit": quota["limit"]},
            )
        return PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.PERMIT,
            action=PolicyAction.ALLOW,
            reason=f"Resource '{resource}' usage within quota",
        )

    def check_quota(
        self, resource: str, user_id: str = ""
    ) -> Tuple[float, float]:
        key = f"{resource}:{user_id}" if user_id and self._quotas.get(resource, {}).get("per_user") else resource
        usage = self._usage.get(key, {}).get("total", 0.0)
        limit = self._quotas.get(resource, {}).get("limit", float("inf"))
        return usage, limit

    def _check_reset(self) -> None:
        now = time.time()
        if now - self._last_reset > self._reset_interval:
            self._usage.clear()
            self._last_reset = now

    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for resource, quota in self._quotas.items():
            usage, limit = self.check_quota(resource)
            result[resource] = {
                "usage": usage,
                "limit": limit,
                "percentage": (usage / limit * 100) if limit > 0 else 0,
            }
        return result


class TimePolicy:
    """Manages time-based access policies."""

    def __init__(self) -> None:
        self._time_ranges: List[Dict[str, Any]] = []
        self._blocked_dates: Set[str] = set()
        self._blocked_weekdays: Set[int] = set()
        self._timezone_offset: float = 0.0

    def add_time_range(
        self,
        name: str,
        start_hour: int,
        start_minute: int,
        end_hour: int,
        end_minute: int,
        weekdays: Optional[List[int]] = None,
        action: PolicyAction = PolicyAction.ALLOW,
    ) -> None:
        self._time_ranges.append({
            "name": name,
            "start_hour": start_hour,
            "start_minute": start_minute,
            "end_hour": end_hour,
            "end_minute": end_minute,
            "weekdays": weekdays or list(range(7)),
            "action": action,
        })

    def block_date(self, date_str: str) -> None:
        self._blocked_dates.add(date_str)

    def block_weekday(self, weekday: int) -> None:
        if 0 <= weekday <= 6:
            self._blocked_weekdays.add(weekday)

    def evaluate(self) -> PolicyDecision:
        now = time.time()
        local_time = time.localtime(now + self._timezone_offset)
        current_hour = local_time.tm_hour
        current_minute = local_time.tm_min
        current_weekday = local_time.tm_wmday
        date_str = time.strftime("%Y-%m-%d", local_time)
        if date_str in self._blocked_dates:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"Access blocked for date: {date_str}",
            )
        if current_weekday in self._blocked_weekdays:
            weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"Access blocked for weekday: {weekday_names[current_weekday]}",
            )
        current_minutes = current_hour * 60 + current_minute
        for tr in self._time_ranges:
            if current_weekday not in tr["weekdays"]:
                continue
            start_minutes = tr["start_hour"] * 60 + tr["start_minute"]
            end_minutes = tr["end_hour"] * 60 + tr["end_minute"]
            if start_minutes <= current_minutes <= end_minutes:
                if tr["action"] == PolicyAction.ALLOW:
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.PERMIT,
                        action=PolicyAction.ALLOW,
                        reason=f"Within allowed time range: {tr['name']}",
                    )
                else:
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.DENY,
                        action=PolicyAction.DENY,
                        reason=f"Within denied time range: {tr['name']}",
                    )
        if self._time_ranges:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason="Outside all defined time ranges",
            )
        return PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.PERMIT,
            action=PolicyAction.ALLOW,
            reason="No time restrictions defined",
        )


class RBACPolicy:
    """Role-Based Access Control policy manager."""

    def __init__(self) -> None:
        self._roles: Dict[str, Set[str]] = {}
        self._user_roles: Dict[str, Set[str]] = {}
        self._role_hierarchy: Dict[str, Set[str]] = {}
        self._permissions: Dict[str, Set[str]] = {}
        self._resource_permissions: Dict[str, Dict[str, Set[str]]] = {}

    def create_role(self, role: str, permissions: Optional[List[str]] = None) -> None:
        self._roles[role] = set(permissions or [])
        if role not in self._permissions:
            self._permissions[role] = set(permissions or [])

    def assign_role(self, user_id: str, role: str) -> None:
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        self._user_roles[user_id].add(role)

    def revoke_role(self, user_id: str, role: str) -> None:
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role)

    def add_permission(self, role: str, permission: str) -> None:
        if role not in self._permissions:
            self._permissions[role] = set()
        self._permissions[role].add(permission)

    def set_parent_role(self, child_role: str, parent_role: str) -> None:
        if child_role not in self._role_hierarchy:
            self._role_hierarchy[child_role] = set()
        self._role_hierarchy[child_role].add(parent_role)

    def set_resource_permission(
        self, resource: str, role: str, actions: List[str]
    ) -> None:
        if resource not in self._resource_permissions:
            self._resource_permissions[resource] = {}
        if role not in self._resource_permissions[resource]:
            self._resource_permissions[resource][role] = set()
        self._resource_permissions[resource][role].update(actions)

    def evaluate(
        self, user_id: str, permission: str, resource: str = ""
    ) -> PolicyDecision:
        user_roles = self._user_roles.get(user_id, set())
        if not user_roles:
            return PolicyDecision(
                decision_id=uuid.uuid4().hex[:12],
                effect=PolicyEffect.DENY,
                action=PolicyAction.DENY,
                reason=f"User '{user_id}' has no assigned roles",
            )
        all_roles = self._get_all_roles(user_roles)
        for role in all_roles:
            if permission in self._permissions.get(role, set()):
                if resource:
                    resource_perms = self._resource_permissions.get(resource, {})
                    allowed_actions = set()
                    for r in all_roles:
                        allowed_actions.update(resource_perms.get(r, set()))
                    if permission in allowed_actions:
                        return PolicyDecision(
                            decision_id=uuid.uuid4().hex[:12],
                            effect=PolicyEffect.PERMIT,
                            action=PolicyAction.ALLOW,
                            reason=f"User '{user_id}' has permission '{permission}' via role '{role}'",
                        )
                    return PolicyDecision(
                        decision_id=uuid.uuid4().hex[:12],
                        effect=PolicyEffect.DENY,
                        action=PolicyAction.DENY,
                        reason=f"Permission '{permission}' not granted for resource '{resource}'",
                    )
                return PolicyDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    effect=PolicyEffect.PERMIT,
                    action=PolicyAction.ALLOW,
                    reason=f"User '{user_id}' has permission '{permission}' via role '{role}'",
                )
        return PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.DENY,
            action=PolicyAction.DENY,
            reason=f"User '{user_id}' lacks permission '{permission}'",
            conditions_matched=[f"roles: {', '.join(all_roles)}"],
        )

    def _get_all_roles(self, roles: Set[str]) -> Set[str]:
        all_roles: Set[str] = set(roles)
        queue = list(roles)
        while queue:
            role = queue.pop(0)
            parents = self._role_hierarchy.get(role, set())
            for parent in parents:
                if parent not in all_roles:
                    all_roles.add(parent)
                    queue.append(parent)
        return all_roles

    def get_user_roles(self, user_id: str) -> Set[str]:
        return self._user_roles.get(user_id, set())

    def get_role_permissions(self, role: str) -> Set[str]:
        return self._permissions.get(role, set())


class PolicyEvaluator:
    """Evaluates custom policy rules against contexts."""

    def __init__(self) -> None:
        self._rules: List[PolicyRule] = []
        self._condition_handlers: Dict[str, Callable[[Any, Dict[str, Any]], bool]] = {
            "equals": self._cond_equals,
            "not_equals": self._cond_not_equals,
            "contains": self._cond_contains,
            "not_contains": self._cond_not_contains,
            "matches": self._cond_matches,
            "greater_than": self._cond_greater_than,
            "less_than": self._cond_less_than,
            "in": self._cond_in,
            "not_in": self._cond_not_in,
            "exists": self._cond_exists,
            "not_exists": self._cond_not_exists,
        }

    def add_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> Optional[PolicyRule]:
        for i, rule in enumerate(self._rules):
            if rule.rule_id == rule_id:
                return self._rules.pop(i)
        return None

    def evaluate(self, context: Dict[str, Any]) -> PolicyDecision:
        for rule in self._rules:
            if not rule.enabled:
                continue
            matched_conditions: List[str] = []
            all_match = True
            for condition in rule.conditions:
                field_name = condition.get("field", "")
                operator = condition.get("operator", "equals")
                expected = condition.get("value")
                actual = self._get_nested_value(context, field_name)
                handler = self._condition_handlers.get(operator)
                if handler is None:
                    all_match = False
                    break
                if not handler(actual, condition):
                    all_match = False
                    break
                matched_conditions.append(f"{field_name} {operator} {expected}")
            if all_match:
                effect = (
                    PolicyEffect.PERMIT
                    if rule.action == PolicyAction.ALLOW
                    else PolicyEffect.DENY
                )
                return PolicyDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    effect=effect,
                    action=rule.action,
                    reason=f"Rule '{rule.name}' matched: {rule.description}",
                    policy_id=rule.rule_id,
                    priority=rule.priority,
                    conditions_matched=matched_conditions,
                )
        return PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.PERMIT,
            action=PolicyAction.ALLOW,
            reason="No rules matched; default allow",
        )

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        parts = path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

    def _cond_equals(self, actual: Any, condition: Dict[str, Any]) -> bool:
        return actual == condition.get("value")

    def _cond_not_equals(self, actual: Any, condition: Dict[str, Any]) -> bool:
        return actual != condition.get("value")

    def _cond_contains(self, actual: Any, condition: Dict[str, Any]) -> bool:
        if isinstance(actual, str) and isinstance(condition.get("value"), str):
            return condition["value"] in actual
        if isinstance(actual, (list, set)):
            return condition.get("value") in actual
        return False

    def _cond_not_contains(self, actual: Any, condition: Dict[str, Any]) -> bool:
        return not self._cond_contains(actual, condition)

    def _cond_matches(self, actual: Any, condition: Dict[str, Any]) -> bool:
        if isinstance(actual, str) and isinstance(condition.get("value"), str):
            import re
            return bool(re.search(condition["value"], actual))
        return False

    def _cond_greater_than(self, actual: Any, condition: Dict[str, Any]) -> bool:
        try:
            return float(actual) > float(condition.get("value", 0))
        except (TypeError, ValueError):
            return False

    def _cond_less_than(self, actual: Any, condition: Dict[str, Any]) -> bool:
        try:
            return float(actual) < float(condition.get("value", 0))
        except (TypeError, ValueError):
            return False

    def _cond_in(self, actual: Any, condition: Dict[str, Any]) -> bool:
        values = condition.get("value", [])
        if isinstance(values, str):
            values = [values]
        return actual in values

    def _cond_not_in(self, actual: Any, condition: Dict[str, Any]) -> bool:
        return not self._cond_in(actual, condition)

    def _cond_exists(self, actual: Any, condition: Dict[str, Any]) -> bool:
        return actual is not None

    def _cond_not_exists(self, actual: Any, condition: Dict[str, Any]) -> bool:
        return actual is None

    def list_rules(self) -> List[PolicyRule]:
        return list(self._rules)


class PolicyEngine:
    """Main policy engine that orchestrates all policy types."""

    def __init__(self, default_action: PolicyAction = PolicyAction.ALLOW) -> None:
        self.default_action: PolicyAction = default_action
        self.path_policy: PathPolicy = PathPolicy()
        self.domain_policy: DomainPolicy = DomainPolicy()
        self.command_policy: CommandPolicy = CommandPolicy()
        self.resource_quota: ResourceQuota = ResourceQuota()
        self.time_policy: TimePolicy = TimePolicy()
        self.rbac_policy: RBACPolicy = RBACPolicy()
        self.evaluator: PolicyEvaluator = PolicyEvaluator()
        self._decision_log: List[PolicyDecision] = []
        self._max_log: int = 10000

    def evaluate_all(
        self,
        context: Dict[str, Any],
        check_path: Optional[str] = None,
        check_domain: Optional[str] = None,
        check_command: Optional[str] = None,
        check_resource: Optional[Tuple[str, float]] = None,
        check_time: bool = False,
        check_rbac: Optional[Tuple[str, str]] = None,
    ) -> PolicyDecision:
        decisions: List[PolicyDecision] = []
        if check_path:
            decisions.append(self.path_policy.evaluate(check_path))
        if check_domain:
            decisions.append(self.domain_policy.evaluate(check_domain))
        if check_command:
            decisions.append(self.command_policy.evaluate(check_command))
        if check_resource:
            resource_name, amount = check_resource
            decisions.append(self.resource_quota.record_usage(resource_name, amount))
        if check_time:
            decisions.append(self.time_policy.evaluate())
        if check_rbac:
            user_id, permission = check_rbac
            decisions.append(self.rbac_policy.evaluate(user_id, permission))
        custom_decision = self.evaluator.evaluate(context)
        decisions.append(custom_decision)
        for d in decisions:
            if d.is_denied:
                self._log_decision(d)
                return d
        self._log_decision(decisions[0] if decisions else PolicyDecision(
            decision_id=uuid.uuid4().hex[:12],
            effect=PolicyEffect.PERMIT,
            action=PolicyAction.ALLOW,
            reason="All policy checks passed",
        ))
        return decisions[0]

    def _log_decision(self, decision: PolicyDecision) -> None:
        self._decision_log.append(decision)
        if len(self._decision_log) > self._max_log:
            self._decision_log = self._decision_log[-self._max_log:]

    def get_decision_log(
        self, limit: int = 100, effect: Optional[PolicyEffect] = None
    ) -> List[PolicyDecision]:
        log = self._decision_log
        if effect:
            log = [d for d in log if d.effect == effect]
        return log[-limit:]
