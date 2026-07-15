"""
DLP Main Engine Module

Data flow monitoring, network egress inspection, memory scanning,
file I/O monitoring, policy enforcement, and incident response.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class DLPSeverity(Enum):
    """Severity levels for DLP incidents."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DLPAction(Enum):
    """Actions that can be taken on DLP violations."""
    ALLOW = "allow"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    REDACT = "redact"
    ALERT = "alert"
    LOG = "log"
    ENCRYPT = "encrypt"


class DataState(Enum):
    """States of data in the DLP context."""
    AT_REST = "at_rest"
    IN_TRANSIT = "in_transit"
    IN_USE = "in_use"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class DataFlowEvent:
    """Represents a single data flow event."""
    event_id: str
    timestamp: float
    source: str
    destination: str
    data_type: str
    data_state: DataState
    data_size: int
    classification: DataClassification = DataClassification.INTERNAL
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    user_id: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "destination": self.destination,
            "data_type": self.data_type,
            "data_state": self.data_state.value,
            "data_size": self.data_size,
            "classification": self.classification.value,
            "content_hash": self.content_hash,
            "session_id": self.session_id,
            "user_id": self.user_id,
        }


@dataclass
class DLPViolation:
    """Represents a DLP policy violation."""
    violation_id: str
    timestamp: float
    severity: DLPSeverity
    rule_id: str
    rule_name: str
    description: str
    data_type: str
    matched_pattern: str
    matched_content: str
    source: str
    destination: str
    action_taken: DLPAction
    data_classification: DataClassification = DataClassification.INTERNAL
    remediation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "description": self.description,
            "data_type": self.data_type,
            "matched_pattern": self.matched_pattern,
            "matched_content": self.matched_content[:200],
            "source": self.source,
            "destination": self.destination,
            "action_taken": self.action_taken.value,
            "data_classification": self.data_classification.value,
            "remediation": self.remediation,
        }


@dataclass
class DLPRule:
    """A DLP policy rule."""
    rule_id: str
    name: str
    description: str
    severity: DLPSeverity
    action: DLPAction
    patterns: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    classifications: List[DataClassification] = field(default_factory=list)
    max_data_size: int = 0
    enabled: bool = True
    priority: int = 0
    source_filter: str = ""
    destination_filter: str = ""

    def matches_event(self, event: DataFlowEvent) -> bool:
        if not self.enabled:
            return False
        if self.data_types and event.data_type not in self.data_types:
            return False
        if self.classifications and event.classification not in self.classifications:
            return False
        if self.max_data_size > 0 and event.data_size > self.max_data_size:
            return True
        if self.source_filter and not self._match_filter(self.source_filter, event.source):
            return False
        if self.destination_filter and not self._match_filter(self.destination_filter, event.destination):
            return False
        return True

    @staticmethod
    def _match_filter(pattern: str, value: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(value, pattern)


@dataclass
class DLPIncident:
    """A DLP incident that may contain multiple violations."""
    incident_id: str
    timestamp: float = field(default_factory=time.time)
    severity: DLPSeverity = DLPSeverity.MEDIUM
    violations: List[DLPViolation] = field(default_factory=list)
    status: str = "open"
    assigned_to: str = ""
    resolution: str = ""
    resolved_at: float = 0.0

    def add_violation(self, violation: DLPViolation) -> None:
        self.violations.append(violation)
        self._update_severity()

    def _update_severity(self) -> None:
        severity_order = [
            DLPSeverity.CRITICAL, DLPSeverity.HIGH,
            DLPSeverity.MEDIUM, DLPSeverity.LOW, DLPSeverity.INFO,
        ]
        for sev in severity_order:
            if any(v.severity == sev for v in self.violations):
                self.severity = sev
                return

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "violation_count": len(self.violations),
            "status": self.status,
            "assigned_to": self.assigned_to,
            "resolution": self.resolution,
            "violations": [v.to_dict() for v in self.violations],
        }


class DataFlowMonitor:
    """Monitors data flows within the system."""

    def __init__(self, buffer_size: int = 10000) -> None:
        self._events: List[DataFlowEvent] = []
        self._buffer_size: int = buffer_size
        self._flow_counts: Counter = Counter()
        self._source_totals: Dict[str, int] = defaultdict(int)
        self._dest_totals: Dict[str, int] = defaultdict(int)
        self._type_totals: Dict[str, int] = defaultdict(int)
        self._sensitive_flow_count: int = 0
        self._callbacks: List[Callable[[DataFlowEvent], None]] = []

    def register_event(self, event: DataFlowEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._buffer_size:
            self._events = self._events[-self._buffer_size:]
        flow_key = f"{event.source}->{event.destination}"
        self._flow_counts[flow_key] += 1
        self._source_totals[event.source] += event.data_size
        self._dest_totals[event.destination] += event.data_size
        self._type_totals[event.data_type] += 1
        if event.classification in (
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
            DataClassification.TOP_SECRET,
        ):
            self._sensitive_flow_count += 1
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def add_callback(self, callback: Callable[[DataFlowEvent], None]) -> None:
        self._callbacks.append(callback)

    def get_recent_events(self, limit: int = 100) -> List[DataFlowEvent]:
        return self._events[-limit:]

    def get_flow_summary(self) -> Dict[str, Any]:
        return {
            "total_events": len(self._events),
            "unique_flows": len(self._flow_counts),
            "top_flows": self._flow_counts.most_common(20),
            "sensitive_flow_count": self._sensitive_flow_count,
            "source_totals": dict(self._source_totals),
            "destination_totals": dict(self._dest_totals),
            "type_distribution": dict(self._type_totals),
        }

    def detect_anomalous_flows(self, threshold_std: float = 2.0) -> List[Dict[str, Any]]:
        if not self._flow_counts:
            return []
        counts = list(self._flow_counts.values())
        if len(counts) < 5:
            return []
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        std = variance ** 0.5
        if std == 0:
            return []
        anomalies: List[Dict[str, Any]] = []
        for flow, count in self._flow_counts.items():
            z_score = (count - mean) / std
            if abs(z_score) > threshold_std:
                anomalies.append({
                    "flow": flow,
                    "count": count,
                    "z_score": z_score,
                    "mean": mean,
                    "std": std,
                })
        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return anomalies


class NetworkEgressInspector:
    """Inspects network egress for data leakage."""

    def __init__(self) -> None:
        self._allowed_destinations: Set[str] = set()
        self._blocked_destinations: Set[str] = set()
        self._allowed_domains: Set[str] = set()
        self._blocked_domains: Set[str] = set()
        self._sensitive_data_patterns: List[Tuple[str, re.Pattern, DLPSeverity]] = [
            ("email", re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), DLPSeverity.MEDIUM),
            ("phone", re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), DLPSeverity.MEDIUM),
            ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), DLPSeverity.CRITICAL),
            ("credit_card", re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), DLPSeverity.CRITICAL),
            ("api_key_pattern", re.compile(r'(?:api[_-]?key|apikey)\s*[:=]\s*\S{8,}'), DLPSeverity.HIGH),
        ]
        self._inspection_log: List[Dict[str, Any]] = []
        self._max_log: int = 10000
        self._data_volume_by_dest: Dict[str, int] = defaultdict(int)
        self._volume_threshold: int = 10 * 1024 * 1024

    def add_allowed_destination(self, dest: str) -> None:
        self._allowed_destinations.add(dest)

    def add_blocked_destination(self, dest: str) -> None:
        self._blocked_destinations.add(dest)

    def add_allowed_domain(self, domain: str) -> None:
        self._allowed_domains.add(domain.lower())

    def add_blocked_domain(self, domain: str) -> None:
        self._blocked_domains.add(domain.lower())

    def inspect(
        self,
        destination: str,
        data: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[DLPViolation]:
        violations: List[DLPViolation] = []
        dest_lower = destination.lower()
        for blocked in self._blocked_destinations:
            if blocked.lower() in dest_lower:
                violations.append(DLPViolation(
                    violation_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    severity=DLPSeverity.HIGH,
                    rule_id="blocked_dest",
                    rule_name="Blocked Destination",
                    description=f"Network egress to blocked destination: {destination}",
                    data_type="network",
                    matched_pattern=blocked,
                    matched_content=destination,
                    source="internal",
                    destination=destination,
                    action_taken=DLPAction.BLOCK,
                ))
        for blocked_domain in self._blocked_domains:
            if blocked_domain in dest_lower:
                violations.append(DLPViolation(
                    violation_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    severity=DLPSeverity.HIGH,
                    rule_id="blocked_domain",
                    rule_name="Blocked Domain",
                    description=f"Network egress to blocked domain: {blocked_domain}",
                    data_type="network",
                    matched_pattern=blocked_domain,
                    matched_content=destination,
                    source="internal",
                    destination=destination,
                    action_taken=DLPAction.BLOCK,
                ))
        for name, pattern, severity in self._sensitive_data_patterns:
            matches = pattern.findall(data)
            if matches:
                violations.append(DLPViolation(
                    violation_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    severity=severity,
                    rule_id=f"egress_{name}",
                    rule_name=f"Egress {name.title()} Detection",
                    description=f"Sensitive data ({name}) detected in network egress",
                    data_type="network",
                    matched_pattern=name,
                    matched_content=matches[0] if matches else "",
                    source="internal",
                    destination=destination,
                    action_taken=DLPAction.BLOCK if severity in (DLPSeverity.CRITICAL, DLPSeverity.HIGH) else DLPAction.ALERT,
                ))
        self._data_volume_by_dest[destination] += len(data)
        if self._data_volume_by_dest[destination] > self._volume_threshold:
            violations.append(DLPViolation(
                violation_id=uuid.uuid4().hex[:12],
                timestamp=time.time(),
                severity=DLPSeverity.MEDIUM,
                rule_id="volume_threshold",
                rule_name="Data Volume Threshold",
                description=f"Data volume to {destination} exceeds threshold",
                data_type="network",
                matched_pattern="volume",
                matched_content=f"{self._data_volume_by_dest[destination]} bytes",
                source="internal",
                destination=destination,
                action_taken=DLPAction.ALERT,
            ))
        self._inspection_log.append({
            "destination": destination,
            "data_size": len(data),
            "violation_count": len(violations),
            "timestamp": time.time(),
        })
        if len(self._inspection_log) > self._max_log:
            self._inspection_log = self._inspection_log[-self._max_log:]
        return violations

    def get_volume_report(self) -> Dict[str, Any]:
        sorted_dests = sorted(
            self._data_volume_by_dest.items(), key=lambda x: -x[1]
        )
        return {
            "destinations": [
                {"destination": d, "bytes": b}
                for d, b in sorted_dests[:20]
            ],
            "total_bytes": sum(self._data_volume_by_dest.values()),
        }


class MemoryScanner:
    """Scans memory regions for sensitive data patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, DLPSeverity]] = [
            ("email", re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), DLPSeverity.MEDIUM),
            ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), DLPSeverity.CRITICAL),
            ("credit_card", re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), DLPSeverity.CRITICAL),
            ("aws_key", re.compile(r'(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}'), DLPSeverity.CRITICAL),
            ("private_key", re.compile(r'-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----'), DLPSeverity.CRITICAL),
            ("password", re.compile(r'(?:password|passwd|pwd)\s*[:=]\s*\S+', re.I), DLPSeverity.HIGH),
            ("connection_string", re.compile(r'(?:mongodb|postgres|mysql|redis)://[^\s]+', re.I), DLPSeverity.HIGH),
            ("jwt", re.compile(r'eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+'), DLPSeverity.HIGH),
        ]
        self._scan_results: List[Dict[str, Any]] = []
        self._max_results: int = 5000

    def scan_string(self, data: str, source: str = "memory") -> List[DLPViolation]:
        violations: List[DLPViolation] = []
        for name, pattern, severity in self._patterns:
            matches = pattern.finditer(data)
            for match in matches:
                violations.append(DLPViolation(
                    violation_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    severity=severity,
                    rule_id=f"mem_{name}",
                    rule_name=f"Memory {name.title()} Detection",
                    description=f"Sensitive data pattern '{name}' found in {source}",
                    data_type="memory",
                    matched_pattern=name,
                    matched_content=match.group(0),
                    source=source,
                    destination="memory",
                    action_taken=DLPAction.ALERT,
                    metadata={"position": match.start()},
                ))
        self._scan_results.append({
            "source": source,
            "data_size": len(data),
            "violation_count": len(violations),
            "timestamp": time.time(),
        })
        if len(self._scan_results) > self._max_results:
            self._scan_results = self._scan_results[-self._max_results:]
        return violations

    def scan_dict(self, data: Dict[str, Any], source: str = "memory") -> List[DLPViolation]:
        violations: List[DLPViolation] = []
        for key, value in data.items():
            key_lower = key.lower()
            sensitive_keys = {"password", "secret", "token", "api_key", "private_key", "credential"}
            if any(sk in key_lower for sk in sensitive_keys):
                violations.append(DLPViolation(
                    violation_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    severity=DLPSeverity.HIGH,
                    rule_id="sensitive_key",
                    rule_name="Sensitive Key in Memory",
                    description=f"Sensitive key '{key}' found in memory structure",
                    data_type="memory",
                    matched_pattern="sensitive_key",
                    matched_content=key,
                    source=source,
                    destination="memory",
                    action_taken=DLPAction.ALERT,
                ))
            if isinstance(value, str):
                violations.extend(self.scan_string(value, f"{source}.{key}"))
            elif isinstance(value, dict):
                violations.extend(self.scan_dict(value, f"{source}.{key}"))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        violations.extend(self.scan_string(item, f"{source}.{key}[{i}]"))
        return violations


class FileIOMonitor:
    """Monitors file I/O operations for data leakage."""

    def __init__(self) -> None:
        self._monitored_paths: Set[str] = set()
        self._protected_paths: Set[str] = set()
        self._allowed_extensions: Set[str] = set()
        self._blocked_extensions: Set[str] = {".key", ".pem", ".p12", ".pfx", ".jks"}
        self._io_log: List[Dict[str, Any]] = []
        self._max_log: int = 10000
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_threshold: int = 100

    def monitor_path(self, path: str) -> None:
        self._monitored_paths.add(path)

    def protect_path(self, path: str) -> None:
        self._protected_paths.add(path)

    def add_allowed_extension(self, ext: str) -> None:
        self._allowed_extensions.add(ext.lower())

    def add_blocked_extension(self, ext: str) -> None:
        self._blocked_extensions.add(ext.lower())

    def check_read(self, file_path: str, user_id: str = "") -> List[DLPViolation]:
        violations: List[DLPViolation] = []
        for protected in self._protected_paths:
            if file_path.startswith(protected):
                violations.append(DLPViolation(
                    violation_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    severity=DLPSeverity.HIGH,
                    rule_id="protected_read",
                    rule_name="Protected Path Read",
                    description=f"Attempt to read protected path: {file_path}",
                    data_type="file",
                    matched_pattern=protected,
                    matched_content=file_path,
                    source=file_path,
                    destination=user_id or "unknown",
                    action_taken=DLPAction.BLOCK,
                ))
        import os
        _, ext = os.path.splitext(file_path)
        if ext.lower() in self._blocked_extensions:
            violations.append(DLPViolation(
                violation_id=uuid.uuid4().hex[:12],
                timestamp=time.time(),
                severity=DLPSeverity.HIGH,
                rule_id="blocked_extension_read",
                rule_name="Blocked Extension Read",
                description=f"Attempt to read file with blocked extension: {ext}",
                data_type="file",
                matched_pattern=ext,
                matched_content=file_path,
                source=file_path,
                destination=user_id or "unknown",
                action_taken=DLPAction.BLOCK,
            ))
        self._access_counts[file_path] += 1
        if self._access_counts[file_path] > self._access_threshold:
            violations.append(DLPViolation(
                violation_id=uuid.uuid4().hex[:12],
                timestamp=time.time(),
                severity=DLPSeverity.MEDIUM,
                rule_id="access_threshold",
                rule_name="File Access Threshold",
                description=f"File access threshold exceeded for: {file_path}",
                data_type="file",
                matched_pattern="threshold",
                matched_content=f"{self._access_counts[file_path]} accesses",
                source=file_path,
                destination=user_id or "unknown",
                action_taken=DLPAction.ALERT,
            ))
        self._log_io("read", file_path, user_id, len(violations))
        return violations

    def check_write(self, file_path: str, data_size: int, user_id: str = "") -> List[DLPViolation]:
        violations: List[DLPViolation] = []
        for protected in self._protected_paths:
            if file_path.startswith(protected):
                violations.append(DLPViolation(
                    violation_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    severity=DLPSeverity.CRITICAL,
                    rule_id="protected_write",
                    rule_name="Protected Path Write",
                    description=f"Attempt to write to protected path: {file_path}",
                    data_type="file",
                    matched_pattern=protected,
                    matched_content=file_path,
                    source=user_id or "unknown",
                    destination=file_path,
                    action_taken=DLPAction.BLOCK,
                ))
        self._log_io("write", file_path, user_id, len(violations))
        return violations

    def _log_io(self, operation: str, path: str, user_id: str, violations: int) -> None:
        self._io_log.append({
            "operation": operation,
            "path": path,
            "user_id": user_id,
            "violation_count": violations,
            "timestamp": time.time(),
        })
        if len(self._io_log) > self._max_log:
            self._io_log = self._io_log[-self._max_log:]

    def get_io_summary(self) -> Dict[str, Any]:
        ops = Counter(entry["operation"] for entry in self._io_log)
        return {
            "total_operations": len(self._io_log),
            "operations_by_type": dict(ops),
            "most_accessed_files": [
                {"path": path, "count": count}
                for path, count in sorted(
                    self._access_counts.items(), key=lambda x: -x[1]
                )[:20]
            ],
        }


class PolicyEnforcer:
    """Enforces DLP policies on data flows."""

    def __init__(self) -> None:
        self._rules: List[DLPRule] = []
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}

    def add_rule(self, rule: DLPRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        compiled: List[re.Pattern] = []
        for pattern in rule.patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass
        self._compiled_patterns[rule.rule_id] = compiled

    def remove_rule(self, rule_id: str) -> Optional[DLPRule]:
        for i, rule in enumerate(self._rules):
            if rule.rule_id == rule_id:
                self._compiled_patterns.pop(rule_id, None)
                return self._rules.pop(i)
        return None

    def enforce(self, event: DataFlowEvent, content: str = "") -> List[DLPViolation]:
        violations: List[DLPViolation] = []
        for rule in self._rules:
            if not rule.matches_event(event):
                continue
            compiled = self._compiled_patterns.get(rule.rule_id, [])
            for pattern in compiled:
                match = pattern.search(content)
                if match:
                    violations.append(DLPViolation(
                        violation_id=uuid.uuid4().hex[:12],
                        timestamp=time.time(),
                        severity=rule.severity,
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        description=rule.description,
                        data_type=event.data_type,
                        matched_pattern=pattern.pattern,
                        matched_content=match.group(0),
                        source=event.source,
                        destination=event.destination,
                        action_taken=rule.action,
                    ))
        return violations

    def get_rules(self) -> List[DLPRule]:
        return list(self._rules)


class IncidentResponder:
    """Handles DLP incident response."""

    def __init__(self) -> None:
        self._incidents: Dict[str, DLPIncident] = {}
        self._response_actions: Dict[str, Callable[[DLPIncident], None]] = {}
        self._auto_response_rules: Dict[DLPSeverity, DLPAction] = {
            DLPSeverity.CRITICAL: DLPAction.BLOCK,
            DLPSeverity.HIGH: DLPAction.QUARANTINE,
            DLPSeverity.MEDIUM: DLPAction.ALERT,
            DLPSeverity.LOW: DLPAction.LOG,
            DLPSeverity.INFO: DLPAction.LOG,
        }
        self._escalation_chain: List[str] = []
        self._notification_callbacks: List[Callable[[DLPIncident], None]] = []

    def create_incident(self, violations: List[DLPViolation]) -> DLPIncident:
        incident = DLPIncident(incident_id=uuid.uuid4().hex[:12])
        for violation in violations:
            incident.add_violation(violation)
        self._incidents[incident.incident_id] = incident
        self._auto_respond(incident)
        return incident

    def register_response_action(
        self, name: str, action: Callable[[DLPIncident], None]
    ) -> None:
        self._response_actions[name] = action

    def add_notification_callback(
        self, callback: Callable[[DLPIncident], None]
    ) -> None:
        self._notification_callbacks.append(callback)

    def resolve_incident(
        self, incident_id: str, resolution: str, resolved_by: str = ""
    ) -> bool:
        incident = self._incidents.get(incident_id)
        if incident is None:
            return False
        incident.status = "resolved"
        incident.resolution = resolution
        incident.assigned_to = resolved_by
        incident.resolved_at = time.time()
        return True

    def escalate_incident(self, incident_id: str) -> bool:
        incident = self._incidents.get(incident_id)
        if incident is None:
            return False
        incident.status = "escalated"
        return True

    def _auto_respond(self, incident: DLPIncident) -> None:
        action = self._auto_response_rules.get(incident.severity, DLPAction.LOG)
        for callback in self._notification_callbacks:
            try:
                callback(incident)
            except Exception:
                pass
        if incident.severity in (DLPSeverity.CRITICAL, DLPSeverity.HIGH):
            incident.status = "escalated"
        else:
            incident.status = "investigating"

    def get_incident(self, incident_id: str) -> Optional[DLPIncident]:
        return self._incidents.get(incident_id)

    def get_open_incidents(self) -> List[DLPIncident]:
        return [
            inc for inc in self._incidents.values()
            if inc.status in ("open", "investigating", "escalated")
        ]

    def get_incident_summary(self) -> Dict[str, Any]:
        status_counts: Dict[str, int] = defaultdict(int)
        severity_counts: Dict[str, int] = defaultdict(int)
        for incident in self._incidents.values():
            status_counts[incident.status] += 1
            severity_counts[incident.severity.value] += 1
        return {
            "total_incidents": len(self._incidents),
            "open_incidents": len(self.get_open_incidents()),
            "by_status": dict(status_counts),
            "by_severity": dict(severity_counts),
        }


class DLPSession:
    """Represents a DLP monitoring session."""

    def __init__(self, session_id: str, user_id: str = "") -> None:
        self.session_id: str = session_id
        self.user_id: str = user_id
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.events: List[DataFlowEvent] = []
        self.violations: List[DLPViolation] = []
        self.incidents: List[DLPIncident] = []

    def add_event(self, event: DataFlowEvent) -> None:
        event.session_id = self.session_id
        self.events.append(event)

    def add_violation(self, violation: DLPViolation) -> None:
        self.violations.append(violation)

    def add_incident(self, incident: DLPIncident) -> None:
        self.incidents.append(incident)

    def close(self) -> Dict[str, Any]:
        self.end_time = time.time()
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "duration_seconds": self.end_time - self.start_time,
            "total_events": len(self.events),
            "total_violations": len(self.violations),
            "total_incidents": len(self.incidents),
        }


class DLPEngine:
    """Main DLP engine orchestrating all components."""

    def __init__(
        self,
        enable_flow_monitor: bool = True,
        enable_egress_inspector: bool = True,
        enable_memory_scanner: bool = True,
        enable_file_monitor: bool = True,
    ) -> None:
        self.flow_monitor: Optional[DataFlowMonitor] = (
            DataFlowMonitor() if enable_flow_monitor else None
        )
        self.egress_inspector: Optional[NetworkEgressInspector] = (
            NetworkEgressInspector() if enable_egress_inspector else None
        )
        self.memory_scanner: Optional[MemoryScanner] = (
            MemoryScanner() if enable_memory_scanner else None
        )
        self.file_monitor: Optional[FileIOMonitor] = (
            FileIOMonitor() if enable_file_monitor else None
        )
        self.policy_enforcer: PolicyEnforcer = PolicyEnforcer()
        self.incident_responder: IncidentResponder = IncidentResponder()
        self._sessions: Dict[str, DLPSession] = {}
        self._global_violations: List[DLPViolation] = []
        self._max_violations: int = 50000

    def create_session(self, session_id: str, user_id: str = "") -> DLPSession:
        session = DLPSession(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        return session

    def close_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self._sessions.get(session_id)
        if session:
            return session.close()
        return None

    def monitor_data_flow(self, event: DataFlowEvent, content: str = "") -> List[DLPViolation]:
        if self.flow_monitor:
            self.flow_monitor.register_event(event)
        violations: List[DLPViolation] = []
        violations.extend(self.policy_enforcer.enforce(event, content))
        if self.memory_scanner and content:
            violations.extend(self.memory_scanner.scan_string(content, event.source))
        for v in violations:
            self._global_violations.append(v)
            session = self._sessions.get(event.session_id)
            if session:
                session.add_violation(v)
        if len(self._global_violations) > self._max_violations:
            self._global_violations = self._global_violations[-self._max_violations:]
        if violations:
            incident = self.incident_responder.create_incident(violations)
            session = self._sessions.get(event.session_id)
            if session:
                session.add_incident(incident)
        return violations

    def inspect_egress(
        self, destination: str, data: str, session_id: str = ""
    ) -> List[DLPViolation]:
        if self.egress_inspector is None:
            return []
        violations = self.egress_inspector.inspect(destination, data)
        for v in violations:
            self._global_violations.append(v)
            session = self._sessions.get(session_id)
            if session:
                session.add_violation(v)
        return violations

    def scan_memory(self, data: str, source: str = "memory") -> List[DLPViolation]:
        if self.memory_scanner is None:
            return []
        return self.memory_scanner.scan_string(data, source)

    def check_file_access(
        self, file_path: str, operation: str, user_id: str = "", session_id: str = ""
    ) -> List[DLPViolation]:
        if self.file_monitor is None:
            return []
        if operation == "read":
            violations = self.file_monitor.check_read(file_path, user_id)
        elif operation == "write":
            violations = self.file_monitor.check_write(file_path, 0, user_id)
        else:
            violations = []
        for v in violations:
            self._global_violations.append(v)
            session = self._sessions.get(session_id)
            if session:
                session.add_violation(v)
        return violations

    def add_rule(self, rule: DLPRule) -> None:
        self.policy_enforcer.add_rule(rule)

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total_violations": len(self._global_violations),
            "active_sessions": len(self._sessions),
        }
        severity_counts: Dict[str, int] = defaultdict(int)
        for v in self._global_violations:
            severity_counts[v.severity.value] += 1
        stats["violations_by_severity"] = dict(severity_counts)
        stats["incident_summary"] = self.incident_responder.get_incident_summary()
        if self.flow_monitor:
            stats["flow_summary"] = self.flow_monitor.get_flow_summary()
        return stats
