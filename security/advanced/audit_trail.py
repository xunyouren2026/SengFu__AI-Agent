"""
Audit Trail Module
==================

Provides an immutable audit trail backed by a Merkle tree for tamper
detection, event querying, and compliance reporting (GDPR / SOC2).

Only uses the Python standard library.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Severity level for audit events."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionCategory(Enum):
    """Category of audited actions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"
    CONFIG_CHANGE = "config_change"
    DATA_EXPORT = "data_export"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """A single audit event record."""

    timestamp: float
    """Unix timestamp of the event."""

    actor: str
    """Identifier of the entity that performed the action."""

    action: str
    """The action performed (e.g. 'read', 'write', 'delete')."""

    target: str
    """The resource or object that was acted upon."""

    result: str
    """Result of the action ('success', 'failure', 'denied')."""

    hash: str = ""
    """SHA-256 hash of this event (computed on creation)."""

    previous_hash: str = ""
    """Hash of the previous event for chain integrity."""

    severity: Severity = Severity.INFO
    """Severity level."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata attached to the event."""

    event_id: str = ""
    """Unique identifier for this event."""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = hashlib.sha256(
                f"{self.timestamp}:{self.actor}:{self.action}:{self.target}".encode()
            ).hexdigest()[:16]
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute the SHA-256 hash of this event."""
        data = json.dumps({
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "result": self.result,
            "previous_hash": self.previous_hash,
            "severity": self.severity.value,
            "metadata": self.metadata,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "result": self.result,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "severity": self.severity.value,
            "metadata": self.metadata,
        }


@dataclass
class AuditQuery:
    """Query filter for searching audit events."""

    time_range: Optional[Tuple[float, float]] = None
    """(start_timestamp, end_timestamp) inclusive."""

    actor: Optional[str] = None
    """Filter by actor identifier (exact match)."""

    action_type: Optional[str] = None
    """Filter by action type (exact match)."""

    severity: Optional[Severity] = None
    """Filter by minimum severity level."""

    target_pattern: Optional[str] = None
    """Filter by target substring match."""

    result: Optional[str] = None
    """Filter by result ('success', 'failure', 'denied')."""

    limit: int = 100
    """Maximum number of events to return."""

    offset: int = 0
    """Number of events to skip."""

    def matches(self, event: AuditEvent) -> bool:
        """Return True if *event* matches this query."""
        if self.time_range is not None:
            start, end = self.time_range
            if not (start <= event.timestamp <= end):
                return False
        if self.actor is not None and event.actor != self.actor:
            return False
        if self.action_type is not None and event.action != self.action_type:
            return False
        if self.severity is not None:
            severity_order = [
                Severity.INFO, Severity.LOW, Severity.MEDIUM,
                Severity.HIGH, Severity.CRITICAL,
            ]
            event_idx = severity_order.index(event.severity)
            query_idx = severity_order.index(self.severity)
            if event_idx < query_idx:
                return False
        if self.target_pattern is not None:
            if self.target_pattern not in event.target:
                return False
        if self.result is not None and event.result != self.result:
            return False
        return True


# ---------------------------------------------------------------------------
# Merkle Tree
# ---------------------------------------------------------------------------

class MerkleTree:
    """Merkle tree for tamper-proof audit log integrity verification.

    Each leaf is the hash of an audit event.  The root hash changes if any
    leaf is modified, enabling efficient integrity checks.
    """

    def __init__(self):
        self._leaves: List[str] = []
        self._layers: List[List[str]] = []

    def add_leaf(self, data: str) -> int:
        """Add a leaf (hash of *data*) and return its index."""
        leaf_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
        self._leaves.append(leaf_hash)
        self._rebuild()
        return len(self._leaves) - 1

    def _rebuild(self) -> None:
        """Rebuild the tree layers from the current leaves."""
        if not self._leaves:
            self._layers = []
            return
        self._layers = [list(self._leaves)]
        current = list(self._leaves)
        while len(current) > 1:
            next_layer: List[str] = []
            for i in range(0, len(current) - 1, 2):
                combined = current[i] + current[i + 1]
                next_layer.append(hashlib.sha256(combined.encode()).hexdigest())
            if len(current) % 2 == 1:
                # Duplicate the last node
                next_layer.append(current[-1])
            self._layers.append(next_layer)
            current = next_layer

    def get_root(self) -> str:
        """Return the Merkle root hash, or empty string if tree is empty."""
        if not self._layers:
            return ""
        return self._layers[-1][0]

    def get_proof(self, index: int) -> List[Tuple[str, bool]]:
        """Return the Merkle proof path for the leaf at *index*.

        Each element is (sibling_hash, is_right) where is_right indicates
        whether the sibling is on the right side.
        """
        if index < 0 or index >= len(self._leaves):
            raise IndexError(f"Index {index} out of range [0, {len(self._leaves)})")

        proof: List[Tuple[str, bool]] = []
        idx = index
        for layer in self._layers[:-1]:
            is_right = idx % 2 == 1
            sibling_idx = idx - 1 if is_right else idx + 1
            if sibling_idx < len(layer):
                proof.append((layer[sibling_idx], is_right))
            else:
                # Duplicate case: sibling is itself
                proof.append((layer[idx], not is_right))
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(leaf: str, proof: List[Tuple[str, bool]], root: str) -> bool:
        """Verify that *leaf* with *proof* produces *root*."""
        current = leaf
        for sibling, is_right in proof:
            if is_right:
                combined = sibling + current
            else:
                combined = current + sibling
            current = hashlib.sha256(combined.encode()).hexdigest()
        return current == root

    def num_leaves(self) -> int:
        """Return the number of leaves in the tree."""
        return len(self._leaves)

    def get_leaves(self) -> List[str]:
        """Return a copy of the leaf hashes."""
        return list(self._leaves)


# ---------------------------------------------------------------------------
# Tamper Detector
# ---------------------------------------------------------------------------

class TamperDetector:
    """Detects tampering in an audit trail by verifying the hash chain
    and Merkle tree integrity."""

    def __init__(self, events: List[AuditEvent], merkle: MerkleTree):
        self._events = events
        self._merkle = merkle

    def detect_tampering(self) -> List[Dict[str, Any]]:
        """Check all events for signs of tampering.

        Returns a list of anomaly reports.
        """
        anomalies: List[Dict[str, Any]] = []

        # 1. Hash chain verification
        for i, event in enumerate(self._events):
            expected_hash = event._compute_hash()
            if event.hash != expected_hash:
                anomalies.append({
                    "type": "hash_mismatch",
                    "event_id": event.event_id,
                    "index": i,
                    "description": f"Event hash does not match computed hash",
                    "severity": Severity.CRITICAL.value,
                })

        # 2. Chain linkage verification
        for i in range(1, len(self._events)):
            if self._events[i].previous_hash != self._events[i - 1].hash:
                anomalies.append({
                    "type": "chain_break",
                    "event_id": self._events[i].event_id,
                    "index": i,
                    "description": "Previous hash reference is broken",
                    "severity": Severity.CRITICAL.value,
                })

        # 3. Merkle tree verification
        leaves = self._merkle.get_leaves()
        for i, event in enumerate(self._events):
            if i < len(leaves):
                expected_leaf = hashlib.sha256(
                    event.hash.encode("utf-8")
                ).hexdigest()
                if leaves[i] != expected_leaf:
                    anomalies.append({
                        "type": "merkle_mismatch",
                        "event_id": event.event_id,
                        "index": i,
                        "description": "Merkle leaf does not match event hash",
                        "severity": Severity.HIGH.value,
                    })

        # 4. Timestamp ordering check
        for i in range(1, len(self._events)):
            if self._events[i].timestamp < self._events[i - 1].timestamp:
                anomalies.append({
                    "type": "timestamp_violation",
                    "event_id": self._events[i].event_id,
                    "index": i,
                    "description": "Event timestamp is before previous event",
                    "severity": Severity.MEDIUM.value,
                })

        return anomalies

    def get_tamper_report(self) -> Dict[str, Any]:
        """Return a comprehensive tamper detection report."""
        anomalies = self.detect_tampering()
        return {
            "is_tampered": len(anomalies) > 0,
            "total_events": len(self._events),
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
            "merkle_root": self._merkle.get_root(),
            "checked_at": time.time(),
        }


# ---------------------------------------------------------------------------
# Compliance Report
# ---------------------------------------------------------------------------

class ComplianceReport:
    """Generate compliance reports for GDPR and SOC2 frameworks."""

    def __init__(self, events: List[AuditEvent]):
        self._events = events

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compliance report."""
        return {
            "summary": self._summary(),
            "gdpr": self.check_gdpr(),
            "soc2": self.check_soc2(),
            "data_retention": self._check_data_retention(),
            "access_control": self._check_access_control(),
            "generated_at": time.time(),
        }

    def _summary(self) -> Dict[str, Any]:
        """Generate a summary of the audit events."""
        if not self._events:
            return {"total_events": 0}
        actors = set(e.actor for e in self._events)
        actions = {}
        for e in self._events:
            actions[e.action] = actions.get(e.action, 0) + 1
        results = {}
        for e in self._events:
            results[e.result] = results.get(e.result, 0) + 1
        severities = {}
        for e in self._events:
            sev = e.severity.value
            severities[sev] = severities.get(sev, 0) + 1
        return {
            "total_events": len(self._events),
            "unique_actors": len(actors),
            "action_counts": actions,
            "result_counts": results,
            "severity_counts": severities,
            "time_range": (
                self._events[0].timestamp,
                self._events[-1].timestamp,
            ),
        }

    def check_gdpr(self) -> Dict[str, Any]:
        """Check GDPR compliance.

        Verifies:
        - Data access events are logged
        - Data deletion requests are recorded
        - Consent-related actions are tracked
        - Data export events are logged
        """
        checks: Dict[str, Any] = {
            "compliant": True,
            "checks": [],
        }

        # Check for data access logging
        access_events = [e for e in self._events if e.action in ("read", "DATA_EXPORT")]
        checks["checks"].append({
            "requirement": "Data access logging",
            "status": "pass" if access_events else "warning",
            "detail": f"{len(access_events)} access events recorded",
        })

        # Check for deletion request logging
        deletion_events = [e for e in self._events if e.action == "delete"]
        checks["checks"].append({
            "requirement": "Deletion request tracking",
            "status": "pass" if deletion_events or not self._events else "warning",
            "detail": f"{len(deletion_events)} deletion events recorded",
        })

        # Check for consent tracking
        consent_events = [e for e in self._events if "consent" in e.target.lower() or "consent" in e.action.lower()]
        checks["checks"].append({
            "requirement": "Consent management",
            "status": "pass" if consent_events or not self._events else "fail",
            "detail": f"{len(consent_events)} consent-related events",
        })

        # Check for data export logging
        export_events = [e for e in self._events if e.action == "DATA_EXPORT"]
        checks["checks"].append({
            "requirement": "Data portability (export)",
            "status": "pass" if export_events or not self._events else "warning",
            "detail": f"{len(export_events)} export events recorded",
        })

        # Determine overall compliance
        statuses = [c["status"] for c in checks["checks"]]
        if "fail" in statuses:
            checks["compliant"] = False
        elif "warning" in statuses:
            checks["compliant"] = None  # Partial compliance

        return checks

    def check_soc2(self) -> Dict[str, Any]:
        """Check SOC2 compliance.

        Verifies:
        - Authentication events are logged
        - Authorization checks are recorded
        - Configuration changes are tracked
        - Security events are monitored
        """
        checks: Dict[str, Any] = {
            "compliant": True,
            "checks": [],
        }

        # Authentication logging
        auth_events = [e for e in self._events if e.action in ("AUTHENTICATE", "authenticate", "login")]
        checks["checks"].append({
            "requirement": "Authentication logging",
            "status": "pass" if auth_events else "warning",
            "detail": f"{len(auth_events)} authentication events",
        })

        # Authorization logging
        authz_events = [e for e in self._events if e.action in ("AUTHORIZE", "authorize")]
        checks["checks"].append({
            "requirement": "Authorization logging",
            "status": "pass" if authz_events else "warning",
            "detail": f"{len(authz_events)} authorization events",
        })

        # Configuration change tracking
        config_events = [e for e in self._events if e.action in ("CONFIG_CHANGE", "config_change")]
        checks["checks"].append({
            "requirement": "Configuration change tracking",
            "status": "pass" if config_events or not self._events else "warning",
            "detail": f"{len(config_events)} configuration change events",
        })

        # Failed access attempts
        denied_events = [e for e in self._events if e.result == "denied"]
        checks["checks"].append({
            "requirement": "Failed access monitoring",
            "status": "pass",
            "detail": f"{len(denied_events)} denied access events recorded",
        })

        # Security event monitoring
        security_events = [e for e in self._events if e.severity in (Severity.HIGH, Severity.CRITICAL)]
        checks["checks"].append({
            "requirement": "Security event monitoring",
            "status": "pass",
            "detail": f"{len(security_events)} high/critical severity events",
        })

        statuses = [c["status"] for c in checks["checks"]]
        if "fail" in statuses:
            checks["compliant"] = False
        elif "warning" in statuses:
            checks["compliant"] = None

        return checks

    def _check_data_retention(self) -> Dict[str, Any]:
        """Check data retention policy compliance."""
        now = time.time()
        thirty_days = 30 * 24 * 3600
        ninety_days = 90 * 24 * 3600
        one_year = 365 * 24 * 3600

        recent = sum(1 for e in self._events if now - e.timestamp < thirty_days)
        medium = sum(1 for e in self._events if thirty_days <= now - e.timestamp < ninety_days)
        old = sum(1 for e in self._events if now - e.timestamp >= ninety_days)

        return {
            "total_events": len(self._events),
            "last_30_days": recent,
            "30_to_90_days": medium,
            "older_than_90_days": old,
            "recommendation": (
                "Consider archiving events older than 90 days"
                if old > 0 else "Retention policy appears compliant"
            ),
        }

    def _check_access_control(self) -> Dict[str, Any]:
        """Check access control patterns."""
        actor_actions: Dict[str, Dict[str, int]] = {}
        for e in self._events:
            if e.actor not in actor_actions:
                actor_actions[e.actor] = {}
            actor_actions[e.actor][e.action] = (
                actor_actions[e.actor].get(e.action, 0) + 1
            )

        high_privilege_actions = {"delete", "CONFIG_CHANGE", "config_change", "DATA_EXPORT"}
        high_privilege_users = []
        for actor, actions in actor_actions.items():
            for a in actions:
                if a in high_privilege_actions:
                    high_privilege_users.append(actor)
                    break

        denied_by_actor: Dict[str, int] = {}
        for e in self._events:
            if e.result == "denied":
                denied_by_actor[e.actor] = denied_by_actor.get(e.actor, 0) + 1

        return {
            "total_actors": len(actor_actions),
            "high_privilege_users": high_privilege_users,
            "denied_access_by_actor": denied_by_actor,
            "recommendation": (
                "Review users with frequent denied access attempts"
                if denied_by_actor else "Access patterns appear normal"
            ),
        }

    def check_compliance(self) -> bool:
        """Return True if both GDPR and SOC2 checks pass."""
        report = self.generate_report()
        gdpr_ok = report["gdpr"]["compliant"] is True
        soc2_ok = report["soc2"]["compliant"] is True
        return gdpr_ok and soc2_ok


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------

class AuditTrail:
    """Immutable, append-only audit trail with Merkle tree integrity."""

    def __init__(self):
        self._events: List[AuditEvent] = []
        self._merkle = MerkleTree()

    def record_event(
        self,
        actor: str,
        action: str,
        target: str,
        result: str = "success",
        severity: Severity = Severity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> AuditEvent:
        """Record a new audit event and return it."""
        ts = timestamp or time.time()
        previous_hash = self._events[-1].hash if self._events else "0" * 64

        event = AuditEvent(
            timestamp=ts,
            actor=actor,
            action=action,
            target=target,
            result=result,
            previous_hash=previous_hash,
            severity=severity,
            metadata=metadata or {},
        )

        self._events.append(event)
        self._merkle.add_leaf(event.hash)
        return event

    def get_events(self, query: Optional[AuditQuery] = None) -> List[AuditEvent]:
        """Query events, optionally filtered by *query*."""
        if query is None:
            return list(self._events)

        matched = [e for e in self._events if query.matches(e)]
        return matched[query.offset: query.offset + query.limit]

    def verify_integrity(self) -> bool:
        """Verify the integrity of the entire audit trail.

        Checks the hash chain and Merkle tree root.
        """
        detector = TamperDetector(self._events, self._merkle)
        report = detector.get_tamper_report()
        return not report["is_tampered"]

    def get_tamper_detector(self) -> TamperDetector:
        """Return a TamperDetector for detailed analysis."""
        return TamperDetector(self._events, self._merkle)

    def get_compliance_report(self) -> ComplianceReport:
        """Return a ComplianceReport for this audit trail."""
        return ComplianceReport(self._events)

    def get_merkle_root(self) -> str:
        """Return the current Merkle root hash."""
        return self._merkle.get_root()

    def get_merkle_proof(self, index: int) -> List[Tuple[str, bool]]:
        """Return the Merkle proof for the event at *index*."""
        return self._merkle.get_proof(index)

    def verify_merkle_proof(
        self, index: int, proof: List[Tuple[str, bool]]
    ) -> bool:
        """Verify the Merkle proof for the event at *index*."""
        leaf_hash = hashlib.sha256(
            self._events[index].hash.encode("utf-8")
        ).hexdigest()
        return self._merkle.verify_proof(
            leaf_hash, proof, self._merkle.get_root()
        )

    def num_events(self) -> int:
        """Return the total number of recorded events."""
        return len(self._events)

    def export_events(self) -> List[Dict[str, Any]]:
        """Export all events as a list of dicts."""
        return [e.to_dict() for e in self._events]
