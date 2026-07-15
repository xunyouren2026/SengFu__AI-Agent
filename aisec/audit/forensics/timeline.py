"""
Forensic Timeline Reconstruction Module
=========================================
Event ordering, timeline gap detection, correlation analysis,
evidence chain management, and timeline visualization.

Pure Python standard library implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from bisect import bisect_left, insort
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EventSeverity(Enum):
    """Severity levels for timeline events."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventCategory(Enum):
    """Categories of forensic events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    PROCESS = "process"
    MEMORY = "memory"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    DATA_ACCESS = "data_access"
    EXFILTRATION = "exfiltration"
    INJECTION = "injection"
    PRIVILEGE_ESC = "privilege_escalation"
    LATERAL_MOVE = "lateral_movement"
    PERSISTENCE = "persistence"
    DEFENSE_EVASION = "defense_evasion"
    UNKNOWN = "unknown"


@dataclass
class TimelineEvent:
    """A single event in the forensic timeline."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    category: EventCategory = EventCategory.UNKNOWN
    severity: EventSeverity = EventSeverity.INFO
    actor: str = ""
    target: str = ""
    source_ip: str = ""
    destination_ip: str = ""
    description: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    correlated_ids: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    sequence_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TimelineEvent):
            return NotImplemented
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.sequence_number < other.sequence_number

    def __hash__(self) -> int:
        return hash(self.event_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimelineEvent):
            return NotImplemented
        return self.event_id == other.event_id

    def compute_hash(self) -> str:
        """Compute a content hash for integrity verification."""
        content = json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor": self.actor,
            "target": self.target,
            "description": self.description,
            "raw_data_keys": sorted(self.raw_data.keys()),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "category": self.category.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "target": self.target,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "description": self.description,
            "raw_data": self.raw_data,
            "evidence_ids": self.evidence_ids,
            "correlated_ids": self.correlated_ids,
            "tags": sorted(self.tags),
            "confidence": self.confidence,
            "sequence_number": self.sequence_number,
            "metadata": self.metadata,
            "hash": self.compute_hash(),
        }


@dataclass
class TimelineGap:
    """A detected gap in the timeline."""
    gap_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    expected_event_types: List[str] = field(default_factory=list)
    severity: EventSeverity = EventSeverity.MEDIUM
    description: str = ""
    surrounding_events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "expected_event_types": self.expected_event_types,
            "severity": self.severity.value,
            "description": self.description,
            "surrounding_events": self.surrounding_events,
        }


@dataclass
class CorrelationResult:
    """Result of correlating two or more events."""
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_ids: List[str] = field(default_factory=list)
    correlation_type: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "event_ids": self.event_ids,
            "correlation_type": self.correlation_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "description": self.description,
        }


@dataclass
class EvidenceChainLink:
    """A single link in the evidence chain."""
    link_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_id: str = ""
    evidence_type: str = ""
    evidence_data: Dict[str, Any] = field(default_factory=dict)
    hash_value: str = ""
    prev_hash: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    custodian: str = ""
    description: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "link_id": self.link_id,
            "event_id": self.event_id,
            "evidence_type": self.evidence_type,
            "evidence_data_keys": sorted(self.evidence_data.keys()),
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp.isoformat(),
        }, sort_keys=True)
        self.hash_value = hashlib.sha256(content.encode()).hexdigest()
        return self.hash_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "event_id": self.event_id,
            "evidence_type": self.evidence_type,
            "evidence_data": self.evidence_data,
            "hash_value": self.hash_value,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp.isoformat(),
            "custodian": self.custodian,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Event Orderer
# ---------------------------------------------------------------------------

class EventOrderer:
    """Orders events chronologically with conflict resolution."""

    TIMESTAMP_TOLERANCE_MS: float = 1.0  # 1ms tolerance for equal timestamps

    def __init__(self) -> None:
        self._events: List[TimelineEvent] = []
        self._sequence_counter: int = 0

    def add_event(self, event: TimelineEvent) -> None:
        """Add an event maintaining sorted order."""
        event.sequence_number = self._sequence_counter
        self._sequence_counter += 1
        insort(self._events, event)

    def add_events(self, events: List[TimelineEvent]) -> None:
        """Add multiple events in sorted order."""
        for event in events:
            self.add_event(event)

    def get_ordered_events(self) -> List[TimelineEvent]:
        """Return events in chronological order."""
        return list(self._events)

    def detect_timestamp_conflicts(self) -> List[List[TimelineEvent]]:
        """Detect events with identical or near-identical timestamps."""
        conflicts: List[List[TimelineEvent]] = []
        i = 0
        while i < len(self._events):
            group: List[TimelineEvent] = [self._events[i]]
            j = i + 1
            while j < len(self._events):
                delta = abs(
                    (self._events[j].timestamp - self._events[i].timestamp).total_seconds()
                )
                if delta <= self.TIMESTAMP_TOLERANCE_MS / 1000.0:
                    group.append(self._events[j])
                    j += 1
                else:
                    break
            if len(group) > 1:
                conflicts.append(group)
            i = j
        return conflicts

    def resolve_conflicts(
        self, conflicts: List[List[TimelineEvent]]
    ) -> List[TimelineEvent]:
        """Resolve timestamp conflicts using heuristic ordering.

        Resolution priority:
        1. Events with causal dependencies (referenced by other events)
        2. Higher severity events first
        3. Authentication events before action events
        4. Original sequence number as tiebreaker
        """
        category_priority = {
            EventCategory.AUTHENTICATION: 0,
            EventCategory.AUTHORIZATION: 1,
            EventCategory.NETWORK: 2,
            EventCategory.FILE_SYSTEM: 3,
            EventCategory.PROCESS: 4,
            EventCategory.DATA_ACCESS: 5,
            EventCategory.EXFILTRATION: 6,
            EventCategory.INJECTION: 7,
            EventCategory.PRIVILEGE_ESC: 8,
            EventCategory.LATERAL_MOVE: 9,
            EventCategory.PERSISTENCE: 10,
            EventCategory.DEFENSE_EVASION: 11,
            EventCategory.UNKNOWN: 12,
        }

        severity_priority = {
            EventSeverity.CRITICAL: 0,
            EventSeverity.HIGH: 1,
            EventSeverity.MEDIUM: 2,
            EventSeverity.LOW: 3,
            EventSeverity.INFO: 4,
        }

        resolved: List[TimelineEvent] = []
        for group in conflicts:
            sorted_group = sorted(
                group,
                key=lambda e: (
                    category_priority.get(e.category, 99),
                    severity_priority.get(e.severity, 99),
                    e.sequence_number,
                ),
            )
            resolved.extend(sorted_group)

        # Rebuild the full ordered list
        conflict_ids = {e.event_id for group in conflicts for e in group}
        result: List[TimelineEvent] = []
        resolved_iter = iter(resolved)
        for event in self._events:
            if event.event_id in conflict_ids:
                continue
            result.append(event)
            # Insert resolved events at the appropriate position
            while True:
                peek = next(resolved_iter, None)
                if peek is None:
                    break
                if peek.timestamp < event.timestamp:
                    result.append(peek)
                else:
                    # Put it back
                    resolved_list = [peek] + list(resolved_iter)
                    resolved_iter = iter(resolved_list)
                    break
        # Append remaining resolved events
        for remaining in resolved_iter:
            result.append(remaining)

        return result

    def get_events_in_range(
        self, start: datetime, end: datetime
    ) -> List[TimelineEvent]:
        """Get events within a time range."""
        start_idx = bisect_left(self._events, TimelineEvent(timestamp=start))
        result: List[TimelineEvent] = []
        for event in self._events[start_idx:]:
            if event.timestamp > end:
                break
            result.append(event)
        return result

    def get_event_by_id(self, event_id: str) -> Optional[TimelineEvent]:
        """Find an event by its ID."""
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None


# ---------------------------------------------------------------------------
# Gap Detector
# ---------------------------------------------------------------------------

class GapDetector:
    """Detects gaps in the forensic timeline."""

    # Expected minimum event intervals by category (seconds)
    EXPECTED_INTERVALS: Dict[EventCategory, float] = {
        EventCategory.AUTHENTICATION: 0.1,
        EventCategory.NETWORK: 0.05,
        EventCategory.FILE_SYSTEM: 0.01,
        EventCategory.PROCESS: 0.01,
        EventCategory.DATA_ACCESS: 0.1,
        EventCategory.EXFILTRATION: 1.0,
    }

    # Maximum expected gap durations by category (seconds)
    MAX_GAP_THRESHOLDS: Dict[EventCategory, float] = {
        EventCategory.AUTHENTICATION: 3600.0,
        EventCategory.NETWORK: 300.0,
        EventCategory.FILE_SYSTEM: 600.0,
        EventCategory.PROCESS: 60.0,
        EventCategory.DATA_ACCESS: 1800.0,
        EventCategory.EXFILTRATION: 600.0,
    }

    def __init__(
        self,
        default_gap_threshold: float = 3600.0,
        min_events_for_analysis: int = 5,
    ) -> None:
        self._default_threshold = default_gap_threshold
        self._min_events = min_events_for_analysis

    def detect_gaps(
        self, events: List[TimelineEvent]
    ) -> List[TimelineGap]:
        """Detect gaps in the event timeline."""
        if len(events) < self._min_events:
            return []

        gaps: List[TimelineGap] = []
        sorted_events = sorted(events)

        # Compute inter-event intervals
        intervals: List[float] = []
        for i in range(1, len(sorted_events)):
            delta = (sorted_events[i].timestamp - sorted_events[i - 1].timestamp).total_seconds()
            intervals.append(delta)

        if not intervals:
            return gaps

        # Compute statistics
        mean_interval = sum(intervals) / len(intervals)
        std_interval = self._std_dev(intervals) if len(intervals) > 1 else 0.0
        threshold = max(mean_interval + 3.0 * std_interval, self._default_threshold)

        # Detect gaps exceeding threshold
        for i, interval in enumerate(intervals):
            if interval > threshold:
                event_before = sorted_events[i]
                event_after = sorted_events[i + 1]

                # Determine expected event types based on context
                expected_types = self._infer_expected_types(
                    event_before, event_after
                )

                severity = self._assess_gap_severity(interval, threshold)

                gap = TimelineGap(
                    start_time=event_before.timestamp,
                    end_time=event_after.timestamp,
                    duration_seconds=interval,
                    expected_event_types=expected_types,
                    severity=severity,
                    description=(
                        f"Gap of {interval:.1f}s detected between "
                        f"'{event_before.event_type}' and '{event_after.event_type}'"
                    ),
                    surrounding_events=[
                        event_before.event_id,
                        event_after.event_id,
                    ],
                )
                gaps.append(gap)

        # Check for missing event type sequences
        sequence_gaps = self._detect_sequence_gaps(sorted_events)
        gaps.extend(sequence_gaps)

        return sorted(gaps, key=lambda g: g.duration_seconds, reverse=True)

    def _detect_sequence_gaps(
        self, events: List[TimelineEvent]
    ) -> List[TimelineGap]:
        """Detect missing expected event sequences."""
        gaps: List[TimelineGap] = []

        # Expected event sequences (attack kill chain)
        expected_sequences: List[List[Tuple[str, EventCategory]]] = [
            [("login_failed", EventCategory.AUTHENTICATION),
             ("login_success", EventCategory.AUTHENTICATION)],
            [("login_success", EventCategory.AUTHENTICATION),
             ("command_exec", EventCategory.PROCESS)],
            [("privilege_escalation", EventCategory.PRIVILEGE_ESC),
             ("lateral_movement", EventCategory.LATERAL_MOVE)],
            [("data_access", EventCategory.DATA_ACCESS),
             ("data_exfil", EventCategory.EXFILTRATION)],
        ]

        event_type_list = [(e.event_type, e.category) for e in events]

        for seq in expected_sequences:
            for i in range(len(event_type_list) - 1):
                if event_type_list[i] == seq[0]:
                    # Check if the next expected event appears within a window
                    found = False
                    for j in range(i + 1, min(i + 20, len(event_type_list))):
                        if event_type_list[j] == seq[1]:
                            found = True
                            break
                    if not found:
                        gap = TimelineGap(
                            start_time=events[i].timestamp,
                            end_time=events[min(i + 1, len(events) - 1)].timestamp,
                            duration_seconds=(
                                events[min(i + 1, len(events) - 1)].timestamp
                                - events[i].timestamp
                            ).total_seconds(),
                            expected_event_types=[s[0] for s in seq],
                            severity=EventSeverity.MEDIUM,
                            description=(
                                f"Missing expected sequence: "
                                f"'{seq[0][0]}' -> '{seq[1][0]}'"
                            ),
                            surrounding_events=[events[i].event_id],
                        )
                        gaps.append(gap)

        return gaps

    def _infer_expected_types(
        self, before: TimelineEvent, after: TimelineEvent
    ) -> List[str]:
        """Infer what event types might be expected in a gap."""
        expected: List[str] = []
        # If authentication followed by data access, expect authorization
        if (before.category == EventCategory.AUTHENTICATION
                and after.category == EventCategory.DATA_ACCESS):
            expected.extend(["authorization_check", "session_validation"])
        # If lateral movement, expect authentication on new host
        if before.category == EventCategory.LATERAL_MOVE:
            expected.extend(["login_attempt", "credential_use"])
        # If privilege escalation, expect action
        if before.category == EventCategory.PRIVILEGE_ESC:
            expected.extend(["command_exec", "file_access", "config_change"])
        return expected

    def _assess_gap_severity(
        self, gap_duration: float, threshold: float
    ) -> EventSeverity:
        """Assess the severity of a gap based on its duration."""
        ratio = gap_duration / threshold
        if ratio > 10.0:
            return EventSeverity.CRITICAL
        elif ratio > 5.0:
            return EventSeverity.HIGH
        elif ratio > 2.0:
            return EventSeverity.MEDIUM
        return EventSeverity.LOW

    @staticmethod
    def _std_dev(values: List[float]) -> float:
        """Compute standard deviation."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        return variance ** 0.5


# ---------------------------------------------------------------------------
# Correlation Analyzer
# ---------------------------------------------------------------------------

class CorrelationAnalyzer:
    """Correlates events to identify related activities."""

    # Correlation rules: (field_matcher, correlation_type, weight)
    CORRELATION_RULES: List[Tuple[str, str, float]] = [
        ("actor", "same_actor", 0.9),
        ("source_ip", "same_source_ip", 0.7),
        ("destination_ip", "same_dest_ip", 0.6),
        ("target", "same_target", 0.8),
        ("source_ip:destination_ip", "network_session", 0.75),
    ]

    # Temporal correlation windows by category (seconds)
    TEMPORAL_WINDOWS: Dict[EventCategory, float] = {
        EventCategory.AUTHENTICATION: 300.0,
        EventCategory.NETWORK: 60.0,
        EventCategory.FILE_SYSTEM: 30.0,
        EventCategory.PROCESS: 10.0,
        EventCategory.DATA_ACCESS: 600.0,
        EventCategory.EXFILTRATION: 1800.0,
    }

    def __init__(
        self,
        temporal_weight: float = 0.3,
        field_weight: float = 0.5,
        pattern_weight: float = 0.2,
        min_confidence: float = 0.3,
    ) -> None:
        self._temporal_weight = temporal_weight
        self._field_weight = field_weight
        self._pattern_weight = pattern_weight
        self._min_confidence = min_confidence
        self._custom_rules: List[Callable[[TimelineEvent, TimelineEvent], Tuple[float, str]]] = []

    def add_custom_rule(
        self,
        rule: Callable[[TimelineEvent, TimelineEvent], Tuple[float, str]],
    ) -> None:
        """Add a custom correlation rule."""
        self._custom_rules.append(rule)

    def correlate_events(
        self, events: List[TimelineEvent]
    ) -> List[CorrelationResult]:
        """Correlate all event pairs and return significant correlations."""
        results: List[CorrelationResult] = []
        sorted_events = sorted(events)

        for i in range(len(sorted_events)):
            for j in range(i + 1, len(sorted_events)):
                event_a = sorted_events[i]
                event_b = sorted_events[j]

                # Quick temporal filter
                delta = (event_b.timestamp - event_a.timestamp).total_seconds()
                max_window = self._get_max_window(event_a, event_b)
                if delta > max_window:
                    continue

                correlation = self._correlate_pair(event_a, event_b)
                if correlation.confidence >= self._min_confidence:
                    results.append(correlation)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def correlate_event_group(
        self, events: List[TimelineEvent]
    ) -> List[CorrelationResult]:
        """Find groups of correlated events using transitive closure."""
        pair_results = self.correlate_events(events)

        # Build adjacency from correlations
        adj: Dict[str, Set[str]] = defaultdict(set)
        for result in pair_results:
            for i, eid_a in enumerate(result.event_ids):
                for eid_b in result.event_ids[i + 1:]:
                    adj[eid_a].add(eid_b)
                    adj[eid_b].add(eid_a)

        # Find connected components
        visited: Set[str] = set()
        groups: List[List[str]] = []
        for eid in adj:
            if eid not in visited:
                group: List[str] = []
                queue = deque([eid])
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    group.append(current)
                    for neighbor in adj[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                if len(group) > 1:
                    groups.append(group)

        # Create group correlation results
        event_map: Dict[str, TimelineEvent] = {e.event_id: e for e in events}
        group_results: List[CorrelationResult] = []
        for group in groups:
            group_events = [event_map[eid] for eid in group if eid in event_map]
            if len(group_events) < 2:
                continue

            categories = set(e.category for e in group_events)
            actors = set(e.actor for e in group_events if e.actor)
            avg_confidence = sum(
                r.confidence for r in pair_results
                if any(eid in r.event_ids for eid in group)
            ) / max(sum(1 for r in pair_results if any(eid in r.event_ids for eid in group)), 1)

            result = CorrelationResult(
                event_ids=group,
                correlation_type="event_group",
                confidence=min(avg_confidence, 1.0),
                evidence=[
                    f"Categories: {', '.join(c.value for c in categories)}",
                    f"Actors: {', '.join(actors) if actors else 'unknown'}",
                    f"Event count: {len(group_events)}",
                ],
                description=(
                    f"Group of {len(group_events)} correlated events "
                    f"across {len(categories)} categories"
                ),
            )
            group_results.append(result)

        return group_results

    def _correlate_pair(
        self, a: TimelineEvent, b: TimelineEvent
    ) -> CorrelationResult:
        """Correlate a pair of events."""
        temporal_score, temporal_ev = self._temporal_correlation(a, b)
        field_score, field_ev, field_type = self._field_correlation(a, b)
        pattern_score, pattern_ev = self._pattern_correlation(a, b)

        # Custom rules
        custom_score = 0.0
        custom_ev: List[str] = []
        for rule in self._custom_rules:
            s, ev = rule(a, b)
            if s > custom_score:
                custom_score = s
                custom_ev.append(ev)

        total_confidence = (
            temporal_score * self._temporal_weight
            + field_score * self._field_weight
            + pattern_score * self._pattern_weight
            + custom_score * 0.1
        )

        all_evidence = temporal_ev + field_ev + pattern_ev + custom_ev
        correlation_type = field_type if field_type else "temporal"

        return CorrelationResult(
            event_ids=[a.event_id, b.event_id],
            correlation_type=correlation_type,
            confidence=min(total_confidence, 1.0),
            evidence=all_evidence,
            description=f"Correlated {a.event_type} with {b.event_type}",
        )

    def _temporal_correlation(
        self, a: TimelineEvent, b: TimelineEvent
    ) -> Tuple[float, List[str]]:
        """Score temporal proximity."""
        delta = abs((b.timestamp - a.timestamp).total_seconds())
        window = self._get_max_window(a, b)
        if delta <= window:
            score = 1.0 - (delta / window)
            return score, [f"Temporal proximity: {delta:.1f}s (window: {window:.1f}s)"]
        return 0.0, []

    def _field_correlation(
        self, a: TimelineEvent, b: TimelineEvent
    ) -> Tuple[float, List[str], str]:
        """Score field-based correlation."""
        best_score = 0.0
        best_type = ""
        evidence: List[str] = []

        for field_spec, corr_type, weight in self.CORRELATION_RULES:
            if ":" in field_spec:
                fields = field_spec.split(":")
                vals_a = [getattr(a, f, "") for f in fields]
                vals_b = [getattr(b, f, "") for f in fields]
                match = all(
                    va and vb and va == vb
                    for va, vb in zip(vals_a, vals_b)
                )
            else:
                val_a = getattr(a, field_spec, "")
                val_b = getattr(b, field_spec, "")
                match = bool(val_a and val_b and val_a == val_b)

            if match and weight > best_score:
                best_score = weight
                best_type = corr_type
                evidence = [f"Field match: {field_spec} = '{getattr(a, field_spec.split(':')[0], '')}'"]

        return best_score, evidence, best_type

    def _pattern_correlation(
        self, a: TimelineEvent, b: TimelineEvent
    ) -> Tuple[float, List[str]]:
        """Score pattern-based correlation."""
        score = 0.0
        evidence: List[str] = []

        # Attack chain progression
        chain_progression = {
            (EventCategory.AUTHENTICATION, EventCategory.AUTHORIZATION): 0.8,
            (EventCategory.AUTHORIZATION, EventCategory.DATA_ACCESS): 0.7,
            (EventCategory.AUTHENTICATION, EventCategory.PROCESS): 0.6,
            (EventCategory.PRIVILEGE_ESC, EventCategory.LATERAL_MOVE): 0.9,
            (EventCategory.LATERAL_MOVE, EventCategory.DATA_ACCESS): 0.8,
            (EventCategory.DATA_ACCESS, EventCategory.EXFILTRATION): 0.9,
            (EventCategory.PROCESS, EventCategory.PERSISTENCE): 0.7,
            (EventCategory.PROCESS, EventCategory.FILE_SYSTEM): 0.6,
        }

        pair = (a.category, b.category)
        if pair in chain_progression:
            score = chain_progression[pair]
            evidence.append(
                f"Attack chain progression: {a.category.value} -> {b.category.value}"
            )

        # Same actor across different categories
        if a.actor and b.actor and a.actor == b.actor and a.category != b.category:
            score = max(score, 0.5)
            evidence.append(f"Same actor '{a.actor}' across different categories")

        return score, evidence

    def _get_max_window(
        self, a: TimelineEvent, b: TimelineEvent
    ) -> float:
        """Get the maximum temporal window for two events."""
        w_a = self.TEMPORAL_WINDOWS.get(a.category, 600.0)
        w_b = self.TEMPORAL_WINDOWS.get(b.category, 600.0)
        return max(w_a, w_b)


# ---------------------------------------------------------------------------
# Evidence Chain
# ---------------------------------------------------------------------------

class EvidenceChain:
    """Manages a chain of custody for forensic evidence."""

    def __init__(self, chain_id: Optional[str] = None) -> None:
        self._chain_id = chain_id or uuid.uuid4().hex[:16]
        self._links: List[EvidenceChainLink] = []
        self._event_index: Dict[str, int] = {}
        self._prev_hash = "0" * 64  # Genesis hash

    @property
    def chain_id(self) -> str:
        return self._chain_id

    @property
    def links(self) -> List[EvidenceChainLink]:
        return list(self._links)

    def add_evidence(
        self,
        event_id: str,
        evidence_type: str,
        evidence_data: Dict[str, Any],
        custodian: str = "",
        description: str = "",
    ) -> EvidenceChainLink:
        """Add evidence to the chain."""
        link = EvidenceChainLink(
            event_id=event_id,
            evidence_type=evidence_type,
            evidence_data=evidence_data,
            prev_hash=self._prev_hash,
            custodian=custodian,
            description=description,
        )
        link.compute_hash()
        self._links.append(link)
        self._event_index[event_id] = len(self._links) - 1
        self._prev_hash = link.hash_value
        return link

    def get_evidence(self, event_id: str) -> Optional[EvidenceChainLink]:
        """Get evidence for a specific event."""
        idx = self._event_index.get(event_id)
        if idx is not None:
            return self._links[idx]
        return None

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the entire evidence chain."""
        errors: List[str] = []
        prev_hash = "0" * 64

        for i, link in enumerate(self._links):
            if link.prev_hash != prev_hash:
                errors.append(
                    f"Link {i} ({link.link_id}): prev_hash mismatch. "
                    f"Expected {prev_hash[:16]}..., got {link.prev_hash[:16]}..."
                )
            # Recompute hash
            expected_hash = self._recompute_hash(link)
            if link.hash_value != expected_hash:
                errors.append(
                    f"Link {i} ({link.link_id}): hash mismatch. "
                    f"Stored {link.hash_value[:16]}..., computed {expected_hash[:16]}..."
                )
            prev_hash = link.hash_value

        return len(errors) == 0, errors

    def _recompute_hash(self, link: EvidenceChainLink) -> str:
        """Recompute the hash for a link."""
        content = json.dumps({
            "link_id": link.link_id,
            "event_id": link.event_id,
            "evidence_type": link.evidence_type,
            "evidence_data_keys": sorted(link.evidence_data.keys()),
            "prev_hash": link.prev_hash,
            "timestamp": link.timestamp.isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_chain_summary(self) -> Dict[str, Any]:
        """Get a summary of the evidence chain."""
        type_counts: Dict[str, int] = defaultdict(int)
        custodians: Set[str] = set()
        for link in self._links:
            type_counts[link.evidence_type] += 1
            if link.custodian:
                custodians.add(link.custodian)

        return {
            "chain_id": self._chain_id,
            "total_links": len(self._links),
            "evidence_types": dict(type_counts),
            "custodians": sorted(custodians),
            "first_timestamp": self._links[0].timestamp.isoformat() if self._links else None,
            "last_timestamp": self._links[-1].timestamp.isoformat() if self._links else None,
            "current_hash": self._prev_hash,
        }

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the full chain as a list of dictionaries."""
        return [link.to_dict() for link in self._links]


# ---------------------------------------------------------------------------
# Timeline Visualizer
# ---------------------------------------------------------------------------

class TimelineVisualizer:
    """Visualizes forensic timelines as ASCII art."""

    SEVERITY_COLORS = {
        EventSeverity.INFO: "INFO",
        EventSeverity.LOW: "LOW ",
        EventSeverity.MEDIUM: "MED ",
        EventSeverity.HIGH: "HIGH",
        EventSeverity.CRITICAL: "CRIT",
    }

    CATEGORY_SYMBOLS: Dict[EventCategory, str] = {
        EventCategory.AUTHENTICATION: "AUTH",
        EventCategory.AUTHORIZATION: "AUTH",
        EventCategory.NETWORK: "NET ",
        EventCategory.FILE_SYSTEM: "FILE",
        EventCategory.PROCESS: "PROC",
        EventCategory.MEMORY: "MEM ",
        EventCategory.DATABASE: "DB  ",
        EventCategory.CONFIGURATION: "CONF",
        EventCategory.DATA_ACCESS: "DATA",
        EventCategory.EXFILTRATION: "EXFL",
        EventCategory.INJECTION: "INJ ",
        EventCategory.PRIVILEGE_ESC: "PRIV",
        EventCategory.LATERAL_MOVE: "LAT ",
        EventCategory.PERSISTENCE: "PERS",
        EventCategory.DEFENSE_EVASION: "DEF ",
        EventCategory.UNKNOWN: "??? ",
    }

    def __init__(self, max_width: int = 100, max_events: int = 100) -> None:
        self._max_width = max_width
        self._max_events = max_events

    def render_timeline(
        self,
        events: List[TimelineEvent],
        title: str = "Forensic Timeline",
        show_details: bool = False,
    ) -> str:
        """Render events as a formatted timeline."""
        lines: List[str] = []
        sorted_events = sorted(events)[: self._max_events]

        if not sorted_events:
            return "  (empty timeline)"

        # Header
        lines.append(f"{'=' * self._max_width}")
        lines.append(f"  {title}")
        lines.append(
            f"  Events: {len(sorted_events)} | "
            f"Range: {sorted_events[0].timestamp.isoformat()} - "
            f"{sorted_events[-1].timestamp.isoformat()}"
        )
        lines.append(f"{'=' * self._max_width}")
        lines.append("")

        # Column headers
        lines.append(
            f"  {'SEQ':>5s}  {'TIME':<20s}  {'SEV':>4s}  {'CAT':>4s}  "
            f"{'EVENT_TYPE':<20s}  {'ACTOR':<15s}  {'TARGET':<15s}"
        )
        lines.append(f"  {'-' * 90}")

        # Event rows
        for i, event in enumerate(sorted_events):
            ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:19]
            sev = self.SEVERITY_COLORS.get(event.severity, "????")
            cat = self.CATEGORY_SYMBOLS.get(event.category, "????")
            actor = event.actor[:14] if event.actor else "-"
            target = event.target[:14] if event.target else "-"
            evt_type = event.event_type[:19] if event.event_type else "-"

            lines.append(
                f"  {i:>5d}  {ts:<20s}  {sev:>4s}  {cat:>4s}  "
                f"{evt_type:<20s}  {actor:<15s}  {target:<15s}"
            )

            if show_details and event.description:
                desc = event.description[:80]
                lines.append(f"         {desc}")

        lines.append(f"  {'-' * 90}")
        return "\n".join(lines)

    def render_gaps(self, gaps: List[TimelineGap]) -> str:
        """Render detected timeline gaps."""
        lines: List[str] = []
        lines.append(f"{'=' * 70}")
        lines.append(f"  TIMELINE GAPS DETECTED: {len(gaps)}")
        lines.append(f"{'=' * 70}")

        if not gaps:
            lines.append("  No gaps detected.")
            return "\n".join(lines)

        for gap in gaps:
            sev = self.SEVERITY_COLORS.get(gap.severity, "????")
            duration_str = self._format_duration(gap.duration_seconds)
            lines.append(f"")
            lines.append(f"  [{sev}] Gap: {gap.description}")
            lines.append(
                f"    Start: {gap.start_time.isoformat()} | "
                f"End: {gap.end_time.isoformat()} | "
                f"Duration: {duration_str}"
            )
            if gap.expected_event_types:
                lines.append(
                    f"    Expected types: {', '.join(gap.expected_event_types)}"
                )

        return "\n".join(lines)

    def render_correlations(
        self, correlations: List[CorrelationResult]
    ) -> str:
        """Render event correlations."""
        lines: List[str] = []
        lines.append(f"{'=' * 70}")
        lines.append(f"  EVENT CORRELATIONS: {len(correlations)}")
        lines.append(f"{'=' * 70}")

        for corr in correlations[:50]:
            conf_bar = self._confidence_bar(corr.confidence)
            lines.append(f"")
            lines.append(
                f"  [{corr.correlation_type}] Confidence: {corr.confidence:.2f} {conf_bar}"
            )
            lines.append(f"    Events: {', '.join(corr.event_ids[:5])}")
            lines.append(f"    {corr.description}")
            for ev in corr.evidence[:3]:
                lines.append(f"      - {ev}")

        return "\n".join(lines)

    def render_evidence_chain(self, chain: EvidenceChain) -> str:
        """Render the evidence chain."""
        lines: List[str] = []
        summary = chain.get_chain_summary()

        lines.append(f"{'=' * 70}")
        lines.append(f"  EVIDENCE CHAIN: {summary['chain_id']}")
        lines.append(f"{'=' * 70}")
        lines.append(f"  Total Links: {summary['total_links']}")
        lines.append(f"  Custodians: {', '.join(summary['custodians']) or 'none'}")
        lines.append(f"  Current Hash: {summary['current_hash'][:32]}...")

        if summary["evidence_types"]:
            lines.append(f"  Evidence Types:")
            for etype, count in sorted(summary["evidence_types"].items()):
                lines.append(f"    {etype}: {count}")

        lines.append(f"")

        for i, link in enumerate(chain.links[:30]):
            lines.append(
                f"  [{i:>3d}] {link.evidence_type:<15s} "
                f"event={link.event_id[:12]}  "
                f"hash={link.hash_value[:16]}...  "
                f"custodian={link.custodian or '-'}"
            )

        return "\n".join(lines)

    def render_compact_timeline(
        self, events: List[TimelineEvent], width: int = 80
    ) -> str:
        """Render a compact single-line-per-event timeline."""
        sorted_events = sorted(events)[:self._max_events]
        if not sorted_events:
            return "  (empty)"

        lines: List[str] = []
        lines.append(f"  {'TIMELINE':^{width}}")
        lines.append(f"  {'-' * width}")

        for event in sorted_events:
            sev = self.SEVERITY_COLORS.get(event.severity, "????")
            cat = self.CATEGORY_SYMBOLS.get(event.category, "????")
            ts = event.timestamp.strftime("%H:%M:%S")
            label = f"{event.event_type}"[:25]
            actor = event.actor[:10] if event.actor else ""
            lines.append(f"  [{ts}] {sev} {cat} {label:<25s} {actor}")

        return "\n".join(lines)

    def render_actor_timeline(
        self, events: List[TimelineEvent], actor: str
    ) -> str:
        """Render a timeline filtered to a specific actor."""
        actor_events = [e for e in events if e.actor == actor]
        return self.render_timeline(
            actor_events,
            title=f"Timeline for Actor: {actor}",
            show_details=True,
        )

    def render_category_timeline(
        self, events: List[TimelineEvent], category: EventCategory
    ) -> str:
        """Render a timeline filtered to a specific category."""
        cat_events = [e for e in events if e.category == category]
        return self.render_timeline(
            cat_events,
            title=f"Timeline for Category: {category.value}",
            show_details=True,
        )

    def _format_duration(self, seconds: float) -> str:
        """Format a duration in human-readable form."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 86400:.1f}d"

    def _confidence_bar(self, confidence: float, width: int = 10) -> str:
        """Create a confidence bar."""
        filled = int(confidence * width)
        return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Forensic Timeline (Main Class)
# ---------------------------------------------------------------------------

class ForensicTimeline:
    """Main forensic timeline reconstruction class."""

    def __init__(self, timeline_id: Optional[str] = None) -> None:
        self._timeline_id = timeline_id or uuid.uuid4().hex[:16]
        self._orderer = EventOrderer()
        self._gap_detector = GapDetector()
        self._correlation_analyzer = CorrelationAnalyzer()
        self._evidence_chain = EvidenceChain()
        self._visualizer = TimelineVisualizer()
        self._metadata: Dict[str, Any] = {}
        self._created_at: datetime = datetime.now()

    @property
    def timeline_id(self) -> str:
        return self._timeline_id

    @property
    def events(self) -> List[TimelineEvent]:
        return self._orderer.get_ordered_events()

    def add_event(self, event: TimelineEvent) -> None:
        """Add an event to the timeline."""
        self._orderer.add_event(event)

    def add_events(self, events: List[TimelineEvent]) -> None:
        """Add multiple events to the timeline."""
        self._orderer.add_events(events)

    def add_event_from_dict(self, event_dict: Dict[str, Any]) -> TimelineEvent:
        """Create and add an event from a dictionary."""
        ts_str = event_dict.get("timestamp")
        timestamp: datetime = datetime.now()
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                pass

        category = EventCategory.UNKNOWN
        cat_str = event_dict.get("category", "")
        try:
            category = EventCategory(cat_str)
        except ValueError:
            pass

        severity = EventSeverity.INFO
        sev_str = event_dict.get("severity", "")
        try:
            severity = EventSeverity(sev_str)
        except ValueError:
            pass

        event = TimelineEvent(
            event_id=event_dict.get("event_id", uuid.uuid4().hex[:16]),
            timestamp=timestamp,
            event_type=event_dict.get("event_type", ""),
            category=category,
            severity=severity,
            actor=event_dict.get("actor", ""),
            target=event_dict.get("target", ""),
            source_ip=event_dict.get("source_ip", ""),
            destination_ip=event_dict.get("destination_ip", ""),
            description=event_dict.get("description", ""),
            raw_data=event_dict.get("raw_data", {}),
            evidence_ids=event_dict.get("evidence_ids", []),
            tags=set(event_dict.get("tags", [])),
            confidence=float(event_dict.get("confidence", 1.0)),
            metadata=event_dict.get("metadata", {}),
        )
        self._orderer.add_event(event)
        return event

    def detect_gaps(self) -> List[TimelineGap]:
        """Detect gaps in the timeline."""
        return self._gap_detector.detect_gaps(self.events)

    def correlate_events(self) -> List[CorrelationResult]:
        """Correlate events in the timeline."""
        return self._correlation_analyzer.correlate_events(self.events)

    def correlate_groups(self) -> List[CorrelationResult]:
        """Find groups of correlated events."""
        return self._correlation_analyzer.correlate_event_group(self.events)

    def add_evidence(
        self,
        event_id: str,
        evidence_type: str,
        evidence_data: Dict[str, Any],
        custodian: str = "",
        description: str = "",
    ) -> Optional[EvidenceChainLink]:
        """Add evidence to the chain of custody."""
        event = self._orderer.get_event_by_id(event_id)
        if event is None:
            return None
        return self._evidence_chain.add_evidence(
            event_id, evidence_type, evidence_data, custodian, description
        )

    def verify_evidence_chain(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the evidence chain."""
        return self._evidence_chain.verify_integrity()

    def get_events_in_range(
        self, start: datetime, end: datetime
    ) -> List[TimelineEvent]:
        """Get events within a time range."""
        return self._orderer.get_events_in_range(start, end)

    def get_events_by_actor(self, actor: str) -> List[TimelineEvent]:
        """Get events for a specific actor."""
        return [e for e in self.events if e.actor == actor]

    def get_events_by_category(
        self, category: EventCategory
    ) -> List[TimelineEvent]:
        """Get events for a specific category."""
        return [e for e in self.events if e.category == category]

    def get_events_by_severity(
        self, severity: EventSeverity
    ) -> List[TimelineEvent]:
        """Get events for a specific severity level."""
        return [e for e in self.events if e.severity == severity]

    def render_timeline(
        self, title: str = "Forensic Timeline", detailed: bool = False
    ) -> str:
        """Render the full timeline."""
        return self._visualizer.render_timeline(
            self.events, title=title, show_details=detailed
        )

    def render_gaps(self) -> str:
        """Render detected gaps."""
        return self._visualizer.render_gaps(self.detect_gaps())

    def render_correlations(self) -> str:
        """Render event correlations."""
        return self._visualizer.render_correlations(self.correlate_events())

    def render_evidence_chain(self) -> str:
        """Render the evidence chain."""
        return self._visualizer.render_evidence_chain(self._evidence_chain)

    def render_compact(self) -> str:
        """Render a compact timeline."""
        return self._visualizer.render_compact_timeline(self.events)

    def render_actor_timeline(self, actor: str) -> str:
        """Render timeline for a specific actor."""
        return self._visualizer.render_actor_timeline(self.events, actor)

    def render_category_timeline(
        self, category: EventCategory
    ) -> str:
        """Render timeline for a specific category."""
        return self._visualizer.render_category_timeline(self.events, category)

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive forensic report."""
        gaps = self.detect_gaps()
        correlations = self.correlate_events()
        chain_valid, chain_errors = self.verify_evidence_chain()

        category_counts: Dict[str, int] = defaultdict(int)
        severity_counts: Dict[str, int] = defaultdict(int)
        actor_counts: Dict[str, int] = defaultdict(int)

        for event in self.events:
            category_counts[event.category.value] += 1
            severity_counts[event.severity.value] += 1
            if event.actor:
                actor_counts[event.actor] += 1

        return {
            "timeline_id": self._timeline_id,
            "created_at": self._created_at.isoformat(),
            "metadata": self._metadata,
            "summary": {
                "total_events": len(self.events),
                "time_range": {
                    "start": self.events[0].timestamp.isoformat() if self.events else None,
                    "end": self.events[-1].timestamp.isoformat() if self.events else None,
                },
                "categories": dict(category_counts),
                "severities": dict(severity_counts),
                "actors": dict(sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)),
            },
            "gaps": {
                "total": len(gaps),
                "items": [g.to_dict() for g in gaps],
            },
            "correlations": {
                "total": len(correlations),
                "items": [c.to_dict() for c in correlations[:50]],
            },
            "evidence_chain": {
                "valid": chain_valid,
                "errors": chain_errors,
                "summary": self._evidence_chain.get_chain_summary(),
            },
        }

    def export_json(self, indent: int = 2) -> str:
        """Export the timeline as JSON."""
        report = self.generate_report()
        report["events"] = [e.to_dict() for e in self.events]
        return json.dumps(report, indent=indent, default=str)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set timeline metadata."""
        self._metadata[key] = value
