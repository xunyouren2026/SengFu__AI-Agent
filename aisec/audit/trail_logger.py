"""
Audit Trail Logger - Full-chain audit trail logging system.

Provides request-response correlation, causal chain tracking, session
reconstruction, timeline building, evidence integrity via hash chains,
and compliance export capabilities.

Classes:
    TrailEntry - Single audit trail record.
    TrailQuery - Query builder for filtering trail entries.
    EvidenceIntegrity - Hash chain for tamper-evident audit logs.
    CausalChain - Tracks causal relationships between events.
    TimelineBuilder - Constructs chronological event timelines.
    SessionReconstructor - Reconstructs full sessions from trail entries.
    ComplianceExporter - Exports audit trails in compliance formats.
    AuditTrail - Central audit trail storage and management.
    TrailLogger - High-level API for logging and querying audit trails.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntrySeverity(Enum):
    """Severity level of an audit trail entry."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EntryCategory(Enum):
    """Category of an audit trail entry."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM = "system"
    NETWORK = "network"
    COMPLIANCE = "compliance"
    ANOMALY = "anomaly"


class ComplianceFormat(Enum):
    """Supported compliance export formats."""
    CSV = "csv"
    JSON = "json"
    CEF = "cef"  # Common Event Format
    LEEF = "leef"  # Log Event Extended Format


# ---------------------------------------------------------------------------
# TrailEntry
# ---------------------------------------------------------------------------

@dataclass
class TrailEntry:
    """A single immutable audit trail record.

    Attributes:
        entry_id: Unique identifier for this entry.
        timestamp: UTC timestamp when the event occurred.
        severity: Severity level of the event.
        category: Category classification of the event.
        actor: Identifier of the entity that triggered the event.
        action: Human-readable description of the action performed.
        resource: Target resource identifier (URI, object ID, etc.).
        request_id: Correlation ID linking request and response entries.
        parent_id: ID of the parent entry for causal chain tracking.
        session_id: Session identifier for session reconstruction.
        metadata: Arbitrary key-value metadata attached to the entry.
        payload_hash: SHA-256 hash of the serialised entry for integrity.
        prev_hash: Hash of the preceding entry to form a hash chain.
    """

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: EntrySeverity = EntrySeverity.INFO
    category: EntryCategory = EntryCategory.SYSTEM
    actor: str = ""
    action: str = ""
    resource: str = ""
    request_id: str = ""
    parent_id: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload_hash: str = ""
    prev_hash: str = ""

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entry to a plain dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "request_id": self.request_id,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrailEntry:
        """Deserialise a dictionary into a TrailEntry."""
        ts = data.get("timestamp", "")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = datetime.now(timezone.utc)

        return cls(
            entry_id=data.get("entry_id", uuid.uuid4().hex),
            timestamp=timestamp,
            severity=EntrySeverity(data.get("severity", "info")),
            category=EntryCategory(data.get("category", "system")),
            actor=data.get("actor", ""),
            action=data.get("action", ""),
            resource=data.get("resource", ""),
            request_id=data.get("request_id", ""),
            parent_id=data.get("parent_id", ""),
            session_id=data.get("session_id", ""),
            metadata=data.get("metadata", {}),
            payload_hash=data.get("payload_hash", ""),
            prev_hash=data.get("prev_hash", ""),
        )

    def canonical_json(self) -> str:
        """Return a deterministic JSON representation for hashing."""
        d = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "request_id": self.request_id,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# TrailQuery
# ---------------------------------------------------------------------------

class TrailQuery:
    """Fluent query builder for filtering :class:`TrailEntry` records.

    Usage::

        results = list(
            TrailQuery()
            .by_actor("alice")
            .by_category(EntryCategory.AUTHENTICATION)
            .by_time_range(start, end)
            .execute(store)
        )
    """

    def __init__(self) -> None:
        self._actor: Optional[str] = None
        self._category: Optional[EntryCategory] = None
        self._severity: Optional[EntrySeverity] = None
        self._session_id: Optional[str] = None
        self._request_id: Optional[str] = None
        self._resource: Optional[str] = None
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
        self._action_contains: Optional[str] = None
        self._metadata_key: Optional[str] = None
        self._metadata_value: Any = None
        self._limit: int = 0
        self._offset: int = 0

    def by_actor(self, actor: str) -> TrailQuery:
        self._actor = actor
        return self

    def by_category(self, category: EntryCategory) -> TrailQuery:
        self._category = category
        return self

    def by_severity(self, severity: EntrySeverity) -> TrailQuery:
        self._severity = severity
        return self

    def by_session(self, session_id: str) -> TrailQuery:
        self._session_id = session_id
        return self

    def by_request(self, request_id: str) -> TrailQuery:
        self._request_id = request_id
        return self

    def by_resource(self, resource: str) -> TrailQuery:
        self._resource = resource
        return self

    def by_time_range(
        self, start: Optional[datetime], end: Optional[datetime]
    ) -> TrailQuery:
        self._start = start
        self._end = end
        return self

    def by_action_contains(self, substring: str) -> TrailQuery:
        self._action_contains = substring
        return self

    def by_metadata(self, key: str, value: Any) -> TrailQuery:
        self._metadata_key = key
        self._metadata_value = value
        return self

    def limit(self, count: int) -> TrailQuery:
        self._limit = count
        return self

    def offset(self, count: int) -> TrailQuery:
        self._offset = count
        return self

    def matches(self, entry: TrailEntry) -> bool:
        """Return True if *entry* satisfies all active filters."""
        if self._actor and entry.actor != self._actor:
            return False
        if self._category and entry.category != self._category:
            return False
        if self._severity and entry.severity != self._severity:
            return False
        if self._session_id and entry.session_id != self._session_id:
            return False
        if self._request_id and entry.request_id != self._request_id:
            return False
        if self._resource and entry.resource != self._resource:
            return False
        if self._start and entry.timestamp < self._start:
            return False
        if self._end and entry.timestamp > self._end:
            return False
        if self._action_contains and self._action_contains not in entry.action:
            return False
        if self._metadata_key is not None:
            if self._metadata_key not in entry.metadata:
                return False
            if entry.metadata[self._metadata_key] != self._metadata_value:
                return False
        return True

    def execute(self, entries: Sequence[TrailEntry]) -> List[TrailEntry]:
        """Run the query against a sequence of entries and return matches."""
        results: List[TrailEntry] = []
        for entry in entries:
            if self.matches(entry):
                results.append(entry)
        results.sort(key=lambda e: e.timestamp)
        if self._offset:
            results = results[self._offset :]
        if self._limit:
            results = results[: self._limit]
        return results


# ---------------------------------------------------------------------------
# EvidenceIntegrity
# ---------------------------------------------------------------------------

class EvidenceIntegrity:
    """Hash-chain based evidence integrity verification.

    Each entry's ``payload_hash`` is the SHA-256 digest of its canonical
    JSON, and ``prev_hash`` references the previous entry's hash, forming
    a tamper-evident chain.
    """

    HASH_ALGORITHM = "sha256"

    @staticmethod
    def compute_hash(data: str) -> str:
        """Return the hex digest of *data* using the configured algorithm."""
        return hashlib.new(EvidenceIntegrity.HASH_ALGORITHM, data.encode("utf-8")).hexdigest()

    @staticmethod
    def seal(entry: TrailEntry, prev_hash: str = "") -> TrailEntry:
        """Compute hashes and return a new sealed entry."""
        entry.prev_hash = prev_hash
        entry.payload_hash = EvidenceIntegrity.compute_hash(entry.canonical_json())
        return entry

    @staticmethod
    def verify_chain(entries: Sequence[TrailEntry]) -> Tuple[bool, List[str]]:
        """Verify the integrity of a complete hash chain.

        Returns:
            A tuple ``(is_valid, issues)`` where *is_valid* is True when
            every link in the chain is intact and *issues* lists human-
            readable descriptions of any problems found.
        """
        issues: List[str] = []
        if not entries:
            return True, issues

        for idx, entry in enumerate(entries):
            # Verify payload hash
            expected_hash = EvidenceIntegrity.compute_hash(entry.canonical_json())
            if entry.payload_hash != expected_hash:
                issues.append(
                    f"Entry {entry.entry_id} at index {idx}: payload hash mismatch "
                    f"(expected {expected_hash}, got {entry.payload_hash})"
                )

            # Verify chain link
            if idx == 0:
                if entry.prev_hash and entry.prev_hash != "":
                    issues.append(
                        f"Entry {entry.entry_id} at index 0: first entry should "
                        f"have empty prev_hash, got {entry.prev_hash}"
                    )
            else:
                if entry.prev_hash != entries[idx - 1].payload_hash:
                    issues.append(
                        f"Entry {entry.entry_id} at index {idx}: prev_hash does not "
                        f"match payload_hash of entry {entries[idx - 1].entry_id}"
                    )

        return len(issues) == 0, issues

    @staticmethod
    def merkle_root(entries: Sequence[TrailEntry]) -> str:
        """Compute the Merkle root hash of a sequence of entries.

        If the number of hashes is odd at any level, the last hash is
        duplicated before moving to the next level.
        """
        if not entries:
            return EvidenceIntegrity.compute_hash("")

        hashes: List[str] = [e.payload_hash for e in entries]
        while len(hashes) > 1:
            next_level: List[str] = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else hashes[i]
                combined = left + right
                next_level.append(EvidenceIntegrity.compute_hash(combined))
            hashes = next_level
        return hashes[0]


# ---------------------------------------------------------------------------
# CausalChain
# ---------------------------------------------------------------------------

@dataclass
class CausalLink:
    """Represents a single causal relationship between two trail entries."""
    cause_id: str
    effect_id: str
    relation_type: str  # e.g. "triggered", "caused", "depended_on"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalChain:
    """Tracks and queries causal relationships between audit events.

    The chain is stored as an adjacency list and supports topological
    ordering, root-cause analysis, and transitive closure computation.
    """

    def __init__(self) -> None:
        self._links: Dict[str, List[CausalLink]] = {}  # cause_id -> links
        self._effects: Dict[str, List[CausalLink]] = {}  # effect_id -> links

    def add_link(self, link: CausalLink) -> None:
        """Register a causal relationship."""
        self._links.setdefault(link.cause_id, []).append(link)
        self._effects.setdefault(link.effect_id, []).append(link)

    def get_effects(self, entry_id: str) -> List[CausalLink]:
        """Return all links where *entry_id* is the cause."""
        return list(self._links.get(entry_id, []))

    def get_causes(self, entry_id: str) -> List[CausalLink]:
        """Return all links where *entry_id* is the effect."""
        return list(self._effects.get(entry_id, []))

    def topological_order(self, entry_ids: Optional[Set[str]] = None) -> List[str]:
        """Return entry IDs in topological (causal) order.

        Uses Kahn's algorithm. If *entry_ids* is provided, only those
        entries are considered; otherwise all known entries are included.
        """
        # Build adjacency and in-degree maps
        all_ids: Set[str] = set()
        adj: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}

        for cause_id, links in self._links.items():
            for link in links:
                if entry_ids is not None and (cause_id not in entry_ids or link.effect_id not in entry_ids):
                    continue
                all_ids.add(cause_id)
                all_ids.add(link.effect_id)
                adj.setdefault(cause_id, []).append(link.effect_id)
                in_degree[link.effect_id] = in_degree.get(link.effect_id, 0) + 1
                if cause_id not in in_degree:
                    in_degree[cause_id] = in_degree.get(cause_id, 0)

        # Nodes with no incoming edges
        queue: List[str] = [nid for nid in all_ids if in_degree.get(nid, 0) == 0]
        heapq.heapify(queue)
        result: List[str] = []

        while queue:
            node = heapq.heappop(queue)
            result.append(node)
            for neighbour in adj.get(node, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    heapq.heappush(queue, neighbour)

        return result

    def root_cause(self, entry_id: str) -> List[str]:
        """Trace back from *entry_id* to find all root causes.

        Performs a breadth-first search backwards through the causal
        graph and returns entry IDs that have no incoming causal links.
        """
        visited: Set[str] = set()
        queue: List[str] = [entry_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            causes = self.get_causes(current)
            if not causes:
                continue
            for link in causes:
                if link.cause_id not in visited:
                    queue.append(link.cause_id)

        # Roots are visited nodes with no causes
        roots: List[str] = []
        for nid in visited:
            if not self._effects.get(nid):
                roots.append(nid)
        return roots

    def transitive_effects(self, entry_id: str) -> Set[str]:
        """Return the set of all transitively effected entry IDs."""
        visited: Set[str] = set()
        stack: List[str] = [entry_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for link in self.get_effects(current):
                if link.effect_id not in visited:
                    stack.append(link.effect_id)
        visited.discard(entry_id)
        return visited


# ---------------------------------------------------------------------------
# TimelineBuilder
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single point on a timeline."""
    timestamp: datetime
    entry_id: str
    summary: str
    severity: EntrySeverity
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimelineBuilder:
    """Constructs chronological timelines from audit trail entries.

    Supports merging multiple streams, gap detection, and annotated
    event summaries.
    """

    def __init__(self) -> None:
        self._events: List[TimelineEvent] = []

    def add_entry(
        self,
        entry: TrailEntry,
        summary_fn: Optional[Callable[[TrailEntry], str]] = None,
    ) -> None:
        """Append a timeline event derived from a trail entry."""
        summary = summary_fn(entry) if summary_fn else self._default_summary(entry)
        self._events.append(
            TimelineEvent(
                timestamp=entry.timestamp,
                entry_id=entry.entry_id,
                summary=summary,
                severity=entry.severity,
                metadata=dict(entry.metadata),
            )
        )

    def add_entries(
        self,
        entries: Sequence[TrailEntry],
        summary_fn: Optional[Callable[[TrailEntry], str]] = None,
    ) -> None:
        """Bulk-add entries to the timeline."""
        for entry in entries:
            self.add_entry(entry, summary_fn)

    def build(self) -> List[TimelineEvent]:
        """Return events sorted chronologically."""
        return sorted(self._events, key=lambda e: e.timestamp)

    def detect_gaps(
        self,
        max_gap_seconds: float = 300.0,
    ) -> List[Tuple[datetime, datetime, float]]:
        """Detect time gaps larger than *max_gap_seconds* between events.

        Returns:
            A list of ``(start, end, gap_seconds)`` tuples.
        """
        sorted_events = self.build()
        gaps: List[Tuple[datetime, datetime, float]] = []
        for i in range(1, len(sorted_events)):
            delta = (sorted_events[i].timestamp - sorted_events[i - 1].timestamp).total_seconds()
            if delta > max_gap_seconds:
                gaps.append(
                    (sorted_events[i - 1].timestamp, sorted_events[i].timestamp, delta)
                )
        return gaps

    def coalesce(
        self,
        window_seconds: float = 5.0,
        group_fn: Optional[Callable[[List[TimelineEvent]], str]] = None,
    ) -> List[Dict[str, Any]]:
        """Group events within *window_seconds* of each other.

        Returns a list of dicts with keys ``start``, ``end``, ``count``,
        and ``summary``.
        """
        sorted_events = self.build()
        if not sorted_events:
            return []

        groups: List[List[TimelineEvent]] = [[sorted_events[0]]]
        for ev in sorted_events[1:]:
            last = groups[-1][-1]
            if (ev.timestamp - last.timestamp).total_seconds() <= window_seconds:
                groups[-1].append(ev)
            else:
                groups.append([ev])

        result: List[Dict[str, Any]] = []
        for group in groups:
            summary = group_fn(group) if group_fn else "; ".join(e.summary for e in group)
            result.append({
                "start": group[0].timestamp.isoformat(),
                "end": group[-1].timestamp.isoformat(),
                "count": len(group),
                "summary": summary,
            })
        return result

    @staticmethod
    def _default_summary(entry: TrailEntry) -> str:
        parts: List[str] = []
        if entry.actor:
            parts.append(f"[{entry.actor}]")
        parts.append(entry.action)
        if entry.resource:
            parts.append(f"on {entry.resource}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# SessionReconstructor
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Reconstructed session from audit trail entries."""
    session_id: str
    actor: str
    start_time: datetime
    end_time: datetime
    entries: List[TrailEntry] = field(default_factory=list)
    request_ids: List[str] = field(default_factory=list)
    resources_accessed: Set[str] = field(default_factory=set)
    actions: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "actor": self.actor,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "entry_count": self.entry_count,
            "request_ids": self.request_ids,
            "resources_accessed": sorted(self.resources_accessed),
            "actions": self.actions,
        }


class SessionReconstructor:
    """Reconstructs user / system sessions from audit trail entries.

    A session is a group of entries sharing the same ``session_id``.
    The reconstructor sorts them chronologically and extracts summary
    statistics.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def ingest(self, entry: TrailEntry) -> None:
        """Add an entry to the appropriate session bucket."""
        sid = entry.session_id or "__unsessioned__"
        if sid not in self._sessions:
            self._sessions[sid] = Session(
                session_id=sid,
                actor=entry.actor,
                start_time=entry.timestamp,
                end_time=entry.timestamp,
            )
        session = self._sessions[sid]
        if entry.timestamp < session.start_time:
            session.start_time = entry.timestamp
        if entry.timestamp > session.end_time:
            session.end_time = entry.timestamp
        session.entries.append(entry)
        if entry.request_id and entry.request_id not in session.request_ids:
            session.request_ids.append(entry.request_id)
        if entry.resource:
            session.resources_accessed.add(entry.resource)
        session.actions.append(entry.action)

    def ingest_all(self, entries: Sequence[TrailEntry]) -> None:
        """Bulk-ingest entries."""
        for entry in entries:
            self.ingest(entry)

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Session]:
        """Return all sessions sorted by start time."""
        return sorted(self._sessions.values(), key=lambda s: s.start_time)

    def find_sessions_by_actor(self, actor: str) -> List[Session]:
        """Return sessions belonging to *actor*."""
        return [s for s in self.list_sessions() if s.actor == actor]

    def find_sessions_by_time_range(
        self, start: datetime, end: datetime
    ) -> List[Session]:
        """Return sessions that overlap with the given time range."""
        results: List[Session] = []
        for session in self.list_sessions():
            if session.end_time >= start and session.start_time <= end:
                results.append(session)
        return results

    def correlate_requests(self, request_id: str) -> List[TrailEntry]:
        """Return all entries across all sessions for a given request_id."""
        results: List[TrailEntry] = []
        for session in self._sessions.values():
            for entry in session.entries:
                if entry.request_id == request_id:
                    results.append(entry)
        results.sort(key=lambda e: e.timestamp)
        return results


# ---------------------------------------------------------------------------
# ComplianceExporter
# ---------------------------------------------------------------------------

class ComplianceExporter:
    """Exports audit trail data in various compliance-friendly formats.

    Supported formats: CSV, JSON, CEF (Common Event Format), and
    LEEF (Log Event Extended Format).
    """

    @staticmethod
    def to_csv(entries: Sequence[TrailEntry]) -> str:
        """Serialise entries to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)
        headers = [
            "entry_id", "timestamp", "severity", "category", "actor",
            "action", "resource", "request_id", "parent_id", "session_id",
            "payload_hash", "prev_hash",
        ]
        writer.writerow(headers)
        for entry in entries:
            writer.writerow([
                entry.entry_id,
                entry.timestamp.isoformat(),
                entry.severity.value,
                entry.category.value,
                entry.actor,
                entry.action,
                entry.resource,
                entry.request_id,
                entry.parent_id,
                entry.session_id,
                entry.payload_hash,
                entry.prev_hash,
            ])
        return output.getvalue()

    @staticmethod
    def to_json(entries: Sequence[TrailEntry], indent: int = 2) -> str:
        """Serialise entries to JSON format."""
        data = [e.to_dict() for e in entries]
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @staticmethod
    def to_cef(entries: Sequence[TrailEntry]) -> str:
        """Serialise entries to Common Event Format (CEF).

        Each entry becomes a CEF syslog line with extension fields.
        """
        lines: List[str] = []
        for entry in entries:
            ext = (
                f"act={entry.action} "
                f"src={entry.actor} "
                f"dst={entry.resource} "
                f"requestId={entry.request_id} "
                f"sessionId={entry.session_id} "
                f"msg={entry.action} "
                f"cat={entry.category.value} "
                f"sev={ComplianceExporter._cef_severity(entry.severity)}"
            )
            # CEF version 0 format: CEF:Version|Device Vendor|Device Product|
            # Device Version|Signature ID|Name|Severity|Extension
            line = (
                f"CEF:0|AGIUnified|AISecAudit|1.0|{entry.entry_id}|"
                f"{entry.category.value}|{ComplianceExporter._cef_severity(entry.severity)}|"
                f"{ext}"
            )
            lines.append(
                f"<134>{entry.timestamp.strftime('%b %d %H:%M:%S')} "
                f"audit-trail {line}"
            )
        return "\n".join(lines)

    @staticmethod
    def to_leef(entries: Sequence[TrailEntry]) -> str:
        """Serialise entries to Log Event Extended Format (LEEF)."""
        lines: List[str] = []
        for entry in entries:
            # LEEF 2.0 header
            cat = entry.category.value.upper().replace(" ", "_")
            ext_parts = [
                f"act={entry.action}",
                f"src={entry.actor}",
                f"dst={entry.resource}",
                f"requestId={entry.request_id}",
                f"sessionId={entry.session_id}",
                f"sev={entry.severity.value}",
                f"cat={cat}",
                f"devTime={entry.timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:23]}Z",
            ]
            if entry.metadata:
                for k, v in entry.metadata.items():
                    ext_parts.append(f"custom-{k}={v}")
            ext = "\t".join(ext_parts)
            line = (
                f"LEEF:2.0|AGIUnified|AISecAudit|1.0|{cat}|{entry.action}|"
                f"{ext}"
            )
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def export(
        entries: Sequence[TrailEntry],
        fmt: ComplianceFormat,
        **kwargs: Any,
    ) -> str:
        """Export entries in the requested format.

        Args:
            entries: Sequence of trail entries to export.
            fmt: Target compliance format.
            **kwargs: Additional keyword arguments passed to the
                      underlying serialiser (e.g. ``indent`` for JSON).

        Returns:
            A string containing the formatted output.
        """
        dispatch: Dict[ComplianceFormat, Callable[..., str]] = {
            ComplianceFormat.CSV: ComplianceExporter.to_csv,
            ComplianceFormat.JSON: ComplianceExporter.to_json,
            ComplianceFormat.CEF: ComplianceExporter.to_cef,
            ComplianceFormat.LEEF: ComplianceExporter.to_leef,
        }
        serialiser = dispatch.get(fmt)
        if serialiser is None:
            raise ValueError(f"Unsupported compliance format: {fmt}")
        return serialiser(entries, **kwargs)

    @staticmethod
    def _cef_severity(severity: EntrySeverity) -> int:
        """Map EntrySeverity to CEF severity (0-10)."""
        mapping = {
            EntrySeverity.DEBUG: 0,
            EntrySeverity.INFO: 3,
            EntrySeverity.WARNING: 5,
            EntrySeverity.ERROR: 7,
            EntrySeverity.CRITICAL: 10,
        }
        return mapping.get(severity, 3)


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------

class AuditTrail:
    """Central storage and management for audit trail entries.

    Provides in-memory storage with hash-chain integrity, indexing by
    multiple fields, and query execution.
    """

    def __init__(self) -> None:
        self._entries: List[TrailEntry] = []
        self._by_id: Dict[str, TrailEntry] = {}
        self._by_request: Dict[str, List[TrailEntry]] = {}
        self._by_session: Dict[str, List[TrailEntry]] = {}
        self._by_actor: Dict[str, List[TrailEntry]] = {}
        self._last_hash: str = ""
        self._integrity = EvidenceIntegrity()

    @property
    def size(self) -> int:
        """Return the number of stored entries."""
        return len(self._entries)

    def append(self, entry: TrailEntry) -> TrailEntry:
        """Append an entry, sealing it into the hash chain.

        Returns the sealed entry with computed hashes.
        """
        sealed = self._integrity.seal(entry, self._last_hash)
        self._entries.append(sealed)
        self._by_id[sealed.entry_id] = sealed
        self._by_request.setdefault(sealed.request_id, []).append(sealed)
        self._by_session.setdefault(sealed.session_id, []).append(sealed)
        self._by_actor.setdefault(sealed.actor, []).append(sealed)
        self._last_hash = sealed.payload_hash
        return sealed

    def get(self, entry_id: str) -> Optional[TrailEntry]:
        """Retrieve an entry by its ID."""
        return self._by_id.get(entry_id)

    def get_by_request(self, request_id: str) -> List[TrailEntry]:
        """Return all entries for a given request correlation ID."""
        return list(self._by_request.get(request_id, []))

    def get_by_session(self, session_id: str) -> List[TrailEntry]:
        """Return all entries for a given session ID."""
        return list(self._by_session.get(session_id, []))

    def get_by_actor(self, actor: str) -> List[TrailEntry]:
        """Return all entries for a given actor."""
        return list(self._by_actor.get(actor, []))

    def query(self, q: Optional[TrailQuery] = None) -> List[TrailEntry]:
        """Execute a query against the stored entries.

        If *q* is None, all entries are returned sorted by timestamp.
        """
        if q is None:
            return sorted(self._entries, key=lambda e: e.timestamp)
        return q.execute(self._entries)

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the hash chain integrity of all stored entries."""
        return self._integrity.verify_chain(self._entries)

    def merkle_root(self) -> str:
        """Compute the Merkle root of all stored entries."""
        return self._integrity.merkle_root(self._entries)

    def entries(self) -> Iterator[TrailEntry]:
        """Iterate over all entries in insertion order."""
        return iter(self._entries)

    def clear(self) -> None:
        """Remove all entries and reset the chain."""
        self._entries.clear()
        self._by_id.clear()
        self._by_request.clear()
        self._by_session.clear()
        self._by_actor.clear()
        self._last_hash = ""


# ---------------------------------------------------------------------------
# TrailLogger
# ---------------------------------------------------------------------------

class TrailLogger:
    """High-level API for logging and querying audit trails.

    Combines :class:`AuditTrail`, :class:`CausalChain`,
    :class:`SessionReconstructor`, :class:`TimelineBuilder`, and
    :class:`ComplianceExporter` into a unified interface.

    Usage::

        logger = TrailLogger()
        logger.log(
            actor="alice",
            action="login",
            resource="/auth",
            category=EntryCategory.AUTHENTICATION,
            session_id="sess-001",
            request_id="req-001",
        )
        results = logger.query(TrailQuery().by_actor("alice"))
    """

    def __init__(self) -> None:
        self._trail = AuditTrail()
        self._causal = CausalChain()
        self._sessions = SessionReconstructor()
        self._timeline = TimelineBuilder()

    # -- logging -------------------------------------------------------------

    def log(
        self,
        actor: str = "",
        action: str = "",
        resource: str = "",
        severity: EntrySeverity = EntrySeverity.INFO,
        category: EntryCategory = EntryCategory.SYSTEM,
        request_id: str = "",
        parent_id: str = "",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrailEntry:
        """Create, seal, and store a new audit trail entry.

        Returns the sealed entry.
        """
        entry = TrailEntry(
            actor=actor,
            action=action,
            resource=resource,
            severity=severity,
            category=category,
            request_id=request_id,
            parent_id=parent_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        sealed = self._trail.append(entry)
        self._sessions.ingest(sealed)
        self._timeline.add_entry(sealed)

        # Auto-register causal link if parent_id is set
        if parent_id:
            self._causal.add_link(
                CausalLink(
                    cause_id=parent_id,
                    effect_id=sealed.entry_id,
                    relation_type="triggered",
                )
            )

        return sealed

    def log_causal(
        self,
        cause_entry_id: str,
        relation_type: str = "triggered",
        **kwargs: Any,
    ) -> TrailEntry:
        """Log a new entry that is causally linked to an existing entry."""
        return self.log(parent_id=cause_entry_id, **kwargs)

    # -- querying ------------------------------------------------------------

    def query(self, q: Optional[TrailQuery] = None) -> List[TrailEntry]:
        """Execute a query against the audit trail."""
        return self._trail.query(q)

    def get(self, entry_id: str) -> Optional[TrailEntry]:
        """Retrieve a single entry by ID."""
        return self._trail.get(entry_id)

    def get_request_chain(self, request_id: str) -> List[TrailEntry]:
        """Return all entries correlated by *request_id*, sorted by time."""
        return self._trail.get_by_request(request_id)

    # -- causal chain --------------------------------------------------------

    def get_effects(self, entry_id: str) -> List[CausalLink]:
        """Return causal links where *entry_id* is the cause."""
        return self._causal.get_effects(entry_id)

    def get_causes(self, entry_id: str) -> List[CausalLink]:
        """Return causal links where *entry_id* is the effect."""
        return self._causal.get_causes(entry_id)

    def topological_order(self) -> List[str]:
        """Return all entry IDs in causal (topological) order."""
        return self._causal.topological_order()

    def root_cause(self, entry_id: str) -> List[str]:
        """Trace back to find root causes of the given entry."""
        return self._causal.root_cause(entry_id)

    def transitive_effects(self, entry_id: str) -> Set[str]:
        """Return all transitively effected entry IDs."""
        return self._causal.transitive_effects(entry_id)

    # -- session reconstruction ----------------------------------------------

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return a reconstructed session by ID."""
        return self._sessions.get_session(session_id)

    def list_sessions(self) -> List[Session]:
        """Return all reconstructed sessions."""
        return self._sessions.list_sessions()

    def find_sessions_by_actor(self, actor: str) -> List[Session]:
        """Return sessions belonging to *actor*."""
        return self._sessions.find_sessions_by_actor(actor)

    def correlate_requests(self, request_id: str) -> List[TrailEntry]:
        """Return all entries across sessions for a given request_id."""
        return self._sessions.correlate_requests(request_id)

    # -- timeline ------------------------------------------------------------

    def build_timeline(self) -> List[TimelineEvent]:
        """Build and return the chronological timeline."""
        return self._timeline.build()

    def detect_timeline_gaps(
        self, max_gap_seconds: float = 300.0
    ) -> List[Tuple[datetime, datetime, float]]:
        """Detect time gaps in the timeline."""
        return self._timeline.detect_gaps(max_gap_seconds)

    def coalesce_timeline(
        self,
        window_seconds: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Group close-together timeline events."""
        return self._timeline.coalesce(window_seconds)

    # -- integrity -----------------------------------------------------------

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the hash chain integrity of the entire trail."""
        return self._trail.verify_integrity()

    def merkle_root(self) -> str:
        """Return the Merkle root hash of all entries."""
        return self._trail.merkle_root()

    # -- export --------------------------------------------------------------

    def export(
        self,
        fmt: ComplianceFormat = ComplianceFormat.JSON,
        query: Optional[TrailQuery] = None,
        **kwargs: Any,
    ) -> str:
        """Export audit trail entries in a compliance format.

        Args:
            fmt: Target export format.
            query: Optional query to filter entries before export.
            **kwargs: Additional arguments for the serialiser.

        Returns:
            Formatted string representation of the entries.
        """
        entries = self.query(query)
        return ComplianceExporter.export(entries, fmt, **kwargs)

    # -- statistics ----------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the audit trail."""
        entries = self._trail.query()
        if not entries:
            return {
                "total_entries": 0,
                "unique_actors": 0,
                "unique_sessions": 0,
                "unique_requests": 0,
                "categories": {},
                "severities": {},
            }

        actors: Set[str] = set()
        sessions: Set[str] = set()
        requests: Set[str] = set()
        categories: Dict[str, int] = {}
        severities: Dict[str, int] = {}

        for entry in entries:
            if entry.actor:
                actors.add(entry.actor)
            if entry.session_id:
                sessions.add(entry.session_id)
            if entry.request_id:
                requests.add(entry.request_id)
            cat = entry.category.value
            categories[cat] = categories.get(cat, 0) + 1
            sev = entry.severity.value
            severities[sev] = severities.get(sev, 0) + 1

        return {
            "total_entries": len(entries),
            "unique_actors": len(actors),
            "unique_sessions": len(sessions),
            "unique_requests": len(requests),
            "categories": categories,
            "severities": severities,
        }
