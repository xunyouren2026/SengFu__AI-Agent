"""
Attack Graph Visualization Module
==================================
Graph construction from audit events, causal relationship inference,
attack path enumeration, critical path analysis, and ASCII art rendering.

Pure Python standard library implementation.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """Type of node in the attack graph."""
    ENTRY = "entry"
    VULNERABILITY = "vulnerability"
    ACTION = "action"
    RESOURCE = "resource"
    EXFILTRATION = "exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    DEFENSE = "defense"
    UNKNOWN = "unknown"


class EdgeType(Enum):
    """Type of edge in the attack graph."""
    ENABLES = "enables"
    EXPLOITS = "exploits"
    LEADS_TO = "leads_to"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    COMMUNICATES = "communicates"
    EVADER = "evades"
    UNKNOWN = "unknown"


@dataclass
class Node:
    """A node in the attack graph representing an entity or event."""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    label: str = ""
    node_type: NodeType = NodeType.UNKNOWN
    properties: Dict[str, Any] = field(default_factory=dict)
    severity: float = 0.0
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    source_event_ids: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.node_id == other.node_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type.value,
            "properties": self.properties,
            "severity": self.severity,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source_event_ids": self.source_event_ids,
        }


@dataclass
class Edge:
    """A directed edge in the attack graph."""
    edge_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.UNKNOWN
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    causal_evidence: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.edge_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.edge_type == other.edge_type
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "causal_evidence": self.causal_evidence,
        }


@dataclass
class AttackPath:
    """A sequence of nodes and edges forming an attack path."""
    path_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    total_severity: float = 0.0
    total_confidence: float = 0.0
    length: int = 0

    def compute_metrics(self) -> None:
        """Compute aggregate metrics for the path."""
        self.length = len(self.nodes)
        if self.nodes:
            self.total_severity = sum(n.severity for n in self.nodes) / max(self.length, 1)
            self.total_confidence = (
                min((n.confidence for n in self.nodes), default=1.0)
                if self.nodes
                else 1.0
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "node_ids": [n.node_id for n in self.nodes],
            "edge_ids": [e.edge_id for e in self.edges],
            "total_severity": self.total_severity,
            "total_confidence": self.total_confidence,
            "length": self.length,
        }


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Constructs an attack graph from audit events."""

    # Patterns that map audit events to node types
    EVENT_TYPE_PATTERNS: Dict[str, Tuple[NodeType, EdgeType]] = {
        "login_failed": (NodeType.ENTRY, EdgeType.ENABLES),
        "login_success": (NodeType.ENTRY, EdgeType.LEADS_TO),
        "privilege_escalation": (NodeType.PRIVILEGE_ESCALATION, EdgeType.EXPLOITS),
        "lateral_movement": (NodeType.LATERAL_MOVEMENT, EdgeType.COMMUNICATES),
        "data_access": (NodeType.RESOURCE, EdgeType.CONTAINS),
        "data_exfil": (NodeType.EXFILTRATION, EdgeType.LEADS_TO),
        "file_write": (NodeType.PERSISTENCE, EdgeType.LEADS_TO),
        "command_exec": (NodeType.ACTION, EdgeType.EXPLOITS),
        "vuln_detected": (NodeType.VULNERABILITY, EdgeType.EXPLOITS),
        "firewall_block": (NodeType.DEFENSE, EdgeType.EVADER),
        "suspicious_process": (NodeType.ACTION, EdgeType.LEADS_TO),
        "network_scan": (NodeType.ENTRY, EdgeType.ENABLES),
        "exploit_attempt": (NodeType.VULNERABILITY, EdgeType.EXPLOITS),
        "credential_access": (NodeType.ENTRY, EdgeType.EXPLOITS),
        "defense_evasion": (NodeType.ACTION, EdgeType.EVADER),
    }

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._reverse_adj: Dict[str, List[str]] = defaultdict(list)
        self._event_node_map: Dict[str, str] = {}

    @property
    def nodes(self) -> Dict[str, Node]:
        return dict(self._nodes)

    @property
    def edges(self) -> Dict[str, Edge]:
        return dict(self._edges)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge to the graph."""
        key = f"{edge.source_id}->{edge.target_id}:{edge.edge_type.value}"
        if key not in self._edges:
            self._edges[key] = edge
            self._adjacency[edge.source_id].append(edge.target_id)
            self._reverse_adj[edge.target_id].append(edge.source_id)

    def add_audit_event(self, event: Dict[str, Any]) -> Optional[str]:
        """Add an audit event and return the created node ID."""
        event_id = event.get("event_id", uuid.uuid4().hex[:12])
        event_type = event.get("event_type", "unknown")
        actor = event.get("actor", "unknown")
        target = event.get("target", "")
        severity = float(event.get("severity", 0.5))
        timestamp_str = event.get("timestamp")
        timestamp: Optional[datetime] = None
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                timestamp = datetime.now()

        node_type, edge_type = self.EVENT_TYPE_PATTERNS.get(
            event_type, (NodeType.UNKNOWN, EdgeType.LEADS_TO)
        )

        node_label = f"{event_type}:{actor}"
        if target:
            node_label += f" -> {target}"

        node = Node(
            node_id=event_id,
            label=node_label,
            node_type=node_type,
            severity=severity,
            confidence=float(event.get("confidence", 1.0)),
            timestamp=timestamp,
            source_event_ids=[event_id],
            properties={
                "actor": actor,
                "target": target,
                "event_type": event_type,
                "raw_event": event.get("details", {}),
            },
        )
        self.add_node(node)
        self._event_node_map[event_id] = event_id

        # Try to connect to related events
        related_events = event.get("related_events", [])
        parent_event = event.get("parent_event_id", "")
        if parent_event and parent_event in self._event_node_map:
            parent_node_id = self._event_node_map[parent_event]
            edge = Edge(
                source_id=parent_node_id,
                target_id=node.node_id,
                edge_type=edge_type,
                weight=severity,
                causal_evidence=[f"Event {event_id} caused by {parent_event}"],
            )
            self.add_edge(edge)

        for rel_id in related_events:
            if rel_id in self._event_node_map:
                rel_node_id = self._event_node_map[rel_id]
                edge = Edge(
                    source_id=rel_node_id,
                    target_id=node.node_id,
                    edge_type=edge_type,
                    weight=severity * 0.8,
                    causal_evidence=[f"Correlation: {rel_id} -> {event_id}"],
                )
                self.add_edge(edge)

        return node.node_id

    def build_from_events(self, events: List[Dict[str, Any]]) -> "AttackGraph":
        """Build a complete attack graph from a list of audit events."""
        for event in events:
            self.add_audit_event(event)
        return AttackGraph(
            nodes=self._nodes,
            edges=self._edges,
            adjacency=dict(self._adjacency),
            reverse_adjacency=dict(self._reverse_adj),
        )

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get successor node IDs."""
        return self._adjacency.get(node_id, [])

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs."""
        return self._reverse_adj.get(node_id, [])


# ---------------------------------------------------------------------------
# Causal Analyzer
# ---------------------------------------------------------------------------

class CausalAnalyzer:
    """Infers causal relationships between events in the attack graph."""

    # Temporal window for causal inference (seconds)
    DEFAULT_TEMPORAL_WINDOW: float = 300.0

    # Actor-based causal patterns
    ACTOR_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
        "reconnaissance": [
            ("network_scan", "exploit_attempt"),
            ("network_scan", "vuln_detected"),
            ("login_failed", "credential_access"),
        ],
        "initial_access": [
            ("exploit_attempt", "login_success"),
            ("credential_access", "login_success"),
            ("phishing", "login_success"),
        ],
        "execution": [
            ("login_success", "command_exec"),
            ("login_success", "file_write"),
            ("exploit_attempt", "command_exec"),
        ],
        "persistence": [
            ("file_write", "command_exec"),
            ("command_exec", "file_write"),
            ("file_write", "file_write"),
        ],
        "privilege_escalation": [
            ("command_exec", "privilege_escalation"),
            ("file_write", "privilege_escalation"),
            ("exploit_attempt", "privilege_escalation"),
        ],
        "lateral_movement": [
            ("privilege_escalation", "lateral_movement"),
            ("command_exec", "lateral_movement"),
            ("credential_access", "lateral_movement"),
        ],
        "exfiltration": [
            ("data_access", "data_exfil"),
            ("lateral_movement", "data_exfil"),
            ("privilege_escalation", "data_exfil"),
        ],
    }

    def __init__(self, temporal_window: float = DEFAULT_TEMPORAL_WINDOW) -> None:
        self._temporal_window = temporal_window
        self._causal_rules: List[Callable[[Dict[str, Any], Dict[str, Any]], float]] = [
            self._temporal_proximity_score,
            self._actor_continuity_score,
            self._target_continuity_score,
            self._pattern_match_score,
        ]

    def infer_causality(
        self, source_event: Dict[str, Any], target_event: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Infer causal relationship strength between two events.

        Returns:
            Tuple of (score, evidence_list).
        """
        total_score = 0.0
        evidence: List[str] = []
        weights = [0.35, 0.25, 0.20, 0.20]

        for rule, weight in zip(self._causal_rules, weights):
            score, ev = rule(source_event, target_event)
            total_score += score * weight
            evidence.extend(ev)

        return min(total_score, 1.0), evidence

    def _temporal_proximity_score(
        self, source: Dict[str, Any], target: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Score based on temporal proximity."""
        evidence: List[str] = []
        src_ts = source.get("timestamp")
        tgt_ts = target.get("timestamp")
        if not src_ts or not tgt_ts:
            return 0.0, evidence

        try:
            src_dt = datetime.fromisoformat(src_ts) if isinstance(src_ts, str) else src_ts
            tgt_dt = datetime.fromisoformat(tgt_ts) if isinstance(tgt_ts, str) else tgt_ts
        except (ValueError, TypeError):
            return 0.0, evidence

        delta = abs((tgt_dt - src_dt).total_seconds())
        if delta <= self._temporal_window:
            score = 1.0 - (delta / self._temporal_window)
            evidence.append(f"Temporal proximity: {delta:.0f}s within {self._temporal_window}s window")
            return score, evidence
        return 0.0, evidence

    def _actor_continuity_score(
        self, source: Dict[str, Any], target: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Score based on actor continuity."""
        evidence: List[str] = []
        src_actor = source.get("actor", "")
        tgt_actor = target.get("actor", "")
        if not src_actor or not tgt_actor:
            return 0.0, evidence

        if src_actor == tgt_actor:
            evidence.append(f"Same actor: {src_actor}")
            return 1.0, evidence
        return 0.0, evidence

    def _target_continuity_score(
        self, source: Dict[str, Any], target: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Score based on target/resource continuity."""
        evidence: List[str] = []
        src_target = source.get("target", "")
        tgt_target = target.get("target", "")
        if not src_target or not tgt_target:
            return 0.0, evidence

        if src_target == tgt_target:
            evidence.append(f"Same target resource: {src_target}")
            return 1.0, evidence

        # Check if targets share a common prefix (e.g., same host)
        src_parts = src_target.split("/")
        tgt_parts = tgt_target.split("/")
        common = 0
        for sp, tp in zip(src_parts, tgt_parts):
            if sp == tp:
                common += 1
            else:
                break
        if common > 0:
            score = common / max(len(src_parts), len(tgt_parts))
            evidence.append(f"Shared target prefix ({common} levels): {'/'.join(src_parts[:common])}")
            return score, evidence
        return 0.0, evidence

    def _pattern_match_score(
        self, source: Dict[str, Any], target: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Score based on known attack pattern matching."""
        evidence: List[str] = []
        src_type = source.get("event_type", "")
        tgt_type = target.get("event_type", "")

        for phase, patterns in self.ACTOR_PATTERNS.items():
            for pat_src, pat_tgt in patterns:
                if src_type == pat_src and tgt_type == pat_tgt:
                    evidence.append(f"Attack pattern match: {phase} ({pat_src} -> {pat_tgt})")
                    return 1.0, evidence
        return 0.0, evidence

    def analyze_graph(
        self, graph: "AttackGraph"
    ) -> List[Tuple[str, str, float, List[str]]]:
        """Analyze all node pairs in the graph for causal relationships.

        Returns:
            List of (source_id, target_id, score, evidence) tuples.
        """
        results: List[Tuple[str, str, float, List[str]]] = []
        node_list = list(graph.nodes.values())

        for i, src_node in enumerate(node_list):
            for tgt_node in node_list[i + 1:]:
                src_event = src_node.properties.get("raw_event", src_node.properties)
                tgt_event = tgt_node.properties.get("raw_event", tgt_node.properties)

                score_fwd, ev_fwd = self.infer_causality(src_event, tgt_event)
                score_bwd, ev_bwd = self.infer_causality(tgt_event, src_event)

                if score_fwd > 0.3:
                    results.append((src_node.node_id, tgt_node.node_id, score_fwd, ev_fwd))
                elif score_bwd > 0.3:
                    results.append((tgt_node.node_id, src_node.node_id, score_bwd, ev_bwd))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def add_causal_edges(
        self, graph: "AttackGraph", threshold: float = 0.4
    ) -> int:
        """Add inferred causal edges to the graph. Returns count of new edges."""
        causal_relations = self.analyze_graph(graph)
        added = 0
        for src_id, tgt_id, score, evidence in causal_relations:
            if score >= threshold:
                # Check if edge already exists
                existing = graph.get_edge(src_id, tgt_id)
                if existing is None:
                    edge = Edge(
                        source_id=src_id,
                        target_id=tgt_id,
                        edge_type=EdgeType.LEADS_TO,
                        weight=score,
                        causal_evidence=evidence,
                    )
                    graph.add_edge(edge)
                    added += 1
        return added


# ---------------------------------------------------------------------------
# Path Enumerator
# ---------------------------------------------------------------------------

class PathEnumerator:
    """Enumerates attack paths through the graph."""

    def __init__(
        self,
        max_path_length: int = 15,
        max_paths: int = 1000,
        min_severity: float = 0.0,
    ) -> None:
        self._max_length = max_path_length
        self._max_paths = max_paths
        self._min_severity = min_severity

    def find_all_paths(
        self,
        graph: "AttackGraph",
        start_id: Optional[str] = None,
        end_id: Optional[str] = None,
    ) -> List[AttackPath]:
        """Find all paths between start and end nodes.

        If start_id or end_id is None, uses entry/exfiltration nodes.
        """
        if start_id is None:
            start_id = self._find_entry_node(graph)
        if end_id is None:
            end_id = self._find_exfil_node(graph)

        if start_id is None or end_id is None:
            return []

        paths: List[AttackPath] = []
        visited: Set[str] = set()

        self._dfs_paths(
            graph, start_id, end_id, visited, [], [], paths
        )

        # Filter by severity and limit
        filtered = []
        for path in paths:
            path.compute_metrics()
            if path.total_severity >= self._min_severity:
                filtered.append(path)

        filtered.sort(key=lambda p: p.total_severity, reverse=True)
        return filtered[: self._max_paths]

    def _dfs_paths(
        self,
        graph: "AttackGraph",
        current: str,
        target: str,
        visited: Set[str],
        current_nodes: List[Node],
        current_edges: List[Edge],
        results: List[AttackPath],
    ) -> None:
        """DFS-based path enumeration."""
        if len(current_nodes) >= self._max_length:
            return

        node = graph.nodes.get(current)
        if node is None:
            return

        visited.add(current)
        current_nodes.append(node)

        if current == target and len(current_nodes) > 1:
            path = AttackPath(
                nodes=list(current_nodes),
                edges=list(current_edges),
            )
            path.compute_metrics()
            results.append(path)
        else:
            for neighbor_id in graph.get_successors(current):
                if neighbor_id not in visited:
                    edge = graph.get_edge(current, neighbor_id)
                    if edge is not None:
                        current_edges.append(edge)
                        self._dfs_paths(
                            graph, neighbor_id, target, visited,
                            current_nodes, current_edges, results,
                        )
                        current_edges.pop()

        current_nodes.pop()
        visited.discard(current)

    def find_shortest_path(
        self, graph: "AttackGraph", start_id: str, end_id: str
    ) -> Optional[AttackPath]:
        """Find the shortest path using BFS."""
        if start_id not in graph.nodes or end_id not in graph.nodes:
            return None

        queue: deque = deque()
        queue.append((start_id, [start_id], []))
        visited: Set[str] = {start_id}

        while queue:
            current, node_ids, edge_keys = queue.popleft()

            if current == end_id:
                nodes = [graph.nodes[nid] for nid in node_ids if nid in graph.nodes]
                edges = [graph.edges[ek] for ek in edge_keys if ek in graph.edges]
                path = AttackPath(nodes=nodes, edges=edges)
                path.compute_metrics()
                return path

            for neighbor_id in graph.get_successors(current):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    edge_key = graph.get_edge_key(current, neighbor_id)
                    if edge_key:
                        queue.append((
                            neighbor_id,
                            node_ids + [neighbor_id],
                            edge_keys + [edge_key],
                        ))

        return None

    def find_most_dangerous_path(
        self, graph: "AttackGraph", start_id: str, end_id: str
    ) -> Optional[AttackPath]:
        """Find the path with highest severity using Dijkstra-like search."""
        if start_id not in graph.nodes or end_id not in graph.nodes:
            return None

        # Use negative severity as distance (max-heap via min-heap with negation)
        heap: List[Tuple[float, int, str, List[str], List[str]]] = []
        counter = 0
        heapq.heappush(heap, (0.0, counter, start_id, [start_id], []))
        best_scores: Dict[str, float] = {start_id: 0.0}

        while heap:
            neg_severity, _, current, node_ids, edge_keys = heapq.heappop(heap)
            current_severity = -neg_severity

            if current == end_id:
                nodes = [graph.nodes[nid] for nid in node_ids if nid in graph.nodes]
                edges = [graph.edges[ek] for ek in edge_keys if ek in graph.edges]
                path = AttackPath(nodes=nodes, edges=edges)
                path.compute_metrics()
                return path

            if current_severity < best_scores.get(current, float("-inf")):
                continue

            for neighbor_id in graph.get_successors(current):
                neighbor_node = graph.nodes.get(neighbor_id)
                if neighbor_node is None:
                    continue

                new_severity = current_severity + neighbor_node.severity
                if new_severity > best_scores.get(neighbor_id, float("-inf")):
                    best_scores[neighbor_id] = new_severity
                    edge_key = graph.get_edge_key(current, neighbor_id)
                    counter += 1
                    heapq.heappush(heap, (
                        -new_severity, counter, neighbor_id,
                        node_ids + [neighbor_id],
                        edge_keys + [edge_key] if edge_key else edge_keys,
                    ))

        return None

    def _find_entry_node(self, graph: "AttackGraph") -> Optional[str]:
        """Find the most likely entry node."""
        entry_types = {NodeType.ENTRY, NodeType.VULNERABILITY}
        for node in graph.nodes.values():
            if node.node_type in entry_types:
                return node.node_id
        # Fallback: node with no predecessors
        for node_id in graph.nodes:
            if not graph.get_predecessors(node_id):
                return node_id
        return None

    def _find_exfil_node(self, graph: "AttackGraph") -> Optional[str]:
        """Find the most likely exfiltration node."""
        for node in graph.nodes.values():
            if node.node_type == NodeType.EXFILTRATION:
                return node.node_id
        # Fallback: node with no successors
        for node_id in graph.nodes:
            if not graph.get_successors(node_id):
                return node_id
        return None


# ---------------------------------------------------------------------------
# Critical Path Analyzer
# ---------------------------------------------------------------------------

class CriticalPathAnalyzer:
    """Analyzes critical paths in the attack graph."""

    def __init__(self) -> None:
        self._centrality_cache: Dict[str, float] = {}

    def compute_betweenness_centrality(
        self, graph: "AttackGraph"
    ) -> Dict[str, float]:
        """Compute betweenness centrality for all nodes using Brandes' algorithm."""
        centrality: Dict[str, float] = {nid: 0.0 for nid in graph.nodes}
        node_ids = list(graph.nodes.keys())

        for source_id in node_ids:
            # BFS from source
            stack: List[str] = []
            predecessors: Dict[str, List[str]] = {nid: [] for nid in node_ids}
            sigma: Dict[str, int] = {nid: 0 for nid in node_ids}
            sigma[source_id] = 1
            distance: Dict[str, float] = {nid: -1.0 for nid in node_ids}
            distance[source_id] = 0.0
            queue: deque = deque([source_id])

            while queue:
                current = queue.popleft()
                stack.append(current)
                for neighbor_id in graph.get_successors(current):
                    if distance[neighbor_id] < 0:
                        queue.append(neighbor_id)
                        distance[neighbor_id] = distance[current] + 1.0
                    if distance[neighbor_id] == distance[current] + 1.0:
                        sigma[neighbor_id] += sigma[current]
                        predecessors[neighbor_id].append(current)

            # Accumulation
            delta: Dict[str, float] = {nid: 0.0 for nid in node_ids}
            while stack:
                node_id = stack.pop()
                for pred_id in predecessors[node_id]:
                    if sigma[node_id] > 0:
                        delta[pred_id] += (
                            sigma[pred_id] / sigma[node_id] * (1.0 + delta[node_id])
                        )
                if node_id != source_id:
                    centrality[node_id] += delta[node_id]

        # Normalize
        n = len(node_ids)
        if n > 2:
            norm = (n - 1) * (n - 2)
            for nid in centrality:
                centrality[nid] /= norm

        self._centrality_cache = centrality
        return centrality

    def compute_degree_centrality(
        self, graph: "AttackGraph"
    ) -> Dict[str, float]:
        """Compute degree centrality (in-degree + out-degree)."""
        centrality: Dict[str, float] = {}
        n = max(len(graph.nodes), 1)

        for node_id in graph.nodes:
            in_deg = len(graph.get_predecessors(node_id))
            out_deg = len(graph.get_successors(node_id))
            centrality[node_id] = (in_deg + out_deg) / (2 * (n - 1)) if n > 1 else 0.0

        return centrality

    def compute_closeness_centrality(
        self, graph: "AttackGraph"
    ) -> Dict[str, float]:
        """Compute closeness centrality using BFS distances."""
        centrality: Dict[str, float] = {}
        node_ids = list(graph.nodes.keys())

        for source_id in node_ids:
            distances = self._bfs_distances(graph, source_id)
            reachable = [d for d in distances.values() if d > 0 and d < float("inf")]
            if reachable:
                avg_dist = sum(reachable) / len(reachable)
                centrality[source_id] = 1.0 / avg_dist if avg_dist > 0 else 0.0
            else:
                centrality[source_id] = 0.0

        return centrality

    def _bfs_distances(
        self, graph: "AttackGraph", source_id: str
    ) -> Dict[str, float]:
        """Compute BFS distances from source to all reachable nodes."""
        distances: Dict[str, float] = {nid: float("inf") for nid in graph.nodes}
        distances[source_id] = 0.0
        queue: deque = deque([source_id])

        while queue:
            current = queue.popleft()
            for neighbor_id in graph.get_successors(current):
                if distances[neighbor_id] == float("inf"):
                    distances[neighbor_id] = distances[current] + 1.0
                    queue.append(neighbor_id)

        return distances

    def find_critical_nodes(
        self, graph: "AttackGraph", top_k: int = 10
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Find the most critical nodes based on combined centrality metrics."""
        betweenness = self.compute_betweenness_centrality(graph)
        degree = self.compute_degree_centrality(graph)
        closeness = self.compute_closeness_centrality(graph)

        combined: Dict[str, float] = {}
        for nid in graph.nodes:
            combined[nid] = (
                betweenness.get(nid, 0.0) * 0.5
                + degree.get(nid, 0.0) * 0.3
                + closeness.get(nid, 0.0) * 0.2
            )

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for nid, score in ranked:
            results.append((
                nid,
                score,
                {
                    "betweenness": betweenness.get(nid, 0.0),
                    "degree": degree.get(nid, 0.0),
                    "closeness": closeness.get(nid, 0.0),
                },
            ))
        return results

    def find_bridges(self, graph: "AttackGraph") -> List[Tuple[str, str]]:
        """Find bridge edges whose removal would disconnect the graph."""
        bridges: List[Tuple[str, str]] = []
        visited: Set[str] = set()
        disc: Dict[str, int] = {}
        low: Dict[str, int] = {}
        timer = [0]

        def dfs(u: str, parent: Optional[str]) -> None:
            visited.add(u)
            disc[u] = low[u] = timer[0]
            timer[0] += 1

            for v in graph.get_successors(u):
                if v not in visited:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent:
                    low[u] = min(low[u], disc[v])

        for node_id in graph.nodes:
            if node_id not in visited:
                dfs(node_id, None)

        return bridges

    def find_attack_surface(
        self, graph: "AttackGraph"
    ) -> List[Tuple[str, NodeType, float]]:
        """Identify the attack surface (entry points and their severity)."""
        surface: List[Tuple[str, NodeType, float]] = []
        entry_types = {
            NodeType.ENTRY,
            NodeType.VULNERABILITY,
            NodeType.RESOURCE,
        }

        for node in graph.nodes.values():
            if node.node_type in entry_types:
                predecessors = graph.get_predecessors(node.node_id)
                # Entry points typically have no predecessors or only defense predecessors
                is_entry = len(predecessors) == 0
                if not is_entry:
                    for pred_id in predecessors:
                        pred_node = graph.nodes.get(pred_id)
                        if pred_node and pred_node.node_type != NodeType.DEFENSE:
                            is_entry = False
                            break
                        else:
                            is_entry = True

                if is_entry:
                    surface.append((node.node_id, node.node_type, node.severity))

        surface.sort(key=lambda x: x[2], reverse=True)
        return surface


# ---------------------------------------------------------------------------
# Graph Renderer
# ---------------------------------------------------------------------------

class GraphRenderer:
    """Renders attack graphs as ASCII art."""

    BOX_CHARS = {
        "tl": "+", "tr": "+", "bl": "+", "br": "+",
        "h": "-", "v": "|",
    }

    NODE_TYPE_SYMBOLS: Dict[NodeType, str] = {
        NodeType.ENTRY: "[IN]",
        NodeType.VULNERABILITY: "[VU]",
        NodeType.ACTION: "[AC]",
        NodeType.RESOURCE: "[RS]",
        NodeType.EXFILTRATION: "[EX]",
        NodeType.PRIVILEGE_ESCALATION: "[PE]",
        NodeType.LATERAL_MOVEMENT: "[LM]",
        NodeType.PERSISTENCE: "[PS]",
        NodeType.DEFENSE: "[DF]",
        NodeType.UNKNOWN: "[??]",
    }

    SEVERITY_BAR_CHARS = "░▒▓█"

    def __init__(self, max_width: int = 120, max_height: int = 40) -> None:
        self._max_width = max_width
        self._max_height = max_height

    def render_node(self, node: Node, detailed: bool = False) -> str:
        """Render a single node as ASCII art."""
        symbol = self.NODE_TYPE_SYMBOLS.get(node.node_type, "[??]")
        sev_bar = self._severity_bar(node.severity)

        lines = [
            f"  {self.BOX_CHARS['tl']}{self.BOX_CHARS['h'] * 40}{self.BOX_CHARS['tr']}",
            f"  {self.BOX_CHARS['v']} {symbol} {node.label[:32]:<32s} {self.BOX_CHARS['v']}",
            f"  {self.BOX_CHARS['v']} ID: {node.node_id:<12s} Sev: {sev_bar} {node.severity:.2f} {self.BOX_CHARS['v']}",
        ]

        if detailed:
            ts_str = node.timestamp.strftime("%Y-%m-%d %H:%M:%S") if node.timestamp else "N/A"
            lines.append(
                f"  {self.BOX_CHARS['v']} Time: {ts_str:<20s} Conf: {node.confidence:.2f} {self.BOX_CHARS['v']}"
            )
            if node.properties.get("actor"):
                lines.append(
                    f"  {self.BOX_CHARS['v']} Actor: {str(node.properties['actor'])[:34]:<34s} {self.BOX_CHARS['v']}"
                )

        lines.append(f"  {self.BOX_CHARS['bl']}{self.BOX_CHARS['h'] * 40}{self.BOX_CHARS['br']}")
        return "\n".join(lines)

    def render_edge(self, edge: Edge) -> str:
        """Render a single edge as ASCII art."""
        arrow_map = {
            EdgeType.ENABLES: "==>",
            EdgeType.EXPLOITS: "**>",
            EdgeType.LEADS_TO: "-->",
            EdgeType.DEPENDS_ON: "-+>",
            EdgeType.CONTAINS: "==>",
            EdgeType.COMMUNICATES: "~~>",
            EdgeType.EVADER: "!!>",
            EdgeType.UNKNOWN: "-->",
        }
        arrow = arrow_map.get(edge.edge_type, "-->")
        return f"    {arrow} [{edge.edge_type.value}] (w={edge.weight:.2f})"

    def render_path(self, path: AttackPath, title: str = "Attack Path") -> str:
        """Render an attack path as ASCII art."""
        lines = [
            f"{'=' * 50}",
            f"  {title}",
            f"  Length: {path.length} | Severity: {path.total_severity:.2f} | Confidence: {path.total_confidence:.2f}",
            f"{'=' * 50}",
        ]

        for i, node in enumerate(path.nodes):
            lines.append(self.render_node(node))
            if i < len(path.edges):
                lines.append(self.render_edge(path.edges[i]))

        return "\n".join(lines)

    def render_graph_summary(self, graph: "AttackGraph") -> str:
        """Render a summary of the entire attack graph."""
        type_counts: Dict[NodeType, int] = defaultdict(int)
        total_severity = 0.0
        high_sev_count = 0

        for node in graph.nodes.values():
            type_counts[node.node_type] += 1
            total_severity += node.severity
            if node.severity >= 0.7:
                high_sev_count += 1

        avg_severity = total_severity / max(len(graph.nodes), 1)

        lines = [
            f"{'#' * 60}",
            f"#  ATTACK GRAPH SUMMARY",
            f"{'#' * 60}",
            f"",
            f"  Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}",
            f"  Avg Severity: {avg_severity:.2f} | High Severity Nodes: {high_sev_count}",
            f"",
            f"  Node Types:",
        ]

        for ntype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            symbol = self.NODE_TYPE_SYMBOLS.get(ntype, "[??]")
            bar = self.BOX_CHARS["h"] * min(count * 2, 40)
            lines.append(f"    {symbol} {ntype.value:<25s} : {count:>4d}  {bar}")

        # Entry points
        lines.append(f"")
        lines.append(f"  Entry Points:")
        entry_nodes = [
            n for n in graph.nodes.values() if n.node_type == NodeType.ENTRY
        ]
        for node in entry_nodes[:5]:
            lines.append(f"    - {node.label} (sev={node.severity:.2f})")

        # Exfiltration points
        lines.append(f"")
        lines.append(f"  Exfiltration Points:")
        exfil_nodes = [
            n for n in graph.nodes.values() if n.node_type == NodeType.EXFILTRATION
        ]
        for node in exfil_nodes[:5]:
            lines.append(f"    - {node.label} (sev={node.severity:.2f})")

        lines.append(f"{'#' * 60}")
        return "\n".join(lines)

    def render_ascii_graph(
        self, graph: "AttackGraph", max_nodes: int = 30
    ) -> str:
        """Render the graph structure as ASCII art with connections."""
        lines: List[str] = []
        rendered_nodes: Set[str] = set()
        node_list = list(graph.nodes.values())[:max_nodes]

        # Build a simple tree-like layout using BFS
        entry_id = self._find_root(graph)
        if entry_id is None:
            return "  (empty graph)"

        layout = self._layout_bfs(graph, entry_id, max_nodes)

        for level_nodes in layout:
            level_line = ""
            for node_id in level_nodes:
                if node_id in rendered_nodes:
                    continue
                rendered_nodes.add(node_id)
                node = graph.nodes[node_id]
                symbol = self.NODE_TYPE_SYMBOLS.get(node.node_type, "[??]")
                label = node.label[:15]
                level_line += f"{symbol} {label:<18s}"
            if level_line.strip():
                lines.append(f"  {level_line}")

            # Draw connections
            conn_line = ""
            for node_id in level_nodes:
                successors = graph.get_successors(node_id)
                next_level_ids = set()
                if layout.index(level_nodes) + 1 < len(layout):
                    next_level_ids = set(layout[layout.index(level_nodes) + 1])
                connected = [s for s in successors if s in next_level_ids]
                if connected:
                    conn_line += f"    |{'':>17s}" * min(len(connected), 3)
            if conn_line.strip():
                lines.append(f"  {conn_line}")

        return "\n".join(lines) if lines else "  (empty graph)"

    def _find_root(self, graph: "AttackGraph") -> Optional[str]:
        """Find the root node for layout."""
        for node in graph.nodes.values():
            if node.node_type == NodeType.ENTRY:
                return node.node_id
        for node_id in graph.nodes:
            if not graph.get_predecessors(node_id):
                return node_id
        return next(iter(graph.nodes), None)

    def _layout_bfs(
        self, graph: "AttackGraph", root: str, max_nodes: int
    ) -> List[List[str]]:
        """BFS-based level layout."""
        levels: List[List[str]] = []
        visited: Set[str] = set()
        queue: deque = deque([root])
        visited.add(root)
        count = 0

        while queue and count < max_nodes:
            level_size = len(queue)
            level: List[str] = []
            for _ in range(level_size):
                node_id = queue.popleft()
                level.append(node_id)
                count += 1
                for neighbor_id in graph.get_successors(node_id):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            levels.append(level)

        return levels

    def _severity_bar(self, severity: float, width: int = 5) -> str:
        """Create a severity bar visualization."""
        filled = int(severity * width)
        empty = width - filled
        return self.SEVERITY_BAR_CHARS[3] * filled + self.SEVERITY_BAR_CHARS[0] * empty

    def render_mermaid(self, graph: "AttackGraph") -> str:
        """Render the graph in Mermaid diagram format."""
        lines = ["graph TD"]

        node_id_map: Dict[str, str] = {}
        for i, (nid, node) in enumerate(graph.nodes.items()):
            safe_id = f"N{i}"
            node_id_map[nid] = safe_id
            label = node.label.replace('"', "'").replace("\n", " ")[:30]
            lines.append(f'    {safe_id}["{node.node_type.value}: {label}"]')

        lines.append("")
        for edge in graph.edges.values():
            src = node_id_map.get(edge.source_id, "?")
            tgt = node_id_map.get(edge.target_id, "?")
            label = edge.edge_type.value
            lines.append(f'    {src} -->|"{label}"| {tgt}')

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Attack Graph (Main Class)
# ---------------------------------------------------------------------------

class AttackGraph:
    """Main attack graph class combining all analysis capabilities."""

    def __init__(
        self,
        nodes: Optional[Dict[str, Node]] = None,
        edges: Optional[Dict[str, Edge]] = None,
        adjacency: Optional[Dict[str, List[str]]] = None,
        reverse_adjacency: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.nodes: Dict[str, Node] = nodes or {}
        self.edges: Dict[str, Edge] = edges or {}
        self._adjacency: Dict[str, List[str]] = adjacency or defaultdict(list)
        self._reverse_adjacency: Dict[str, List[str]] = reverse_adjacency or defaultdict(list)
        self._builder = GraphBuilder()
        self._causal_analyzer = CausalAnalyzer()
        self._path_enumerator = PathEnumerator()
        self._critical_analyzer = CriticalPathAnalyzer()
        self._renderer = GraphRenderer()

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge to the graph."""
        key = f"{edge.source_id}->{edge.target_id}:{edge.edge_type.value}"
        self.edges[key] = edge
        if edge.target_id not in self._adjacency.get(edge.source_id, []):
            self._adjacency[edge.source_id].append(edge.target_id)
        if edge.source_id not in self._reverse_adjacency.get(edge.target_id, []):
            self._reverse_adjacency[edge.target_id].append(edge.source_id)

    def get_successors(self, node_id: str) -> List[str]:
        """Get successor node IDs."""
        return self._adjacency.get(node_id, [])

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs."""
        return self._reverse_adjacency.get(node_id, [])

    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        """Get an edge between two nodes."""
        for edge in self.edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None

    def get_edge_key(self, source_id: str, target_id: str) -> Optional[str]:
        """Get the edge key between two nodes."""
        for key, edge in self.edges.items():
            if edge.source_id == source_id and edge.target_id == target_id:
                return key
        return None

    def infer_causal_edges(self, threshold: float = 0.4) -> int:
        """Infer and add causal edges to the graph."""
        return self._causal_analyzer.add_causal_edges(self, threshold)

    def find_attack_paths(
        self,
        start_id: Optional[str] = None,
        end_id: Optional[str] = None,
    ) -> List[AttackPath]:
        """Find all attack paths in the graph."""
        return self._path_enumerator.find_all_paths(self, start_id, end_id)

    def find_shortest_attack_path(
        self, start_id: str, end_id: str
    ) -> Optional[AttackPath]:
        """Find the shortest attack path."""
        return self._path_enumerator.find_shortest_path(self, start_id, end_id)

    def find_most_dangerous_path(
        self, start_id: str, end_id: str
    ) -> Optional[AttackPath]:
        """Find the most dangerous attack path."""
        return self._path_enumerator.find_most_dangerous_path(self, start_id, end_id)

    def find_critical_nodes(self, top_k: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
        """Find the most critical nodes."""
        return self._critical_analyzer.find_critical_nodes(self, top_k)

    def find_bridges(self) -> List[Tuple[str, str]]:
        """Find bridge edges."""
        return self._critical_analyzer.find_bridges(self)

    def find_attack_surface(self) -> List[Tuple[str, NodeType, float]]:
        """Identify the attack surface."""
        return self._critical_analyzer.find_attack_surface(self)

    def render_summary(self) -> str:
        """Render a graph summary."""
        return self._renderer.render_graph_summary(self)

    def render_ascii(self, max_nodes: int = 30) -> str:
        """Render the graph as ASCII art."""
        return self._renderer.render_ascii_graph(self, max_nodes)

    def render_path(self, path: AttackPath, title: str = "Attack Path") -> str:
        """Render a specific attack path."""
        return self._renderer.render_path(path, title)

    def render_mermaid(self) -> str:
        """Render as Mermaid diagram."""
        return self._renderer.render_mermaid(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
            "metadata": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the graph to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_events(cls, events: List[Dict[str, Any]]) -> "AttackGraph":
        """Build an attack graph from audit events."""
        builder = GraphBuilder()
        return builder.build_from_events(events)

    def compute_graph_hash(self) -> str:
        """Compute a hash of the graph structure for integrity checking."""
        node_str = ",".join(
            f"{nid}:{n.node_type.value}:{n.severity:.2f}"
            for nid, n in sorted(self.nodes.items())
        )
        edge_str = ",".join(
            f"{e.source_id}->{e.target_id}:{e.edge_type.value}"
            for e in sorted(self.edges.values(), key=lambda x: x.source_id)
        )
        content = f"{node_str}|{edge_str}"
        return hashlib.sha256(content.encode()).hexdigest()
