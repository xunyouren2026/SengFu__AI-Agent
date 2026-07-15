"""
Knowledge Graph Module - AGI Unified Framework
Comprehensive knowledge graph with real algorithms, pure Python implementation.

Components:
- Entity / Relation data structures
- KnowledgeGraph core (adjacency list, CRUD, traversal, statistics)
- GraphEmbedding (TransE, TransH, TransR, DistMult, ComplEx, RotatE)
- GraphAlgorithm (PageRank, HITS, Community, Centrality, ShortestPath)
- Reasoning (RuleMining, LinkPrediction, PathReasoning, SubgraphMatching)
- TemporalKnowledgeGraph (time-aware reasoning)
- KnowledgeGraphManager (high-level build, query, merge, completion)
"""

from __future__ import annotations

import math
import random
import uuid
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graph operations."""
    embedding_dim: int = 100
    learning_rate: float = 0.01
    margin: float = 1.0
    negative_samples: int = 5
    max_epochs: int = 100
    batch_size: int = 256
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 100
    hits_iterations: int = 100
    community_resolution: float = 1.0
    temporal_decay_rate: float = 0.01
    seed: int = 42


# ============================================================
# 2. Entity
# ============================================================

@dataclass
class Entity:
    """Knowledge graph entity node."""
    id: str
    type: str = "generic"
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.id == other.id

    def __repr__(self) -> str:
        return f"Entity(id={self.id!r}, type={self.type!r})"


# ============================================================
# 3. Relation
# ============================================================

@dataclass
class Relation:
    """Knowledge graph relation edge."""
    subject: str
    predicate: str
    object: str
    weight: float = 1.0
    timestamp: Optional[float] = None

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Relation) and self.key == other.key

    def __repr__(self) -> str:
        return f"Relation({self.subject!r} --{self.predicate!r}--> {self.object!r})"


# ============================================================
# 4. KnowledgeGraph - Core data structure
# ============================================================

class KnowledgeGraph:
    """Core knowledge graph with adjacency list representation."""

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        self.config = config or KnowledgeGraphConfig()
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[Tuple[str, str, str], Relation] = {}
        self._adj_out: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._adj_in: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._predicate_index: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    # --- Entity CRUD ---

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        edges_to_remove = [
            (s, p, o) for s, p, o in self._relations
            if s == entity_id or o == entity_id
        ]
        for key in edges_to_remove:
            self.remove_relation(*key)

    def entities(self) -> List[Entity]:
        return list(self._entities.values())

    def entity_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for e in self._entities.values():
            counts[e.type] += 1
        return dict(counts)

    # --- Relation CRUD ---

    def add_relation(self, relation: Relation) -> None:
        key = relation.key
        self._relations[key] = relation
        self._adj_out[relation.subject].append(
            (relation.object, relation.predicate, relation.weight)
        )
        self._adj_in[relation.object].append(
            (relation.subject, relation.predicate, relation.weight)
        )
        self._predicate_index[relation.predicate].append(
            (relation.subject, relation.object, relation.weight)
        )

    def get_relation(self, subject: str, predicate: str, obj: str) -> Optional[Relation]:
        return self._relations.get((subject, predicate, obj))

    def remove_relation(self, subject: str, predicate: str, obj: str) -> None:
        key = (subject, predicate, obj)
        self._relations.pop(key, None)
        self._adj_out[subject] = [
            (o, p, w) for o, p, w in self._adj_out[subject]
            if not (o == obj and p == predicate)
        ]
        self._adj_in[obj] = [
            (s, p, w) for s, p, w in self._adj_in[obj]
            if not (s == subject and p == predicate)
        ]
        self._predicate_index[predicate] = [
            (s, o, w) for s, o, w in self._predicate_index[predicate]
            if not (s == subject and o == obj)
        ]

    def relations(self) -> List[Relation]:
        return list(self._relations.values())

    def neighbors(self, entity_id: str) -> List[Tuple[str, str, float]]:
        return self._adj_out.get(entity_id, [])

    def predecessors(self, entity_id: str) -> List[Tuple[str, str, float]]:
        return self._adj_in.get(entity_id, [])

    # --- Graph traversal ---

    def bfs(self, start: str, max_depth: int = 10) -> List[str]:
        visited: List[str] = []
        seen: Set[str] = {start}
        queue: deque = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            visited.append(node)
            if depth >= max_depth:
                continue
            for neighbor, _, _ in self._adj_out.get(node, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return visited

    def dfs(self, start: str, max_depth: int = 10) -> List[str]:
        visited: List[str] = []
        seen: Set[str] = set()
        stack: list = [(start, 0)]
        while stack:
            node, depth = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            visited.append(node)
            if depth >= max_depth:
                continue
            for neighbor, _, _ in reversed(self._adj_out.get(node, [])):
                if neighbor not in seen:
                    stack.append((neighbor, depth + 1))
        return visited

    def find_paths(self, source: str, target: str, max_depth: int = 6) -> List[List[str]]:
        """Find all simple paths between source and target using DFS."""
        results: List[List[str]] = []
        stack: list = [(source, [source])]
        while stack:
            node, path = stack.pop()
            if node == target and len(path) > 1:
                results.append(path)
                continue
            if len(path) > max_depth:
                continue
            for neighbor, _, _ in self._adj_out.get(node, []):
                if neighbor not in set(path):
                    stack.append((neighbor, path + [neighbor]))
        return results

    # --- Subgraph extraction ---

    def extract_subgraph(self, entity_ids: Set[str]) -> "KnowledgeGraph":
        sub = KnowledgeGraph(self.config)
        for eid in entity_ids:
            if eid in self._entities:
                sub.add_entity(self._entities[eid])
        for rel in self._relations.values():
            if rel.subject in entity_ids and rel.object in entity_ids:
                sub.add_relation(rel)
        return sub

    def ego_graph(self, center: str, radius: int = 1) -> "KnowledgeGraph":
        ids = set(self.bfs(center, max_depth=radius))
        return self.extract_subgraph(ids)

    # --- Graph statistics ---

    def num_entities(self) -> int:
        return len(self._entities)

    def num_relations(self) -> int:
        return len(self._relations)

    def degree_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for eid in self._entities:
            out_deg = len(self._adj_out.get(eid, []))
            in_deg = len(self._adj_in.get(eid, []))
            total = out_deg + in_deg
            dist[eid] = total
        return dist

    def avg_degree(self) -> float:
        if not self._entities:
            return 0.0
        return sum(self.degree_distribution().values()) / len(self._entities)

    def density(self) -> float:
        n = len(self._entities)
        if n < 2:
            return 0.0
        m = len(self._relations)
        return 2.0 * m / (n * (n - 1))

    def clustering_coefficient(self, entity_id: str) -> float:
        neighbors_out = {o for o, _, _ in self._adj_out.get(entity_id, [])}
        neighbors_in = {s for s, _, _ in self._adj_in.get(entity_id, [])}
        neighbors_all = neighbors_out | neighbors_in
        k = len(neighbors_all)
        if k < 2:
            return 0.0
        triangles = 0
        for n1 in neighbors_all:
            for n2 in neighbors_all:
                if n1 >= n2:
                    continue
                if any(o == n2 for o, _, _ in self._adj_out.get(n1, [])):
                    triangles += 1
        possible = k * (k - 1) / 2.0
        return triangles / possible if possible > 0 else 0.0

    def avg_clustering_coefficient(self) -> float:
        if not self._entities:
            return 0.0
        total = sum(self.clustering_coefficient(eid) for eid in self._entities)
        return total / len(self._entities)

    def avg_path_length(self, sample_size: int = 100) -> float:
        eids = list(self._entities.keys())
        if len(eids) < 2:
            return 0.0
        sampled = random.sample(eids, min(sample_size, len(eids)))
        total = 0.0
        count = 0
        for i, s in enumerate(sampled):
            for t in sampled[i + 1:]:
                dist = self._bfs_shortest_distance(s, t)
                if dist is not None:
                    total += dist
                    count += 1
        return total / count if count > 0 else 0.0

    def _bfs_shortest_distance(self, source: str, target: str) -> Optional[int]:
        if source == target:
            return 0
        visited: Set[str] = {source}
        queue: deque = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            for neighbor, _, _ in self._adj_out.get(node, []):
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None


# ============================================================
# 5. GraphEmbedding - KGE algorithms
# ============================================================

class GraphEmbedding:
    """Knowledge graph embedding algorithms with negative sampling training."""

    def __init__(self, graph: KnowledgeGraph, config: Optional[KnowledgeGraphConfig] = None):
        self.graph = graph
        self.config = config or graph.config
        self.entity_embeddings: Dict[str, List[float]] = {}
        self.relation_embeddings: Dict[str, List[float]] = {}
        self._entity_ids: List[str] = []
        self._relation_types: List[str] = []
        self._triples: List[Tuple[str, str, str]] = []
        self._rng = random.Random(self.config.seed)

    def _init_embeddings(self) -> None:
        dim = self.config.embedding_dim
        self._entity_ids = list(self.graph._entities.keys())
        self._relation_types = list(self.graph._predicate_index.keys())
        self._triples = [(r.subject, r.predicate, r.object) for r in self.graph.relations()]
        bound = 1.0 / math.sqrt(dim)
        for eid in self._entity_ids:
            self.entity_embeddings[eid] = [
                self._rng.uniform(-bound, bound) for _ in range(dim)
            ]
        for pid in self._relation_types:
            self.relation_embeddings[pid] = [
                self._rng.uniform(-bound, bound) for _ in range(dim)
            ]

    def _sample_negative(self, h: str, r: str, t: str) -> Tuple[str, str, str]:
        existing = set(self._triples)
        for _ in range(100):
            nh = self._rng.choice(self._entity_ids)
            nt = self._rng.choice(self._entity_ids)
            if self._rng.random() < 0.5:
                if (nh, r, t) not in existing:
                    return (nh, r, t)
            else:
                if (h, r, nt) not in existing:
                    return (h, r, nt)
        return (h, r, t)

    @staticmethod
    def _l2_norm(vec: List[float]) -> float:
        return math.sqrt(sum(v * v for v in vec))

    @staticmethod
    def _l2_distance(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _vec_add(a: List[float], b: List[float]) -> List[float]:
        return [x + y for x, y in zip(a, b)]

    @staticmethod
    def _vec_sub(a: List[float], b: List[float]) -> List[float]:
        return [x - y for x, y in zip(a, b)]

    @staticmethod
    def _vec_scale(a: List[float], s: float) -> List[float]:
        return [x * s for x in a]

    @staticmethod
    def _vec_elementwise(a: List[float], b: List[float]) -> List[float]:
        return [x * y for x, y in zip(a, b)]

    # --- TransE: h + r ≈ t ---

    def train_transe(self) -> None:
        self._init_embeddings()
        lr = self.config.learning_rate
        margin = self.config.margin
        neg_k = self.config.negative_samples
        for epoch in range(self.config.max_epochs):
            self._rng.shuffle(self._triples)
            total_loss = 0.0
            for h, r, t in self._triples:
                h_emb = self.entity_embeddings[h]
                r_emb = self.relation_embeddings[r]
                t_emb = self.entity_embeddings[t]
                pos_score = self._l2_distance(
                    self._vec_add(h_emb, r_emb), t_emb
                )
                for _ in range(neg_k):
                    nh, nr, nt = self._sample_negative(h, r, t)
                    nh_emb = self.entity_embeddings[nh]
                    nt_emb = self.entity_embeddings[nt]
                    neg_score = self._l2_distance(
                        self._vec_add(nh_emb, r_emb), nt_emb
                    )
                    loss = margin + pos_score - neg_score
                    if loss > 0:
                        total_loss += loss
                        grad_pos = self._vec_scale(
                            self._vec_sub(self._vec_add(h_emb, r_emb), t_emb),
                            2.0 * lr / max(self._l2_norm(self._vec_add(h_emb, r_emb)) + 1e-8, 1e-8)
                        )
                        grad_neg = self._vec_scale(
                            self._vec_sub(self._vec_add(nh_emb, r_emb), nt_emb),
                            2.0 * lr / max(self._l2_norm(self._vec_add(nh_emb, r_emb)) + 1e-8, 1e-8)
                        )
                        for i in range(len(h_emb)):
                            h_emb[i] -= grad_pos[i]
                            t_emb[i] += grad_pos[i]
                            nh_emb[i] += grad_neg[i]
                            nt_emb[i] -= grad_neg[i]

    # --- TransH: projection to relation-specific hyperplane ---

    def train_transh(self) -> None:
        self._init_embeddings()
        dim = self.config.embedding_dim
        lr = self.config.learning_rate
        margin = self.config.margin
        neg_k = self.config.negative_samples
        wr: Dict[str, List[float]] = {}
        for pid in self._relation_types:
            norm = [self._rng.gauss(0, 0.1) for _ in range(dim)]
            n = self._l2_norm(norm)
            wr[pid] = [x / n for x in norm]
        for epoch in range(self.config.max_epochs):
            self._rng.shuffle(self._triples)
            for h, r, t in self._triples:
                h_emb = self.entity_embeddings[h]
                t_emb = self.entity_embeddings[t]
                r_emb = self.relation_embeddings[r]
                w_r = wr[r]
                h_proj = self._vec_sub(h_emb, self._vec_scale(w_r, self._dot(h_emb, w_r)))
                t_proj = self._vec_sub(t_emb, self._vec_scale(w_r, self._dot(t_emb, w_r)))
                pos_score = self._l2_distance(self._vec_add(h_proj, r_emb), t_proj)
                for _ in range(neg_k):
                    nh, nr, nt = self._sample_negative(h, r, t)
                    nh_emb = self.entity_embeddings[nh]
                    nt_emb = self.entity_embeddings[nt]
                    nh_proj = self._vec_sub(nh_emb, self._vec_scale(w_r, self._dot(nh_emb, w_r)))
                    nt_proj = self._vec_sub(nt_emb, self._vec_scale(w_r, self._dot(nt_emb, w_r)))
                    neg_score = self._l2_distance(self._vec_add(nh_proj, r_emb), nt_proj)
                    loss = margin + pos_score - neg_score
                    if loss > 0:
                        grad = lr * 0.5
                        diff_pos = self._vec_sub(self._vec_add(h_proj, r_emb), t_proj)
                        diff_neg = self._vec_sub(self._vec_add(nh_proj, r_emb), nt_proj)
                        for i in range(dim):
                            h_emb[i] -= grad * diff_pos[i]
                            t_emb[i] += grad * diff_pos[i]
                            nh_emb[i] += grad * diff_neg[i]
                            nt_emb[i] -= grad * diff_neg[i]

    # --- TransR: separate entity/relation spaces ---

    def train_transr(self) -> None:
        self._init_embeddings()
        dim = self.config.embedding_dim
        rel_dim = max(dim // 2, 16)
        lr = self.config.learning_rate
        margin = self.config.margin
        neg_k = self.config.negative_samples
        rel_embs: Dict[str, List[float]] = {}
        mr: Dict[str, List[List[float]]] = {}
        for pid in self._relation_types:
            rel_embs[pid] = [self._rng.uniform(-0.1, 0.1) for _ in range(rel_dim)]
            mat = [[self._rng.gauss(0, 0.01) for _ in range(dim)] for _ in range(rel_dim)]
            mr[pid] = mat
        for epoch in range(self.config.max_epochs):
            self._rng.shuffle(self._triples)
            for h, r, t in self._triples:
                h_e = self.entity_embeddings[h]
                t_e = self.entity_embeddings[t]
                r_e = rel_embs[r]
                M = mr[r]
                h_r = [sum(M[i][j] * h_e[j] for j in range(dim)) for i in range(rel_dim)]
                t_r = [sum(M[i][j] * t_e[j] for j in range(dim)) for i in range(rel_dim)]
                pos_score = self._l2_distance(self._vec_add(h_r, r_e), t_r)
                for _ in range(neg_k):
                    nh, nr, nt = self._sample_negative(h, r, t)
                    nh_e = self.entity_embeddings[nh]
                    nt_e = self.entity_embeddings[nt]
                    nh_r = [sum(M[i][j] * nh_e[j] for j in range(dim)) for i in range(rel_dim)]
                    nt_r = [sum(M[i][j] * nt_e[j] for j in range(dim)) for i in range(rel_dim)]
                    neg_score = self._l2_distance(self._vec_add(nh_r, r_e), nt_r)
                    loss = margin + pos_score - neg_score
                    if loss > 0:
                        g = lr * 0.5
                        for i in range(rel_dim):
                            dp = self._vec_add(h_r, r_e)[i] - t_r[i]
                            dn = self._vec_add(nh_r, r_e)[i] - nt_r[i]
                            for j in range(dim):
                                h_e[j] -= g * dp * M[i][j]
                                t_e[j] += g * dp * M[i][j]
                                nh_e[j] += g * dn * M[i][j]
                                nt_e[j] -= g * dn * M[i][j]

    # --- DistMult: h ⊙ r ⊙ t ---

    def train_distmult(self) -> None:
        self._init_embeddings()
        lr = self.config.learning_rate
        margin = self.config.margin
        neg_k = self.config.negative_samples
        for epoch in range(self.config.max_epochs):
            self._rng.shuffle(self._triples)
            for h, r, t in self._triples:
                h_emb = self.entity_embeddings[h]
                r_emb = self.relation_embeddings[r]
                t_emb = self.entity_embeddings[t]
                pos_score = self._dot(h_emb, self._vec_elementwise(r_emb, t_emb))
                for _ in range(neg_k):
                    nh, nr, nt = self._sample_negative(h, r, t)
                    nh_emb = self.entity_embeddings[nh]
                    nt_emb = self.entity_embeddings[nt]
                    neg_score = self._dot(nh_emb, self._vec_elementwise(r_emb, nt_emb))
                    loss = margin - pos_score + neg_score
                    if loss > 0:
                        for i in range(len(h_emb)):
                            g = lr * (r_emb[i] * t_emb[i] - r_emb[i] * nt_emb[i])
                            h_emb[i] += g
                            nh_emb[i] -= g
                            g_t = lr * (r_emb[i] * h_emb[i] - r_emb[i] * nh_emb[i])
                            t_emb[i] += g_t
                            nt_emb[i] -= g_t

    # --- ComplEx: complex-valued h ⊙ r ⊙ conj(t) ---

    def train_complex(self) -> None:
        self._init_embeddings()
        dim = self.config.embedding_dim
        lr = self.config.learning_rate
        neg_k = self.config.negative_samples
        real: Dict[str, List[float]] = {}
        imag: Dict[str, List[float]] = {}
        r_real: Dict[str, List[float]] = {}
        r_imag: Dict[str, List[float]] = {}
        for eid in self._entity_ids:
            real[eid] = self.entity_embeddings[eid][:dim // 2]
            imag[eid] = self.entity_embeddings[eid][dim // 2:]
        for pid in self._relation_types:
            r_real[pid] = self.relation_embeddings[pid][:dim // 2]
            r_imag[pid] = self.relation_embeddings[pid][dim // 2:]
        half = dim // 2
        for epoch in range(self.config.max_epochs):
            self._rng.shuffle(self._triples)
            for h, r, t in self._triples:
                hr, hi = real[h], imag[h]
                rr, ri = r_real[r], r_imag[r]
                tr, ti = real[t], imag[t]
                score = sum(
                    (hr[i] * rr[i] - hi[i] * ri[i]) * tr[i]
                    + (hr[i] * ri[i] + hi[i] * rr[i]) * ti[i]
                    for i in range(half)
                )
                for _ in range(neg_k):
                    nh, nr, nt = self._sample_negative(h, r, t)
                    nhr, nhi = real[nh], imag[nh]
                    ntr, nti = real[nt], imag[nt]
                    neg = sum(
                        (nhr[i] * rr[i] - nhi[i] * ri[i]) * ntr[i]
                        + (nhr[i] * ri[i] + nhi[i] * rr[i]) * nti[i]
                        for i in range(half)
                    )
                    grad = lr * (neg - score)
                    for i in range(half):
                        hr[i] += grad * (rr[i] * tr[i] + ri[i] * ti[i])
                        hi[i] += grad * (-ri[i] * tr[i] + rr[i] * ti[i])
                        tr[i] += grad * (hr[i] * rr[i] + hi[i] * ri[i])
                        ti[i] += grad * (hr[i] * ri[i] - hi[i] * rr[i])

    # --- RotatE: t = h ∘ r (rotation in complex plane) ---

    def train_rotate(self) -> None:
        self._init_embeddings()
        dim = self.config.embedding_dim
        lr = self.config.learning_rate
        margin = self.config.margin
        neg_k = self.config.negative_samples
        half = dim // 2
        phase: Dict[str, List[float]] = {}
        for pid in self._relation_types:
            phase[pid] = [self._rng.uniform(0, 2 * math.pi) for _ in range(half)]
        for epoch in range(self.config.max_epochs):
            self._rng.shuffle(self._triples)
            for h, r, t in self._triples:
                h_emb = self.entity_embeddings[h]
                t_emb = self.entity_embeddings[t]
                r_ph = phase[r]
                pos_score = sum(
                    (h_emb[i] * math.cos(r_ph[i]) - h_emb[i + half] * math.sin(r_ph[i]) - t_emb[i]) ** 2
                    + (h_emb[i] * math.sin(r_ph[i]) + h_emb[i + half] * math.cos(r_ph[i]) - t_emb[i + half]) ** 2
                    for i in range(half)
                )
                for _ in range(neg_k):
                    nh, nr, nt = self._sample_negative(h, r, t)
                    nh_emb = self.entity_embeddings[nh]
                    nt_emb = self.entity_embeddings[nt]
                    neg_score = sum(
                        (nh_emb[i] * math.cos(r_ph[i]) - nh_emb[i + half] * math.sin(r_ph[i]) - nt_emb[i]) ** 2
                        + (nh_emb[i] * math.sin(r_ph[i]) + nh_emb[i + half] * math.cos(r_ph[i]) - nt_emb[i + half]) ** 2
                        for i in range(half)
                    )
                    loss = margin + pos_score - neg_score
                    if loss > 0:
                        g = lr * 0.5
                        for i in range(half):
                            cos_r = math.cos(r_ph[i])
                            sin_r = math.sin(r_ph[i])
                            re_h = h_emb[i] * cos_r - h_emb[i + half] * sin_r
                            im_h = h_emb[i] * sin_r + h_emb[i + half] * cos_r
                            re_n = nh_emb[i] * cos_r - nh_emb[i + half] * sin_r
                            im_n = nh_emb[i] * sin_r + nh_emb[i + half] * cos_r
                            h_emb[i] -= g * 2 * (re_h - t_emb[i]) * cos_r
                            h_emb[i + half] -= g * 2 * (im_h - t_emb[i + half]) * sin_r
                            t_emb[i] += g * 2 * (re_h - t_emb[i])
                            t_emb[i + half] += g * 2 * (im_h - t_emb[i + half])
                            nh_emb[i] += g * 2 * (re_n - nt_emb[i]) * cos_r
                            nh_emb[i + half] += g * 2 * (im_n - nt_emb[i + half]) * sin_r
                            nt_emb[i] -= g * 2 * (re_n - nt_emb[i])
                            nt_emb[i + half] -= g * 2 * (im_n - nt_emb[i + half])

    def score_triple(self, h: str, r: str, t: str) -> float:
        """Score a triple using current embeddings (negative = more plausible)."""
        h_emb = self.entity_embeddings.get(h)
        r_emb = self.relation_embeddings.get(r)
        t_emb = self.entity_embeddings.get(t)
        if not h_emb or not r_emb or not t_emb:
            return float('inf')
        return self._l2_distance(self._vec_add(h_emb, r_emb), t_emb)


# ============================================================
# 6. GraphAlgorithm
# ============================================================

class GraphAlgorithm:
    """Classical graph algorithms for knowledge graph analysis."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    # --- PageRank ---

    def pagerank(self, damping: float = 0.85, max_iter: int = 100,
                 tol: float = 1e-6) -> Dict[str, float]:
        nodes = list(self.graph._entities.keys())
        n = len(nodes)
        if n == 0:
            return {}
        out_deg = {nid: len(self.graph._adj_out.get(nid, [])) for nid in nodes}
        rank = {nid: 1.0 / n for nid in nodes}
        for _ in range(max_iter):
            new_rank: Dict[str, float] = {}
            for nid in nodes:
                s = 0.0
                for src, _, _ in self.graph._adj_in.get(nid, []):
                    if out_deg[src] > 0:
                        s += rank[src] / out_deg[src]
                new_rank[nid] = (1 - damping) / n + damping * s
            diff = sum(abs(new_rank[nid] - rank[nid]) for nid in nodes)
            rank = new_rank
            if diff < tol:
                break
        total = sum(rank.values())
        if total > 0:
            rank = {k: v / total for k, v in rank.items()}
        return rank

    # --- HITS: Hubs and Authorities ---

    def hits(self, max_iter: int = 100, tol: float = 1e-6) -> Tuple[Dict[str, float], Dict[str, float]]:
        nodes = list(self.graph._entities.keys())
        auth = {nid: 1.0 for nid in nodes}
        hub = {nid: 1.0 for nid in nodes}
        for _ in range(max_iter):
            new_auth: Dict[str, float] = {}
            for nid in nodes:
                new_auth[nid] = sum(hub[src] for src, _, _ in self.graph._adj_in.get(nid, []))
            norm_a = math.sqrt(sum(v * v for v in new_auth.values())) or 1.0
            auth = {k: v / norm_a for k, v in new_auth.items()}
            new_hub: Dict[str, float] = {}
            for nid in nodes:
                new_hub[nid] = sum(auth[tgt] for tgt, _, _ in self.graph._adj_out.get(nid, []))
            norm_h = math.sqrt(sum(v * v for v in new_hub.values())) or 1.0
            hub = {k: v / norm_h for k, v in new_hub.items()}
            diff = sum(abs(auth[nid] - new_auth.get(nid, 0)) for nid in nodes)
            if diff < tol:
                break
        return hub, auth

    # --- Personalized PageRank ---

    def personalized_pagerank(self, personalization: Dict[str, float],
                               damping: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        nodes = list(self.graph._entities.keys())
        n = len(nodes)
        if n == 0:
            return {}
        p_total = sum(personalization.values())
        if p_total == 0:
            return {nid: 0.0 for nid in nodes}
        p_norm = {k: v / p_total for k, v in personalization.items()}
        out_deg = {nid: len(self.graph._adj_out.get(nid, [])) for nid in nodes}
        rank = {nid: 1.0 / n for nid in nodes}
        for _ in range(max_iter):
            new_rank: Dict[str, float] = {}
            for nid in nodes:
                s = sum(rank[src] / out_deg[src] for src, _, _ in self.graph._adj_in.get(nid, [])
                        if out_deg[src] > 0)
                new_rank[nid] = (1 - damping) * p_norm.get(nid, 0) + damping * s
            rank = new_rank
        total = sum(rank.values())
        if total > 0:
            rank = {k: v / total for k, v in rank.items()}
        return rank

    # --- Community Detection: Louvain ---

    def community_detection(self, resolution: float = 1.0) -> Dict[str, int]:
        """Louvain community detection via modularity optimization."""
        nodes = list(self.graph._entities.keys())
        n = len(nodes)
        if n == 0:
            return {}
        m = len(self.graph._relations)
        if m == 0:
            return {nid: i for i, nid in enumerate(nodes)}
        node_idx = {nid: i for i, nid in enumerate(nodes)}
        community = list(range(n))
        adj: Dict[int, List[int]] = defaultdict(list)
        for rel in self.graph._relations.values():
            si = node_idx.get(rel.subject)
            oi = node_idx.get(rel.object)
            if si is not None and oi is not None:
                adj[si].append(oi)
                adj[oi].append(si)
        k = [len(adj[i]) for i in range(n)]
        sigma_tot = [0.0] * n
        for i in range(n):
            sigma_tot[community[i]] += k[i]
        improved = True
        while improved:
            improved = False
            for i in range(n):
                ci = community[i]
                ki = k[i]
                best_delta = 0.0
                best_c = ci
                neighbor_communities: Set[int] = set()
                for j in adj[i]:
                    neighbor_communities.add(community[j])
                for cj in neighbor_communities:
                    if cj == ci:
                        continue
                    ki_in = sum(1 for j in adj[i] if community[j] == cj)
                    delta = (ki_in / (2 * m)) - (resolution * sigma_tot[cj] * ki / (4 * m * m))
                    if delta > best_delta:
                        best_delta = delta
                        best_c = cj
                if best_c != ci:
                    community[i] = best_c
                    improved = True
        return {nodes[i]: community[i] for i in range(n)}

    # --- Shortest Path: Dijkstra ---

    def dijkstra(self, source: str) -> Dict[str, float]:
        dist: Dict[str, float] = {source: 0.0}
        heap = [(0.0, source)]
        visited: Set[str] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            for v, _, w in self.graph._adj_out.get(u, []):
                nd = d + (1.0 / w if w > 0 else float('inf'))
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """BFS shortest path."""
        if source == target:
            return [source]
        visited: Set[str] = {source}
        queue: deque = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            for neighbor, _, _ in self.graph._adj_out.get(node, []):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # --- Centrality ---

    def degree_centrality(self) -> Dict[str, float]:
        n = self.graph.num_entities()
        if n <= 1:
            return {eid: 0.0 for eid in self.graph._entities}
        deg = self.graph.degree_distribution()
        return {eid: d / (n - 1) for eid, d in deg.items()}

    def betweenness_centrality(self) -> Dict[str, float]:
        nodes = list(self.graph._entities.keys())
        cb: Dict[str, float] = {n: 0.0 for n in nodes}
        for s in nodes:
            stack: List[str] = []
            pred: Dict[str, List[str]] = {n: [] for n in nodes}
            sigma: Dict[str, int] = {n: 0 for n in nodes}
            sigma[s] = 1
            dist: Dict[str, int] = {n: -1 for n in nodes}
            dist[s] = 0
            queue: deque = deque([s])
            while queue:
                v = queue.popleft()
                stack.append(v)
                for w, _, _ in self.graph._adj_out.get(v, []):
                    if dist.get(w, -1) < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist.get(w, -1) == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            delta: Dict[str, float] = {n: 0.0 for n in nodes}
            while stack:
                w = stack.pop()
                for v in pred.get(w, []):
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    cb[w] += delta[w]
        total = len(nodes)
        scale = 1.0 / ((total - 1) * (total - 2)) if total > 2 else 1.0
        return {n: v * scale for n, v in cb.items()}

    def closeness_centrality(self) -> Dict[str, float]:
        nodes = list(self.graph._entities.keys())
        result: Dict[str, float] = {}
        for s in nodes:
            dist_map = self.dijkstra(s)
            reachable = sum(1 for d in dist_map.values() if d < float('inf') and d > 0)
            if reachable == 0:
                result[s] = 0.0
            else:
                total_dist = sum(d for d in dist_map.values() if d < float('inf') and d > 0)
                result[s] = (reachable - 1) / (len(nodes) - 1) * reachable / total_dist if total_dist > 0 else 0.0
        return result

    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        nodes = list(self.graph._entities.keys())
        n = len(nodes)
        if n == 0:
            return {}
        centrality = {nid: 1.0 / n for nid in nodes}
        for _ in range(max_iter):
            new_c: Dict[str, float] = {}
            for nid in nodes:
                new_c[nid] = sum(centrality[tgt] for tgt, _, _ in self.graph._adj_out.get(nid, []))
            norm = math.sqrt(sum(v * v for v in new_c.values())) or 1.0
            new_c = {k: v / norm for k, v in new_c.items()}
            diff = sum(abs(new_c[nid] - centrality[nid]) for nid in nodes)
            centrality = new_c
            if diff < tol:
                break
        return centrality

    # --- Connected Components ---

    def connected_components(self) -> List[Set[str]]:
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for nid in self.graph._entities:
            if nid in visited:
                continue
            component: Set[str] = set()
            queue: deque = deque([nid])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor, _, _ in self.graph._adj_out.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
                for neighbor, _, _ in self.graph._adj_in.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
        return components

    # --- Triadic Closure ---

    def triadic_closure(self) -> List[Tuple[str, str, float]]:
        """Predict missing edges via friend-of-friend (common neighbors)."""
        predictions: List[Tuple[str, str, float]] = []
        existing = set(self.graph._relations.keys())
        nodes = list(self.graph._entities.keys())
        for i, u in enumerate(nodes):
            u_neighbors = {v for v, _, _ in self.graph._adj_out.get(u, [])}
            u_neighbors |= {v for v, _, _ in self.graph._adj_in.get(u, [])}
            for v in nodes[i + 1:]:
                if (u, "related_to", v) in existing or (v, "related_to", u) in existing:
                    continue
                v_neighbors = {w for w, _, _ in self.graph._adj_out.get(v, [])}
                v_neighbors |= {w for w, _, _ in self.graph._adj_in.get(v, [])}
                common = len(u_neighbors & v_neighbors)
                if common > 0:
                    predictions.append((u, v, float(common)))
        predictions.sort(key=lambda x: -x[2])
        return predictions


# ============================================================
# 7. Reasoning
# ============================================================

class Reasoning:
    """Knowledge graph reasoning capabilities."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    # --- Rule Mining ---

    def mine_rules(self, min_support: float = 0.01,
                    min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Mine association rules: body predicates -> head predicate."""
        predicate_triples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for rel in self.graph.relations():
            predicate_triples[rel.predicate].append((rel.subject, rel.object))
        total = max(len(self.graph.relations()), 1)
        rules: List[Dict[str, Any]] = []
        predicates = list(predicate_triples.keys())
        for i, p1 in enumerate(predicates):
            p1_set = set(predicate_triples[p1])
            support_p1 = len(p1_set) / total
            if support_p1 < min_support:
                continue
            for p2 in predicates[i + 1:]:
                p2_set = set(predicate_triples[p2])
                support_p2 = len(p2_set) / total
                if support_p2 < min_support:
                    continue
                intersection = p1_set & p2_set
                support_both = len(intersection) / total
                if support_both < min_support:
                    continue
                conf_p1_p2 = len(intersection) / len(p1_set) if p1_set else 0
                conf_p2_p1 = len(intersection) / len(p2_set) if p2_set else 0
                if conf_p1_p2 >= min_confidence:
                    rules.append({
                        "body": p1, "head": p2,
                        "support": support_both, "confidence": conf_p1_p2,
                        "lift": conf_p1_p2 / support_p2 if support_p2 > 0 else 0
                    })
                if conf_p2_p1 >= min_confidence:
                    rules.append({
                        "body": p2, "head": p1,
                        "support": support_both, "confidence": conf_p2_p1,
                        "lift": conf_p2_p1 / support_p1 if support_p1 > 0 else 0
                    })
        rules.sort(key=lambda r: -r["confidence"])
        return rules

    # --- Link Prediction ---

    def predict_links(self, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Predict missing links using common neighbors + Adamic-Adar index."""
        existing = set(self.graph._relations.keys())
        nodes = list(self.graph._entities.keys())
        scores: List[Tuple[str, str, float]] = []
        for i, u in enumerate(nodes):
            u_nbrs = set(self.graph._adj_out.get(u, [])) | set(
                n for n, _, _ in self.graph._adj_in.get(u, [])
            )
            for v in nodes[i + 1:]:
                if any((u, p, v) in existing or (v, p, u) in existing
                       for p in self.graph._predicate_index):
                    continue
                v_nbrs = set(self.graph._adj_out.get(v, [])) | set(
                    n for n, _, _ in self.graph._adj_in.get(v, [])
                )
                common = u_nbrs & v_nbrs
                aa = sum(1.0 / math.log(len(self.graph._adj_out.get(c, []))
                       + len(self.graph._adj_in.get(c, [])) + 1) for c in common)
                cn = len(common)
                score = cn + aa
                if score > 0:
                    scores.append((u, v, score))
        scores.sort(key=lambda x: -x[2])
        return scores[:top_k]

    # --- Path Reasoning ---

    def reason_path(self, source: str, target: str,
                     max_hops: int = 4) -> List[Dict[str, Any]]:
        """Multi-hop reasoning: find explanatory paths between entities."""
        paths = self.graph.find_paths(source, target, max_depth=max_hops)
        results: List[Dict[str, Any]] = []
        for path in paths:
            edges = []
            for i in range(len(path) - 1):
                s, o = path[i], path[i + 1]
                rel = self.graph.get_relation(s, "related_to", o)
                if rel is None:
                    for r in self.graph._relations.values():
                        if r.subject == s and r.object == o:
                            rel = r
                            break
                edges.append({
                    "subject": s, "object": o,
                    "predicate": rel.predicate if rel else "inferred"
                })
            results.append({"path": path, "edges": edges, "length": len(path) - 1})
        results.sort(key=lambda x: x["length"])
        return results

    # --- Subgraph Matching ---

    def match_subgraph(self, pattern_entities: Dict[str, str],
                        pattern_edges: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """Match a subgraph pattern against the knowledge graph.
        pattern_entities: {variable_name: entity_type}
        pattern_edges: [(var1, predicate, var2), ...]
        Returns list of bindings {variable_name: entity_id}.
        """
        candidates: Dict[str, List[str]] = {}
        for var, etype in pattern_entities.items():
            candidates[var] = [
                e.id for e in self.graph.entities() if e.type == etype
            ]
        bindings_list: List[Dict[str, str]] = [{}]
        for var1, pred, var2 in pattern_edges:
            new_bindings: List[Dict[str, str]] = []
            for binding in bindings_list:
                c1 = candidates.get(var1, list(self.graph._entities.keys()))
                c2 = candidates.get(var2, list(self.graph._entities.keys()))
                if var1 in binding:
                    c1 = [binding[var1]]
                if var2 in binding:
                    c2 = [binding[var2]]
                for e1 in c1:
                    for e2 in c2:
                        if e1 == e2:
                            continue
                        rel = self.graph.get_relation(e1, pred, e2)
                        if rel is not None:
                            b = dict(binding)
                            b[var1] = e1
                            b[var2] = e2
                            new_bindings.append(b)
            bindings_list = new_bindings
            if not bindings_list:
                break
        return bindings_list


# ============================================================
# 8. TemporalKnowledgeGraph
# ============================================================

class TemporalKnowledgeGraph(KnowledgeGraph):
    """Time-aware knowledge graph with temporal reasoning and decay."""

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        super().__init__(config)
        self._time_index: Dict[float, List[Relation]] = defaultdict(list)
        self._entity_temporal: Dict[str, List[float]] = defaultdict(list)

    def add_relation(self, relation: Relation) -> None:
        super().add_relation(relation)
        if relation.timestamp is not None:
            self._time_index[relation.timestamp].append(relation)
            self._entity_temporal[relation.subject].append(relation.timestamp)
            self._entity_temporal[relation.object].append(relation.timestamp)

    def relations_at_time(self, timestamp: float, window: float = 0.0) -> List[Relation]:
        """Get relations within a time window."""
        result: List[Relation] = []
        for t, rels in self._time_index.items():
            if abs(t - timestamp) <= window:
                result.extend(rels)
        return result

    def temporal_importance(self, entity_id: str, current_time: float) -> float:
        """Compute time-decayed importance of an entity."""
        decay = self.config.temporal_decay_rate
        timestamps = self._entity_temporal.get(entity_id, [])
        if not timestamps:
            return 0.0
        importance = sum(math.exp(-decay * (current_time - t)) for t in timestamps)
        return importance

    def temporal_snapshot(self, timestamp: float, window: float = 1.0) -> KnowledgeGraph:
        """Extract a knowledge graph snapshot at a given time."""
        rels = self.relations_at_time(timestamp, window)
        entity_ids: Set[str] = set()
        for r in rels:
            entity_ids.add(r.subject)
            entity_ids.add(r.object)
        return self.extract_subgraph(entity_ids)

    def temporal_reasoning(self, entity_id: str, predicate: str,
                            current_time: float) -> List[Tuple[str, float]]:
        """Predict likely objects for (entity, predicate, ?) using temporal decay."""
        scores: Dict[str, float] = defaultdict(float)
        decay = self.config.temporal_decay_rate
        for rel in self.graph.relations():
            if rel.subject == entity_id and rel.predicate == predicate and rel.timestamp is not None:
                age = current_time - rel.timestamp
                scores[rel.object] += math.exp(-decay * age) * rel.weight
        result = sorted(scores.items(), key=lambda x: -x[1])
        return result

    def temporal_evolution(self, entity_id: str) -> List[Tuple[float, str, str]]:
        """Get chronological events involving an entity."""
        events: List[Tuple[float, str, str]] = []
        for rel in self.graph.relations():
            if rel.subject == entity_id and rel.timestamp is not None:
                events.append((rel.timestamp, rel.predicate, rel.object))
            elif rel.object == entity_id and rel.timestamp is not None:
                events.append((rel.timestamp, rel.predicate, rel.subject))
        events.sort()
        return events


# ============================================================
# 9. KnowledgeGraphManager
# ============================================================

class KnowledgeGraphManager:
    """High-level knowledge graph operations."""

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        self.config = config or KnowledgeGraphConfig()
        self.graph = KnowledgeGraph(self.config)
        self.algorithms = GraphAlgorithm(self.graph)
        self.reasoning = Reasoning(self.graph)
        self.embedding: Optional[GraphEmbedding] = None

    # --- Build from text ---

    def build_from_text(self, text: str) -> None:
        """Simulate entity and relation extraction from text."""
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        entity_types = {
            "person": {"he", "she", "they", "mr", "mrs", "dr", "professor", "ceo", "president"},
            "location": {"city", "country", "capital", "river", "mountain", "ocean", "continent"},
            "organization": {"company", "university", "government", "institute", "corporation"},
            "concept": {"theory", "idea", "method", "algorithm", "science", "technology"},
            "event": {"war", "revolution", "discovery", "invention", "election"},
        }
        relation_keywords = {
            "is_a": {"is a", "are a", "was a", "were a"},
            "part_of": {"part of", "belongs to", "member of", "located in"},
            "related_to": {"related to", "connected to", "associated with", "linked to"},
            "created_by": {"created by", "invented by", "discovered by", "founded by"},
            "located_in": {"located in", "situated in", "found in", "based in"},
            "works_for": {"works for", "employed by", "serves at"},
        }
        import re
        word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        entity_id_counter = 0
        entity_map: Dict[str, str] = {}
        for sentence in sentences:
            words = word_pattern.findall(sentence.lower())
            for word in words:
                if len(word) < 3:
                    continue
                if word not in entity_map:
                    etype = "generic"
                    for t, keywords in entity_types.items():
                        if word in keywords:
                            etype = t
                            break
                    eid = f"e_{entity_id_counter}"
                    entity_id_counter += 1
                    entity_map[word] = eid
                    self.graph.add_entity(Entity(id=eid, type=etype, properties={"label": word}))
            for rel_name, keywords in relation_keywords.items():
                for kw in keywords:
                    if kw in sentence.lower():
                        kw_words = kw.split()
                        for kw_word in kw_words:
                            if kw_word in entity_map:
                                idx = sentence.lower().index(kw)
                                before = sentence[:idx].strip().split()
                                after = sentence[idx + len(kw):].strip().split()
                                for bw in before:
                                    bw_lower = bw.lower().strip(".,;:!?")
                                    if bw_lower in entity_map and bw_lower != kw_word:
                                        for aw in after:
                                            aw_lower = aw.lower().strip(".,;:!?")
                                            if aw_lower in entity_map and aw_lower != kw_word:
                                                self.graph.add_relation(Relation(
                                                    subject=entity_map[bw_lower],
                                                    predicate=rel_name,
                                                    object=entity_map[aw_lower],
                                                    timestamp=entity_id_counter * 0.1
                                                ))
                                break

    # --- SPARQL-like Query ---

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """Simple SPARQL-like query interface.
        Supports: SELECT ?x WHERE { ?x predicate object }
                  SELECT ?x ?y WHERE { ?x predicate ?y }
        """
        import re
        query_lower = query_str.strip().lower()
        select_match = re.match(r'select\s+(.*?)\s+where\s*\{(.*?)\}', query_lower, re.DOTALL)
        if not select_match:
            return []
        variables = [v.strip().lstrip("?") for v in select_match.group(1).split()]
        pattern = select_match.group(2).strip()
        triples_pattern = re.findall(r'(\?\w+|\w+)\s+(\w+)\s+(\?\w+|\w+)', pattern)
        bindings_list: List[Dict[str, str]] = [{}]
        for subj_pat, pred, obj_pat in triples_pattern:
            new_bindings: List[Dict[str, str]] = []
            for binding in bindings_list:
                subj_resolved = binding.get(subj_pat.lstrip("?"), subj_pat) if subj_pat.startswith("?") else subj_pat
                obj_resolved = binding.get(obj_pat.lstrip("?"), obj_pat) if obj_pat.startswith("?") else obj_pat
                if subj_pat.startswith("?") and subj_resolved == subj_pat:
                    candidates_s = list(self.graph._entities.keys())
                else:
                    candidates_s = [subj_resolved]
                if obj_pat.startswith("?") and obj_resolved == obj_pat:
                    candidates_o = list(self.graph._entities.keys())
                else:
                    candidates_o = [obj_resolved]
                for s in candidates_s:
                    for o in candidates_o:
                        rel = self.graph.get_relation(s, pred, o)
                        if rel is not None:
                            b = dict(binding)
                            if subj_pat.startswith("?"):
                                b[subj_pat.lstrip("?")] = s
                            if obj_pat.startswith("?"):
                                b[obj_pat.lstrip("?")] = o
                            new_bindings.append(b)
            bindings_list = new_bindings
        results: List[Dict[str, Any]] = []
        for binding in bindings_list:
            row: Dict[str, Any] = {}
            for var in variables:
                eid = binding.get(var)
                if eid:
                    entity = self.graph.get_entity(eid)
                    row[var] = entity.properties.get("label", eid) if entity else eid
                else:
                    row[var] = None
            results.append(row)
        return results

    # --- Graph Merging ---

    def merge(self, other: KnowledgeGraph,
              entity_matcher: Optional[Any] = None) -> None:
        """Merge another knowledge graph into this one."""
        matched: Dict[str, str] = {}
        if entity_matcher:
            for e in other.entities():
                match = entity_matcher(e, self.graph.entities())
                if match:
                    matched[e.id] = match.id
        for e in other.entities():
            if e.id not in matched:
                self.graph.add_entity(e)
        for rel in other.relations():
            s = matched.get(rel.subject, rel.subject)
            o = matched.get(rel.object, rel.object)
            if self.graph.get_relation(s, rel.predicate, o) is None:
                self.graph.add_relation(Relation(
                    subject=s, predicate=rel.predicate, object=o,
                    weight=rel.weight, timestamp=rel.timestamp
                ))

    # --- Knowledge Completion ---

    def complete(self, method: str = "pagerank", top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """Suggest missing triples for knowledge completion."""
        if method == "pagerank":
            pr = self.algorithms.pagerank()
            top_entities = sorted(pr.items(), key=lambda x: -x[1])[:top_k]
            suggestions: List[Tuple[str, str, str, float]] = []
            existing = set(self.graph._relations.keys())
            predicates = list(self.graph._predicate_index.keys())
            for eid, score in top_entities:
                for pred in predicates:
                    for _, oid, _ in self.graph._predicate_index[pred]:
                        if (eid, pred, oid) not in existing and (oid, pred, eid) not in existing:
                            suggestions.append((eid, pred, oid, score * 0.1))
                            if len(suggestions) >= top_k:
                                return suggestions
            return suggestions
        elif method == "embedding":
            if self.embedding is None:
                self.embedding = GraphEmbedding(self.graph, self.config)
                self.embedding.train_transe()
            suggestions: List[Tuple[str, str, str, float]] = []
            existing = set(self.graph._relations.keys())
            eids = list(self.graph._entities.keys())
            pids = list(self.graph._predicate_index.keys())
            for _ in range(top_k * 10):
                h = random.choice(eids)
                r = random.choice(pids)
                t = random.choice(eids)
                if (h, r, t) not in existing:
                    score = self.embedding.score_triple(h, r, t)
                    suggestions.append((h, r, t, -score))
            suggestions.sort(key=lambda x: -x[3])
            return suggestions[:top_k]
        elif method == "triadic":
            preds = self.algorithms.triadic_closure()
            return [(u, "related_to", v, s) for u, v, s in preds[:top_k]]
        return []

    # --- Statistics summary ---

    def summary(self) -> Dict[str, Any]:
        return {
            "num_entities": self.graph.num_entities(),
            "num_relations": self.graph.num_relations(),
            "entity_types": self.graph.entity_types(),
            "avg_degree": self.graph.avg_degree(),
            "density": self.graph.density(),
            "avg_clustering": self.graph.avg_clustering_coefficient(),
            "avg_path_length": self.graph.avg_path_length(),
            "num_components": len(self.algorithms.connected_components()),
        }
