"""
Hierarchical Memory System Implementation
==========================================
Implements a multi-level memory hierarchy inspired by human cognition:
  - SensoryMemory:  echoic / iconic / haptic buffers with decay-based forgetting
  - WorkingMemory:  Baddeley's model (phonological loop, visuospatial sketchpad, episodic buffer)
  - ShortTermMemory: chunking, primacy/recency effects
  - LongTermMemory:  episodic, semantic, procedural sub-systems with consolidation
  - HierarchicalMemoryManager: orchestrates all levels, Ebbinghaus forgetting, reconsolidation

Pure Python -- only stdlib modules (math, random, time, collections, copy).
"""

from __future__ import annotations

import math
import random
import time
import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _normalize(v: List[float]) -> List[float]:
    """L2-normalize a vector."""
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        return v[:]
    return [x / mag for x in v]


def _random_vector(dim: int) -> List[float]:
    """Generate a random unit vector of given dimension."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    return _normalize(v)


# ---------------------------------------------------------------------------
# 1. MemoryItem
# ---------------------------------------------------------------------------

@dataclass
class MemoryItem:
    """Base memory item carrying content, embedding, metadata."""
    content: str
    embedding: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5          # 0..1
    access_count: int = 0
    emotional_valence: float = 0.0   # -1 (negative) .. +1 (positive)
    memory_type: str = "generic"     # episodic / semantic / procedural
    metadata: Dict[str, Any] = field(default_factory=dict)
    item_id: str = ""

    def __post_init__(self):
        if not self.item_id:
            self.item_id = f"mem_{id(self):016x}"

    def touch(self):
        """Record an access event."""
        self.access_count += 1

    def similarity_to(self, other: "MemoryItem") -> float:
        if not self.embedding or not other.embedding:
            return 0.0
        return _cosine_similarity(self.embedding, other.embedding)


# ---------------------------------------------------------------------------
# 2. MemoryStore (abstract base)
# ---------------------------------------------------------------------------

class MemoryStore:
    """Abstract base class for all memory stores."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._items: Dict[str, MemoryItem] = {}

    # -- core CRUD ----------------------------------------------------------

    def add(self, item: MemoryItem) -> bool:
        """Add a memory item to the store.
        
        Default implementation stores the item directly. If the store is at
        capacity, the least recently accessed item is evicted.
        
        Args:
            item: The MemoryItem to add.
            
        Returns:
            True if the item was added successfully.
        """
        if not isinstance(item, MemoryItem):
            return False
        
        # Evict if at capacity
        if self.size() >= self.capacity and item.item_id not in self._items:
            self._evict_one()
        
        self._items[item.item_id] = item
        return True
    
    def _evict_one(self) -> None:
        """Evict a single item to make room. Default: remove oldest by timestamp."""
        if not self._items:
            return
        oldest_id = min(self._items, key=lambda k: self._items[k].timestamp)
        del self._items[oldest_id]

    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def delete(self, item_id: str) -> bool:
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def search(self, query: List[float], top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """Return top-k items most similar to *query* embedding."""
        scored = [(item, _cosine_similarity(query, item.embedding))
                  for item in self._items.values() if item.embedding]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def update(self, item_id: str, **kwargs) -> bool:
        item = self._items.get(item_id)
        if item is None:
            return False
        for k, v in kwargs.items():
            if hasattr(item, k):
                setattr(item, k, v)
        return True

    def size(self) -> int:
        return len(self._items)

    def all_items(self) -> List[MemoryItem]:
        return list(self._items.values())


# ---------------------------------------------------------------------------
# 3. SensoryMemory
# ---------------------------------------------------------------------------

class SensoryBufferType(Enum):
    ECHOIC = auto()     # auditory
    ICONIC = auto()     # visual
    HAPTIC = auto()     # touch / proprioception


class SensoryMemory(MemoryStore):
    """
    Very short-term sensory buffers with exponential decay.
    Durations (approximate, in seconds):
        iconic  ~ 0.2-0.5 s
        echoic  ~ 3-5 s
        haptic  ~ 2 s
    """

    DEFAULT_DURATIONS = {
        SensoryBufferType.ICONIC: 0.5,
        SensoryBufferType.ECHOIC: 4.0,
        SensoryBufferType.HAPTIC: 2.0,
    }

    def __init__(self, buffer_type: SensoryBufferType = SensoryBufferType.ECHOIC,
                 duration: Optional[float] = None, capacity: int = 50):
        super().__init__(capacity=capacity)
        self.buffer_type = buffer_type
        self.duration = duration or self.DEFAULT_DURATIONS[buffer_type]
        self._decay_rate = math.log(2) / self.duration  # half-life based

    def add(self, item: MemoryItem) -> bool:
        if self.size() >= self.capacity:
            self._evict_oldest()
        item.metadata["sensory_buffer"] = self.buffer_type.name
        self._items[item.item_id] = item
        return True

    def retrieve_active(self) -> List[MemoryItem]:
        """Return all items still within decay window."""
        now = time.time()
        active = []
        expired = []
        for item in self._items.values():
            age = now - item.timestamp
            strength = math.exp(-self._decay_rate * age)
            if strength > 0.01:
                item.metadata["_strength"] = strength
                active.append(item)
            else:
                expired.append(item.item_id)
        for eid in expired:
            del self._items[eid]
        active.sort(key=lambda m: m.metadata.get("_strength", 0), reverse=True)
        return active

    def decay_strength(self, item: MemoryItem) -> float:
        age = time.time() - item.timestamp
        return math.exp(-self._decay_rate * age)

    def _evict_oldest(self):
        if not self._items:
            return
        oldest_id = min(self._items, key=lambda k: self._items[k].timestamp)
        del self._items[oldest_id]


# ---------------------------------------------------------------------------
# 4. WorkingMemory  (Baddeley's model)
# ---------------------------------------------------------------------------

class WorkingMemoryComponent(Enum):
    PHONOLOGICAL_LOOP = auto()
    VISUOSPATIAL_SKETCHPAD = auto()
    EPISODIC_BUFFER = auto()


class WorkingMemory(MemoryStore):
    """
    Baddeley's working memory model.
    Capacity: 7 +/- 2 items (Miller's law).
    Forgetting: interference-based (proactive + retroactive).
    Rehearsal: subvocal rehearsal refreshes phonological loop items.
    """

    def __init__(self, capacity: int = 7, rehearsal_rate: float = 1.5):
        super().__init__(capacity=capacity)
        self.rehearsal_rate = rehearsal_rate          # seconds per rehearsal cycle
        self._phonological_loop: OrderedDict[str, MemoryItem] = OrderedDict()
        self._visuospatial: OrderedDict[str, MemoryItem] = OrderedDict()
        self._episodic_buffer: OrderedDict[str, MemoryItem] = OrderedDict()
        self._last_rehearsal: float = time.time()
        self._interference_matrix: Dict[str, float] = defaultdict(float)

    # -- add / retrieve -----------------------------------------------------

    def add(self, item: MemoryItem, component: WorkingMemoryComponent = WorkingMemoryComponent.EPISODIC_BUFFER) -> bool:
        total = len(self._phonological_loop) + len(self._visuospatial) + len(self._episodic_buffer)
        if total >= self.capacity:
            self._evict_weakest()
        store = self._get_store(component)
        store[item.item_id] = item
        self._items[item.item_id] = item
        item.metadata["wm_component"] = component.name
        return True

    def retrieve_from(self, component: WorkingMemoryComponent) -> List[MemoryItem]:
        store = self._get_store(component)
        return list(store.values())

    def get_all_active(self) -> List[MemoryItem]:
        """Return all items across components, applying interference decay."""
        self._apply_interference()
        self._rehearse()
        return list(self._items.values())

    # -- rehearsal ----------------------------------------------------------

    def _rehearse(self):
        """Subvocal rehearsal refreshes phonological loop items."""
        now = time.time()
        if now - self._last_rehearsal < self.rehearsal_rate:
            return
        self._last_rehearsal = now
        for item in self._phonological_loop.values():
            item.timestamp = now   # refresh timestamp
            item.access_count += 1

    # -- interference -------------------------------------------------------

    def _compute_interference(self, new_item: MemoryItem):
        """Compute proactive & retroactive interference for existing items."""
        for existing in self._items.values():
            if existing.item_id == new_item.item_id:
                continue
            sim = new_item.similarity_to(existing)
            # Proactive: existing interferes with new
            # Retroactive: new interferes with existing
            interference = sim * 0.1
            self._interference_matrix[existing.item_id] += interference
            self._interference_matrix[new_item.item_id] += interference

    def _apply_interference(self):
        """Reduce activation of items based on accumulated interference."""
        to_remove = []
        for item_id, interference in self._interference_matrix.items():
            if item_id not in self._items:
                continue
            item = self._items[item_id]
            # Activation decays with interference
            activation = 1.0 - min(interference, 1.0)
            if activation < 0.1:
                to_remove.append(item_id)
        for item_id in to_remove:
            self.delete(item_id)
            self._interference_matrix.pop(item_id, None)
            for store in (self._phonological_loop, self._visuospatial, self._episodic_buffer):
                store.pop(item_id, None)

    # -- eviction -----------------------------------------------------------

    def _evict_weakest(self):
        """Remove the item with lowest activation across all components."""
        if not self._items:
            return
        weakest_id = min(self._items,
                         key=lambda k: self._activation_score(self._items[k]))
        self.delete(weakest_id)
        for store in (self._phonological_loop, self._visuospatial, self._episodic_buffer):
            store.pop(weakest_id, None)

    def _activation_score(self, item: MemoryItem) -> float:
        age = time.time() - item.timestamp
        interference = self._interference_matrix.get(item.item_id, 0.0)
        recency = math.exp(-0.1 * age)
        return recency * (1.0 - interference) * (0.5 + 0.5 * item.importance)

    # -- helpers ------------------------------------------------------------

    def _get_store(self, component: WorkingMemoryComponent) -> OrderedDict:
        if component == WorkingMemoryComponent.PHONOLOGICAL_LOOP:
            return self._phonological_loop
        elif component == WorkingMemoryComponent.VISUOSPATIAL_SKETCHPAD:
            return self._visuospatial
        return self._episodic_buffer

    def delete(self, item_id: str) -> bool:
        result = super().delete(item_id)
        for store in (self._phonological_loop, self._visuospatial, self._episodic_buffer):
            store.pop(item_id, None)
        self._interference_matrix.pop(item_id, None)
        return result


# ---------------------------------------------------------------------------
# 5. ShortTermMemory
# ---------------------------------------------------------------------------

class ShortTermMemory(MemoryStore):
    """
    Medium-duration store (~15-30 seconds without rehearsal).
    Features:
      - Chunking: groups related items
      - Primacy / recency effects in serial position
      - Displacement when capacity exceeded
    """

    def __init__(self, capacity: int = 30, duration: float = 30.0, chunk_size: int = 4):
        super().__init__(capacity=capacity)
        self.duration = duration
        self.chunk_size = chunk_size
        self._serial_order: List[str] = []  # maintains insertion order

    def add(self, item: MemoryItem) -> bool:
        if self.size() >= self.capacity:
            self._displace()
        self._items[item.item_id] = item
        self._serial_order.append(item.item_id)
        item.metadata["serial_position"] = len(self._serial_order) - 1
        return True

    def retrieve_by_serial_position(self, position: int) -> Optional[MemoryItem]:
        if 0 <= position < len(self._serial_order):
            item_id = self._serial_order[position]
            return self._items.get(item_id)
        return None

    def primacy_recency_curve(self) -> List[float]:
        """
        Compute activation based on serial position.
        Primacy effect: first items have higher activation.
        Recency effect: last items have higher activation.
        Modeled as U-shaped curve.
        """
        n = len(self._serial_order)
        if n == 0:
            return []
        curve = []
        for i in range(n):
            primacy = math.exp(-0.3 * i)       # decays from start
            recency = math.exp(-0.3 * (n - 1 - i))  # decays from end
            activation = 0.5 * primacy + 0.5 * recency
            curve.append(activation)
        return curve

    def get_active_items(self) -> List[Tuple[MemoryItem, float]]:
        """Return items with their activation scores, filtering expired."""
        now = time.time()
        curve = self.primacy_recency_curve()
        active = []
        expired_ids = []
        for idx, item_id in enumerate(self._serial_order):
            item = self._items.get(item_id)
            if item is None:
                continue
            age = now - item.timestamp
            if age > self.duration:
                expired_ids.append(item_id)
                continue
            time_decay = math.exp(-age / self.duration)
            serial_activation = curve[idx] if idx < len(curve) else 0.5
            activation = time_decay * serial_activation * (0.5 + 0.5 * item.importance)
            active.append((item, activation))
        for eid in expired_ids:
            self.delete(eid)
            if eid in self._serial_order:
                self._serial_order.remove(eid)
        active.sort(key=lambda x: x[1], reverse=True)
        return active

    def chunk_items(self, items: List[MemoryItem]) -> List[List[MemoryItem]]:
        """Group items into chunks based on similarity."""
        if not items:
            return []
        chunks = []
        current_chunk = [items[0]]
        for item in items[1:]:
            # Check if item is similar enough to current chunk centroid
            centroid = self._chunk_centroid(current_chunk)
            if centroid and item.similarity_to(MemoryItem(content="", embedding=centroid)) > 0.5:
                current_chunk.append(item)
            else:
                chunks.append(current_chunk)
                current_chunk = [item]
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _chunk_centroid(self, items: List[MemoryItem]) -> Optional[List[float]]:
        if not items or not items[0].embedding:
            return None
        dim = len(items[0].embedding)
        centroid = [0.0] * dim
        count = 0
        for item in items:
            if item.embedding:
                for i in range(dim):
                    centroid[i] += item.embedding[i]
                count += 1
        if count == 0:
            return None
        return [c / count for c in centroid]

    def _displace(self):
        """Remove oldest (FIFO) item."""
        if self._serial_order:
            oldest = self._serial_order.pop(0)
            self.delete(oldest)

    def delete(self, item_id: str) -> bool:
        result = super().delete(item_id)
        if item_id in self._serial_order:
            self._serial_order.remove(item_id)
        return result


# ---------------------------------------------------------------------------
# 6. LongTermMemory
# ---------------------------------------------------------------------------

class EpisodicMemory(MemoryStore):
    """
    Event-based memory: stores what happened, when, where.
    Retrieval uses temporal context and content similarity.
    """

    def __init__(self, capacity: int = 100000):
        super().__init__(capacity=capacity)
        self._temporal_index: List[Tuple[float, str]] = []  # (timestamp, item_id)

    def add(self, item: MemoryItem) -> bool:
        if self.size() >= self.capacity:
            self._evict_least_important()
        item.memory_type = "episodic"
        self._items[item.item_id] = item
        self._temporal_index.append((item.timestamp, item.item_id))
        self._temporal_index.sort()
        return True

    def retrieve_by_time_range(self, start: float, end: float) -> List[MemoryItem]:
        """Retrieve episodic memories within a time range."""
        results = []
        for ts, item_id in self._temporal_index:
            if start <= ts <= end:
                item = self._items.get(item_id)
                if item:
                    results.append(item)
        return results

    def retrieve_recent(self, n: int = 10) -> List[MemoryItem]:
        """Retrieve the n most recent episodic memories."""
        recent = sorted(self._temporal_index, key=lambda x: x[0], reverse=True)[:n]
        return [self._items[iid] for _, iid in recent if iid in self._items]

    def retrieve_by_context(self, context_embedding: List[float],
                            time_weight: float = 0.3,
                            top_k: int = 10) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieve memories by content similarity weighted with temporal recency.
        """
        now = time.time()
        max_age = max((now - item.timestamp for item in self._items.values()), default=1.0)
        scored = []
        for item in self._items.values():
            content_sim = _cosine_similarity(context_embedding, item.embedding) if item.embedding else 0.0
            recency = 1.0 - ((now - item.timestamp) / max_age)
            score = (1.0 - time_weight) * content_sim + time_weight * recency
            scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _evict_least_important(self):
        if not self._items:
            return
        # Importance = access_count * importance_score * recency
        now = time.time()
        weakest = min(self._items,
                      key=lambda k: self._items[k].access_count * self._items[k].importance
                                    * math.exp(-0.0001 * (now - self._items[k].timestamp)))
        self.delete(weakest)
        self._temporal_index = [(t, i) for t, i in self._temporal_index if i != weakest]


class SemanticMemory(MemoryStore):
    """
    Fact-based memory with concept graph and spreading activation.
    """

    def __init__(self, capacity: int = 50000):
        super().__init__(capacity=capacity)
        # Concept graph: node -> {neighbor: weight}
        self._concept_graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Concept -> item_id mapping
        self._concept_to_items: Dict[str, List[str]] = defaultdict(list)

    def add(self, item: MemoryItem) -> bool:
        if self.size() >= self.capacity:
            self._evict_least_accessed()
        item.memory_type = "semantic"
        self._items[item.item_id] = item
        # Extract concepts from metadata or content
        concepts = item.metadata.get("concepts", self._extract_concepts(item.content))
        for concept in concepts:
            self._concept_to_items[concept].append(item.item_id)
            # Link concepts together
            for other_concept in concepts:
                if other_concept != concept:
                    self._concept_graph[concept][other_concept] += 0.1
        return True

    def add_concept_link(self, concept_a: str, concept_b: str, weight: float = 1.0):
        """Manually add a link between two concepts."""
        self._concept_graph[concept_a][concept_b] += weight
        self._concept_graph[concept_b][concept_a] += weight

    def spreading_activation(self, seed_concepts: List[str],
                             max_steps: int = 3,
                             decay: float = 0.5,
                             activation_threshold: float = 0.01) -> Dict[str, float]:
        """
        Spreading activation from seed concepts through the concept graph.
        Returns concept -> activation level.
        """
        activation = {}
        # Initialize seed activations
        for concept in seed_concepts:
            activation[concept] = 1.0
        # Spread
        for step in range(max_steps):
            new_activation = dict(activation)
            for concept, act in activation.items():
                if act < activation_threshold:
                    continue
                for neighbor, weight in self._concept_graph.get(concept, {}).items():
                    spread = act * weight * decay
                    new_activation[neighbor] = new_activation.get(neighbor, 0.0) + spread
            activation = new_activation
        # Filter by threshold
        return {c: a for c, a in activation.items() if a >= activation_threshold}

    def retrieve_by_concepts(self, concepts: List[str], top_k: int = 10) -> List[Tuple[MemoryItem, float]]:
        """Retrieve items related to given concepts via spreading activation."""
        activation = self.spreading_activation(concepts)
        scored_items = []
        seen = set()
        for concept, act in activation.items():
            for item_id in self._concept_to_items.get(concept, []):
                if item_id in seen:
                    continue
                seen.add(item_id)
                item = self._items.get(item_id)
                if item:
                    scored_items.append((item, act))
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]

    def _extract_concepts(self, content: str) -> List[str]:
        """Simple concept extraction: split on whitespace, lowercase, filter short."""
        words = content.lower().split()
        return [w for w in words if len(w) > 3]

    def _evict_least_accessed(self):
        if not self._items:
            return
        weakest = min(self._items, key=lambda k: self._items[k].access_count)
        self.delete(weakest)


class ProceduralMemory(MemoryStore):
    """
    Skill-based memory with reinforcement learning integration.
    Stores (state, action, reward) tuples and supports policy retrieval.
    """

    def __init__(self, capacity: int = 20000, learning_rate: float = 0.1,
                 discount_factor: float = 0.95):
        super().__init__(capacity=capacity)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # Q-table approximation: state_hash -> {action: value}
        self._q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Skill -> list of item_ids
        self._skill_index: Dict[str, List[str]] = defaultdict(list)

    def add(self, item: MemoryItem) -> bool:
        if self.size() >= self.capacity:
            self._evict_worst_skill()
        item.memory_type = "procedural"
        self._items[item.item_id] = item
        skill = item.metadata.get("skill", "default")
        self._skill_index[skill].append(item.item_id)
        # Update Q-value if experience tuple is provided
        state = item.metadata.get("state")
        action = item.metadata.get("action")
        reward = item.metadata.get("reward")
        next_state = item.metadata.get("next_state")
        if state is not None and action is not None and reward is not None:
            self._update_q_value(str(state), str(action), reward,
                                 str(next_state) if next_state else None)
        return True

    def _update_q_value(self, state: str, action: str, reward: float,
                        next_state: Optional[str] = None):
        """Q-learning update: Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))"""
        current_q = self._q_table[state][action]
        if next_state and self._q_table[next_state]:
            max_next_q = max(self._q_table[next_state].values())
            target = reward + self.discount_factor * max_next_q
        else:
            target = reward
        self._q_table[state][action] = current_q + self.learning_rate * (target - current_q)

    def get_best_action(self, state: str) -> Optional[str]:
        """Retrieve the best action for a given state."""
        if state not in self._q_table or not self._q_table[state]:
            return None
        return max(self._q_table[state], key=self._q_table[state].get)

    def get_q_value(self, state: str, action: str) -> float:
        return self._q_table[state][action]

    def retrieve_skill(self, skill_name: str, top_k: int = 5) -> List[MemoryItem]:
        """Retrieve memories associated with a skill."""
        item_ids = self._skill_index.get(skill_name, [])
        items = [self._items[iid] for iid in item_ids if iid in self._items]
        items.sort(key=lambda m: m.access_count * m.importance, reverse=True)
        return items[:top_k]

    def _evict_worst_skill(self):
        if not self._items:
            return
        weakest = min(self._items,
                      key=lambda k: self._items[k].importance * (1 + self._items[k].access_count))
        self.delete(weakest)


class LongTermMemory:
    """
    Persistent storage combining episodic, semantic, and procedural memory.
    """

    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()

    def consolidate(self, item: MemoryItem, memory_type: str = "episodic") -> bool:
        """Consolidate a memory item into the appropriate LTM subsystem."""
        if memory_type == "episodic":
            return self.episodic.add(item)
        elif memory_type == "semantic":
            return self.semantic.add(item)
        elif memory_type == "procedural":
            return self.procedural.add(item)
        return False

    def retrieve(self, query: List[float], top_k: int = 10,
                 episodic_weight: float = 0.4,
                 semantic_weight: float = 0.4,
                 procedural_weight: float = 0.2) -> List[Tuple[MemoryItem, float, str]]:
        """Retrieve from all LTM subsystems with weighted fusion."""
        epi_results = self.episodic.search(query, top_k * 2)
        sem_results = self.semantic.search(query, top_k * 2)
        proc_results = self.procedural.search(query, top_k * 2)
        fused = {}
        for item, score in epi_results:
            fused[item.item_id] = (item, score * episodic_weight, "episodic")
        for item, score in sem_results:
            if item.item_id in fused:
                old = fused[item.item_id]
                fused[item.item_id] = (item, old[1] + score * semantic_weight, old[2])
            else:
                fused[item.item_id] = (item, score * semantic_weight, "semantic")
        for item, score in proc_results:
            if item.item_id in fused:
                old = fused[item.item_id]
                fused[item.item_id] = (item, old[1] + score * procedural_weight, old[2])
            else:
                fused[item.item_id] = (item, score * procedural_weight, "procedural")
        results = [(v[0], v[1], v[2]) for v in fused.values()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def total_size(self) -> int:
        return self.episodic.size() + self.semantic.size() + self.procedural.size()


# ---------------------------------------------------------------------------
# 7. HierarchicalMemoryManager
# ---------------------------------------------------------------------------

class ConsolidationStrategy(Enum):
    ALL = auto()
    IMPORTANT_ONLY = auto()
    REHEARSED_ONLY = auto()
    EMOTIONAL_ONLY = auto()


class HierarchicalMemoryManager:
    """
    Orchestrates all memory levels with:
      - Consolidation pipeline (STM -> LTM during "sleep")
      - Multi-modal retrieval
      - Ebbinghaus forgetting curves
      - Memory reconsolidation (retrieval-dependent updating)
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

        # Memory hierarchy
        self.sensory = {
            SensoryBufferType.ICONIC: SensoryMemory(SensoryBufferType.ICONIC),
            SensoryBufferType.ECHOIC: SensoryMemory(SensoryBufferType.ECHOIC),
            SensoryBufferType.HAPTIC: SensoryMemory(SensoryBufferType.HAPTIC),
        }
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

        # Forgetting curve parameters (Ebbinghaus)
        self._forgetting_base_strength = 1.0
        self._forgetting_decay_rate = 1.5   # controls steepness

        # Reconsolidation tracking
        self._reconsolidation_history: Dict[str, List[float]] = defaultdict(list)

        # Statistics
        self._stats = {
            "total_consolidated": 0,
            "total_forgotten": 0,
            "total_retrieved": 0,
            "consolidation_runs": 0,
        }

    # -----------------------------------------------------------------------
    # Encoding pipeline: sensory -> working -> STM
    # -----------------------------------------------------------------------

    def perceive(self, content: str, embedding: Optional[List[float]] = None,
                 sensory_type: SensoryBufferType = SensoryBufferType.ECHOIC,
                 importance: float = 0.5,
                 emotional_valence: float = 0.0,
                 metadata: Optional[Dict] = None) -> MemoryItem:
        """Full perception pipeline: create item, register in sensory memory."""
        if embedding is None:
            embedding = _random_vector(self.embedding_dim)
        item = MemoryItem(
            content=content,
            embedding=embedding,
            importance=importance,
            emotional_valence=emotional_valence,
            metadata=metadata or {},
        )
        self.sensory[sensory_type].add(item)
        return item

    def attend(self, item: MemoryItem,
               component: WorkingMemoryComponent = WorkingMemoryComponent.EPISODIC_BUFFER) -> bool:
        """Move item from sensory to working memory (attention)."""
        return self.working.add(item, component)

    def encode(self, item: MemoryItem) -> bool:
        """Move item from working memory to short-term memory."""
        return self.short_term.add(item)

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(self, query: List[float], top_k: int = 10) -> List[Tuple[MemoryItem, float, str]]:
        """
        Hierarchical retrieval: check STM first, then LTM.
        Returns (item, score, source) tuples.
        """
        results = []
        # Check working memory
        for item in self.working.get_all_active():
            sim = _cosine_similarity(query, item.embedding) if item.embedding else 0.0
            if sim > 0.1:
                results.append((item, sim, "working"))

        # Check short-term memory
        for item, activation in self.short_term.get_active_items():
            sim = _cosine_similarity(query, item.embedding) if item.embedding else 0.0
            combined = 0.6 * sim + 0.4 * activation
            if combined > 0.1:
                results.append((item, combined, "short_term"))

        # Check long-term memory
        ltm_results = self.long_term.retrieve(query, top_k)
        results.extend(ltm_results)

        # Sort by score, deduplicate
        seen = set()
        unique = []
        for item, score, source in sorted(results, key=lambda x: x[1], reverse=True):
            if item.item_id not in seen:
                seen.add(item.item_id)
                unique.append((item, score, source))
                item.touch()
                self._stats["total_retrieved"] += 1

        return unique[:top_k]

    def recall_recent(self, n: int = 10) -> List[MemoryItem]:
        """Recall recent memories across all stores."""
        items = []
        items.extend(self.working.get_all_active())
        items.extend([i for i, _ in self.short_term.get_active_items()])
        items.extend(self.long_term.episodic.retrieve_recent(n))
        items.sort(key=lambda m: m.timestamp, reverse=True)
        return items[:n]

    # -----------------------------------------------------------------------
    # Consolidation (STM -> LTM)
    # -----------------------------------------------------------------------

    def consolidate(self, strategy: ConsolidationStrategy = ConsolidationStrategy.ALL,
                    importance_threshold: float = 0.3,
                    min_access_count: int = 1) -> int:
        """
        Consolidation pass: transfer eligible STM items to LTM.
        Simulates what happens during sleep/rest.
        """
        count = 0
        # Gather candidates from short-term memory
        candidates = self.short_term.get_active_items()
        for item, activation in candidates:
            if not self._is_consolidation_eligible(item, strategy, importance_threshold, min_access_count):
                continue
            # Determine LTM type
            mem_type = self._classify_memory_type(item)
            success = self.long_term.consolidate(item, mem_type)
            if success:
                count += 1
                self._stats["total_consolidated"] += 1
                # Apply Ebbinghaus forgetting curve
                item.metadata["consolidation_time"] = time.time()
                item.metadata["initial_strength"] = activation
        self._stats["consolidation_runs"] += 1
        return count

    def _is_consolidation_eligible(self, item: MemoryItem,
                                    strategy: ConsolidationStrategy,
                                    importance_threshold: float,
                                    min_access_count: int) -> bool:
        if strategy == ConsolidationStrategy.ALL:
            return True
        elif strategy == ConsolidationStrategy.IMPORTANT_ONLY:
            return item.importance >= importance_threshold
        elif strategy == ConsolidationStrategy.REHEARSED_ONLY:
            return item.access_count >= min_access_count
        elif strategy == ConsolidationStrategy.EMOTIONAL_ONLY:
            return abs(item.emotional_valence) > 0.3
        return False

    def _classify_memory_type(self, item: MemoryItem) -> str:
        """Classify which LTM subsystem a memory belongs to."""
        if item.metadata.get("skill"):
            return "procedural"
        if item.metadata.get("concepts"):
            return "semantic"
        if item.metadata.get("event_time") or item.metadata.get("location"):
            return "episodic"
        # Default: high emotional valence -> episodic, else semantic
        if abs(item.emotional_valence) > 0.3:
            return "episodic"
        return "semantic"

    # -----------------------------------------------------------------------
    # Ebbinghaus Forgetting Curve
    # -----------------------------------------------------------------------

    def ebbinghaus_retention(self, item: MemoryItem) -> float:
        """
        Compute retention strength using the Ebbinghaus forgetting curve.
        R(t) = S * exp(-k * t)
        where S = initial strength, k = decay rate, t = time since consolidation.
        Modified by repetition (each access resets the curve partially).
        """
        consolidation_time = item.metadata.get("consolidation_time", item.timestamp)
        initial_strength = item.metadata.get("initial_strength", 0.5)
        elapsed = time.time() - consolidation_time
        # Base retention
        retention = initial_strength * math.exp(-self._forgetting_decay_rate * elapsed / 3600.0)
        # Boost from repetitions (access_count)
        repetition_boost = min(item.access_count * 0.1, 0.5)
        # Emotional modulation
        emotional_mod = 1.0 + 0.3 * abs(item.emotional_valence)
        # Importance modulation
        importance_mod = 0.5 + 0.5 * item.importance
        final_retention = min(retention * (1 + repetition_boost) * emotional_mod * importance_mod, 1.0)
        return final_retention

    def apply_forgetting(self, retention_threshold: float = 0.05) -> int:
        """
        Apply forgetting: remove LTM items whose retention drops below threshold.
        Returns number of forgotten items.
        """
        forgotten = 0
        for store in (self.long_term.episodic, self.long_term.semantic, self.long_term.procedural):
            to_remove = []
            for item in store.all_items():
                retention = self.ebbinghaus_retention(item)
                if retention < retention_threshold:
                    to_remove.append(item.item_id)
            for item_id in to_remove:
                store.delete(item_id)
                forgotten += 1
        self._stats["total_forgotten"] += forgotten
        return forgotten

    # -----------------------------------------------------------------------
    # Memory Reconsolidation
    # -----------------------------------------------------------------------

    def reconsolidate(self, item: MemoryItem, new_information: Optional[Dict] = None) -> bool:
        """
        Reconsolidation: when a memory is retrieved, it becomes labile and
        can be updated with new information before being re-stored.
        """
        # Record retrieval for reconsolidation tracking
        self._reconsolidation_history[item.item_id].append(time.time())

        # Apply new information
        if new_information:
            for key, value in new_information.items():
                if key == "importance":
                    # Blend old and new importance
                    item.importance = 0.7 * item.importance + 0.3 * value
                elif key == "emotional_valence":
                    item.emotional_valence = 0.7 * item.emotional_valence + 0.3 * value
                elif key == "content":
                    item.metadata["previous_content"] = item.content
                    item.content = value
                else:
                    item.metadata[key] = value

        # Reset consolidation time (memory is re-consolidated)
        item.metadata["consolidation_time"] = time.time()
        item.metadata["initial_strength"] = self.ebbinghaus_retention(item)
        item.metadata["reconsolidation_count"] = item.metadata.get("reconsolidation_count", 0) + 1

        # Update in LTM
        if item.memory_type == "episodic":
            return self.long_term.episodic.update(item.item_id,
                                                   content=item.content,
                                                   importance=item.importance,
                                                   emotional_valence=item.emotional_valence,
                                                   metadata=item.metadata)
        elif item.memory_type == "semantic":
            return self.long_term.semantic.update(item.item_id,
                                                   content=item.content,
                                                   importance=item.importance,
                                                   metadata=item.metadata)
        elif item.memory_type == "procedural":
            return self.long_term.procedural.update(item.item_id,
                                                     content=item.content,
                                                     importance=item.importance,
                                                     metadata=item.metadata)
        return False

    def get_reconsolidation_strength(self, item: MemoryItem) -> float:
        """
        Memories that have been reconsolidated many times become more stable.
        Stability increases logarithmically with reconsolidation count.
        """
        count = item.metadata.get("reconsolidation_count", 0)
        return 1.0 + 0.5 * math.log(1 + count)

    # -----------------------------------------------------------------------
    # Statistics & Diagnostics
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "sensory": {k.name: v.size() for k, v in self.sensory.items()},
            "working_memory": self.working.size(),
            "short_term_memory": self.short_term.size(),
            "long_term_memory": {
                "episodic": self.long_term.episodic.size(),
                "semantic": self.long_term.semantic.size(),
                "procedural": self.long_term.procedural.size(),
                "total": self.long_term.total_size(),
            },
        }

    def memory_report(self) -> str:
        """Generate a human-readable memory status report."""
        stats = self.get_stats()
        lines = [
            "=" * 60,
            "  HIERARCHICAL MEMORY SYSTEM - STATUS REPORT",
            "=" * 60,
            f"  Sensory Memory:",
        ]
        for buf_type, count in stats["sensory"].items():
            lines.append(f"    {buf_type}: {count} items")
        lines.extend([
            f"  Working Memory: {stats['working_memory']} items",
            f"  Short-Term Memory: {stats['short_term_memory']} items",
            f"  Long-Term Memory:",
            f"    Episodic:   {stats['long_term_memory']['episodic']} items",
            f"    Semantic:   {stats['long_term_memory']['semantic']} items",
            f"    Procedural: {stats['long_term_memory']['procedural']} items",
            f"    Total:      {stats['long_term_memory']['total']} items",
            "-" * 60,
            f"  Total consolidated: {stats['total_consolidated']}",
            f"  Total forgotten:    {stats['total_forgotten']}",
            f"  Total retrieved:    {stats['total_retrieved']}",
            f"  Consolidation runs: {stats['consolidation_runs']}",
            "=" * 60,
        ])
        return "\n".join(lines)
