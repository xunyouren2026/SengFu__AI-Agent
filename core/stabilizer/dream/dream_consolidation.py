"""
Dream / Consolidation System
=============================
Simulates sleep-based memory processing inspired by neuroscience:
  - DreamPhase:       REM / NREM cycle modeling, hippocampal replay
  - MemoryReplay:     random, prioritized, and generative experience replay
  - DreamGenerator:   creative recombination, counterfactual simulation
  - ConsolidationManager: full sleep pipeline with synaptic homeostasis
                          and Complementary Learning Systems (CLS) theory

Pure Python -- only stdlib modules (math, random, time, collections).
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _l2_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _normalize(v: List[float]) -> List[float]:
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        return v[:]
    return [x / mag for x in v]


def _mean_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    result = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            result[i] += v[i]
    return [x / len(vectors) for x in result]


def _random_vector(dim: int) -> List[float]:
    v = [random.gauss(0, 1) for _ in range(dim)]
    return _normalize(v)


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryTrace:
    """A memory trace used during consolidation."""
    trace_id: str
    content: str
    embedding: List[float]
    source: str = ""              # episodic, semantic, procedural
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    emotional_valence: float = 0.0
    access_count: int = 0
    hippocampal_index: int = -1   # index in hippocampal buffer
    neocortical_strength: float = 0.0
    replay_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamEvent:
    """A single event within a dream sequence."""
    event_id: str
    source_traces: List[str]      # trace_ids that contributed
    embedding: List[float]
    content: str
    novelty: float = 0.0          # how novel / creative the combination is
    emotional_intensity: float = 0.0
    coherence: float = 0.0        # how coherent the dream fragment is
    phase: str = ""               # REM or NREM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SleepCycle:
    """Record of a single REM/NREM cycle."""
    cycle_number: int
    phase: str                    # REM or NREM
    duration: float               # seconds
    events_generated: int
    memories_replayed: int
    memories_consolidated: int
    synaptic_change: float        # net synaptic weight change


# ---------------------------------------------------------------------------
# 1. DreamPhase -- Sleep phase simulation
# ---------------------------------------------------------------------------

class SleepPhase(Enum):
    NREM1 = auto()    # light sleep
    NREM2 = auto()    # spindle sleep
    NREM3 = auto()    # slow-wave (deep) sleep
    REM = auto()      # rapid eye movement


class DreamPhase:
    """
    Simulates sleep phases with realistic REM/NREM cycling.

    A typical 90-minute sleep cycle:
      N1 (5%) -> N2 (45%) -> N3 (25%) -> REM (25%)

    NREM: slow-wave sleep, dominant in early cycles, hippocampal replay
    REM:  dream sleep, dominant in late cycles, creative recombination
    """

    # Phase durations as fractions of a cycle
    PHASE_DISTRIBUTION = {
        SleepPhase.NREM1: 0.05,
        SleepPhase.NREM2: 0.45,
        SleepPhase.NREM3: 0.25,
        SleepPhase.REM: 0.25,
    }

    # REM proportion increases across cycles (first cycle ~10%, last ~40%)
    REM_PROGRESSION = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    def __init__(self, cycle_duration: float = 90.0 * 60,  # 90 minutes in seconds
                 num_cycles: int = 5):
        self.cycle_duration = cycle_duration
        self.num_cycles = num_cycles
        self._current_cycle = 0
        self._current_phase = SleepPhase.NREM1
        self._phase_start_time = 0.0
        self._cycle_log: List[SleepCycle] = []

    def get_phase_sequence(self) -> List[Tuple[SleepPhase, float]]:
        """
        Generate the full sequence of (phase, duration) for all sleep cycles.
        """
        sequence = []
        for cycle in range(self.num_cycles):
            rem_frac = self.REM_PROGRESSION[min(cycle, len(self.REM_PROGRESSION) - 1)]
            nrem_frac = 1.0 - rem_frac
            # Distribute NREM across sub-phases
            nrem1_dur = self.cycle_duration * 0.05 * nrem_frac
            nrem2_dur = self.cycle_duration * (0.45 / 0.75) * nrem_frac
            nrem3_dur = self.cycle_duration * (0.25 / 0.75) * nrem_frac
            rem_dur = self.cycle_duration * rem_frac
            sequence.extend([
                (SleepPhase.NREM1, nrem1_dur),
                (SleepPhase.NREM2, nrem2_dur),
                (SleepPhase.NREM3, nrem3_dur),
                (SleepPhase.REM, rem_dur),
            ])
        return sequence

    def hippocampal_replay_rate(self, phase: SleepPhase) -> float:
        """
        Rate of hippocampal replay depends on sleep phase.
        Highest during NREM3 (slow-wave sleep), moderate during REM.
        """
        rates = {
            SleepPhase.NREM1: 0.1,
            SleepPhase.NREM2: 0.4,
            SleepPhase.NREM3: 0.9,   # sharp-wave ripples
            SleepPhase.REM: 0.5,
        }
        return rates[phase]

    def replay_speed_factor(self, phase: SleepPhase) -> float:
        """
        Hippocampal replay is compressed in time.
        During SWS, replay can be 10-20x faster than real-time experience.
        """
        speeds = {
            SleepPhase.NREM1: 1.0,
            SleepPhase.NREM2: 5.0,
            SleepPhase.NREM3: 15.0,  # highly compressed
            SleepPhase.REM: 3.0,
        }
        return speeds[phase]

    def plasticity_level(self, phase: SleepPhase) -> float:
        """
        Synaptic plasticity level during each phase.
        NREM3: high plasticity for system consolidation
        REM: moderate plasticity for memory integration
        """
        levels = {
            SleepPhase.NREM1: 0.1,
            SleepPhase.NREM2: 0.4,
            SleepPhase.NREM3: 0.8,
            SleepPhase.REM: 0.6,
        }
        return levels[phase]

    def should_replay(self, phase: SleepPhase) -> bool:
        """Determine if a replay event should occur based on phase."""
        rate = self.hippocampal_replay_rate(phase)
        return random.random() < rate

    def get_cycle_log(self) -> List[SleepCycle]:
        return self._cycle_log


# ---------------------------------------------------------------------------
# 2. MemoryReplay -- Experience replay during consolidation
# ---------------------------------------------------------------------------

class ReplayStrategy(Enum):
    RANDOM = auto()
    TEMPORAL = auto()         # replay in temporal order
    PRIORITIZED = auto()      # by importance / TD-error
    SEQUENTIAL = auto()       # replay sequences (temporal contiguity)
    REVERSE = auto()          # reverse temporal order


class MemoryReplay:
    """
    Implements various experience replay strategies during sleep consolidation.
    """

    def __init__(self, replay_buffer_capacity: int = 5000):
        self._buffer: List[MemoryTrace] = []
        self._capacity = replay_buffer_capacity
        self._replay_history: List[Dict[str, Any]] = []
        self._priority_sum = 0.0

    def add_trace(self, trace: MemoryTrace):
        """Add a memory trace to the replay buffer."""
        if len(self._buffer) >= self._capacity:
            # Remove lowest priority trace
            self._buffer.sort(key=lambda t: t.importance)
            removed = self._buffer.pop(0)
            self._priority_sum -= removed.importance
        self._buffer.append(trace)
        self._priority_sum += trace.importance

    def replay(self, strategy: ReplayStrategy = ReplayStrategy.PRIORITIZED,
               batch_size: int = 32) -> List[MemoryTrace]:
        """Replay a batch of memory traces using the specified strategy."""
        if not self._buffer:
            return []

        if strategy == ReplayStrategy.RANDOM:
            return self._random_replay(batch_size)
        elif strategy == ReplayStrategy.TEMPORAL:
            return self._temporal_replay(batch_size)
        elif strategy == ReplayStrategy.PRIORITIZED:
            return self._prioritized_replay(batch_size)
        elif strategy == ReplayStrategy.SEQUENTIAL:
            return self._sequential_replay(batch_size)
        elif strategy == ReplayStrategy.REVERSE:
            return self._reverse_replay(batch_size)
        return []

    def _random_replay(self, batch_size: int) -> List[MemoryTrace]:
        """Uniform random sampling."""
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def _temporal_replay(self, batch_size: int) -> List[MemoryTrace]:
        """Replay traces in chronological order (simulates episodic replay)."""
        sorted_traces = sorted(self._buffer, key=lambda t: t.timestamp)
        # Pick a random starting point and take a contiguous sequence
        if len(sorted_traces) <= batch_size:
            return sorted_traces
        start = random.randint(0, len(sorted_traces) - batch_size)
        return sorted_traces[start:start + batch_size]

    def _prioritized_replay(self, batch_size: int) -> List[MemoryTrace]:
        """Sample proportional to importance (priority replay)."""
        if self._priority_sum == 0:
            return self._random_replay(batch_size)
        n = min(batch_size, len(self._buffer))
        sampled = []
        weights = [t.importance / self._priority_sum for t in self._buffer]
        # Weighted sampling without replacement
        remaining = list(range(len(self._buffer)))
        remaining_weights = weights[:]
        for _ in range(n):
            if not remaining:
                break
            total_w = sum(remaining_weights)
            if total_w == 0:
                idx = random.choice(remaining)
            else:
                r = random.random() * total_w
                cumulative = 0.0
                idx = remaining[0]
                for i, w in zip(remaining, remaining_weights):
                    cumulative += w
                    if cumulative >= r:
                        idx = i
                        break
            sampled.append(self._buffer[idx])
            pos = remaining.index(idx)
            remaining.pop(pos)
            remaining_weights.pop(pos)
        return sampled

    def _sequential_replay(self, batch_size: int) -> List[MemoryTrace]:
        """
        Replay temporally contiguous sequences.
        Finds sequences of traces that are close in time.
        """
        if len(self._buffer) <= batch_size:
            return sorted(self._buffer, key=lambda t: t.timestamp)
        sorted_traces = sorted(self._buffer, key=lambda t: t.timestamp)
        # Find a starting point, prefer points with temporal neighbors
        start = random.randint(0, len(sorted_traces) - batch_size)
        # Extend sequence by finding temporally close neighbors
        sequence = [sorted_traces[start]]
        candidates = sorted_traces[:start] + sorted_traces[start + 1:]
        candidates.sort(key=lambda t: abs(t.timestamp - sequence[-1].timestamp))
        for trace in candidates:
            if len(sequence) >= batch_size:
                break
            time_gap = abs(trace.timestamp - sequence[-1].timestamp)
            if time_gap < 3600:  # within 1 hour
                sequence.append(trace)
        return sequence

    def _reverse_replay(self, batch_size: int) -> List[MemoryTrace]:
        """Reverse temporal order replay (useful for credit assignment)."""
        sorted_traces = sorted(self._buffer, key=lambda t: t.timestamp, reverse=True)
        return sorted_traces[:batch_size]

    def update_trace(self, trace_id: str, **kwargs) -> bool:
        """Update a trace's properties after replay."""
        for trace in self._buffer:
            if trace.trace_id == trace_id:
                for k, v in kwargs.items():
                    if hasattr(trace, k):
                        setattr(trace, k, v)
                return True
        return False

    def record_replay(self, traces: List[MemoryTrace], strategy: ReplayStrategy,
                      phase: SleepPhase):
        """Record a replay event for analysis."""
        self._replay_history.append({
            "timestamp": time.time(),
            "strategy": strategy.name,
            "phase": phase.name,
            "num_traces": len(traces),
            "trace_ids": [t.trace_id for t in traces],
        })
        # Increment replay counts
        for trace in traces:
            trace.replay_count += 1
            trace.access_count += 1

    def get_buffer_size(self) -> int:
        return len(self._buffer)

    def get_replay_stats(self) -> Dict[str, Any]:
        if not self._buffer:
            return {"buffer_size": 0}
        return {
            "buffer_size": len(self._buffer),
            "avg_replay_count": sum(t.replay_count for t in self._buffer) / len(self._buffer),
            "avg_importance": sum(t.importance for t in self._buffer) / len(self._buffer),
            "total_replays": len(self._replay_history),
            "source_distribution": self._count_sources(),
        }

    def _count_sources(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for t in self._buffer:
            counts[t.source] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# 3. DreamGenerator -- Novel combinations from memories
# ---------------------------------------------------------------------------

class RecombinationStrategy(Enum):
    INTERPOLATION = auto()       # linear interpolation between memories
    ADDITIVE = auto()            # additive combination
    STRUCTURAL = auto()          # structure-preserving recombination
    COUNTERFACTUAL = auto()      # negate / modify specific dimensions
    ANALOGICAL = auto()          # map structure from one domain to another


class DreamGenerator:
    """
    Generates novel memory combinations during REM sleep.
    Implements creative recombination and counterfactual simulation.
    """

    def __init__(self, dim: int = 64, creativity: float = 0.5,
                 coherence_threshold: float = 0.3):
        self.dim = dim
        self.creativity = creativity       # 0 = conservative, 1 = wild
        self.coherence_threshold = coherence_threshold
        self._dream_log: List[DreamEvent] = []
        self._event_counter = 0

    def generate_dream_fragment(self, traces: List[MemoryTrace],
                                strategy: RecombinationStrategy = RecombinationStrategy.INTERPOLATION,
                                phase: SleepPhase = SleepPhase.REM) -> DreamEvent:
        """
        Generate a single dream fragment by combining memory traces.
        """
        if len(traces) < 1:
            return self._empty_dream(phase)
        if len(traces) == 1:
            return self._single_trace_dream(traces[0], phase)

        self._event_counter += 1
        source_ids = [t.trace_id for t in traces]

        if strategy == RecombinationStrategy.INTERPOLATION:
            embedding, novelty = self._interpolate(traces)
        elif strategy == RecombinationStrategy.ADDITIVE:
            embedding, novelty = self._additive_combine(traces)
        elif strategy == RecombinationStrategy.STRUCTURAL:
            embedding, novelty = self._structural_recombine(traces)
        elif strategy == RecombinationStrategy.COUNTERFACTUAL:
            embedding, novelty = self._counterfactual(traces)
        elif strategy == RecombinationStrategy.ANALOGICAL:
            embedding, novelty = self._analogical_map(traces)
        else:
            embedding, novelty = self._interpolate(traces)

        # Compute coherence: average pairwise similarity of source traces
        coherence = self._compute_coherence(traces)

        # Emotional intensity: blend source emotions with creativity modulation
        emotional_intensity = self._blend_emotions(traces)

        # Generate content description
        content = self._generate_content(traces, strategy)

        event = DreamEvent(
            event_id=f"dream_{self._event_counter:06d}",
            source_traces=source_ids,
            embedding=_normalize(embedding),
            content=content,
            novelty=min(novelty * (0.5 + 0.5 * self.creativity), 1.0),
            emotional_intensity=emotional_intensity,
            coherence=coherence,
            phase=phase.name,
            metadata={"strategy": strategy.name, "num_sources": len(traces)},
        )
        self._dream_log.append(event)
        return event

    def _interpolate(self, traces: List[MemoryTrace]) -> Tuple[List[float], float]:
        """Linear interpolation between memory embeddings."""
        weights = _softmax([t.importance for t in traces])
        result = [0.0] * self.dim
        for trace, w in zip(traces, weights):
            for i in range(min(len(trace.embedding), self.dim)):
                result[i] += w * trace.embedding[i]
        # Novelty: distance from nearest source
        novelty = min(_l2_distance(result, t.embedding) for t in traces
                      if t.embedding) if traces else 0.0
        novelty = min(novelty / 2.0, 1.0)  # normalize
        return result, novelty

    def _additive_combine(self, traces: List[MemoryTrace]) -> Tuple[List[float], float]:
        """Additive superposition of memory embeddings."""
        result = [0.0] * self.dim
        for trace in traces:
            for i in range(min(len(trace.embedding), self.dim)):
                result[i] += trace.embedding[i]
        # Normalize to prevent explosion
        result = _normalize(result)
        novelty = 1.0 - max(_cosine_similarity(result, t.embedding)
                            for t in traces if t.embedding) if traces else 0.0
        return result, novelty

    def _structural_recombine(self, traces: List[MemoryTrace]) -> Tuple[List[float], float]:
        """
        Structure-preserving recombination:
        Take structure (relationships) from one trace and content from another.
        """
        if len(traces) < 2:
            return self._interpolate(traces)
        # Use first trace as "structure", second as "content"
        structure = traces[0].embedding
        content = traces[1].embedding
        # Structural recombination: use magnitude from structure, direction from content
        s_mag = math.sqrt(sum(x * x for x in structure))
        c_norm = _normalize(content)
        result = [s_mag * x for x in c_norm]
        # Add noise proportional to creativity
        noise = [random.gauss(0, 0.1 * self.creativity) for _ in range(self.dim)]
        result = [r + n for r, n in zip(result, noise)]
        novelty = 1.0 - _cosine_similarity(result, structure)
        return result, novelty

    def _counterfactual(self, traces: List[MemoryTrace]) -> Tuple[List[float], float]:
        """
        Counterfactual simulation: negate or modify specific dimensions.
        "What if this aspect were different?"
        """
        if not traces:
            return [0.0] * self.dim, 0.0
        base = traces[0].embedding[:]
        # Select dimensions to modify (proportional to creativity)
        n_modify = max(1, int(self.dim * self.creativity * 0.3))
        dims_to_modify = random.sample(range(min(len(base), self.dim)), n_modify)
        for d in dims_to_modify:
            # Negate and add noise
            base[d] = -base[d] + random.gauss(0, 0.2)
        # If multiple traces, blend in aspects of others
        for trace in traces[1:]:
            blend_dims = random.sample(range(min(len(trace.embedding), self.dim)),
                                       max(1, int(self.dim * 0.1)))
            for d in blend_dims:
                base[d] = 0.5 * base[d] + 0.5 * trace.embedding[d]
        novelty = 1.0 - _cosine_similarity(base, traces[0].embedding)
        return base, novelty

    def _analogical_map(self, traces: List[MemoryTrace]) -> Tuple[List[float], float]:
        """
        Analogical mapping: find relational structure shared between traces
        and project it into a new space.
        """
        if len(traces) < 2:
            return self._interpolate(traces)
        # Compute relational structure as difference vectors
        base = traces[0].embedding[:]
        for i in range(1, len(traces)):
            diff = [traces[i].embedding[j] - base[j]
                    for j in range(min(len(base), len(traces[i].embedding), self.dim))]
            # Apply relational transform with creativity modulation
            scale = 0.5 + self.creativity * random.uniform(-0.5, 0.5)
            for j in range(min(len(diff), self.dim)):
                base[j] += diff[j] * scale * 0.3
        novelty = 1.0 - max(_cosine_similarity(base, t.embedding)
                            for t in traces if t.embedding) if traces else 0.0
        return base, novelty

    def _compute_coherence(self, traces: List[MemoryTrace]) -> float:
        """Average pairwise cosine similarity among source traces."""
        if len(traces) < 2:
            return 1.0
        similarities = []
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                sim = _cosine_similarity(traces[i].embedding, traces[j].embedding)
                similarities.append(sim)
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _blend_emotions(self, traces: List[MemoryTrace]) -> float:
        """Blend emotional valences, amplifying with creativity."""
        if not traces:
            return 0.0
        avg_valence = sum(t.emotional_valence for t in traces) / len(traces)
        # Creativity amplifies emotional intensity
        intensity = abs(avg_valence) * (1.0 + self.creativity * 0.5)
        return min(intensity, 1.0)

    def _generate_content(self, traces: List[MemoryTrace],
                          strategy: RecombinationStrategy) -> str:
        """Generate a textual description of the dream fragment."""
        parts = [t.content[:30] for t in traces[:3]]
        strategy_desc = {
            RecombinationStrategy.INTERPOLATION: "blend of",
            RecombinationStrategy.ADDITIVE: "fusion of",
            RecombinationStrategy.STRUCTURAL: "restructuring of",
            RecombinationStrategy.COUNTERFACTUAL: "counterfactual about",
            RecombinationStrategy.ANALOGICAL: "analogy between",
        }
        desc = strategy_desc.get(strategy, "combination of")
        return f"[{desc}] {' + '.join(parts)}"

    def _empty_dream(self, phase: SleepPhase) -> DreamEvent:
        self._event_counter += 1
        return DreamEvent(
            event_id=f"dream_{self._event_counter:06d}",
            source_traces=[],
            embedding=_random_vector(self.dim),
            content="[empty dream fragment]",
            phase=phase.name,
        )

    def _single_trace_dream(self, trace: MemoryTrace,
                            phase: SleepPhase) -> DreamEvent:
        self._event_counter += 1
        # Add noise to create a slightly modified version
        noisy = [e + random.gauss(0, 0.05 * self.creativity)
                 for e in trace.embedding]
        return DreamEvent(
            event_id=f"dream_{self._event_counter:06d}",
            source_traces=[trace.trace_id],
            embedding=_normalize(noisy),
            content=f"[replay] {trace.content[:50]}",
            novelty=0.1 * self.creativity,
            emotional_intensity=abs(trace.emotional_valence),
            coherence=1.0,
            phase=phase.name,
            metadata={"strategy": "single_replay"},
        )

    def generate_dream_sequence(self, traces: List[MemoryTrace],
                                num_events: int = 10,
                                phase: SleepPhase = SleepPhase.REM) -> List[DreamEvent]:
        """Generate a sequence of dream events forming a narrative."""
        if not traces:
            return []
        events = []
        available = list(traces)
        for _ in range(num_events):
            if not available:
                break
            # Select 1-3 traces for this event
            n_sources = min(random.randint(1, 3), len(available))
            selected = random.sample(available, n_sources)
            strategy = random.choice(list(RecombinationStrategy))
            event = self.generate_dream_fragment(selected, strategy, phase)
            events.append(event)
            # Occasionally reuse traces (persistence in dreams)
            if random.random() < 0.3:
                available = [t for t in available if t.trace_id not in event.source_traces]
                available.extend(selected[:1])  # keep one for continuity
            else:
                available = [t for t in available if t.trace_id not in event.source_traces]
            if not available:
                available = list(traces)  # reset
        return events

    def get_dream_log(self) -> List[DreamEvent]:
        return self._dream_log

    def get_dream_stats(self) -> Dict[str, Any]:
        if not self._dream_log:
            return {"total_events": 0}
        return {
            "total_events": len(self._dream_log),
            "avg_novelty": sum(e.novelty for e in self._dream_log) / len(self._dream_log),
            "avg_coherence": sum(e.coherence for e in self._dream_log) / len(self._dream_log),
            "avg_emotional_intensity": sum(e.emotional_intensity for e in self._dream_log) / len(self._dream_log),
            "phase_distribution": self._count_phases(),
        }

    def _count_phases(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for e in self._dream_log:
            counts[e.phase] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# 4. ConsolidationManager -- Full sleep/consolidation pipeline
# ---------------------------------------------------------------------------

class ConsolidationResult:
    """Result of a consolidation session."""

    def __init__(self):
        self.total_duration: float = 0.0
        self.cycles_completed: int = 0
        self.memories_replayed: int = 0
        self.memories_consolidated: int = 0
        self.dream_events_generated: int = 0
        self.synaptic_downscaling: float = 0.0
        self.cycle_details: List[SleepCycle] = []
        self.consolidated_trace_ids: List[str] = []
        self.dream_events: List[DreamEvent] = []


class ConsolidationManager:
    """
    Full sleep/consolidation pipeline implementing:
      - Complementary Learning Systems (CLS) theory
        (hippocampal fast learning + neocortical slow integration)
      - Synaptic Homeostasis Hypothesis (SHY)
        (synaptic downscaling during sleep to restore homeostasis)
      - Memory reorganization during sleep
    """

    def __init__(self, dim: int = 64,
                 hippocampal_capacity: int = 500,
                 neocortical_capacity: int = 10000,
                 learning_rate: float = 0.01,
                 consolidation_threshold: float = 0.7):
        self.dim = dim
        self.learning_rate = learning_rate
        self.consolidation_threshold = consolidation_threshold

        # Complementary Learning Systems
        self.hippocampal_buffer: List[MemoryTrace] = []  # fast learning
        self.neocortical_memory: List[MemoryTrace] = []  # slow learning
        self.hippocampal_capacity = hippocampal_capacity
        self.neocortical_capacity = neocortical_capacity

        # Synaptic weights (simplified model)
        self._synaptic_weights: Dict[str, float] = defaultdict(float)
        self._synaptic_baseline = 0.5

        # Subsystems
        self.dream_phase = DreamPhase()
        self.memory_replay = MemoryReplay()
        self.dream_generator = DreamGenerator(dim=dim)

        # Consolidation statistics
        self._total_consolidation_time = 0.0
        self._total_sessions = 0

    # -----------------------------------------------------------------------
    # CLS: Fast encoding into hippocampus
    # -----------------------------------------------------------------------

    def encode_fast(self, content: str, embedding: List[float],
                    source: str = "episodic",
                    importance: float = 0.5,
                    emotional_valence: float = 0.0,
                    metadata: Optional[Dict] = None) -> MemoryTrace:
        """
        Fast encoding into hippocampal buffer (pattern separation).
        New memories are rapidly encoded with distinct representations.
        """
        # Pattern separation: add noise to make representations more distinct
        separated = [e + random.gauss(0, 0.05) for e in embedding]
        separated = _normalize(separated)

        trace = MemoryTrace(
            trace_id=f"hc_{random.getrandbits(64):016x}",
            content=content,
            embedding=separated,
            source=source,
            importance=importance,
            emotional_valence=emotional_valence,
            hippocampal_index=len(self.hippocampal_buffer),
            neocortical_strength=0.0,
            metadata=metadata or {},
        )

        # Add to hippocampal buffer
        if len(self.hippocampal_buffer) >= self.hippocampal_capacity:
            # Displace weakest trace
            self.hippocampal_buffer.sort(key=lambda t: t.importance * (1 + t.access_count))
            displaced = self.hippocampal_buffer.pop(0)
            # Try to consolidate displaced trace before losing it
            self._attempt_emergency_consolidation(displaced)
        self.hippocampal_buffer.append(trace)

        # Initialize synaptic weight
        self._synaptic_weights[trace.trace_id] = self._synaptic_baseline

        return trace

    def _attempt_emergency_consolidation(self, trace: MemoryTrace):
        """Quick consolidation of a trace about to be displaced."""
        if trace.importance > 0.3:
            self._transfer_to_neocortex(trace)

    # -----------------------------------------------------------------------
    # CLS: Slow consolidation to neocortex
    # -----------------------------------------------------------------------

    def _transfer_to_neocortex(self, trace: MemoryTrace):
        """
        Transfer a hippocampal trace to neocortical slow memory.
        Implements the CLS interleaved learning: hippocampal traces are
        replayed to neocortex, which gradually integrates them.
        """
        # Check if similar memory already exists in neocortex
        best_match = None
        best_sim = 0.0
        for neo_trace in self.neocortical_memory:
            sim = _cosine_similarity(trace.embedding, neo_trace.embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = neo_trace

        if best_match and best_sim > self.consolidation_threshold:
            # Merge with existing neocortical memory (interleaved learning)
            alpha = self.learning_rate
            for i in range(min(len(trace.embedding), len(best_match.embedding), self.dim)):
                best_match.embedding[i] = (1 - alpha) * best_match.embedding[i] + alpha * trace.embedding[i]
            best_match.embedding = _normalize(best_match.embedding)
            best_match.importance = max(best_match.importance, trace.importance)
            best_match.neocortical_strength = min(best_match.neocortical_strength + 0.1, 1.0)
            best_match.access_count += trace.access_count
            # Update synaptic weight
            self._synaptic_weights[best_match.trace_id] = min(
                self._synaptic_weights[best_match.trace_id] + 0.05, 2.0)
        else:
            # Create new neocortical trace
            if len(self.neocortical_memory) >= self.neocortical_capacity:
                self._prune_neocortex()
            neo_trace = MemoryTrace(
                trace_id=f"neo_{random.getrandbits(64):016x}",
                content=trace.content,
                embedding=trace.embedding[:],
                source=trace.source,
                importance=trace.importance * 0.8,  # slight decay during transfer
                emotional_valence=trace.emotional_valence,
                neocortical_strength=0.1,  # starts weak
                metadata=dict(trace.metadata),
            )
            self.neocortical_memory.append(neo_trace)
            self._synaptic_weights[neo_trace.trace_id] = self._synaptic_baseline

    def _prune_neocortex(self):
        """Remove weakest neocortical memories."""
        self.neocortical_memory.sort(
            key=lambda t: t.neocortical_strength * t.importance * (1 + t.access_count))
        removed = self.neocortical_memory.pop(0)
        self._synaptic_weights.pop(removed.trace_id, None)

    # -----------------------------------------------------------------------
    # Synaptic Homeostasis Hypothesis (SHY)
    # -----------------------------------------------------------------------

    def synaptic_downscaling(self, scaling_factor: float = 0.9) -> float:
        """
        Implement synaptic downscaling (Tononi & Cirelli, 2006).
        During sleep, synaptic weights are proportionally downscaled to
        restore homeostasis. Stronger synapses are preserved better.
        
        Formula: w_new = w_old * scaling_factor + baseline * (1 - scaling_factor)
        This pushes all weights toward baseline while preserving relative differences.
        """
        total_change = 0.0
        for trace_id in list(self._synaptic_weights.keys()):
            old_w = self._synaptic_weights[trace_id]
            new_w = old_w * scaling_factor + self._synaptic_baseline * (1 - scaling_factor)
            self._synaptic_weights[trace_id] = new_w
            total_change += abs(new_w - old_w)

            # Update corresponding neocortical trace strength
            for neo_trace in self.neocortical_memory:
                if neo_trace.trace_id == trace_id:
                    neo_trace.neocortical_strength = new_w
                    break

        avg_change = total_change / max(len(self._synaptic_weights), 1)
        return avg_change

    def renormalize_synapses(self):
        """Ensure synaptic weights stay within valid range."""
        for trace_id in self._synaptic_weights:
            w = self._synaptic_weights[trace_id]
            w = max(0.01, min(w, 2.0))
            self._synaptic_weights[trace_id] = w

    # -----------------------------------------------------------------------
    # Full sleep pipeline
    # -----------------------------------------------------------------------

    def sleep(self, speed_factor: float = 100.0,
              creativity: float = 0.5) -> ConsolidationResult:
        """
        Run a full sleep/consolidation session.

        speed_factor: how much to accelerate simulation (100 = 100x real time)
        creativity: dream generator creativity level

        Pipeline per cycle:
          1. NREM: hippocampal replay -> neocortical consolidation
          2. REM:  dream generation -> creative recombination
          3. End:  synaptic downscaling
        """
        result = ConsolidationResult()
        start_time = time.time()

        self.dream_generator.creativity = creativity
        phase_sequence = self.dream_phase.get_phase_sequence()

        # Load hippocampal traces into replay buffer
        for trace in self.hippocampal_buffer:
            self.memory_replay.add_trace(trace)

        cycle_num = 0
        current_cycle_events = 0
        current_cycle_replayed = 0
        current_cycle_consolidated = 0
        current_cycle_start = 0.0
        current_cycle_phase = ""

        for phase, duration in phase_sequence:
            simulated_duration = duration / speed_factor
            is_new_cycle = (phase == SleepPhase.NREM1)

            if is_new_cycle and cycle_num > 0:
                # Record completed cycle
                cycle_record = SleepCycle(
                    cycle_number=cycle_num,
                    phase=current_cycle_phase,
                    duration=simulated_duration,
                    events_generated=current_cycle_events,
                    memories_replayed=current_cycle_replayed,
                    memories_consolidated=current_cycle_consolidated,
                    synaptic_change=0.0,
                )
                result.cycle_details.append(cycle_record)
                current_cycle_events = 0
                current_cycle_replayed = 0
                current_cycle_consolidated = 0

            if is_new_cycle:
                cycle_num += 1
                current_cycle_start = time.time()
                current_cycle_phase = phase.name

            # --- NREM processing: hippocampal replay and consolidation ---
            if phase in (SleepPhase.NREM2, SleepPhase.NREM3):
                replay_rate = self.dream_phase.hippocampal_replay_rate(phase)
                num_replays = max(1, int(len(self.hippocampal_buffer) * replay_rate * 0.1))

                # Choose replay strategy based on phase
                if phase == SleepPhase.NREM3:
                    strategy = ReplayStrategy.SEQUENTIAL  # temporal replay in SWS
                else:
                    strategy = ReplayStrategy.PRIORITIZED

                replayed_traces = self.memory_replay.replay(strategy, num_replays)
                self.memory_replay.record_replay(replayed_traces, strategy, phase)
                current_cycle_replayed += len(replayed_traces)
                result.memories_replayed += len(replayed_traces)

                # Consolidate replayed traces to neocortex
                for trace in replayed_traces:
                    self._transfer_to_neocortex(trace)
                    current_cycle_consolidated += 1
                    result.memories_consolidated += 1
                    result.consolidated_trace_ids.append(trace.trace_id)

                # Strengthen synaptic weights for replayed traces
                for trace in replayed_traces:
                    self._synaptic_weights[trace.trace_id] = min(
                        self._synaptic_weights.get(trace.trace_id, 0.5) + 0.02, 2.0)

            # --- REM processing: dream generation ---
            elif phase == SleepPhase.REM:
                # Generate dream events from hippocampal traces
                num_events = max(1, int(5 * creativity))
                if self.hippocampal_buffer:
                    dream_events = self.dream_generator.generate_dream_sequence(
                        self.hippocampal_buffer, num_events, phase)
                    result.dream_events.extend(dream_events)
                    current_cycle_events += len(dream_events)
                    result.dream_events_generated += len(dream_events)

                    # Novel dream events can create new memory traces
                    for event in dream_events:
                        if event.novelty > 0.3 and event.coherence > self.dream_generator.coherence_threshold:
                            # Consolidate creative insights
                            new_trace = MemoryTrace(
                                trace_id=f"insight_{random.getrandbits(64):016x}",
                                content=event.content,
                                embedding=event.embedding,
                                source="dream_insight",
                                importance=event.novelty * event.coherence,
                                emotional_valence=event.emotional_intensity,
                                neocortical_strength=0.2,
                                metadata={"dream_event_id": event.event_id},
                            )
                            self.neocortical_memory.append(new_trace)
                            self._synaptic_weights[new_trace.trace_id] = self._synaptic_baseline * 1.2

        # Record final cycle
        if cycle_num > 0:
            cycle_record = SleepCycle(
                cycle_number=cycle_num,
                phase=current_cycle_phase,
                duration=time.time() - current_cycle_start,
                events_generated=current_cycle_events,
                memories_replayed=current_cycle_replayed,
                memories_consolidated=current_cycle_consolidated,
                synaptic_change=0.0,
            )
            result.cycle_details.append(cycle_record)

        # --- Post-sleep: synaptic downscaling (SHY) ---
        downscale_amount = self.synaptic_downscaling(scaling_factor=0.85)
        self.renormalize_synapses()
        result.synaptic_downscaling = downscale_amount

        # Clean up: remove consolidated traces from hippocampus
        self._cleanup_hippocampus()

        result.cycles_completed = cycle_num
        result.total_duration = time.time() - start_time
        self._total_consolidation_time += result.total_duration
        self._total_sessions += 1

        return result

    def _cleanup_hippocampus(self):
        """
        Remove well-consolidated traces from hippocampal buffer.
        Traces with high neocortical strength can be freed from hippocampus.
        """
        remaining = []
        for trace in self.hippocampal_buffer:
            neo_strength = self._get_neocortical_strength(trace)
            if neo_strength > 0.7 and trace.replay_count > 3:
                # Well consolidated: remove from hippocampus
                pass
            else:
                remaining.append(trace)
        self.hippocampal_buffer = remaining

    def _get_neocortical_strength(self, trace: MemoryTrace) -> float:
        """Find the neocortical strength for a hippocampal trace."""
        best = 0.0
        for neo in self.neocortical_memory:
            sim = _cosine_similarity(trace.embedding, neo.embedding)
            if sim > 0.7:
                best = max(best, neo.neocortical_strength)
        return best

    # -----------------------------------------------------------------------
    # Memory reorganization
    # -----------------------------------------------------------------------

    def reorganize_memories(self) -> Dict[str, Any]:
        """
        Reorganize neocortical memories:
        1. Cluster similar memories
        2. Merge redundant memories
        3. Strengthen important associations
        """
        if len(self.neocortical_memory) < 2:
            return {"status": "too_few_memories"}

        merged_count = 0
        to_remove = set()

        # Simple agglomerative clustering
        n = len(self.neocortical_memory)
        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                sim = _cosine_similarity(
                    self.neocortical_memory[i].embedding,
                    self.neocortical_memory[j].embedding)
                if sim > 0.95:  # very similar: merge
                    # Merge j into i
                    alpha = 0.5
                    for k in range(min(len(self.neocortical_memory[i].embedding),
                                       len(self.neocortical_memory[j].embedding),
                                       self.dim)):
                        self.neocortical_memory[i].embedding[k] = (
                            (1 - alpha) * self.neocortical_memory[i].embedding[k] +
                            alpha * self.neocortical_memory[j].embedding[k])
                    self.neocortical_memory[i].embedding = _normalize(
                        self.neocortical_memory[i].embedding)
                    self.neocortical_memory[i].importance = max(
                        self.neocortical_memory[i].importance,
                        self.neocortical_memory[j].importance)
                    self.neocortical_memory[i].neocortical_strength = max(
                        self.neocortical_memory[i].neocortical_strength,
                        self.neocortical_memory[j].neocortical_strength)
                    self.neocortical_memory[i].access_count += self.neocortical_memory[j].access_count
                    to_remove.add(j)
                    merged_count += 1

        # Remove merged traces
        if to_remove:
            self.neocortical_memory = [t for i, t in enumerate(self.neocortical_memory)
                                       if i not in to_remove]
            for i in to_remove:
                trace = self.neocortical_memory[i] if i < len(self.neocortical_memory) else None
                # We already filtered, so just clean synaptic weights

        return {
            "status": "success",
            "memories_before": n,
            "memories_after": len(self.neocortical_memory),
            "merged": merged_count,
        }

    # -----------------------------------------------------------------------
    # Retrieval from consolidated memory
    # -----------------------------------------------------------------------

    def retrieve(self, query: List[float], top_k: int = 10) -> List[Tuple[MemoryTrace, float]]:
        """Retrieve from neocortical (consolidated) memory."""
        scored = []
        for trace in self.neocortical_memory:
            sim = _cosine_similarity(query, trace.embedding) if trace.embedding else 0.0
            # Weight by neocortical strength
            weighted_sim = sim * (0.5 + 0.5 * trace.neocortical_strength)
            scored.append((trace, weighted_sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_from_hippocampus(self, query: List[float], top_k: int = 5) -> List[Tuple[MemoryTrace, float]]:
        """Retrieve recent, unconsolidated memories from hippocampus."""
        scored = []
        for trace in self.hippocampal_buffer:
            sim = _cosine_similarity(query, trace.embedding) if trace.embedding else 0.0
            scored.append((trace, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hippocampal_size": len(self.hippocampal_buffer),
            "neocortical_size": len(self.neocortical_memory),
            "total_synaptic_weights": len(self._synaptic_weights),
            "avg_synaptic_weight": (sum(self._synaptic_weights.values()) /
                                    max(len(self._synaptic_weights), 1)),
            "total_consolidation_time": self._total_consolidation_time,
            "total_sessions": self._total_sessions,
            "replay_stats": self.memory_replay.get_replay_stats(),
            "dream_stats": self.dream_generator.get_dream_stats(),
        }

    def sleep_report(self, result: ConsolidationResult) -> str:
        """Generate a human-readable sleep/consolidation report."""
        lines = [
            "=" * 60,
            "  SLEEP / CONSOLIDATION REPORT",
            "=" * 60,
            f"  Session duration:       {result.total_duration:.2f}s",
            f"  Cycles completed:       {result.cycles_completed}",
            f"  Memories replayed:      {result.memories_replayed}",
            f"  Memories consolidated:  {result.memories_consolidated}",
            f"  Dream events generated: {result.dream_events_generated}",
            f"  Synaptic downscaling:   {result.synaptic_downscaling:.4f}",
            "-" * 60,
            "  Cycle details:",
        ]
        for cycle in result.cycle_details:
            lines.append(
                f"    Cycle {cycle.cycle_number}: {cycle.phase} | "
                f"replayed={cycle.memories_replayed} "
                f"consolidated={cycle.memories_consolidated} "
                f"dreams={cycle.events_generated}"
            )
        lines.extend([
            "-" * 60,
            "  Memory status:",
            f"    Hippocampal buffer: {len(self.hippocampal_buffer)} traces",
            f"    Neocortical memory: {len(self.neocortical_memory)} traces",
            f"    Avg synaptic weight: {sum(self._synaptic_weights.values()) / max(len(self._synaptic_weights), 1):.3f}",
            "=" * 60,
        ])
        return "\n".join(lines)
