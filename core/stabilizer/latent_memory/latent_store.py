"""
Latent / Potential Memory Store
================================
Implements compressed memory representations and efficient indexing:
  - LatentMemory:       vector quantization, sparse distributed representations
  - MemoryIndex:        locality-sensitive hashing (LSH), inverted file index
  - MemoryConsolidator: replay buffer, prioritized experience replay, memory distillation

Pure Python -- only stdlib modules (math, random, collections, struct, hashlib).
"""

from __future__ import annotations

import math
import random
import hashlib
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set


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


def _hash_floats(vec: List[float], num_bits: int = 64) -> int:
    """Deterministic hash of a float vector."""
    data = struct.pack(f"{len(vec)}d", *vec)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << num_bits) - 1)


# ---------------------------------------------------------------------------
# 1. LatentMemory -- Compressed memory representations
# ---------------------------------------------------------------------------

class CompressionMethod(Enum):
    VECTOR_QUANTIZATION = auto()
    SPARSE_DISTRIBUTED = auto()
    PRODUCT_QUANTIZATION = auto()


@dataclass
class LatentMemoryItem:
    """A compressed memory representation."""
    item_id: str
    original_id: str          # reference to source MemoryItem
    compressed_vector: List[float]
    codebook_index: int = -1  # VQ codebook index
    sparse_vector: Dict[int, float] = field(default_factory=dict)  # SDR: index -> weight
    reconstruction_error: float = 0.0
    compression_ratio: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorQuantizer:
    """
    Online k-means vector quantization.
    Compresses high-dimensional vectors into discrete codebook entries.
    """

    def __init__(self, codebook_size: int = 256, dim: int = 64,
                 learning_rate: float = 0.01, decay: float = 0.9999):
        self.codebook_size = codebook_size
        self.dim = dim
        self.learning_rate = learning_rate
        self.decay = decay
        self.codebook: List[List[float]] = [_random_vector(dim) for _ in range(codebook_size)]
        self.usage_counts: List[int] = [0] * codebook_size
        self._total_updates = 0

    def encode(self, vector: List[float]) -> Tuple[int, float]:
        """Find nearest codebook entry. Returns (index, distance)."""
        best_idx = 0
        best_dist = float("inf")
        for i, centroid in enumerate(self.codebook):
            dist = _l2_distance(vector, centroid)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx, best_dist

    def decode(self, index: int) -> List[float]:
        """Retrieve codebook vector."""
        return self.codebook[index][:]

    def update(self, vector: List[float], index: int):
        """Move codebook entry toward vector (online k-means step)."""
        lr = self.learning_rate * (self.decay ** self._total_updates)
        centroid = self.codebook[index]
        for i in range(self.dim):
            centroid[i] += lr * (vector[i] - centroid[i])
        self.usage_counts[index] += 1
        self._total_updates += 1

    def compress(self, vector: List[float]) -> LatentMemoryItem:
        """Compress a vector into a latent memory item."""
        idx, error = self.encode(vector)
        self.update(vector, idx)
        original_size = len(vector) * 8  # 8 bytes per float
        compressed_size = math.log2(self.codebook_size) / 8  # index in bytes
        return LatentMemoryItem(
            item_id=f"latent_{_hash_floats(vector):016x}",
            original_id="",
            compressed_vector=self.decode(idx),
            codebook_index=idx,
            reconstruction_error=error,
            compression_ratio=original_size / max(compressed_size, 1e-9),
        )

    def get_utilization(self) -> float:
        """Fraction of codebook entries that have been used."""
        used = sum(1 for c in self.usage_counts if c > 0)
        return used / self.codebook_size

    def prune_unused(self, threshold: float = 0.01):
        """Reinitialize rarely-used codebook entries."""
        total = max(sum(self.usage_counts), 1)
        for i in range(self.codebook_size):
            if self.usage_counts[i] / total < threshold:
                self.codebook[i] = _random_vector(self.dim)
                self.usage_counts[i] = 0


class SparseDistributedRepresentation:
    """
    Sparse Distributed Representation (SDR) encoder.
    Each vector is encoded as a small set of active indices with weights.
    Inspired by neocortical column activation patterns.
    """

    def __init__(self, dim: int = 4096, sparsity: float = 0.02,
                 seed: int = 42):
        self.dim = dim
        self.sparsity = sparsity
        self.active_count = max(int(dim * sparsity), 1)
        self.rng = random.Random(seed)
        # Random projection matrix (stored implicitly as hash functions)
        self._projection_seeds = [self.rng.randint(0, 2**31 - 1) for _ in range(dim)]

    def encode(self, vector: List[float]) -> Dict[int, float]:
        """
        Encode a dense vector into a sparse representation.
        Uses random projection to map dense -> sparse indices.
        """
        if not vector:
            return {}
        # Compute scores for each sparse dimension via random projection
        scores = []
        for i in range(self.dim):
            rng = random.Random(self._projection_seeds[i])
            projection = sum(rng.gauss(0, 1) * v for v in vector)
            scores.append((i, projection))
        # Select top-k active indices
        scores.sort(key=lambda x: x[1], reverse=True)
        sparse = {}
        for idx, score in scores[:self.active_count]:
            # Normalize score to [0, 1] range using sigmoid
            normalized = 1.0 / (1.0 + math.exp(-score * 0.1))
            sparse[idx] = normalized
        return sparse

    def decode(self, sparse: Dict[int, float]) -> List[float]:
        """
        Decode sparse representation back to dense vector.
        Returns a vector of dimension equal to the original input (approximate).
        """
        if not sparse:
            return []
        # We store original dim in metadata; here we use sparse dim as approximation
        result = [0.0] * self.dim
        for idx, weight in sparse.items():
            if 0 <= idx < self.dim:
                result[idx] = weight
        return result

    def similarity(self, sparse_a: Dict[int, float], sparse_b: Dict[int, float]) -> float:
        """Overlap-based similarity between two SDRs."""
        if not sparse_a or not sparse_b:
            return 0.0
        common_keys = set(sparse_a.keys()) & set(sparse_b.keys())
        if not common_keys:
            return 0.0
        # Cosine similarity on overlapping dimensions
        dot = sum(sparse_a[k] * sparse_b[k] for k in common_keys)
        mag_a = math.sqrt(sum(v * v for v in sparse_a.values()))
        mag_b = math.sqrt(sum(v * v for v in sparse_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def union(self, sparse_a: Dict[int, float], sparse_b: Dict[int, float]) -> Dict[int, float]:
        """Union of two SDRs (max pooling)."""
        result = dict(sparse_a)
        for k, v in sparse_b.items():
            result[k] = max(result.get(k, 0.0), v)
        return result

    def intersection(self, sparse_a: Dict[int, float], sparse_b: Dict[int, float]) -> Dict[int, float]:
        """Intersection of two SDRs (min pooling on common keys)."""
        common = set(sparse_a.keys()) & set(sparse_b.keys())
        return {k: min(sparse_a[k], sparse_b[k]) for k in common}


class ProductQuantizer:
    """
    Product Quantization: splits vector into subvectors and quantizes each independently.
    Provides much better compression than flat VQ for high-dimensional vectors.
    """

    def __init__(self, dim: int = 64, n_subquantizers: int = 8,
                 sub_codebook_size: int = 256):
        assert dim % n_subquantizers == 0, "dim must be divisible by n_subquantizers"
        self.dim = dim
        self.n_subquantizers = n_subquantizers
        self.sub_dim = dim // n_subquantizers
        self.sub_codebook_size = sub_codebook_size
        # Each sub-quantizer has its own codebook
        self.sub_codebooks: List[List[List[float]]] = []
        for _ in range(n_subquantizers):
            codebook = [_random_vector(self.sub_dim) for _ in range(sub_codebook_size)]
            self.sub_codebooks.append(codebook)

    def encode(self, vector: List[float]) -> List[int]:
        """Encode vector into list of sub-codebook indices."""
        codes = []
        for sq in range(self.n_subquantizers):
            start = sq * self.sub_dim
            end = start + self.sub_dim
            subvec = vector[start:end]
            # Find nearest centroid in sub-codebook
            best_idx = 0
            best_dist = float("inf")
            for i, centroid in enumerate(self.sub_codebooks[sq]):
                dist = _l2_distance(subvec, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            codes.append(best_idx)
        return codes

    def decode(self, codes: List[int]) -> List[float]:
        """Reconstruct vector from sub-codebook indices."""
        result = []
        for sq in range(self.n_subquantizers):
            result.extend(self.sub_codebooks[sq][codes[sq]])
        return result

    def compute_distance_table(self, query: List[float]) -> List[List[float]]:
        """Precompute distance table for asymmetric distance computation."""
        table = []
        for sq in range(self.n_subquantizers):
            start = sq * self.sub_dim
            end = start + self.sub_dim
            subvec = query[start:end]
            distances = []
            for centroid in self.sub_codebooks[sq]:
                dist = _l2_distance(subvec, centroid)
                distances.append(dist)
            table.append(distances)
        return table

    def asymmetric_distance(self, codes: List[int], dist_table: List[List[float]]) -> float:
        """Fast approximate distance using precomputed distance table."""
        total = 0.0
        for sq in range(self.n_subquantizers):
            total += dist_table[sq][codes[sq]] ** 2
        return math.sqrt(total)

    def update_subcodebook(self, sq: int, subvec: List[float], idx: int, lr: float = 0.01):
        """Online update of a sub-codebook entry."""
        centroid = self.sub_codebooks[sq][idx]
        for i in range(self.sub_dim):
            centroid[i] += lr * (subvec[i] - centroid[i])


class LatentMemory:
    """
    Compressed memory store using vector quantization and sparse representations.
    """

    def __init__(self, dim: int = 64, codebook_size: int = 256,
                 sdr_dim: int = 4096, sdr_sparsity: float = 0.02,
                 method: CompressionMethod = CompressionMethod.VECTOR_QUANTIZATION):
        self.dim = dim
        self.method = method
        self.vq = VectorQuantizer(codebook_size, dim)
        self.sdr = SparseDistributedRepresentation(sdr_dim, sdr_sparsity)
        self.pq = ProductQuantizer(dim) if method == CompressionMethod.PRODUCT_QUANTIZATION else None
        self._items: Dict[str, LatentMemoryItem] = {}
        self._original_to_latent: Dict[str, str] = {}  # original_id -> latent_id

    def store(self, vector: List[float], original_id: str = "",
              metadata: Optional[Dict] = None) -> LatentMemoryItem:
        """Compress and store a vector."""
        import time as _time
        if self.method == CompressionMethod.VECTOR_QUANTIZATION:
            item = self.vq.compress(vector)
        elif self.method == CompressionMethod.SPARSE_DISTRIBUTED:
            sparse = self.sdr.encode(vector)
            item = LatentMemoryItem(
                item_id=f"latent_sdr_{_hash_floats(vector):016x}",
                original_id=original_id,
                compressed_vector=vector,  # keep original for decoding
                sparse_vector=sparse,
                compression_ratio=len(vector) / max(len(sparse) * 2, 1),
            )
        elif self.method == CompressionMethod.PRODUCT_QUANTIZATION and self.pq:
            codes = self.pq.encode(vector)
            reconstructed = self.pq.decode(codes)
            error = _l2_distance(vector, reconstructed)
            item = LatentMemoryItem(
                item_id=f"latent_pq_{_hash_floats(vector):016x}",
                original_id=original_id,
                compressed_vector=reconstructed,
                reconstruction_error=error,
                compression_ratio=len(vector) * 8 / (len(codes) * math.log2(256) / 8),
                metadata={"pq_codes": codes},
            )
        else:
            item = self.vq.compress(vector)

        item.original_id = original_id
        item.timestamp = _time.time()
        if metadata:
            item.metadata.update(metadata)
        self._items[item.item_id] = item
        if original_id:
            self._original_to_latent[original_id] = item.item_id
        return item

    def retrieve(self, query: List[float], top_k: int = 5) -> List[Tuple[LatentMemoryItem, float]]:
        """Approximate nearest neighbor search in latent space."""
        if not self._items:
            return []
        if self.method == CompressionMethod.PRODUCT_QUANTIZATION and self.pq:
            return self._pq_search(query, top_k)
        scored = []
        for item in self._items.values():
            sim = _cosine_similarity(query, item.compressed_vector)
            scored.append((item, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _pq_search(self, query: List[float], top_k: int) -> List[Tuple[LatentMemoryItem, float]]:
        """Product quantization-based approximate search."""
        dist_table = self.pq.compute_distance_table(query)
        scored = []
        for item in self._items.values():
            codes = item.metadata.get("pq_codes")
            if codes is None:
                sim = _cosine_similarity(query, item.compressed_vector)
            else:
                dist = self.pq.asymmetric_distance(codes, dist_table)
                sim = 1.0 / (1.0 + dist)
            scored.append((item, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_by_original_id(self, original_id: str) -> Optional[LatentMemoryItem]:
        latent_id = self._original_to_latent.get(original_id)
        if latent_id:
            return self._items.get(latent_id)
        return None

    def size(self) -> int:
        return len(self._items)

    def average_reconstruction_error(self) -> float:
        if not self._items:
            return 0.0
        return sum(item.reconstruction_error for item in self._items.values()) / len(self._items)


# ---------------------------------------------------------------------------
# 2. MemoryIndex -- Fast approximate nearest neighbor search
# ---------------------------------------------------------------------------

class LSHFamily(Enum):
    SIMHASH = auto()       # cosine similarity
    RANDOM_PROJECTION = auto()  # L2 distance (E2LSH-style)


class LSHIndex:
    """
    Locality-Sensitive Hashing for approximate nearest neighbor search.
    Uses random hyperplane hashing for cosine similarity.
    """

    def __init__(self, dim: int = 64, num_tables: int = 8, num_hashes: int = 8,
                 family: LSHFamily = LSHFamily.RANDOM_PROJECTION):
        self.dim = dim
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.family = family
        # Generate random hyperplanes for each table
        self.hyperplanes: List[List[List[float]]] = []
        for _ in range(num_tables):
            planes = [_random_vector(dim) for _ in range(num_hashes)]
            self.hyperplanes.append(planes)
        # Hash tables: table_idx -> {hash_key -> [item_ids]}
        self.tables: List[Dict[int, List[str]]] = [defaultdict(list) for _ in range(num_tables)]
        # Item storage
        self._vectors: Dict[str, List[float]] = {}

    def _compute_hash(self, vector: List[float], table_idx: int) -> int:
        """Compute LSH hash for a vector in a specific table."""
        planes = self.hyperplanes[table_idx]
        hash_bits = 0
        for i, plane in enumerate(planes):
            # Dot product > 0 => bit = 1
            dot = sum(v * p for v, p in zip(vector, plane))
            if dot > 0:
                hash_bits |= (1 << i)
        return hash_bits

    def insert(self, item_id: str, vector: List[float]):
        """Insert a vector into the LSH index."""
        self._vectors[item_id] = vector
        for t in range(self.num_tables):
            h = self._compute_hash(vector, t)
            self.tables[t][h].append(item_id)

    def query(self, query: List[float], top_k: int = 10,
              num_probes: int = 2) -> List[Tuple[str, float]]:
        """
        Query the LSH index.
        num_probes: number of nearby hash buckets to check per table.
        """
        candidate_ids: Set[str] = set()
        for t in range(self.num_tables):
            h = self._compute_hash(query, t)
            # Check exact bucket and nearby buckets
            for probe in range(num_probes):
                # Flip one bit at a time for nearby buckets
                if probe == 0:
                    buckets_to_check = [h]
                else:
                    bit_to_flip = probe - 1
                    buckets_to_check = [h ^ (1 << bit_to_flip)]
                for bucket_key in buckets_to_check:
                    candidate_ids.update(self.tables[t].get(bucket_key, []))

        # Compute exact distances for candidates
        scored = []
        for cid in candidate_ids:
            vec = self._vectors.get(cid)
            if vec:
                sim = _cosine_similarity(query, vec)
                scored.append((cid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def remove(self, item_id: str) -> bool:
        """Remove an item from the index."""
        if item_id not in self._vectors:
            return False
        vector = self._vectors[item_id]
        for t in range(self.num_tables):
            h = self._compute_hash(vector, t)
            bucket = self.tables[t].get(h, [])
            if item_id in bucket:
                bucket.remove(item_id)
        del self._vectors[item_id]
        return True

    def size(self) -> int:
        return len(self._vectors)


class InvertedFileIndex:
    """
    Inverted File Index (IVF) for approximate nearest neighbor search.
    Partitions the vector space into clusters (Voronoi cells) using online k-means,
    then searches only the nearest clusters at query time.
    """

    def __init__(self, dim: int = 64, n_clusters: int = 16,
                 n_probe: int = 3):
        self.dim = dim
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        # Cluster centroids
        self.centroids: List[List[float]] = [_random_vector(dim) for _ in range(n_clusters)]
        # Inverted lists: cluster_id -> [(item_id, vector)]
        self.inverted_lists: Dict[int, List[Tuple[str, List[float]]]] = defaultdict(list)
        self._all_items: Dict[str, int] = {}  # item_id -> cluster_id
        self._total_assigned = 0

    def _find_nearest_cluster(self, vector: List[float]) -> int:
        best = 0
        best_dist = float("inf")
        for i, centroid in enumerate(self.centroids):
            dist = _l2_distance(vector, centroid)
            if dist < best_dist:
                best_dist = dist
                best = i
        return best

    def insert(self, item_id: str, vector: List[float]):
        """Insert a vector into the IVF index."""
        cluster_id = self._find_nearest_cluster(vector)
        self.inverted_lists[cluster_id].append((item_id, vector))
        self._all_items[item_id] = cluster_id
        self._total_assigned += 1
        # Periodically update centroids
        if self._total_assigned % 100 == 0:
            self._update_centroids()

    def _update_centroids(self):
        """Recompute cluster centroids from assigned vectors."""
        for cid in range(self.n_clusters):
            items = self.inverted_lists.get(cid, [])
            if items:
                vectors = [v for _, v in items]
                self.centroids[cid] = _mean_vectors(vectors)

    def query(self, query: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Query the IVF index, searching only the n_probe nearest clusters."""
        # Find nearest clusters
        cluster_distances = []
        for i, centroid in enumerate(self.centroids):
            dist = _l2_distance(query, centroid)
            cluster_distances.append((i, dist))
        cluster_distances.sort(key=lambda x: x[1])
        nearest_clusters = [c for c, _ in cluster_distances[:self.n_probe]]

        # Search within those clusters
        scored = []
        for cid in nearest_clusters:
            for item_id, vector in self.inverted_lists.get(cid, []):
                sim = _cosine_similarity(query, vector)
                scored.append((item_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def remove(self, item_id: str) -> bool:
        if item_id not in self._all_items:
            return False
        cluster_id = self._all_items[item_id]
        items = self.inverted_lists[cluster_id]
        self.inverted_lists[cluster_id] = [(iid, v) for iid, v in items if iid != item_id]
        del self._all_items[item_id]
        return True

    def size(self) -> int:
        return len(self._all_items)

    def cluster_sizes(self) -> Dict[int, int]:
        return {cid: len(items) for cid, items in self.inverted_lists.items()}


class MemoryIndex:
    """
    Combined memory index using both LSH and IVF for fast approximate search.
    """

    def __init__(self, dim: int = 64, use_lsh: bool = True, use_ivf: bool = True):
        self.dim = dim
        self.use_lsh = use_lsh
        self.use_ivf = use_ivf
        self.lsh = LSHIndex(dim) if use_lsh else None
        self.ivf = InvertedFileIndex(dim) if use_ivf else None
        # Fallback exact search
        self._vectors: Dict[str, List[float]] = {}

    def insert(self, item_id: str, vector: List[float]):
        self._vectors[item_id] = vector
        if self.lsh:
            self.lsh.insert(item_id, vector)
        if self.ivf:
            self.ivf.insert(item_id, vector)

    def query(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Query using available indices, merging results."""
        results: Dict[str, float] = {}

        if self.lsh:
            lsh_results = self.lsh.query(query_vector, top_k * 2)
            for item_id, score in lsh_results:
                results[item_id] = max(results.get(item_id, 0.0), score)

        if self.ivf:
            ivf_results = self.ivf.query(query_vector, top_k * 2)
            for item_id, score in ivf_results:
                results[item_id] = max(results.get(item_id, 0.0), score)

        if not self.lsh and not self.ivf:
            # Exact search fallback
            for item_id, vec in self._vectors.items():
                results[item_id] = _cosine_similarity(query_vector, vec)

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def remove(self, item_id: str) -> bool:
        if item_id not in self._vectors:
            return False
        if self.lsh:
            self.lsh.remove(item_id)
        if self.ivf:
            self.ivf.remove(item_id)
        del self._vectors[item_id]
        return True

    def size(self) -> int:
        return len(self._vectors)


# ---------------------------------------------------------------------------
# 3. MemoryConsolidator -- Offline memory processing
# ---------------------------------------------------------------------------

class ReplayType(Enum):
    RANDOM = auto()
    PRIORITIZED = auto()
    GENERATIVE = auto()


@dataclass
class ReplaySample:
    """A single experience replay sample."""
    item_id: str
    vector: List[float]
    reward: float = 0.0
    priority: float = 1.0
    weight: float = 1.0  # importance sampling weight
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """
    Prioritized experience replay buffer using sum-tree for efficient sampling.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha          # priority exponent
        self.beta = beta            # importance sampling exponent
        self.beta_increment = beta_increment
        self._buffer: List[ReplaySample] = []
        self._priorities: List[float] = []
        self._max_priority = 1.0
        self._position = 0

    def add(self, sample: ReplaySample):
        """Add a sample to the buffer."""
        priority = sample.priority ** self.alpha
        if len(self._buffer) < self.capacity:
            self._buffer.append(sample)
            self._priorities.append(priority)
        else:
            self._buffer[self._position] = sample
            self._priorities[self._position] = priority
            self._position = (self._position + 1) % self.capacity
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size: int) -> List[Tuple[ReplaySample, float]]:
        """
        Sample a batch using prioritized sampling.
        Returns list of (sample, importance_sampling_weight).
        """
        n = len(self._buffer)
        if n == 0:
            return []
        batch = []
        # Compute sampling probabilities
        total = sum(self._priorities)
        if total == 0:
            indices = random.sample(range(n), min(batch_size, n))
        else:
            probs = [p / total for p in self._priorities]
            indices = []
            for _ in range(min(batch_size, n)):
                r = random.random()
                cumulative = 0.0
                for i, prob in enumerate(probs):
                    cumulative += prob
                    if cumulative >= r:
                        indices.append(i)
                        break
                else:
                    indices.append(n - 1)

        # Compute importance sampling weights
        max_prob = max(self._priorities) / total if total > 0 else 1.0
        for idx in indices:
            sample = self._buffer[idx]
            prob = self._priorities[idx] / total if total > 0 else 1.0
            weight = (prob / max_prob) ** (-self.beta) if max_prob > 0 else 1.0
            sample.weight = weight
            batch.append((sample, weight))

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        return batch

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled indices."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = priority ** self.alpha
                self._max_priority = max(self._max_priority, self._priorities[idx])

    def __len__(self) -> int:
        return len(self._buffer)


class MemoryConsolidator:
    """
    Offline memory consolidation system.
    Features:
      - Replay buffer management with prioritization
      - Experience replay (random, prioritized, generative)
      - Memory distillation: compress many memories into fewer representative ones
    """

    def __init__(self, dim: int = 64, replay_capacity: int = 10000,
                 distillation_threshold: int = 100):
        self.dim = dim
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.latent_memory = LatentMemory(dim)
        self.memory_index = MemoryIndex(dim)
        self.distillation_threshold = distillation_threshold
        self._consolidation_count = 0
        self._distillation_stats = {
            "memories_in": 0,
            "memories_out": 0,
            "compression_ratio": 0.0,
        }

    def add_experience(self, vector: List[float], item_id: str = "",
                       reward: float = 0.0, priority: float = 1.0,
                       metadata: Optional[Dict] = None) -> str:
        """Add a new experience to the consolidation pipeline."""
        sample = ReplaySample(
            item_id=item_id or f"exp_{random.getrandbits(64):016x}",
            vector=vector,
            reward=reward,
            priority=priority,
            metadata=metadata or {},
        )
        self.replay_buffer.add(sample)
        # Also add to latent memory
        self.latent_memory.store(vector, item_id=sample.item_id, metadata=metadata)
        # Add to search index
        self.memory_index.insert(sample.item_id, vector)
        return sample.item_id

    def replay(self, batch_size: int = 32,
               replay_type: ReplayType = ReplayType.PRIORITIZED) -> List[ReplaySample]:
        """
        Perform experience replay.
        Returns a batch of samples for re-processing.
        """
        if replay_type == ReplayType.RANDOM:
            return self._random_replay(batch_size)
        elif replay_type == ReplayType.PRIORITIZED:
            return self._prioritized_replay(batch_size)
        elif replay_type == ReplayType.GENERATIVE:
            return self._generative_replay(batch_size)
        return []

    def _random_replay(self, batch_size: int) -> List[ReplaySample]:
        """Uniform random sampling from replay buffer."""
        n = len(self.replay_buffer)
        if n == 0:
            return []
        samples = random.sample(self.replay_buffer._buffer, min(batch_size, n))
        return samples

    def _prioritized_replay(self, batch_size: int) -> List[ReplaySample]:
        """Prioritized replay based on TD-error / importance."""
        batch = self.replay_buffer.sample(batch_size)
        return [sample for sample, _ in batch]

    def _generative_replay(self, batch_size: int) -> List[ReplaySample]:
        """
        Generative replay: create new samples by interpolating between
        existing memories. Simulates dreaming / creative recombination.
        """
        n = len(self.replay_buffer)
        if n < 2:
            return list(self.replay_buffer._buffer[:batch_size])

        generated = []
        for _ in range(min(batch_size, n)):
            # Pick two random samples
            a, b = random.sample(self.replay_buffer._buffer, 2)
            # Interpolate vectors
            alpha = random.random()
            interp_vec = [alpha * x + (1 - alpha) * y
                          for x, y in zip(a.vector, b.vector)]
            interp_vec = _normalize(interp_vec)
            # Interpolate rewards
            interp_reward = alpha * a.reward + (1 - alpha) * b.reward
            generated.append(ReplaySample(
                item_id=f"gen_{random.getrandbits(64):016x}",
                vector=interp_vec,
                reward=interp_reward,
                priority=max(a.priority, b.priority) * 0.8,
                metadata={"generated": True, "parent_a": a.item_id, "parent_b": b.item_id},
            ))
        return generated

    def distill(self, max_clusters: int = 50, min_cluster_size: int = 3) -> Dict[str, Any]:
        """
        Memory distillation: compress many memories into fewer representative ones.
        Uses online clustering to group similar memories, then keeps the centroid
        of each cluster as the distilled memory.
        """
        if self.latent_memory.size() < self.distillation_threshold:
            return {"status": "not_enough_memories", "current_size": self.latent_memory.size()}

        # Gather all compressed vectors
        items = list(self.latent_memory._items.values())
        vectors = [item.compressed_vector for item in items if item.compressed_vector]

        if len(vectors) < min_cluster_size:
            return {"status": "too_few_vectors", "count": len(vectors)}

        # Online clustering (simplified k-means)
        centroids = [_random_vector(self.dim) for _ in range(max_clusters)]
        assignments: Dict[int, List[int]] = defaultdict(list)

        # Run k-means iterations
        for iteration in range(10):
            assignments.clear()
            for i, vec in enumerate(vectors):
                best_cluster = 0
                best_dist = float("inf")
                for c, centroid in enumerate(centroids):
                    dist = _l2_distance(vec, centroid)
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = c
                assignments[best_cluster].append(i)

            # Update centroids
            for c, indices in assignments.items():
                if indices:
                    cluster_vectors = [vectors[i] for i in indices]
                    centroids[c] = _mean_vectors(cluster_vectors)

        # Filter clusters by minimum size
        valid_clusters = {c: indices for c, indices in assignments.items()
                          if len(indices) >= min_cluster_size}

        # Create distilled memories
        distilled_count = 0
        distilled_items = []
        for c, indices in valid_clusters.items():
            cluster_vectors = [vectors[i] for i in indices]
            centroid = _mean_vectors(cluster_vectors)
            # Compute cluster quality (average distance to centroid)
            avg_dist = sum(_l2_distance(v, centroid) for v in cluster_vectors) / len(cluster_vectors)
            distilled_item = self.latent_memory.store(
                centroid,
                original_id=f"distilled_{c}",
                metadata={
                    "distilled": True,
                    "source_count": len(indices),
                    "avg_distance": avg_dist,
                    "cluster_id": c,
                },
            )
            distilled_items.append(distilled_item)
            distilled_count += 1

        self._distillation_stats = {
            "memories_in": len(vectors),
            "memories_out": distilled_count,
            "compression_ratio": len(vectors) / max(distilled_count, 1),
            "clusters": len(valid_clusters),
            "iterations": 10,
        }
        self._consolidation_count += 1

        return {
            "status": "success",
            **self._distillation_stats,
            "distilled_ids": [item.item_id for item in distilled_items],
        }

    def search(self, query: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search consolidated memories."""
        return self.memory_index.query(query, top_k)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "replay_buffer_size": len(self.replay_buffer),
            "latent_memory_size": self.latent_memory.size(),
            "index_size": self.memory_index.size(),
            "consolidation_count": self._consolidation_count,
            "distillation_stats": self._distillation_stats,
            "avg_reconstruction_error": self.latent_memory.average_reconstruction_error(),
        }
