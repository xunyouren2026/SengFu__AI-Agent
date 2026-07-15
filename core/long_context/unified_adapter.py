"""
统一核心适配器模块

将统一核心算法适配为长上下文模块的接口。
保持原有API不变，内部使用统一核心实现。

适配器组件：
- ContextMemoryAdapter: 将统一记忆适配为长上下文记忆
- ContextAttentionAdapter: 将统一注意力适配为文本注意力
- ContextChunkerAdapter: 将统一分块器适配为文本语义分块
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
import time
import hashlib

# 导入统一核心
from ..unified_algorithms.unified_memory import (
    UnifiedMemoryBank,
    UnifiedLightweightMemory,
    UnifiedHierarchicalMemory,
    UnifiedAdaptiveCompressor,
    MemoryEntry,
    MemoryQuery,
)
from ..unified_algorithms.unified_attention import (
    UnifiedSlidingWindowAttention,
    UnifiedDynamicRouting,
    UnifiedSparseAttention,
    AttentionContext,
    RoutingDecision,
)
from ..unified_algorithms.unified_chunking import (
    UnifiedChunker,
    UnifiedOverlapFusion,
    UnifiedBoundaryDetector,
    UnifiedProgressiveLoader,
    Chunk,
    ChunkingResult,
    Boundary,
)
from ..unified_algorithms.unified_config import (
    UnifiedAlgorithmConfig,
    MemoryRetrievalMode,
    CompressionStrategy,
    AttentionPattern,
    ChunkingStrategy,
    BoundaryType,
)


# ============================================================================
# 工具函数
# ============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_vector(v: List[float]) -> List[float]:
    """L2归一化向量"""
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0:
        return v[:]
    return [x / norm for x in v]


def compute_hash(content: str) -> str:
    """计算内容哈希"""
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ============================================================================
# 数据类（保持与原模块兼容）
# ============================================================================

@dataclass
class ContextChunk:
    """
    上下文分块 - 与memory_systems.py兼容
    """
    chunk_id: str
    content: str
    embedding: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    access_count: int = 0
    position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_keyframe: bool = False
    keyframe_score: float = 0.0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    def touch(self):
        """记录访问"""
        self.access_count += 1
        self.metadata['last_access'] = time.time()

    def similarity_to(self, other: 'ContextChunk') -> float:
        """计算与另一块的相似度"""
        if not self.embedding or not other.embedding:
            return 0.0
        return cosine_similarity(self.embedding, other.embedding)


@dataclass
class MemorySlot:
    """
    记忆槽位 - 与memory_systems.py兼容
    """
    slot_id: str
    key: List[float]
    value: List[float]
    rank: int = 8
    alpha: float = 1.0
    timestamp: float = field(default_factory=time.time)
    update_count: int = 0


@dataclass
class ContextSegment:
    """
    上下文片段 - 与context_manager.py兼容
    """
    segment_id: str
    content: str
    start_pos: int = 0
    end_pos: int = 0
    embedding: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    boundary_type: Any = None
    boundary_score: float = 0.0
    prev_segment_id: Optional[str] = None
    next_segment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    overlap_prev: str = ""
    overlap_next: str = ""

    def get_full_content(self) -> str:
        """获取完整内容（包含重叠）"""
        return self.overlap_prev + self.content + self.overlap_next

    def similarity_to(self, other: 'ContextSegment') -> float:
        """计算与另一片段的相似度"""
        if not self.embedding or not other.embedding:
            return 0.0
        return cosine_similarity(self.embedding, other.embedding)


@dataclass
class TokenInfo:
    """
    Token信息 - 与attention_mechanisms.py兼容
    """
    token_id: int
    embedding: List[float]
    position: int
    importance: float = 0.5
    zone: Any = None
    last_access: float = 0.0
    access_count: int = 0


# ============================================================================
# 1. ContextMemoryAdapter - 统一记忆适配器
# ============================================================================

class ContextMemoryAdapter:
    """
    上下文记忆适配器

    将UnifiedMemoryBank适配为ContextMemoryBank的接口。
    保持原有API不变，内部使用统一记忆核心。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 64,
        max_chunks: int = 10000,
        dim: int = 768,
        keyframe_ratio: float = 0.1,
        config: Optional[UnifiedAlgorithmConfig] = None,
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_chunks = max_chunks
        self.dim = dim
        self.keyframe_ratio = keyframe_ratio

        # 创建统一记忆库配置
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.config = self.config.update(
            memory_capacity=max_chunks,
            memory_retrieval_mode=MemoryRetrievalMode.HYBRID,
            compression_strategy=CompressionStrategy.ADAPTIVE,
        )

        # 内部统一记忆库
        self._unified_memory: UnifiedMemoryBank[ContextChunk] = UnifiedMemoryBank(self.config)

        # 存储映射
        self.chunks: Dict[str, ContextChunk] = {}
        self.keyframes: Set[str] = set()
        self.chunk_order: List[str] = []

        # 索引
        self._embedding_index: List[Tuple[str, List[float]]] = []
        self._index_dirty = True

        # 统计
        self.stats = {
            'total_chunks': 0,
            'total_keyframes': 0,
            'total_accesses': 0,
            'cache_hits': 0
        }

    def add_context(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        添加上下文，自动分块

        保持与原ContextMemoryBank相同的API。
        """
        # 分块
        chunks = self._chunk_content(content)
        chunk_ids = []
        prev_id = None

        for i, chunk_content in enumerate(chunks):
            chunk_id = f"chunk_{compute_hash(chunk_content)}_{int(time.time() * 1000)}"

            # 生成嵌入（如果没有提供）
            chunk_embedding = embedding if (embedding and i == 0) else self._generate_embedding(chunk_content)

            # 计算关键帧分数
            keyframe_score = self._compute_keyframe_score(chunk_content, importance, i, len(chunks))
            is_keyframe = keyframe_score > (1.0 - self.keyframe_ratio)

            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                embedding=chunk_embedding,
                importance=importance,
                position=i,
                metadata=metadata or {},
                is_keyframe=is_keyframe,
                keyframe_score=keyframe_score,
                prev_chunk_id=prev_id,
                next_chunk_id=None
            )

            # 更新前一个块的next引用
            if prev_id and prev_id in self.chunks:
                self.chunks[prev_id].next_chunk_id = chunk_id

            # 存储到统一记忆库
            entry = MemoryEntry(
                data=chunk,
                timestamp=time.time(),
                importance=importance,
                metadata={'chunk_id': chunk_id, 'is_keyframe': is_keyframe}
            )
            self._unified_memory.store(entry)

            self.chunks[chunk_id] = chunk
            chunk_ids.append(chunk_id)

            if is_keyframe:
                self.keyframes.add(chunk_id)

            prev_id = chunk_id

        self.chunk_order.extend(chunk_ids)
        self._index_dirty = True
        self._enforce_capacity()

        self.stats['total_chunks'] = len(self.chunks)
        self.stats['total_keyframes'] = len(self.keyframes)

        return chunk_ids

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        use_keyframes: bool = True,
        context_window: int = 1
    ) -> List[ContextChunk]:
        """
        检索相关上下文

        使用统一记忆库的检索功能。
        """
        self.stats['total_accesses'] += 1

        # 使用统一记忆库检索
        query = MemoryQuery(
            query_data=ContextChunk(
                chunk_id="query",
                content="",
                embedding=query_embedding
            ),
            top_k=top_k,
            threshold=0.0,
            recency_weight=0.3
        )

        entries = self._unified_memory.retrieve(query)

        # 转换为ContextChunk列表
        result = []
        seen = set()

        for entry in entries:
            chunk = entry.data
            if isinstance(chunk, ContextChunk):
                # 扩展上下文窗口
                self._expand_context_window(chunk, context_window, result, seen)

        # 按位置排序
        result.sort(key=lambda x: x.position)

        return result

    def _expand_context_window(
        self,
        chunk: ContextChunk,
        window_size: int,
        result: List[ContextChunk],
        seen: Set[str]
    ):
        """扩展上下文窗口"""
        # 向前扩展
        current = chunk
        for _ in range(window_size):
            if current.prev_chunk_id and current.prev_chunk_id not in seen:
                prev_chunk = self.chunks.get(current.prev_chunk_id)
                if prev_chunk:
                    result.append(prev_chunk)
                    seen.add(current.prev_chunk_id)
                    current = prev_chunk

        # 添加当前块
        if chunk.chunk_id not in seen:
            result.append(chunk)
            seen.add(chunk.chunk_id)
            chunk.touch()

        # 向后扩展
        current = chunk
        for _ in range(window_size):
            if current.next_chunk_id and current.next_chunk_id not in seen:
                next_chunk = self.chunks.get(current.next_chunk_id)
                if next_chunk:
                    result.append(next_chunk)
                    seen.add(current.next_chunk_id)
                    current = next_chunk

    def retrieve_by_position(
        self,
        start_pos: int,
        end_pos: int
    ) -> List[ContextChunk]:
        """按位置范围检索"""
        result = []
        for chunk in self.chunks.values():
            if start_pos <= chunk.position <= end_pos:
                result.append(chunk)
        result.sort(key=lambda x: x.position)
        return result

    def progressive_load(
        self,
        chunk_ids: List[str],
        batch_size: int = 10
    ) -> List[List[ContextChunk]]:
        """渐进式加载"""
        batches = []

        # 第一批：关键帧
        keyframe_batch = [self.chunks[cid] for cid in chunk_ids
                         if cid in self.chunks and self.chunks[cid].is_keyframe]
        if keyframe_batch:
            batches.append(keyframe_batch)

        # 后续批次：普通块
        normal_chunks = [self.chunks[cid] for cid in chunk_ids
                        if cid in self.chunks and not self.chunks[cid].is_keyframe]

        for i in range(0, len(normal_chunks), batch_size):
            batch = normal_chunks[i:i + batch_size]
            batches.append(batch)

        return batches

    def _chunk_content(self, content: str) -> List[str]:
        """将内容分块"""
        chunks = []
        for i in range(0, len(content), self.chunk_size - self.overlap_size):
            chunk = content[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [content]

    def _generate_embedding(self, content: str) -> List[float]:
        """生成简单嵌入"""
        embedding = [0.0] * self.dim
        for i, char in enumerate(content):
            idx = i % self.dim
            embedding[idx] += ord(char) / 255.0
        return normalize_vector(embedding)

    def _compute_keyframe_score(
        self,
        content: str,
        importance: float,
        position: int,
        total_chunks: int
    ) -> float:
        """计算关键帧分数"""
        length_score = min(len(content) / self.chunk_size, 1.0)
        position_score = 1.0 if position == 0 or position == total_chunks - 1 else 0.5
        score = (importance * 0.4 + length_score * 0.3 + position_score * 0.3)
        return score

    def _enforce_capacity(self):
        """强制执行容量限制"""
        while len(self.chunks) > self.max_chunks:
            oldest_id = min(
                self.chunks.keys(),
                key=lambda k: (self.chunks[k].access_count, self.chunks[k].timestamp)
            )
            self._remove_chunk(oldest_id)

    def _remove_chunk(self, chunk_id: str):
        """移除块并更新引用"""
        if chunk_id not in self.chunks:
            return

        chunk = self.chunks[chunk_id]

        # 更新相邻引用
        if chunk.prev_chunk_id and chunk.prev_chunk_id in self.chunks:
            self.chunks[chunk.prev_chunk_id].next_chunk_id = chunk.next_chunk_id
        if chunk.next_chunk_id and chunk.next_chunk_id in self.chunks:
            self.chunks[chunk.next_chunk_id].prev_chunk_id = chunk.prev_chunk_id

        self.keyframes.discard(chunk_id)

        if chunk_id in self.chunk_order:
            self.chunk_order.remove(chunk_id)

        del self.chunks[chunk_id]
        self._index_dirty = True

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        unified_stats = self._unified_memory.get_stats()
        return {
            **self.stats,
            'capacity': self.max_chunks,
            'utilization': len(self.chunks) / self.max_chunks,
            'keyframe_ratio': len(self.keyframes) / max(len(self.chunks), 1),
            'unified_memory_stats': unified_stats
        }


# ============================================================================
# 2. ContextAttentionAdapter - 统一注意力适配器
# ============================================================================

class ContextAttentionAdapter:
    """
    上下文注意力适配器

    将UnifiedSlidingWindowAttention适配为SlidingWindowContextAttention的接口。
    保持原有API不变，内部使用统一注意力核心。
    """

    def __init__(
        self,
        dim: int = 768,
        window_size: int = 512,
        global_summary_size: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
        config: Optional[UnifiedAlgorithmConfig] = None,
    ):
        self.dim = dim
        self.window_size = window_size
        self.global_summary_size = global_summary_size
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.dropout = dropout

        # 创建统一注意力配置
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.config = self.config.update(
            attention_window_size=window_size,
            attention_cache_zones=3,
        )

        # 内部统一注意力机制
        self._unified_attention = UnifiedSlidingWindowAttention[
            List[float]
        ](
            window_size=window_size,
            num_zones=3,
            config=self.config
        )

        # 投影矩阵（保持与原API兼容）
        self.w_q = self._init_weight(dim, dim)
        self.w_k = self._init_weight(dim, dim)
        self.w_v = self._init_weight(dim, dim)
        self.w_o = self._init_weight(dim, dim)

        # 三区缓存
        self.current_tokens: List[TokenInfo] = []
        self.local_cache: List[TokenInfo] = []
        self.global_summary: List[TokenInfo] = []

        # 缓存统计
        self.cache_stats = {
            'current_hits': 0,
            'local_hits': 0,
            'global_hits': 0,
            'total_queries': 0
        }

    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重"""
        import random
        import math
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]

    def forward(
        self,
        x: List[List[float]],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        前向传播

        使用统一注意力核心计算。
        """
        seq_len = len(x)

        # 更新缓存
        if use_cache:
            self._update_cache(x)

        # 使用统一注意力计算
        # 将输入转换为统一注意力格式
        output = self._unified_attention.compute(x, x, x)

        # 更新统计
        self.cache_stats['total_queries'] += seq_len
        self.cache_stats['local_hits'] += min(seq_len * self.window_size, seq_len * seq_len)
        self.cache_stats['global_hits'] += seq_len * self.global_summary_size

        return output

    def _update_cache(self, x: List[List[float]]):
        """更新三区缓存"""
        seq_len = len(x)

        # 更新当前区
        current_size = min(16, seq_len)
        self.current_tokens = [
            TokenInfo(
                token_id=seq_len - current_size + i,
                embedding=x[seq_len - current_size + i],
                position=seq_len - current_size + i,
            )
            for i in range(current_size)
        ]

        # 更新局部缓存
        if seq_len > self.window_size:
            local_start = seq_len - self.window_size
            self.local_cache = [
                TokenInfo(
                    token_id=local_start + i,
                    embedding=x[local_start + i],
                    position=local_start + i,
                )
                for i in range(self.window_size)
            ]
        else:
            self.local_cache = [
                TokenInfo(
                    token_id=i,
                    embedding=x[i],
                    position=i,
                )
                for i in range(seq_len)
            ]

        # 更新全局摘要
        self.global_summary = [
            TokenInfo(
                token_id=i * (seq_len // self.global_summary_size) if seq_len >= self.global_summary_size else i,
                embedding=x[i * (seq_len // self.global_summary_size) if seq_len >= self.global_summary_size else i],
                position=i * (seq_len // self.global_summary_size) if seq_len >= self.global_summary_size else i,
            )
            for i in range(min(self.global_summary_size, seq_len))
        ]

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.cache_stats['total_queries']
        return {
            **self.cache_stats,
            'current_size': len(self.current_tokens),
            'local_size': len(self.local_cache),
            'global_size': len(self.global_summary),
            'local_hit_rate': self.cache_stats['local_hits'] / max(total, 1),
            'global_hit_rate': self.cache_stats['global_hits'] / max(total, 1)
        }


# ============================================================================
# 3. ContextChunkerAdapter - 统一分块器适配器
# ============================================================================

class ContextChunkerAdapter:
    """
    上下文分块器适配器

    将UnifiedChunker适配为ContextChunker的接口。
    保持原有API不变，内部使用统一分块核心。
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap_size: int = 64,
        respect_boundaries: bool = True,
        config: Optional[UnifiedAlgorithmConfig] = None,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.respect_boundaries = respect_boundaries

        # 创建统一分块器配置
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.config = self.config.update(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap_size,
            chunking_strategy=ChunkingStrategy.SEMANTIC if respect_boundaries else ChunkingStrategy.FIXED_SIZE,
        )

        # 内部统一分块器
        self._unified_chunker = UnifiedChunker[str](
            strategy=self.config.chunking_strategy,
            chunk_size=max_chunk_size,
            overlap=overlap_size,
            config=self.config
        )

        # 边界检测器
        self._boundary_detector: Optional[UnifiedBoundaryDetector[str]] = None
        if respect_boundaries:
            self._boundary_detector = UnifiedBoundaryDetector[str](
                boundary_type=BoundaryType.TOPIC,
                threshold=0.5,
                min_segment_size=10
            )
            self._unified_chunker.set_boundary_detector(self._boundary_detector)

        # 融合器
        self._overlap_fusion = UnifiedOverlapFusion[str](
            blend_window=overlap_size
        )

        self.chunk_history: List[Dict] = []

    def chunk(
        self,
        content: str,
        embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> List[ContextSegment]:
        """
        分块处理

        使用统一分块器进行分块。
        """
        if not content:
            return []

        # 将内容转换为字符列表进行分块
        char_list = list(content)

        # 使用统一分块器
        result: ChunkingResult[str] = self._unified_chunker.chunk(char_list)

        # 转换回ContextSegment
        segments = []
        for i, chunk in enumerate(result.chunks):
            segment_content = ''.join(chunk.data)

            segment = ContextSegment(
                segment_id=f"seg_{compute_hash(segment_content)}_{i}",
                content=segment_content,
                start_pos=chunk.start,
                end_pos=chunk.end,
                boundary_type=chunk.boundary_type,
            )

            # 生成嵌入
            if embedding_fn:
                segment.embedding = embedding_fn(segment_content)

            segments.append(segment)

        # 应用重叠
        segments = self._apply_overlap(segments)

        # 记录历史
        self.chunk_history.append({
            'original_length': len(content),
            'num_segments': len(segments),
            'avg_segment_size': sum(len(s.content) for s in segments) / max(len(segments), 1),
        })

        return segments

    def _apply_overlap(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """应用重叠区域"""
        if len(segments) <= 1 or self.overlap_size <= 0:
            return segments

        for i in range(len(segments)):
            # 与前一片段重叠
            if i > 0:
                prev_segment = segments[i - 1]
                overlap_text = prev_segment.content[-self.overlap_size:] if len(prev_segment.content) > self.overlap_size else prev_segment.content
                segments[i].overlap_prev = overlap_text
                segments[i].prev_segment_id = prev_segment.segment_id

            # 与后一片段重叠
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                overlap_text = next_segment.content[:self.overlap_size] if len(next_segment.content) > self.overlap_size else next_segment.content
                segments[i].overlap_next = overlap_text
                segments[i].next_segment_id = next_segment.segment_id

        return segments

    def merge_chunks(
        self,
        segments: List[ContextSegment],
        merge_threshold: float = 0.8
    ) -> List[ContextSegment]:
        """合并相似块"""
        if len(segments) <= 1:
            return segments

        # 转换为统一分块器的Chunk格式
        chunks = []
        for i, seg in enumerate(segments):
            chunk = Chunk(
                data=list(seg.content),
                index=i,
                start=seg.start_pos,
                end=seg.end_pos,
                metadata=seg.metadata,
                boundary_type=seg.boundary_type
            )
            chunks.append(chunk)

        # 使用统一分块器合并
        result = self._unified_chunker.merge_chunks(chunks, max_gap=1)

        # 转换回ContextSegment
        merged_segments = []
        for chunk in result.chunks:
            content = ''.join(chunk.data)
            segment = ContextSegment(
                segment_id=f"seg_{compute_hash(content)}_{chunk.index}",
                content=content,
                start_pos=chunk.start,
                end_pos=chunk.end,
                metadata=chunk.metadata,
                boundary_type=chunk.boundary_type
            )
            merged_segments.append(segment)

        return merged_segments


# ============================================================================
# 4. 动态路由适配器
# ============================================================================

class DynamicRoutingAdapter:
    """
    动态路由适配器

    将UnifiedDynamicRouting适配为DynamicContextRouting的接口。
    """

    def __init__(
        self,
        dim: int = 768,
        num_important: int = 128,
        num_compressed: int = 256,
        compression_ratio: float = 0.5,
        routing_temperature: float = 0.5,
        config: Optional[UnifiedAlgorithmConfig] = None,
    ):
        self.dim = dim
        self.num_important = num_important
        self.num_compressed = num_compressed
        self.compression_ratio = compression_ratio
        self.routing_temperature = routing_temperature

        # 创建统一动态路由
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self._unified_routing = UnifiedDynamicRouting[List[float]](
            threshold=0.5,
            capacity_factor=(num_important + num_compressed) / 1000.0,
            config=self.config
        )

        # 投影矩阵（保持兼容）
        self.importance_proj = self._init_weight(dim, 1)
        self.compress_proj = self._init_weight(dim, int(dim * compression_ratio))
        self.decompress_proj = self._init_weight(int(dim * compression_ratio), dim)

        self.routing_history: List[Dict] = []

    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重"""
        import random
        import math
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]

    def route(
        self,
        tokens: List[List[float]],
        query: Optional[List[float]] = None
    ) -> Tuple[List[List[float]], List[int], List[int], List[int]]:
        """
        路由token

        使用统一动态路由。
        """
        seq_len = len(tokens)

        # 计算重要性分数
        importances = self._compute_importance(tokens, query)

        # 使用统一路由
        decision = self._unified_routing.route(tokens, importances)

        # 分类
        selected_indices = decision.token_indices
        important_indices = selected_indices[:min(self.num_important, len(selected_indices))]
        compressed_indices = selected_indices[self.num_important:self.num_important + self.num_compressed]
        skipped_indices = [i for i in range(seq_len) if i not in selected_indices]

        # 收集路由后的token
        routed_tokens = []

        # 重要token：完整保留
        for idx in important_indices:
            routed_tokens.append(tokens[idx])

        # 压缩token：降维
        for idx in compressed_indices:
            compressed = self._compress_token(tokens[idx])
            routed_tokens.append(compressed)

        # 记录路由历史
        self.routing_history.append({
            'seq_len': seq_len,
            'important': len(important_indices),
            'compressed': len(compressed_indices),
            'skipped': len(skipped_indices),
            'compression_ratio': self.compression_ratio
        })

        return routed_tokens, important_indices, compressed_indices, skipped_indices

    def _compute_importance(
        self,
        tokens: List[List[float]],
        query: Optional[List[float]]
    ) -> List[float]:
        """计算token重要性"""
        importances = []

        for token in tokens:
            # 基础重要性：投影分数
            base_importance = sum(
                token[i] * self.importance_proj[i][0]
                for i in range(self.dim)
            )

            # 如果有查询，加入相关性
            if query is not None:
                relevance = cosine_similarity(token, query)
                base_importance += relevance * 0.5

            # L2范数作为重要性的补充
            norm = sum(x * x for x in token) ** 0.5
            base_importance += norm * 0.1

            importances.append(max(0, base_importance))

        return importances

    def _compress_token(self, token: List[float]) -> List[float]:
        """压缩token"""
        compressed_dim = int(self.dim * self.compression_ratio)
        compressed = [
            sum(token[i] * self.compress_proj[i][j] for i in range(self.dim))
            for j in range(compressed_dim)
        ]
        # 升维回来（近似）
        reconstructed = [
            sum(compressed[j] * self.decompress_proj[j][i] for j in range(compressed_dim))
            for i in range(self.dim)
        ]
        return reconstructed

    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        unified_stats = self._unified_routing.get_routing_stats()

        if not self.routing_history:
            return {**unified_stats, 'routing_history': 0}

        avg_important = sum(h['important'] for h in self.routing_history) / len(self.routing_history)
        avg_compressed = sum(h['compressed'] for h in self.routing_history) / len(self.routing_history)
        avg_skipped = sum(h['skipped'] for h in self.routing_history) / len(self.routing_history)

        return {
            **unified_stats,
            'routing_history': len(self.routing_history),
            'avg_important': avg_important,
            'avg_compressed': avg_compressed,
            'avg_skipped': avg_skipped,
        }


# ============================================================================
# 5. 边界检测适配器
# ============================================================================

class BoundaryDetectorAdapter:
    """
    边界检测适配器

    将UnifiedBoundaryDetector适配为BoundaryDetector的接口。
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        window_size: int = 3,
        config: Optional[UnifiedAlgorithmConfig] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

        # 创建统一边界检测器
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self._unified_detector = UnifiedBoundaryDetector[str](
            boundary_type=BoundaryType.TOPIC,
            threshold=similarity_threshold,
            min_segment_size=window_size * 3
        )

        self.detection_history: List[Dict] = []

    def detect_boundaries(
        self,
        segments: List[ContextSegment]
    ) -> List[Tuple[int, Any, float]]:
        """
        检测片段间的边界

        使用统一边界检测器。
        """
        # 转换为统一格式
        data = [seg.content for seg in segments]

        # 使用统一检测器
        boundaries = self._unified_detector.detect_boundaries(data)

        # 转换结果
        result = []
        for boundary in boundaries:
            result.append((
                boundary.position,
                boundary.type,
                boundary.confidence
            ))

        # 记录历史
        self.detection_history.append({
            'num_segments': len(segments),
            'boundaries_found': len(result),
            'avg_strength': sum(b[2] for b in result) / max(len(result), 1)
        })

        return result

    def split_at_boundaries(
        self,
        segments: List[ContextSegment]
    ) -> List[List[ContextSegment]]:
        """在边界处分割片段"""
        boundaries = self.detect_boundaries(segments)
        boundary_indices = set(b[0] for b in boundaries)

        groups = []
        current_group = []

        for i, segment in enumerate(segments):
            current_group.append(segment)

            if i in boundary_indices and current_group:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        return groups


# ============================================================================
# 6. 渐进式加载适配器
# ============================================================================

class ProgressiveLoaderAdapter:
    """
    渐进式加载适配器

    将UnifiedProgressiveLoader适配为ProgressiveLoader的接口。
    """

    def __init__(
        self,
        num_stages: int = 3,
        keyframe_ratio: float = 0.2,
        config: Optional[UnifiedAlgorithmConfig] = None,
    ):
        self.num_stages = num_stages
        self.keyframe_ratio = keyframe_ratio

        # 创建统一渐进式加载器
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self._unified_loader = UnifiedProgressiveLoader[str](
            chunk_size=512,
            prefetch_size=num_stages,
            config=self.config
        )

        self.load_history: List[Dict] = []

    def create_loading_plan(
        self,
        segments: List[ContextSegment]
    ) -> List[List[str]]:
        """
        创建加载计划

        使用统一渐进式加载器。
        """
        if not segments:
            return []

        # 按重要性排序
        sorted_segments = sorted(
            segments,
            key=lambda s: (s.boundary_score, s.metadata.get('importance', 0)),
            reverse=True
        )

        # 阶段1：关键片段
        num_keyframes = max(1, int(len(segments) * self.keyframe_ratio))
        stage1 = [s.segment_id for s in sorted_segments[:num_keyframes]]

        # 阶段2：中等重要片段
        remaining = [s for s in sorted_segments[num_keyframes:]]
        mid_point = len(remaining) // 2
        stage2 = [s.segment_id for s in remaining[:mid_point]]

        # 阶段3：剩余片段
        stage3 = [s.segment_id for s in remaining[mid_point:]]

        plan = [stage1, stage2, stage3]

        # 记录
        self.load_history.append({
            'total_segments': len(segments),
            'plan': [len(p) for p in plan],
            'timestamp': time.time()
        })

        return plan

    def load_progressive(
        self,
        segments: List[ContextSegment],
        callback: Optional[Callable[[int, List[ContextSegment]], bool]] = None
    ) -> Any:
        """
        渐进式加载
        """
        plan = self.create_loading_plan(segments)
        segment_map = {s.segment_id: s for s in segments}

        loaded_segments = []
        pending_segments = [s.segment_id for s in segments]
        is_complete = False

        for stage_idx, stage_ids in enumerate(plan):
            # 加载当前阶段
            loaded = [segment_map[sid] for sid in stage_ids if sid in segment_map]
            loaded_segments.extend(stage_ids)

            # 从未加载列表中移除
            for sid in stage_ids:
                if sid in pending_segments:
                    pending_segments.remove(sid)

            # 回调
            if callback:
                should_continue = callback(stage_idx, loaded)
                if not should_continue:
                    break

        is_complete = len(pending_segments) == 0

        # 返回兼容的状态对象
        return type('ProgressiveLoadState', (), {
            'stage': len(plan) - 1,
            'total_stages': len(plan),
            'loaded_segments': loaded_segments,
            'pending_segments': pending_segments,
            'is_complete': is_complete
        })()

    def get_iterator(
        self,
        segments: List[ContextSegment]
    ):
        """
        获取渐进式迭代器
        """
        plan = self.create_loading_plan(segments)
        segment_map = {s.segment_id: s for s in segments}

        for stage_ids in plan:
            loaded = [segment_map[sid] for sid in stage_ids if sid in segment_map]
            yield loaded
