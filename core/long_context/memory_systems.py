"""
长上下文记忆系统 - 借鉴视频生成长视频技术
==========================================

本模块实现了多种长上下文记忆系统，灵感来源于视频生成领域处理长视频的技术。

重构说明：
- 内部使用core/unified_algorithms/统一核心
- 通过unified_adapter.py适配器保持API兼容
- 原有API完全保持不变

核心组件：
1. ContextMemoryBank: 外部上下文记忆库（基于UnifiedMemoryBank）
2. LightweightContextMemory: 轻量级可训练记忆
3. HierarchicalContextMemory: 分级记忆系统
4. AdaptiveContextCompressor: 自适应上下文压缩器
5. MemoryFusionLayer: 记忆融合层

纯Python实现，仅使用标准库。
"""

from __future__ import annotations

import math
import random
import time
import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from enum import Enum, auto

# 导入统一核心适配器
from .unified_adapter import (
    ContextMemoryAdapter,
    ContextChunk,
    MemorySlot,
    cosine_similarity,
    normalize_vector,
    compute_hash,
)

# 导入统一核心（用于高级功能）
from ..unified_algorithms.unified_memory import (
    UnifiedMemoryBank,
    UnifiedLightweightMemory,
    UnifiedHierarchicalMemory,
    UnifiedAdaptiveCompressor,
    MemoryEntry,
    MemoryQuery,
)
from ..unified_algorithms.unified_config import (
    UnifiedAlgorithmConfig,
    MemoryRetrievalMode,
    CompressionStrategy,
)


# ============================================================================
# 工具函数（保持向后兼容）
# ============================================================================

# 工具函数现在从unified_adapter导入，这里保留别名以保持兼容性
def euclidean_distance(a: List[float], b: List[float]) -> float:
    """计算欧氏距离"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def random_vector(dim: int, scale: float = 1.0) -> List[float]:
    """生成随机向量"""
    return [random.gauss(0, scale) for _ in range(dim)]


def softmax(x: List[float], temperature: float = 1.0) -> List[float]:
    """计算softmax"""
    if not x:
        return []
    max_x = max(x)
    exp_x = [math.exp((xi - max_x) / temperature) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法"""
    m, n = len(a), len(b[0])
    k = len(b)
    result = [[sum(a[i][l] * b[l][j] for l in range(k)) for j in range(n)] for i in range(m)]
    return result


def transpose(x: List[List[float]]) -> List[List[float]]:
    """矩阵转置"""
    return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]


def vector_add(a: List[float], b: List[float]) -> List[float]:
    """向量加法"""
    return [x + y for x, y in zip(a, b)]


def vector_scale(v: List[float], scale: float) -> List[float]:
    """向量缩放"""
    return [x * scale for x in v]


# ============================================================================
# 数据类定义（从适配器导入，保持兼容）
# ============================================================================

# ContextChunk, MemorySlot 现在从unified_adapter导入
# 这里保留MemoryLevel定义

class MemoryLevel(Enum):
    """记忆层级"""
    SHORT_TERM = auto()   # 短期记忆：最近的几轮
    MEDIUM_TERM = auto()  # 中期记忆：压缩的历史
    LONG_TERM = auto()    # 长期记忆：归档存储


# ============================================================================
# 1. ContextMemoryBank - 外部上下文记忆库
# ============================================================================

class ContextMemoryBank:
    """
    外部上下文记忆库
    
    借鉴视频生成中的MemoryBank技术：
    - 将长上下文存储在外部存储器，避免显存溢出
    - 支持分块存储和检索
    - 使用关键帧机制快速定位重要内容
    - 支持渐进式加载（类似视频的渐进解码）
    
    重构说明：
    - 内部使用ContextMemoryAdapter包装UnifiedMemoryBank
    - 保持原有API完全不变
    
    Attributes:
        chunk_size: 每个块的最大token数
        overlap_size: 相邻块之间的重叠大小
        max_chunks: 最大存储块数
        dim: 嵌入向量维度
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 64,
        max_chunks: int = 10000,
        dim: int = 768,
        keyframe_ratio: float = 0.1
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_chunks = max_chunks
        self.dim = dim
        self.keyframe_ratio = keyframe_ratio
        
        # 内部使用适配器（包装UnifiedMemoryBank）
        self._adapter = ContextMemoryAdapter(
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            max_chunks=max_chunks,
            dim=dim,
            keyframe_ratio=keyframe_ratio
        )
        
        # 保持原有属性（委托给适配器）
        self.chunks: Dict[str, ContextChunk] = self._adapter.chunks
        self.keyframes: Set[str] = self._adapter.keyframes
        self.chunk_order: List[str] = self._adapter.chunk_order
        
        # 索引
        self._embedding_index: List[Tuple[str, List[float]]] = []
        self._index_dirty = True
        
        # 统计
        self.stats = self._adapter.stats
    
    def add_context(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        添加上下文，自动分块
        
        Args:
            content: 原始文本内容
            embedding: 可选的嵌入向量（用于第一个块）
            importance: 内容重要性
            metadata: 元数据
            
        Returns:
            生成的块ID列表
        """
        # 委托给适配器
        chunk_ids = self._adapter.add_context(content, embedding, importance, metadata)
        
        # 更新索引标记
        self._index_dirty = True
        
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
        
        Args:
            query_embedding: 查询嵌入
            top_k: 返回的关键块数
            use_keyframes: 是否优先使用关键帧
            context_window: 每侧扩展的上下文块数
            
        Returns:
            相关上下文块列表（按时间顺序）
        """
        # 委托给适配器
        return self._adapter.retrieve(query_embedding, top_k, use_keyframes, context_window)
    
    def retrieve_by_position(
        self,
        start_pos: int,
        end_pos: int
    ) -> List[ContextChunk]:
        """按位置范围检索"""
        return self._adapter.retrieve_by_position(start_pos, end_pos)

    def progressive_load(
        self,
        chunk_ids: List[str],
        batch_size: int = 10
    ) -> List[List[ContextChunk]]:
        """
        渐进式加载

        借鉴视频的渐进式解码：
        - 先加载关键帧（低分辨率预览）
        - 然后逐步加载完整内容

        Returns:
            分批的块列表
        """
        return self._adapter.progressive_load(chunk_ids, batch_size)
    
    def _chunk_content(self, content: str) -> List[str]:
        """将内容分块"""
        return self._adapter._chunk_content(content)

    def _generate_embedding(self, content: str) -> List[float]:
        """生成简单嵌入"""
        return self._adapter._generate_embedding(content)

    def _compute_keyframe_score(
        self,
        content: str,
        importance: float,
        position: int,
        total_chunks: int
    ) -> float:
        """计算关键帧分数"""
        return self._adapter._compute_keyframe_score(content, importance, position, total_chunks)

    def _rebuild_index(self):
        """重建嵌入索引"""
        self._embedding_index = [
            (chunk_id, chunk.embedding)
            for chunk_id, chunk in self.chunks.items()
        ]
        self._index_dirty = False

    def _enforce_capacity(self):
        """强制执行容量限制"""
        self._adapter._enforce_capacity()

    def _remove_chunk(self, chunk_id: str):
        """移除块并更新引用"""
        self._adapter._remove_chunk(chunk_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._adapter.get_stats()


# ============================================================================
# 2. LightweightContextMemory - 轻量级可训练记忆
# ============================================================================

class LightweightContextMemory:
    """
    轻量级可训练记忆

    借鉴LoRA（Low-Rank Adaptation）技术：
    - 使用低秩矩阵近似全参数更新
    - 高效适应长上下文模式
    - 支持增量学习和记忆融合

    重构说明：
    - 内部使用UnifiedLightweightMemory作为底层
    - 保持原有API完全不变

    Attributes:
        dim: 特征维度
        rank: 低秩维度
        num_slots: 记忆槽位数
        alpha: 缩放因子
    """

    def __init__(
        self,
        dim: int = 768,
        rank: int = 8,
        num_slots: int = 100,
        alpha: float = 1.0
    ):
        self.dim = dim
        self.rank = rank
        self.num_slots = num_slots
        self.alpha = alpha

        # 内部使用统一轻量级记忆
        self._unified_memory = UnifiedLightweightMemory[Dict](
            dim=dim,
            rank=rank,
            capacity=num_slots,
            config=UnifiedAlgorithmConfig.default_config()
        )

        # LoRA矩阵（保持兼容）
        self.A: List[List[float]] = self._init_lora_matrix(dim, rank)
        self.B: List[List[float]] = self._init_lora_matrix(rank, dim)

        # 记忆槽位（委托给统一核心）
        self.slots: Dict[str, MemorySlot] = {}
        self.slot_order: deque = deque(maxlen=num_slots)

        # 梯度累积
        self.grad_accumulator: Dict[str, List[float]] = {}

        # 统计
        self.stats = {
            'updates': 0,
            'queries': 0,
            'adaptations': 0
        }
    
    def _init_lora_matrix(self, rows: int, cols: int) -> List[List[float]]:
        """初始化LoRA矩阵（Kaiming初始化）"""
        scale = math.sqrt(2.0 / (rows + cols))
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]
    
    def adapt(
        self,
        context_vectors: List[List[float]],
        target_vectors: List[List[float]],
        learning_rate: float = 0.01
    ) -> float:
        """
        适应新上下文
        
        Args:
            context_vectors: 上下文向量列表
            target_vectors: 目标输出向量列表
            learning_rate: 学习率
            
        Returns:
            适应损失
        """
        if len(context_vectors) != len(target_vectors):
            raise ValueError("Context and target vectors must have same length")
        
        total_loss = 0.0
        
        for ctx_vec, tgt_vec in zip(context_vectors, target_vectors):
            # 前向传播
            adapted = self.forward(ctx_vec)
            
            # 计算损失（MSE）
            loss = sum((a - t) ** 2 for a, t in zip(adapted, tgt_vec)) / len(tgt_vec)
            total_loss += loss
            
            # 反向传播（简化版梯度计算）
            error = [a - t for a, t in zip(adapted, tgt_vec)]
            
            # 更新A和B（梯度下降）
            self._update_lora(ctx_vec, error, learning_rate)
        
        self.stats['adaptations'] += 1
        return total_loss / len(context_vectors)
    
    def forward(self, x: List[float]) -> List[float]:
        """
        前向传播: y = x + alpha/rank * x @ B @ A
        
        Args:
            x: 输入向量 [dim]
            
        Returns:
            适应后的向量 [dim]
        """
        # x @ B: [1, dim] @ [dim, rank] = [1, rank]
        xB = [sum(x[i] * self.B[i][j] for i in range(self.dim)) for j in range(self.rank)]
        
        # (x @ B) @ A: [1, rank] @ [rank, dim] = [1, dim]
        xBA = [sum(xB[j] * self.A[j][i] for j in range(self.rank)) for i in range(self.dim)]
        
        # 缩放并残差连接
        scale = self.alpha / self.rank
        return [x[i] + scale * xBA[i] for i in range(self.dim)]
    
    def store_memory(
        self,
        key: List[float],
        value: List[float],
        slot_name: Optional[str] = None
    ) -> str:
        """
        存储记忆槽位

        Args:
            key: 查询键
            value: 记忆值
            slot_name: 可选的槽位名称

        Returns:
            槽位ID
        """
        slot_id = slot_name or f"slot_{compute_hash(str(key))}_{int(time.time() * 1000)}"

        # 压缩键值到低秩表示
        compressed_key = self._compress(key)
        compressed_value = self._compress(value)

        slot = MemorySlot(
            slot_id=slot_id,
            key=compressed_key,
            value=compressed_value,
            rank=self.rank,
            alpha=self.alpha
        )

        # 同时存储到统一轻量级记忆
        self._unified_memory.store({
            'slot_id': slot_id,
            'key': compressed_key,
            'value': compressed_value,
            'timestamp': time.time()
        })

        # LRU淘汰
        if len(self.slots) >= self.num_slots and self.slot_order:
            oldest = self.slot_order.popleft()
            self.slots.pop(oldest, None)

        self.slots[slot_id] = slot
        self.slot_order.append(slot_id)

        return slot_id
    
    def retrieve_memory(
        self,
        query: List[float],
        top_k: int = 3
    ) -> List[Tuple[List[float], float]]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            top_k: 返回数量
            
        Returns:
            (记忆值, 相似度) 列表
        """
        self.stats['queries'] += 1
        
        compressed_query = self._compress(query)
        
        # 计算相似度
        scored = []
        for slot in self.slots.values():
            sim = cosine_similarity(compressed_query, slot.key)
            # 解压值
            value = self._decompress(slot.value)
            scored.append((value, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def _compress(self, vector: List[float]) -> List[float]:
        """压缩向量到低秩空间"""
        # 使用A矩阵压缩: x @ A
        return [sum(vector[i] * self.A[j][i] for i in range(self.dim)) for j in range(self.rank)]
    
    def _decompress(self, compressed: List[float]) -> List[float]:
        """从低秩空间解压"""
        # 使用B矩阵解压: x @ B
        return [sum(compressed[j] * self.B[i][j] for j in range(self.rank)) for i in range(self.dim)]
    
    def _update_lora(
        self,
        x: List[float],
        error: List[float],
        learning_rate: float
    ):
        """更新LoRA参数"""
        # 简化的梯度更新
        # dL/dB = x^T @ (error @ A^T)
        # dL/dA = (x @ B)^T @ error
        
        scale = learning_rate * self.alpha / self.rank
        
        # 更新B
        for i in range(self.dim):
            for j in range(self.rank):
                grad = x[i] * error[i] * self.A[j][i]
                self.B[i][j] -= scale * grad
        
        # 更新A
        for j in range(self.rank):
            for i in range(self.dim):
                xB_j = sum(x[k] * self.B[k][j] for k in range(self.dim))
                grad = xB_j * error[i]
                self.A[j][i] -= scale * grad
        
        self.stats['updates'] += 1
    
    def merge_memories(self, other: 'LightweightContextMemory', weight: float = 0.5):
        """
        合并另一个轻量级记忆
        
        Args:
            other: 另一个记忆实例
            weight: 合并权重
        """
        # 合并LoRA矩阵
        for i in range(self.dim):
            for j in range(self.rank):
                self.B[i][j] = (1 - weight) * self.B[i][j] + weight * other.B[i][j]
        
        for j in range(self.rank):
            for i in range(self.dim):
                self.A[j][i] = (1 - weight) * self.A[j][i] + weight * other.A[j][i]
        
        # 合并槽位
        for slot_id, slot in other.slots.items():
            if slot_id not in self.slots:
                self.store_memory(
                    self._decompress(slot.key),
                    self._decompress(slot.value),
                    slot_id
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        unified_stats = self._unified_memory.get_stats()
        return {
            **self.stats,
            'dim': self.dim,
            'rank': self.rank,
            'compression_ratio': self.dim / self.rank,
            'num_slots': len(self.slots),
            'parameters': (self.dim * self.rank + self.rank * self.dim),
            'unified_memory_stats': unified_stats
        }


# ============================================================================
# 3. HierarchicalContextMemory - 分级记忆系统
# ============================================================================

class HierarchicalContextMemory:
    """
    分级上下文记忆系统

    三层架构：
    - 短期记忆：最近的几轮对话，完整保留
    - 中期记忆：压缩的历史，保留关键信息
    - 长期记忆：归档存储，高度压缩

    借鉴视频处理中的多分辨率思想：
    - 短期：高分辨率（完整上下文）
    - 中期：中分辨率（摘要）
    - 长期：低分辨率（关键帧）

    重构说明：
    - 内部使用UnifiedHierarchicalMemory作为底层
    - 保持原有API完全不变
    """

    def __init__(
        self,
        short_term_capacity: int = 10,
        medium_term_capacity: int = 100,
        long_term_capacity: int = 1000,
        dim: int = 768,
        compression_ratio: float = 0.5
    ):
        self.dim = dim
        self.compression_ratio = compression_ratio

        # 内部使用统一分级记忆
        self._unified_memory = UnifiedHierarchicalMemory[ContextChunk](
            short_term_size=short_term_capacity,
            medium_term_size=medium_term_capacity,
            long_term_size=long_term_capacity,
            config=UnifiedAlgorithmConfig.default_config()
        )

        # 三层存储（保持兼容）
        self.short_term: OrderedDict[str, ContextChunk] = OrderedDict()
        self.medium_term: Dict[str, ContextChunk] = {}
        self.long_term: Dict[str, ContextChunk] = {}

        # 容量
        self.capacities = {
            MemoryLevel.SHORT_TERM: short_term_capacity,
            MemoryLevel.MEDIUM_TERM: medium_term_capacity,
            MemoryLevel.LONG_TERM: long_term_capacity
        }

        # 压缩器
        self.compression_matrix = self._init_compression_matrix()

        # 迁移阈值
        self.migration_threshold = {
            MemoryLevel.SHORT_TERM: 5,   # 访问5次后迁移到中期
            MemoryLevel.MEDIUM_TERM: 10  # 访问10次后迁移到长期
        }

        # 统计
        self.stats = {
            'migrations': {level: 0 for level in MemoryLevel},
            'compressions': 0,
            'retrievals': {level: 0 for level in MemoryLevel}
        }
    
    def _init_compression_matrix(self) -> List[List[float]]:
        """初始化压缩矩阵"""
        compressed_dim = int(self.dim * self.compression_ratio)
        scale = math.sqrt(1.0 / self.dim)
        return [[random.gauss(0, scale) for _ in range(compressed_dim)] 
                for _ in range(self.dim)]
    
    def add(
        self,
        content: str,
        embedding: List[float],
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        添加新记忆到短期记忆
        
        Returns:
            记忆ID
        """
        memory_id = f"mem_{compute_hash(content)}_{int(time.time() * 1000)}"
        
        chunk = ContextChunk(
            chunk_id=memory_id,
            content=content,
            embedding=embedding[:],
            importance=importance,
            metadata=metadata or {}
        )
        
        # 添加到短期记忆
        self.short_term[memory_id] = chunk
        
        # 检查容量并迁移
        self._enforce_capacity(MemoryLevel.SHORT_TERM)
        
        return memory_id
    
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        levels: Optional[List[MemoryLevel]] = None
    ) -> Dict[MemoryLevel, List[Tuple[ContextChunk, float]]]:
        """
        分层检索
        
        Args:
            query_embedding: 查询嵌入
            top_k: 每层返回数量
            levels: 要检索的层级（None表示全部）
            
        Returns:
            各层检索结果
        """
        levels = levels or list(MemoryLevel)
        results = {}
        
        for level in levels:
            store = self._get_store(level)
            scored = []
            
            for chunk in store.values():
                sim = cosine_similarity(query_embedding, chunk.embedding)
                # 根据层级调整分数
                level_boost = {
                    MemoryLevel.SHORT_TERM: 1.2,
                    MemoryLevel.MEDIUM_TERM: 1.0,
                    MemoryLevel.LONG_TERM: 0.8
                }[level]
                scored.append((chunk, sim * level_boost))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            results[level] = scored[:top_k]
            
            # 更新统计
            self.stats['retrievals'][level] += len(results[level])
            
            # 更新访问计数
            for chunk, _ in results[level]:
                chunk.touch()
        
        return results
    
    def migrate(self, force: bool = False):
        """
        执行记忆迁移
        
        将频繁访问的短期记忆迁移到中期
        将频繁访问的中期记忆迁移到长期
        """
        # 短期 -> 中期
        to_migrate = []
        for memory_id, chunk in list(self.short_term.items()):
            if chunk.access_count >= self.migration_threshold[MemoryLevel.SHORT_TERM] or force:
                to_migrate.append(memory_id)
        
        for memory_id in to_migrate:
            self._migrate_single(memory_id, MemoryLevel.SHORT_TERM, MemoryLevel.MEDIUM_TERM)
        
        # 中期 -> 长期
        to_migrate = []
        for memory_id, chunk in list(self.medium_term.items()):
            if chunk.access_count >= self.migration_threshold[MemoryLevel.MEDIUM_TERM] or force:
                to_migrate.append(memory_id)
        
        for memory_id in to_migrate:
            self._migrate_single(memory_id, MemoryLevel.MEDIUM_TERM, MemoryLevel.LONG_TERM)
    
    def _migrate_single(
        self,
        memory_id: str,
        from_level: MemoryLevel,
        to_level: MemoryLevel
    ):
        """迁移单个记忆"""
        from_store = self._get_store(from_level)
        to_store = self._get_store(to_level)
        
        if memory_id not in from_store:
            return
        
        chunk = from_store[memory_id]
        
        # 根据目标层级压缩
        if to_level == MemoryLevel.LONG_TERM:
            chunk = self._compress_chunk(chunk)
        elif to_level == MemoryLevel.MEDIUM_TERM:
            chunk = self._summarize_chunk(chunk)
        
        # 迁移
        to_store[memory_id] = chunk
        del from_store[memory_id]
        
        # 更新统计
        self.stats['migrations'][from_level] += 1
        
        # 检查目标层级容量
        self._enforce_capacity(to_level)
    
    def _compress_chunk(self, chunk: ContextChunk) -> ContextChunk:
        """压缩记忆块"""
        # 压缩嵌入
        compressed_embedding = self._compress_vector(chunk.embedding)
        
        # 压缩内容（简化：只保留前N个字符）
        compressed_content = chunk.content[:len(chunk.content) // 2]
        
        return ContextChunk(
            chunk_id=chunk.chunk_id,
            content=compressed_content,
            embedding=compressed_embedding,
            timestamp=chunk.timestamp,
            importance=chunk.importance,
            access_count=chunk.access_count,
            position=chunk.position,
            metadata={**chunk.metadata, 'compressed': True}
        )
    
    def _summarize_chunk(self, chunk: ContextChunk) -> ContextChunk:
        """摘要记忆块（简化版）"""
        # 实际应用中使用摘要模型
        # 这里简化处理：保留关键句
        sentences = chunk.content.split('.')
        key_sentences = sentences[:max(1, len(sentences) // 2)]
        summary = '. '.join(key_sentences)
        
        return ContextChunk(
            chunk_id=chunk.chunk_id,
            content=summary,
            embedding=chunk.embedding,  # 保持嵌入不变
            timestamp=chunk.timestamp,
            importance=chunk.importance,
            access_count=chunk.access_count,
            position=chunk.position,
            metadata={**chunk.metadata, 'summarized': True}
        )
    
    def _compress_vector(self, vector: List[float]) -> List[float]:
        """压缩向量"""
        compressed_dim = int(self.dim * self.compression_ratio)
        return [sum(vector[i] * self.compression_matrix[i][j] 
                   for i in range(self.dim)) for j in range(compressed_dim)]
    
    def _get_store(self, level: MemoryLevel) -> Dict[str, ContextChunk]:
        """获取对应层级的存储"""
        return {
            MemoryLevel.SHORT_TERM: self.short_term,
            MemoryLevel.MEDIUM_TERM: self.medium_term,
            MemoryLevel.LONG_TERM: self.long_term
        }[level]
    
    def _enforce_capacity(self, level: MemoryLevel):
        """强制执行容量限制"""
        store = self._get_store(level)
        capacity = self.capacities[level]
        
        while len(store) > capacity:
            # 移除最不重要且访问最少的
            to_remove = min(
                store.keys(),
                key=lambda k: store[k].importance * (1 + store[k].access_count)
            )
            del store[to_remove]
    
    def get_all_memories(
        self,
        level: Optional[MemoryLevel] = None
    ) -> List[ContextChunk]:
        """获取所有记忆"""
        if level:
            return list(self._get_store(level).values())
        
        all_memories = []
        for lvl in MemoryLevel:
            all_memories.extend(self._get_store(lvl).values())
        return all_memories
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        unified_stats = self._unified_memory.get_stats()
        return {
            **self.stats,
            'sizes': {
                level: len(self._get_store(level))
                for level in MemoryLevel
            },
            'total_memories': sum(len(self._get_store(lvl)) for lvl in MemoryLevel),
            'compression_ratio': self.compression_ratio,
            'unified_memory_stats': unified_stats
        }


# ============================================================================
# 4. AdaptiveContextCompressor - 自适应上下文压缩器
# ============================================================================

class AdaptiveContextCompressor:
    """
    自适应上下文压缩器

    根据内容重要性动态调整压缩率：
    - 重要内容：低压缩率，保留更多信息
    - 次要内容：高压缩率，节省存储

    借鉴视频编码中的率失真优化：
    - 在信息保留和存储效率之间权衡

    重构说明：
    - 内部使用UnifiedAdaptiveCompressor作为底层
    - 保持原有API完全不变
    """

    def __init__(
        self,
        dim: int = 768,
        min_compression: float = 0.1,
        max_compression: float = 0.9,
        target_size: int = 4096
    ):
        self.dim = dim
        self.min_compression = min_compression
        self.max_compression = max_compression
        self.target_size = target_size

        # 内部使用统一自适应压缩器
        self._unified_compressor = UnifiedAdaptiveCompressor[ContextChunk](
            config=UnifiedAlgorithmConfig.default_config()
        )

        # 重要性评估器（保持兼容）
        self.importance_weights = random_vector(dim, 0.1)

        # 压缩历史
        self.compression_history: List[Dict] = []
    
    def compress(
        self,
        chunks: List[ContextChunk],
        importance_threshold: float = 0.5
    ) -> List[ContextChunk]:
        """
        自适应压缩上下文
        
        Args:
            chunks: 上下文块列表
            importance_threshold: 重要性阈值
            
        Returns:
            压缩后的块列表
        """
        if not chunks:
            return []
        
        # 评估每个块的重要性
        scored_chunks = []
        for chunk in chunks:
            importance = self._evaluate_importance(chunk)
            scored_chunks.append((chunk, importance))
        
        # 按重要性排序
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # 确定压缩率
        total_size = sum(len(c.content) for c in chunks)
        if total_size <= self.target_size:
            # 不需要压缩
            return chunks
        
        # 自适应分配压缩率
        compressed = []
        remaining_size = self.target_size
        
        for chunk, importance in scored_chunks:
            # 重要性高的块使用低压缩率
            if importance > importance_threshold:
                compression_rate = self.min_compression + (1 - importance) * 0.2
            else:
                compression_rate = self.max_compression - importance * 0.2
            
            # 压缩
            compressed_chunk = self._apply_compression(chunk, compression_rate)
            compressed.append(compressed_chunk)
            
            # 记录历史
            self.compression_history.append({
                'chunk_id': chunk.chunk_id,
                'original_size': len(chunk.content),
                'compressed_size': len(compressed_chunk.content),
                'compression_rate': compression_rate,
                'importance': importance
            })
        
        # 按原始位置排序
        compressed.sort(key=lambda x: x.position)
        
        return compressed
    
    def _evaluate_importance(self, chunk: ContextChunk) -> float:
        """评估块的重要性"""
        # 基于多个因素
        factors = []
        
        # 1. 显式重要性
        factors.append(chunk.importance)
        
        # 2. 访问频率
        access_score = min(chunk.access_count / 10, 1.0)
        factors.append(access_score)
        
        # 3. 内容长度（适中的长度更重要）
        length_score = 1.0 - abs(len(chunk.content) - 200) / 400
        factors.append(max(0, length_score))
        
        # 4. 关键帧标记
        if chunk.is_keyframe:
            factors.append(1.0)
        
        # 加权平均
        return sum(factors) / len(factors)
    
    def _apply_compression(
        self,
        chunk: ContextChunk,
        compression_rate: float
    ) -> ContextChunk:
        """应用压缩"""
        # 压缩内容
        original_len = len(chunk.content)
        target_len = int(original_len * (1 - compression_rate))
        
        if target_len >= original_len:
            return chunk
        
        # 简化压缩：保留开头和结尾，中间省略
        head_len = target_len // 2
        tail_len = target_len - head_len
        
        compressed_content = (
            chunk.content[:head_len] + 
            " ... [compressed] ... " + 
            chunk.content[-tail_len:] if tail_len > 0 else ""
        )
        
        # 压缩嵌入
        compressed_embedding = self._compress_embedding(chunk.embedding, compression_rate)
        
        return ContextChunk(
            chunk_id=chunk.chunk_id,
            content=compressed_content,
            embedding=compressed_embedding,
            timestamp=chunk.timestamp,
            importance=chunk.importance,
            access_count=chunk.access_count,
            position=chunk.position,
            metadata={
                **chunk.metadata,
                'compressed': True,
                'compression_rate': compression_rate,
                'original_length': original_len
            },
            is_keyframe=chunk.is_keyframe,
            keyframe_score=chunk.keyframe_score,
            prev_chunk_id=chunk.prev_chunk_id,
            next_chunk_id=chunk.next_chunk_id
        )
    
    def _compress_embedding(
        self,
        embedding: List[float],
        compression_rate: float
    ) -> List[float]:
        """压缩嵌入向量"""
        # 使用PCA风格的降维（简化版）
        target_dim = max(1, int(self.dim * (1 - compression_rate)))
        
        # 均匀采样维度
        step = self.dim // target_dim
        compressed = [embedding[i * step] for i in range(target_dim)]
        
        return normalize_vector(compressed)
    
    def decompress(self, chunk: ContextChunk) -> ContextChunk:
        """解压（近似还原）"""
        if not chunk.metadata.get('compressed'):
            return chunk
        
        # 标记为已解压
        new_metadata = dict(chunk.metadata)
        new_metadata['decompressed'] = True
        del new_metadata['compressed']
        
        return ContextChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,  # 文本压缩不可逆
            embedding=chunk.embedding,  # 嵌入压缩不可逆
            timestamp=chunk.timestamp,
            importance=chunk.importance,
            access_count=chunk.access_count,
            position=chunk.position,
            metadata=new_metadata,
            is_keyframe=chunk.is_keyframe,
            keyframe_score=chunk.keyframe_score,
            prev_chunk_id=chunk.prev_chunk_id,
            next_chunk_id=chunk.next_chunk_id
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.compression_history:
            return {'compression_history': 0}
        
        avg_compression = sum(
            h['compression_rate'] for h in self.compression_history
        ) / len(self.compression_history)
        
        return {
            'compression_history': len(self.compression_history),
            'average_compression_rate': avg_compression,
            'total_original_size': sum(h['original_size'] for h in self.compression_history),
            'total_compressed_size': sum(h['compressed_size'] for h in self.compression_history)
        }


# ============================================================================
# 5. MemoryFusionLayer - 记忆融合层
# ============================================================================

class MemoryFusionLayer:
    """
    记忆融合层
    
    将来自多个源的记忆进行智能融合：
    - 短期工作记忆
    - 检索到的历史记忆
    - 外部知识
    
    使用注意力机制加权融合
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        fusion_temperature: float = 1.0
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.fusion_temperature = fusion_temperature
        
        # 融合权重投影
        self.query_proj = self._init_projection(dim, dim)
        self.key_proj = self._init_projection(dim, dim)
        self.value_proj = self._init_projection(dim, dim)
        self.output_proj = self._init_projection(dim, dim)
        
        # 源权重（可学习）
        self.source_weights: Dict[str, float] = {
            'short_term': 1.0,
            'retrieved': 0.8,
            'external': 0.6
        }
    
    def _init_projection(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化投影矩阵"""
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] 
                for _ in range(in_dim)]
    
    def fuse(
        self,
        query: List[float],
        memories: Dict[str, List[ContextChunk]],
        top_k_per_source: int = 3
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        融合多源记忆
        
        Args:
            query: 查询向量
            memories: 各源记忆 {'source_name': [chunks]}
            top_k_per_source: 每源取top-k
            
        Returns:
            (融合后的向量, 各源注意力权重)
        """
        # 收集所有记忆
        all_memories = []
        source_indices = []
        
        for source_name, chunks in memories.items():
            # 按相似度排序
            scored = [(c, cosine_similarity(query, c.embedding)) for c in chunks]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # 取top-k
            weight = self.source_weights.get(source_name, 0.5)
            for chunk, sim in scored[:top_k_per_source]:
                all_memories.append((chunk, sim * weight))
                source_indices.append(source_name)
        
        if not all_memories:
            return query, {}
        
        # 计算注意力
        attention_weights = self._compute_attention(query, all_memories)
        
        # 加权融合
        fused = [0.0] * self.dim
        source_attentions: Dict[str, List[float]] = {}
        
        for i, ((chunk, base_sim), attn) in enumerate(zip(all_memories, attention_weights)):
            source = source_indices[i]
            if source not in source_attentions:
                source_attentions[source] = []
            source_attentions[source].append(attn)
            
            for j in range(self.dim):
                fused[j] += attn * chunk.embedding[j]
        
        # 残差连接
        for j in range(self.dim):
            fused[j] = 0.7 * query[j] + 0.3 * fused[j]
        
        # 计算各源平均注意力
        avg_source_attention = {
            source: sum(attns) / len(attns) if attns else 0.0
            for source, attns in source_attentions.items()
        }
        
        return normalize_vector(fused), avg_source_attention
    
    def _compute_attention(
        self,
        query: List[float],
        memories: List[Tuple[ContextChunk, float]]
    ) -> List[float]:
        """计算注意力权重"""
        # 投影
        q_proj = [sum(query[i] * self.query_proj[i][j] for i in range(self.dim)) 
                  for j in range(self.dim)]
        
        # 计算分数
        scores = []
        for chunk, base_sim in memories:
            k_proj = [sum(chunk.embedding[i] * self.key_proj[i][j] for i in range(self.dim))
                     for j in range(self.dim)]
            
            # 点积注意力
            score = sum(q_proj[j] * k_proj[j] for j in range(self.dim))
            score = score / math.sqrt(self.dim)  # 缩放
            score += base_sim  # 加入基础相似度
            scores.append(score)
        
        # Softmax
        return softmax(scores, self.fusion_temperature)
    
    def update_source_weight(self, source_name: str, weight: float):
        """更新源权重"""
        self.source_weights[source_name] = max(0.0, min(1.0, weight))
    
    def cross_fusion(
        self,
        memory_sets: List[List[ContextChunk]],
        fusion_method: str = 'attention'
    ) -> List[ContextChunk]:
        """
        跨记忆集融合
        
        Args:
            memory_sets: 多个记忆集
            fusion_method: 融合方法 ('attention', 'average', 'max')
            
        Returns:
            融合后的记忆
        """
        if not memory_sets:
            return []
        
        if fusion_method == 'average':
            return self._average_fusion(memory_sets)
        elif fusion_method == 'max':
            return self._max_fusion(memory_sets)
        else:  # attention
            return self._attention_fusion(memory_sets)
    
    def _average_fusion(
        self,
        memory_sets: List[List[ContextChunk]]
    ) -> List[ContextChunk]:
        """平均融合"""
        # 收集所有唯一块
        all_chunks = {}
        for memory_set in memory_sets:
            for chunk in memory_set:
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = []
                all_chunks[chunk.chunk_id].append(chunk)
        
        fused = []
        for chunk_id, chunks in all_chunks.items():
            # 平均嵌入
            avg_embedding = [
                sum(c.embedding[i] for c in chunks) / len(chunks)
                for i in range(self.dim)
            ]
            
            # 取最新内容
            latest_chunk = max(chunks, key=lambda c: c.timestamp)
            
            fused_chunk = ContextChunk(
                chunk_id=chunk_id,
                content=latest_chunk.content,
                embedding=avg_embedding,
                timestamp=latest_chunk.timestamp,
                importance=max(c.importance for c in chunks),
                access_count=sum(c.access_count for c in chunks),
                position=latest_chunk.position,
                metadata={**latest_chunk.metadata, 'fused': True, 'fusion_count': len(chunks)}
            )
            fused.append(fused_chunk)
        
        return fused
    
    def _max_fusion(
        self,
        memory_sets: List[List[ContextChunk]]
    ) -> List[ContextChunk]:
        """最大池化融合"""
        all_chunks = {}
        for memory_set in memory_sets:
            for chunk in memory_set:
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk
                elif chunk.importance > all_chunks[chunk.chunk_id].importance:
                    all_chunks[chunk.chunk_id] = chunk
        
        return list(all_chunks.values())
    
    def _attention_fusion(
        self,
        memory_sets: List[List[ContextChunk]]
    ) -> List[ContextChunk]:
        """注意力融合"""
        # 简化版：使用第一个集合作为查询
        if not memory_sets[0]:
            return []
        
        query_chunk = memory_sets[0][0]
        fused, _ = self.fuse(
            query_chunk.embedding,
            {f'set_{i}': ms for i, ms in enumerate(memory_sets)}
        )
        
        # 创建融合后的虚拟块
        return [ContextChunk(
            chunk_id='fused_query',
            content=query_chunk.content,
            embedding=fused,
            timestamp=time.time(),
            importance=query_chunk.importance,
            metadata={'fused': True, 'fusion_method': 'attention'}
        )]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'source_weights': self.source_weights.copy(),
            'dim': self.dim,
            'num_heads': self.num_heads,
            'parameters': (
                self.dim * self.dim * 4  # 4个投影矩阵
            )
        }
