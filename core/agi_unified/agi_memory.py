"""
AGI记忆系统
===========

基于UnifiedMemoryBank的AGI记忆系统实现。

特点：
- 分层记忆架构（工作记忆、短期记忆、长期记忆）
- 自适应压缩和检索
- 基于重要性的记忆管理
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
import time
import hashlib

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


@dataclass
class AGIMemoryItem:
    """AGI记忆项"""
    item_id: str
    content: Any
    embedding: List[float] = field(default_factory=list)
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    memory_level: str = "short_term"  # working, short_term, long_term
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self):
        """记录访问"""
        self.access_count += 1
        self.metadata['last_access'] = time.time()


class AGIMemorySystem:
    """
    AGI记忆系统

    基于统一核心的三层记忆架构：
    1. 工作记忆：当前活跃信息，容量有限
    2. 短期记忆：近期信息，中等容量
    3. 长期记忆：归档信息，大容量

    Attributes:
        dim: 嵌入维度
        working_capacity: 工作记忆容量
        short_capacity: 短期记忆容量
        long_capacity: 长期记忆容量
    """

    def __init__(
        self,
        dim: int = 768,
        working_capacity: int = 10,
        short_capacity: int = 100,
        long_capacity: int = 10000,
        compression_ratio: float = 0.5
    ):
        self.dim = dim
        self.working_capacity = working_capacity
        self.short_capacity = short_capacity
        self.long_capacity = long_capacity
        self.compression_ratio = compression_ratio

        # 创建统一配置
        self.config = UnifiedAlgorithmConfig.default_config()

        # 三层记忆系统（使用统一分级记忆）
        self._hierarchical_memory = UnifiedHierarchicalMemory[AGIMemoryItem](
            dim=dim,
            num_levels=3,
            compression_ratio=compression_ratio,
            config=self.config
        )

        # 自适应压缩器
        self._compressor = UnifiedAdaptiveCompressor[AGIMemoryItem](
            config=self.config
        )

        # 记忆存储
        self.working_memory: Dict[str, AGIMemoryItem] = {}
        self.short_memory: Dict[str, AGIMemoryItem] = {}
        self.long_memory: Dict[str, AGIMemoryItem] = {}

        # 索引
        self._embedding_index: List[Tuple[str, List[float]]] = []

        # 统计
        self.stats = {
            'stores': 0,
            'retrievals': 0,
            'migrations': {'working_to_short': 0, 'short_to_long': 0},
            'compressions': 0
        }

    def store(
        self,
        content: Any,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        存储记忆

        Args:
            content: 记忆内容
            embedding: 嵌入向量
            importance: 重要性分数
            metadata: 元数据

        Returns:
            记忆ID
        """
        item_id = f"mem_{hashlib.md5(str(content).encode()).hexdigest()[:16]}"

        # 生成嵌入（如果没有提供）
        if embedding is None:
            embedding = self._generate_embedding(content)

        item = AGIMemoryItem(
            item_id=item_id,
            content=content,
            embedding=embedding,
            importance=importance,
            memory_level="working",
            metadata=metadata or {}
        )

        # 存储到工作记忆
        self.working_memory[item_id] = item

        # 同时存储到统一记忆系统
        entry = MemoryEntry(
            data=item,
            timestamp=time.time(),
            importance=importance,
            metadata={'level': 'working', 'item_id': item_id}
        )
        self._hierarchical_memory.store(entry, level=0)

        self.stats['stores'] += 1

        # 检查容量并迁移
        self._enforce_capacity()

        return item_id

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        memory_levels: Optional[List[str]] = None
    ) -> List[AGIMemoryItem]:
        """
        检索记忆

        Args:
            query_embedding: 查询嵌入
            top_k: 返回数量
            memory_levels: 要检索的记忆层级

        Returns:
            记忆项列表
        """
        self.stats['retrievals'] += 1

        memory_levels = memory_levels or ['working', 'short_term', 'long_term']

        results = []

        # 从各层级检索
        if 'working' in memory_levels:
            results.extend(self._search_level(self.working_memory, query_embedding, top_k))

        if 'short_term' in memory_levels:
            results.extend(self._search_level(self.short_memory, query_embedding, top_k))

        if 'long_term' in memory_levels:
            results.extend(self._search_level(self.long_memory, query_embedding, top_k))

        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)

        # 返回top-k
        return [item for item, _ in results[:top_k]]

    def _search_level(
        self,
        memory: Dict[str, AGIMemoryItem],
        query_embedding: List[float],
        top_k: int
    ) -> List[Tuple[AGIMemoryItem, float]]:
        """在指定层级搜索"""
        results = []

        for item in memory.values():
            if item.embedding:
                similarity = self._cosine_similarity(query_embedding, item.embedding)
                # 加权：重要性高的优先
                weighted_score = similarity * (1 + item.importance)
                results.append((item, weighted_score))
                item.touch()

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def migrate_memory(self, item_id: str, from_level: str, to_level: str):
        """
        迁移记忆

        Args:
            item_id: 记忆ID
            from_level: 源层级
            to_level: 目标层级
        """
        source = self._get_memory_by_level(from_level)
        target = self._get_memory_by_level(to_level)

        if item_id not in source:
            return

        item = source[item_id]

        # 如果是迁移到长期记忆，进行压缩
        if to_level == 'long_term':
            item = self._compress_item(item)

        item.memory_level = to_level
        target[item_id] = item
        del source[item_id]

        # 更新统计
        key = f"{from_level}_to_{to_level}"
        if key in self.stats['migrations']:
            self.stats['migrations'][key] += 1

    def _compress_item(self, item: AGIMemoryItem) -> AGIMemoryItem:
        """压缩记忆项"""
        # 压缩嵌入
        target_dim = int(self.dim * self.compression_ratio)
        if len(item.embedding) > target_dim:
            # 均匀采样
            step = len(item.embedding) // target_dim
            item.embedding = [item.embedding[i * step] for i in range(target_dim)]

        item.metadata['compressed'] = True
        self.stats['compressions'] += 1

        return item

    def _enforce_capacity(self):
        """强制执行容量限制"""
        # 工作记忆 -> 短期记忆
        while len(self.working_memory) > self.working_capacity:
            oldest_id = min(
                self.working_memory.keys(),
                key=lambda k: (
                    self.working_memory[k].importance,
                    self.working_memory[k].access_count,
                    self.working_memory[k].timestamp
                )
            )
            self.migrate_memory(oldest_id, 'working', 'short_term')

        # 短期记忆 -> 长期记忆
        while len(self.short_memory) > self.short_capacity:
            oldest_id = min(
                self.short_memory.keys(),
                key=lambda k: (
                    self.short_memory[k].importance,
                    self.short_memory[k].access_count,
                    self.short_memory[k].timestamp
                )
            )
            self.migrate_memory(oldest_id, 'short_term', 'long_term')

        # 长期记忆容量限制
        while len(self.long_memory) > self.long_capacity:
            oldest_id = min(
                self.long_memory.keys(),
                key=lambda k: self.long_memory[k].timestamp
            )
            del self.long_memory[oldest_id]

    def _get_memory_by_level(self, level: str) -> Dict[str, AGIMemoryItem]:
        """根据层级获取记忆字典"""
        if level == 'working':
            return self.working_memory
        elif level == 'short_term':
            return self.short_memory
        elif level == 'long_term':
            return self.long_memory
        else:
            return {}

    def _generate_embedding(self, content: Any) -> List[float]:
        """生成嵌入（简化版）"""
        content_str = str(content)
        embedding = [0.0] * self.dim
        for i, char in enumerate(content_str):
            idx = i % self.dim
            embedding[idx] += ord(char) / 255.0

        # 归一化
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        unified_stats = self._hierarchical_memory.get_stats()
        return {
            **self.stats,
            'working_size': len(self.working_memory),
            'short_size': len(self.short_memory),
            'long_size': len(self.long_memory),
            'total_memories': len(self.working_memory) + len(self.short_memory) + len(self.long_memory),
            'unified_stats': unified_stats
        }

    def clear(self, level: Optional[str] = None):
        """
        清除记忆

        Args:
            level: 要清除的层级（None表示全部）
        """
        if level is None:
            self.working_memory.clear()
            self.short_memory.clear()
            self.long_memory.clear()
        else:
            memory = self._get_memory_by_level(level)
            memory.clear()
