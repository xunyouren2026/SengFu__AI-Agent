"""
统一记忆系统模块

提供通用的记忆管理功能，支持AGI和视频生成系统的记忆需求。
支持序列数据（视频帧/文本token）的统一存储、检索和压缩。

核心组件：
- UnifiedMemoryBank: 通用记忆库
- UnifiedLightweightMemory: 轻量级可训练记忆
- UnifiedHierarchicalMemory: 分级记忆系统
- UnifiedAdaptiveCompressor: 自适应压缩器
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable, Tuple, Union
from collections import deque
import heapq
import math
from enum import Enum, auto

from .unified_config import (
    UnifiedAlgorithmConfig, 
    MemoryRetrievalMode, 
    CompressionStrategy,
    T, S, K, V
)


# ============================================================================
# 记忆条目和数据结构
# ============================================================================

@dataclass
class MemoryEntry(Generic[T]):
    """
    记忆条目
    
    存储单个记忆单元，支持任意类型的数据。
    包含时间戳、重要性分数和元数据。
    
    Attributes:
        data: 记忆数据
        timestamp: 时间戳（用于时序排序）
        importance: 重要性分数 (0.0-1.0)
        metadata: 额外元数据
        compressed: 是否已压缩
    """
    data: T
    timestamp: float = field(default_factory=lambda: 0.0)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    
    def __post_init__(self):
        """验证重要性分数"""
        self.importance = max(0.0, min(1.0, self.importance))


@dataclass
class MemoryQuery(Generic[T]):
    """
    记忆查询
    
    用于检索记忆的结构化查询。
    
    Attributes:
        query_data: 查询数据
        top_k: 返回结果数量
        threshold: 相似度阈值
        recency_weight: 时间衰减权重
    """
    query_data: T
    top_k: int = 5
    threshold: float = 0.0
    recency_weight: float = 0.3


# ============================================================================
# 抽象基类
# ============================================================================

class BaseMemory(ABC, Generic[T]):
    """
    记忆系统抽象基类
    
    所有记忆系统的基类，定义通用接口。
    """
    
    def __init__(self, config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化记忆系统
        
        Args:
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.default_config()
    
    @abstractmethod
    def store(self, entry: MemoryEntry[T]) -> bool:
        """
        存储记忆
        
        Args:
            entry: 记忆条目
            
        Returns:
            存储是否成功
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: MemoryQuery[T]) -> List[MemoryEntry[T]]:
        """
        检索记忆
        
        Args:
            query: 查询条件
            
        Returns:
            匹配的记忆条目列表
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """
        获取记忆数量
        
        Returns:
            当前记忆条目数
        """
        pass
    
    def similarity(self, a: T, b: T) -> float:
        """
        计算两个数据的相似度
        
        子类可以重写此方法以实现特定的相似度计算。
        默认使用简单的哈希比较。
        
        Args:
            a: 第一个数据
            b: 第二个数据
            
        Returns:
            相似度分数 (0.0-1.0)
        """
        # 默认实现：基于哈希的简单相似度
        try:
            if a == b:
                return 1.0
            # 尝试计算向量相似度
            if hasattr(a, '__len__') and hasattr(b, '__len__'):
                return self._vector_similarity(a, b)
            return 0.0
        except Exception:
            return 0.0
    
    def _vector_similarity(self, a: Any, b: Any) -> float:
        """
        计算向量相似度（余弦相似度）
        
        Args:
            a: 第一个向量
            b: 第二个向量
            
        Returns:
            余弦相似度
        """
        try:
            # 转换为数值列表
            vec_a = [float(x) for x in a]
            vec_b = [float(x) for x in b]
            
            if len(vec_a) != len(vec_b):
                return 0.0
            
            dot = sum(x * y for x, y in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(x * x for x in vec_a))
            norm_b = math.sqrt(sum(x * x for x in vec_b))
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0


# ============================================================================
# 统一记忆库
# ============================================================================

class UnifiedMemoryBank(BaseMemory[T]):
    """
    通用记忆库
    
    支持序列数据（视频帧/文本token）的统一存储和检索。
    支持多种检索模式（相似度/可学习/混合）。
    
    Attributes:
        entries: 记忆条目列表
        capacity: 最大容量
        retrieval_mode: 检索模式
    """
    
    def __init__(self, config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化记忆库
        
        Args:
            config: 算法配置
        """
        super().__init__(config)
        self.capacity = self.config.memory_capacity
        self.retrieval_mode = self.config.memory_retrieval_mode
        self.entries: List[MemoryEntry[T]] = []
        self._timestamp_counter = 0.0
        
        # 可学习检索的参数（简化模拟）
        self._learnable_weights: Dict[str, float] = {}
    
    def store(self, entry: MemoryEntry[T]) -> bool:
        """
        存储记忆条目
        
        如果超过容量，会根据重要性移除最不重要的条目。
        
        Args:
            entry: 记忆条目
            
        Returns:
            存储是否成功
        """
        # 分配时间戳
        if entry.timestamp == 0.0:
            entry.timestamp = self._timestamp_counter
            self._timestamp_counter += 1.0
        
        # 检查容量
        if len(self.entries) >= self.capacity:
            # 移除重要性最低的条目
            self._evict_least_important()
        
        self.entries.append(entry)
        return True
    
    def store_batch(self, entries: List[MemoryEntry[T]]) -> int:
        """
        批量存储记忆
        
        Args:
            entries: 记忆条目列表
            
        Returns:
            成功存储的数量
        """
        count = 0
        for entry in entries:
            if self.store(entry):
                count += 1
        return count
    
    def retrieve(self, query: MemoryQuery[T]) -> List[MemoryEntry[T]]:
        """
        检索记忆
        
        根据配置的检索模式进行检索。
        
        Args:
            query: 查询条件
            
        Returns:
            匹配的记忆条目列表
        """
        if not self.entries:
            return []
        
        if self.retrieval_mode == MemoryRetrievalMode.SIMILARITY:
            return self._retrieve_by_similarity(query)
        elif self.retrieval_mode == MemoryRetrievalMode.LEARNABLE:
            return self._retrieve_by_learnable(query)
        else:  # HYBRID
            return self._retrieve_hybrid(query)
    
    def _retrieve_by_similarity(self, query: MemoryQuery[T]) -> List[MemoryEntry[T]]:
        """基于相似度的检索"""
        scored_entries = []
        max_timestamp = max(e.timestamp for e in self.entries) if self.entries else 1.0
        
        for entry in self.entries:
            sim = self.similarity(query.query_data, entry.data)
            
            # 应用时间衰减
            recency = entry.timestamp / max_timestamp if max_timestamp > 0 else 1.0
            combined_score = (1 - query.recency_weight) * sim + query.recency_weight * recency
            
            if combined_score >= query.threshold:
                scored_entries.append((combined_score, entry))
        
        # 按分数排序并返回 top_k
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:query.top_k]]
    
    def _retrieve_by_learnable(self, query: MemoryQuery[T]) -> List[MemoryEntry[T]]:
        """基于可学习权重的检索"""
        # 简化实现：使用带权重的相似度
        scored_entries = []
        
        for entry in self.entries:
            base_sim = self.similarity(query.query_data, entry.data)
            
            # 应用可学习权重
            weight_key = f"entry_{id(entry)}"
            weight = self._learnable_weights.get(weight_key, 1.0)
            weighted_sim = base_sim * weight
            
            if weighted_sim >= query.threshold:
                scored_entries.append((weighted_sim, entry))
        
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:query.top_k]]
    
    def _retrieve_hybrid(self, query: MemoryQuery[T]) -> List[MemoryEntry[T]]:
        """混合检索（结合相似度和可学习权重）"""
        scored_entries = []
        max_timestamp = max(e.timestamp for e in self.entries) if self.entries else 1.0
        
        for entry in self.entries:
            sim = self.similarity(query.query_data, entry.data)
            weight_key = f"entry_{id(entry)}"
            weight = self._learnable_weights.get(weight_key, 1.0)
            
            recency = entry.timestamp / max_timestamp if max_timestamp > 0 else 1.0
            
            # 混合分数
            combined_score = 0.5 * sim * weight + 0.3 * entry.importance + 0.2 * recency
            
            if combined_score >= query.threshold:
                scored_entries.append((combined_score, entry))
        
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:query.top_k]]
    
    def _evict_least_important(self) -> None:
        """移除重要性最低的记忆条目"""
        if not self.entries:
            return
        
        # 找到重要性最低的条目
        min_idx = min(range(len(self.entries)), 
                     key=lambda i: self.entries[i].importance)
        self.entries.pop(min_idx)
    
    def clear(self) -> None:
        """清空所有记忆"""
        self.entries.clear()
        self._timestamp_counter = 0.0
        self._learnable_weights.clear()
    
    def size(self) -> int:
        """获取记忆数量"""
        return len(self.entries)
    
    def update_learnable_weight(self, entry_id: str, weight: float) -> None:
        """
        更新可学习权重
        
        Args:
            entry_id: 条目标识
            weight: 新权重
        """
        self._learnable_weights[entry_id] = weight
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆库统计信息
        
        Returns:
            统计信息字典
        """
        if not self.entries:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_importance': 0.0,
            }
        
        importances = [e.importance for e in self.entries]
        return {
            'size': len(self.entries),
            'capacity': self.capacity,
            'utilization': len(self.entries) / self.capacity,
            'avg_importance': sum(importances) / len(importances),
            'max_importance': max(importances),
            'min_importance': min(importances),
        }


# ============================================================================
# 轻量级可训练记忆
# ============================================================================

class UnifiedLightweightMemory(BaseMemory[T]):
    """
    轻量级可训练记忆
    
    设计用于资源受限环境的轻量级记忆系统。
    使用固定大小的循环缓冲区，支持快速读写。
    
    Attributes:
        buffer: 循环缓冲区
        max_size: 最大大小
        trainable_params: 可训练参数（简化模拟）
    """
    
    def __init__(self, max_size: int = 100, 
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化轻量级记忆
        
        Args:
            max_size: 最大记忆数量
            config: 算法配置
        """
        super().__init__(config)
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        
        # 可训练参数（简化模拟）
        self._attention_weights: List[float] = [1.0] * max_size
        self._learning_rate = 0.01
    
    def store(self, entry: MemoryEntry[T]) -> bool:
        """
        存储记忆
        
        Args:
            entry: 记忆条目
            
        Returns:
            存储是否成功
        """
        self.buffer.append(entry)
        return True
    
    def retrieve(self, query: MemoryQuery[T]) -> List[MemoryEntry[T]]:
        """
        检索记忆
        
        Args:
            query: 查询条件
            
        Returns:
            匹配的记忆条目列表
        """
        if not self.buffer:
            return []
        
        scored_entries = []
        buffer_list = list(self.buffer)
        
        for i, entry in enumerate(buffer_list):
            sim = self.similarity(query.query_data, entry.data)
            # 应用注意力权重
            weight = self._attention_weights[i] if i < len(self._attention_weights) else 1.0
            weighted_sim = sim * weight
            
            if weighted_sim >= query.threshold:
                scored_entries.append((weighted_sim, entry))
        
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:query.top_k]]
    
    def train_step(self, query: MemoryQuery[T], target_idx: int) -> float:
        """
        训练步骤（简化模拟）
        
        根据查询和目标索引更新注意力权重。
        
        Args:
            query: 查询
            target_idx: 目标条目索引
            
        Returns:
            损失值
        """
        buffer_list = list(self.buffer)
        if target_idx >= len(buffer_list):
            return 0.0
        
        # 计算当前预测
        similarities = []
        for entry in buffer_list:
            sim = self.similarity(query.query_data, entry.data)
            similarities.append(sim)
        
        # 更新权重（简化梯度下降）
        loss = 0.0
        for i, sim in enumerate(similarities):
            if i == target_idx:
                # 增加目标权重
                target = 1.0
            else:
                # 减少非目标权重
                target = 0.0
            
            error = target - sim * self._attention_weights[i]
            self._attention_weights[i] += self._learning_rate * error * sim
            self._attention_weights[i] = max(0.1, min(2.0, self._attention_weights[i]))
            loss += error ** 2
        
        return loss / len(similarities) if similarities else 0.0
    
    def clear(self) -> None:
        """清空记忆"""
        self.buffer.clear()
        self._attention_weights = [1.0] * self.max_size
    
    def size(self) -> int:
        """获取记忆数量"""
        return len(self.buffer)
    
    def get_recent(self, n: int = 10) -> List[MemoryEntry[T]]:
        """
        获取最近的 n 个记忆
        
        Args:
            n: 数量
            
        Returns:
            最近的记忆条目列表
        """
        buffer_list = list(self.buffer)
        return buffer_list[-n:] if n < len(buffer_list) else buffer_list


# ============================================================================
# 分级记忆系统
# ============================================================================

class MemoryLevel(Enum):
    """记忆级别"""
    SHORT_TERM = auto()   # 短期记忆（最近）
    MEDIUM_TERM = auto()  # 中期记忆
    LONG_TERM = auto()    # 长期记忆（重要）


class UnifiedHierarchicalMemory(BaseMemory[T]):
    """
    分级记忆系统
    
    将记忆分为短期、中期、长期三个级别。
    支持记忆在不同级别之间的晋升和降级。
    
    Attributes:
        short_term: 短期记忆（容量小，访问快）
        medium_term: 中期记忆
        long_term: 长期记忆（容量大，访问慢）
        promotion_threshold: 晋升阈值
    """
    
    def __init__(self, 
                 short_term_size: int = 100,
                 medium_term_size: int = 1000,
                 long_term_size: int = 10000,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化分级记忆
        
        Args:
            short_term_size: 短期记忆容量
            medium_term_size: 中期记忆容量
            long_term_size: 长期记忆容量
            config: 算法配置
        """
        super().__init__(config)
        
        self.short_term = UnifiedMemoryBank(
            config.update(memory_capacity=short_term_size)
        )
        self.medium_term = UnifiedMemoryBank(
            config.update(memory_capacity=medium_term_size)
        )
        self.long_term = UnifiedMemoryBank(
            config.update(memory_capacity=long_term_size)
        )
        
        self.promotion_threshold = 0.7
        self.demotion_threshold = 0.3
        self.access_counts: Dict[int, int] = {}
    
    def store(self, entry: MemoryEntry[T]) -> bool:
        """
        存储记忆（默认存入短期记忆）
        
        Args:
            entry: 记忆条目
            
        Returns:
            存储是否成功
        """
        return self.short_term.store(entry)
    
    def retrieve(self, query: MemoryQuery[T]) -> List[MemoryEntry[T]]:
        """
        检索记忆（按级别顺序检索）
        
        先检索短期记忆，再中期，最后长期。
        
        Args:
            query: 查询条件
            
        Returns:
            合并的记忆条目列表
        """
        results = []
        
        # 按优先级检索
        for level, memory in [
            (MemoryLevel.SHORT_TERM, self.short_term),
            (MemoryLevel.MEDIUM_TERM, self.medium_term),
            (MemoryLevel.LONG_TERM, self.long_term)
        ]:
            level_results = memory.retrieve(query)
            
            # 更新访问计数
            for entry in level_results:
                entry_id = id(entry)
                self.access_counts[entry_id] = self.access_counts.get(entry_id, 0) + 1
            
            results.extend(level_results)
        
        # 去重并排序
        seen = set()
        unique_results = []
        for entry in results:
            entry_id = id(entry)
            if entry_id not in seen:
                seen.add(entry_id)
                unique_results.append(entry)
        
        return unique_results[:query.top_k]
    
    def consolidate(self) -> None:
        """
        记忆巩固
        
        根据访问频率和重要性，在不同级别之间移动记忆。
        """
        # 短期 -> 中期
        for entry in list(self.short_term.entries):
            entry_id = id(entry)
            access_count = self.access_counts.get(entry_id, 0)
            
            if entry.importance >= self.promotion_threshold or access_count >= 3:
                self.medium_term.store(entry)
                # 从短期移除
                self.short_term.entries.remove(entry)
        
        # 中期 -> 长期
        for entry in list(self.medium_term.entries):
            entry_id = id(entry)
            access_count = self.access_counts.get(entry_id, 0)
            
            if entry.importance >= 0.9 or access_count >= 10:
                self.long_term.store(entry)
                # 从中期移除
                self.medium_term.entries.remove(entry)
        
        # 降级检查
        for entry in list(self.medium_term.entries):
            entry_id = id(entry)
            if entry.importance <= self.demotion_threshold:
                self.short_term.store(entry)
                self.medium_term.entries.remove(entry)
    
    def clear(self) -> None:
        """清空所有级别的记忆"""
        self.short_term.clear()
        self.medium_term.clear()
        self.long_term.clear()
        self.access_counts.clear()
    
    def size(self) -> int:
        """获取总记忆数量"""
        return (self.short_term.size() + 
                self.medium_term.size() + 
                self.long_term.size())
    
    def get_level_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取各级别统计信息
        
        Returns:
            各级别统计信息
        """
        return {
            'short_term': self.short_term.get_stats(),
            'medium_term': self.medium_term.get_stats(),
            'long_term': self.long_term.get_stats(),
        }


# ============================================================================
# 自适应压缩器
# ============================================================================

class UnifiedAdaptiveCompressor(Generic[T]):
    """
    自适应压缩器
    
    根据数据特征和配置自动选择压缩策略。
    支持多种压缩算法和自适应压缩率调整。
    
    Attributes:
        strategy: 压缩策略
        target_ratio: 目标压缩率
        quality_threshold: 质量阈值
    """
    
    def __init__(self, config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化压缩器
        
        Args:
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.strategy = self.config.compression_strategy
        self.target_ratio = self.config.memory_compression_rate
        self.quality_threshold = 0.9
        
        # 压缩历史（用于自适应调整）
        self._compression_history: List[Tuple[float, float]] = []
    
    def compress(self, data: T, ratio: Optional[float] = None) -> T:
        """
        压缩数据
        
        Args:
            data: 原始数据
            ratio: 压缩率（可选，默认使用配置值）
            
        Returns:
            压缩后的数据
        """
        if ratio is None:
            ratio = self.target_ratio
        
        if self.strategy == CompressionStrategy.LOSSLESS:
            return self._lossless_compress(data)
        elif self.strategy == CompressionStrategy.FIXED_RATIO:
            return self._fixed_ratio_compress(data, ratio)
        else:  # ADAPTIVE
            return self._adaptive_compress(data)
    
    def _lossless_compress(self, data: T) -> T:
        """无损压缩（简化实现）"""
        # 实际实现中可以使用更复杂的算法
        # 这里仅做示例
        return data
    
    def _fixed_ratio_compress(self, data: T, ratio: float) -> T:
        """固定压缩率压缩"""
        if hasattr(data, '__len__'):
            # 对于序列数据，进行采样
            length = len(data)  # type: ignore
            new_length = max(1, int(length * ratio))
            
            if isinstance(data, (list, tuple)):
                # 均匀采样
                step = length / new_length
                indices = [int(i * step) for i in range(new_length)]
                return type(data)(data[i] for i in indices)  # type: ignore
            elif isinstance(data, str):
                # 字符串压缩（简化）
                step = length / new_length
                return ''.join(data[int(i * step)] for i in range(new_length))  # type: ignore
        
        return data
    
    def _adaptive_compress(self, data: T) -> T:
        """自适应压缩"""
        # 根据历史调整压缩率
        if self._compression_history:
            avg_quality = sum(q for _, q in self._compression_history) / len(self._compression_history)
            if avg_quality < self.quality_threshold:
                # 质量不足，降低压缩率
                adjusted_ratio = min(1.0, self.target_ratio * 1.1)
            else:
                # 质量充足，可以提高压缩率
                adjusted_ratio = max(0.1, self.target_ratio * 0.95)
        else:
            adjusted_ratio = self.target_ratio
        
        compressed = self._fixed_ratio_compress(data, adjusted_ratio)
        
        # 记录压缩质量（简化估计）
        quality = self._estimate_quality(data, compressed)
        self._compression_history.append((adjusted_ratio, quality))
        
        # 限制历史记录大小
        if len(self._compression_history) > 100:
            self._compression_history.pop(0)
        
        return compressed
    
    def _estimate_quality(self, original: T, compressed: T) -> float:
        """
        估计压缩质量
        
        Args:
            original: 原始数据
            compressed: 压缩后数据
            
        Returns:
            质量分数 (0.0-1.0)
        """
        # 简化实现：基于相似度估计
        if hasattr(original, '__len__') and hasattr(compressed, '__len__'):
            orig_len = len(original)  # type: ignore
            comp_len = len(compressed)  # type: ignore
            
            if orig_len == 0:
                return 1.0
            
            # 压缩率越低，质量估计越低
            compression_ratio = comp_len / orig_len
            return min(1.0, compression_ratio * 1.5)
        
        return 1.0
    
    def decompress(self, data: T) -> T:
        """
        解压缩数据
        
        对于某些压缩算法，可能需要解压缩。
        
        Args:
            data: 压缩数据
            
        Returns:
            解压缩后的数据
        """
        # 简化实现：某些压缩是可逆的，某些不是
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取压缩器统计信息
        
        Returns:
            统计信息
        """
        if not self._compression_history:
            return {
                'strategy': self.strategy.name,
                'target_ratio': self.target_ratio,
                'history_size': 0,
            }
        
        ratios = [r for r, _ in self._compression_history]
        qualities = [q for _, q in self._compression_history]
        
        return {
            'strategy': self.strategy.name,
            'target_ratio': self.target_ratio,
            'avg_actual_ratio': sum(ratios) / len(ratios),
            'avg_quality': sum(qualities) / len(qualities),
            'history_size': len(self._compression_history),
        }
