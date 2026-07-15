"""
分层记忆系统 - Hierarchical Memory System

实现三层记忆架构：
1. 短期记忆（Working Memory）- 当前任务相关，容量小，访问快
2. 情景记忆（Episodic Memory）- 经历序列，中等容量
3. 长期记忆（Long-term Memory）- 压缩存储，大容量，检索慢

支持：
- 记忆编码/存储/检索
- 遗忘机制
- 记忆巩固
- 向量化检索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import deque
import time
import hashlib


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    # 短期记忆
    working_capacity: int = 100
    working_feature_dim: int = 256
    
    # 情景记忆
    episodic_capacity: int = 10000
    episodic_feature_dim: int = 512
    
    # 长期记忆
    longterm_capacity: int = 1000000
    longterm_feature_dim: int = 1024
    longterm_compressed_dim: int = 128
    
    # 检索
    retrieval_top_k: int = 10
    retrieval_threshold: float = 0.7
    
    # 遗忘
    forget_rate: float = 0.01
    importance_decay: float = 0.99
    
    # 巩固
    consolidation_interval: int = 1000
    consolidation_batch: int = 100
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MemoryItem:
    """记忆项"""
    key: torch.Tensor  # 键向量
    value: torch.Tensor  # 值向量
    timestamp: float  # 时间戳
    importance: float  # 重要性
    access_count: int  # 访问次数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def update_importance(self, decay: float) -> None:
        """衰减重要性"""
        self.importance *= decay


class WorkingMemory:
    """
    短期工作记忆
    
    容量小，访问快，存储当前任务相关信息。
    使用LRU策略管理。
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.capacity = config.working_capacity
        self.feature_dim = config.working_feature_dim
        self.device = config.device
        
        # 存储
        self.memory: Dict[str, MemoryItem] = {}
        self.access_order: deque = deque()  # LRU顺序
        
        # 统计
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def _make_key(self, key_vector: torch.Tensor) -> str:
        """生成存储键"""
        return hashlib.md5(key_vector.cpu().numpy().tobytes()).hexdigest()
    
    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        存储记忆
        
        Returns:
            被淘汰的键（如果有）
        """
        key = key.to(self.device)
        value = value.to(self.device)
        
        storage_key = self._make_key(key)
        
        # 如果已存在，更新
        if storage_key in self.memory:
            self.memory[storage_key].value = value
            self.memory[storage_key].importance = importance
            self.memory[storage_key].access_count += 1
            self.memory[storage_key].timestamp = time.time()
            
            # 更新LRU顺序
            self.access_order.remove(storage_key)
            self.access_order.append(storage_key)
            return None
        
        # 检查容量
        evicted_key = None
        if len(self.memory) >= self.capacity:
            # LRU淘汰
            evicted_key = self.access_order.popleft()
            del self.memory[evicted_key]
            self.stats['evictions'] += 1
        
        # 存储
        self.memory[storage_key] = MemoryItem(
            key=key,
            value=value,
            timestamp=time.time(),
            importance=importance,
            access_count=1,
            metadata=metadata or {}
        )
        self.access_order.append(storage_key)
        
        return evicted_key
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """
        检索记忆
        
        Returns:
            (记忆项, 相似度) 列表
        """
        query = query.to(self.device)
        
        if not self.memory:
            self.stats['misses'] += 1
            return []
        
        # 计算相似度
        results = []
        for storage_key, item in self.memory.items():
            similarity = F.cosine_similarity(
                query.unsqueeze(0),
                item.key.unsqueeze(0)
            ).item()
            results.append((item, similarity, storage_key))
        
        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 更新访问
        for item, sim, storage_key in results[:top_k]:
            item.access_count += 1
            self.access_order.remove(storage_key)
            self.access_order.append(storage_key)
        
        self.stats['hits'] += min(top_k, len(results))
        
        return [(item, sim) for item, sim, _ in results[:top_k]]
    
    def clear(self) -> None:
        """清空记忆"""
        self.memory.clear()
        self.access_order.clear()
    
    def get_size(self) -> int:
        return len(self.memory)


class EpisodicMemory:
    """
    情景记忆
    
    存储经历序列，支持时序检索。
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.capacity = config.episodic_capacity
        self.feature_dim = config.episodic_feature_dim
        self.device = config.device
        
        # 存储（按时间顺序）
        self.episodes: List[MemoryItem] = []
        
        # 索引（用于快速检索）
        self.key_matrix: Optional[torch.Tensor] = None
        self._index_dirty = True
        
        # 统计
        self.stats = {'stored': 0, 'retrieved': 0}
    
    def _rebuild_index(self) -> None:
        """重建检索索引"""
        if not self.episodes:
            self.key_matrix = None
            return
        
        keys = torch.stack([e.key for e in self.episodes])
        self.key_matrix = F.normalize(keys, dim=-1)
        self._index_dirty = False
    
    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> None:
        """存储情景"""
        key = key.to(self.device)
        value = value.to(self.device)
        
        # 检查容量
        if len(self.episodes) >= self.capacity:
            # 移除最旧且最不重要的
            min_idx = min(
                range(len(self.episodes)),
                key=lambda i: self.episodes[i].importance * (1 + self.episodes[i].access_count)
            )
            self.episodes.pop(min_idx)
        
        self.episodes.append(MemoryItem(
            key=key,
            value=value,
            timestamp=time.time(),
            importance=importance,
            access_count=0,
            metadata=metadata or {}
        ))
        
        self._index_dirty = True
        self.stats['stored'] += 1
    
    def store_sequence(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importances: Optional[torch.Tensor] = None
    ) -> None:
        """存储序列"""
        batch_size = keys.shape[0]
        
        for i in range(batch_size):
            imp = importances[i].item() if importances is not None else 1.0
            self.store(keys[i], values[i], imp)
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 10,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """
        检索情景
        
        Args:
            query: 查询向量
            top_k: 返回数量
            time_range: 时间范围过滤
        """
        query = query.to(self.device)
        query = F.normalize(query.unsqueeze(0), dim=-1).squeeze(0)
        
        if not self.episodes:
            return []
        
        # 重建索引（如果需要）
        if self._index_dirty:
            self._rebuild_index()
        
        if self.key_matrix is None:
            return []
        
        # 计算相似度
        similarities = torch.matmul(self.key_matrix, query)
        
        # 时间过滤
        if time_range:
            t_min, t_max = time_range
            mask = torch.tensor([
                t_min <= e.timestamp <= t_max
                for e in self.episodes
            ], device=self.device)
            similarities = similarities * mask - (1 - mask) * 1e10
        
        # 获取top-k
        top_indices = torch.topk(similarities, min(top_k, len(self.episodes))).indices
        
        results = []
        for idx in top_indices:
            item = self.episodes[idx.item()]
            item.access_count += 1
            results.append((item, similarities[idx].item()))
        
        self.stats['retrieved'] += len(results)
        return results
    
    def retrieve_recent(self, n: int = 10) -> List[MemoryItem]:
        """检索最近的n个情景"""
        return self.episodes[-n:]
    
    def forget(self, threshold: float = 0.1) -> int:
        """
        遗忘不重要的记忆
        
        Returns:
            遗忘的数量
        """
        before = len(self.episodes)
        
        self.episodes = [
            e for e in self.episodes
            if e.importance >= threshold or e.access_count > 0
        ]
        
        self._index_dirty = True
        return before - len(self.episodes)
    
    def decay_importance(self, decay: float) -> None:
        """衰减所有重要性"""
        for e in self.episodes:
            e.update_importance(decay)


class LongTermMemory:
    """
    长期记忆
    
    大容量压缩存储，支持向量检索。
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.capacity = config.longterm_capacity
        self.feature_dim = config.longterm_feature_dim
        self.compressed_dim = config.longterm_compressed_dim
        self.device = config.device
        
        # 压缩编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.compressed_dim)
        ).to(self.device)
        
        # 存储
        self.compressed_keys: Optional[torch.Tensor] = None
        self.values: List[torch.Tensor] = []
        self.metadata: List[Dict] = []
        
        # 重要性
        self.importances: Optional[torch.Tensor] = None
        
        # 统计
        self.stats = {'stored': 0, 'compressed': 0}
    
    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> None:
        """存储到长期记忆"""
        key = key.to(self.device)
        value = value.to(self.device)
        
        # 压缩
        with torch.no_grad():
            compressed_key = self.encoder(key.unsqueeze(0)).squeeze(0)
        
        # 添加到存储
        if self.compressed_keys is None:
            self.compressed_keys = compressed_key.unsqueeze(0)
            self.importances = torch.tensor([importance], device=self.device)
        else:
            self.compressed_keys = torch.cat([
                self.compressed_keys,
                compressed_key.unsqueeze(0)
            ])
            self.importances = torch.cat([
                self.importances,
                torch.tensor([importance], device=self.device)
            ])
        
        self.values.append(value)
        self.metadata.append(metadata or {})
        
        # 检查容量
        if len(self.values) > self.capacity:
            # 移除最不重要的
            min_idx = torch.argmin(self.importances).item()
            
            mask = torch.ones(len(self.values), dtype=torch.bool, device=self.device)
            mask[min_idx] = False
            
            self.compressed_keys = self.compressed_keys[mask]
            self.importances = self.importances[mask]
            
            self.values.pop(min_idx)
            self.metadata.pop(min_idx)
        
        self.stats['stored'] += 1
        self.stats['compressed'] += 1
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 10
    ) -> List[Tuple[torch.Tensor, float, Dict]]:
        """
        检索长期记忆
        
        Returns:
            (值, 相似度, 元数据) 列表
        """
        query = query.to(self.device)
        
        if self.compressed_keys is None or len(self.values) == 0:
            return []
        
        # 压缩查询
        with torch.no_grad():
            compressed_query = self.encoder(query.unsqueeze(0)).squeeze(0)
        
        # 计算相似度
        similarities = F.cosine_similarity(
            compressed_query.unsqueeze(0),
            self.compressed_keys,
            dim=-1
        )
        
        # 加权重要性
        weighted_sim = similarities * self.importances
        
        # 获取top-k
        top_indices = torch.topk(weighted_sim, min(top_k, len(self.values))).indices
        
        results = []
        for idx in top_indices:
            results.append((
                self.values[idx.item()],
                similarities[idx].item(),
                self.metadata[idx.item()]
            ))
        
        return results
    
    def consolidate(
        self,
        episodic_memory: EpisodicMemory,
        batch_size: int = 100
    ) -> int:
        """
        从情景记忆巩固到长期记忆
        
        选择重要且频繁访问的情景。
        """
        candidates = [
            e for e in episodic_memory.episodes
            if e.importance > 0.5 or e.access_count > 3
        ]
        
        # 按重要性排序
        candidates.sort(key=lambda e: e.importance * (1 + e.access_count), reverse=True)
        
        consolidated = 0
        for item in candidates[:batch_size]:
            self.store(
                item.key,
                item.value,
                item.importance,
                item.metadata
            )
            consolidated += 1
        
        return consolidated


class HierarchicalMemory:
    """
    分层记忆系统
    
    整合三层记忆，提供统一接口。
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        
        # 三层记忆
        self.working = WorkingMemory(config)
        self.episodic = EpisodicMemory(config)
        self.longterm = LongTermMemory(config)
        
        # 巩固计数器
        self.step_count = 0
    
    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: float = 1.0,
        metadata: Optional[Dict] = None,
        to_working: bool = True,
        to_episodic: bool = True
    ) -> None:
        """
        存储记忆
        
        Args:
            key: 键向量
            value: 值向量
            importance: 重要性
            metadata: 元数据
            to_working: 是否存到工作记忆
            to_episodic: 是否存到情景记忆
        """
        if to_working:
            self.working.store(key, value, importance, metadata)
        
        if to_episodic:
            self.episodic.store(key, value, importance, metadata)
        
        self.step_count += 1
        
        # 定期巩固
        if self.step_count % self.config.consolidation_interval == 0:
            self.consolidate()
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 10,
        sources: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[Any, float]]]:
        """
        从所有层检索
        
        Args:
            query: 查询向量
            top_k: 每层返回数量
            sources: 要检索的层（None表示全部）
        """
        sources = sources or ['working', 'episodic', 'longterm']
        results = {}
        
        if 'working' in sources:
            results['working'] = self.working.retrieve(query, top_k)
        
        if 'episodic' in sources:
            results['episodic'] = self.episodic.retrieve(query, top_k)
        
        if 'longterm' in sources:
            results['longterm'] = self.longterm.retrieve(query, top_k)
        
        return results
    
    def retrieve_best(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[Any, float, str]]:
        """
        从所有层检索最佳结果
        
        Returns:
            (值, 相似度, 来源) 列表
        """
        all_results = self.retrieve(query, top_k * 2)
        
        combined = []
        for source, items in all_results.items():
            for item, sim in items:
                if hasattr(item, 'value'):
                    combined.append((item.value, sim, source))
                else:
                    combined.append((item[0], sim, source))
        
        # 按相似度排序
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined[:top_k]
    
    def consolidate(self) -> Dict[str, int]:
        """
        记忆巩固
        
        将重要的情景记忆转移到长期记忆。
        """
        # 情景 -> 长期
        to_longterm = self.longterm.consolidate(
            self.episodic,
            self.config.consolidation_batch
        )
        
        # 遗忘不重要的情景
        forgotten = self.episodic.forget(self.config.forget_rate)
        
        # 衰减重要性
        self.episodic.decay_importance(self.config.importance_decay)
        
        return {
            'to_longterm': to_longterm,
            'forgotten': forgotten
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'working_size': self.working.get_size(),
            'episodic_size': len(self.episodic.episodes),
            'longterm_size': len(self.longterm.values),
            'working_stats': self.working.stats,
            'episodic_stats': self.episodic.stats,
            'longterm_stats': self.longterm.stats
        }
    
    def clear_all(self) -> None:
        """清空所有记忆"""
        self.working.clear()
        self.episodic.episodes.clear()
        self.longterm.values.clear()
        self.longterm.compressed_keys = None
        self.longterm.metadata.clear()
