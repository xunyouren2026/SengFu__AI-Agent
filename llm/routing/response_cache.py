"""
语义缓存引擎 (Semantic Cache Engine)

该模块提供基于语义相似度的智能缓存功能，支持：
- 语义相似度匹配
- Embedding向量检索
- TTL过期策略
- 缓存预热

核心功能：
1. 语义缓存存储
2. 向量相似度计算
3. 缓存命中分析
4. 预热和清理策略
5. 统计和监控

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, TypeVar, Sequence
)
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import heapq

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """
    缓存策略
    """
    EXACT = auto()            # 精确匹配
    SEMANTIC = auto()         # 语义匹配
    HYBRID = auto()           # 混合模式


@dataclass
class CacheConfig:
    """
    缓存配置
    
    Attributes:
        max_size: 最大缓存条目数
        max_memory_mb: 最大内存占用 (MB)
        default_ttl_seconds: 默认TTL (秒)
        min_similarity: 最小相似度阈值
        cache_strategy: 缓存策略
        enable_embedding: 是否启用Embedding
        embedding_model: Embedding模型
        vector_dim: 向量维度
        warmup_enabled: 是否启用预热
    """
    max_size: int = 10000
    max_memory_mb: float = 512.0
    default_ttl_seconds: float = 3600.0
    min_similarity: float = 0.85
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    enable_embedding: bool = True
    embedding_model: str = "default"
    vector_dim: int = 1536
    warmup_enabled: bool = False


@dataclass
class CacheEntry:
    """
    缓存条目
    
    Attributes:
        key: 缓存键
        value: 缓存值
        created_at: 创建时间
        expires_at: 过期时间
        hit_count: 命中次数
        last_accessed: 最后访问时间
        embedding: 语义向量
        metadata: 其他元数据
    """
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def ttl_remaining(self) -> float:
        """剩余TTL (秒)"""
        if self.expires_at is None:
            return float('inf')
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, remaining)
    
    def touch(self) -> None:
        """更新访问时间"""
        self.last_accessed = datetime.now()
        self.hit_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value_preview": str(self.value)[:100] if self.value else None,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed.isoformat(),
            "ttl_remaining": self.ttl_remaining,
        }


@dataclass
class SimilarityMatch:
    """
    相似度匹配结果
    
    Attributes:
        entry: 缓存条目
        similarity: 相似度分数
        match_type: 匹配类型 (exact, semantic)
    """
    entry: CacheEntry
    similarity: float
    match_type: str = "unknown"


@dataclass
class CacheResult:
    """
    缓存结果
    
    Attributes:
        hit: 是否命中
        value: 缓存值
        match: 匹配详情
        reason: 命中/未命中原因
    """
    hit: bool
    value: Any
    match: Optional[SimilarityMatch] = None
    reason: str = ""


class EmbeddingFunction(ABC):
    """Embedding函数抽象基类"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        将文本编码为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        pass
    
    @abstractmethod
    def similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        计算两个向量的相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数 (0-1)
        """
        pass


class DefaultEmbedding(EmbeddingFunction):
    """
    默认Embedding实现
    
    使用简单的字符级TF-IDF作为后备方案。
    """
    
    def __init__(self, dim: int = 1536):
        self._dim = dim
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """简单编码：使用字符哈希"""
        vectors = []
        for text in texts:
            vec = [0.0] * self._dim
            # 简单哈希
            for i, char in enumerate(text[:self._dim]):
                vec[i] = (ord(char) % 100) / 100.0
            vectors.append(vec)
        return vectors
    
    def similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class SemanticCache:
    """
    语义缓存引擎
    
    Features:
        - 精确匹配和语义匹配
        - Embedding向量存储
        - LRU淘汰策略
        - TTL过期机制
        - 缓存预热
        - 统计监控
    
    Example:
        ```python
        # 创建缓存
        cache = SemanticCache(config=CacheConfig(
            max_size=1000,
            min_similarity=0.85
        ))
        
        # 存储
        cache.set("prompt_key", {"result": "response"})
        
        # 精确查询
        result = cache.get("prompt_key")
        
        # 语义查询
        result = cache.get_semantic("What is AI?")
        
        # 预热
        cache.warmup(preload_data)
        ```
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedding_func: Optional[EmbeddingFunction] = None
    ):
        """
        初始化语义缓存。
        
        Args:
            config: 缓存配置
            embedding_func: Embedding函数
        """
        self._config = config or CacheConfig()
        self._embedding_func = embedding_func or DefaultEmbedding(self._config.vector_dim)
        
        # 精确匹配缓存
        self._exact_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # 语义缓存
        self._semantic_cache: Dict[str, CacheEntry] = {}
        self._semantic_keys: List[str] = []  # 用于高效检索
        
        # 统计
        self._stats = {
            "hits": 0,
            "misses": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "evictions": 0,
            "expirations": 0,
            "total_items": 0,
        }
        
        self._lock = threading.RLock()
        
        # 启动后台清理
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        if self._config.warmup_enabled:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """启动后台清理线程"""
        if self._cleanup_thread is not None:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """后台清理循环"""
        while self._running:
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
            
            time.sleep(60)  # 每分钟清理一次
    
    def _generate_key(self, prompt: Any) -> str:
        """生成缓存键"""
        if isinstance(prompt, str):
            text = prompt
        else:
            text = json.dumps(prompt, sort_keys=True)
        
        # 使用MD5哈希作为键
        return hashlib.md5(text.encode()).hexdigest()
    
    def _generate_semantic_key(self, prompt: str) -> str:
        """生成语义缓存键（用于embedding索引）"""
        # 使用更长的前缀确保唯一性
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]
    
    def set(
        self,
        key: Any,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        设置缓存。
        
        Args:
            key: 缓存键 (可以是任意可序列化对象)
            value: 缓存值
            ttl: 过期时间 (秒)
            metadata: 元数据
            
        Returns:
            缓存键
        """
        with self._lock:
            # 生成键
            cache_key = self._generate_key(key)
            
            # 计算过期时间
            expires_at = None
            if ttl is None:
                ttl = self._config.default_ttl_seconds
            if ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # 创建缓存条目
            entry = CacheEntry(
                key=cache_key,
                value=value,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            # 生成embedding
            if self._config.enable_embedding and isinstance(key, str):
                entry.embedding = self._embedding_func.encode([key])[0]
            
            # 存储到精确缓存
            if cache_key in self._exact_cache:
                self._exact_cache.move_to_end(cache_key)
            self._exact_cache[cache_key] = entry
            
            # 如果是语义缓存模式，也添加到语义索引
            if self._config.cache_strategy in [CacheStrategy.SEMANTIC, CacheStrategy.HYBRID]:
                semantic_key = self._generate_semantic_key(str(key))
                self._semantic_cache[semantic_key] = entry
                self._semantic_keys.append(semantic_key)
            
            # 检查容量
            self._evict_if_needed()
            
            # 更新统计
            self._stats["total_items"] = len(self._exact_cache)
            
            return cache_key
    
    def get(
        self,
        key: Any,
        update_access: bool = True
    ) -> CacheResult:
        """
        获取缓存（精确匹配）。
        
        Args:
            key: 缓存键
            update_access: 是否更新访问时间
            
        Returns:
            缓存结果
        """
        with self._lock:
            cache_key = self._generate_key(key)
            
            # 查找精确匹配
            if cache_key in self._exact_cache:
                entry = self._exact_cache[cache_key]
                
                # 检查是否过期
                if entry.is_expired:
                    self._remove_entry(cache_key)
                    self._stats["misses"] += 1
                    self._stats["expirations"] += 1
                    return CacheResult(
                        hit=False,
                        value=None,
                        reason="Expired"
                    )
                
                # 命中
                if update_access:
                    entry.touch()
                    self._exact_cache.move_to_end(cache_key)
                
                self._stats["hits"] += 1
                self._stats["exact_hits"] += 1
                
                return CacheResult(
                    hit=True,
                    value=entry.value,
                    match=SimilarityMatch(
                        entry=entry,
                        similarity=1.0,
                        match_type="exact"
                    ),
                    reason="Exact match"
                )
            
            # 未命中
            self._stats["misses"] += 1
            return CacheResult(
                hit=False,
                value=None,
                reason="Not found"
            )
    
    def get_semantic(
        self,
        prompt: str,
        min_similarity: Optional[float] = None
    ) -> CacheResult:
        """
        获取缓存（语义匹配）。
        
        Args:
            prompt: 查询文本
            min_similarity: 最小相似度
            
        Returns:
            缓存结果
        """
        if not self._config.enable_embedding:
            return CacheResult(
                hit=False,
                value=None,
                reason="Semantic cache disabled"
            )
        
        min_sim = min_similarity or self._config.min_similarity
        
        with self._lock:
            # 编码查询
            query_embedding = self._embedding_func.encode([prompt])[0]
            
            # 遍历查找最相似的
            best_match: Optional[Tuple[str, float, CacheEntry]] = None
            
            for cache_key, entry in self._exact_cache.items():
                # 跳过过期条目
                if entry.is_expired:
                    continue
                
                if entry.embedding is None:
                    continue
                
                # 计算相似度
                similarity = self._embedding_func.similarity(
                    query_embedding,
                    entry.embedding
                )
                
                if similarity >= min_sim:
                    if best_match is None or similarity > best_match[1]:
                        best_match = (cache_key, similarity, entry)
            
            if best_match:
                cache_key, similarity, entry = best_match
                
                # 更新访问
                entry.touch()
                self._exact_cache.move_to_end(cache_key)
                
                self._stats["hits"] += 1
                self._stats["semantic_hits"] += 1
                
                return CacheResult(
                    hit=True,
                    value=entry.value,
                    match=SimilarityMatch(
                        entry=entry,
                        similarity=similarity,
                        match_type="semantic"
                    ),
                    reason=f"Semantic match (similarity: {similarity:.3f})"
                )
            
            # 未命中
            self._stats["misses"] += 1
            return CacheResult(
                hit=False,
                value=None,
                reason="No semantic match found"
            )
    
    def _remove_entry(self, cache_key: str) -> None:
        """移除缓存条目"""
        if cache_key in self._exact_cache:
            entry = self._exact_cache.pop(cache_key)
            
            # 从语义缓存中移除
            if entry.embedding:
                semantic_key = self._generate_semantic_key(entry.key)
                if semantic_key in self._semantic_cache:
                    del self._semantic_cache[semantic_key]
                if semantic_key in self._semantic_keys:
                    self._semantic_keys.remove(semantic_key)
    
    def _evict_if_needed(self) -> None:
        """必要时淘汰"""
        while len(self._exact_cache) >= self._config.max_size:
            # LRU淘汰：移除最老的条目
            oldest_key = next(iter(self._exact_cache))
            self._remove_entry(oldest_key)
            self._stats["evictions"] += 1
    
    def _cleanup_expired(self) -> None:
        """清理过期条目"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._exact_cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats["expirations"] += 1
    
    def delete(self, key: Any) -> bool:
        """
        删除缓存条目。
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        with self._lock:
            cache_key = self._generate_key(key)
            if cache_key in self._exact_cache:
                self._remove_entry(cache_key)
                return True
            return False
    
    def clear(self) -> int:
        """
        清空缓存。
        
        Returns:
            清空的条目数
        """
        with self._lock:
            count = len(self._exact_cache)
            self._exact_cache.clear()
            self._semantic_cache.clear()
            self._semantic_keys.clear()
            return count
    
    def warmup(self, data: List[Tuple[Any, Any]]) -> int:
        """
        预热缓存。
        
        Args:
            data: 预热数据 [(key, value), ...]
            
        Returns:
            预热的条目数
        """
        count = 0
        for key, value in data:
            try:
                self.set(key, value)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to warmup entry: {e}")
        
        logger.info(f"Warmed up {count} cache entries")
        return count
    
    def invalidate(self, pattern: str) -> int:
        """
        使匹配的缓存失效。
        
        Args:
            pattern: 匹配模式
            
        Returns:
            失效的条目数
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._exact_cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total 
                if total > 0 else 0.0
            )
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "exact_hits": self._stats["exact_hits"],
                "semantic_hits": self._stats["semantic_hits"],
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
                "total_items": len(self._exact_cache),
                "max_size": self._config.max_size,
                "memory_usage_estimate_mb": self._estimate_memory_usage(),
            }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用"""
        # 粗略估算
        import sys
        total_size = 0
        
        for entry in self._exact_cache.values():
            total_size += sys.getsizeof(entry)
            if entry.value:
                total_size += sys.getsizeof(str(entry.value))
            if entry.embedding:
                total_size += sys.getsizeof(entry.embedding)
        
        return total_size / (1024 * 1024)  # MB
    
    def get_all_entries(self) -> List[CacheEntry]:
        """获取所有缓存条目"""
        with self._lock:
            return list(self._exact_cache.values())
    
    def get_recent_entries(self, limit: int = 10) -> List[CacheEntry]:
        """获取最近访问的条目"""
        with self._lock:
            entries = sorted(
                self._exact_cache.values(),
                key=lambda e: e.last_accessed,
                reverse=True
            )
            return entries[:limit]
    
    def get_hot_entries(self, limit: int = 10) -> List[CacheEntry]:
        """获取最热门的条目"""
        with self._lock:
            entries = sorted(
                self._exact_cache.values(),
                key=lambda e: e.hit_count,
                reverse=True
            )
            return entries[:limit]
    
    def export_cache(self) -> Dict[str, Any]:
        """导出缓存数据"""
        with self._lock:
            entries_data = []
            
            for entry in self._exact_cache.values():
                if entry.is_expired:
                    continue
                
                entries_data.append({
                    "key": entry.key,
                    "value": str(entry.value)[:500],  # 限制大小
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                    "hit_count": entry.hit_count,
                })
            
            return {
                "entries": entries_data,
                "stats": self.get_stats(),
                "export_time": datetime.now().isoformat(),
            }
    
    def import_cache(self, data: Dict[str, Any]) -> int:
        """导入缓存数据"""
        imported = 0
        entries = data.get("entries", [])
        
        for entry_data in entries:
            try:
                entry = CacheEntry(
                    key=entry_data["key"],
                    value=entry_data["value"],
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    expires_at=(
                        datetime.fromisoformat(entry_data["expires_at"])
                        if entry_data.get("expires_at") else None
                    ),
                    hit_count=entry_data.get("hit_count", 0),
                )
                
                self._exact_cache[entry.key] = entry
                imported += 1
                
            except Exception as e:
                logger.warning(f"Failed to import entry: {e}")
        
        return imported
    
    def reset_stats(self) -> None:
        """重置统计"""
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "exact_hits": 0,
                "semantic_hits": 0,
                "evictions": 0,
                "expirations": 0,
                "total_items": len(self._exact_cache),
            }
    
    def __del__(self):
        """析构时停止清理线程"""
        self._running = False
