"""
Response Cache - 响应缓存系统
智能缓存LLM响应，支持语义匹配和缓存淘汰
"""

import re
import time
import logging
import threading
import hashlib
import json
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import math

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"           # 最近最少使用
    LFU = "lfu"           # 最少使用频率
    TTL = "ttl"           # 时间过期
    SEMANTIC = "semantic" # 语义匹配
    HYBRID = "hybrid"     # 混合策略


class CacheStatus(Enum):
    """缓存状态"""
    HIT = "hit"           # 命中
    MISS = "miss"         # 未命中
    EXPIRED = "expired"   # 过期
    EVICTED = "evicted"   # 淘汰
    SEMANTIC_HIT = "semantic_hit"  # 语义命中


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    query: str
    response: str
    model: str
    tokens_used: int
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # 生存时间（秒）
    
    # 语义信息
    query_embedding: Optional[List[float]] = None
    keywords: Set[str] = field(default_factory=set)
    
    # 统计信息
    hit_count: int = 0
    miss_count: int = 0
    
    # 额外数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """获取条目年龄"""
        return time.time() - self.created_at
    
    @property
    def hit_rate(self) -> float:
        """获取命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def access(self):
        """记录访问"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "query": self.query[:100],
            "model": self.model,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "hit_rate": self.hit_rate,
            "age": self.age
        }


@dataclass
class CacheConfig:
    """缓存配置"""
    # 容量配置
    max_entries: int = 10000
    max_memory_mb: int = 500
    
    # TTL配置
    default_ttl: Optional[float] = 3600.0  # 默认1小时
    max_ttl: float = 86400.0  # 最大24小时
    
    # 语义缓存配置
    enable_semantic_cache: bool = True
    semantic_similarity_threshold: float = 0.95
    max_semantic_candidates: int = 10
    
    # 淘汰配置
    eviction_strategy: CacheStrategy = CacheStrategy.HYBRID
    eviction_batch_size: int = 100
    
    # 性能配置
    enable_compression: bool = True
    enable_persistence: bool = False
    persistence_path: Optional[str] = None
    
    # 统计配置
    track_statistics: bool = True


class CacheKeyGenerator:
    """缓存键生成器"""
    
    def __init__(self):
        self._normalizers = self._init_normalizers()
    
    def _init_normalizers(self) -> List[Callable[[str], str]]:
        """初始化标准化器"""
        return [
            lambda x: x.lower().strip(),
            lambda x: re.sub(r'\s+', ' ', x),
            lambda x: re.sub(r'[^\w\s\u4e00-\u9fff?！？。，,.!?]', '', x),
        ]
    
    def generate(
        self, 
        query: str, 
        model: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成缓存键"""
        # 标准化查询
        normalized = query
        for normalizer in self._normalizers:
            normalized = normalizer(normalized)
        
        # 构建键字符串
        key_parts = [normalized, model]
        
        if params:
            # 只包含影响输出的参数
            relevant_params = {
                k: v for k, v in params.items()
                if k in ['temperature', 'max_tokens', 'top_p', 'system_prompt']
            }
            if relevant_params:
                key_parts.append(json.dumps(relevant_params, sort_keys=True))
        
        key_string = '|'.join(key_parts)
        
        # 生成哈希
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def generate_semantic_key(self, query: str) -> str:
        """生成语义缓存键（用于索引）"""
        # 提取关键词
        keywords = self._extract_keywords(query)
        return '|'.join(sorted(keywords)[:10])
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取关键词"""
        # 简单分词
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        # 过滤短词
        return {w for w in words if len(w) > 1}


class SemanticCacheIndex:
    """语义缓存索引"""
    
    def __init__(
        self, 
        similarity_threshold: float = 0.95,
        embedding_model: Optional[Any] = None
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        
        # 向量索引
        self._embeddings: Dict[str, List[float]] = {}
        self._keys: List[str] = []
        
        self._lock = threading.Lock()
    
    def add(self, key: str, query: str, embedding: Optional[List[float]] = None):
        """添加条目"""
        with self._lock:
            if embedding:
                self._embeddings[key] = embedding
            elif self.embedding_model:
                try:
                    self._embeddings[key] = self.embedding_model.encode(query)
                except Exception as e:
                    logger.warning(f"嵌入生成失败: {e}")
                    return
            
            if key not in self._keys:
                self._keys.append(key)
    
    def remove(self, key: str):
        """移除条目"""
        with self._lock:
            self._embeddings.pop(key, None)
            if key in self._keys:
                self._keys.remove(key)
    
    def find_similar(
        self, 
        query: str, 
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        查找相似条目
        
        Returns:
            [(key, similarity), ...]
        """
        if not self._embeddings:
            return []
        
        # 获取查询嵌入
        if not query_embedding:
            if self.embedding_model:
                try:
                    query_embedding = self.embedding_model.encode(query)
                except Exception:
                    return []
            else:
                return []
        
        # 计算相似度
        similarities = []
        with self._lock:
            for key, embedding in self._embeddings.items():
                sim = self._cosine_similarity(query_embedding, embedding)
                if sim >= self.similarity_threshold:
                    similarities.append((key, sim))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


class EvictionManager:
    """淘汰管理器"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.LRU):
        self.strategy = strategy
        
        # LRU队列
        self._lru_queue = OrderedDict()
        
        # LFU计数
        self._lfu_counts: Dict[str, int] = {}
        
        self._lock = threading.Lock()
    
    def record_access(self, key: str):
        """记录访问"""
        with self._lock:
            # LRU更新
            if key in self._lru_queue:
                self._lru_queue.move_to_end(key)
            else:
                self._lru_queue[key] = True
            
            # LFU更新
            self._lfu_counts[key] = self._lfu_counts.get(key, 0) + 1
    
    def get_eviction_candidates(
        self, 
        entries: Dict[str, CacheEntry],
        count: int
    ) -> List[str]:
        """获取淘汰候选"""
        with self._lock:
            if self.strategy == CacheStrategy.LRU:
                return self._get_lru_candidates(count)
            elif self.strategy == CacheStrategy.LFU:
                return self._get_lfu_candidates(count)
            elif self.strategy == CacheStrategy.TTL:
                return self._get_ttl_candidates(entries, count)
            else:
                return self._get_hybrid_candidates(entries, count)
    
    def _get_lru_candidates(self, count: int) -> List[str]:
        """获取LRU候选"""
        candidates = []
        for key in self._lru_queue:
            candidates.append(key)
            if len(candidates) >= count:
                break
        return candidates
    
    def _get_lfu_candidates(self, count: int) -> List[str]:
        """获取LFU候选"""
        if not self._lfu_counts:
            return []
        
        sorted_items = sorted(
            self._lfu_counts.items(),
            key=lambda x: x[1]
        )
        return [key for key, _ in sorted_items[:count]]
    
    def _get_ttl_candidates(
        self, 
        entries: Dict[str, CacheEntry],
        count: int
    ) -> List[str]:
        """获取TTL候选"""
        expired = [
            key for key, entry in entries.items()
            if entry.is_expired
        ]
        return expired[:count]
    
    def _get_hybrid_candidates(
        self, 
        entries: Dict[str, CacheEntry],
        count: int
    ) -> List[str]:
        """获取混合策略候选"""
        scores = {}
        
        for key, entry in entries.items():
            # 综合评分：访问频率 + 最近访问时间 + 年龄
            freq_score = entry.access_count / (entry.age / 3600 + 1)
            recency_score = 1.0 / (time.time() - entry.last_accessed + 1)
            age_penalty = entry.age / 86400  # 天为单位
            
            scores[key] = freq_score * 0.4 + recency_score * 0.4 - age_penalty * 0.2
        
        # 选择评分最低的
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        return [key for key, _ in sorted_items[:count]]
    
    def remove_key(self, key: str):
        """移除键"""
        with self._lock:
            self._lru_queue.pop(key, None)
            self._lfu_counts.pop(key, None)


class ResponseCache:
    """响应缓存主类"""
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedding_model: Optional[Any] = None
    ):
        self.config = config or CacheConfig()
        self.embedding_model = embedding_model
        
        # 组件
        self.key_generator = CacheKeyGenerator()
        self.semantic_index = SemanticCacheIndex(
            self.config.semantic_similarity_threshold,
            embedding_model
        ) if self.config.enable_semantic_cache else None
        self.eviction_manager = EvictionManager(self.config.eviction_strategy)
        
        # 存储
        self._cache: Dict[str, CacheEntry] = {}
        self._memory_usage = 0
        
        # 统计
        self._stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "evictions": 0,
            "total_tokens_saved": 0
        }
        
        self._lock = threading.Lock()
    
    def get(
        self, 
        query: str, 
        model: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], CacheStatus]:
        """
        获取缓存响应
        
        Returns:
            (response, status)
        """
        # 生成键
        key = self.key_generator.generate(query, model, params)
        
        with self._lock:
            # 精确匹配
            if key in self._cache:
                entry = self._cache[key]
                
                # 检查过期
                if entry.is_expired:
                    self._remove_entry(key)
                    self._stats["misses"] += 1
                    return None, CacheStatus.EXPIRED
                
                # 命中
                entry.access()
                entry.hit_count += 1
                self.eviction_manager.record_access(key)
                
                self._stats["hits"] += 1
                self._stats["total_tokens_saved"] += entry.tokens_used
                
                return entry.response, CacheStatus.HIT
        
        # 语义匹配
        if self.config.enable_semantic_cache and self.semantic_index:
            similar = self.semantic_index.find_similar(query)
            
            if similar:
                best_key, similarity = similar[0]
                
                with self._lock:
                    if best_key in self._cache:
                        entry = self._cache[best_key]
                        
                        if not entry.is_expired:
                            entry.access()
                            entry.hit_count += 1
                            
                            self._stats["semantic_hits"] += 1
                            self._stats["total_tokens_saved"] += entry.tokens_used
                            
                            return entry.response, CacheStatus.SEMANTIC_HIT
        
        # 未命中
        with self._lock:
            self._stats["misses"] += 1
        
        return None, CacheStatus.MISS
    
    def set(
        self,
        query: str,
        response: str,
        model: str,
        tokens_used: int,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        设置缓存
        
        Returns:
            cache_key
        """
        # 生成键
        key = self.key_generator.generate(query, model, params)
        
        # 计算TTL
        if ttl is None:
            ttl = self.config.default_ttl
        ttl = min(ttl, self.config.max_ttl) if ttl else None
        
        # 提取关键词
        keywords = self.key_generator._extract_keywords(query)
        
        # 创建条目
        entry = CacheEntry(
            key=key,
            query=query,
            response=response,
            model=model,
            tokens_used=tokens_used,
            ttl=ttl,
            keywords=keywords,
            metadata=metadata or {}
        )
        
        # 估算内存使用
        entry_size = len(query) + len(response) + 500  # 粗略估算
        
        with self._lock:
            # 检查容量
            self._ensure_capacity(entry_size)
            
            # 存储
            self._cache[key] = entry
            self._memory_usage += entry_size
            
            # 更新淘汰管理器
            self.eviction_manager.record_access(key)
        
        # 更新语义索引
        if self.semantic_index:
            self.semantic_index.add(key, query)
        
        return key
    
    def _ensure_capacity(self, required_size: int):
        """确保有足够容量"""
        # 检查条目数量
        while len(self._cache) >= self.config.max_entries:
            self._evict(self.config.eviction_batch_size)
        
        # 检查内存使用
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        while self._memory_usage + required_size > max_memory_bytes:
            self._evict(self.config.eviction_batch_size)
    
    def _evict(self, count: int):
        """执行淘汰"""
        candidates = self.eviction_manager.get_eviction_candidates(
            self._cache, count
        )
        
        for key in candidates:
            self._remove_entry(key)
            self._stats["evictions"] += 1
    
    def _remove_entry(self, key: str):
        """移除条目"""
        if key in self._cache:
            entry = self._cache[key]
            entry_size = len(entry.query) + len(entry.response) + 500
            self._memory_usage -= entry_size
            del self._cache[key]
            
            # 更新索引
            self.eviction_manager.remove_key(key)
            if self.semantic_index:
                self.semantic_index.remove(key)
    
    def invalidate(
        self, 
        query: Optional[str] = None,
        model: Optional[str] = None,
        key: Optional[str] = None
    ):
        """使缓存失效"""
        with self._lock:
            if key:
                self._remove_entry(key)
            elif query and model:
                cache_key = self.key_generator.generate(query, model)
                self._remove_entry(cache_key)
            else:
                # 清空所有
                self._cache.clear()
                self._memory_usage = 0
                self.eviction_manager = EvictionManager(self.config.eviction_strategy)
                if self.semantic_index:
                    self.semantic_index = SemanticCacheIndex(
                        self.config.semantic_similarity_threshold,
                        self.embedding_model
                    )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self._stats["hits"] + self._stats["misses"] + self._stats["semantic_hits"]
        hit_rate = (
            (self._stats["hits"] + self._stats["semantic_hits"]) / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            "total_entries": len(self._cache),
            "memory_usage_mb": self._memory_usage / (1024 * 1024),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "semantic_hits": self._stats["semantic_hits"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
            "total_tokens_saved": self._stats["total_tokens_saved"],
            "config": {
                "max_entries": self.config.max_entries,
                "max_memory_mb": self.config.max_memory_mb,
                "strategy": self.config.eviction_strategy.value,
                "semantic_cache_enabled": self.config.enable_semantic_cache
            }
        }
    
    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        return self._cache.get(key)
    
    def get_entries_by_model(self, model: str) -> List[CacheEntry]:
        """按模型获取条目"""
        return [e for e in self._cache.values() if e.model == model]
    
    def get_top_entries(self, by: str = "hit_count", limit: int = 10) -> List[CacheEntry]:
        """获取热门条目"""
        entries = list(self._cache.values())
        
        if by == "hit_count":
            entries.sort(key=lambda e: e.hit_count, reverse=True)
        elif by == "access_count":
            entries.sort(key=lambda e: e.access_count, reverse=True)
        elif by == "tokens_used":
            entries.sort(key=lambda e: e.tokens_used, reverse=True)
        
        return entries[:limit]
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        count = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                count += 1
        
        return count
    
    def warmup(self, entries: List[Dict[str, Any]]):
        """预热缓存"""
        for entry_data in entries:
            self.set(
                query=entry_data["query"],
                response=entry_data["response"],
                model=entry_data.get("model", "default"),
                tokens_used=entry_data.get("tokens_used", 0),
                ttl=entry_data.get("ttl"),
                metadata=entry_data.get("metadata")
            )


class CacheMiddleware:
    """缓存中间件"""
    
    def __init__(self, cache: ResponseCache):
        self.cache = cache
    
    def wrap_llm_call(
        self,
        llm_func: Callable,
        model: str,
        default_params: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """包装LLM调用函数"""
        def wrapped(query: str, **kwargs) -> str:
            # 合并参数
            params = {**(default_params or {}), **kwargs}
            
            # 尝试从缓存获取
            response, status = self.cache.get(query, model, params)
            
            if status in [CacheStatus.HIT, CacheStatus.SEMANTIC_HIT]:
                logger.debug(f"Cache {status.value} for query: {query[:50]}...")
                return response
            
            # 调用LLM
            response = llm_func(query, **kwargs)
            
            # 缓存结果
            tokens_used = len(response.split())  # 简单估算
            self.cache.set(
                query=query,
                response=response,
                model=model,
                tokens_used=tokens_used,
                params=params
            )
            
            return response
        
        return wrapped


# 工厂函数
def create_cache(
    config: Optional[CacheConfig] = None,
    embedding_model: Optional[Any] = None
) -> ResponseCache:
    """创建缓存"""
    return ResponseCache(config, embedding_model)


# 便捷函数
def cached_response(
    query: str,
    model: str = "default",
    cache: Optional[ResponseCache] = None
) -> Tuple[Optional[str], CacheStatus]:
    """便捷缓存查询函数"""
    if cache is None:
        cache = ResponseCache()
    return cache.get(query, model)
