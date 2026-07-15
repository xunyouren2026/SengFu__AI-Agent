"""
AGI Unified Framework - Cache Manager
KV缓存管理器和语义缓存，支持前缀匹配和LRU淘汰
"""

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import Message, Usage


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stored_entries: int = 0
    total_memory_bytes: int = 0
    saved_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    @property
    def saved_tokens_ratio(self) -> float:
        total = self.hits + self.misses
        return self.saved_tokens / max(total, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "stored_entries": self.stored_entries,
            "total_memory_bytes": self.total_memory_bytes,
            "saved_tokens": self.saved_tokens,
            "saved_tokens_ratio": round(self.saved_tokens_ratio, 2),
        }


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str = ""
    prefix_key: str = ""
    kv_cache: Any = None
    token_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_size: int = 0

    def touch(self):
        """更新访问时间"""
        self.last_accessed = time.time()
        self.access_count += 1


class KVCacheManager:
    """
    KV缓存管理器

    功能：
    - 基于消息前缀哈希的缓存键计算
    - 前缀匹配的缓存查找
    - LRU缓存淘汰策略
    - 缓存统计
    - 内存管理
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_bytes: int = 0,  # 0表示不限制
        default_ttl: float = 3600.0,  # 默认TTL（秒）
    ):
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_bytes
        self._default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._prefix_index: Dict[str, List[str]] = {}  # prefix_key -> [cache_keys]
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def get_cache_key(self, messages: List[Message], params: Optional[Dict[str, Any]] = None) -> str:
        """
        计算缓存键（基于消息前缀哈希）

        Args:
            messages: 消息列表
            params: 可选的参数（影响缓存键）

        Returns:
            str: 缓存键
        """
        # 序列化消息内容
        msg_str = "|".join(
            f"{m.role}:{m.content}" for m in messages
        )

        # 添加参数到哈希
        if params:
            param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
            msg_str += f"|params:{param_str}"

        # 计算SHA256哈希
        hash_obj = hashlib.sha256(msg_str.encode("utf-8"))
        return hash_obj.hexdigest()

    def get_prefix_key(self, messages: List[Message]) -> str:
        """
        计算前缀键（用于前缀匹配）

        使用除最后一条消息外的所有消息作为前缀。

        Args:
            messages: 消息列表

        Returns:
            str: 前缀键
        """
        if len(messages) <= 1:
            return ""

        prefix_msgs = messages[:-1]
        return self.get_cache_key(prefix_msgs)

    def lookup(self, messages: List[Message]) -> Optional[CacheEntry]:
        """
        查找前缀匹配的缓存

        Args:
            messages: 消息列表

        Returns:
            Optional[CacheEntry]: 匹配的缓存条目
        """
        prefix_key = self.get_prefix_key(messages)

        with self._lock:
            self._stats.misses += 1

            if not prefix_key:
                return None

            cache_keys = self._prefix_index.get(prefix_key, [])
            for key in cache_keys:
                entry = self._cache.get(key)
                if entry is not None:
                    # 检查TTL
                    if time.time() - entry.created_at > self._default_ttl:
                        self._evict(key)
                        continue

                    entry.touch()
                    # 移动到OrderedDict末尾（LRU）
                    self._cache.move_to_end(key)
                    self._stats.hits += 1
                    self._stats.misses -= 1
                    self._stats.saved_tokens += entry.token_count
                    return entry

        return None

    def lookup_exact(self, messages: List[Message], params: Optional[Dict[str, Any]] = None) -> Optional[CacheEntry]:
        """
        精确匹配查找

        Args:
            messages: 消息列表
            params: 参数

        Returns:
            Optional[CacheEntry]: 匹配的缓存条目
        """
        key = self.get_cache_key(messages, params)

        with self._lock:
            self._stats.misses += 1

            entry = self._cache.get(key)
            if entry is not None:
                if time.time() - entry.created_at > self._default_ttl:
                    self._evict(key)
                    return None

                entry.touch()
                self._cache.move_to_end(key)
                self._stats.hits += 1
                self._stats.misses -= 1
                self._stats.saved_tokens += entry.token_count
                return entry

        return None

    def store(
        self,
        messages: List[Message],
        kv_cache: Any,
        token_count: int = 0,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        存储KV缓存

        Args:
            messages: 消息列表
            kv_cache: KV缓存数据
            token_count: Token数量
            params: 参数
            metadata: 元数据

        Returns:
            str: 缓存键
        """
        key = self.get_cache_key(messages, params)
        prefix_key = self.get_prefix_key(messages)

        # 估算内存大小
        memory_size = self._estimate_memory_size(kv_cache)

        entry = CacheEntry(
            key=key,
            prefix_key=prefix_key,
            kv_cache=kv_cache,
            token_count=token_count,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            metadata=metadata or {},
            memory_size=memory_size,
        )

        with self._lock:
            # 如果已存在，先删除
            if key in self._cache:
                self._evict(key)

            # 检查容量限制
            while len(self._cache) >= self._max_entries:
                self._evict_oldest()

            # 检查内存限制
            if self._max_memory_bytes > 0:
                while self._stats.total_memory_bytes + memory_size > self._max_memory_bytes:
                    if len(self._cache) == 0:
                        break
                    self._evict_oldest()

            # 存储缓存
            self._cache[key] = entry
            self._cache.move_to_end(key)

            # 更新前缀索引
            if prefix_key:
                if prefix_key not in self._prefix_index:
                    self._prefix_index[prefix_key] = []
                self._prefix_index[prefix_key].append(key)

            # 更新统计
            self._stats.stored_entries = len(self._cache)
            self._stats.total_memory_bytes += memory_size

        return key

    def _evict(self, key: str) -> None:
        """淘汰指定缓存条目"""
        entry = self._cache.pop(key, None)
        if entry:
            self._stats.evictions += 1
            self._stats.total_memory_bytes -= entry.memory_size
            self._stats.stored_entries = len(self._cache)

            # 清理前缀索引
            if entry.prefix_key and entry.prefix_key in self._prefix_index:
                keys = self._prefix_index[entry.prefix_key]
                if key in keys:
                    keys.remove(key)
                if not keys:
                    del self._prefix_index[entry.prefix_key]

    def _evict_oldest(self) -> None:
        """淘汰最旧的缓存条目（LRU）"""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._evict(oldest_key)

    def evict(self, key: Optional[str] = None) -> int:
        """
        缓存淘汰

        Args:
            key: 指定淘汰的缓存键，为None时淘汰最旧的

        Returns:
            int: 淘汰的条目数
        """
        with self._lock:
            if key:
                if key in self._cache:
                    self._evict(key)
                    return 1
                return 0
            else:
                self._evict_oldest()
                return 1

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._prefix_index.clear()
            self._stats = CacheStats()

    def _estimate_memory_size(self, data: Any) -> int:
        """估算数据的内存大小"""
        try:
            serialized = json.dumps(data, ensure_ascii=False)
            return len(serialized.encode("utf-8"))
        except (TypeError, ValueError):
            return 1024  # 默认估算1KB

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return self._stats.to_dict()

    def get_entries_info(self) -> List[Dict[str, Any]]:
        """获取所有缓存条目的信息"""
        with self._lock:
            return [
                {
                    "key": entry.key[:16] + "...",
                    "prefix_key": entry.prefix_key[:16] + "..." if entry.prefix_key else "",
                    "token_count": entry.token_count,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "memory_size": entry.memory_size,
                    "age_seconds": round(time.time() - entry.created_at, 2),
                }
                for entry in self._cache.values()
            ]


class SemanticCache:
    """
    语义缓存

    基于文本相似度的缓存匹配，当输入与缓存中的输入语义相似时
    直接返回缓存结果，避免重复调用LLM。

    使用TF-IDF向量和余弦相似度进行匹配（不依赖外部库）。
    """

    def __init__(
        self,
        threshold: float = 0.85,
        max_entries: int = 500,
        default_ttl: float = 1800.0,
    ):
        self._threshold = threshold
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._entries: List[Dict[str, Any]] = []
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """简易分词"""
        text = text.lower()
        # 按空白字符和标点分割
        tokens = []
        current = []
        for char in text:
            if char.isalnum() or char in "_-":
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """计算词频（TF）"""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total = len(tokens)
        for token in tf:
            tf[token] /= max(total, 1)
        return tf

    def _update_idf(self, tokens: List[str]) -> None:
        """更新逆文档频率（IDF）"""
        self._doc_count += 1
        seen = set()
        for token in tokens:
            if token not in seen:
                seen.add(token)
                self._vocabulary[token] = self._vocabulary.get(token, 0) + 1

        # 重新计算IDF
        for token in self._vocabulary:
            df = self._vocabulary[token]
            self._idf[token] = math.log((self._doc_count + 1) / (df + 1)) + 1

    def _compute_tfidf(self, tokens: List[str]) -> Dict[str, float]:
        """计算TF-IDF向量"""
        tf = self._compute_tf(tokens)
        tfidf = {}
        for token, freq in tf.items():
            idf = self._idf.get(token, 1.0)
            tfidf[token] = freq * idf
        return tfidf

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 稀疏向量1（字典形式）
            vec2: 稀疏向量2（字典形式）

        Returns:
            float: 余弦相似度 [0, 1]
        """
        # 点积
        dot_product = 0.0
        for token in vec1:
            if token in vec2:
                dot_product += vec1[token] * vec2[token]

        # 模
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def similarity_check(self, text1: str, text2: str) -> float:
        """
        计算两段文本的语义相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度 [0, 1]
        """
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        vec1 = self._compute_tfidf(tokens1)
        vec2 = self._compute_tfidf(tokens2)

        return self._cosine_similarity(vec1, vec2)

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        语义缓存查找

        Args:
            query: 查询文本

        Returns:
            匹配的缓存结果，包含similarity和response
        """
        query_tokens = self._tokenize(query)
        query_vec = self._compute_tfidf(query_tokens)

        with self._lock:
            self._stats.misses += 1
            now = time.time()

            best_match = None
            best_similarity = 0.0

            for entry in self._entries:
                # 检查TTL
                if now - entry["created_at"] > self._default_ttl:
                    continue

                sim = self._cosine_similarity(query_vec, entry["vector"])
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = entry

            if best_match and best_similarity >= self._threshold:
                self._stats.hits += 1
                self._stats.misses -= 1
                self._stats.saved_tokens += best_match.get("token_count", 0)
                return {
                    "response": best_match["response"],
                    "similarity": best_similarity,
                    "cached_query": best_match["query"],
                    "token_count": best_match.get("token_count", 0),
                }

        return None

    def store(
        self,
        query: str,
        response: str,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        存储到语义缓存

        Args:
            query: 查询文本
            response: 响应文本
            token_count: Token数量
            metadata: 元数据
        """
        tokens = self._tokenize(query)
        vector = self._compute_tfidf(tokens)

        entry = {
            "query": query,
            "response": response,
            "vector": vector,
            "token_count": token_count,
            "created_at": time.time(),
            "metadata": metadata or {},
        }

        with self._lock:
            self._update_idf(tokens)
            self._entries.append(entry)

            # LRU淘汰
            while len(self._entries) > self._max_entries:
                self._entries.pop(0)

            self._stats.stored_entries = len(self._entries)

    def clear(self) -> None:
        """清空语义缓存"""
        with self._lock:
            self._entries.clear()
            self._vocabulary.clear()
            self._idf.clear()
            self._doc_count = 0
            self._stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                **self._stats.to_dict(),
                "threshold": self._threshold,
                "vocabulary_size": len(self._vocabulary),
                "doc_count": self._doc_count,
            }
