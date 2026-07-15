"""
AGI Unified Framework - Embedding Engine Module
嵌入生成引擎：批量嵌入、缓存、归一化、降维、相似度计算

提供高效的文本嵌入生成和处理能力，支持多种相似度算法和降维技术。
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Set, Tuple, Union


class SimilarityMetric(str, Enum):
    """相似度度量类型"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    HAMMING = "hamming"
    JACCARD = "jaccard"
    PEARSON = "pearson"


class NormalizationType(str, Enum):
    """归一化类型"""
    L2 = "l2"
    L1 = "l1"
    MAX = "max"
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    NONE = "none"


class ReductionMethod(str, Enum):
    """降维方法"""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    RANDOM_PROJECTION = "random_projection"
    TRUNCATED_SVD = "truncated_svd"


@dataclass
class EmbeddingVector:
    """嵌入向量"""
    vector: List[float]
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    model: str = ""
    
    @property
    def dimension(self) -> int:
        return len(self.vector)
    
    def __len__(self) -> int:
        return len(self.vector)
    
    def __getitem__(self, idx: int) -> float:
        return self.vector[idx]


@dataclass
class SimilarityResult:
    """相似度结果"""
    query_idx: int
    target_idx: int
    score: float
    distance: float
    
    def __lt__(self, other: SimilarityResult) -> bool:
        return self.score > other.score  # 降序排列


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    embedding: EmbeddingVector
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def touch(self) -> None:
        """更新访问时间"""
        self.access_count += 1
        self.last_access = time.time()


class EmbeddingCache:
    """
    嵌入缓存
    
    实现LRU缓存策略，存储和复用嵌入向量以减少重复计算。
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # 使用OrderedDict实现LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, text: str, model: str, **params) -> str:
        """生成缓存键"""
        key_data = f"{model}:{text}:{sorted(params.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(self, text: str, model: str, **params) -> Optional[EmbeddingVector]:
        """
        获取缓存的嵌入
        
        Args:
            text: 文本
            model: 模型名称
            **params: 其他参数
            
        Returns:
            嵌入向量或None
        """
        key = self._generate_key(text, model, **params)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # 检查TTL
                if self.ttl_seconds is not None:
                    if time.time() - entry.last_access > self.ttl_seconds:
                        del self._cache[key]
                        self._misses += 1
                        return None
                
                # 更新访问记录
                entry.touch()
                self._cache.move_to_end(key)
                self._hits += 1
                return entry.embedding
            
            self._misses += 1
            return None
    
    def put(self, text: str, embedding: EmbeddingVector, model: str, **params) -> None:
        """
        存储嵌入到缓存
        
        Args:
            text: 文本
            embedding: 嵌入向量
            model: 模型名称
            **params: 其他参数
        """
        key = self._generate_key(text, model, **params)
        
        with self._lock:
            # 如果已存在，更新
            if key in self._cache:
                self._cache[key].embedding = embedding
                self._cache[key].touch()
                self._cache.move_to_end(key)
                return
            
            # 检查容量
            if len(self._cache) >= self.max_size:
                # 淘汰最久未使用的
                self._cache.popitem(last=False)
            
            # 添加新条目
            entry = CacheEntry(key=key, embedding=embedding)
            self._cache[key] = entry
    
    def invalidate(self, text: str, model: str, **params) -> bool:
        """
        使缓存条目失效
        
        Args:
            text: 文本
            model: 模型名称
            **params: 其他参数
            
        Returns:
            是否成功删除
        """
        key = self._generate_key(text, model, **params)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """
        清空缓存
        
        Returns:
            清除的条目数
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0.0,
            }
    
    def get_popular_keys(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        获取最常访问的键
        
        Args:
            n: 返回数量
            
        Returns:
            (键, 访问次数)列表
        """
        with self._lock:
            items = [(k, v.access_count) for k, v in self._cache.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            return items[:n]


class EmbeddingNormalizer:
    """
    嵌入归一化器
    
    提供多种向量归一化方法，优化嵌入向量的质量和可比性。
    """
    
    def __init__(self, method: NormalizationType = NormalizationType.L2):
        self.method = method
    
    def normalize(self, vector: List[float]) -> List[float]:
        """
        归一化向量
        
        Args:
            vector: 输入向量
            
        Returns:
            归一化后的向量
        """
        if self.method == NormalizationType.NONE:
            return vector.copy()
        
        elif self.method == NormalizationType.L2:
            return self._l2_normalize(vector)
        
        elif self.method == NormalizationType.L1:
            return self._l1_normalize(vector)
        
        elif self.method == NormalizationType.MAX:
            return self._max_normalize(vector)
        
        elif self.method == NormalizationType.Z_SCORE:
            return self._z_score_normalize(vector)
        
        elif self.method == NormalizationType.MIN_MAX:
            return self._min_max_normalize(vector)
        
        return vector.copy()
    
    def normalize_batch(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        批量归一化
        
        Args:
            vectors: 向量列表
            
        Returns:
            归一化后的向量列表
        """
        return [self.normalize(v) for v in vectors]
    
    def _l2_normalize(self, vector: List[float]) -> List[float]:
        """L2归一化"""
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector.copy()
        return [x / norm for x in vector]
    
    def _l1_normalize(self, vector: List[float]) -> List[float]:
        """L1归一化"""
        norm = sum(abs(x) for x in vector)
        if norm == 0:
            return vector.copy()
        return [x / norm for x in vector]
    
    def _max_normalize(self, vector: List[float]) -> List[float]:
        """Max归一化"""
        max_val = max(abs(x) for x in vector)
        if max_val == 0:
            return vector.copy()
        return [x / max_val for x in vector]
    
    def _z_score_normalize(self, vector: List[float]) -> List[float]:
        """Z-Score归一化"""
        mean = sum(vector) / len(vector)
        variance = sum((x - mean) ** 2 for x in vector) / len(vector)
        std = math.sqrt(variance)
        
        if std == 0:
            return [0.0] * len(vector)
        return [(x - mean) / std for x in vector]
    
    def _min_max_normalize(self, vector: List[float]) -> List[float]:
        """Min-Max归一化"""
        min_val = min(vector)
        max_val = max(vector)
        
        if max_val == min_val:
            return [0.5] * len(vector)
        
        return [(x - min_val) / (max_val - min_val) for x in vector]


class DimensionalityReducer:
    """
    降维器
    
    实现多种降维算法，减少嵌入向量的维度同时保留重要信息。
    """
    
    def __init__(self, method: ReductionMethod = ReductionMethod.PCA, target_dim: int = 128):
        self.method = method
        self.target_dim = target_dim
        self._fitted = False
        
        # PCA相关
        self._mean: Optional[List[float]] = None
        self._components: Optional[List[List[float]]] = None
        
        # 随机投影相关
        self._projection_matrix: Optional[List[List[float]]] = None
        
        # 统计信息
        self._input_dim: int = 0
    
    def fit(self, vectors: List[List[float]]) -> None:
        """
        拟合降维模型
        
        Args:
            vectors: 训练向量
        """
        if not vectors:
            return
        
        self._input_dim = len(vectors[0])
        
        if self.method == ReductionMethod.PCA:
            self._fit_pca(vectors)
        elif self.method == ReductionMethod.RANDOM_PROJECTION:
            self._fit_random_projection()
        elif self.method == ReductionMethod.TRUNCATED_SVD:
            self._fit_truncated_svd(vectors)
        
        self._fitted = True
    
    def transform(self, vector: List[float]) -> List[float]:
        """
        降维变换
        
        Args:
            vector: 输入向量
            
        Returns:
            降维后的向量
        """
        if not self._fitted:
            raise RuntimeError("Reducer must be fitted before transform")
        
        if self.method == ReductionMethod.PCA:
            return self._transform_pca(vector)
        elif self.method == ReductionMethod.RANDOM_PROJECTION:
            return self._transform_random_projection(vector)
        elif self.method == ReductionMethod.TRUNCATED_SVD:
            return self._transform_truncated_svd(vector)
        
        # 其他方法：简单截断
        return vector[:self.target_dim]
    
    def fit_transform(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        拟合并变换
        
        Args:
            vectors: 输入向量
            
        Returns:
            降维后的向量
        """
        self.fit(vectors)
        return [self.transform(v) for v in vectors]
    
    def _fit_pca(self, vectors: List[List[float]]) -> None:
        """拟合PCA（简化实现）"""
        # 计算均值
        n = len(vectors)
        self._mean = [sum(v[i] for v in vectors) / n for i in range(self._input_dim)]
        
        # 中心化
        centered = [[v[i] - self._mean[i] for i in range(self._input_dim)] for v in vectors]
        
        # 简化的PCA：使用随机投影近似
        self._components = self._generate_random_matrix(self.target_dim, self._input_dim)
    
    def _transform_pca(self, vector: List[float]) -> List[float]:
        """PCA变换"""
        if self._mean is None or self._components is None:
            return vector[:self.target_dim]
        
        # 中心化
        centered = [vector[i] - self._mean[i] for i in range(len(vector))]
        
        # 投影
        result = []
        for component in self._components:
            projection = sum(c * x for c, x in zip(component, centered))
            result.append(projection)
        
        return result
    
    def _fit_random_projection(self) -> None:
        """拟合随机投影"""
        self._projection_matrix = self._generate_random_matrix(self.target_dim, self._input_dim)
    
    def _transform_random_projection(self, vector: List[float]) -> List[float]:
        """随机投影变换"""
        if self._projection_matrix is None:
            return vector[:self.target_dim]
        
        result = []
        for row in self._projection_matrix:
            projection = sum(r * x for r, x in zip(row, vector))
            result.append(projection)
        
        return result
    
    def _fit_truncated_svd(self, vectors: List[List[float]]) -> None:
        """拟合截断SVD（使用PCA近似）"""
        self._fit_pca(vectors)
    
    def _transform_truncated_svd(self, vector: List[float]) -> List[float]:
        """截断SVD变换"""
        return self._transform_pca(vector)
    
    def _generate_random_matrix(self, rows: int, cols: int) -> List[List[float]]:
        """生成随机投影矩阵"""
        # 使用高斯随机矩阵
        return [[random.gauss(0, 1.0 / math.sqrt(cols)) for _ in range(cols)] for _ in range(rows)]


class SimilarityComputer:
    """
    相似度计算器
    
    实现多种向量相似度度量算法，支持批量计算和Top-K检索。
    """
    
    def __init__(self, metric: SimilarityMetric = SimilarityMetric.COSINE):
        self.metric = metric
    
    def compute(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        if self.metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_similarity(vec1, vec2)
        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            return self._dot_product(vec1, vec2)
        elif self.metric == SimilarityMetric.MANHATTAN:
            return self._manhattan_similarity(vec1, vec2)
        elif self.metric == SimilarityMetric.PEARSON:
            return self._pearson_correlation(vec1, vec2)
        
        return self._cosine_similarity(vec1, vec2)
    
    def compute_batch(
        self,
        query: List[float],
        candidates: List[List[float]],
    ) -> List[SimilarityResult]:
        """
        批量计算相似度
        
        Args:
            query: 查询向量
            candidates: 候选向量列表
            
        Returns:
            相似度结果列表
        """
        results = []
        for i, candidate in enumerate(candidates):
            score = self.compute(query, candidate)
            distance = 1.0 - score if score <= 1.0 else 1.0 / score
            results.append(SimilarityResult(
                query_idx=0,
                target_idx=i,
                score=score,
                distance=distance,
            ))
        return results
    
    def top_k(
        self,
        query: List[float],
        candidates: List[List[float]],
        k: int = 5,
    ) -> List[SimilarityResult]:
        """
        查找Top-K最相似的向量
        
        Args:
            query: 查询向量
            candidates: 候选向量列表
            k: 返回数量
            
        Returns:
            Top-K相似度结果
        """
        results = self.compute_batch(query, candidates)
        
        # 使用堆排序获取Top-K
        if self.metric in (SimilarityMetric.EUCLIDEAN, SimilarityMetric.MANHATTAN):
            # 距离越小越好
            return heapq.nsmallest(k, results, key=lambda x: x.distance)
        else:
            # 相似度越大越好
            return heapq.nlargest(k, results, key=lambda x: x.score)
    
    def compute_matrix(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        计算相似度矩阵
        
        Args:
            vectors: 向量列表
            
        Returns:
            相似度矩阵
        """
        n = len(vectors)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            matrix[i][i] = 1.0  # 自身相似度为1
            for j in range(i + 1, n):
                sim = self.compute(vectors[i], vectors[j])
                matrix[i][j] = sim
                matrix[j][i] = sim
        
        return matrix
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _euclidean_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """欧氏距离相似度（转换为相似度）"""
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
        # 转换为相似度：1 / (1 + distance)
        return 1.0 / (1.0 + distance)
    
    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """点积"""
        return sum(a * b for a, b in zip(vec1, vec2))
    
    def _manhattan_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """曼哈顿距离相似度"""
        distance = sum(abs(a - b) for a, b in zip(vec1, vec2))
        return 1.0 / (1.0 + distance)
    
    def _pearson_correlation(self, vec1: List[float], vec2: List[float]) -> float:
        """皮尔逊相关系数"""
        n = len(vec1)
        mean1 = sum(vec1) / n
        mean2 = sum(vec2) / n
        
        numerator = sum((a - mean1) * (b - mean2) for a, b in zip(vec1, vec2))
        std1 = math.sqrt(sum((a - mean1) ** 2 for a in vec1))
        std2 = math.sqrt(sum((b - mean2) ** 2 for b in vec2))
        
        if std1 == 0 or std2 == 0:
            return 0.0
        
        return numerator / (std1 * std2)


class BatchEmbedder:
    """
    批量嵌入器
    
    管理批量嵌入生成，支持动态批大小和并发处理。
    """
    
    def __init__(
        self,
        embed_func: Callable[[List[str]], List[List[float]]],
        batch_size: int = 32,
        max_workers: int = 4,
    ):
        self.embed_func = embed_func
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self._stats = {
            "total_embedded": 0,
            "total_batches": 0,
            "total_time_ms": 0.0,
        }
        self._lock = threading.Lock()
    
    def embed(self, texts: List[str]) -> Iterator[Tuple[str, List[float]]]:
        """
        批量嵌入文本
        
        Args:
            texts: 文本列表
            
        Yields:
            (文本, 嵌入向量)元组
        """
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            start_time = time.time()
            embeddings = self.embed_func(batch)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 更新统计
            with self._lock:
                self._stats["total_embedded"] += len(batch)
                self._stats["total_batches"] += 1
                self._stats["total_time_ms"] += elapsed_ms
            
            for text, embedding in zip(batch, embeddings):
                yield (text, embedding)
    
    def embed_parallel(
        self,
        texts: List[str],
    ) -> List[Tuple[str, List[float]]]:
        """
        并行批量嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            (文本, 嵌入向量)列表
        """
        results = []
        
        # 分割批次
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        # 使用线程池并行处理
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._embed_batch, batch) for batch in batches]
            
            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)
        
        return results
    
    def _embed_batch(self, batch: List[str]) -> List[Tuple[str, List[float]]]:
        """嵌入单个批次"""
        start_time = time.time()
        embeddings = self.embed_func(batch)
        elapsed_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            self._stats["total_embedded"] += len(batch)
            self._stats["total_batches"] += 1
            self._stats["total_time_ms"] += elapsed_ms
        
        return list(zip(batch, embeddings))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()
            if stats["total_batches"] > 0:
                stats["avg_batch_time_ms"] = stats["total_time_ms"] / stats["total_batches"]
                stats["avg_time_per_text_ms"] = stats["total_time_ms"] / max(1, stats["total_embedded"])
            return stats


class EmbeddingEngine:
    """
    嵌入生成引擎
    
    整合缓存、归一化、降维和相似度计算，提供完整的嵌入处理流程。
    """
    
    def __init__(
        self,
        embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
        model_name: str = "default",
        cache_size: int = 10000,
        normalization: NormalizationType = NormalizationType.L2,
        target_dim: Optional[int] = None,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        self.model_name = model_name
        self.embed_func = embed_func
        
        # 组件
        self.cache = EmbeddingCache(max_size=cache_size)
        self.normalizer = EmbeddingNormalizer(method=normalization)
        self.reducer: Optional[DimensionalityReducer] = None
        self.similarity = SimilarityComputer(metric=similarity_metric)
        self.batch_embedder: Optional[BatchEmbedder] = None
        
        if target_dim is not None:
            self.reducer = DimensionalityReducer(
                method=ReductionMethod.PCA,
                target_dim=target_dim,
            )
        
        if embed_func is not None:
            self.batch_embedder = BatchEmbedder(embed_func)
        
        self._lock = threading.RLock()
    
    def embed(self, text: str, use_cache: bool = True) -> EmbeddingVector:
        """
        嵌入单个文本
        
        Args:
            text: 文本
            use_cache: 是否使用缓存
            
        Returns:
            嵌入向量
        """
        # 检查缓存
        if use_cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return cached
        
        # 生成嵌入
        if self.embed_func is None:
            raise RuntimeError("No embedding function provided")
        
        embeddings = self.embed_func([text])
        vector = embeddings[0]
        
        # 归一化
        vector = self.normalizer.normalize(vector)
        
        # 降维
        if self.reducer is not None:
            if not self.reducer._fitted:
                # 首次需要拟合
                self.reducer.fit([vector])
            vector = self.reducer.transform(vector)
        
        result = EmbeddingVector(
            vector=vector,
            text=text,
            model=self.model_name,
        )
        
        # 存入缓存
        if use_cache:
            self.cache.put(text, result, self.model_name)
        
        return result
    
    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> List[EmbeddingVector]:
        """
        批量嵌入
        
        Args:
            texts: 文本列表
            use_cache: 是否使用缓存
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        results = []
        texts_to_embed = []
        indices = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            if use_cache:
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    results.append((i, cached))
                    continue
            
            texts_to_embed.append(text)
            indices.append(i)
        
        # 嵌入未缓存的文本
        if texts_to_embed and self.batch_embedder is not None:
            for text, vector in self.batch_embedder.embed(texts_to_embed):
                # 归一化
                vector = self.normalizer.normalize(vector)
                
                embedding = EmbeddingVector(
                    vector=vector,
                    text=text,
                    model=self.model_name,
                )
                
                # 存入缓存
                if use_cache:
                    self.cache.put(text, embedding, self.model_name)
                
                results.append((indices[len([r for r in results if r[0] < len(results)])], embedding))
        
        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def fit_reducer(self, vectors: List[List[float]]) -> None:
        """
        拟合降维器
        
        Args:
            vectors: 训练向量
        """
        if self.reducer is not None:
            self.reducer.fit(vectors)
    
    def search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            corpus: 文档集合
            top_k: 返回数量
            
        Returns:
            (文档, 相似度)列表
        """
        # 嵌入查询
        query_embedding = self.embed(query)
        
        # 嵌入文档
        corpus_embeddings = self.embed_batch(corpus)
        
        # 计算相似度
        candidates = [e.vector for e in corpus_embeddings]
        top_results = self.similarity.top_k(query_embedding.vector, candidates, top_k)
        
        return [(corpus[r.target_idx], r.score) for r in top_results]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        return self.similarity.compute(emb1.vector, emb2.vector)
    
    def cluster(
        self,
        texts: List[str],
        n_clusters: int = 3,
    ) -> Dict[int, List[str]]:
        """
        简单聚类（K-Means简化版）
        
        Args:
            texts: 文本列表
            n_clusters: 聚类数
            
        Returns:
            聚类结果
        """
        if len(texts) < n_clusters:
            return {i: [text] for i, text in enumerate(texts)}
        
        # 嵌入所有文本
        embeddings = self.embed_batch(texts)
        vectors = [e.vector for e in embeddings]
        
        # 随机初始化中心点
        random.seed(42)
        centroids = random.sample(vectors, n_clusters)
        
        # 迭代优化
        for _ in range(10):  # 固定迭代次数
            # 分配点到最近的中心
            clusters: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
            
            for i, vec in enumerate(vectors):
                similarities = [self.similarity.compute(vec, c) for c in centroids]
                closest = max(range(n_clusters), key=lambda j: similarities[j])
                clusters[closest].append(i)
            
            # 更新中心点
            for k in range(n_clusters):
                if clusters[k]:
                    cluster_vectors = [vectors[i] for i in clusters[k]]
                    centroids[k] = [
                        sum(v[i] for v in cluster_vectors) / len(cluster_vectors)
                        for i in range(len(cluster_vectors[0]))
                    ]
        
        # 构建结果
        result: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for k, indices in clusters.items():
            result[k] = [texts[i] for i in indices]
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        stats = {
            "cache": self.cache.get_stats(),
            "model": self.model_name,
            "normalization": self.normalizer.method.value,
            "similarity_metric": self.similarity.metric.value,
        }
        
        if self.batch_embedder is not None:
            stats["batch"] = self.batch_embedder.get_stats()
        
        if self.reducer is not None:
            stats["reducer"] = {
                "method": self.reducer.method.value,
                "target_dim": self.reducer.target_dim,
                "fitted": self.reducer._fitted,
            }
        
        return stats
    
    def clear_cache(self) -> int:
        """清除缓存"""
        return self.cache.clear()
