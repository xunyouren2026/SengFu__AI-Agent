"""
向量存储模块

提供向量存储的抽象接口和多种实现，包括朴素向量存储和HNSW近似最近邻索引。
仅使用Python标准库。
"""

import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# VectorDocument: 向量文档
# ============================================================

@dataclass
class VectorDocument:
    """向量文档，包含文档内容、向量和元数据。"""

    id: str
    content: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = "unknown"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """将文档转换为字典。"""
        return {
            "id": self.id,
            "content": self.content,
            "vector": self.vector,
            "metadata": self.metadata,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        """从字典创建文档。"""
        return cls(
            id=data["id"],
            content=data["content"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
            embedding_model=data.get("embedding_model", "unknown"),
            created_at=data.get("created_at", time.time()),
        )


# ============================================================
# VectorStore: 向量存储抽象基类
# ============================================================

class VectorStore(ABC):
    """向量存储抽象基类。"""

    @abstractmethod
    def add(self, doc_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加向量到存储。

        Args:
            doc_id: 文档ID
            vector: 向量
            metadata: 元数据
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """搜索最近邻。

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filter_fn: 元数据过滤函数

        Returns:
            [(doc_id, score, metadata), ...] 按分数降序排列
        """
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """删除文档。

        Args:
            doc_id: 文档ID

        Returns:
            是否成功删除
        """
        pass

    @abstractmethod
    def update(self, doc_id: str, vector: List[float]) -> bool:
        """更新文档向量。

        Args:
            doc_id: 文档ID
            vector: 新向量

        Returns:
            是否成功更新
        """
        pass

    def size(self) -> int:
        """返回存储中的文档数量。"""
        return 0

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """获取文档。"""
        return None

    def clear(self) -> None:
        """清空存储。"""
        pass


# ============================================================
# 距离度量函数
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度。

    Args:
        a: 向量A
        b: 向量B

    Returns:
        余弦相似度，范围[-1, 1]
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不匹配: {len(a)} != {len(b)}")

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(a)):
        dot_product += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """计算欧氏距离。

    Args:
        a: 向量A
        b: 向量B

    Returns:
        欧氏距离，范围[0, +inf)
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不匹配: {len(a)} != {len(b)}")

    sum_sq = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sum_sq += diff * diff

    return math.sqrt(sum_sq)


def inner_product(a: List[float], b: List[float]) -> float:
    """计算内积（点积）。

    Args:
        a: 向量A
        b: 向量B

    Returns:
        内积值
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不匹配: {len(a)} != {len(b)}")

    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]

    return result


def euclidean_to_similarity(distance: float) -> float:
    """将欧氏距离转换为相似度分数（越大越好）。

    使用 1 / (1 + distance) 映射。

    Args:
        distance: 欧氏距离

    Returns:
        相似度分数
    """
    return 1.0 / (1.0 + distance)


# ============================================================
# NaiveVectorStore: 朴素向量存储（暴力搜索）
# ============================================================

class NaiveVectorStore(VectorStore):
    """朴素向量存储，基于字典存储，暴力搜索。

    支持余弦相似度、欧氏距离和内积三种距离度量。
    支持元数据过滤。
    """

    METRIC_COSINE = "cosine"
    METRIC_EUCLIDEAN = "euclidean"
    METRIC_INNER_PRODUCT = "inner_product"

    def __init__(
        self,
        metric: str = "cosine",
        dimension: Optional[int] = None,
    ):
        """初始化朴素向量存储。

        Args:
            metric: 距离度量方式，可选 "cosine", "euclidean", "inner_product"
            dimension: 向量维度（可选，用于验证）
        """
        self._store: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._metric = metric
        self._dimension = dimension

    @property
    def metric(self) -> str:
        """获取当前距离度量方式。"""
        return self._metric

    def _validate_vector(self, vector: List[float]) -> None:
        """验证向量维度。"""
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self._dimension}, 实际 {len(vector)}"
            )

    def _compute_score(self, a: List[float], b: List[float]) -> float:
        """计算两个向量之间的分数。

        Args:
            a: 向量A
            b: 向量B

        Returns:
            分数（越大越好）
        """
        if self._metric == self.METRIC_COSINE:
            return cosine_similarity(a, b)
        elif self._metric == self.METRIC_EUCLIDEAN:
            dist = euclidean_distance(a, b)
            return euclidean_to_similarity(dist)
        elif self._metric == self.METRIC_INNER_PRODUCT:
            return inner_product(a, b)
        else:
            raise ValueError(f"未知的距离度量: {self._metric}")

    def _apply_filter(
        self,
        doc_ids: List[str],
        filter_fn: Optional[callable] = None,
    ) -> List[str]:
        """应用元数据过滤。

        Args:
            doc_ids: 候选文档ID列表
            filter_fn: 过滤函数，接受metadata字典返回bool

        Returns:
            过滤后的文档ID列表
        """
        if filter_fn is None:
            return doc_ids

        filtered = []
        for doc_id in doc_ids:
            metadata = self._metadata.get(doc_id, {})
            if filter_fn(metadata):
                filtered.append(doc_id)

        return filtered

    def add(
        self,
        doc_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加向量到存储。"""
        self._validate_vector(vector)
        self._store[doc_id] = list(vector)
        self._metadata[doc_id] = metadata if metadata is not None else {}

    def add_batch(
        self,
        doc_ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """批量添加向量。

        Args:
            doc_ids: 文档ID列表
            vectors: 向量列表
            metadatas: 元数据列表
        """
        for i, doc_id in enumerate(doc_ids):
            metadata = None
            if metadatas is not None and i < len(metadatas):
                metadata = metadatas[i]
            self.add(doc_id, vectors[i], metadata)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """搜索最近邻（暴力搜索）。

        遍历所有向量计算分数，排序后返回top_k结果。

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filter_fn: 元数据过滤函数

        Returns:
            [(doc_id, score, metadata), ...] 按分数降序排列
        """
        self._validate_vector(query_vector)

        # 获取候选文档ID
        candidate_ids = list(self._store.keys())
        candidate_ids = self._apply_filter(candidate_ids, filter_fn)

        # 计算分数
        results = []
        for doc_id in candidate_ids:
            vector = self._store[doc_id]
            score = self._compute_score(query_vector, vector)
            results.append((doc_id, score, self._metadata[doc_id]))

        # 按分数降序排列
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def delete(self, doc_id: str) -> bool:
        """删除文档。"""
        if doc_id in self._store:
            del self._store[doc_id]
            self._metadata.pop(doc_id, None)
            return True
        return False

    def update(self, doc_id: str, vector: List[float]) -> bool:
        """更新文档向量。"""
        if doc_id not in self._store:
            return False

        self._validate_vector(vector)
        self._store[doc_id] = list(vector)
        return True

    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """更新文档元数据。

        Args:
            doc_id: 文档ID
            metadata: 新元数据（会合并到现有元数据）

        Returns:
            是否成功更新
        """
        if doc_id not in self._store:
            return False

        self._metadata[doc_id].update(metadata)
        return True

    def size(self) -> int:
        """返回存储中的文档数量。"""
        return len(self._store)

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """获取文档。"""
        if doc_id not in self._store:
            return None

        return VectorDocument(
            id=doc_id,
            content="",
            vector=self._store[doc_id],
            metadata=self._metadata[doc_id],
        )

    def get_vector(self, doc_id: str) -> Optional[List[float]]:
        """获取文档向量。"""
        return self._store.get(doc_id)

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档元数据。"""
        return self._metadata.get(doc_id)

    def list_ids(self) -> List[str]:
        """列出所有文档ID。"""
        return list(self._store.keys())

    def clear(self) -> None:
        """清空存储。"""
        self._store.clear()
        self._metadata.clear()


# ============================================================
# HNSWIndex: HNSW近似最近邻索引
# ============================================================

class HNSWIndex:
    """HNSW (Hierarchical Navigable Small World) 近似最近邻索引。

    基于分层可导航小世界图结构，支持高效的近似最近邻搜索。
    使用贪心搜索在多层图中查找最近邻。

    参数:
        M: 每个节点的最大连接数
        ef_construction: 构建时的搜索宽度
        max_level: 最大层数
        metric: 距离度量方式
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        max_level: int = 16,
        metric: str = "cosine",
    ):
        """初始化HNSW索引。

        Args:
            M: 每个节点的最大连接数
            ef_construction: 构建时的搜索宽度
            max_level: 最大层数
            metric: 距离度量方式
        """
        self._M = M
        self._M_max0 = 2 * M  # 第0层的最大连接数
        self._ef_construction = ef_construction
        self._max_level = max_level
        self._metric = metric

        # 节点存储: doc_id -> vector
        self._vectors: Dict[str, List[float]] = {}
        # 邻接表: doc_id -> {level: set of neighbor doc_ids}
        self._graphs: Dict[str, Dict[int, set]] = {}
        # 节点层级: doc_id -> level
        self._levels: Dict[str, int] = {}
        # 入口节点
        self._entry_point: Optional[str] = None
        self._entry_level: int = -1

        # 多层概率分布参数
        self._ml = 1.0 / math.log(M)

    def _distance(self, a: List[float], b: List[float]) -> float:
        """计算距离（越小越相似）。"""
        if self._metric == "cosine":
            return 1.0 - cosine_similarity(a, b)
        elif self._metric == "euclidean":
            return euclidean_distance(a, b)
        elif self._metric == "inner_product":
            return -inner_product(a, b)
        else:
            raise ValueError(f"未知的距离度量: {self._metric}")

    def _random_level(self) -> int:
        """生成随机层级（指数分布）。"""
        level = 0
        while random.random() < math.exp(-1.0 / self._ml) and level < self._max_level - 1:
            level += 1
        return level

    def _search_layer(
        self,
        query: List[float],
        entry_points: List[str],
        ef: int,
        level: int,
    ) -> List[Tuple[str, float]]:
        """在单层图中搜索最近邻。

        使用贪心搜索算法，从入口点出发，不断向更近的邻居移动。

        Args:
            query: 查询向量
            entry_points: 入口点列表
            ef: 搜索宽度
            level: 搜索层级

        Returns:
            [(doc_id, distance), ...] 按距离升序排列
        """
        visited = set(entry_points)

        # 计算入口点到查询的距离
        candidates = []
        results = []
        for ep in entry_points:
            if ep in self._vectors:
                dist = self._distance(query, self._vectors[ep])
                candidates.append((dist, ep))
                results.append((dist, ep))

        # 候选堆（最小堆，用列表模拟）
        candidates.sort()
        results.sort()

        while candidates:
            # 取最近的候选
            c_dist, c_id = candidates[0]
            # 取结果中最远的
            f_dist = results[-1][0] if results else float("inf")

            if c_dist > f_dist and len(results) >= ef:
                break

            candidates.pop(0)

            # 获取邻居
            neighbors = set()
            if c_id in self._graphs and level in self._graphs[c_id]:
                neighbors = self._graphs[c_id][level]

            for neighbor_id in neighbors:
                if neighbor_id not in visited and neighbor_id in self._vectors:
                    visited.add(neighbor_id)
                    n_dist = self._distance(query, self._vectors[neighbor_id])
                    f_dist = results[-1][0] if results else float("inf")

                    if n_dist < f_dist or len(results) < ef:
                        candidates.append((n_dist, neighbor_id))
                        candidates.sort()
                        results.append((n_dist, neighbor_id))
                        results.sort()
                        if len(results) > ef:
                            results.pop()

        return results

    def _select_neighbors_simple(
        self,
        query: List[float],
        candidates: List[Tuple[str, float]],
        M: int,
    ) -> List[Tuple[str, float]]:
        """简单邻居选择策略。

        选择距离最近的M个候选作为邻居。

        Args:
            query: 查询向量
            candidates: 候选邻居列表
            M: 最大邻居数

        Returns:
            选中的邻居列表
        """
        candidates.sort(key=lambda x: x[1])
        return candidates[:M]

    def _select_neighbors_heuristic(
        self,
        query: List[float],
        candidates: List[Tuple[str, float]],
        M: int,
    ) -> List[Tuple[str, float]]:
        """启发式邻居选择策略。

        在保持多样性的同时选择最近的邻居。
        对于每个候选，如果它与已选邻居中任何一个的距离小于它与查询的距离，
        则跳过该候选（避免选择过于相似的邻居）。

        Args:
            query: 查询向量
            candidates: 候选邻居列表
            M: 最大邻居数

        Returns:
            选中的邻居列表
        """
        candidates.sort(key=lambda x: x[1])
        selected = []

        for cand_dist, cand_id in candidates:
            if len(selected) >= M:
                break

            # 检查与已选邻居的多样性
            is_good = True
            for sel_dist, sel_id in selected:
                if sel_id in self._vectors and cand_id in self._vectors:
                    inter_dist = self._distance(
                        self._vectors[sel_id], self._vectors[cand_id]
                    )
                    if inter_dist < cand_dist:
                        is_good = False
                        break

            if is_good:
                selected.append((cand_dist, cand_id))

        return selected

    def _add_connection(self, node_id: str, neighbor_id: str, level: int) -> None:
        """在指定层级添加连接。"""
        if node_id not in self._graphs:
            self._graphs[node_id] = {}

        if level not in self._graphs[node_id]:
            self._graphs[node_id][level] = set()

        self._graphs[node_id][level].add(neighbor_id)

    def _insert(self, doc_id: str, vector: List[float]) -> None:
        """插入单个向量到HNSW索引。

        使用分层插入算法：
        1. 为新节点分配随机层级
        2. 从入口点开始，从上到下在各层搜索最近邻
        3. 在新节点的层级及以下层建立连接

        Args:
            doc_id: 文档ID
            vector: 向量
        """
        level = self._random_level()
        self._vectors[doc_id] = vector
        self._levels[doc_id] = level
        self._graphs[doc_id] = {}

        # 如果索引为空，设为入口点
        if self._entry_point is None:
            self._entry_point = doc_id
            self._entry_level = level
            return

        # 从顶层开始搜索
        ep = [self._entry_point]
        curr_level = self._entry_level

        # 从顶层向下到新节点层级+1，只搜索不连接
        for lc in range(curr_level, level, -1):
            results = self._search_layer(vector, ep, ef=1, level=lc)
            ep = [results[0][1]] if results else ep

        # 在新节点层级到第0层，搜索并建立连接
        for lc in range(min(level, curr_level), -1, -1):
            results = self._search_layer(vector, ep, ef=self._ef_construction, level=lc)

            M_max = self._M_max0 if lc == 0 else self._M
            neighbors = self._select_neighbors_heuristic(vector, results, M_max)

            # 建立双向连接
            for n_dist, n_id in neighbors:
                self._add_connection(doc_id, n_id, lc)
                self._add_connection(n_id, doc_id, lc)

                # 如果邻居连接数超过上限，修剪
                n_neighbors = self._graphs.get(n_id, {}).get(lc, set())
                if len(n_neighbors) > M_max:
                    n_candidates = [
                        (self._distance(self._vectors[n_id], self._vectors[nn_id]), nn_id)
                        for nn_id in n_neighbors
                        if nn_id in self._vectors
                    ]
                    pruned = self._select_neighbors_heuristic(
                        self._vectors[n_id], n_candidates, M_max
                    )
                    self._graphs[n_id][lc] = {p_id for _, p_id in pruned}

            ep = [r_id for _, r_id in results[:self._ef_construction]]

        # 更新入口点
        if level > self._entry_level:
            self._entry_point = doc_id
            self._entry_level = level

    def build(self, vectors: Dict[str, List[float]]) -> None:
        """构建HNSW索引。

        Args:
            vectors: {doc_id: vector} 字典
        """
        self._vectors.clear()
        self._graphs.clear()
        self._levels.clear()
        self._entry_point = None
        self._entry_level = -1

        for doc_id, vector in vectors.items():
            self._insert(doc_id, vector)

    def add(self, doc_id: str, vector: List[float]) -> None:
        """添加单个向量。"""
        self._insert(doc_id, vector)

    def search(
        self,
        query: List[float],
        k: int = 10,
        ef: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """搜索最近邻。

        Args:
            query: 查询向量
            k: 返回数量
            ef: 搜索宽度（越大越精确，越慢）

        Returns:
            [(doc_id, distance), ...] 按距离升序排列
        """
        if self._entry_point is None:
            return []

        if ef is None:
            ef = max(k, self._ef_construction)

        ep = [self._entry_point]
        curr_level = self._entry_level

        # 从顶层向下搜索到第1层
        for lc in range(curr_level, 0, -1):
            results = self._search_layer(query, ep, ef=1, level=lc)
            ep = [results[0][1]] if results else ep

        # 在第0层搜索
        results = self._search_layer(query, ep, ef=ef, level=0)

        return results[:k]

    def delete(self, doc_id: str) -> bool:
        """从索引中删除文档。

        注意：HNSW的删除是软删除，仅移除向量数据，
        图中的连接可能残留（惰性删除策略）。

        Args:
            doc_id: 文档ID

        Returns:
            是否成功删除
        """
        if doc_id not in self._vectors:
            return False

        # 从所有邻居的连接中移除
        if doc_id in self._graphs:
            for level, neighbors in self._graphs[doc_id].items():
                for neighbor_id in neighbors:
                    if neighbor_id in self._graphs and level in self._graphs[neighbor_id]:
                        self._graphs[neighbor_id][level].discard(doc_id)

        del self._vectors[doc_id]
        self._graphs.pop(doc_id, None)
        self._levels.pop(doc_id, None)

        # 如果删除的是入口点，选择新的入口点
        if self._entry_point == doc_id:
            if self._vectors:
                self._entry_point = next(iter(self._vectors))
                self._entry_level = self._levels.get(self._entry_point, 0)
            else:
                self._entry_point = None
                self._entry_level = -1

        return True

    def size(self) -> int:
        """返回索引中的文档数量。"""
        return len(self._vectors)

    def get_vector(self, doc_id: str) -> Optional[List[float]]:
        """获取文档向量。"""
        return self._vectors.get(doc_id)

    def get_neighbors(self, doc_id: str, level: int = 0) -> set:
        """获取文档在指定层级的邻居。"""
        if doc_id in self._graphs and level in self._graphs[doc_id]:
            return self._graphs[doc_id][level].copy()
        return set()

    def get_level(self, doc_id: str) -> int:
        """获取文档的层级。"""
        return self._levels.get(doc_id, 0)

    def clear(self) -> None:
        """清空索引。"""
        self._vectors.clear()
        self._graphs.clear()
        self._levels.clear()
        self._entry_point = None
        self._entry_level = -1
