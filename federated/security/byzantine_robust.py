"""
Byzantine Robust Aggregator Module
拜占庭鲁棒聚合器模块

实现联邦学习中的拜占庭容错聚合算法，包括：
1. Krum 算法 - 选择最接近其他更新的单个更新
2. Multi-Krum 算法 - 选择多个最接近的更新取平均
3. Trimmed Mean - 去掉极值后取平均
4. Coordinate-wise Median - 坐标中位数聚合
5. Bulyan 算法 - Krum 选择 + TrimmedMean 聚合
6. FlTrust - 基于信任评分的加权聚合
7. Byzantine 检测器 - 基于余弦相似度和更新范数

Author: AGI Unified Framework
"""

import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set

logger = logging.getLogger(__name__)


# ============== 拜占庭配置 ==============

@dataclass
class ByzantineConfig:
    """
    拜占庭鲁棒聚合配置

    Attributes:
        max_byzantine: 最大恶意节点数 f
        aggregation_method: 聚合方法名称
        krum_candidates: Multi-Krum 选择的候选数量
        trim_count: TrimmedMean 每端修剪的数量（默认自动计算）
        detection_threshold: 拜占庭检测阈值
        trust_decay: FlTrust 信任衰减因子
        fltrust_root_update: FlTrust 是否使用服务器模型作为根信任
    """
    max_byzantine: int = 1
    aggregation_method: str = "krum"
    krum_candidates: int = 1
    trim_count: Optional[int] = None
    detection_threshold: float = 2.0
    trust_decay: float = 0.9
    fltrust_root_update: bool = True

    # 支持的聚合方法
    SUPPORTED_METHODS: Set[str] = field(default_factory=lambda: {
        "krum", "multi_krum", "trimmed_mean", "median",
        "bulyan", "fltrust", "mean",
    })

    def __post_init__(self) -> None:
        if self.max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0，当前值: {self.max_byzantine}")
        if self.aggregation_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的聚合方法: {self.aggregation_method}，"
                f"支持的方法: {self.SUPPORTED_METHODS}"
            )
        if self.krum_candidates < 1:
            raise ValueError(f"krum_candidates 必须 >= 1，当前值: {self.krum_candidates}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_byzantine": self.max_byzantine,
            "aggregation_method": self.aggregation_method,
            "krum_candidates": self.krum_candidates,
            "trim_count": self.trim_count,
            "detection_threshold": self.detection_threshold,
            "trust_decay": self.trust_decay,
        }


# ============== 工具函数 ==============

def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    计算两个向量之间的欧几里得距离

    ||a - b||_2 = sqrt(sum((a_i - b_i)^2))

    Args:
        a: 向量 a
        b: 向量 b

    Returns:
        欧几里得距离
    """
    min_len = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min_len)))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    计算两个向量的余弦相似度

    cos(a, b) = (a . b) / (||a|| * ||b||)

    Args:
        a: 向量 a
        b: 向量 b

    Returns:
        余弦相似度，范围 [-1, 1]
    """
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0

    dot = sum(a[i] * b[i] for i in range(min_len))
    norm_a = math.sqrt(sum(a[i] ** 2 for i in range(min_len)))
    norm_b = math.sqrt(sum(b[i] ** 2 for i in range(min_len)))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _vector_norm(vector: List[float]) -> float:
    """计算向量的L2范数"""
    return math.sqrt(sum(x * x for x in vector))


def _mean_vectors(vectors: List[List[float]]) -> List[float]:
    """计算多个向量的逐元素均值"""
    if not vectors:
        return []
    n = len(vectors)
    dim = max(len(v) for v in vectors)
    result = [0.0] * dim
    for vec in vectors:
        for i in range(len(vec)):
            result[i] += vec[i] / n
    return result


def _pad_vectors(vectors: List[List[float]]) -> Tuple[List[List[float]], int]:
    """
    将所有向量填充到相同长度

    Args:
        vectors: 向量列表

    Returns:
        (填充后的向量列表, 维度)
    """
    if not vectors:
        return ([], 0)
    dim = max(len(v) for v in vectors)
    padded = [v + [0.0] * (dim - len(v)) for v in vectors]
    return (padded, dim)


# ============== Krum 聚合器 ==============

class KrumAggregator:
    """
    Krum 聚合算法

    Krum 选择与其他更新距离之和最小的单个更新。
    基于假设：诚实节点的更新应该相互接近，而恶意节点的更新会偏离。

    算法:
    1. 对每个更新 i，计算 score_i = sum_{j in S_i} ||u_i - u_j||^2
       其中 S_i 是距离 u_i 最近的 n - f - 2 个更新的集合
    2. 选择 score 最小的更新作为聚合结果

    要求: n >= 2f + 3（至少需要 2f+3 个节点才能容忍 f 个恶意节点）

    参考文献:
    P. Blanchard et al., "Machine learning with adversaries: Byzantine tolerant gradient descent",
    NeurIPS 2017.
    """

    def __init__(self, max_byzantine: int = 1) -> None:
        if max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0，当前值: {max_byzantine}")
        self._f = max_byzantine

    def aggregate(self, updates: List[List[float]]) -> List[float]:
        """
        使用 Krum 算法选择最可靠的更新

        Args:
            updates: 客户端更新列表

        Returns:
            被选中的更新向量

        Raises:
            ValueError: 当更新数量不足以执行 Krum 时
        """
        if not updates:
            return []

        n = len(updates)
        f = self._f

        # Krum 要求 n >= 2f + 3
        if n < 2 * f + 3:
            logger.warning(
                f"Krum 需要至少 {2 * f + 3} 个更新，当前只有 {n} 个，"
                f"回退到中位数聚合"
            )
            return self._fallback_median(updates)

        # 计算所有更新对之间的距离
        padded, dim = _pad_vectors(updates)
        scores = self._compute_krum_scores(padded)

        # 选择 score 最小的更新
        best_idx = min(range(n), key=lambda i: scores[i])
        return list(padded[best_idx])

    def _compute_krum_scores(self, updates: List[List[float]]) -> List[float]:
        """
        计算每个更新的 Krum 分数

        score_i = sum_{j in nearest n-f-2} ||u_i - u_j||^2

        Args:
            updates: 已填充到相同长度的更新列表

        Returns:
            每个更新的 Krum 分数列表
        """
        n = len(updates)
        f = self._f
        # 选择最近的 n - f - 2 个邻居
        num_neighbors = n - f - 2

        scores: List[float] = []
        for i in range(n):
            # 计算更新 i 到所有其他更新的距离平方
            distances: List[float] = []
            for j in range(n):
                if i != j:
                    dist_sq = sum(
                        (updates[i][d] - updates[j][d]) ** 2
                        for d in range(len(updates[i]))
                    )
                    distances.append(dist_sq)

            # 取最近的 num_neighbors 个距离
            distances.sort()
            score = sum(distances[:num_neighbors])
            scores.append(score)

        return scores

    def _fallback_median(self, updates: List[List[float]]) -> List[float]:
        """回退到坐标中位数"""
        padded, dim = _pad_vectors(updates)
        result: List[float] = []
        for d in range(dim):
            values = sorted(u[d] for u in padded)
            n = len(values)
            if n % 2 == 0:
                result.append((values[n // 2 - 1] + values[n // 2]) / 2.0)
            else:
                result.append(values[n // 2])
        return result


# ============== Multi-Krum 聚合器 ==============

class MultiKrumAggregator:
    """
    Multi-Krum 聚合算法

    Multi-Krum 是 Krum 的扩展，选择前 m 个 score 最小的更新，
    然后取它们的平均值。相比 Krum，Multi-Krum 能更好地利用
    多个诚实节点的信息。

    算法:
    1. 计算每个更新的 Krum 分数
    2. 选择 score 最小的 m 个更新
    3. 对选中的更新取平均

    要求: n >= 2f + 2m + 1

    参考文献:
    P. Blanchard et al., "Machine learning with adversaries: Byzantine tolerant gradient descent",
    NeurIPS 2017.
    """

    def __init__(self, max_byzantine: int = 1, num_candidates: int = 2) -> None:
        if max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0，当前值: {max_byzantine}")
        if num_candidates < 1:
            raise ValueError(f"num_candidates 必须 >= 1，当前值: {num_candidates}")
        self._f = max_byzantine
        self._m = num_candidates

    def aggregate(self, updates: List[List[float]]) -> List[float]:
        """
        使用 Multi-Krum 算法聚合更新

        Args:
            updates: 客户端更新列表

        Returns:
            聚合后的更新向量
        """
        if not updates:
            return []

        n = len(updates)
        f = self._f
        m = self._m

        # 确保 m 不超过可用数量
        m = min(m, n - f)

        # Multi-Krum 要求 n >= 2f + 2m + 1
        if n < 2 * f + 2 * m + 1:
            logger.warning(
                f"Multi-Krum 需要至少 {2 * f + 2 * m + 1} 个更新，"
                f"当前只有 {n} 个，回退到 Krum"
            )
            krum = KrumAggregator(f)
            return krum.aggregate(updates)

        padded, dim = _pad_vectors(updates)

        # 计算 Krum 分数
        krum_agg = KrumAggregator(f)
        scores = krum_agg._compute_krum_scores(padded)

        # 选择 score 最小的 m 个更新
        indexed_scores = [(scores[i], i) for i in range(n)]
        indexed_scores.sort(key=lambda x: x[0])
        selected_indices = [idx for _, idx in indexed_scores[:m]]
        selected = [padded[i] for i in selected_indices]

        return _mean_vectors(selected)


# ============== Trimmed Mean 聚合器 ==============

class TrimmedMeanAggregator:
    """
    Trimmed Mean 聚合算法

    对每个坐标维度，去掉最大和最小的各 f 个值后取平均。
    这能有效抵御最多 f 个恶意节点发送极端值的攻击。

    算法:
    对每个坐标 d:
    1. 收集所有更新在第 d 维的值
    2. 排序
    3. 去掉最大的 f 个和最小的 f 个
    4. 对剩余值取平均

    要求: n >= 4f + 1

    参考文献:
    Y. Chen et al., "Distributed machine learning with Byzantine gradient descent",
    arXiv:1705.05113.
    """

    def __init__(self, max_byzantine: int = 1, trim_count: Optional[int] = None) -> None:
        if max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0，当前值: {max_byzantine}")
        self._f = max_byzantine
        self._trim_count = trim_count  # 如果指定，覆盖 f

    def aggregate(self, updates: List[List[float]]) -> List[float]:
        """
        使用 Trimmed Mean 算法聚合更新

        Args:
            updates: 客户端更新列表

        Returns:
            聚合后的更新向量
        """
        if not updates:
            return []

        n = len(updates)
        f = self._trim_count if self._trim_count is not None else self._f

        # Trimmed Mean 要求 n > 2f
        if n <= 2 * f:
            logger.warning(
                f"Trimmed Mean 需要至少 {2 * f + 1} 个更新，"
                f"当前只有 {n} 个，回退到中位数"
            )
            return self._fallback_median(updates)

        padded, dim = _pad_vectors(updates)
        result: List[float] = []

        for d in range(dim):
            # 收集该坐标的所有值并排序
            values = sorted(u[d] for u in padded)
            # 去掉最大和最小的各 f 个
            trimmed = values[f : n - f]
            # 取平均
            result.append(sum(trimmed) / len(trimmed))

        return result

    def _fallback_median(self, updates: List[List[float]]) -> List[float]:
        """回退到坐标中位数"""
        padded, dim = _pad_vectors(updates)
        result: List[float] = []
        for d in range(dim):
            values = sorted(u[d] for u in padded)
            n = len(values)
            if n % 2 == 0:
                result.append((values[n // 2 - 1] + values[n // 2]) / 2.0)
            else:
                result.append(values[n // 2])
        return result


# ============== Median 聚合器 ==============

class MedianAggregator:
    """
    坐标中位数聚合器

    对每个坐标维度独立计算中位数。
    这是最简单的拜占庭鲁棒聚合方法之一，能容忍最多
    接近一半的恶意节点。

    算法:
    对每个坐标 d:
    1. 收集所有更新在第 d 维的值
    2. 排序
    3. 取中位数

    中位数聚合对维度间相关性不敏感，适合高维场景。

    参考文献:
    Y. Chen et al., "Distributed machine learning with Byzantine gradient descent",
    arXiv:1705.05113.
    """

    def __init__(self, max_byzantine: int = 1) -> None:
        if max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0，当前值: {max_byzantine}")
        self._f = max_byzantine

    def aggregate(self, updates: List[List[float]]) -> List[float]:
        """
        使用坐标中位数聚合更新

        Args:
            updates: 客户端更新列表

        Returns:
            聚合后的更新向量
        """
        if not updates:
            return []

        padded, dim = _pad_vectors(updates)
        n = len(padded)
        result: List[float] = []

        for d in range(dim):
            values = sorted(u[d] for u in padded)
            if n % 2 == 0:
                result.append((values[n // 2 - 1] + values[n // 2]) / 2.0)
            else:
                result.append(values[n // 2])

        return result


# ============== Bulyan 聚合器 ==============

class BulyanAggregator:
    """
    Bulyan 聚合算法

    Bulyan 是两阶段算法，结合了 Krum 的选择能力和 Trimmed Mean 的鲁棒性:

    阶段 1: 使用 Krum 选择 2f 个候选更新
    阶段 2: 对候选更新执行 Trimmed Mean（每端去掉 f 个）

    Bulyan 能同时抵御方向攻击和幅值攻击。

    要求: n >= 4f + 3

    参考文献:
    E. Guerraoui et al., "The Hidden Peril of Byzantine Attacks on Machine Learning",
    NeurIPS 2018.
    """

    def __init__(self, max_byzantine: int = 1) -> None:
        if max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0，当前值: {max_byzantine}")
        self._f = max_byzantine

    def aggregate(self, updates: List[List[float]]) -> List[float]:
        """
        使用 Bulyan 算法聚合更新

        Args:
            updates: 客户端更新列表

        Returns:
            聚合后的更新向量
        """
        if not updates:
            return []

        n = len(updates)
        f = self._f

        # Bulyan 要求 n >= 4f + 3
        if n < 4 * f + 3:
            logger.warning(
                f"Bulyan 需要至少 {4 * f + 3} 个更新，"
                f"当前只有 {n} 个，回退到 Trimmed Mean"
            )
            tm = TrimmedMeanAggregator(f)
            return tm.aggregate(updates)

        # 阶段 1: 使用 Krum 选择 2f 个候选
        krum = KrumAggregator(f)
        padded, dim = _pad_vectors(updates)
        scores = krum._compute_krum_scores(padded)

        # 选择 score 最小的 2f 个
        num_candidates = 2 * f
        indexed_scores = [(scores[i], i) for i in range(n)]
        indexed_scores.sort(key=lambda x: x[0])
        candidate_indices = [idx for _, idx in indexed_scores[:num_candidates]]
        candidates = [padded[i] for i in candidate_indices]

        # 阶段 2: 对候选执行 Trimmed Mean（每端去掉 f 个）
        # 但候选只有 2f 个，去掉 f 个后只剩 f 个
        # 标准 Bulyan: 对 2f 个候选做 coordinate-wise trimmed mean，每端去掉 f 个
        # 实际上 2f - 2f = 0，所以需要调整
        # 正确做法: 选择 n - 2f 个候选（Krum 选出足够多的候选）
        # 重新实现: 使用 Multi-Krum 选择 n - 2f 个候选
        num_select = n - 2 * f
        if num_select < 2 * f + 1:
            # 候选太少，直接用选出的候选做 Trimmed Mean
            num_select = 2 * f

        selected_indices = [idx for _, idx in indexed_scores[:num_select]]
        selected = [padded[i] for i in selected_indices]

        # 对选中的更新做 Trimmed Mean
        tm = TrimmedMeanAggregator(f)
        return tm.aggregate(selected)


# ============== FlTrust 聚合器 ==============

class FlTrustAggregator:
    """
    FlTrust (Federated Learning with Trust) 聚合算法

    基于信任评分的加权聚合。服务器维护一个根信任模型，
    根据每个客户端更新与根模型的余弦相似度计算信任权重。

    算法:
    1. 服务器维护一个根模型（初始为全局模型）
    2. 对每个客户端更新，计算与根模型的余弦相似度
    3. 将相似度作为信任权重（过滤掉相似度为负的更新）
    4. 使用信任权重进行加权平均

    参考文献:
    H. Cai et al., "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping",
    NeurIPS 2021.
    """

    def __init__(
        self,
        max_byzantine: int = 1,
        trust_decay: float = 0.9,
        root_update: bool = True,
    ) -> None:
        if max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0")
        if not (0 < trust_decay <= 1.0):
            raise ValueError(f"trust_decay 必须在 (0, 1] 之间，当前值: {trust_decay}")
        self._f = max_byzantine
        self._trust_decay = trust_decay
        self._root_update = root_update
        self._root_model: Optional[List[float]] = None
        self._trust_scores: Dict[str, float] = {}

    @property
    def root_model(self) -> Optional[List[float]]:
        return self._root_model

    def set_root_model(self, model: List[float]) -> None:
        """设置根信任模型"""
        self._root_model = list(model)

    def aggregate(
        self,
        updates: List[List[float]],
        client_ids: Optional[List[str]] = None,
    ) -> List[float]:
        """
        使用 FlTrust 算法聚合更新

        Args:
            updates: 客户端更新列表
            client_ids: 客户端ID列表（可选，用于跟踪信任评分）

        Returns:
            聚合后的更新向量
        """
        if not updates:
            return []

        n = len(updates)
        padded, dim = _pad_vectors(updates)

        # 如果没有根模型，使用均值作为初始根模型
        if self._root_model is None:
            self._root_model = _mean_vectors(padded)

        # 确保根模型维度匹配
        root = list(self._root_model)
        if len(root) < dim:
            root.extend([0.0] * (dim - len(root)))
        elif len(root) > dim:
            root = root[:dim]

        # 计算每个更新的信任权重
        weights: List[float] = []
        for i in range(n):
            sim = _cosine_similarity(root, padded[i])
            # 只保留非负相似度
            weight = max(0.0, sim)
            weights.append(weight)

        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            # 所有权重为 0，回退到均匀平均
            logger.warning("所有信任权重为 0，回退到均匀平均")
            return _mean_vectors(padded)

        # 加权平均
        result: List[float] = []
        for d in range(dim):
            weighted_sum = sum(weights[i] * padded[i][d] for i in range(n))
            result.append(weighted_sum / total_weight)

        # 更新根模型
        if self._root_update:
            # 根模型向聚合结果移动
            new_root: List[float] = []
            for d in range(dim):
                new_root.append(self._trust_decay * root[d] + (1 - self._trust_decay) * result[d])
            self._root_model = new_root

        # 记录信任评分
        if client_ids is not None:
            for i, cid in enumerate(client_ids):
                if i < len(weights):
                    self._trust_scores[cid] = weights[i]

        return result

    def get_trust_scores(self) -> Dict[str, float]:
        """获取所有客户端的信任评分"""
        return dict(self._trust_scores)

    def get_low_trust_clients(self, threshold: float = 0.1) -> List[str]:
        """
        获取低信任度的客户端

        Args:
            threshold: 信任度阈值

        Returns:
            信任度低于阈值的客户端ID列表
        """
        return [
            cid for cid, score in self._trust_scores.items()
            if score < threshold
        ]

    def reset_trust(self) -> None:
        """重置信任评分和根模型"""
        self._root_model = None
        self._trust_scores.clear()


# ============== 拜占庭节点检测器 ==============

class ByzantineDetector:
    """
    拜占庭节点检测器

    基于余弦相似度和更新范数检测可能的拜占庭节点。

    检测策略:
    1. 余弦相似度检测: 计算每个更新与聚合均值的余弦相似度，
       相似度低于阈值的更新被标记为可疑
    2. 范数异常检测: 计算每个更新的范数，范数偏离中位数过远的
       更新被标记为可疑
    3. 综合评分: 结合余弦相似度和范数异常度给出综合评分

    注意: 检测器不直接排除节点，而是提供可疑度评分供聚合器参考。
    """

    def __init__(
        self,
        max_byzantine: int = 1,
        similarity_threshold: float = 0.0,
        norm_threshold: float = 3.0,
    ) -> None:
        """
        Args:
            max_byzantine: 预期最大恶意节点数
            similarity_threshold: 余弦相似度阈值（低于此值标记为可疑）
            norm_threshold: 范数异常阈值（标准差倍数）
        """
        if max_byzantine < 0:
            raise ValueError(f"max_byzantine 必须 >= 0")
        self._f = max_byzantine
        self._sim_threshold = similarity_threshold
        self._norm_threshold = norm_threshold

    def detect(
        self,
        updates: List[List[float]],
        client_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        检测拜占庭节点

        Args:
            updates: 客户端更新列表
            client_ids: 客户端ID列表（可选）

        Returns:
            检测结果字典，包含:
            - suspicious_indices: 可疑更新索引列表
            - suspicious_ids: 可疑客户端ID列表
            - similarity_scores: 余弦相似度列表
            - norm_scores: 范数异常度列表
            - combined_scores: 综合评分列表
        """
        if not updates:
            return {
                "suspicious_indices": [],
                "suspicious_ids": [],
                "similarity_scores": [],
                "norm_scores": [],
                "combined_scores": [],
            }

        n = len(updates)
        padded, dim = _pad_vectors(updates)

        # 计算聚合均值
        mean = _mean_vectors(padded)

        # 计算余弦相似度
        sim_scores: List[float] = []
        for i in range(n):
            sim = _cosine_similarity(mean, padded[i])
            sim_scores.append(sim)

        # 计算范数
        norms = [_vector_norm(u) for u in padded]
        median_norm = sorted(norms)[n // 2]

        # 计算范数异常度（MAD-based）
        deviations = [abs(norm - median_norm) for norm in norms]
        mad = sorted(deviations)[n // 2]  # Median Absolute Deviation
        if mad == 0:
            mad = 1e-8  # 防止除零

        norm_scores: List[float] = []
        for i in range(n):
            # 标准化范数异常度
            score = abs(norms[i] - median_norm) / (mad * self._norm_threshold)
            norm_scores.append(min(score, 1.0))  # 限制在 [0, 1]

        # 综合评分（余弦相似度 + 范数异常度的加权组合）
        combined_scores: List[float] = []
        for i in range(n):
            # 相似度越低越可疑，范数异常度越高越可疑
            sim_factor = max(0.0, 1.0 - sim_scores[i]) / 2.0  # 归一化到 [0, 0.5]
            norm_factor = norm_scores[i] / 2.0  # 归一化到 [0, 0.5]
            combined_scores.append(sim_factor + norm_factor)

        # 确定可疑节点
        suspicious_indices: List[int] = []
        for i in range(n):
            if sim_scores[i] < self._sim_threshold:
                suspicious_indices.append(i)
            elif combined_scores[i] > 0.6:
                suspicious_indices.append(i)

        # 限制可疑节点数量不超过 f
        if len(suspicious_indices) > self._f:
            # 按综合评分排序，只保留最可疑的 f 个
            suspicious_indices.sort(key=lambda idx: combined_scores[idx], reverse=True)
            suspicious_indices = suspicious_indices[:self._f]

        suspicious_ids: List[str] = []
        if client_ids is not None:
            suspicious_ids = [
                client_ids[i] for i in suspicious_indices if i < len(client_ids)
            ]

        return {
            "suspicious_indices": suspicious_indices,
            "suspicious_ids": suspicious_ids,
            "similarity_scores": sim_scores,
            "norm_scores": norm_scores,
            "combined_scores": combined_scores,
        }

    def filter_updates(
        self,
        updates: List[List[float]],
        client_ids: Optional[List[str]] = None,
    ) -> Tuple[List[List[float]], List[str]]:
        """
        过滤掉可疑更新

        Args:
            updates: 客户端更新列表
            client_ids: 客户端ID列表

        Returns:
            (过滤后的更新列表, 过滤后的客户端ID列表)
        """
        result = self.detect(updates, client_ids)
        suspicious = set(result["suspicious_indices"])

        filtered_updates = [u for i, u in enumerate(updates) if i not in suspicious]
        filtered_ids: List[str] = []
        if client_ids is not None:
            filtered_ids = [
                cid for i, cid in enumerate(client_ids) if i not in suspicious
            ]

        return (filtered_updates, filtered_ids)


# ============== 拜占庭鲁棒聚合器（统一入口） ==============

class ByzantineRobustAggregator:
    """
    拜占庭鲁棒聚合器 - 统一入口

    根据配置选择合适的聚合策略，并提供统一的接口。

    支持的聚合方法:
    - krum: Krum 算法
    - multi_krum: Multi-Krum 算法
    - trimmed_mean: Trimmed Mean
    - median: 坐标中位数
    - bulyan: Bulyan 算法
    - fltrust: FlTrust 信任聚合
    - mean: 简单平均（无鲁棒性，用于对比）

    使用示例:
        config = ByzantineConfig(max_byzantine=2, aggregation_method="bulyan")
        aggregator = ByzantineRobustAggregator(config)
        result = aggregator.aggregate(updates)
    """

    def __init__(self, config: Optional[ByzantineConfig] = None) -> None:
        self._config = config or ByzantineConfig()
        self._f = self._config.max_byzantine
        self._method = self._config.aggregation_method

        # 初始化各聚合器
        self._krum = KrumAggregator(self._f)
        self._multi_krum = MultiKrumAggregator(
            self._f, self._config.krum_candidates
        )
        self._trimmed_mean = TrimmedMeanAggregator(
            self._f, self._config.trim_count
        )
        self._median = MedianAggregator(self._f)
        self._bulyan = BulyanAggregator(self._f)
        self._fltrust = FlTrustAggregator(
            self._f,
            self._config.trust_decay,
            self._config.fltrust_root_update,
        )
        self._detector = ByzantineDetector(
            self._f,
            similarity_threshold=self._config.detection_threshold,
        )

        # 聚合历史
        self._history: List[Dict[str, Any]] = []

    @property
    def config(self) -> ByzantineConfig:
        return self._config

    def aggregate(
        self,
        updates: List[List[float]],
        client_ids: Optional[List[str]] = None,
        use_detection: bool = True,
    ) -> List[float]:
        """
        根据配置的聚合方法聚合更新

        Args:
            updates: 客户端更新列表
            client_ids: 客户端ID列表（可选）
            use_detection: 是否在聚合前执行拜占庭检测

        Returns:
            聚合后的更新向量
        """
        if not updates:
            return []

        # 移除空更新
        valid_updates = [u for u in updates if u]
        if not valid_updates:
            return []

        # 可选: 先执行拜占庭检测
        detection_result: Optional[Dict[str, Any]] = None
        if use_detection and len(valid_updates) > 2 * self._f + 1:
            detection_result = self._detector.detect(valid_updates, client_ids)
            suspicious = set(detection_result["suspicious_indices"])
            if suspicious:
                logger.info(
                    f"拜占庭检测发现 {len(suspicious)} 个可疑更新: "
                    f"{detection_result['suspicious_indices']}"
                )

        # 根据方法选择聚合器
        method = self._method

        if method == "krum":
            result = self._krum.aggregate(valid_updates)
        elif method == "multi_krum":
            result = self._multi_krum.aggregate(valid_updates)
        elif method == "trimmed_mean":
            result = self._trimmed_mean.aggregate(valid_updates)
        elif method == "median":
            result = self._median.aggregate(valid_updates)
        elif method == "bulyan":
            result = self._bulyan.aggregate(valid_updates)
        elif method == "fltrust":
            result = self._fltrust.aggregate(valid_updates, client_ids)
        elif method == "mean":
            result = _mean_vectors(valid_updates)
        else:
            logger.warning(f"未知聚合方法 '{method}'，回退到 Trimmed Mean")
            result = self._trimmed_mean.aggregate(valid_updates)

        # 记录历史
        record: Dict[str, Any] = {
            "method": method,
            "num_updates": len(valid_updates),
            "result_norm": _vector_norm(result) if result else 0.0,
        }
        if detection_result is not None:
            record["suspicious_count"] = len(detection_result["suspicious_indices"])
        self._history.append(record)

        return result

    def detect(
        self,
        updates: List[List[float]],
        client_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        执行拜占庭检测（不执行聚合）

        Args:
            updates: 客户端更新列表
            client_ids: 客户端ID列表

        Returns:
            检测结果
        """
        return self._detector.detect(updates, client_ids)

    def set_root_model(self, model: List[float]) -> None:
        """设置 FlTrust 的根模型"""
        self._fltrust.set_root_model(model)

    def get_trust_scores(self) -> Dict[str, float]:
        """获取 FlTrust 的信任评分"""
        return self._fltrust.get_trust_scores()

    def get_history(self) -> List[Dict[str, Any]]:
        """获取聚合历史"""
        return list(self._history)

    def get_summary(self) -> Dict[str, Any]:
        """获取聚合器摘要信息"""
        total_suspicious = sum(
            h.get("suspicious_count", 0) for h in self._history
        )
        return {
            "method": self._method,
            "max_byzantine": self._f,
            "total_rounds": len(self._history),
            "total_suspicious_detected": total_suspicious,
            "config": self._config.to_dict(),
        }

    def reset(self) -> None:
        """重置聚合器状态"""
        self._history.clear()
        self._fltrust.reset_trust()


# ============== 模块入口 ==============

if __name__ == "__main__":
    print("=== 拜占庭鲁棒聚合器演示 ===\n")

    # 构造测试数据: 5个正常更新 + 2个恶意更新
    random.seed(42)
    true_update = [0.5, -0.3, 0.2, -0.1, 0.4]
    normal_updates = [
        [v + random.gauss(0, 0.05) for v in true_update] for _ in range(5)
    ]
    byzantine_updates = [
        [10.0, -5.0, 8.0, -3.0, 12.0],   # 幅值攻击
        [-10.0, 5.0, -8.0, 3.0, -12.0],   # 反向攻击
    ]
    all_updates = normal_updates + byzantine_updates

    print(f"正常更新数量: {len(normal_updates)}")
    print(f"恶意更新数量: {len(byzantine_updates)}")
    print(f"总更新数量:   {len(all_updates)}")
    print(f"真实更新:     {true_update}")
    print()

    # 1. Krum
    print("1. Krum 聚合:")
    krum = KrumAggregator(max_byzantine=2)
    result = krum.aggregate(all_updates)
    print(f"   结果: {[f'{v:.4f}' for v in result]}")

    # 2. Multi-Krum
    print("\n2. Multi-Krum 聚合:")
    mk = MultiKrumAggregator(max_byzantine=2, num_candidates=3)
    result = mk.aggregate(all_updates)
    print(f"   结果: {[f'{v:.4f}' for v in result]}")

    # 3. Trimmed Mean
    print("\n3. Trimmed Mean 聚合:")
    tm = TrimmedMeanAggregator(max_byzantine=2)
    result = tm.aggregate(all_updates)
    print(f"   结果: {[f'{v:.4f}' for v in result]}")

    # 4. Median
    print("\n4. Median 聚合:")
    med = MedianAggregator(max_byzantine=2)
    result = med.aggregate(all_updates)
    print(f"   结果: {[f'{v:.4f}' for v in result]}")

    # 5. Bulyan
    print("\n5. Bulyan 聚合:")
    bulyan = BulyanAggregator(max_byzantine=1)
    result = bulyan.aggregate(all_updates)
    print(f"   结果: {[f'{v:.4f}' for v in result]}")

    # 6. FlTrust
    print("\n6. FlTrust 聚合:")
    fltrust = FlTrustAggregator(max_byzantine=2)
    result = fltrust.aggregate(all_updates)
    print(f"   结果: {[f'{v:.4f}' for v in result]}")

    # 7. 拜占庭检测
    print("\n7. 拜占庭检测:")
    detector = ByzantineDetector(max_byzantine=2, similarity_threshold=0.0)
    detection = detector.detect(all_updates)
    print(f"   可疑索引: {detection['suspicious_indices']}")
    print(f"   余弦相似度: {[f'{s:.4f}' for s in detection['similarity_scores']]}")
    print(f"   综合评分:   {[f'{s:.4f}' for s in detection['combined_scores']]}")

    # 8. 统一入口
    print("\n8. 统一聚合器:")
    config = ByzantineConfig(max_byzantine=2, aggregation_method="trimmed_mean")
    aggregator = ByzantineRobustAggregator(config)
    result = aggregator.aggregate(all_updates)
    print(f"   方法: {config.aggregation_method}")
    print(f"   结果: {[f'{v:.4f}' for v in result]}")
    print(f"   摘要: {aggregator.get_summary()}")

    print("\n=== 演示完成 ===")
