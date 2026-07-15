"""
后门检测 - 检测模型中的后门
"""
import math
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class BackdoorType(Enum):
    """后门类型"""
    TRIGGER_BASED = "trigger_based"
    CLEAN_LABEL = "clean_label"
    INVISIBLE = "invisible"
    PHYSICAL = "physical"
    SEMANTIC = "semantic"


@dataclass
class TriggerCandidate:
    """触发器候选"""
    pattern: List[Any]
    position: Tuple[int, ...]
    size: Tuple[int, ...]
    confidence: float
    backdoor_type: BackdoorType


@dataclass
class BackdoorDetectionResult:
    """后门检测结果"""
    is_detected: bool
    backdoor_type: Optional[BackdoorType]
    trigger: Optional[TriggerCandidate]
    target_label: Optional[int]
    confidence: float
    affected_samples: int
    details: Dict[str, Any] = field(default_factory=dict)


class BackdoorDetector:
    """后门检测器"""
    
    def __init__(self):
        self._detection_methods = [
            self._detect_by_activation_clustering,
            self._detect_by_spectral_signature,
            self._detect_by_neuron_inspection,
        ]
        self._threshold = 0.5
    
    def detect(
        self,
        model_predict: callable,
        clean_data: List[Tuple[List[float], int]],
        test_data: List[Tuple[List[float], int]]
    ) -> BackdoorDetectionResult:
        """检测后门"""
        results = []
        
        for method in self._detection_methods:
            try:
                result = method(model_predict, clean_data, test_data)
                results.append(result)
            except Exception:
                pass
        
        if not results:
            return BackdoorDetectionResult(
                is_detected=False,
                backdoor_type=None,
                trigger=None,
                target_label=None,
                confidence=0.0,
                affected_samples=0
            )
        
        # 选择置信度最高的结果
        best_result = max(results, key=lambda r: r.confidence)
        
        return best_result
    
    def _detect_by_activation_clustering(
        self,
        model_predict: callable,
        clean_data: List[Tuple[List[float], int]],
        test_data: List[Tuple[List[float], int]]
    ) -> BackdoorDetectionResult:
        """通过激活聚类检测"""
        # 提取激活模式（简化）
        clean_patterns = [self._extract_pattern(x) for x, _ in clean_data]
        test_patterns = [self._extract_pattern(x) for x, _ in test_data]
        
        # 计算聚类
        clean_clusters = self._simple_clustering(clean_patterns)
        test_clusters = self._simple_clustering(test_patterns)
        
        # 检测异常聚类
        anomaly_score = self._compare_clusters(clean_clusters, test_clusters)
        
        is_detected = anomaly_score > self._threshold
        
        return BackdoorDetectionResult(
            is_detected=is_detected,
            backdoor_type=BackdoorType.TRIGGER_BASED if is_detected else None,
            trigger=None,
            target_label=None,
            confidence=anomaly_score,
            affected_samples=len(test_data) if is_detected else 0,
            details={"method": "activation_clustering"}
        )
    
    def _detect_by_spectral_signature(
        self,
        model_predict: callable,
        clean_data: List[Tuple[List[float], int]],
        test_data: List[Tuple[List[float], int]]
    ) -> BackdoorDetectionResult:
        """通过频谱签名检测"""
        # 计算频谱特征
        clean_spectrum = self._compute_spectrum(clean_data)
        test_spectrum = self._compute_spectrum(test_data)
        
        # 检测异常
        spectral_diff = self._spectral_distance(clean_spectrum, test_spectrum)
        
        is_detected = spectral_diff > self._threshold
        
        return BackdoorDetectionResult(
            is_detected=is_detected,
            backdoor_type=BackdoorType.INVISIBLE if is_detected else None,
            trigger=None,
            target_label=None,
            confidence=spectral_diff,
            affected_samples=len(test_data) if is_detected else 0,
            details={"method": "spectral_signature"}
        )
    
    def _detect_by_neuron_inspection(
        self,
        model_predict: callable,
        clean_data: List[Tuple[List[float], int]],
        test_data: List[Tuple[List[float], int]]
    ) -> BackdoorDetectionResult:
        """通过神经元检测"""
        # 检测休眠神经元
        dormant_neurons = self._find_dormant_neurons(clean_data, test_data)
        
        # 检测高激活神经元
        high_activation = self._find_high_activation_neurons(clean_data, test_data)
        
        anomaly_score = (len(dormant_neurons) + len(high_activation)) / max(len(test_data), 1)
        anomaly_score = min(1.0, anomaly_score * 10)
        
        is_detected = anomaly_score > self._threshold
        
        return BackdoorDetectionResult(
            is_detected=is_detected,
            backdoor_type=BackdoorType.CLEAN_LABEL if is_detected else None,
            trigger=None,
            target_label=None,
            confidence=anomaly_score,
            affected_samples=len(dormant_neurons) + len(high_activation),
            details={
                "method": "neuron_inspection",
                "dormant_neurons": len(dormant_neurons),
                "high_activation_neurons": len(high_activation)
            }
        )
    
    def _extract_pattern(self, data: List[float]) -> List[float]:
        """提取模式"""
        # 简化：返回数据的统计特征
        if not data:
            return [0, 0, 0]
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        max_val = max(data)
        
        return [mean, math.sqrt(variance), max_val]
    
    def _simple_clustering(self, patterns: List[List[float]], k: int = 3) -> List[List[List[float]]]:
        """简单聚类"""
        if not patterns:
            return []
        
        # K-means简化实现
        # 随机初始化中心
        centers = random.sample(patterns, min(k, len(patterns)))
        
        for _ in range(10):  # 迭代次数
            clusters = [[] for _ in range(k)]
            
            for pattern in patterns:
                # 找最近的中心
                distances = [
                    sum((p - c) ** 2 for p, c in zip(pattern, center))
                    for center in centers
                ]
                nearest = distances.index(min(distances))
                clusters[nearest].append(pattern)
            
            # 更新中心
            for i, cluster in enumerate(clusters):
                if cluster:
                    centers[i] = [
                        sum(p[j] for p in cluster) / len(cluster)
                        for j in range(len(cluster[0]))
                    ]
        
        return clusters
    
    def _compare_clusters(
        self,
        clean_clusters: List[List[List[float]]],
        test_clusters: List[List[List[float]]]
    ) -> float:
        """比较聚类"""
        if not clean_clusters or not test_clusters:
            return 0.0
        
        # 计算聚类大小差异
        clean_sizes = [len(c) for c in clean_clusters if c]
        test_sizes = [len(c) for c in test_clusters if c]
        
        if not clean_sizes or not test_sizes:
            return 0.0
        
        # 计算差异分数
        clean_total = sum(clean_sizes)
        test_total = sum(test_sizes)
        
        # 检查是否有异常大的聚类
        max_test_ratio = max(test_sizes) / test_total if test_total > 0 else 0
        max_clean_ratio = max(clean_sizes) / clean_total if clean_total > 0 else 0
        
        return abs(max_test_ratio - max_clean_ratio)
    
    def _compute_spectrum(self, data: List[Tuple[List[float], int]]) -> List[float]:
        """计算频谱"""
        if not data:
            return []
        
        # 简化：计算数据的协方差矩阵特征值
        features = [x for x, _ in data]
        if not features or not features[0]:
            return []
        
        # 简化为方差
        n_features = len(features[0])
        variances = []
        
        for i in range(n_features):
            col = [f[i] if i < len(f) else 0 for f in features]
            mean = sum(col) / len(col)
            var = sum((x - mean) ** 2 for x in col) / len(col)
            variances.append(var)
        
        return sorted(variances, reverse=True)
    
    def _spectral_distance(self, s1: List[float], s2: List[float]) -> float:
        """频谱距离"""
        if not s1 or not s2:
            return 0.0
        
        # L2距离
        min_len = min(len(s1), len(s2))
        distance = math.sqrt(sum((s1[i] - s2[i]) ** 2 for i in range(min_len)))
        
        # 归一化
        norm = math.sqrt(sum(s ** 2 for s in s1[:min_len]))
        if norm > 0:
            return distance / norm
        return 0.0
    
    def _find_dormant_neurons(
        self,
        clean_data: List[Tuple[List[float], int]],
        test_data: List[Tuple[List[float], int]]
    ) -> List[int]:
        """查找休眠神经元"""
        # 简化：查找在测试数据中激活但在干净数据中不激活的特征
        dormant = []
        
        clean_features = [x for x, _ in clean_data]
        test_features = [x for x, _ in test_data]
        
        if not clean_features or not test_features:
            return dormant
        
        n_features = min(len(clean_features[0]), len(test_features[0]))
        
        for i in range(n_features):
            clean_act = sum(1 for f in clean_features if i < len(f) and abs(f[i]) > 0.1)
            test_act = sum(1 for f in test_features if i < len(f) and abs(f[i]) > 0.1)
            
            clean_ratio = clean_act / len(clean_features)
            test_ratio = test_act / len(test_features)
            
            # 如果在测试数据中激活但在干净数据中不激活
            if test_ratio > 0.5 and clean_ratio < 0.1:
                dormant.append(i)
        
        return dormant
    
    def _find_high_activation_neurons(
        self,
        clean_data: List[Tuple[List[float], int]],
        test_data: List[Tuple[List[float], int]]
    ) -> List[int]:
        """查找高激活神经元"""
        high_activation = []
        
        clean_features = [x for x, _ in clean_data]
        test_features = [x for x, _ in test_data]
        
        if not clean_features or not test_features:
            return high_activation
        
        n_features = min(len(clean_features[0]), len(test_features[0]))
        
        for i in range(n_features):
            clean_vals = [f[i] if i < len(f) else 0 for f in clean_features]
            test_vals = [f[i] if i < len(f) else 0 for f in test_features]
            
            clean_mean = sum(abs(v) for v in clean_vals) / len(clean_vals)
            test_mean = sum(abs(v) for v in test_vals) / len(test_vals)
            
            # 如果测试数据激活显著高于干净数据
            if test_mean > clean_mean * 3 and test_mean > 0.5:
                high_activation.append(i)
        
        return high_activation
    
    def scan_for_triggers(
        self,
        model_predict: callable,
        data: List[Tuple[List[float], int]],
        num_candidates: int = 100
    ) -> List[TriggerCandidate]:
        """扫描触发器"""
        candidates = []
        
        # 生成候选触发器
        for _ in range(num_candidates):
            # 随机位置和大小
            position = (random.randint(0, 10), random.randint(0, 10))
            size = (random.randint(1, 3), random.randint(1, 3))
            
            # 随机模式
            pattern = [random.random() for _ in range(size[0] * size[1])]
            
            # 测试触发器
            confidence = self._test_trigger(model_predict, data, pattern, position)
            
            if confidence > 0.3:
                candidates.append(TriggerCandidate(
                    pattern=pattern,
                    position=position,
                    size=size,
                    confidence=confidence,
                    backdoor_type=BackdoorType.TRIGGER_BASED
                ))
        
        # 按置信度排序
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates[:10]  # 返回前10个
    
    def _test_trigger(
        self,
        model_predict: callable,
        data: List[Tuple[List[float], int]],
        pattern: List[float],
        position: Tuple[int, int]
    ) -> float:
        """测试触发器"""
        # 简化：返回随机置信度
        return random.random() * 0.5
