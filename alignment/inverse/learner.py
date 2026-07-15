"""
逆向宪章学习 - 从人类偏好数据学习宪章规则

实现从偏好数据中自动发现、生成和验证宪章规则的完整流程。
所有实现使用纯Python，不依赖任何外部库。
"""

import math
import random
import hashlib
import json
import time
import re
import copy
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum


# ==================== 基础数据模型 ====================

@dataclass
class ConstitutionalRule:
    """宪章规则"""
    rule_id: str
    description: str
    category: str = "general"
    severity: str = "medium"
    rationale: str = ""
    examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Constitution:
    """宪章"""
    constitution_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    rules: List[ConstitutionalRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_rule(self, rule: ConstitutionalRule) -> None:
        self.rules.append(rule)
        self.updated_at = time.time()

    def get_enabled_rules(self) -> List[ConstitutionalRule]:
        return [r for r in self.rules if r.enabled]


# ==================== 偏好数据模型 ====================

@dataclass
class PreferenceData:
    """偏好数据 - 人类偏好标注"""
    prompt: str
    chosen_response: str
    rejected_response: str
    preference_label: int  # 1=chosen better, 0=rejected better
    category: str = "general"
    source: str = "human"
    metadata: dict = field(default_factory=dict)


# ==================== Pattern 数据类 ====================

@dataclass
class Pattern:
    """发现的模式"""
    pattern_id: str
    feature_signature: List[float]
    representative_features: Dict[str, float]
    member_count: int
    category: str = "general"
    description: str = ""


# ==================== PreferenceDataset ====================

class PreferenceDataset:
    """偏好数据集"""

    def __init__(self):
        self._data: List[PreferenceData] = []

    def add(self, data: PreferenceData) -> None:
        """添加一条偏好数据"""
        self._data.append(data)

    def add_batch(self, batch: List[PreferenceData]) -> None:
        """批量添加偏好数据"""
        self._data.extend(batch)

    def remove(self, index: int) -> PreferenceData:
        """移除指定索引的数据"""
        if 0 <= index < len(self._data):
            return self._data.pop(index)
        raise IndexError(f"Index {index} out of range for dataset of size {len(self._data)}")

    def get_batch(self, batch_size: int) -> List[PreferenceData]:
        """获取一个批次的数据"""
        if batch_size >= len(self._data):
            return list(self._data)
        indices = random.sample(range(len(self._data)), batch_size)
        return [self._data[i] for i in indices]

    def shuffle(self) -> None:
        """随机打乱数据"""
        random.shuffle(self._data)

    def split(self, ratio: float) -> Tuple["PreferenceDataset", "PreferenceDataset"]:
        """按比例分割数据集"""
        shuffled = list(self._data)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * ratio)
        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]

        train_set = PreferenceDataset()
        train_set._data = train_data

        val_set = PreferenceDataset()
        val_set._data = val_data

        return train_set, val_set

    def get_category_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = Counter(p.category for p in self._data)
        return dict(distribution)

    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        if not self._data:
            return {
                "total_samples": 0,
                "categories": {},
                "avg_prompt_length": 0,
                "avg_chosen_length": 0,
                "avg_rejected_length": 0,
                "label_distribution": {},
            }

        prompt_lengths = [len(p.prompt) for p in self._data]
        chosen_lengths = [len(p.chosen_response) for p in self._data]
        rejected_lengths = [len(p.rejected_response) for p in self._data]

        label_dist = Counter(p.preference_label for p in self._data)

        return {
            "total_samples": len(self._data),
            "categories": self.get_category_distribution(),
            "avg_prompt_length": round(sum(prompt_lengths) / len(prompt_lengths), 2),
            "avg_chosen_length": round(sum(chosen_lengths) / len(chosen_lengths), 2),
            "avg_rejected_length": round(sum(rejected_lengths) / len(rejected_lengths), 2),
            "label_distribution": dict(label_dist),
            "sources": dict(Counter(p.source for p in self._data)),
        }

    def export_json(self) -> str:
        """导出为JSON字符串"""
        export_data = []
        for p in self._data:
            export_data.append({
                "prompt": p.prompt,
                "chosen_response": p.chosen_response,
                "rejected_response": p.rejected_response,
                "preference_label": p.preference_label,
                "category": p.category,
                "source": p.source,
                "metadata": p.metadata,
            })
        return json.dumps(export_data, ensure_ascii=False, indent=2)

    def import_json(self, json_str: str) -> None:
        """从JSON字符串导入"""
        data = json.loads(json_str)
        for item in data:
            self._data.append(PreferenceData(
                prompt=item["prompt"],
                chosen_response=item["chosen_response"],
                rejected_response=item["rejected_response"],
                preference_label=item.get("preference_label", 1),
                category=item.get("category", "general"),
                source=item.get("source", "imported"),
                metadata=item.get("metadata", {}),
            ))

    def balance_categories(self) -> None:
        """平衡各类别数据 - 欠采样多数类到与最少类相同数量"""
        if not self._data:
            return

        category_groups = defaultdict(list)
        for p in self._data:
            category_groups[p.category].append(p)

        min_count = min(len(group) for group in category_groups.values())

        balanced = []
        for category, group in category_groups.items():
            if len(group) > min_count:
                sampled = random.sample(group, min_count)
            else:
                sampled = group
            balanced.extend(sampled)

        random.shuffle(balanced)
        self._data = balanced

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> PreferenceData:
        return self._data[index]


# ==================== RuleCandidate ====================

@dataclass
class RuleCandidate:
    """规则候选"""
    description: str
    category: str
    patterns: List[str]
    confidence: float
    support_count: int
    violation_count: int


# ==================== FeatureExtractor ====================

class FeatureExtractor:
    """特征提取器 - 从文本中提取多种特征"""

    # 正面情感词表
    POSITIVE_WORDS = {
        "good", "great", "excellent", "wonderful", "amazing", "fantastic",
        "helpful", "kind", "friendly", "polite", "respectful", "considerate",
        "thankful", "grateful", "happy", "joyful", "pleasant", "positive",
        "beneficial", "valuable", "important", "useful", "effective",
        "accurate", "correct", "right", "proper", "appropriate", "suitable",
        "clear", "concise", "informative", "comprehensive", "thorough",
        "safe", "secure", "protected", "fair", "equitable", "just",
        "honest", "truthful", "transparent", "open", "genuine",
        "empathetic", "caring", "supportive", "encouraging", "motivating",
    }

    # 负面情感词表
    NEGATIVE_WORDS = {
        "bad", "terrible", "horrible", "awful", "worst", "poor",
        "harmful", "dangerous", "risky", "unsafe", "threatening",
        "rude", "impolite", "disrespectful", "offensive", "insulting",
        "angry", "hostile", "aggressive", "violent", "abusive",
        "negative", "unhelpful", "useless", "worthless", "pointless",
        "inaccurate", "wrong", "incorrect", "misleading", "deceptive",
        "unfair", "biased", "discriminatory", "prejudiced", "unjust",
        "dishonest", "lying", "fraudulent", "manipulative", "deceitful",
        "hateful", "toxic", "malicious", "cruel", "hurtful",
        "anxious", "worried", "fearful", "scared", "terrified",
    }

    # 技术术语
    TECHNICAL_TERMS = {
        "algorithm", "api", "database", "function", "variable", "parameter",
        "optimization", "neural", "network", "model", "training", "inference",
        "gradient", "loss", "accuracy", "precision", "recall", "f1",
        "embedding", "transformer", "attention", "encoder", "decoder",
        "classification", "regression", "clustering", "dimensionality",
        "hyperparameter", "regularization", "normalization", "activation",
        "backpropagation", "convolutional", "recurrent", "generative",
        "discriminative", "supervised", "unsupervised", "reinforcement",
        "probability", "distribution", "entropy", "likelihood", "bayesian",
        "vector", "matrix", "tensor", "scalar", "dimension", "feature",
    }

    # 被动语态标记
    PASSIVE_MARKERS = {
        "was", "were", "been", "being", "is", "are",
        "was done", "were done", "has been", "have been",
        "will be", "would be", "could be", "should be",
        "can be", "may be", "might be", "must be",
    }

    def extract_text_features(self, text: str) -> Dict[str, float]:
        """从文本中提取多种特征"""
        if not text:
            return self._empty_features()

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        features = {}

        # 1. 长度特征
        features["length"] = float(len(text))
        features["word_count"] = float(len(words))
        features["sentence_count"] = float(len(sentences))
        features["avg_sentence_length"] = (
            len(words) / len(sentences) if sentences else 0.0
        )
        features["avg_word_length"] = (
            sum(len(w) for w in words) / len(words) if words else 0.0
        )

        # 2. 情感特征
        text_lower = text.lower()
        words_lower = set(w.lower().strip(".,!?;:\"'()[]{}") for w in words)

        positive_count = sum(1 for w in words_lower if w in self.POSITIVE_WORDS)
        negative_count = sum(1 for w in words_lower if w in self.NEGATIVE_WORDS)

        total_sentiment_words = positive_count + negative_count
        features["positive_word_count"] = float(positive_count)
        features["negative_word_count"] = float(negative_count)
        features["sentiment_score"] = (
            (positive_count - negative_count) / total_sentiment_words
            if total_sentiment_words > 0 else 0.0
        )
        features["sentiment_word_density"] = (
            total_sentiment_words / len(words) if words else 0.0
        )

        # 3. 结构特征
        question_marks = text.count("?")
        exclamation_marks = text.count("!")
        features["question_ratio"] = (
            question_marks / len(sentences) if sentences else 0.0
        )
        features["exclamation_ratio"] = (
            exclamation_marks / len(sentences) if sentences else 0.0
        )

        # 4. 被动语态比例
        passive_count = 0
        for marker in self.PASSIVE_MARKERS:
            passive_count += text_lower.count(f" {marker} ")
        # 简化估计: 被动标记数 / 句子数
        features["passive_ratio"] = (
            passive_count / len(sentences) if sentences else 0.0
        )

        # 5. 技术术语密度
        tech_count = sum(1 for w in words_lower if w in self.TECHNICAL_TERMS)
        features["technical_term_density"] = (
            tech_count / len(words) if words else 0.0
        )
        features["technical_term_count"] = float(tech_count)

        # 6. 词汇多样性
        features["vocabulary_diversity"] = (
            len(words_lower) / len(words) if words else 0.0
        )

        # 7. 大写比例
        uppercase_chars = sum(1 for c in text if c.isupper())
        features["uppercase_ratio"] = (
            uppercase_chars / len(text) if text else 0.0
        )

        return features

    def extract_difference_features(
        self, text_a: str, text_b: str
    ) -> Dict[str, float]:
        """提取两段文本的特征差异"""
        features_a = self.extract_text_features(text_a)
        features_b = self.extract_text_features(text_b)

        diff_features = {}
        all_keys = set(features_a.keys()) | set(features_b.keys())

        for key in all_keys:
            val_a = features_a.get(key, 0.0)
            val_b = features_b.get(key, 0.0)
            diff_features[f"{key}_diff"] = val_a - val_b
            diff_features[f"{key}_ratio"] = (
                val_a / val_b if val_b != 0 else (1.0 if val_a > 0 else 0.0)
            )

        return diff_features

    def extract_preference_features(
        self, preference: PreferenceData
    ) -> Dict[str, float]:
        """提取偏好数据的特征(chosen vs rejected的差异)"""
        return self.extract_difference_features(
            preference.chosen_response,
            preference.rejected_response,
        )

    def _empty_features(self) -> Dict[str, float]:
        """返回空特征字典"""
        return {
            "length": 0.0,
            "word_count": 0.0,
            "sentence_count": 0.0,
            "avg_sentence_length": 0.0,
            "avg_word_length": 0.0,
            "positive_word_count": 0.0,
            "negative_word_count": 0.0,
            "sentiment_score": 0.0,
            "sentiment_word_density": 0.0,
            "question_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "passive_ratio": 0.0,
            "technical_term_density": 0.0,
            "technical_term_count": 0.0,
            "vocabulary_diversity": 0.0,
            "uppercase_ratio": 0.0,
        }


# ==================== SimpleKMeans ====================

class SimpleKMeans:
    """简易K-means聚类 - 纯Python实现"""

    def __init__(self, seed: int = 42):
        self._centroids: List[List[float]] = []
        self._labels: List[int] = []
        self._k: int = 0
        self._rng = random.Random(seed)

    def fit(
        self, data: List[List[float]], k: int, max_iter: int = 100
    ) -> List[int]:
        """
        K-means聚类

        Args:
            data: 数据点列表，每个点是一个特征向量
            k: 聚类数
            max_iter: 最大迭代次数

        Returns:
            每个数据点的聚类标签
        """
        if not data:
            return []
        if k <= 0:
            return [0] * len(data)
        if k >= len(data):
            return list(range(len(data)))

        self._k = k
        n = len(data)
        dim = len(data[0])

        # K-means++ 初始化质心
        self._centroids = self._initialize_centroids(data, k)

        self._labels = [0] * n

        for iteration in range(max_iter):
            # 分配步骤: 将每个点分配到最近的质心
            new_labels = []
            for point in data:
                closest = self._find_closest_centroid(point)
                new_labels.append(closest)

            # 检查收敛
            if new_labels == self._labels:
                break

            self._labels = new_labels

            # 更新步骤: 重新计算质心
            for cluster_id in range(k):
                cluster_points = [
                    data[i] for i in range(n) if self._labels[i] == cluster_id
                ]

                if cluster_points:
                    new_centroid = []
                    for d in range(dim):
                        avg = sum(p[d] for p in cluster_points) / len(cluster_points)
                        new_centroid.append(avg)
                    self._centroids[cluster_id] = new_centroid

        return self._labels

    def predict(self, point: List[float]) -> int:
        """预测单个点的聚类标签"""
        if not self._centroids:
            return 0
        return self._find_closest_centroid(point)

    def _find_closest_centroid(self, point: List[float]) -> int:
        """找到最近的质心"""
        min_dist = float("inf")
        closest = 0

        for i, centroid in enumerate(self._centroids):
            dist = self._euclidean_distance(point, centroid)
            if dist < min_dist:
                min_dist = dist
                closest = i

        return closest

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """计算欧氏距离"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _initialize_centroids(
        self, data: List[List[float]], k: int
    ) -> List[List[float]]:
        """K-means++ 初始化质心"""
        n = len(data)
        centroids = []

        # 随机选择第一个质心
        first_idx = self._rng.randint(0, n - 1)
        centroids.append(list(data[first_idx]))

        for _ in range(1, k):
            # 计算每个点到最近质心的距离
            distances = []
            for point in data:
                min_dist = min(
                    self._euclidean_distance(point, c) for c in centroids
                )
                distances.append(min_dist * min_dist)  # 使用距离平方

            # 按距离平方的比例选择下一个质心
            total_dist = sum(distances)
            if total_dist == 0:
                # 所有距离为0，随机选择
                idx = self._rng.randint(0, n - 1)
            else:
                # 轮盘赌选择
                r = self._rng.random() * total_dist
                cumulative = 0.0
                idx = 0
                for i, d in enumerate(distances):
                    cumulative += d
                    if cumulative >= r:
                        idx = i
                        break

            centroids.append(list(data[idx]))

        return centroids


# ==================== InverseConstitutionalLearner ====================

class InverseConstitutionalLearner:
    """逆向宪章学习器 - 从人类偏好数据学习宪章规则"""

    # 特征名称到规则描述的映射模板
    FEATURE_RULE_TEMPLATES = {
        "sentiment_score_diff": "Responses should maintain a {direction} sentiment tone",
        "length_diff": "Responses should be {direction} in length",
        "word_count_diff": "Responses should contain {direction} words",
        "sentence_count_diff": "Responses should have {direction} sentences",
        "avg_sentence_length_diff": "Sentences should be {direction} in length",
        "positive_word_count_diff": "Responses should use {direction} positive language",
        "negative_word_count_diff": "Responses should use {direction} negative language",
        "question_ratio_diff": "Responses should ask {direction} questions",
        "exclamation_ratio_diff": "Responses should use {direction} exclamations",
        "passive_ratio_diff": "Responses should use {direction} passive voice",
        "technical_term_density_diff": "Responses should be {direction} technical",
        "vocabulary_diversity_diff": "Responses should have {direction} vocabulary diversity",
        "uppercase_ratio_diff": "Responses should use {direction} uppercase characters",
    }

    def __init__(self):
        self._dataset: Optional[PreferenceDataset] = None
        self._candidates: List[RuleCandidate] = []
        self._learned_rules: List[ConstitutionalRule] = []
        self._feature_extractor = FeatureExtractor()
        self._kmeans = SimpleKMeans()
        self._learning_report: Dict[str, Any] = {}

    def learn_from_preferences(
        self, dataset: PreferenceDataset, max_rules: int = 20
    ) -> Constitution:
        """
        完整学习流程: 从偏好数据中学习宪章

        a) 特征提取: 从chosen/rejected中提取差异特征
        b) 模式发现: 聚类相似差异，发现重复模式
        c) 规则生成: 将模式转化为宪章规则候选
        d) 规则筛选: 基于支持度和置信度筛选
        e) 规则验证: 在验证集上验证规则有效性
        """
        self._dataset = dataset
        self._learning_report = {
            "dataset_size": len(dataset),
            "max_rules": max_rules,
            "start_time": time.time(),
        }

        # 分割训练集和验证集
        train_set, val_set = dataset.split(0.8)

        # a) 特征提取
        all_features = []
        all_categories = []
        for pref in train_set:
            features = self._extract_features(pref)
            all_features.append(features)
            all_categories.append(pref.category)

        self._learning_report["features_extracted"] = len(all_features)

        # b) 模式发现
        patterns = self._discover_patterns(all_features, all_categories)
        self._learning_report["patterns_discovered"] = len(patterns)

        # c) 规则生成
        candidates = self._generate_rule_candidates(patterns)
        self._learning_report["candidates_generated"] = len(candidates)

        # d) 规则筛选
        filtered = self._filter_candidates(candidates, min_confidence=0.6)
        self._learning_report["candidates_filtered"] = len(filtered)

        # 限制规则数量
        filtered = sorted(filtered, key=lambda c: c.confidence, reverse=True)
        filtered = filtered[:max_rules]

        # e) 规则验证
        validation_results = self._validate_rules(filtered, val_set)
        self._learning_report["validation_results"] = validation_results

        # 转换为ConstitutionalRule
        self._learned_rules = []
        for candidate in filtered:
            rule = ConstitutionalRule(
                rule_id=hashlib.md5(
                    candidate.description.encode()
                ).hexdigest()[:12],
                description=candidate.description,
                category=candidate.category,
                severity=self._confidence_to_severity(candidate.confidence),
                rationale=(
                    f"Learned from {candidate.support_count} preference samples "
                    f"with {candidate.confidence:.2%} confidence. "
                    f"Patterns: {', '.join(candidate.patterns[:3])}"
                ),
                examples=candidate.patterns[:3],
            )
            self._learned_rules.append(rule)

        self._learning_report["rules_learned"] = len(self._learned_rules)
        self._learning_report["end_time"] = time.time()
        self._learning_report["duration"] = (
            self._learning_report["end_time"] - self._learning_report["start_time"]
        )

        constitution = Constitution(
            constitution_id="learned_" + hashlib.md5(
                str(time.time()).encode()
            ).hexdigest()[:12],
            name="Learned Constitution",
            description=(
                f"Constitution learned from {len(dataset)} preference samples "
                f"with {len(self._learned_rules)} rules"
            ),
            version="1.0.0",
            rules=self._learned_rules,
            metadata={
                "learning_method": "inverse_constitutional_learning",
                "dataset_size": len(dataset),
                "validation_results": validation_results,
            },
        )

        return constitution

    def _extract_features(self, preference: PreferenceData) -> Dict[str, float]:
        """从偏好数据中提取特征"""
        return self._feature_extractor.extract_preference_features(preference)

    def _discover_patterns(
        self,
        features: List[Dict[str, float]],
        categories: List[str],
    ) -> List[Pattern]:
        """
        模式发现: 基于特征相似度聚类，从每个簇中提取代表性模式

        使用纯Python实现的K-means聚类算法
        """
        if not features:
            return []

        # 收集所有特征名称(保持一致性)
        feature_keys = sorted(set(k for f in features for k in f.keys()))

        # 将特征字典转换为向量
        vectors = []
        for f in features:
            vec = [f.get(k, 0.0) for k in feature_keys]
            vectors.append(vec)

        # 确定聚类数
        n = len(vectors)
        k = min(max(3, n // 5), 15)  # 至少3个簇，最多15个

        if k >= n:
            k = max(1, n - 1)

        # 执行K-means聚类
        labels = self._kmeans.fit(vectors, k, max_iter=100)

        # 从每个簇中提取代表性模式
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        patterns = []
        for cluster_id, member_indices in clusters.items():
            if len(member_indices) < 2:
                continue

            # 计算簇的中心特征
            member_features = [features[i] for i in member_indices]
            center = {}
            for key in feature_keys:
                values = [f.get(key, 0.0) for f in member_features]
                center[key] = sum(values) / len(values)

            # 找出最具区分力的特征(绝对值最大的差异特征)
            diff_keys = [k for k in feature_keys if k.endswith("_diff")]
            significant_features = {}
            for key in diff_keys:
                val = center.get(key, 0.0)
                if abs(val) > 0.01:  # 只保留有意义的特征
                    significant_features[key] = val

            if not significant_features:
                continue

            # 按绝对值排序
            sorted_features = sorted(
                significant_features.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            # 生成模式描述
            top_features = sorted_features[:5]
            pattern_desc_parts = []
            for feat_name, feat_val in top_features:
                direction = "higher" if feat_val > 0 else "lower"
                clean_name = feat_name.replace("_diff", "").replace("_", " ")
                pattern_desc_parts.append(f"{clean_name}: {direction}")

            # 确定模式类别
            member_cats = [categories[i] for i in member_indices]
            cat_counter = Counter(member_cats)
            dominant_cat = cat_counter.most_common(1)[0][0]

            pattern = Pattern(
                pattern_id=f"pattern_{cluster_id}",
                feature_signature=[center.get(k, 0.0) for k in feature_keys],
                representative_features=dict(top_features),
                member_count=len(member_indices),
                category=dominant_cat,
                description="; ".join(pattern_desc_parts),
            )
            patterns.append(pattern)

        # 按成员数排序
        patterns.sort(key=lambda p: p.member_count, reverse=True)
        return patterns

    def _generate_rule_candidates(
        self, patterns: List[Pattern]
    ) -> List[RuleCandidate]:
        """将模式转化为宪章规则候选"""
        candidates = []

        for pattern in patterns:
            # 从代表性特征生成规则描述
            rule_descriptions = []
            pattern_names = []

            for feat_name, feat_val in pattern.representative_features.items():
                template = self.FEATURE_RULE_TEMPLATES.get(feat_name)
                if template is None:
                    continue

                direction = "more" if feat_val > 0 else "less"
                description = template.format(direction=direction)
                rule_descriptions.append(description)
                pattern_names.append(feat_name)

            if not rule_descriptions:
                continue

            # 合并描述生成规则
            combined_desc = rule_descriptions[0]
            if len(rule_descriptions) > 1:
                combined_desc = (
                    rule_descriptions[0] + ", and " +
                    ", ".join(rule_descriptions[1:-1]) +
                    ", and " + rule_descriptions[-1]
                )

            # 计算置信度: 基于模式大小和特征显著性
            total_significance = sum(
                abs(v) for v in pattern.representative_features.values()
            )
            confidence = min(
                0.5 + 0.3 * (pattern.member_count / max(len(patterns), 1))
                + 0.2 * min(total_significance, 1.0),
                1.0,
            )

            candidate = RuleCandidate(
                description=combined_desc,
                category=pattern.category,
                patterns=pattern_names,
                confidence=round(confidence, 4),
                support_count=pattern.member_count,
                violation_count=0,
            )
            candidates.append(candidate)

        return candidates

    def _filter_candidates(
        self, candidates: List[RuleCandidate], min_confidence: float = 0.6
    ) -> List[RuleCandidate]:
        """基于支持度和置信度筛选规则候选"""
        filtered = []

        for candidate in candidates:
            # 置信度过滤
            if candidate.confidence < min_confidence:
                continue

            # 支持度过滤: 至少需要2个样本支持
            if candidate.support_count < 2:
                continue

            # 去重: 检查是否与已选候选过于相似
            is_duplicate = False
            for existing in filtered:
                if self._rule_similarity(candidate.description, existing.description) > 0.8:
                    # 保留置信度更高的
                    if candidate.confidence > existing.confidence:
                        filtered.remove(existing)
                    else:
                        is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(candidate)

        return filtered

    def _validate_rules(
        self,
        candidates: List[RuleCandidate],
        validation_set: PreferenceDataset,
    ) -> Dict[str, float]:
        """在验证集上验证规则有效性"""
        if not validation_set or not candidates:
            return {}

        results = {}
        total = len(validation_set)
        correct = 0

        for candidate in candidates:
            rule_correct = 0
            rule_total = 0

            for pref in validation_set:
                features = self._extract_features(pref)
                rule_total += 1

                # 检查偏好数据是否符合规则
                rule_satisfied = self._check_rule_against_features(
                    candidate, features
                )

                # 如果规则预测正确(偏好标签与规则方向一致)
                if rule_satisfied == (pref.preference_label == 1):
                    rule_correct += 1

            accuracy = rule_correct / rule_total if rule_total > 0 else 0.0
            results[candidate.description[:50]] = round(accuracy, 4)

        return results

    def _check_rule_against_features(
        self, candidate: RuleCandidate, features: Dict[str, float]
    ) -> bool:
        """检查特征是否符合规则候选"""
        match_count = 0
        total_checks = 0

        for pattern_name in candidate.patterns:
            diff_key = f"{pattern_name}"
            val = features.get(diff_key, 0.0)

            if abs(val) < 0.001:
                continue

            total_checks += 1
            # 如果chosen更好(label=1), 差异应该为正(正面特征更多)
            if val > 0:
                match_count += 1

        if total_checks == 0:
            return True

        return (match_count / total_checks) > 0.5

    def _rule_similarity(self, desc_a: str, desc_b: str) -> float:
        """计算两个规则描述的相似度"""
        words_a = set(desc_a.lower().split())
        words_b = set(desc_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    def _confidence_to_severity(self, confidence: float) -> str:
        """将置信度映射到严重度"""
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        else:
            return "low"

    def get_learned_constitution(self) -> Optional[Constitution]:
        """获取学习到的宪章"""
        if not self._learned_rules:
            return None

        return Constitution(
            constitution_id="learned_" + hashlib.md5(
                str(time.time()).encode()
            ).hexdigest()[:12],
            name="Learned Constitution",
            description=f"Constitution with {len(self._learned_rules)} learned rules",
            version="1.0.0",
            rules=self._learned_rules,
        )

    def get_learning_report(self) -> dict:
        """获取学习报告"""
        return copy.deepcopy(self._learning_report)

    def active_learning(
        self, query_strategy: str = "uncertainty", n: int = 10
    ) -> List[PreferenceData]:
        """
        主动学习: 选择最有信息量的样本请求人类标注

        策略:
        - uncertainty: 选择特征差异最不确定的样本(接近0)
        - diversity: 选择特征最多样化的样本
        - margin: 选择规则判断最不确定的样本
        """
        if not self._dataset or len(self._dataset) == 0:
            return []

        all_features = []
        for pref in self._dataset:
            features = self._extract_features(pref)
            all_features.append(features)

        if query_strategy == "uncertainty":
            # 选择特征差异绝对值最小的样本(最不确定)
            scored = []
            for i, features in enumerate(all_features):
                diff_keys = [k for k in features if k.endswith("_diff")]
                if not diff_keys:
                    scored.append((i, 0.0))
                    continue

                total_abs = sum(abs(features.get(k, 0.0)) for k in diff_keys)
                avg_abs = total_abs / len(diff_keys)
                # 不确定性 = 1 - 归一化的平均绝对差异
                uncertainty = 1.0 / (1.0 + avg_abs)
                scored.append((i, uncertainty))

            scored.sort(key=lambda x: x[1], reverse=True)

        elif query_strategy == "diversity":
            # 选择特征最多样化的样本(使用贪心最大距离选择)
            if len(all_features) <= n:
                return list(self._dataset)

            feature_keys = sorted(
                set(k for f in all_features for k in f.keys())
            )
            vectors = [
                [f.get(k, 0.0) for k in feature_keys]
                for f in all_features
            ]

            selected_indices = [0]  # 随机选第一个
            min_distances = [
                self._min_distance_to_selected(vectors[i], vectors, selected_indices)
                for i in range(len(vectors))
            ]

            for _ in range(n - 1):
                # 选择与已选集合距离最远的点
                farthest_idx = max(
                    range(len(vectors)),
                    key=lambda i: min_distances[i] if i not in selected_indices else -1,
                )
                selected_indices.append(farthest_idx)

                # 更新最小距离
                for i in range(len(vectors)):
                    if i not in selected_indices:
                        dist = self._euclidean_distance(
                            vectors[i], vectors[farthest_idx]
                        )
                        min_distances[i] = min(min_distances[i], dist)

            return [self._dataset[i] for i in selected_indices]

        elif query_strategy == "margin":
            # 选择规则判断边界附近的样本
            scored = []
            for i, features in enumerate(all_features):
                diff_keys = [k for k in features if k.endswith("_diff")]
                if not diff_keys:
                    scored.append((i, 0.5))
                    continue

                # 计算正负特征的比例
                positive = sum(
                    1 for k in diff_keys if features.get(k, 0.0) > 0
                )
                negative = len(diff_keys) - positive
                total = positive + negative

                # margin: 越接近0.5越不确定
                margin = abs(positive / total - 0.5) if total > 0 else 0.5
                uncertainty = 1.0 - margin
                scored.append((i, uncertainty))

            scored.sort(key=lambda x: x[1], reverse=True)

        else:
            # 默认随机选择
            indices = random.sample(range(len(self._dataset)), min(n, len(self._dataset)))
            return [self._dataset[i] for i in indices]

        selected_indices = [idx for idx, _ in scored[:n]]
        return [self._dataset[i] for i in selected_indices]

    def _min_distance_to_selected(
        self,
        point: List[float],
        all_points: List[List[float]],
        selected: List[int],
    ) -> float:
        """计算点到已选集合的最小距离"""
        if not selected:
            return float("inf")
        return min(
            self._euclidean_distance(point, all_points[j])
            for j in selected
        )

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """计算欧氏距离"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
