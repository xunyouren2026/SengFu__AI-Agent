"""
Constitutional AI Engine - 宪章AI引擎

本模块实现了完整的宪章AI审查引擎，包括规则定义、文本审查、
风险评估、审计日志和效果评估等功能。所有实现使用纯Python，
不依赖任何外部库。
"""

import math
import random
import re
import hashlib
import json
import time
import threading
import statistics
import functools
import copy
from typing import (
    List, Dict, Tuple, Optional, Any, Set, Deque, Callable
)
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum


# ============================================================================
# 辅助函数：纯Python实现的TF向量余弦相似度
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """将文本分词为小写token列表"""
    text = text.lower().strip()
    # 按非字母数字字符分割，保留中文单字
    tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]', text)
    return tokens


def _build_tf_vector(text: str) -> Dict[str, float]:
    """构建文本的TF（词频）向量"""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf: Dict[str, float] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0.0) + 1.0
    total = len(tokens)
    for key in tf:
        tf[key] /= total
    return tf


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """计算两个TF向量的余弦相似度"""
    if not vec_a or not vec_b:
        return 0.0
    # 取两个向量的交集键
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not common_keys:
        return 0.0
    dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _generate_id() -> str:
    """生成唯一ID"""
    raw = f"{time.time()}-{random.random()}-{threading.get_ident()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================================
# RuleViolation - 规则违规记录
# ============================================================================

@dataclass
class RuleViolation:
    """规则违规记录，包含违规详情和修正建议"""
    rule_id: str
    rule_name: str
    category: str
    severity: float
    matched_text: str
    position: int
    context: str
    explanation: str
    suggested_fix: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RuleViolation":
        return cls(**d)


# ============================================================================
# ConstitutionalRule - 宪章规则
# ============================================================================

@dataclass
class ConstitutionalRule:
    """宪章规则定义，支持关键词匹配和TF向量语义相似度检测"""
    id: str = field(default_factory=_generate_id)
    name: str = ""
    description: str = ""
    category: str = "safety"  # safety/privacy/fairness/honesty/kindness
    severity: float = 0.5     # 0-1
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    enabled: bool = True

    # 内部缓存：规则的TF向量
    _description_vector: Dict[str, float] = field(default_factory=dict, repr=False)
    _condition_vectors: List[Dict[str, float]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """初始化后构建TF向量缓存"""
        self._description_vector = _build_tf_vector(self.description)
        self._condition_vectors = [_build_tf_vector(c) for c in self.conditions]

    def check(self, text: str) -> Optional[RuleViolation]:
        """
        检测文本是否违反本规则。
        使用关键词匹配 + TF向量余弦相似度进行检测。
        返回RuleViolation或None。
        """
        if not self.enabled:
            return None

        # 阶段1：关键词匹配
        keyword_result = self._keyword_match(text)
        if keyword_result is not None:
            return keyword_result

        # 阶段2：TF向量语义相似度检测
        semantic_result = self._semantic_match(text)
        if semantic_result is not None:
            return semantic_result

        return None

    def _keyword_match(self, text: str) -> Optional[RuleViolation]:
        """基于关键词的匹配检测"""
        text_lower = text.lower()
        best_match = None
        best_score = 0.0

        # 从描述和条件中提取关键词
        all_patterns = list(self.conditions)
        if not all_patterns:
            all_patterns = self._extract_keywords(self.description)

        for pattern in all_patterns:
            pattern_lower = pattern.lower()
            idx = text_lower.find(pattern_lower)
            if idx != -1:
                # 计算匹配得分：基于匹配长度和位置
                match_len = len(pattern)
                position_factor = 1.0 - (idx / max(len(text), 1)) * 0.3
                score = (match_len / max(len(text), 1)) * position_factor
                if score > best_score:
                    best_score = score
                    context_start = max(0, idx - 40)
                    context_end = min(len(text), idx + len(pattern) + 40)
                    context = text[context_start:context_end]
                    best_match = {
                        "matched_text": text[idx:idx + len(pattern)],
                        "position": idx,
                        "context": context,
                        "score": score,
                        "match_type": "keyword"
                    }

        if best_match is not None:
            # 检查是否命中例外
            if self._check_exceptions(text):
                return None
            severity = self.calculate_severity(text, best_match)
            explanation = self._generate_violation_explanation(best_match)
            suggested_fix = self._generate_suggestion(best_match, text)
            return RuleViolation(
                rule_id=self.id,
                rule_name=self.name,
                category=self.category,
                severity=severity,
                matched_text=best_match["matched_text"],
                position=best_match["position"],
                context=best_match["context"],
                explanation=explanation,
                suggested_fix=suggested_fix
            )
        return None

    def _semantic_match(self, text: str) -> Optional[RuleViolation]:
        """基于TF向量余弦相似度的语义匹配"""
        if not self._description_vector and not self._condition_vectors:
            return None

        text_vector = _build_tf_vector(text)
        if not text_vector:
            return None

        best_similarity = 0.0
        best_source = ""

        # 与描述向量比较
        if self._description_vector:
            sim = _cosine_similarity(text_vector, self._description_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_source = "description"

        # 与条件向量比较
        for i, cond_vec in enumerate(self._condition_vectors):
            sim = _cosine_similarity(text_vector, cond_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_source = f"condition_{i}"

        # 相似度阈值：基于规则严重度动态调整
        threshold = 0.3 + (1.0 - self.severity) * 0.2

        if best_similarity >= threshold:
            if self._check_exceptions(text):
                return None
            # 找到最相似的片段
            matched_segment = self._find_most_similar_segment(text, text_vector)
            severity = self.calculate_severity(text, {
                "score": best_similarity,
                "match_type": "semantic",
                "matched_text": matched_segment
            })
            explanation = (
                f"文本语义与规则'{self.name}'相似度较高 "
                f"(相似度={best_similarity:.3f}, 来源={best_source})"
            )
            return RuleViolation(
                rule_id=self.id,
                rule_name=self.name,
                category=self.category,
                severity=min(severity, 1.0),
                matched_text=matched_segment,
                position=text.find(matched_segment) if matched_segment in text else 0,
                context=text[:200] if len(text) > 200 else text,
                explanation=explanation,
                suggested_fix=self._generate_suggestion(
                    {"matched_text": matched_segment, "match_type": "semantic"}, text
                )
            )
        return None

    def _find_most_similar_segment(self, text: str, text_vector: Dict[str, float]) -> str:
        """在文本中找到与规则最相似的片段"""
        # 将文本按句子分割
        sentences = re.split(r'[。！？.!?\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return text[:100]

        best_sentence = sentences[0]
        best_sim = 0.0

        for sentence in sentences:
            sent_vector = _build_tf_vector(sentence)
            if not sent_vector:
                continue
            # 与所有条件向量比较，取最大值
            max_sim = 0.0
            if self._description_vector:
                max_sim = max(max_sim, _cosine_similarity(sent_vector, self._description_vector))
            for cond_vec in self._condition_vectors:
                max_sim = max(max_sim, _cosine_similarity(sent_vector, cond_vec))
            if max_sim > best_sim:
                best_sim = max_sim
                best_sentence = sentence

        return best_sentence[:100]

    def _check_exceptions(self, text: str) -> bool:
        """检查文本是否命中例外条件"""
        text_lower = text.lower()
        for exc in self.exceptions:
            if exc.lower() in text_lower:
                return True
        return False

    def _extract_keywords(self, text: str) -> List[str]:
        """从描述文本中提取关键词"""
        # 移除常见停用词后提取有意义的词组
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
            'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
            'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
            'if', 'when', 'where', 'how', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'it', 'its',
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它',
            '们', '那', '些', '么', '什么', '如何', '如何', '可以', '应该',
            '必须', '需要', '能够', '以及', '或者', '但是', '然而', '因此',
        }
        tokens = _tokenize(text)
        keywords = [t for t in tokens if t not in stop_words and len(t) > 1]
        # 也生成二元组作为关键词
        bigrams = []
        for i in range(len(tokens) - 1):
            if tokens[i] not in stop_words and tokens[i + 1] not in stop_words:
                bigrams.append(f"{tokens[i]} {tokens[i + 1]}")
        return keywords + bigrams

    def calculate_severity(self, text: str, match_info: dict) -> float:
        """
        根据匹配位置、上下文、重复次数等因素计算综合严重度。
        返回0-1之间的浮点数。
        """
        base_severity = self.severity
        score = match_info.get("score", 0.5)
        match_type = match_info.get("match_type", "keyword")

        # 因子1：匹配得分权重
        score_factor = 0.3 + score * 0.4

        # 因子2：位置权重（文本开头和结尾的违规更严重）
        position = match_info.get("position", 0)
        text_len = max(len(text), 1)
        relative_pos = position / text_len
        # 开头权重高，中间低，结尾中等
        if relative_pos < 0.2:
            position_factor = 0.9
        elif relative_pos > 0.8:
            position_factor = 0.7
        else:
            position_factor = 0.5

        # 因子3：重复出现次数
        matched_text = match_info.get("matched_text", "")
        repeat_count = 0
        if matched_text:
            repeat_count = text.lower().count(matched_text.lower())
        repeat_factor = min(1.0, 0.5 + repeat_count * 0.15)

        # 因子4：上下文中的强化词
        context = match_info.get("context", text)
        intensifiers = [
            "very", "extremely", "highly", "absolutely", "completely",
            "definitely", "certainly", "seriously", "severely",
            "非常", "极其", "绝对", "严重", "彻底", "完全"
        ]
        intensifier_count = sum(1 for w in intensifiers if w in context.lower())
        intensifier_factor = min(1.0, 0.6 + intensifier_count * 0.15)

        # 因子5：匹配类型权重
        type_factor = 0.8 if match_type == "keyword" else 0.6

        # 综合计算
        final_severity = (
            base_severity * 0.3
            + score_factor * 0.2
            + position_factor * 0.15
            + repeat_factor * 0.15
            + intensifier_factor * 0.1
            + type_factor * 0.1
        )
        return max(0.0, min(1.0, final_severity))

    def _generate_violation_explanation(self, match_info: dict) -> str:
        """生成违规解释"""
        match_type = match_info.get("match_type", "keyword")
        matched_text = match_info.get("matched_text", "")
        if match_type == "keyword":
            return (
                f"文本中包含违反规则'{self.name}'的内容: "
                f"'{matched_text}'。该规则属于{self.category}类别，"
                f"基础严重度为{self.severity:.2f}。"
            )
        else:
            return (
                f"文本语义与规则'{self.name}'高度相似。"
                f"该规则属于{self.category}类别。"
            )

    def _generate_suggestion(self, match_info: dict, text: str) -> str:
        """生成修正建议"""
        matched_text = match_info.get("matched_text", "")
        if not matched_text:
            return "建议重新审视文本内容，确保符合相关规则要求。"

        match_type = match_info.get("match_type", "keyword")
        if match_type == "keyword":
            return (
                f"建议移除或替换文本中的'{matched_text}'部分，"
                f"使用更符合规则'{self.name}'要求的表述方式。"
            )
        else:
            return (
                f"建议重新表述相关内容，避免与规则'{self.name}'产生语义冲突。"
            )

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "severity": self.severity,
            "conditions": self.conditions,
            "exceptions": self.exceptions,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConstitutionalRule":
        """从字典反序列化"""
        return cls(
            id=d.get("id", _generate_id()),
            name=d.get("name", ""),
            description=d.get("description", ""),
            category=d.get("category", "safety"),
            severity=d.get("severity", 0.5),
            conditions=d.get("conditions", []),
            exceptions=d.get("exceptions", []),
            enabled=d.get("enabled", True),
        )


# ============================================================================
# ConstitutionalReviewResult - 审查结果
# ============================================================================

@dataclass
class ConstitutionalReviewResult:
    """宪章审查结果"""
    passed: bool
    risk_score: float
    violations: List[RuleViolation]
    explanation: str
    suggested_fix: str
    review_time: float
    strictness_level: int

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "risk_score": self.risk_score,
            "violations": [v.to_dict() for v in self.violations],
            "explanation": self.explanation,
            "suggested_fix": self.suggested_fix,
            "review_time": self.review_time,
            "strictness_level": self.strictness_level,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConstitutionalReviewResult":
        violations = [RuleViolation.from_dict(v) for v in d.get("violations", [])]
        return cls(
            passed=d.get("passed", True),
            risk_score=d.get("risk_score", 0.0),
            violations=violations,
            explanation=d.get("explanation", ""),
            suggested_fix=d.get("suggested_fix", ""),
            review_time=d.get("review_time", 0.0),
            strictness_level=d.get("strictness_level", 3),
        )


# ============================================================================
# Constitution - 宪章
# ============================================================================

class Constitution:
    """AI宪章，管理一组宪章规则"""

    VALID_CATEGORIES = {"safety", "privacy", "fairness", "honesty", "kindness"}

    def __init__(self, version: str = "1.0", description: str = ""):
        self.rules: List[ConstitutionalRule] = []
        self.version = version
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.description = description

    def add_rule(self, rule: ConstitutionalRule) -> None:
        """添加规则"""
        # 检查ID是否重复
        for existing in self.rules:
            if existing.id == rule.id:
                raise ValueError(f"规则ID '{rule.id}' 已存在")
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则，返回是否成功"""
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                self.rules.pop(i)
                return True
        return False

    def get_rules_by_category(self, category: str) -> List[ConstitutionalRule]:
        """按类别获取规则"""
        return [r for r in self.rules if r.category == category]

    def validate_consistency(self) -> List[str]:
        """
        检查规则间冲突。
        例如："不说谎" vs "保护隐私" 的冲突。
        返回冲突描述列表。
        """
        conflicts: List[str] = []
        enabled_rules = [r for r in self.rules if r.enabled]

        # 定义已知的冲突规则对
        conflict_pairs = [
            ("honesty", "privacy", "诚实规则可能要求披露信息，而隐私规则要求保密"),
            ("fairness", "kindness", "公平规则可能要求平等对待，而善意规则可能要求特殊关怀"),
            ("safety", "honesty", "安全规则可能要求限制信息，而诚实规则要求完全透明"),
            ("privacy", "safety", "隐私规则可能限制数据共享，而安全规则可能需要数据共享来防范风险"),
        ]

        for cat_a, cat_b, reason in conflict_pairs:
            rules_a = [r for r in enabled_rules if r.category == cat_a]
            rules_b = [r for r in enabled_rules if r.category == cat_b]
            if rules_a and rules_b:
                # 检查具体规则间的语义冲突
                for ra in rules_a:
                    for rb in rules_b:
                        conflict_score = self._compute_rule_conflict(ra, rb)
                        if conflict_score > 0.4:
                            conflicts.append(
                                f"规则'{ra.name}'({cat_a})与规则'{rb.name}'({cat_b})存在潜在冲突: "
                                f"{reason} (冲突度={conflict_score:.2f})"
                            )

        # 检查同类别中严重度差异过大的规则
        for cat in self.VALID_CATEGORIES:
            cat_rules = [r for r in enabled_rules if r.category == cat]
            if len(cat_rules) > 1:
                severities = [r.severity for r in cat_rules]
                max_sev = max(severities)
                min_sev = min(severities)
                if max_sev - min_sev > 0.6:
                    conflicts.append(
                        f"类别'{cat}'中规则严重度差异过大 "
                        f"(最高={max_sev:.2f}, 最低={min_sev:.2f})，"
                        f"可能导致审查不一致"
                    )

        return conflicts

    def _compute_rule_conflict(self, rule_a: ConstitutionalRule, rule_b: ConstitutionalRule) -> float:
        """计算两条规则之间的冲突度"""
        # 基于描述的语义相似度来估算冲突度
        vec_a = _build_tf_vector(rule_a.description)
        vec_b = _build_tf_vector(rule_b.description)
        similarity = _cosine_similarity(vec_a, vec_b)

        # 如果描述相似度高但属于不同类别，冲突度更高
        if rule_a.category != rule_b.category:
            return similarity * 0.8
        else:
            # 同类别中，如果描述相似度高但严重度差异大，也有冲突
            severity_diff = abs(rule_a.severity - rule_b.severity)
            return similarity * 0.3 + severity_diff * 0.5

    def merge(self, other: "Constitution") -> "Constitution":
        """
        合并两套宪章，解决冲突。
        策略：保留两方所有规则，对冲突规则取较高严重度。
        """
        merged = Constitution(
            version=f"{self.version}+{other.version}",
            description=f"Merged: {self.description} | {other.description}"
        )

        # 用字典跟踪已添加的规则（按名称去重）
        name_to_rule: Dict[str, ConstitutionalRule] = {}

        for rule in self.rules:
            name_to_rule[rule.name] = copy.deepcopy(rule)

        for rule in other.rules:
            if rule.name in name_to_rule:
                # 冲突解决：取较高严重度
                existing = name_to_rule[rule.name]
                if rule.severity > existing.severity:
                    # 合并条件和例外
                    merged_conditions = list(set(existing.conditions + rule.conditions))
                    merged_exceptions = list(set(existing.exceptions + rule.exceptions))
                    rule.conditions = merged_conditions
                    rule.exceptions = merged_exceptions
                    name_to_rule[rule.name] = copy.deepcopy(rule)
            else:
                name_to_rule[rule.name] = copy.deepcopy(rule)

        for rule in name_to_rule.values():
            merged.add_rule(rule)

        return merged

    def export_yaml(self) -> str:
        """导出为简易YAML格式"""
        lines = [
            f"version: \"{self.version}\"",
            f"created_at: \"{self.created_at}\"",
            f"description: \"{self.description}\"",
            "rules:"
        ]
        for rule in self.rules:
            lines.append(f"  - id: \"{rule.id}\"")
            lines.append(f"    name: \"{rule.name}\"")
            lines.append(f"    description: \"{rule.description}\"")
            lines.append(f"    category: \"{rule.category}\"")
            lines.append(f"    severity: {rule.severity}")
            if rule.conditions:
                lines.append("    conditions:")
                for cond in rule.conditions:
                    lines.append(f"      - \"{cond}\"")
            if rule.exceptions:
                lines.append("    exceptions:")
                for exc in rule.exceptions:
                    lines.append(f"      - \"{exc}\"")
            lines.append(f"    enabled: {str(rule.enabled).lower()}")
        return "\n".join(lines)

    @classmethod
    def import_yaml(cls, yaml_str: str) -> "Constitution":
        """从简易YAML格式导入"""
        # 使用简易YAML解析器
        data = _simple_yaml_parse(yaml_str)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Constitution":
        """从字典反序列化"""
        constitution = cls(
            version=d.get("version", "1.0"),
            description=d.get("description", ""),
        )
        if "created_at" in d:
            constitution.created_at = d["created_at"]
        for rule_data in d.get("rules", []):
            constitution.add_rule(ConstitutionalRule.from_dict(rule_data))
        return constitution

    def get_rule_count(self) -> int:
        """获取规则总数"""
        return len(self.rules)

    def get_category_distribution(self) -> Dict[str, int]:
        """获取各类别规则数量分布"""
        dist: Dict[str, int] = defaultdict(int)
        for rule in self.rules:
            dist[rule.category] += 1
        return dict(dist)


# ============================================================================
# 简易YAML解析器
# ============================================================================

def _simple_yaml_parse(yaml_str: str) -> dict:
    """
    简易YAML解析器，支持基本键值对和列表。
    不依赖PyYAML。
    """
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[List[Any]] = None
    in_list = False
    list_item_key: Optional[str] = None
    list_item_dict: Optional[Dict[str, Any]] = None
    list_item_list: Optional[List[str]] = None
    list_item_list_key: Optional[str] = None

    lines = yaml_str.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 跳过空行和注释
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # 计算缩进级别
        indent = len(line) - len(line.lstrip())

        if indent == 0 and ":" in stripped:
            # 顶层键值对
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not value:
                # 值在下一行（可能是列表或字典）
                current_key = key
                current_list = None
                in_list = False
                list_item_dict = None
                list_item_list = None
                list_item_list_key = None
            else:
                # 直接值
                # 尝试转换为数字或布尔
                parsed_value = _parse_yaml_value(value)
                result[key] = parsed_value
                current_key = None

        elif indent == 2 and current_key and stripped.startswith("- "):
            # 列表项开始
            if current_list is None:
                current_list = []
                result[current_key] = current_list

            item_content = stripped[2:].strip().strip('"').strip("'")
            if ":" in item_content:
                # 列表项是字典
                list_item_dict = {}
                list_item_list = None
                list_item_list_key = None
                k, _, v = item_content.partition(":")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                list_item_dict[k] = _parse_yaml_value(v)
                list_item_key = k
            else:
                list_item_dict = None
                list_item_list = None
                current_list.append(_parse_yaml_value(item_content))
            in_list = True

        elif indent == 4 and in_list:
            # 列表项的子属性
            if list_item_dict is not None:
                if stripped.startswith("- "):
                    # 子列表
                    item_val = stripped[2:].strip().strip('"').strip("'")
                    if list_item_list is None:
                        list_item_list = []
                        list_item_dict[list_item_list_key] = list_item_list
                    list_item_list.append(item_val)
                elif ":" in stripped:
                    # 提交之前的字典
                    if list_item_list is not None and list_item_list_key is not None:
                        pass  # 已在dict中
                    k, _, v = stripped.partition(":")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if not v:
                        list_item_list_key = k
                        list_item_list = None
                    else:
                        list_item_dict[k] = _parse_yaml_value(v)
                        list_item_list = None
                        list_item_list_key = None

        elif indent == 6 and in_list and list_item_dict is not None and list_item_list_key:
            # 子列表的内容
            if stripped.startswith("- "):
                item_val = stripped[2:].strip().strip('"').strip("'")
                if list_item_list is None:
                    list_item_list = []
                    list_item_dict[list_item_list_key] = list_item_list
                list_item_list.append(item_val)

        i += 1

    # 提交最后一个列表项
    if current_list is not None and list_item_dict is not None:
        current_list.append(list_item_dict)

    return result


def _parse_yaml_value(value: str) -> Any:
    """解析YAML值，自动转换类型"""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ============================================================================
# ConstitutionalEngine - 核心审查引擎
# ============================================================================

class ConstitutionalEngine:
    """宪章AI核心审查引擎"""

    # 类别权重映射
    CATEGORY_WEIGHTS: Dict[str, float] = {
        "safety": 1.0,
        "privacy": 0.9,
        "fairness": 0.85,
        "honesty": 0.8,
        "kindness": 0.7,
    }

    # 严格度对应的阈值倍数
    STRICTNESS_MULTIPLIERS = {
        1: 0.5,   # 非常宽松
        2: 0.75,  # 宽松
        3: 1.0,   # 标准
        4: 1.25,  # 严格
        5: 1.5,   # 非常严格
    }

    def __init__(self, constitution: Optional[Constitution] = None,
                 strictness_level: int = 3):
        self.constitution = constitution or Constitution()
        self.strictness_level = max(1, min(5, strictness_level))
        self._violation_history: Deque[RuleViolation] = deque(maxlen=1000)
        self._stats: Dict[str, Any] = defaultdict(int)
        self._stats["total_reviews"] = 0
        self._stats["total_violations"] = 0
        self._stats["passed_reviews"] = 0
        self._stats["failed_reviews"] = 0
        self._stats["category_violations"] = defaultdict(int)
        self._stats["avg_risk_score"] = 0.0
        self._stats["risk_scores"] = []

    def review(self, text: str, context: str = "") -> ConstitutionalReviewResult:
        """
        完整审查流程：
        1. 预处理文本
        2. 逐规则检测
        3. 计算风险评分
        4. 生成解释和修正建议
        """
        start_time = time.time()

        # 预处理
        processed_text = self._preprocess(text)

        # 如果提供了上下文，合并到文本中一起检测
        review_text = processed_text
        if context:
            review_text = f"{context}\n{processed_text}"

        # 逐规则检测
        violations: List[RuleViolation] = []
        for rule in self.constitution.rules:
            if not rule.enabled:
                continue
            violation = self._check_single_rule(review_text, rule)
            if violation is not None:
                violations.append(violation)

        # 按严重度排序
        violations.sort(key=lambda v: v.severity, reverse=True)

        # 计算风险评分
        risk_score = self._calculate_risk_score(violations)

        # 判断是否通过
        # 通过阈值根据严格度调整
        threshold = 0.3 / self.STRICTNESS_MULTIPLIERS[self.strictness_level]
        passed = risk_score < threshold and len(violations) == 0

        # 生成解释
        explanation = self._generate_explanation(violations)

        # 生成修正建议
        suggested_fix = self._suggest_fixes(violations, processed_text) if violations else ""

        # 记录统计
        review_time = time.time() - start_time
        self._update_stats(violations, risk_score, passed)

        # 记录违规历史
        for v in violations:
            self._violation_history.append(v)

        return ConstitutionalReviewResult(
            passed=passed,
            risk_score=risk_score,
            violations=violations,
            explanation=explanation,
            suggested_fix=suggested_fix,
            review_time=review_time,
            strictness_level=self.strictness_level,
        )

    def _preprocess(self, text: str) -> str:
        """文本预处理：标准化、去除不可见字符"""
        if not text:
            return ""
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 去除零宽字符和其他不可见Unicode字符
        text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]', '', text)
        # 去除控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # 标准化空白字符：多个空格合并为一个
        text = re.sub(r' {2,}', ' ', text)
        # 标准化多个换行为两个
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除首尾空白
        text = text.strip()
        return text

    def _check_single_rule(self, text: str, rule: ConstitutionalRule) -> Optional[RuleViolation]:
        """单规则检测，考虑严格度调整"""
        # 根据严格度调整规则的严重度阈值
        multiplier = self.STRICTNESS_MULTIPLIERS[self.strictness_level]

        # 在高严格度下，临时降低匹配阈值
        original_severity = rule.severity
        if self.strictness_level >= 4:
            # 严格模式下，降低有效严重度阈值使更多内容被检测
            rule.severity = min(1.0, rule.severity * multiplier)

        try:
            violation = rule.check(text)
        finally:
            rule.severity = original_severity

        return violation

    def _calculate_risk_score(self, violations: List[RuleViolation]) -> float:
        """
        综合风险评分计算。
        使用加权平均 + 类别惩罚 + 违规数量惩罚。
        """
        if not violations:
            return 0.0

        # 加权平均严重度
        total_weight = 0.0
        weighted_sum = 0.0
        category_counts: Dict[str, int] = defaultdict(int)

        for v in violations:
            cat_weight = self.CATEGORY_WEIGHTS.get(v.category, 0.5)
            weighted_sum += v.severity * cat_weight
            total_weight += cat_weight
            category_counts[v.category] += 1

        avg_risk = weighted_sum / max(total_weight, 1.0)

        # 类别惩罚：如果多个类别同时违规，增加风险
        category_penalty = 1.0
        num_categories = len(category_counts)
        if num_categories >= 3:
            category_penalty = 1.3
        elif num_categories == 2:
            category_penalty = 1.15

        # 违规数量惩罚
        count_penalty = 1.0 + min(len(violations) * 0.05, 0.5)

        # 最高严重度加成
        max_severity = max(v.severity for v in violations)
        max_severity_bonus = max_severity * 0.2

        final_score = (
            avg_risk * category_penalty * count_penalty
            + max_severity_bonus
        )
        return max(0.0, min(1.0, final_score))

    def _generate_explanation(self, violations: List[RuleViolation]) -> str:
        """生成人类可读的审查结果解释"""
        if not violations:
            return "文本通过宪章审查，未发现违规内容。"

        parts: List[str] = []
        parts.append(f"共发现 {len(violations)} 项违规：")

        # 按类别分组
        by_category: Dict[str, List[RuleViolation]] = defaultdict(list)
        for v in violations:
            by_category[v.category].append(v)

        category_names = {
            "safety": "安全",
            "privacy": "隐私",
            "fairness": "公平",
            "honesty": "诚实",
            "kindness": "善意",
        }

        for cat, cat_violations in sorted(by_category.items()):
            cat_name = category_names.get(cat, cat)
            avg_sev = statistics.mean(v.severity for v in cat_violations)
            parts.append(
                f"  [{cat_name}] {len(cat_violations)}项违规, "
                f"平均严重度={avg_sev:.2f}"
            )
            for v in cat_violations[:3]:  # 最多显示3条
                parts.append(f"    - {v.explanation}")

        if len(violations) > 3:
            parts.append(f"  ...还有 {len(violations) - 3} 项违规未显示")

        return "\n".join(parts)

    def _suggest_fixes(self, violations: List[RuleViolation], text: str) -> str:
        """
        建议修正方案，基于替换/删除/重写策略。
        """
        if not violations:
            return ""

        suggestions: List[str] = []
        suggestions.append("修正建议：")

        # 策略1：直接替换/删除违规文本
        for v in violations[:5]:
            if v.matched_text and v.matched_text in text:
                suggestions.append(
                    f"  [替换] 将'{v.matched_text}'替换为更合适的表述 "
                    f"(规则: {v.rule_name})"
                )

        # 策略2：如果同一位置有多个违规，建议重写
        position_groups: Dict[int, List[RuleViolation]] = defaultdict(list)
        for v in violations:
            # 将位置量化到50字符的区间
            bucket = v.position // 50
            position_groups[bucket].append(v)

        for bucket, group_violations in position_groups.items():
            if len(group_violations) > 1:
                start_pos = bucket * 50
                end_pos = min(start_pos + 50, len(text))
                segment = text[start_pos:end_pos]
                suggestions.append(
                    f"  [重写] 位置{start_pos}-{end_pos}存在{len(group_violations)}项违规，"
                    f"建议重写该段落: '{segment[:30]}...'"
                )

        # 策略3：整体建议
        high_severity = [v for v in violations if v.severity > 0.7]
        if high_severity:
            categories = set(v.category for v in high_severity)
            suggestions.append(
                f"  [警告] 存在{len(high_severity)}项高严重度违规，"
                f"涉及类别: {', '.join(categories)}，建议全面修改后重新提交"
            )

        return "\n".join(suggestions)

    def batch_review(self, texts: List[str]) -> List[ConstitutionalReviewResult]:
        """批量审查多个文本"""
        results: List[ConstitutionalReviewResult] = []
        for text in texts:
            result = self.review(text)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取审查统计信息"""
        stats = dict(self._stats)
        stats["category_violations"] = dict(stats.get("category_violations", {}))
        stats["strictness_level"] = self.strictness_level
        stats["rule_count"] = self.constitution.get_rule_count()
        stats["history_size"] = len(self._violation_history)

        # 计算平均风险分
        risk_scores = stats.get("risk_scores", [])
        if risk_scores:
            stats["avg_risk_score"] = statistics.mean(risk_scores)
            stats["max_risk_score"] = max(risk_scores)
            stats["min_risk_score"] = min(risk_scores)
            if len(risk_scores) > 1:
                stats["risk_score_std"] = statistics.stdev(risk_scores)
            else:
                stats["risk_score_std"] = 0.0
        else:
            stats["avg_risk_score"] = 0.0
            stats["max_risk_score"] = 0.0
            stats["min_risk_score"] = 0.0
            stats["risk_score_std"] = 0.0

        # 通过率
        total = stats.get("total_reviews", 0)
        if total > 0:
            stats["pass_rate"] = stats.get("passed_reviews", 0) / total
        else:
            stats["pass_rate"] = 0.0

        return stats

    def set_strictness(self, level: int) -> None:
        """动态调整严格度级别(1-5)"""
        if not 1 <= level <= 5:
            raise ValueError(f"严格度级别必须在1-5之间，当前值: {level}")
        self.strictness_level = level

    def add_custom_rule(self, name: str, description: str, category: str,
                        patterns: List[str]) -> ConstitutionalRule:
        """添加自定义规则"""
        if category not in Constitution.VALID_CATEGORIES:
            raise ValueError(
                f"无效类别: {category}，"
                f"有效类别: {', '.join(Constitution.VALID_CATEGORIES)}"
            )
        rule = ConstitutionalRule(
            name=name,
            description=description,
            category=category,
            severity=0.6,
            conditions=patterns,
            exceptions=[],
            enabled=True,
        )
        self.constitution.add_rule(rule)
        return rule

    def _update_stats(self, violations: List[RuleViolation],
                      risk_score: float, passed: bool) -> None:
        """更新内部统计"""
        self._stats["total_reviews"] += 1
        self._stats["total_violations"] += len(violations)
        if passed:
            self._stats["passed_reviews"] += 1
        else:
            self._stats["failed_reviews"] += 1
        for v in violations:
            self._stats["category_violations"][v.category] += 1
        self._stats["risk_scores"].append(risk_score)
        # 保留最近1000个风险分
        if len(self._stats["risk_scores"]) > 1000:
            self._stats["risk_scores"] = self._stats["risk_scores"][-1000:]


# ============================================================================
# ConstitutionalAuditLog - 审计日志
# ============================================================================

class ConstitutionalAuditLog:
    """宪章审查审计日志，支持查询和趋势分析"""

    def __init__(self):
        self._logs: List[dict] = []
        self._lock = threading.Lock()

    def log(self, review_result: ConstitutionalReviewResult,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """记录审查结果到审计日志"""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "passed": review_result.passed,
            "risk_score": review_result.risk_score,
            "violation_count": len(review_result.violations),
            "strictness_level": review_result.strictness_level,
            "review_time": review_result.review_time,
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "category": v.category,
                    "severity": v.severity,
                }
                for v in review_result.violations
            ],
            "metadata": metadata or {},
        }
        with self._lock:
            self._logs.append(entry)

    def query(self, rule_id: Optional[str] = None,
              category: Optional[str] = None,
              time_range: Optional[Tuple[str, str]] = None) -> List[dict]:
        """
        查询审计日志。
        支持按规则ID、类别、时间范围过滤。
        """
        with self._lock:
            results = []
            for entry in self._logs:
                # 时间范围过滤
                if time_range:
                    start_time, end_time = time_range
                    entry_time = entry["timestamp"]
                    if entry_time < start_time or entry_time > end_time:
                        continue

                # 规则ID过滤
                if rule_id:
                    found = any(
                        v["rule_id"] == rule_id
                        for v in entry["violations"]
                    )
                    if not found:
                        continue

                # 类别过滤
                if category:
                    found = any(
                        v["category"] == category
                        for v in entry["violations"]
                    )
                    if not found:
                        continue

                results.append(copy.deepcopy(entry))
            return results

    def get_violation_trend(self, window: int = 100) -> Dict[str, List[float]]:
        """
        获取违规趋势数据。
        返回按时间窗口聚合的违规数量和风险评分趋势。
        """
        with self._lock:
            if not self._logs:
                return {"violation_counts": [], "risk_scores": [], "pass_rates": []}

            violation_counts: List[float] = []
            risk_scores: List[float] = []
            pass_rates: List[float] = []

            # 滑动窗口计算
            logs = self._logs
            total = len(logs)
            step = max(1, total // window)

            for i in range(0, total, step):
                chunk = logs[i:i + step]
                if not chunk:
                    continue
                counts = [e["violation_count"] for e in chunk]
                risks = [e["risk_score"] for e in chunk]
                passed = [1 if e["passed"] else 0 for e in chunk]

                violation_counts.append(statistics.mean(counts))
                risk_scores.append(statistics.mean(risks))
                pass_rates.append(statistics.mean(passed))

            return {
                "violation_counts": violation_counts,
                "risk_scores": risk_scores,
                "pass_rates": pass_rates,
            }

    def export_report(self, format: str = "json") -> str:
        """导出审计报告"""
        with self._lock:
            if format == "json":
                return json.dumps(self._logs, indent=2, ensure_ascii=False)
            elif format == "text":
                lines: List[str] = []
                lines.append("=" * 60)
                lines.append("宪章审查审计报告")
                lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append(f"总记录数: {len(self._logs)}")
                lines.append("=" * 60)

                total_reviews = len(self._logs)
                total_violations = sum(e["violation_count"] for e in self._logs)
                passed_count = sum(1 for e in self._logs if e["passed"])
                avg_risk = (
                    statistics.mean([e["risk_score"] for e in self._logs])
                    if self._logs else 0.0
                )

                lines.append(f"\n总审查次数: {total_reviews}")
                lines.append(f"总违规次数: {total_violations}")
                lines.append(f"通过次数: {passed_count}")
                lines.append(f"通过率: {passed_count / max(total_reviews, 1):.2%}")
                lines.append(f"平均风险分: {avg_risk:.4f}")

                # 按类别统计
                cat_stats: Dict[str, int] = defaultdict(int)
                for entry in self._logs:
                    for v in entry["violations"]:
                        cat_stats[v["category"]] += 1

                lines.append("\n按类别违规统计:")
                for cat, count in sorted(cat_stats.items(), key=lambda x: -x[1]):
                    lines.append(f"  {cat}: {count}")

                return "\n".join(lines)
            else:
                raise ValueError(f"不支持的格式: {format}")

    def get_most_violated_rules(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """获取最常被违反的规则排名"""
        with self._lock:
            rule_counts: Dict[str, int] = defaultdict(int)
            rule_names: Dict[str, str] = {}

            for entry in self._logs:
                for v in entry["violations"]:
                    rule_counts[v["rule_id"]] += 1
                    rule_names[v["rule_id"]] = v["rule_name"]

            sorted_rules = sorted(
                rule_counts.items(), key=lambda x: -x[1]
            )[:top_n]

            return [
                (rule_names[rid], count) for rid, count in sorted_rules
            ]


# ============================================================================
# ConstitutionalEvaluator - 宪章有效性评估
# ============================================================================

class ConstitutionalEvaluator:
    """评估宪章规则的有效性"""

    def evaluate_precision(self, gold_standard: List[bool],
                           reviews: List[ConstitutionalReviewResult]) -> float:
        """
        计算精确率。
        gold_standard[i] = True 表示第i个文本确实应该被标记为违规。
        reviews[i].passed = False 表示引擎标记为违规。
        精确率 = TP / (TP + FP)
        """
        if len(gold_standard) != len(reviews):
            raise ValueError("gold_standard和reviews长度必须相同")

        tp = 0  # 正确标记为违规
        fp = 0  # 错误标记为违规

        for gold, review in zip(gold_standard, reviews):
            predicted_violation = not review.passed
            if gold and predicted_violation:
                tp += 1
            elif not gold and predicted_violation:
                fp += 1

        if tp + fp == 0:
            return 1.0  # 没有预测为违规的，精确率视为完美
        return tp / (tp + fp)

    def evaluate_recall(self, gold_standard: List[bool],
                        reviews: List[ConstitutionalReviewResult]) -> float:
        """
        计算召回率。
        召回率 = TP / (TP + FN)
        """
        if len(gold_standard) != len(reviews):
            raise ValueError("gold_standard和reviews长度必须相同")

        tp = 0
        fn = 0  # 漏报

        for gold, review in zip(gold_standard, reviews):
            predicted_violation = not review.passed
            if gold and predicted_violation:
                tp += 1
            elif gold and not predicted_violation:
                fn += 1

        if tp + fn == 0:
            return 1.0  # 没有实际违规的，召回率视为完美
        return tp / (tp + fn)

    def evaluate_f1(self, precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def cross_validate(self, texts: List[str], labels: List[bool],
                       k: int = 5) -> Dict[str, float]:
        """
        K折交叉验证。
        将数据分为K份，轮流用K-1份训练（调整规则）、1份测试。
        返回平均精确率、召回率、F1。
        """
        if len(texts) != len(labels):
            raise ValueError("texts和labels长度必须相同")
        if len(texts) < k:
            raise ValueError(f"数据量({len(texts)})小于折数({k})")

        n = len(texts)
        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)

        fold_size = n // k
        precisions: List[float] = []
        recalls: List[float] = []
        f1_scores: List[float] = []

        for fold in range(k):
            # 划分训练集和测试集
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < k - 1 else n

            test_indices = set(indices[test_start:test_end])
            train_texts = [texts[i] for i in range(n) if i not in test_indices]
            train_labels = [labels[i] for i in range(n) if i not in test_indices]
            test_texts = [texts[i] for i in range(n) if i in test_indices]
            test_labels = [labels[i] for i in range(n) if i in test_indices]

            # 在训练集上分析违规模式，动态调整规则权重
            # （简化实现：使用标准引擎进行审查）
            engine = ConstitutionalEngine(
                constitution=self._create_optimized_constitution(
                    train_texts, train_labels
                )
            )

            # 在测试集上评估
            test_reviews = engine.batch_review(test_texts)
            precision = self.evaluate_precision(test_labels, test_reviews)
            recall = self.evaluate_recall(test_labels, test_reviews)
            f1 = self.evaluate_f1(precision, recall)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {
            "precision_mean": statistics.mean(precisions),
            "precision_std": statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
            "recall_mean": statistics.mean(recalls),
            "recall_std": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            "f1_mean": statistics.mean(f1_scores),
            "f1_std": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
            "folds": k,
        }

    def _create_optimized_constitution(self, texts: List[str],
                                       labels: List[bool]) -> Constitution:
        """基于训练数据创建优化后的宪章（简化实现）"""
        constitution = Constitution(version="optimized", description="交叉验证优化宪章")

        # 分析违规文本中的常见模式
        violation_texts = [t for t, l in zip(texts, labels) if l]
        safe_texts = [t for t, l in zip(texts, labels) if not l]

        # 从违规文本中提取高频词作为规则条件
        all_tokens: Dict[str, int] = defaultdict(int)
        for text in violation_texts:
            tokens = _tokenize(text)
            for token in tokens:
                if len(token) > 2:
                    all_tokens[token] += 1

        # 也统计安全文本中的词频（用于排除）
        safe_tokens: Dict[str, int] = defaultdict(int)
        for text in safe_texts:
            tokens = _tokenize(text)
            for token in tokens:
                safe_tokens[token] += 1

        # 选择在违规文本中高频但在安全文本中低频的词作为规则条件
        discriminative = []
        for token, count in all_tokens.items():
            safe_count = safe_tokens.get(token, 0)
            if count >= 2 and safe_count == 0:
                discriminative.append((token, count))

        discriminative.sort(key=lambda x: -x[1])
        top_patterns = [t for t, c in discriminative[:10]]

        if top_patterns:
            rule = ConstitutionalRule(
                name="cross_validation_rule",
                description="基于交叉验证训练数据自动生成的规则",
                category="safety",
                severity=0.7,
                conditions=top_patterns,
                exceptions=[],
                enabled=True,
            )
            constitution.add_rule(rule)

        return constitution

    def ablation_study(self, texts: List[str],
                       labels: List[bool]) -> Dict[str, float]:
        """
        消融实验：逐个移除规则，观察效果变化。
        返回每个规则移除后的F1分数。
        """
        if len(texts) != len(labels):
            raise ValueError("texts和labels长度必须相同")

        # 基线：使用所有规则
        base_constitution = Constitution(
            version="ablation_base",
            description="消融实验基线"
        )
        # 创建一组基础规则
        base_rules = self._create_base_rules_for_ablation(texts, labels)
        for rule in base_rules:
            base_constitution.add_rule(rule)

        base_engine = ConstitutionalEngine(constitution=base_constitution)
        base_reviews = base_engine.batch_review(texts)
        base_precision = self.evaluate_precision(labels, base_reviews)
        base_recall = self.evaluate_recall(labels, base_reviews)
        base_f1 = self.evaluate_f1(base_precision, base_recall)

        results: Dict[str, float] = {
            "base_f1": base_f1,
            "base_precision": base_precision,
            "base_recall": base_recall,
        }

        # 逐个移除规则
        for rule in base_constitution.rules:
            ablation_constitution = Constitution(
                version="ablation",
                description=f"移除规则: {rule.name}"
            )
            for r in base_constitution.rules:
                if r.id != rule.id:
                    ablation_constitution.add_rule(
                        ConstitutionalRule.from_dict(r.to_dict())
                    )

            engine = ConstitutionalEngine(constitution=ablation_constitution)
            reviews = engine.batch_review(texts)
            precision = self.evaluate_precision(labels, reviews)
            recall = self.evaluate_recall(labels, reviews)
            f1 = self.evaluate_f1(precision, recall)
            results[f"without_{rule.name}"] = f1

        return results

    def _create_base_rules_for_ablation(self, texts: List[str],
                                         labels: List[bool]) -> List[ConstitutionalRule]:
        """为消融实验创建基础规则集"""
        rules: List[ConstitutionalRule] = []

        # 分析违规文本，按类别生成规则
        violation_texts = [t for t, l in zip(texts, labels) if l]

        # 安全类规则
        safety_patterns = self._extract_discriminative_patterns(
            violation_texts, [t for t, l in zip(texts, labels) if not l],
            min_count=2
        )
        if safety_patterns:
            rules.append(ConstitutionalRule(
                name="safety_rule",
                description="检测不安全内容",
                category="safety",
                severity=0.8,
                conditions=safety_patterns[:5],
                exceptions=[],
                enabled=True,
            ))

        # 隐私类规则
        privacy_patterns = [p for p in safety_patterns if any(
            w in p.lower() for w in ["password", "secret", "private", "personal", "密码", "隐私"]
        )]
        if privacy_patterns:
            rules.append(ConstitutionalRule(
                name="privacy_rule",
                description="检测隐私泄露风险",
                category="privacy",
                severity=0.7,
                conditions=privacy_patterns[:5],
                exceptions=[],
                enabled=True,
            ))

        # 公平类规则
        fairness_patterns = [p for p in safety_patterns if any(
            w in p.lower() for w in ["discriminat", "bias", "unfair", "歧视", "偏见"]
        )]
        if fairness_patterns:
            rules.append(ConstitutionalRule(
                name="fairness_rule",
                description="检测不公平内容",
                category="fairness",
                severity=0.7,
                conditions=fairness_patterns[:5],
                exceptions=[],
                enabled=True,
            ))

        # 诚实类规则
        honesty_patterns = [p for p in safety_patterns if any(
            w in p.lower() for w in ["lie", "deceit", "fake", "false", "谎言", "虚假"]
        )]
        if honesty_patterns:
            rules.append(ConstitutionalRule(
                name="honesty_rule",
                description="检测不诚实内容",
                category="honesty",
                severity=0.6,
                conditions=honesty_patterns[:5],
                exceptions=[],
                enabled=True,
            ))

        # 善意类规则
        kindness_patterns = [p for p in safety_patterns if any(
            w in p.lower() for w in ["harm", "hurt", "abuse", "伤害", "虐待"]
        )]
        if kindness_patterns:
            rules.append(ConstitutionalRule(
                name="kindness_rule",
                description="检测有害内容",
                category="kindness",
                severity=0.8,
                conditions=kindness_patterns[:5],
                exceptions=[],
                enabled=True,
            ))

        return rules

    def _extract_discriminative_patterns(self, positive_texts: List[str],
                                          negative_texts: List[str],
                                          min_count: int = 2) -> List[str]:
        """提取区分性模式（在正样本中高频，负样本中低频）"""
        pos_token_counts: Dict[str, int] = defaultdict(int)
        neg_token_counts: Dict[str, int] = defaultdict(int)

        for text in positive_texts:
            for token in _tokenize(text):
                if len(token) > 1:
                    pos_token_counts[token] += 1

        for text in negative_texts:
            for token in _tokenize(text):
                if len(token) > 1:
                    neg_token_counts[token] += 1

        # 计算区分度
        patterns: List[Tuple[str, float]] = []
        all_tokens = set(pos_token_counts.keys()) | set(neg_token_counts.keys())
        for token in all_tokens:
            pos_count = pos_token_counts.get(token, 0)
            neg_count = neg_token_counts.get(token, 0)
            if pos_count >= min_count:
                # 区分度 = 正样本频率 / (正样本频率 + 负样本频率 + 1)
                discriminativity = pos_count / (pos_count + neg_count + 1)
                patterns.append((token, discriminativity))

        patterns.sort(key=lambda x: -x[1])
        return [p for p, _ in patterns[:20]]

    def calibration_analysis(self, reviews: List[ConstitutionalReviewResult]) -> Dict[str, float]:
        """
        校准分析：评估预测风险评分的校准程度。
        将预测风险分箱，计算每个箱内的实际违规率。
        理想情况下，预测风险为0.3的箱内应有约30%的实际违规。
        """
        if not reviews:
            return {}

        # 将风险评分分箱
        num_bins = 10
        bin_edges = [i / num_bins for i in range(num_bins + 1)]

        bin_predicted: List[float] = []
        bin_actual: List[float] = []
        bin_counts: List[int] = []

        for i in range(num_bins):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            bin_reviews = [
                r for r in reviews
                if low <= r.risk_score < high
            ]

            if not bin_reviews:
                continue

            predicted_mean = statistics.mean(r.risk_score for r in bin_reviews)
            actual_rate = sum(1 for r in bin_reviews if not r.passed) / len(bin_reviews)

            bin_predicted.append(predicted_mean)
            bin_actual.append(actual_rate)
            bin_counts.append(len(bin_reviews))

        # 计算校准误差
        if bin_predicted and bin_actual:
            # 加权平均校准误差
            total = sum(bin_counts)
            weights = [c / total for c in bin_counts]
            calibration_error = sum(
                abs(p - a) * w
                for p, a, w in zip(bin_predicted, bin_actual, weights)
            )

            # ECE (Expected Calibration Error)
            ece = calibration_error

            # 相关系数（预测vs实际）
            if len(bin_predicted) >= 2:
                correlation = self._pearson_correlation(bin_predicted, bin_actual)
            else:
                correlation = 0.0
        else:
            ece = 0.0
            correlation = 0.0

        return {
            "calibration_error": ece,
            "correlation": correlation,
            "num_bins_used": len(bin_predicted),
            "bin_predicted_means": bin_predicted,
            "bin_actual_rates": bin_actual,
            "bin_counts": bin_counts,
        }

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """计算Pearson相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if denom_x == 0 or denom_y == 0:
            return 0.0
        return numerator / (denom_x * denom_y)
