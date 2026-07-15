"""
Constitutional AI Critic - 宪章AI自我批评生成器

本模块实现了完整的自我批评生成系统，包括规则评估、违规评分、
改进建议生成、多视角批评和批评链追踪功能。所有实现使用纯Python，
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
    List, Dict, Tuple, Optional, Any, Set, Callable
)
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# 辅助函数
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """将文本分词为小写token列表"""
    text = text.lower().strip()
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


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ============================================================================
# 数据结构
# ============================================================================

class Severity(Enum):
    """违规严重程度"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CritiquePerspective(Enum):
    """批评视角"""
    SAFETY = "safety"
    ETHICS = "ethics"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    FAIRNESS = "fairness"
    PRIVACY = "privacy"
    HELPFULNESS = "helpfulness"


@dataclass
class ConstitutionRule:
    """宪章规则"""
    rule_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    severity: Severity = Severity.MEDIUM
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    threshold: float = 0.5
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Violation:
    """违规记录"""
    violation_id: str = ""
    rule_id: str = ""
    rule_name: str = ""
    severity: Severity = Severity.MEDIUM
    score: float = 0.0
    location: str = ""
    description: str = ""
    matched_text: str = ""
    context: str = ""
    timestamp: float = 0.0


@dataclass
class ImprovementSuggestion:
    """改进建议"""
    suggestion_id: str = ""
    violation_id: str = ""
    priority: int = 0
    description: str = ""
    original_text: str = ""
    suggested_text: str = ""
    reasoning: str = ""
    expected_impact: float = 0.0


@dataclass
class PerspectiveCritique:
    """单视角批评"""
    perspective: CritiquePerspective = CritiquePerspective.SAFETY
    score: float = 0.0
    violations: List[Violation] = field(default_factory=list)
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class CritiqueReport:
    """批评报告"""
    report_id: str = ""
    content_hash: str = ""
    overall_score: float = 0.0
    severity_counts: Dict[str, int] = field(default_factory=dict)
    violations: List[Violation] = field(default_factory=list)
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)
    perspective_critiques: List[PerspectiveCritique] = field(default_factory=list)
    chain_depth: int = 0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueChainEntry:
    """批评链条目"""
    chain_id: str = ""
    iteration: int = 0
    content: str = ""
    report: Optional[CritiqueReport] = None
    revision_applied: bool = False
    improvement_delta: float = 0.0
    timestamp: float = 0.0


# ============================================================================
# RuleEvaluator - 规则评估器
# ============================================================================

class RuleEvaluator:
    """规则评估器：对内容进行逐条规则评估"""

    def __init__(self, rules: Optional[List[ConstitutionRule]] = None):
        self._rules: List[ConstitutionRule] = rules or []
        self._pattern_cache: Dict[str, re.Pattern] = {}
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)
        self._build_keyword_index()

    def add_rule(self, rule: ConstitutionRule) -> None:
        """添加规则"""
        self._rules.append(rule)
        self._build_keyword_index()

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        self._build_keyword_index()
        return len(self._rules) < before

    def set_rules(self, rules: List[ConstitutionRule]) -> None:
        """设置规则列表"""
        self._rules = list(rules)
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        """构建关键词索引"""
        self._keyword_index.clear()
        for rule in self._rules:
            for kw in rule.keywords:
                self._keyword_index[kw.lower()].append(rule.rule_id)

    def _compile_pattern(self, pattern_str: str) -> re.Pattern:
        """编译正则表达式（带缓存）"""
        if pattern_str not in self._pattern_cache:
            try:
                self._pattern_cache[pattern_str] = re.compile(
                    pattern_str, re.IGNORECASE | re.DOTALL
                )
            except re.error:
                self._pattern_cache[pattern_str] = re.compile(
                    re.escape(pattern_str), re.IGNORECASE
                )
        return self._pattern_cache[pattern_str]

    def evaluate_single_rule(
        self, content: str, rule: ConstitutionRule
    ) -> Tuple[float, List[str]]:
        """
        评估单条规则。
        返回 (违规分数, 匹配到的文本片段列表)
        """
        if not content.strip():
            return 0.0, []

        matched_segments: List[str] = []
        total_score = 0.0

        # 关键词匹配
        keyword_matches = 0
        content_lower = content.lower()
        for kw in rule.keywords:
            kw_lower = kw.lower()
            if kw_lower in content_lower:
                keyword_matches += 1
                # 提取上下文
                idx = content_lower.index(kw_lower)
                start = max(0, idx - 30)
                end = min(len(content), idx + len(kw) + 30)
                segment = content[start:end]
                if segment not in matched_segments:
                    matched_segments.append(segment)

        keyword_score = keyword_matches / max(len(rule.keywords), 1)
        total_score += keyword_score * 0.4

        # 正则模式匹配
        pattern_matches = 0
        for pattern_str in rule.patterns:
            pattern = self._compile_pattern(pattern_str)
            matches = pattern.findall(content)
            if matches:
                pattern_matches += 1
                for m in matches:
                    match_text = m if isinstance(m, str) else str(m)
                    if match_text and match_text not in matched_segments:
                        matched_segments.append(match_text[:100])

        pattern_score = pattern_matches / max(len(rule.patterns), 1)
        total_score += pattern_score * 0.4

        # 语义相似度（基于TF向量）
        if rule.description:
            content_vec = _build_tf_vector(content)
            desc_vec = _build_tf_vector(rule.description)
            semantic_sim = _cosine_similarity(content_vec, desc_vec)
            total_score += semantic_sim * 0.2

        # 应用阈值
        final_score = max(0.0, min(1.0, total_score))
        if final_score < rule.threshold:
            final_score = 0.0

        return final_score, matched_segments

    def evaluate_all_rules(
        self, content: str
    ) -> List[Tuple[ConstitutionRule, float, List[str]]]:
        """评估所有规则"""
        results: List[Tuple[ConstitutionRule, float, List[str]]] = []
        for rule in self._rules:
            score, segments = self.evaluate_single_rule(content, rule)
            if score > 0:
                results.append((rule, score, segments))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def evaluate_by_category(
        self, content: str, category: str
    ) -> List[Tuple[ConstitutionRule, float, List[str]]]:
        """按类别评估规则"""
        category_rules = [r for r in self._rules if r.category == category]
        results: List[Tuple[ConstitutionRule, float, List[str]]] = []
        for rule in category_rules:
            score, segments = self.evaluate_single_rule(content, rule)
            if score > 0:
                results.append((rule, score, segments))
        return results


# ============================================================================
# ViolationScorer - 违规评分器
# ============================================================================

class ViolationScorer:
    """违规评分器：综合计算违规严重程度和分数"""

    def __init__(
        self,
        severity_weights: Optional[Dict[Severity, float]] = None,
        overlap_penalty: float = 0.3,
        context_multiplier: float = 1.2,
    ):
        self._severity_weights = severity_weights or {
            Severity.NONE: 0.0,
            Severity.LOW: 0.2,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.8,
            Severity.CRITICAL: 1.0,
        }
        self._overlap_penalty = overlap_penalty
        self._context_multiplier = context_multiplier

    def score_violation(
        self,
        rule: ConstitutionRule,
        raw_score: float,
        matched_segments: List[str],
        content: str,
    ) -> Violation:
        """计算单条违规的最终分数"""
        severity = self._classify_severity(raw_score, rule)
        weighted_score = raw_score * self._severity_weights[severity]

        # 上下文增强：检查匹配片段周围的上下文
        context_score = self._evaluate_context(matched_segments, content)
        weighted_score *= (1.0 + context_score * self._context_multiplier)

        # 规则权重
        weighted_score *= rule.weight

        # 限制在[0, 1]范围
        weighted_score = max(0.0, min(1.0, weighted_score))

        violation = Violation(
            violation_id=_generate_id(),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=severity,
            score=weighted_score,
            location=self._find_location(matched_segments, content),
            description=f"违反规则 [{rule.name}]: {rule.description}",
            matched_text="; ".join(matched_segments[:3]),
            context=content[:200],
            timestamp=time.time(),
        )
        return violation

    def _classify_severity(
        self, raw_score: float, rule: ConstitutionRule
    ) -> Severity:
        """根据分数和规则严重程度分类"""
        if raw_score >= 0.9:
            return Severity.CRITICAL
        elif raw_score >= 0.7:
            return Severity.HIGH
        elif raw_score >= 0.5:
            return Severity.MEDIUM
        elif raw_score >= 0.3:
            return Severity.LOW
        else:
            return Severity.NONE

    def _evaluate_context(
        self, segments: List[str], content: str
    ) -> float:
        """评估匹配片段的上下文严重性"""
        if not segments or not content:
            return 0.0

        context_score = 0.0
        # 检查匹配片段是否出现在内容开头或结尾（更重要的位置）
        content_len = len(content)
        for seg in segments:
            pos = content.find(seg)
            if pos >= 0:
                relative_pos = pos / max(content_len, 1)
                # 开头和结尾位置权重更高
                if relative_pos < 0.1:
                    context_score += 0.3
                elif relative_pos > 0.9:
                    context_score += 0.2
                else:
                    context_score += 0.1

        return min(1.0, context_score / max(len(segments), 1))

    def _find_location(
        self, segments: List[str], content: str
    ) -> str:
        """找到违规位置描述"""
        if not segments:
            return ""
        lines = content.split("\n")
        for seg in segments:
            for i, line in enumerate(lines):
                if seg[:20] in line:
                    return f"第{i + 1}行"
        return f"内容中(共{len(lines)}行)"

    def score_violations_batch(
        self,
        evaluations: List[Tuple[ConstitutionRule, float, List[str]]],
        content: str,
    ) -> List[Violation]:
        """批量评分违规"""
        violations: List[Violation] = []
        seen_rules: Set[str] = set()

        for rule, score, segments in evaluations:
            # 检查是否与已有违规重叠
            if rule.rule_id in seen_rules:
                score *= (1.0 - self._overlap_penalty)
            seen_rules.add(rule.rule_id)

            violation = self.score_violation(rule, score, segments, content)
            violations.append(violation)

        violations.sort(key=lambda v: v.score, reverse=True)
        return violations


# ============================================================================
# SuggestionGenerator - 改进建议生成器
# ============================================================================

class SuggestionGenerator:
    """改进建议生成器：基于违规生成具体的改进建议"""

    def __init__(
        self,
        max_suggestions_per_violation: int = 3,
        min_expected_impact: float = 0.1,
    ):
        self._max_suggestions = max_suggestions_per_violation
        self._min_impact = min_expected_impact
        self._templates: Dict[str, List[str]] = {
            "safety": [
                "建议移除或改写涉及不安全内容的表述，使用更安全的替代方案",
                "考虑从更中性的角度重新表述，避免潜在的安全风险",
                "建议增加安全声明或上下文限定，降低误解风险",
            ],
            "ethics": [
                "建议从伦理角度重新审视表述，确保公平和尊重",
                "考虑添加多元视角，避免单一立场的偏见",
                "建议使用更包容的语言，尊重不同群体的感受",
            ],
            "accuracy": [
                "建议核实相关事实和数据，确保信息准确",
                "考虑添加引用来源或数据支撑，增强可信度",
                "建议使用更精确的表述，避免模糊或误导性描述",
            ],
            "clarity": [
                "建议简化表述，使用更清晰直接的语言",
                "考虑添加示例或解释，帮助读者理解",
                "建议重组内容结构，提高逻辑连贯性",
            ],
            "privacy": [
                "建议移除或匿名化个人信息，保护隐私",
                "考虑使用概括性描述替代具体个人数据",
                "建议添加隐私保护声明，明确数据使用范围",
            ],
            "default": [
                "建议重新审视相关内容，确保符合规范要求",
                "考虑从不同角度评估内容的适当性",
                "建议参考最佳实践进行改进",
            ],
        }

    def generate_suggestions(
        self,
        violation: Violation,
        content: str,
        existing_suggestions: Optional[List[ImprovementSuggestion]] = None,
    ) -> List[ImprovementSuggestion]:
        """为单条违规生成改进建议"""
        suggestions: List[ImprovementSuggestion] = []
        existing_ids = {s.suggestion_id for s in (existing_suggestions or [])}

        # 确定建议模板类别
        category = self._infer_category(violation)
        templates = self._templates.get(category, self._templates["default"])

        for idx, template in enumerate(templates):
            if len(suggestions) >= self._max_suggestions:
                break

            suggestion = self._build_suggestion(
                violation=violation,
                content=content,
                template=template,
                priority=idx + 1,
                existing_ids=existing_ids,
            )
            if suggestion and suggestion.expected_impact >= self._min_impact:
                suggestions.append(suggestion)
                existing_ids.add(suggestion.suggestion_id)

        return suggestions

    def _infer_category(self, violation: Violation) -> str:
        """推断违规类别"""
        rule_name_lower = violation.rule_name.lower()
        desc_lower = violation.description.lower()
        combined = rule_name_lower + " " + desc_lower

        category_keywords = {
            "safety": ["安全", "safety", "危险", "danger", "harm", "伤害"],
            "ethics": ["伦理", "ethics", "道德", "moral", "公平", "fairness"],
            "accuracy": ["准确", "accuracy", "事实", "fact", "错误", "error"],
            "clarity": ["清晰", "clarity", "模糊", "ambiguous", "复杂", "complex"],
            "privacy": ["隐私", "privacy", "个人", "personal", "数据", "data"],
        }

        best_category = "default"
        best_score = 0.0
        for cat, keywords in category_keywords.items():
            score = sum(1.0 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_category = cat

        return best_category

    def _build_suggestion(
        self,
        violation: Violation,
        content: str,
        template: str,
        priority: int,
        existing_ids: Set[str],
    ) -> Optional[ImprovementSuggestion]:
        """构建单条建议"""
        suggestion_id = _generate_id()
        if suggestion_id in existing_ids:
            return None

        # 生成建议文本
        original_text = violation.matched_text[:100] if violation.matched_text else ""
        suggested_text = self._generate_suggested_text(
            original_text, template, violation
        )
        expected_impact = self._estimate_impact(violation, template)

        return ImprovementSuggestion(
            suggestion_id=suggestion_id,
            violation_id=violation.violation_id,
            priority=priority,
            description=template,
            original_text=original_text,
            suggested_text=suggested_text,
            reasoning=f"基于违规 [{violation.rule_name}] (分数: {violation.score:.2f})",
            expected_impact=expected_impact,
        )

    def _generate_suggested_text(
        self,
        original: str,
        template: str,
        violation: Violation,
    ) -> str:
        """生成建议的替换文本"""
        if not original:
            return ""

        # 基于违规严重程度决定修改策略
        if violation.severity in (Severity.CRITICAL, Severity.HIGH):
            # 高严重度：大幅改写
            words = original.split()
            if len(words) > 3:
                return "[建议完全重写此部分内容]"
            return "[建议移除此内容]"

        # 中低严重度：微调
        modified = original
        # 添加限定词
        qualifiers = ["在适当的情况下", "在合规范围内", "以安全的方式"]
        if len(modified) > 10:
            qualifier = qualifiers[hash(violation.violation_id) % len(qualifiers)]
            modified = qualifier + "，" + modified

        return modified

    def _estimate_impact(
        self, violation: Violation, template: str
    ) -> float:
        """估算建议的预期影响"""
        base_impact = violation.score * 0.8
        # 模板越长通常越具体，影响越大
        length_factor = min(len(template) / 100.0, 1.0) * 0.2
        return min(1.0, base_impact + length_factor)

    def generate_batch_suggestions(
        self,
        violations: List[Violation],
        content: str,
    ) -> List[ImprovementSuggestion]:
        """批量生成建议"""
        all_suggestions: List[ImprovementSuggestion] = []
        for violation in violations:
            suggestions = self.generate_suggestions(
                violation, content, all_suggestions
            )
            all_suggestions.extend(suggestions)
        all_suggestions.sort(key=lambda s: s.expected_impact, reverse=True)
        return all_suggestions


# ============================================================================
# MultiPerspectiveCritic - 多视角批评器
# ============================================================================

class MultiPerspectiveCritic:
    """多视角批评器：从多个维度评估内容"""

    def __init__(
        self,
        perspectives: Optional[List[CritiquePerspective]] = None,
        weights: Optional[Dict[CritiquePerspective, float]] = None,
    ):
        self._perspectives = perspectives or list(CritiquePerspective)
        self._weights = weights or {
            CritiquePerspective.SAFETY: 0.25,
            CritiquePerspective.ETHICS: 0.20,
            CritiquePerspective.ACCURACY: 0.15,
            CritiquePerspective.CLARITY: 0.10,
            CritiquePerspective.COMPLETENESS: 0.10,
            CritiquePerspective.FAIRNESS: 0.10,
            CritiquePerspective.PRIVACY: 0.05,
            CritiquePerspective.HELPFULNESS: 0.05,
        }
        self._perspective_keywords: Dict[CritiquePerspective, List[str]] = {
            CritiquePerspective.SAFETY: [
                "危险", "安全", "伤害", "暴力", "武器", "攻击",
                "dangerous", "harm", "violence", "weapon", "attack",
                "kill", "death", "illegal", "crime",
            ],
            CritiquePerspective.ETHICS: [
                "伦理", "道德", "正义", "公平", "歧视", "偏见",
                "ethics", "moral", "justice", "discrimination", "bias",
                "racist", "sexist", "prejudice",
            ],
            CritiquePerspective.ACCURACY: [
                "准确", "正确", "错误", "事实", "数据", "统计",
                "accurate", "correct", "wrong", "fact", "data",
                "statistic", "source", "citation",
            ],
            CritiquePerspective.CLARITY: [
                "清晰", "模糊", "复杂", "简单", "易懂", "晦涩",
                "clear", "ambiguous", "complex", "simple", "understand",
                "confusing", "obscure",
            ],
            CritiquePerspective.COMPLETENESS: [
                "完整", "全面", "遗漏", "缺失", "部分", "充分",
                "complete", "comprehensive", "missing", "partial",
                "sufficient", "thorough",
            ],
            CritiquePerspective.FAIRNESS: [
                "公平", "公正", "平衡", "客观", "片面", "偏袒",
                "fair", "balanced", "objective", "biased", "one-sided",
                "neutral", "impartial",
            ],
            CritiquePerspective.PRIVACY: [
                "隐私", "个人信息", "数据保护", "姓名", "地址",
                "privacy", "personal", "data", "name", "address",
                "phone", "email", "identity",
            ],
            CritiquePerspective.HELPFULNESS: [
                "有用", "帮助", "实用", "建议", "指导", "解决方案",
                "helpful", "useful", "practical", "advice", "guide",
                "solution", "recommendation",
            ],
        }

    def critique_from_perspective(
        self,
        content: str,
        perspective: CritiquePerspective,
    ) -> PerspectiveCritique:
        """从单个视角进行批评"""
        keywords = self._perspective_keywords[perspective]
        content_lower = content.lower()

        # 关键词命中分析
        hits: List[str] = []
        for kw in keywords:
            if kw.lower() in content_lower:
                hits.append(kw)

        # 计算该视角的分数（0=差, 1=好）
        hit_ratio = len(hits) / max(len(keywords), 1)
        # 命中率高不一定好——对于安全和隐私，命中多意味着风险高
        if perspective in (CritiquePerspective.SAFETY, CritiquePerspective.PRIVACY):
            # 安全和隐私：命中越多，分数越低（风险越高）
            score = max(0.0, 1.0 - hit_ratio * 2.0)
        else:
            # 其他视角：命中越多，内容越相关，但需要综合评估
            score = min(1.0, hit_ratio * 1.5 + 0.3)

        # 内容长度和结构评估
        length_score = self._evaluate_content_structure(content, perspective)
        score = score * 0.7 + length_score * 0.3

        # 生成推理
        reasoning = self._generate_perspective_reasoning(
            content, perspective, hits, score
        )

        # 置信度
        confidence = self._compute_confidence(hits, keywords, content)

        return PerspectiveCritique(
            perspective=perspective,
            score=score,
            violations=[],
            suggestions=[],
            reasoning=reasoning,
            confidence=confidence,
        )

    def _evaluate_content_structure(
        self, content: str, perspective: CritiquePerspective
    ) -> float:
        """评估内容结构"""
        if not content.strip():
            return 0.0

        # 基本结构指标
        sentences = re.split(r'[。！？.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = (
            sum(len(s) for s in sentences) / max(len(sentences), 1)
        )

        # 句子长度适中得分高
        if 10 <= avg_sentence_len <= 50:
            length_score = 1.0
        elif 5 <= avg_sentence_len <= 100:
            length_score = 0.7
        else:
            length_score = 0.4

        # 段落结构
        paragraphs = content.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        structure_score = min(1.0, len(paragraphs) / 3.0) if paragraphs else 0.3

        return (length_score + structure_score) / 2.0

    def _generate_perspective_reasoning(
        self,
        content: str,
        perspective: CritiquePerspective,
        hits: List[str],
        score: float,
    ) -> str:
        """生成视角推理文本"""
        perspective_names = {
            CritiquePerspective.SAFETY: "安全性",
            CritiquePerspective.ETHICS: "伦理性",
            CritiquePerspective.ACCURACY: "准确性",
            CritiquePerspective.CLARITY: "清晰度",
            CritiquePerspective.COMPLETENESS: "完整性",
            CritiquePerspective.FAIRNESS: "公平性",
            CritiquePerspective.PRIVACY: "隐私保护",
            CritiquePerspective.HELPFULNESS: "有用性",
        }
        name = perspective_names.get(perspective, perspective.value)

        if score >= 0.8:
            assessment = "表现良好"
        elif score >= 0.6:
            assessment = "基本合格，存在改进空间"
        elif score >= 0.4:
            assessment = "需要关注，存在明显问题"
        else:
            assessment = "严重不足，需要立即改进"

        hit_info = f"（命中关键词: {', '.join(hits[:5])}）" if hits else ""
        return (
            f"从{name}视角评估: {assessment}，"
            f"得分 {score:.2f}{hit_info}"
        )

    def _compute_confidence(
        self, hits: List[str], keywords: List[str], content: str
    ) -> float:
        """计算评估置信度"""
        if not content.strip():
            return 0.0
        content_len = len(content)
        # 内容越长，评估越可靠
        length_factor = min(1.0, content_len / 500.0)
        # 关键词命中越多，置信度越高
        hit_factor = min(1.0, len(hits) / max(len(keywords) * 0.3, 1))
        return length_factor * 0.4 + hit_factor * 0.6

    def critique_all_perspectives(
        self, content: str
    ) -> List[PerspectiveCritique]:
        """从所有视角进行批评"""
        critiques: List[PerspectiveCritique] = []
        for perspective in self._perspectives:
            critique = self.critique_from_perspective(content, perspective)
            critiques.append(critique)
        return critiques

    def compute_weighted_score(
        self, critiques: List[PerspectiveCritique]
    ) -> float:
        """计算加权综合分数"""
        total_weight = 0.0
        weighted_sum = 0.0
        for critique in critiques:
            w = self._weights.get(critique.perspective, 0.0)
            weighted_sum += critique.score * w * critique.confidence
            total_weight += w * critique.confidence
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight


# ============================================================================
# CritiqueChain - 批评链
# ============================================================================

class CritiqueChain:
    """批评链：追踪多轮批评-修订的迭代过程"""

    def __init__(
        self,
        max_iterations: int = 5,
        improvement_threshold: float = 0.05,
        convergence_threshold: float = 0.01,
    ):
        self._max_iterations = max_iterations
        self._improvement_threshold = improvement_threshold
        self._convergence_threshold = convergence_threshold
        self._entries: List[CritiqueChainEntry] = []
        self._chain_id: str = _generate_id()

    @property
    def chain_id(self) -> str:
        return self._chain_id

    @property
    def entries(self) -> List[CritiqueChainEntry]:
        return list(self._entries)

    @property
    def current_iteration(self) -> int:
        return len(self._entries)

    def add_entry(
        self,
        content: str,
        report: CritiqueReport,
        revision_applied: bool = False,
    ) -> CritiqueChainEntry:
        """添加批评链条目"""
        iteration = len(self._entries)
        improvement_delta = 0.0

        if self._entries:
            prev_score = self._entries[-1].report.overall_score if self._entries[-1].report else 0.0
            improvement_delta = report.overall_score - prev_score

        entry = CritiqueChainEntry(
            chain_id=self._chain_id,
            iteration=iteration,
            content=content,
            report=report,
            revision_applied=revision_applied,
            improvement_delta=improvement_delta,
            timestamp=time.time(),
        )
        self._entries.append(entry)
        return entry

    def should_continue(self) -> bool:
        """判断是否应该继续迭代"""
        if len(self._entries) >= self._max_iterations:
            return False

        if len(self._entries) < 2:
            return True

        # 检查改进幅度
        last_delta = self._entries[-1].improvement_delta
        if abs(last_delta) < self._convergence_threshold:
            return False

        # 检查分数是否已经足够高
        last_report = self._entries[-1].report
        if last_report and last_report.overall_score >= 0.95:
            return False

        return True

    def get_best_iteration(self) -> Optional[int]:
        """获取最佳迭代次数"""
        if not self._entries:
            return None
        best_idx = 0
        best_score = -1.0
        for i, entry in enumerate(self._entries):
            if entry.report and entry.report.overall_score > best_score:
                best_score = entry.report.overall_score
                best_idx = i
        return best_idx

    def get_improvement_trajectory(self) -> List[float]:
        """获取改进轨迹（分数列表）"""
        trajectory: List[float] = []
        for entry in self._entries:
            if entry.report:
                trajectory.append(entry.report.overall_score)
        return trajectory

    def compute_total_improvement(self) -> float:
        """计算总改进量"""
        if len(self._entries) < 2:
            return 0.0
        first_score = self._entries[0].report.overall_score if self._entries[0].report else 0.0
        last_score = self._entries[-1].report.overall_score if self._entries[-1].report else 0.0
        return last_score - first_score

    def get_summary(self) -> Dict[str, Any]:
        """获取批评链摘要"""
        return {
            "chain_id": self._chain_id,
            "total_iterations": len(self._entries),
            "total_improvement": self.compute_total_improvement(),
            "best_iteration": self.get_best_iteration(),
            "trajectory": self.get_improvement_trajectory(),
            "converged": not self.should_continue() if self._entries else False,
        }


# ============================================================================
# ConstitutionCritic - 宪章批评器（主入口）
# ============================================================================

class ConstitutionCritic:
    """
    宪章批评器：整合所有组件，提供统一的自我批评接口。

    使用方法:
        critic = ConstitutionCritic(rules)
        report = critic.criticize(content)
    """

    def __init__(
        self,
        rules: Optional[List[ConstitutionRule]] = None,
        max_chain_iterations: int = 5,
        enable_multi_perspective: bool = True,
    ):
        self._rule_evaluator = RuleEvaluator(rules)
        self._violation_scorer = ViolationScorer()
        self._suggestion_generator = SuggestionGenerator()
        self._multi_perspective = MultiPerspectiveCritic()
        self._critique_chain = CritiqueChain(
            max_iterations=max_chain_iterations
        )
        self._enable_multi_perspective = enable_multi_perspective

    @property
    def rule_evaluator(self) -> RuleEvaluator:
        return self._rule_evaluator

    @property
    def violation_scorer(self) -> ViolationScorer:
        return self._violation_scorer

    @property
    def suggestion_generator(self) -> SuggestionGenerator:
        return self._suggestion_generator

    @property
    def multi_perspective_critic(self) -> MultiPerspectiveCritic:
        return self._multi_perspective

    @property
    def critique_chain(self) -> CritiqueChain:
        return self._critique_chain

    def add_rule(self, rule: ConstitutionRule) -> None:
        """添加规则"""
        self._rule_evaluator.add_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        return self._rule_evaluator.remove_rule(rule_id)

    def criticize(self, content: str) -> CritiqueReport:
        """
        对内容进行完整的批评分析。

        步骤:
        1. 规则评估
        2. 违规评分
        3. 建议生成
        4. 多视角批评（可选）
        5. 生成报告
        """
        # 步骤1: 规则评估
        evaluations = self._rule_evaluator.evaluate_all_rules(content)

        # 步骤2: 违规评分
        violations = self._violation_scorer.score_violations_batch(
            evaluations, content
        )

        # 步骤3: 建议生成
        suggestions = self._suggestion_generator.generate_batch_suggestions(
            violations, content
        )

        # 步骤4: 多视角批评
        perspective_critiques: List[PerspectiveCritique] = []
        if self._enable_multi_perspective:
            perspective_critiques = (
                self._multi_perspective.critique_all_perspectives(content)
            )

        # 步骤5: 计算综合分数
        overall_score = self._compute_overall_score(
            violations, perspective_critiques
        )

        # 统计严重程度
        severity_counts: Dict[str, int] = {}
        for sev in Severity:
            severity_counts[sev.value] = sum(
                1 for v in violations if v.severity == sev
            )

        # 生成报告
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        report = CritiqueReport(
            report_id=_generate_id(),
            content_hash=content_hash,
            overall_score=overall_score,
            severity_counts=severity_counts,
            violations=violations,
            suggestions=suggestions,
            perspective_critiques=perspective_critiques,
            chain_depth=self._critique_chain.current_iteration,
            timestamp=time.time(),
            metadata={
                "content_length": len(content),
                "rule_count": len(self._rule_evaluator._rules),
                "perspective_count": len(perspective_critiques),
            },
        )

        return report

    def _compute_overall_score(
        self,
        violations: List[Violation],
        perspective_critiques: List[PerspectiveCritique],
    ) -> float:
        """计算综合分数"""
        if not violations and not perspective_critiques:
            return 1.0

        # 基于违规的分数
        violation_penalty = sum(v.score for v in violations)
        violation_score = max(0.0, 1.0 - violation_penalty)

        # 基于多视角的分数
        if perspective_critiques:
            perspective_score = self._multi_perspective.compute_weighted_score(
                perspective_critiques
            )
        else:
            perspective_score = 1.0

        # 加权综合
        if perspective_critiques:
            overall = violation_score * 0.6 + perspective_score * 0.4
        else:
            overall = violation_score

        return max(0.0, min(1.0, overall))

    def criticize_with_chain(
        self, content: str
    ) -> Tuple[CritiqueReport, CritiqueChain]:
        """执行带批评链的批评"""
        self._critique_chain = CritiqueChain(
            max_iterations=self._critique_chain._max_iterations
        )
        report = self.criticize(content)
        self._critique_chain.add_entry(content, report)
        return report, self._critique_chain

    def quick_critique(self, content: str) -> Dict[str, Any]:
        """快速批评（返回简化结果）"""
        report = self.criticize(content)
        return {
            "score": report.overall_score,
            "violation_count": len(report.violations),
            "suggestion_count": len(report.suggestions),
            "severity_summary": report.severity_counts,
            "top_violations": [
                {
                    "rule": v.rule_name,
                    "score": v.score,
                    "severity": v.severity.value,
                }
                for v in report.violations[:3]
            ],
        }

    def batch_critique(
        self, contents: List[str]
    ) -> List[CritiqueReport]:
        """批量批评"""
        return [self.criticize(content) for content in contents]

    def compare_contents(
        self, content_a: str, content_b: str
    ) -> Dict[str, Any]:
        """比较两个内容的批评结果"""
        report_a = self.criticize(content_a)
        report_b = self.criticize(content_b)

        return {
            "content_a_score": report_a.overall_score,
            "content_b_score": report_b.overall_score,
            "score_difference": report_b.overall_score - report_a.overall_score,
            "content_a_violations": len(report_a.violations),
            "content_b_violations": len(report_b.violations),
            "improved": report_b.overall_score > report_a.overall_score,
        }
