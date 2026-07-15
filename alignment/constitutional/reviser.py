"""
Constitutional AI Reviser - 宪章AI内容修订引擎

本模块实现了完整的内容修订系统，包括基于违规的修订、迭代精炼循环、
风格保持、最小编辑策略和修订质量评分功能。所有实现使用纯Python，
不依赖任何外部库。
"""

import math
import random
import re
import hashlib
import json
import time
import threading
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


def _diff_ratio(s1: str, s2: str) -> float:
    """计算两个字符串的相似度比率"""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = _levenshtein_distance(s1, s2)
    return 1.0 - dist / max_len


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


class RevisionStrategy(Enum):
    """修订策略"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    FULL_REWRITE = "full_rewrite"


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
class RevisionEdit:
    """单次修订编辑"""
    edit_id: str = ""
    start_pos: int = 0
    end_pos: int = 0
    original_text: str = ""
    revised_text: str = ""
    violation_id: str = ""
    strategy: RevisionStrategy = RevisionStrategy.MINIMAL
    confidence: float = 0.0


@dataclass
class RevisionHistoryEntry:
    """修订历史条目"""
    entry_id: str = ""
    iteration: int = 0
    original_content: str = ""
    revised_content: str = ""
    edits: List[RevisionEdit] = field(default_factory=list)
    violations_addressed: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    style_preservation: float = 0.0
    timestamp: float = 0.0


@dataclass
class RevisionReport:
    """修订报告"""
    report_id: str = ""
    original_content: str = ""
    final_content: str = ""
    total_iterations: int = 0
    total_edits: int = 0
    violations_before: int = 0
    violations_after: int = 0
    quality_score: float = 0.0
    style_preservation: float = 0.0
    edit_distance: int = 0
    similarity_ratio: float = 0.0
    history: List[RevisionHistoryEntry] = field(default_factory=list)
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# StylePreserver - 风格保持器
# ============================================================================

class StylePreserver:
    """
    风格保持器：在修订过程中保持原文风格特征。

    分析维度:
    - 句子长度分布
    - 词汇多样性
    - 标点符号使用模式
    - 段落结构
    - 语气特征
    """

    def __init__(self, preservation_weight: float = 0.7):
        self._preservation_weight = preservation_weight

    def extract_style_features(self, text: str) -> Dict[str, Any]:
        """提取文本风格特征"""
        if not text.strip():
            return self._empty_features()

        sentences = self._split_sentences(text)
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        tokens = _tokenize(text)

        # 句子长度特征
        sentence_lengths = [len(s) for s in sentences]
        avg_sentence_len = (
            sum(sentence_lengths) / max(len(sentence_lengths), 1)
        )
        std_sentence_len = self._std(sentence_lengths) if sentence_lengths else 0.0

        # 词汇多样性
        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / max(len(tokens), 1)

        # 标点符号模式
        punctuation = self._analyze_punctuation(text)

        # 段落结构
        para_lengths = [len(p) for p in paragraphs]
        avg_para_len = sum(para_lengths) / max(len(para_lengths), 1)

        # 语气特征
        tone = self._analyze_tone(text)

        return {
            "avg_sentence_length": avg_sentence_len,
            "std_sentence_length": std_sentence_len,
            "vocabulary_diversity": ttr,
            "punctuation_pattern": punctuation,
            "avg_paragraph_length": avg_para_len,
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "total_length": len(text),
            "tone": tone,
            "token_count": len(tokens),
        }

    def _empty_features(self) -> Dict[str, Any]:
        """返回空特征字典"""
        return {
            "avg_sentence_length": 0.0,
            "std_sentence_length": 0.0,
            "vocabulary_diversity": 0.0,
            "punctuation_pattern": {},
            "avg_paragraph_length": 0.0,
            "paragraph_count": 0,
            "sentence_count": 0,
            "total_length": 0,
            "tone": {},
            "token_count": 0,
        }

    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        sentences = re.split(r'[。！？.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def _analyze_punctuation(self, text: str) -> Dict[str, float]:
        """分析标点符号使用模式"""
        punct_marks = "。，！？、；：""''…—,.!?;:'\"-"
        total_chars = len(text)
        if total_chars == 0:
            return {}

        pattern: Dict[str, float] = {}
        for mark in punct_marks:
            count = text.count(mark)
            if count > 0:
                pattern[mark] = count / total_chars

        return pattern

    def _analyze_tone(self, text: str) -> Dict[str, float]:
        """分析语气特征"""
        tone_indicators = {
            "formal": ["因此", "综上所述", "鉴于", "故", "therefore", "thus", "hence"],
            "casual": ["哈哈", "嗯", "好吧", "嘛", "oh", "well", "like", "yeah"],
            "assertive": ["必须", "一定", "毫无疑问", "显然", "must", "certainly", "definitely"],
            "tentative": ["可能", "也许", "大概", "或许", "maybe", "perhaps", "might"],
            "questioning": ["？", "?", "是否", "怎样", "如何", "what", "how", "why"],
        }

        text_lower = text.lower()
        tone: Dict[str, float] = {}
        for tone_type, indicators in tone_indicators.items():
            count = sum(1 for ind in indicators if ind in text_lower)
            tone[tone_type] = count / max(len(indicators), 1)

        return tone

    def compute_preservation_score(
        self, original: str, revised: str
    ) -> float:
        """计算风格保持分数"""
        orig_features = self.extract_style_features(original)
        rev_features = self.extract_style_features(revised)

        if not orig_features["total_length"] or not rev_features["total_length"]:
            return 0.0

        scores: List[float] = []

        # 句子长度保持
        orig_avg = orig_features["avg_sentence_length"]
        rev_avg = rev_features["avg_sentence_length"]
        if orig_avg > 0:
            length_ratio = min(rev_avg, orig_avg) / max(rev_avg, orig_avg)
            scores.append(length_ratio)

        # 词汇多样性保持
        orig_ttr = orig_features["vocabulary_diversity"]
        rev_ttr = rev_features["vocabulary_diversity"]
        if orig_ttr > 0:
            ttr_ratio = min(rev_ttr, orig_ttr) / max(rev_ttr, orig_ttr)
            scores.append(ttr_ratio)

        # 段落结构保持
        orig_paras = orig_features["paragraph_count"]
        rev_paras = rev_features["paragraph_count"]
        if orig_paras > 0:
            para_ratio = min(rev_paras, orig_paras) / max(rev_paras, orig_paras)
            scores.append(para_ratio)

        # 标点符号模式保持
        orig_punct = orig_features["punctuation_pattern"]
        rev_punct = rev_features["punctuation_pattern"]
        if orig_punct and rev_punct:
            all_puncts = set(orig_punct.keys()) | set(rev_punct.keys())
            punct_diffs = []
            for p in all_puncts:
                o_val = orig_punct.get(p, 0.0)
                r_val = rev_punct.get(p, 0.0)
                if o_val > 0 or r_val > 0:
                    diff = abs(o_val - r_val) / max(o_val, r_val, 0.001)
                    punct_diffs.append(1.0 - min(diff, 1.0))
            if punct_diffs:
                scores.append(sum(punct_diffs) / len(punct_diffs))

        # 语气保持
        orig_tone = orig_features["tone"]
        rev_tone = rev_features["tone"]
        if orig_tone and rev_tone:
            all_tones = set(orig_tone.keys()) | set(rev_tone.keys())
            tone_diffs = []
            for t in all_tones:
                o_val = orig_tone.get(t, 0.0)
                r_val = rev_tone.get(t, 0.0)
                diff = abs(o_val - r_val)
                tone_diffs.append(1.0 - min(diff, 1.0))
            if tone_diffs:
                scores.append(sum(tone_diffs) / len(tone_diffs))

        if not scores:
            return 1.0

        return sum(scores) / len(scores)

    def suggest_style_aware_replacement(
        self,
        original_segment: str,
        suggested_replacement: str,
        full_original: str,
    ) -> str:
        """生成风格感知的替换文本"""
        features = self.extract_style_features(full_original)

        # 调整句子长度
        orig_len = len(original_segment)
        sug_len = len(suggested_replacement)

        if orig_len > 0 and abs(orig_len - sug_len) > orig_len * 0.3:
            # 长度差异过大，需要调整
            if sug_len > orig_len:
                # 建议文本过长，尝试精简
                words = suggested_replacement.split()
                target_words = max(1, int(len(words) * orig_len / max(sug_len, 1)))
                suggested_replacement = " ".join(words[:target_words])
            else:
                # 建议文本过短，添加上下文适配词
                tone = features.get("tone", {})
                if tone.get("formal", 0) > tone.get("casual", 0):
                    suggested_replacement += "，基于以上分析"
                else:
                    suggested_replacement += "，简单来说"

        return suggested_replacement


# ============================================================================
# MinimalEditStrategy - 最小编辑策略
# ============================================================================

class MinimalEditStrategy:
    """
    最小编辑策略：在满足修订目标的前提下，尽可能少地修改原文。

    策略:
    - 精确定位违规文本
    - 生成最小化的替换方案
    - 避免不必要的上下文修改
    - 保持原文结构和连贯性
    """

    def __init__(
        self,
        max_edit_ratio: float = 0.3,
        context_window: int = 50,
    ):
        self._max_edit_ratio = max_edit_ratio
        self._context_window = context_window

    def find_edit_region(
        self, content: str, violation: Violation
    ) -> Tuple[int, int]:
        """找到需要编辑的区域"""
        if not violation.matched_text:
            return 0, 0

        # 尝试精确匹配
        matched = violation.matched_text
        pos = content.find(matched)
        if pos >= 0:
            return pos, pos + len(matched)

        # 模糊匹配：尝试匹配前N个字符
        for prefix_len in range(len(matched), 5, -1):
            prefix = matched[:prefix_len]
            pos = content.find(prefix)
            if pos >= 0:
                return pos, pos + len(prefix)

        # 基于上下文搜索
        if violation.context:
            ctx_pos = content.find(violation.context[:50])
            if ctx_pos >= 0:
                return ctx_pos, ctx_pos + len(violation.context[:50])

        return 0, 0

    def generate_minimal_edit(
        self,
        content: str,
        violation: Violation,
        suggestion: ImprovementSuggestion,
    ) -> Optional[RevisionEdit]:
        """生成最小化编辑"""
        start, end = self.find_edit_region(content, violation)
        if start == end:
            return None

        original_text = content[start:end]
        suggested_text = suggestion.suggested_text

        # 如果建议文本为空，生成默认替换
        if not suggested_text:
            suggested_text = self._generate_default_replacement(
                original_text, violation
            )

        # 检查编辑比例
        edit_ratio = len(suggested_text) / max(len(original_text), 1)
        if edit_ratio > 2.0 or edit_ratio < 0.1:
            # 编辑比例过大，使用更保守的策略
            suggested_text = self._constrain_edit(
                original_text, suggested_text
            )

        return RevisionEdit(
            edit_id=_generate_id(),
            start_pos=start,
            end_pos=end,
            original_text=original_text,
            revised_text=suggested_text,
            violation_id=violation.violation_id,
            strategy=RevisionStrategy.MINIMAL,
            confidence=self._compute_edit_confidence(
                original_text, suggested_text, violation
            ),
        )

    def _generate_default_replacement(
        self, original: str, violation: Violation
    ) -> str:
        """生成默认替换文本"""
        if violation.severity in (Severity.CRITICAL, Severity.HIGH):
            return "[已移除不合规内容]"
        elif violation.severity == Severity.MEDIUM:
            return "[已修订]"
        else:
            # 低严重度：尝试保留大部分原文
            words = original.split()
            if len(words) > 4:
                return " ".join(words[:2]) + " [已调整]"
            return "[已微调]"

    def _constrain_edit(
        self, original: str, suggested: str
    ) -> str:
        """约束编辑范围"""
        max_len = int(len(original) * 1.5)
        if len(suggested) > max_len:
            suggested = suggested[:max_len]
        return suggested

    def _compute_edit_confidence(
        self,
        original: str,
        suggested: str,
        violation: Violation,
    ) -> float:
        """计算编辑置信度"""
        # 基于违规严重程度
        severity_conf = {
            Severity.CRITICAL: 0.9,
            Severity.HIGH: 0.8,
            Severity.MEDIUM: 0.7,
            Severity.LOW: 0.5,
            Severity.NONE: 0.3,
        }
        base_conf = severity_conf.get(violation.severity, 0.5)

        # 基于建议的预期影响
        impact_bonus = min(0.2, violation.score * 0.2)

        return min(1.0, base_conf + impact_bonus)

    def apply_edits(
        self, content: str, edits: List[RevisionEdit]
    ) -> str:
        """应用编辑到内容"""
        if not edits:
            return content

        # 按位置排序（从后往前，避免位置偏移）
        sorted_edits = sorted(edits, key=lambda e: e.start_pos, reverse=True)

        result = content
        for edit in sorted_edits:
            if edit.start_pos < 0 or edit.end_pos > len(result):
                continue
            if edit.start_pos >= edit.end_pos:
                continue
            result = (
                result[:edit.start_pos]
                + edit.revised_text
                + result[edit.end_pos:]
            )

        return result

    def resolve_overlapping_edits(
        self, edits: List[RevisionEdit]
    ) -> List[RevisionEdit]:
        """解决重叠编辑"""
        if not edits:
            return []

        sorted_edits = sorted(edits, key=lambda e: e.start_pos)
        resolved: List[RevisionEdit] = [sorted_edits[0]]

        for edit in sorted_edits[1:]:
            last = resolved[-1]
            if edit.start_pos < last.end_pos:
                # 重叠：保留置信度更高的
                if edit.confidence > last.confidence:
                    resolved[-1] = edit
            else:
                resolved.append(edit)

        return resolved


# ============================================================================
# ViolationReviser - 违规修订器
# ============================================================================

class ViolationReviser:
    """违规修订器：基于违规记录和建议生成修订"""

    def __init__(
        self,
        style_preserver: Optional[StylePreserver] = None,
        edit_strategy: Optional[MinimalEditStrategy] = None,
    ):
        self._style_preserver = style_preserver or StylePreserver()
        self._edit_strategy = edit_strategy or MinimalEditStrategy()

    def revise_for_violation(
        self,
        content: str,
        violation: Violation,
        suggestion: Optional[ImprovementSuggestion] = None,
    ) -> Optional[RevisionEdit]:
        """为单条违规生成修订"""
        if not suggestion:
            suggestion = ImprovementSuggestion(
                suggestion_id=_generate_id(),
                violation_id=violation.violation_id,
                priority=1,
                description="自动修订",
                original_text=violation.matched_text,
                suggested_text="",
                reasoning=violation.description,
                expected_impact=violation.score,
            )

        edit = self._edit_strategy.generate_minimal_edit(
            content, violation, suggestion
        )
        if edit is None:
            return None

        # 应用风格保持
        edit.revised_text = self._style_preserver.suggest_style_aware_replacement(
            edit.original_text, edit.revised_text, content
        )

        return edit

    def revise_for_violations(
        self,
        content: str,
        violations: List[Violation],
        suggestions: List[ImprovementSuggestion],
    ) -> Tuple[str, List[RevisionEdit]]:
        """为多条违规批量生成修订"""
        suggestion_map: Dict[str, ImprovementSuggestion] = {}
        for sug in suggestions:
            suggestion_map[sug.violation_id] = sug

        edits: List[RevisionEdit] = []
        for violation in violations:
            suggestion = suggestion_map.get(violation.violation_id)
            edit = self.revise_for_violation(content, violation, suggestion)
            if edit is not None:
                edits.append(edit)

        # 解决重叠编辑
        edits = self._edit_strategy.resolve_overlapping_edits(edits)

        # 应用编辑
        revised_content = self._edit_strategy.apply_edits(content, edits)

        return revised_content, edits


# ============================================================================
# RevisionScorer - 修订质量评分器
# ============================================================================

class RevisionScorer:
    """修订质量评分器：评估修订的质量和效果"""

    def __init__(self):
        self._weights = {
            "violation_reduction": 0.35,
            "style_preservation": 0.25,
            "semantic_preservation": 0.20,
            "fluency": 0.10,
            "edit_efficiency": 0.10,
        }

    def score_revision(
        self,
        original: str,
        revised: str,
        violations_before: List[Violation],
        violations_after: List[Violation],
        edits: List[RevisionEdit],
    ) -> Dict[str, float]:
        """评估修订质量"""
        scores: Dict[str, float] = {}

        # 违规减少度
        scores["violation_reduction"] = self._score_violation_reduction(
            violations_before, violations_after
        )

        # 风格保持度
        style_preserver = StylePreserver()
        scores["style_preservation"] = style_preserver.compute_preservation_score(
            original, revised
        )

        # 语义保持度
        scores["semantic_preservation"] = self._score_semantic_preservation(
            original, revised
        )

        # 流畅度
        scores["fluency"] = self._score_fluency(revised)

        # 编辑效率
        scores["edit_efficiency"] = self._score_edit_efficiency(
            original, revised, edits, violations_before
        )

        # 综合分数
        total_weight = sum(self._weights.values())
        overall = sum(
            scores[key] * weight
            for key, weight in self._weights.items()
        ) / total_weight

        scores["overall"] = overall
        return scores

    def _score_violation_reduction(
        self,
        before: List[Violation],
        after: List[Violation],
    ) -> float:
        """评分违规减少程度"""
        if not before:
            return 1.0

        before_severity_sum = sum(v.score for v in before)
        after_severity_sum = sum(v.score for v in after)

        if before_severity_sum == 0:
            return 1.0

        reduction = 1.0 - after_severity_sum / before_severity_sum
        return max(0.0, min(1.0, reduction))

    def _score_semantic_preservation(
        self, original: str, revised: str
    ) -> float:
        """评分语义保持程度"""
        if not original and not revised:
            return 1.0

        # TF向量余弦相似度
        orig_vec = _build_tf_vector(original)
        rev_vec = _build_tf_vector(revised)
        tf_sim = _cosine_similarity(orig_vec, rev_vec)

        # 编辑距离相似度
        edit_sim = _diff_ratio(original, revised)

        # 综合两种度量
        return tf_sim * 0.6 + edit_sim * 0.4

    def _score_fluency(self, text: str) -> float:
        """评分文本流畅度"""
        if not text.strip():
            return 0.0

        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.5

        # 句子长度变化（适度变化表示流畅）
        lengths = [len(s) for s in sentences]
        if len(lengths) < 2:
            return 0.8

        avg_len = sum(lengths) / len(lengths)
        if avg_len == 0:
            return 0.5

        length_variation = (
            sum(abs(l - avg_len) for l in lengths) / len(lengths) / avg_len
        )
        # 适度的长度变化（0.2-0.5）最流畅
        if length_variation < 0.2:
            variation_score = 0.6
        elif length_variation < 0.5:
            variation_score = 1.0
        elif length_variation < 0.8:
            variation_score = 0.7
        else:
            variation_score = 0.4

        # 段落连贯性
        paragraphs = text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        coherence_score = min(1.0, len(paragraphs) / 3.0) if paragraphs else 0.5

        return variation_score * 0.6 + coherence_score * 0.4

    def _score_edit_efficiency(
        self,
        original: str,
        revised: str,
        edits: List[RevisionEdit],
        violations: List[Violation],
    ) -> float:
        """评分编辑效率"""
        if not violations:
            return 1.0

        if not edits:
            return 0.0

        # 每个违规对应的编辑数
        edits_per_violation = len(edits) / len(violations)
        efficiency = max(0.0, 1.0 - (edits_per_violation - 1.0) * 0.3)

        # 编辑覆盖的违规比例
        addressed = set(e.violation_id for e in edits)
        coverage = len(addressed) / len(violations)

        return efficiency * 0.4 + coverage * 0.6


# ============================================================================
# IterativeRefiner - 迭代精炼器
# ============================================================================

class IterativeRefiner:
    """
    迭代精炼器：通过多轮批评-修订循环逐步改进内容。

    流程:
    1. 对内容进行批评
    2. 根据批评结果生成修订
    3. 评估修订质量
    4. 判断是否需要继续迭代
    5. 重复直到满足终止条件
    """

    def __init__(
        self,
        max_iterations: int = 5,
        target_score: float = 0.9,
        min_improvement: float = 0.02,
        style_weight: float = 0.3,
    ):
        self._max_iterations = max_iterations
        self._target_score = target_score
        self._min_improvement = min_improvement
        self._style_weight = style_weight
        self._reviser = ViolationReviser()
        self._scorer = RevisionScorer()
        self._style_preserver = StylePreserver()

    def refine(
        self,
        content: str,
        critic: Any,
    ) -> RevisionReport:
        """
        迭代精炼内容。

        Args:
            content: 原始内容
            critic: ConstitutionCritic实例，用于批评

        Returns:
            RevisionReport: 修订报告
        """
        history: List[RevisionHistoryEntry] = []
        current_content = content
        prev_score = 0.0

        for iteration in range(self._max_iterations):
            # 步骤1: 批评
            report = critic.criticize(current_content)

            # 步骤2: 检查终止条件
            if report.overall_score >= self._target_score:
                break

            if iteration > 0:
                improvement = report.overall_score - prev_score
                if improvement < self._min_improvement:
                    break

            prev_score = report.overall_score

            # 步骤3: 生成修订
            revised_content, edits = self._reviser.revise_for_violations(
                current_content,
                report.violations,
                report.suggestions,
            )

            if not edits:
                break

            # 步骤4: 评估修订质量
            post_report = critic.criticize(revised_content)
            quality_scores = self._scorer.score_revision(
                current_content,
                revised_content,
                report.violations,
                post_report.violations,
                edits,
            )

            # 步骤5: 决定是否接受修订
            if quality_scores["overall"] < 0.3:
                # 修订质量太差，拒绝
                break

            # 记录历史
            entry = RevisionHistoryEntry(
                entry_id=_generate_id(),
                iteration=iteration,
                original_content=current_content,
                revised_content=revised_content,
                edits=edits,
                violations_addressed=[v.violation_id for v in report.violations],
                quality_score=quality_scores["overall"],
                style_preservation=quality_scores["style_preservation"],
                timestamp=time.time(),
            )
            history.append(entry)

            # 更新当前内容
            current_content = revised_content

        # 生成最终报告
        final_report = critic.criticize(current_content)
        initial_report = critic.criticize(content)

        return RevisionReport(
            report_id=_generate_id(),
            original_content=content,
            final_content=current_content,
            total_iterations=len(history),
            total_edits=sum(len(e.edits) for e in history),
            violations_before=len(initial_report.violations),
            violations_after=len(final_report.violations),
            quality_score=final_report.overall_score,
            style_preservation=self._style_preserver.compute_preservation_score(
                content, current_content
            ),
            edit_distance=_levenshtein_distance(content, current_content),
            similarity_ratio=_diff_ratio(content, current_content),
            history=history,
            timestamp=time.time(),
            metadata={
                "target_score": self._target_score,
                "max_iterations": self._max_iterations,
                "initial_score": initial_report.overall_score,
                "final_score": final_report.overall_score,
            },
        )


# ============================================================================
# RevisionHistory - 修订历史管理
# ============================================================================

class RevisionHistory:
    """修订历史管理器"""

    def __init__(self, max_entries: int = 100):
        self._entries: List[RevisionHistoryEntry] = []
        self._max_entries = max_entries
        self._by_content_hash: Dict[str, List[str]] = defaultdict(list)

    def add_entry(self, entry: RevisionHistoryEntry) -> None:
        """添加历史条目"""
        self._entries.append(entry)
        content_hash = hashlib.sha256(
            entry.original_content.encode()
        ).hexdigest()[:16]
        self._by_content_hash[content_hash].append(entry.entry_id)

        # 限制条目数量
        if len(self._entries) > self._max_entries:
            removed = self._entries.pop(0)
            removed_hash = hashlib.sha256(
                removed.original_content.encode()
            ).hexdigest()[:16]
            if removed.entry_id in self._by_content_hash[removed_hash]:
                self._by_content_hash[removed_hash].remove(removed.entry_id)

    def get_entries(
        self, limit: int = 10, offset: int = 0
    ) -> List[RevisionHistoryEntry]:
        """获取历史条目"""
        return self._entries[offset:offset + limit]

    def get_by_content(self, content: str) -> List[RevisionHistoryEntry]:
        """根据原始内容查找历史"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        entry_ids = self._by_content_hash.get(content_hash, [])
        return [
            e for e in self._entries if e.entry_id in entry_ids
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._entries:
            return {
                "total_entries": 0,
                "avg_quality_score": 0.0,
                "avg_style_preservation": 0.0,
                "total_edits": 0,
            }

        return {
            "total_entries": len(self._entries),
            "avg_quality_score": sum(
                e.quality_score for e in self._entries
            ) / len(self._entries),
            "avg_style_preservation": sum(
                e.style_preservation for e in self._entries
            ) / len(self._entries),
            "total_edits": sum(len(e.edits) for e in self._entries),
            "avg_edits_per_revision": sum(
                len(e.edits) for e in self._entries
            ) / len(self._entries),
        }

    def clear(self) -> None:
        """清空历史"""
        self._entries.clear()
        self._by_content_hash.clear()


# ============================================================================
# ContentReviser - 内容修订器（主入口）
# ============================================================================

class ContentReviser:
    """
    内容修订器：整合所有组件，提供统一的内容修订接口。

    使用方法:
        reviser = ContentReviser()
        report = reviser.revise(content, critic)
    """

    def __init__(
        self,
        max_iterations: int = 5,
        target_score: float = 0.9,
        style_preservation_weight: float = 0.7,
        max_edit_ratio: float = 0.3,
    ):
        self._style_preserver = StylePreserver(style_preservation_weight)
        self._edit_strategy = MinimalEditStrategy(max_edit_ratio=max_edit_ratio)
        self._violation_reviser = ViolationReviser(
            self._style_preserver, self._edit_strategy
        )
        self._revision_scorer = RevisionScorer()
        self._iterative_refiner = IterativeRefiner(
            max_iterations=max_iterations,
            target_score=target_score,
        )
        self._history = RevisionHistory()

    @property
    def style_preserver(self) -> StylePreserver:
        return self._style_preserver

    @property
    def edit_strategy(self) -> MinimalEditStrategy:
        return self._edit_strategy

    @property
    def violation_reviser(self) -> ViolationReviser:
        return self._violation_reviser

    @property
    def revision_scorer(self) -> RevisionScorer:
        return self._revision_scorer

    @property
    def iterative_refiner(self) -> IterativeRefiner:
        return self._iterative_refiner

    @property
    def history(self) -> RevisionHistory:
        return self._history

    def revise(
        self,
        content: str,
        critic: Any,
        strategy: RevisionStrategy = RevisionStrategy.MODERATE,
    ) -> RevisionReport:
        """
        修订内容。

        Args:
            content: 原始内容
            critic: ConstitutionCritic实例
            strategy: 修订策略

        Returns:
            RevisionReport: 修订报告
        """
        if strategy == RevisionStrategy.FULL_REWRITE:
            return self._full_rewrite(content, critic)

        # 执行批评
        report = critic.criticize(content)

        if not report.violations:
            return RevisionReport(
                report_id=_generate_id(),
                original_content=content,
                final_content=content,
                total_iterations=0,
                total_edits=0,
                violations_before=0,
                violations_after=0,
                quality_score=1.0,
                style_preservation=1.0,
                edit_distance=0,
                similarity_ratio=1.0,
                timestamp=time.time(),
            )

        if strategy == RevisionStrategy.MINIMAL:
            return self._minimal_revision(content, critic, report)
        elif strategy == RevisionStrategy.AGGRESSIVE:
            return self._aggressive_revision(content, critic, report)
        else:
            return self._iterative_refiner.refine(content, critic)

    def _minimal_revision(
        self,
        content: str,
        critic: Any,
        report: Any,
    ) -> RevisionReport:
        """最小化修订"""
        # 只处理高严重度的违规
        high_severity = [
            v for v in report.violations
            if v.severity in (Severity.HIGH, Severity.CRITICAL)
        ]
        high_suggestions = [
            s for s in report.suggestions
            if s.violation_id in {v.violation_id for v in high_severity}
        ]

        revised, edits = self._violation_reviser.revise_for_violations(
            content, high_severity, high_suggestions
        )

        post_report = critic.criticize(revised)
        scores = self._revision_scorer.score_revision(
            content, revised, report.violations, post_report.violations, edits
        )

        return RevisionReport(
            report_id=_generate_id(),
            original_content=content,
            final_content=revised,
            total_iterations=1,
            total_edits=len(edits),
            violations_before=len(report.violations),
            violations_after=len(post_report.violations),
            quality_score=scores["overall"],
            style_preservation=scores["style_preservation"],
            edit_distance=_levenshtein_distance(content, revised),
            similarity_ratio=_diff_ratio(content, revised),
            timestamp=time.time(),
        )

    def _aggressive_revision(
        self,
        content: str,
        critic: Any,
        report: Any,
    ) -> RevisionReport:
        """激进修订"""
        revised, edits = self._violation_reviser.revise_for_violations(
            content, report.violations, report.suggestions
        )

        post_report = critic.criticize(revised)
        scores = self._revision_scorer.score_revision(
            content, revised, report.violations, post_report.violations, edits
        )

        # 如果还有违规，再迭代一轮
        if post_report.violations and scores["overall"] > 0.4:
            revised2, edits2 = self._violation_reviser.revise_for_violations(
                revised, post_report.violations, post_report.suggestions
            )
            final_report = critic.criticize(revised2)
            final_scores = self._revision_scorer.score_revision(
                revised, revised2,
                post_report.violations, final_report.violations, edits2
            )
            if final_scores["overall"] > scores["overall"]:
                revised = revised2
                post_report = final_report
                scores = final_scores
                edits.extend(edits2)

        return RevisionReport(
            report_id=_generate_id(),
            original_content=content,
            final_content=revised,
            total_iterations=2,
            total_edits=len(edits),
            violations_before=len(report.violations),
            violations_after=len(post_report.violations),
            quality_score=scores["overall"],
            style_preservation=scores["style_preservation"],
            edit_distance=_levenshtein_distance(content, revised),
            similarity_ratio=_diff_ratio(content, revised),
            timestamp=time.time(),
        )

    def _full_rewrite(
        self, content: str, critic: Any
    ) -> RevisionReport:
        """完全重写（保留核心语义）"""
        # 提取核心语义关键词
        tokens = _tokenize(content)
        if not tokens:
            return RevisionReport(
                report_id=_generate_id(),
                original_content=content,
                final_content=content,
                total_iterations=0,
                total_edits=0,
                violations_before=0,
                violations_after=0,
                quality_score=1.0,
                style_preservation=1.0,
                edit_distance=0,
                similarity_ratio=1.0,
                timestamp=time.time(),
            )

        # 获取原始批评
        report = critic.criticize(content)
        violated_keywords = set()
        for v in report.violations:
            if v.matched_text:
                for word in _tokenize(v.matched_text):
                    violated_keywords.add(word)

        # 保留非违规关键词
        safe_tokens = [t for t in tokens if t not in violated_keywords]

        # 生成重写内容
        sentences = re.split(r'[。！？.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        rewritten_parts: List[str] = []
        for sentence in sentences:
            sent_tokens = _tokenize(sentence)
            safe_sent_tokens = [
                t for t in sent_tokens if t not in violated_keywords
            ]
            if safe_sent_tokens:
                # 保持句子结构但移除违规内容
                rewritten = " ".join(safe_sent_tokens)
                rewritten_parts.append(rewritten)

        rewritten = "。".join(rewritten_parts)
        if not rewritten:
            rewritten = "内容已根据合规要求进行重写。"

        post_report = critic.criticize(rewritten)
        scores = self._revision_scorer.score_revision(
            content, rewritten, report.violations, post_report.violations, []
        )

        return RevisionReport(
            report_id=_generate_id(),
            original_content=content,
            final_content=rewritten,
            total_iterations=1,
            total_edits=0,
            violations_before=len(report.violations),
            violations_after=len(post_report.violations),
            quality_score=scores["overall"],
            style_preservation=scores["style_preservation"],
            edit_distance=_levenshtein_distance(content, rewritten),
            similarity_ratio=_diff_ratio(content, rewritten),
            timestamp=time.time(),
        )

    def quick_revise(
        self, content: str, critic: Any
    ) -> Dict[str, Any]:
        """快速修订（返回简化结果）"""
        report = self.revise(content, critic)
        return {
            "original_length": len(report.original_content),
            "revised_length": len(report.final_content),
            "iterations": report.total_iterations,
            "edits": report.total_edits,
            "quality_score": report.quality_score,
            "style_preservation": report.style_preservation,
            "violations_fixed": report.violations_before - report.violations_after,
        }

    def batch_revise(
        self, contents: List[str], critic: Any
    ) -> List[RevisionReport]:
        """批量修订"""
        return [self.revise(content, critic) for content in contents]
