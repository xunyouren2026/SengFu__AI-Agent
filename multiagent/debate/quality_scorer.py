"""
辩论质量评分模块
基于论点新颖性、逻辑性、证据充分性打分
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import math
import statistics
from collections import defaultdict

from .protocol import Argument, Rebuttal, Evidence, DebateState


class QualityDimension(Enum):
    """质量维度枚举"""
    NOVELTY = "novelty"                    # 新颖性
    LOGICAL_CONSISTENCY = "logical_consistency"  # 逻辑一致性
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"  # 证据充分性
    CLARITY = "clarity"                    # 清晰度
    RELEVANCE = "relevance"                # 相关性
    RHETORICAL_EFFECTIVENESS = "rhetorical_effectiveness"  # 修辞效果
    STRUCTURAL_COMPLETENESS = "structural_completeness"  # 结构完整性


@dataclass
class DimensionScore:
    """维度评分"""
    dimension: QualityDimension
    score: float                           # 0-1分数
    weight: float                          # 权重
    details: Dict[str, Any] = field(default_factory=dict)
    
    def weighted_score(self) -> float:
        """获取加权分数"""
        return self.score * self.weight


@dataclass
class QualityAssessment:
    """质量评估结果"""
    target_id: str
    target_type: str                       # "argument" 或 "rebuttal"
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    overall_score: float = 0.0
    grade: str = "C"                       # A/B/C/D/F
    percentile: float = 0.0                # 百分位
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def calculate_overall(self) -> float:
        """计算综合得分"""
        if not self.dimension_scores:
            return 0.0
        
        total_weight = sum(d.weight for d in self.dimension_scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(d.weighted_score() for d in self.dimension_scores)
        self.overall_score = weighted_sum / total_weight
        return self.overall_score
    
    def determine_grade(self) -> str:
        """确定等级"""
        score = self.overall_score
        if score >= 0.9:
            self.grade = "A"
        elif score >= 0.8:
            self.grade = "B"
        elif score >= 0.7:
            self.grade = "C"
        elif score >= 0.6:
            self.grade = "D"
        else:
            self.grade = "F"
        return self.grade


class NoveltyScorer:
    """
    新颖性评分器
    评估论点的原创性和新颖程度
    """
    
    def __init__(self) -> None:
        self.seen_arguments: Set[str] = set()
        self.concept_frequency: Dict[str, int] = defaultdict(int)
    
    def score(self, argument: Argument, all_arguments: List[Argument]) -> DimensionScore:
        """
        评估新颖性
        
        基于：
        1. 与已有论点的相似度
        2. 概念的独特性
        3. 论证角度的新颖性
        """
        scores = []
        details = {}
        
        # 1. 文本相似度新颖性
        similarity_scores = []
        for other in all_arguments:
            if other.argument_id != argument.argument_id:
                sim = self._text_similarity(argument.content, other.content)
                similarity_scores.append(sim)
        
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            novelty_from_similarity = 1.0 - avg_similarity
            scores.append(novelty_from_similarity)
            details["text_uniqueness"] = novelty_from_similarity
        else:
            scores.append(1.0)  # 第一个论点默认为新颖
            details["text_uniqueness"] = 1.0
        
        # 2. 概念新颖性
        concepts = self._extract_concepts(argument.content)
        concept_scores = []
        for concept in concepts:
            freq = self.concept_frequency.get(concept, 0)
            # 出现次数越少，越新颖
            novelty = 1.0 / (1 + math.log(1 + freq))
            concept_scores.append(novelty)
            self.concept_frequency[concept] += 1
        
        if concept_scores:
            avg_concept_novelty = sum(concept_scores) / len(concept_scores)
            scores.append(avg_concept_novelty)
            details["concept_novelty"] = avg_concept_novelty
        
        # 3. 论证类型新颖性
        type_novelty = self._assess_argument_type_novelty(argument, all_arguments)
        scores.append(type_novelty)
        details["approach_novelty"] = type_novelty
        
        final_score = sum(scores) / len(scores) if scores else 0.5
        
        return DimensionScore(
            dimension=QualityDimension.NOVELTY,
            score=final_score,
            weight=0.15,
            details=details
        )
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """提取概念（简化版）"""
        # 提取2-4个字的词组作为概念
        words = text.split()
        concepts = []
        for word in words:
            if 2 <= len(word) <= 6:
                concepts.append(word.lower())
        return concepts[:10]  # 限制概念数量
    
    def _assess_argument_type_novelty(
        self,
        argument: Argument,
        all_arguments: List[Argument]
    ) -> float:
        """评估论证类型的新颖性"""
        same_type_count = sum(
            1 for a in all_arguments
            if a.argument_type == argument.argument_type
            and a.argument_id != argument.argument_id
        )
        
        total = len(all_arguments) - 1
        if total == 0:
            return 1.0
        
        # 同类型越少，越新颖
        return 1.0 - (same_type_count / total)


class LogicalConsistencyScorer:
    """
    逻辑一致性评分器
    评估论点的逻辑结构和推理有效性
    """
    
    LOGICAL_MARKERS = {
        "premise": ["因为", "由于", "鉴于", "基于", "如果", "假设"],
        "conclusion": ["所以", "因此", "由此", "从而", "结论", "故"],
        "conditional": ["如果", "那么", "则", "就"],
        "causal": ["导致", "引起", "造成", "产生"],
    }
    
    def score(self, argument: Argument) -> DimensionScore:
        """评估逻辑一致性"""
        scores = []
        details = {}
        
        content = argument.content
        
        # 1. 结构完整性检查
        has_premise = any(m in content for m in self.LOGICAL_MARKERS["premise"])
        has_conclusion = any(m in content for m in self.LOGICAL_MARKERS["conclusion"])
        
        structure_score = 0.0
        if has_premise:
            structure_score += 0.4
        if has_conclusion:
            structure_score += 0.4
        if has_premise and has_conclusion:
            structure_score += 0.2
        
        scores.append(structure_score)
        details["structural_integrity"] = structure_score
        
        # 2. 逻辑连接词丰富度
        total_markers = sum(
            len([m for m in markers if m in content])
            for markers in self.LOGICAL_MARKERS.values()
        )
        marker_score = min(1.0, total_markers / 5)  # 最多5个连接词得满分
        scores.append(marker_score)
        details["logical_markers"] = marker_score
        
        # 3. 逻辑一致性检查（简化）
        consistency_score = self._check_internal_consistency(content)
        scores.append(consistency_score)
        details["internal_consistency"] = consistency_score
        
        final_score = sum(scores) / len(scores)
        
        return DimensionScore(
            dimension=QualityDimension.LOGICAL_CONSISTENCY,
            score=final_score,
            weight=0.25,
            details=details
        )
    
    def _check_internal_consistency(self, content: str) -> float:
        """检查内部一致性"""
        # 简化检查：查找矛盾表述
        contradictions = [
            ("所有", "有些不"),
            ("总是", "有时不"),
            ("必然", "可能不"),
        ]
        
        penalty = 0
        for pos, neg in contradictions:
            if pos in content and neg in content:
                penalty += 0.2
        
        return max(0.0, 1.0 - penalty)


class EvidenceSufficiencyScorer:
    """
    证据充分性评分器
    评估论点的证据支持程度
    """
    
    def score(self, argument: Argument) -> DimensionScore:
        """评估证据充分性"""
        scores = []
        details = {}
        
        evidence_list = argument.evidence_list
        
        # 1. 证据数量
        evidence_count = len(evidence_list)
        count_score = min(1.0, evidence_count / 3)  # 3个证据得满分
        scores.append(count_score)
        details["evidence_quantity"] = count_score
        
        # 2. 证据质量
        if evidence_list:
            quality_scores = [e.quality_score() for e in evidence_list]
            avg_quality = sum(quality_scores) / len(quality_scores)
            scores.append(avg_quality)
            details["average_evidence_quality"] = avg_quality
        else:
            scores.append(0.0)
            details["average_evidence_quality"] = 0.0
        
        # 3. 证据多样性
        if evidence_list:
            sources = set(e.source for e in evidence_list)
            diversity_score = min(1.0, len(sources) / 2)  # 2个不同来源得满分
            scores.append(diversity_score)
            details["source_diversity"] = diversity_score
        else:
            scores.append(0.0)
            details["source_diversity"] = 0.0
        
        # 4. 引用明确性
        citation_score = self._assess_citation_clarity(argument.content)
        scores.append(citation_score)
        details["citation_clarity"] = citation_score
        
        final_score = sum(scores) / len(scores)
        
        return DimensionScore(
            dimension=QualityDimension.EVIDENCE_SUFFICIENCY,
            score=final_score,
            weight=0.25,
            details=details
        )
    
    def _assess_citation_clarity(self, content: str) -> float:
        """评估引用明确性"""
        citation_markers = [
            "根据", "引用", "数据显示", "研究表明",
            "如图", "见", "来源", "参考"
        ]
        
        matches = sum(1 for m in citation_markers if m in content)
        return min(1.0, matches / 2)


class ClarityScorer:
    """
    清晰度评分器
    评估论点的表达清晰度
    """
    
    def score(self, argument: Argument) -> DimensionScore:
        """评估清晰度"""
        scores = []
        details = {}
        
        content = argument.content
        
        # 1. 长度适中性
        length = len(content)
        if 100 <= length <= 500:
            length_score = 1.0
        elif length < 100:
            length_score = 0.5 + length / 200
        else:
            length_score = max(0.5, 1.0 - (length - 500) / 1000)
        
        scores.append(length_score)
        details["length_appropriateness"] = length_score
        
        # 2. 结构清晰度
        structure_markers = ["首先", "其次", "最后", "第一", "第二", "总结"]
        structure_count = sum(1 for m in structure_markers if m in content)
        structure_score = min(1.0, structure_count * 0.25 + 0.5)
        scores.append(structure_score)
        details["structural_clarity"] = structure_score
        
        # 3. 语言简洁性
        words = content.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            # 平均词长适中为佳
            simplicity_score = 1.0 - abs(avg_word_length - 4) / 4
            simplicity_score = max(0.3, min(1.0, simplicity_score))
        else:
            simplicity_score = 0.5
        scores.append(simplicity_score)
        details["language_simplicity"] = simplicity_score
        
        # 4. 避免模糊表述
        vague_terms = ["可能", "大概", "也许", "差不多", "一些", "某些"]
        vague_count = sum(content.count(term) for term in vague_terms)
        precision_score = max(0.3, 1.0 - vague_count * 0.1)
        scores.append(precision_score)
        details["precision"] = precision_score
        
        final_score = sum(scores) / len(scores)
        
        return DimensionScore(
            dimension=QualityDimension.CLARITY,
            score=final_score,
            weight=0.15,
            details=details
        )


class RelevanceScorer:
    """
    相关性评分器
    评估论点与主题的相关程度
    """
    
    def score(self, argument: Argument, topic: str = "") -> DimensionScore:
        """评估相关性"""
        scores = []
        details = {}
        
        content = argument.content.lower()
        topic_lower = topic.lower() if topic else ""
        
        # 1. 主题词匹配
        if topic_lower:
            topic_words = set(topic_lower.split())
            content_words = set(content.split())
            
            if topic_words:
                overlap = len(topic_words & content_words)
                topic_score = min(1.0, overlap / len(topic_words) * 1.5)
                scores.append(topic_score)
                details["topic_overlap"] = topic_score
        
        # 2. 论点类型相关性
        type_relevance = {
            "FACTUAL": 0.9,
            "NORMATIVE": 0.8,
            "CAUSAL": 0.85,
            "ANALOGICAL": 0.7,
            "AUTHORITY": 0.75,
            "PRAGMATIC": 0.8,
        }
        type_score = type_relevance.get(argument.argument_type.name, 0.7)
        scores.append(type_score)
        details["argument_type_relevance"] = type_score
        
        # 3. 目标论点相关性（如果是反驳）
        if argument.target_argument_id:
            scores.append(0.9)  # 有明确目标的得分较高
            details["target_specificity"] = 0.9
        else:
            scores.append(0.7)
            details["target_specificity"] = 0.7
        
        final_score = sum(scores) / len(scores) if scores else 0.7
        
        return DimensionScore(
            dimension=QualityDimension.RELEVANCE,
            score=final_score,
            weight=0.10,
            details=details
        )


class QualityScorer:
    """
    辩论质量评分器
    综合多个维度评估辩论质量
    """
    
    def __init__(self) -> None:
        self.novelty_scorer = NoveltyScorer()
        self.logical_scorer = LogicalConsistencyScorer()
        self.evidence_scorer = EvidenceSufficiencyScorer()
        self.clarity_scorer = ClarityScorer()
        self.relevance_scorer = RelevanceScorer()
        
        self.assessment_history: List[QualityAssessment] = []
        self.score_distribution: Dict[str, List[float]] = defaultdict(list)
    
    def assess_argument(
        self,
        argument: Argument,
        all_arguments: List[Argument],
        topic: str = ""
    ) -> QualityAssessment:
        """
        评估单个论点
        
        Args:
            argument: 要评估的论点
            all_arguments: 所有论点列表（用于新颖性计算）
            topic: 辩论主题
            
        Returns:
            质量评估结果
        """
        assessment = QualityAssessment(
            target_id=argument.argument_id,
            target_type="argument"
        )
        
        # 计算各维度分数
        dimension_scores = [
            self.novelty_scorer.score(argument, all_arguments),
            self.logical_scorer.score(argument),
            self.evidence_scorer.score(argument),
            self.clarity_scorer.score(argument),
            self.relevance_scorer.score(argument, topic),
        ]
        
        assessment.dimension_scores = dimension_scores
        assessment.calculate_overall()
        assessment.determine_grade()
        
        # 识别优势和劣势
        self._identify_strengths_weaknesses(assessment)
        
        # 生成改进建议
        assessment.improvement_suggestions = self._generate_suggestions(assessment)
        
        # 记录历史
        self.assessment_history.append(assessment)
        self.score_distribution[argument.argument_id].append(assessment.overall_score)
        
        return assessment
    
    def assess_rebuttal(
        self,
        rebuttal: Rebuttal,
        target_argument: Argument
    ) -> QualityAssessment:
        """评估反驳质量"""
        assessment = QualityAssessment(
            target_id=rebuttal.rebuttal_id,
            target_type="rebuttal"
        )
        
        # 反驳的特殊评分逻辑
        scores = []
        details = {}
        
        # 1. 有效性
        effectiveness = rebuttal.effectiveness_score
        scores.append(DimensionScore(
            dimension=QualityDimension.LOGICAL_CONSISTENCY,
            score=effectiveness,
            weight=0.3,
            details={"effectiveness": effectiveness}
        ))
        
        # 2. 针对性
        relevance = 0.9 if rebuttal.target_argument_id else 0.5
        scores.append(DimensionScore(
            dimension=QualityDimension.RELEVANCE,
            score=relevance,
            weight=0.3,
            details={"target_specificity": relevance}
        ))
        
        # 3. 内容清晰度
        content_length = len(rebuttal.content)
        clarity = 1.0 if 50 <= content_length <= 300 else 0.7
        scores.append(DimensionScore(
            dimension=QualityDimension.CLARITY,
            score=clarity,
            weight=0.2,
            details={"length_appropriateness": clarity}
        ))
        
        # 4. 谬误检测
        fallacy_count = len(rebuttal.fallacies_detected)
        logical_score = max(0.3, 1.0 - fallacy_count * 0.2)
        scores.append(DimensionScore(
            dimension=QualityDimension.LOGICAL_CONSISTENCY,
            score=logical_score,
            weight=0.2,
            details={"fallacy_free": logical_score}
        ))
        
        assessment.dimension_scores = scores
        assessment.calculate_overall()
        assessment.determine_grade()
        
        self.assessment_history.append(assessment)
        
        return assessment
    
    def _identify_strengths_weaknesses(self, assessment: QualityAssessment) -> None:
        """识别优势和劣势"""
        for dim_score in assessment.dimension_scores:
            if dim_score.score >= 0.8:
                assessment.strengths.append(
                    f"{dim_score.dimension.value}: 表现优秀 ({dim_score.score:.2f})"
                )
            elif dim_score.score < 0.5:
                assessment.weaknesses.append(
                    f"{dim_score.dimension.value}: 需要改进 ({dim_score.score:.2f})"
                )
    
    def _generate_suggestions(self, assessment: QualityAssessment) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        for dim_score in assessment.dimension_scores:
            if dim_score.score < 0.6:
                if dim_score.dimension == QualityDimension.NOVELTY:
                    suggestions.append("尝试从新的角度论证，避免重复已有观点")
                elif dim_score.dimension == QualityDimension.LOGICAL_CONSISTENCY:
                    suggestions.append("加强逻辑结构，明确前提和结论的关系")
                elif dim_score.dimension == QualityDimension.EVIDENCE_SUFFICIENCY:
                    suggestions.append("提供更多高质量的证据支持您的主张")
                elif dim_score.dimension == QualityDimension.CLARITY:
                    suggestions.append("简化表达，使用更清晰的结构和语言")
                elif dim_score.dimension == QualityDimension.RELEVANCE:
                    suggestions.append("确保论点紧密围绕核心议题展开")
        
        return suggestions
    
    def assess_debate_overall(self, debate_state: DebateState) -> Dict[str, Any]:
        """评估整场辩论的整体质量"""
        arguments = list(debate_state.arguments.values())
        rebuttals = list(debate_state.rebuttals.values())
        
        if not arguments:
            return {"error": "没有论点可供评估"}
        
        # 评估所有论点
        argument_assessments = [
            self.assess_argument(arg, arguments, debate_state.topic)
            for arg in arguments
        ]
        
        # 计算统计数据
        overall_scores = [a.overall_score for a in argument_assessments]
        
        # 按维度统计
        dimension_averages = defaultdict(list)
        for assessment in argument_assessments:
            for dim_score in assessment.dimension_scores:
                dimension_averages[dim_score.dimension.value].append(dim_score.score)
        
        return {
            "debate_id": debate_state.debate_id,
            "topic": debate_state.topic,
            "total_arguments": len(arguments),
            "total_rebuttals": len(rebuttals),
            "average_quality_score": statistics.mean(overall_scores),
            "median_quality_score": statistics.median(overall_scores),
            "score_std_deviation": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            "dimension_averages": {
                dim: statistics.mean(scores)
                for dim, scores in dimension_averages.items()
            },
            "grade_distribution": self._calculate_grade_distribution(argument_assessments),
            "top_arguments": [
                {
                    "argument_id": a.target_id,
                    "score": a.overall_score,
                    "grade": a.grade,
                }
                for a in sorted(argument_assessments, key=lambda x: x.overall_score, reverse=True)[:3]
            ],
        }
    
    def _calculate_grade_distribution(
        self,
        assessments: List[QualityAssessment]
    ) -> Dict[str, int]:
        """计算等级分布"""
        distribution = defaultdict(int)
        for a in assessments:
            distribution[a.grade] += 1
        return dict(distribution)
    
    def get_historical_stats(self) -> Dict[str, Any]:
        """获取历史统计"""
        if not self.assessment_history:
            return {"message": "暂无评估记录"}
        
        all_scores = [a.overall_score for a in self.assessment_history]
        
        return {
            "total_assessments": len(self.assessment_history),
            "average_score": statistics.mean(all_scores),
            "median_score": statistics.median(all_scores),
            "min_score": min(all_scores),
            "max_score": max(all_scores),
            "score_std_deviation": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
        }


__all__ = [
    "QualityDimension",
    "DimensionScore",
    "QualityAssessment",
    "NoveltyScorer",
    "LogicalConsistencyScorer",
    "EvidenceSufficiencyScorer",
    "ClarityScorer",
    "RelevanceScorer",
    "QualityScorer",
]
