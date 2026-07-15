"""
辩论知识提取模块
从辩论过程中归纳胜方论点形成新知识
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from uuid import uuid4
from collections import defaultdict
import statistics
import re

from .protocol import (
    Argument, Rebuttal, Verdict, DebateState,
    Stance, ArgumentType
)


class KnowledgeType(Enum):
    """知识类型枚举"""
    FACTUAL = "factual"           # 事实性知识
    PROCEDURAL = "procedural"     # 过程性知识
    STRATEGIC = "strategic"       # 策略性知识
    EVIDENTIAL = "evidential"     # 证据性知识
    RHETORICAL = "rhetorical"     # 修辞性知识
    META = "meta"                 # 元知识


class ExtractionMethod(Enum):
    """提取方法枚举"""
    WINNER_ANALYSIS = "winner_analysis"       # 胜方分析
    PATTERN_MINING = "pattern_mining"         # 模式挖掘
    CONTRASTIVE = "contrastive"               # 对比分析
    EVIDENCE_SYNTHESIS = "evidence_synthesis" # 证据综合
    FAILURE_ANALYSIS = "failure_analysis"     # 失败分析


@dataclass
class KnowledgePattern:
    """知识模式"""
    pattern_id: str = field(default_factory=lambda: str(uuid4())[:8])
    pattern_type: KnowledgeType = KnowledgeType.FACTUAL
    description: str = ""
    template: str = ""                    # 模式模板
    conditions: List[str] = field(default_factory=list)  # 适用条件
    examples: List[str] = field(default_factory=list)    # 示例
    confidence: float = 0.0
    support_count: int = 0                # 支持次数
    success_rate: float = 0.0
    source_debates: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "template": self.template,
            "conditions": self.conditions,
            "examples": self.examples,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "success_rate": self.success_rate,
            "source_debates": list(self.source_debates),
        }


@dataclass
class ExtractedKnowledge:
    """提取的知识"""
    knowledge_id: str = field(default_factory=lambda: str(uuid4())[:8])
    knowledge_type: KnowledgeType = KnowledgeType.FACTUAL
    content: str = ""
    summary: str = ""
    source_arguments: List[str] = field(default_factory=list)
    source_debate_id: str = ""
    extraction_method: ExtractionMethod = ExtractionMethod.WINNER_ANALYSIS
    confidence: float = 0.0
    applicability: List[str] = field(default_factory=list)  # 适用场景
    counter_examples: List[str] = field(default_factory=list)  # 反例
    related_knowledge: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "knowledge_id": self.knowledge_id,
            "knowledge_type": self.knowledge_type.value,
            "content": self.content,
            "summary": self.summary,
            "source_arguments": self.source_arguments,
            "source_debate_id": self.source_debate_id,
            "extraction_method": self.extraction_method.value,
            "confidence": self.confidence,
            "applicability": self.applicability,
            "counter_examples": self.counter_examples,
            "related_knowledge": self.related_knowledge,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class LearningInsight:
    """学习洞察"""
    insight_id: str = field(default_factory=lambda: str(uuid4())[:8])
    category: str = ""
    description: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    importance: float = 0.0
    actionability: float = 0.0  # 可操作性
    recommendations: List[str] = field(default_factory=list)


class WinnerAnalyzer:
    """
    胜方分析器
    分析获胜论点的共同特征
    """
    
    def __init__(self) -> None:
        self.winning_patterns: List[Dict[str, Any]] = []
        self.losing_patterns: List[Dict[str, Any]] = []
    
    def analyze(
        self,
        arguments: List[Argument],
        verdict: Verdict
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        分析胜方和败方论点
        
        Returns:
            (胜方特征列表, 败方特征列表)
        """
        winning_stance = verdict.winning_stance
        
        winning_args = [a for a in arguments if a.stance == winning_stance]
        losing_args = [a for a in arguments if a.stance != winning_stance and 
                       a.stance != Stance.NEUTRAL]
        
        # 分析胜方特征
        winning_features = self._extract_features(winning_args)
        for feature in winning_features:
            feature["outcome"] = "win"
            self.winning_patterns.append(feature)
        
        # 分析败方特征
        losing_features = self._extract_features(losing_args)
        for feature in losing_features:
            feature["outcome"] = "lose"
            self.losing_patterns.append(feature)
        
        return winning_features, losing_features
    
    def _extract_features(
        self,
        arguments: List[Argument]
    ) -> List[Dict[str, Any]]:
        """提取论点特征"""
        if not arguments:
            return []
        
        features = []
        
        # 1. 论点类型分布
        type_dist = defaultdict(int)
        for arg in arguments:
            type_dist[arg.argument_type.name] += 1
        
        features.append({
            "feature_type": "argument_type_distribution",
            "data": dict(type_dist),
            "sample_size": len(arguments),
        })
        
        # 2. 平均证据数量
        avg_evidence = sum(len(a.evidence_list) for a in arguments) / len(arguments)
        features.append({
            "feature_type": "average_evidence_count",
            "data": {"value": avg_evidence},
            "sample_size": len(arguments),
        })
        
        # 3. 平均置信度
        avg_confidence = sum(a.confidence for a in arguments) / len(arguments)
        features.append({
            "feature_type": "average_confidence",
            "data": {"value": avg_confidence},
            "sample_size": len(arguments),
        })
        
        # 4. 证据质量
        evidence_qualities = []
        for arg in arguments:
            if arg.evidence_list:
                qualities = [e.quality_score() for e in arg.evidence_list]
                evidence_qualities.extend(qualities)
        
        if evidence_qualities:
            avg_quality = sum(evidence_qualities) / len(evidence_qualities)
            features.append({
                "feature_type": "average_evidence_quality",
                "data": {"value": avg_quality},
                "sample_size": len(evidence_qualities),
            })
        
        # 5. 内容特征
        content_features = self._analyze_content_features(arguments)
        features.extend(content_features)
        
        return features
    
    def _analyze_content_features(
        self,
        arguments: List[Argument]
    ) -> List[Dict[str, Any]]:
        """分析内容特征"""
        features = []
        
        # 结构标记使用频率
        structure_markers = ["因为", "所以", "首先", "其次", "因此", "然而"]
        marker_counts = defaultdict(int)
        
        for arg in arguments:
            for marker in structure_markers:
                if marker in arg.content:
                    marker_counts[marker] += 1
        
        if marker_counts:
            features.append({
                "feature_type": "structure_marker_usage",
                "data": dict(marker_counts),
                "sample_size": len(arguments),
            })
        
        # 平均论点长度
        avg_length = sum(len(a.content) for a in arguments) / len(arguments)
        features.append({
            "feature_type": "average_content_length",
            "data": {"value": avg_length},
            "sample_size": len(arguments),
        })
        
        return features
    
    def get_discriminative_features(self) -> List[Dict[str, Any]]:
        """
        获取区分性特征
        比较胜方和败方的差异
        """
        if not self.winning_patterns or not self.losing_patterns:
            return []
        
        discriminative = []
        
        # 比较各特征
        win_features = {f["feature_type"]: f for f in self.winning_patterns}
        lose_features = {f["feature_type"]: f for f in self.losing_patterns}
        
        for feature_type in win_features:
            if feature_type in lose_features:
                win_data = win_features[feature_type]["data"]
                lose_data = lose_features[feature_type]["data"]
                
                # 计算差异
                if "value" in win_data and "value" in lose_data:
                    diff = win_data["value"] - lose_data["value"]
                    discriminative.append({
                        "feature_type": feature_type,
                        "winning_value": win_data["value"],
                        "losing_value": lose_data["value"],
                        "difference": diff,
                        "advantage": "winning" if diff > 0 else "losing",
                    })
        
        return discriminative


class PatternMiner:
    """
    模式挖掘器
    从论点中挖掘有效论证模式
    """
    
    ARGUMENT_PATTERNS = {
        "evidence_chain": {
            "template": "提出主张 -> 提供证据 -> 解释关联",
            "indicators": ["因为", "根据", "数据显示"],
        },
        "counter_example": {
            "template": "对方观点 -> 反例 -> 结论修正",
            "indicators": ["然而", "但是", "反过来看"],
        },
        "analogy": {
            "template": "相似案例 -> 类比推理 -> 结论",
            "indicators": ["类似", "好比", "同样"],
        },
        "causal_chain": {
            "template": "原因 -> 中间过程 -> 结果",
            "indicators": ["导致", "引起", "造成"],
        },
        "authority_appeal": {
            "template": "权威来源 -> 权威观点 -> 支持结论",
            "indicators": ["专家", "研究", "学者"],
        },
    }
    
    def __init__(self) -> None:
        self.discovered_patterns: Dict[str, KnowledgePattern] = {}
    
    def mine_patterns(
        self,
        arguments: List[Argument],
        quality_scores: Optional[Dict[str, float]] = None
    ) -> List[KnowledgePattern]:
        """
        挖掘论证模式
        
        Args:
            arguments: 论点列表
            quality_scores: 质量分数映射
            
        Returns:
            发现的模式列表
        """
        patterns = []
        
        for pattern_name, pattern_config in self.ARGUMENT_PATTERNS.items():
            matching_args = []
            
            for arg in arguments:
                # 检查是否匹配模式指标
                indicators = pattern_config["indicators"]
                match_count = sum(1 for ind in indicators if ind in arg.content)
                
                if match_count > 0:
                    matching_args.append((arg, match_count))
            
            if matching_args:
                # 计算模式质量
                if quality_scores:
                    scores = [
                        quality_scores.get(arg.argument_id, 0.5)
                        for arg, _ in matching_args
                    ]
                    avg_quality = sum(scores) / len(scores)
                else:
                    avg_quality = 0.5
                
                pattern = KnowledgePattern(
                    pattern_type=KnowledgeType.PROCEDURAL,
                    description=f"论证模式: {pattern_name}",
                    template=pattern_config["template"],
                    conditions=[f"包含指标词: {', '.join(pattern_config['indicators'])}"],
                    examples=[arg.content[:100] for arg, _ in matching_args[:3]],
                    confidence=avg_quality,
                    support_count=len(matching_args),
                )
                
                patterns.append(pattern)
                self.discovered_patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def identify_successful_sequences(
        self,
        arguments: List[Argument],
        rebuttals: List[Rebuttal]
    ) -> List[Dict[str, Any]]:
        """
        识别成功的论证序列
        
        Returns:
            成功序列列表
        """
        sequences = []
        
        # 按时间排序
        sorted_args = sorted(arguments, key=lambda a: a.timestamp)
        
        # 识别论点-反驳序列
        for rebuttal in rebuttals:
            target_id = rebuttal.target_argument_id
            target_arg = next(
                (a for a in sorted_args if a.argument_id == target_id),
                None
            )
            
            if target_arg:
                sequence = {
                    "type": "argument_rebuttal",
                    "argument": {
                        "id": target_arg.argument_id,
                        "stance": target_arg.stance.name,
                        "content": target_arg.content[:100],
                    },
                    "rebuttal": {
                        "id": rebuttal.rebuttal_id,
                        "effectiveness": rebuttal.effectiveness_score,
                        "content": rebuttal.content[:100],
                    },
                    "effectiveness": rebuttal.effectiveness_score,
                }
                sequences.append(sequence)
        
        return sequences


class ContrastiveAnalyzer:
    """
    对比分析器
    对比成功和失败的论证策略
    """
    
    def analyze_contrast(
        self,
        winning_arguments: List[Argument],
        losing_arguments: List[Argument]
    ) -> List[LearningInsight]:
        """
        对比分析
        
        Returns:
            学习洞察列表
        """
        insights = []
        
        # 1. 论点类型对比
        type_insight = self._compare_argument_types(
            winning_arguments, losing_arguments
        )
        if type_insight:
            insights.append(type_insight)
        
        # 2. 证据使用对比
        evidence_insight = self._compare_evidence_usage(
            winning_arguments, losing_arguments
        )
        if evidence_insight:
            insights.append(evidence_insight)
        
        # 3. 置信度对比
        confidence_insight = self._compare_confidence(
            winning_arguments, losing_arguments
        )
        if confidence_insight:
            insights.append(confidence_insight)
        
        # 4. 内容结构对比
        structure_insight = self._compare_structure(
            winning_arguments, losing_arguments
        )
        if structure_insight:
            insights.append(structure_insight)
        
        return insights
    
    def _compare_argument_types(
        self,
        winning: List[Argument],
        losing: List[Argument]
    ) -> Optional[LearningInsight]:
        """比较论点类型"""
        if not winning or not losing:
            return None
        
        # 统计类型分布
        win_types = defaultdict(int)
        lose_types = defaultdict(int)
        
        for arg in winning:
            win_types[arg.argument_type.name] += 1
        for arg in losing:
            lose_types[arg.argument_type.name] += 1
        
        # 找出差异最大的类型
        all_types = set(win_types.keys()) | set(lose_types.keys())
        max_diff = 0
        diff_type = None
        
        for t in all_types:
            win_ratio = win_types[t] / len(winning) if winning else 0
            lose_ratio = lose_types[t] / len(losing) if losing else 0
            diff = abs(win_ratio - lose_ratio)
            
            if diff > max_diff:
                max_diff = diff
                diff_type = t
        
        if diff_type and max_diff > 0.1:
            win_ratio = win_types[diff_type] / len(winning)
            lose_ratio = lose_types[diff_type] / len(losing)
            
            return LearningInsight(
                category="argument_type",
                description=f"胜方更倾向于使用{diff_type}类型论点",
                supporting_evidence=[
                    f"胜方{diff_type}类型占比: {win_ratio:.2%}",
                    f"败方{diff_type}类型占比: {lose_ratio:.2%}",
                ],
                importance=max_diff,
                recommendations=[
                    f"考虑增加{diff_type}类型论点的使用",
                ]
            )
        
        return None
    
    def _compare_evidence_usage(
        self,
        winning: List[Argument],
        losing: List[Argument]
    ) -> Optional[LearningInsight]:
        """比较证据使用"""
        if not winning or not losing:
            return None
        
        win_evidence = sum(len(a.evidence_list) for a in winning) / len(winning)
        lose_evidence = sum(len(a.evidence_list) for a in losing) / len(losing)
        
        diff = win_evidence - lose_evidence
        
        if abs(diff) > 0.3:
            direction = "更多" if diff > 0 else "更少"
            
            return LearningInsight(
                category="evidence_usage",
                description=f"胜方平均使用了{direction}的证据",
                supporting_evidence=[
                    f"胜方平均证据数: {win_evidence:.2f}",
                    f"败方平均证据数: {lose_evidence:.2f}",
                ],
                importance=abs(diff) / 2,
                recommendations=[
                    f"{'增加' if diff > 0 else '优化'}证据的使用",
                ]
            )
        
        return None
    
    def _compare_confidence(
        self,
        winning: List[Argument],
        losing: List[Argument]
    ) -> Optional[LearningInsight]:
        """比较置信度"""
        if not winning or not losing:
            return None
        
        win_conf = sum(a.confidence for a in winning) / len(winning)
        lose_conf = sum(a.confidence for a in losing) / len(losing)
        
        diff = win_conf - lose_conf
        
        if abs(diff) > 0.1:
            return LearningInsight(
                category="confidence",
                description=f"胜方论点平均置信度{'更高' if diff > 0 else '更低'}",
                supporting_evidence=[
                    f"胜方平均置信度: {win_conf:.2f}",
                    f"败方平均置信度: {lose_conf:.2f}",
                ],
                importance=abs(diff),
                recommendations=[
                    "调整论点置信度的表达方式",
                ]
            )
        
        return None
    
    def _compare_structure(
        self,
        winning: List[Argument],
        losing: List[Argument]
    ) -> Optional[LearningInsight]:
        """比较内容结构"""
        structure_markers = ["因为", "所以", "首先", "其次", "因此"]
        
        win_structure = 0
        lose_structure = 0
        
        for arg in winning:
            win_structure += sum(1 for m in structure_markers if m in arg.content)
        for arg in losing:
            lose_structure += sum(1 for m in structure_markers if m in arg.content)
        
        if winning:
            win_structure /= len(winning)
        if losing:
            lose_structure /= len(losing)
        
        diff = win_structure - lose_structure
        
        if abs(diff) > 0.3:
            return LearningInsight(
                category="structure",
                description=f"胜方论点结构{'更' if diff > 0 else '不'}清晰",
                supporting_evidence=[
                    f"胜方平均结构标记数: {win_structure:.2f}",
                    f"败方平均结构标记数: {lose_structure:.2f}",
                ],
                importance=abs(diff) / 2,
                recommendations=[
                    "使用更清晰的结构组织论点",
                    "使用逻辑连接词增强论证连贯性",
                ]
            )
        
        return None


class KnowledgeSynthesizer:
    """
    知识综合器
    将多个来源的知识综合成新知识
    """
    
    def __init__(self) -> None:
        self.knowledge_base: Dict[str, ExtractedKnowledge] = {}
    
    def synthesize(
        self,
        knowledge_items: List[ExtractedKnowledge],
        synthesis_type: str = "merge"
    ) -> ExtractedKnowledge:
        """
        综合多个知识项
        
        Args:
            knowledge_items: 知识项列表
            synthesis_type: 综合类型
            
        Returns:
            综合后的知识
        """
        if not knowledge_items:
            raise ValueError("知识列表不能为空")
        
        if synthesis_type == "merge":
            return self._merge_knowledge(knowledge_items)
        elif synthesis_type == "generalize":
            return self._generalize_knowledge(knowledge_items)
        else:
            return knowledge_items[0]
    
    def _merge_knowledge(
        self,
        items: List[ExtractedKnowledge]
    ) -> ExtractedKnowledge:
        """合并知识"""
        # 合并内容
        combined_content = " | ".join(item.summary or item.content[:100] 
                                      for item in items)
        
        # 合并来源
        all_sources = []
        for item in items:
            all_sources.extend(item.source_arguments)
        
        # 计算综合置信度
        confidences = [item.confidence for item in items]
        combined_confidence = sum(confidences) / len(confidences)
        
        # 合并适用场景
        all_applicability = []
        for item in items:
            all_applicability.extend(item.applicability)
        
        return ExtractedKnowledge(
            knowledge_type=items[0].knowledge_type,
            content=combined_content,
            summary=f"综合{len(items)}个知识源",
            source_arguments=list(set(all_sources)),
            extraction_method=ExtractionMethod.EVIDENCE_SYNTHESIS,
            confidence=combined_confidence,
            applicability=list(set(all_applicability)),
            related_knowledge=[item.knowledge_id for item in items],
        )
    
    def _generalize_knowledge(
        self,
        items: List[ExtractedKnowledge]
    ) -> ExtractedKnowledge:
        """泛化知识"""
        # 提取共同模式
        common_elements = self._find_common_elements(items)
        
        return ExtractedKnowledge(
            knowledge_type=KnowledgeType.STRATEGIC,
            content=f"泛化模式: {common_elements}",
            summary=f"从{len(items)}个实例中泛化",
            source_arguments=[],
            extraction_method=ExtractionMethod.PATTERN_MINING,
            confidence=sum(i.confidence for i in items) / len(items),
            related_knowledge=[item.knowledge_id for item in items],
        )
    
    def _find_common_elements(
        self,
        items: List[ExtractedKnowledge]
    ) -> str:
        """找出共同元素"""
        # 简化实现：找关键词交集
        all_words = []
        for item in items:
            words = set(item.content.split())
            all_words.append(words)
        
        if all_words:
            common = set.intersection(*all_words)
            return ", ".join(list(common)[:5])
        
        return ""
    
    def add_knowledge(self, knowledge: ExtractedKnowledge) -> None:
        """添加知识到知识库"""
        self.knowledge_base[knowledge.knowledge_id] = knowledge
    
    def get_knowledge(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        min_confidence: float = 0.0
    ) -> List[ExtractedKnowledge]:
        """获取知识"""
        results = list(self.knowledge_base.values())
        
        if knowledge_type:
            results = [k for k in results if k.knowledge_type == knowledge_type]
        
        results = [k for k in results if k.confidence >= min_confidence]
        
        return results


class LearningFromDebate:
    """
    辩论知识提取器
    主类，协调所有学习和提取功能
    """
    
    def __init__(self) -> None:
        self.winner_analyzer = WinnerAnalyzer()
        self.pattern_miner = PatternMiner()
        self.contrastive_analyzer = ContrastiveAnalyzer()
        self.knowledge_synthesizer = KnowledgeSynthesizer()
        
        self.extracted_knowledge: List[ExtractedKnowledge] = []
        self.learning_history: List[Dict[str, Any]] = []
    
    def extract_from_debate(
        self,
        debate_state: DebateState,
        verdict: Verdict,
        quality_scores: Optional[Dict[str, float]] = None
    ) -> List[ExtractedKnowledge]:
        """
        从辩论中提取知识
        
        Args:
            debate_state: 辩论状态
            verdict: 裁决结果
            quality_scores: 质量分数映射
            
        Returns:
            提取的知识列表
        """
        arguments = list(debate_state.arguments.values())
        rebuttals = list(debate_state.rebuttals.values())
        
        extracted = []
        
        # 1. 胜方分析
        winner_knowledge = self._extract_winner_knowledge(
            arguments, verdict, debate_state.debate_id
        )
        extracted.extend(winner_knowledge)
        
        # 2. 模式挖掘
        pattern_knowledge = self._extract_pattern_knowledge(
            arguments, rebuttals, quality_scores, debate_state.debate_id
        )
        extracted.extend(pattern_knowledge)
        
        # 3. 对比分析
        contrast_knowledge = self._extract_contrast_knowledge(
            arguments, verdict, debate_state.debate_id
        )
        extracted.extend(contrast_knowledge)
        
        # 4. 证据综合
        evidence_knowledge = self._extract_evidence_knowledge(
            arguments, verdict, debate_state.debate_id
        )
        extracted.extend(evidence_knowledge)
        
        # 记录历史
        self.learning_history.append({
            "debate_id": debate_state.debate_id,
            "timestamp": datetime.now().isoformat(),
            "knowledge_count": len(extracted),
            "verdict": verdict.winning_stance.name if verdict.winning_stance else None,
        })
        
        # 添加到知识库
        self.extracted_knowledge.extend(extracted)
        for k in extracted:
            self.knowledge_synthesizer.add_knowledge(k)
        
        return extracted
    
    def _extract_winner_knowledge(
        self,
        arguments: List[Argument],
        verdict: Verdict,
        debate_id: str
    ) -> List[ExtractedKnowledge]:
        """提取胜方知识"""
        knowledge_list = []
        
        # 分析胜方特征
        win_features, _ = self.winner_analyzer.analyze(arguments, verdict)
        
        # 提取胜方论点
        winning_args = [
            a for a in arguments 
            if a.stance == verdict.winning_stance
        ]
        
        for arg in winning_args:
            knowledge = ExtractedKnowledge(
                knowledge_type=self._map_argument_type_to_knowledge(arg.argument_type),
                content=arg.content,
                summary=f"胜方论点: {arg.content[:100]}",
                source_arguments=[arg.argument_id],
                source_debate_id=debate_id,
                extraction_method=ExtractionMethod.WINNER_ANALYSIS,
                confidence=arg.confidence * verdict.confidence,
                applicability=[arg.argument_type.name],
            )
            knowledge_list.append(knowledge)
        
        # 提取区分性特征作为知识
        discriminative = self.winner_analyzer.get_discriminative_features()
        for feature in discriminative:
            if abs(feature.get("difference", 0)) > 0.1:
                knowledge = ExtractedKnowledge(
                    knowledge_type=KnowledgeType.STRATEGIC,
                    content=f"策略洞察: {feature['feature_type']}",
                    summary=f"胜方在{feature['feature_type']}上表现更好",
                    source_debate_id=debate_id,
                    extraction_method=ExtractionMethod.WINNER_ANALYSIS,
                    confidence=abs(feature["difference"]),
                    metadata={"feature": feature},
                )
                knowledge_list.append(knowledge)
        
        return knowledge_list
    
    def _extract_pattern_knowledge(
        self,
        arguments: List[Argument],
        rebuttals: List[Rebuttal],
        quality_scores: Optional[Dict[str, float]],
        debate_id: str
    ) -> List[ExtractedKnowledge]:
        """提取模式知识"""
        knowledge_list = []
        
        # 挖掘论证模式
        patterns = self.pattern_miner.mine_patterns(arguments, quality_scores)
        
        for pattern in patterns:
            knowledge = ExtractedKnowledge(
                knowledge_type=KnowledgeType.PROCEDURAL,
                content=pattern.template,
                summary=pattern.description,
                source_debate_id=debate_id,
                extraction_method=ExtractionMethod.PATTERN_MINING,
                confidence=pattern.confidence,
                applicability=pattern.conditions,
                metadata={"pattern": pattern.to_dict()},
            )
            knowledge_list.append(knowledge)
        
        # 识别成功序列
        sequences = self.pattern_miner.identify_successful_sequences(
            arguments, rebuttals
        )
        
        for seq in sequences[:5]:  # 限制数量
            knowledge = ExtractedKnowledge(
                knowledge_type=KnowledgeType.PROCEDURAL,
                content=f"论证序列: {seq['type']}",
                summary=f"有效性: {seq['effectiveness']:.2f}",
                source_debate_id=debate_id,
                extraction_method=ExtractionMethod.PATTERN_MINING,
                confidence=seq["effectiveness"],
            )
            knowledge_list.append(knowledge)
        
        return knowledge_list
    
    def _extract_contrast_knowledge(
        self,
        arguments: List[Argument],
        verdict: Verdict,
        debate_id: str
    ) -> List[ExtractedKnowledge]:
        """提取对比知识"""
        knowledge_list = []
        
        winning_stance = verdict.winning_stance
        winning_args = [a for a in arguments if a.stance == winning_stance]
        losing_args = [a for a in arguments 
                       if a.stance != winning_stance and a.stance != Stance.NEUTRAL]
        
        # 对比分析
        insights = self.contrastive_analyzer.analyze_contrast(
            winning_args, losing_args
        )
        
        for insight in insights:
            knowledge = ExtractedKnowledge(
                knowledge_type=KnowledgeType.STRATEGIC,
                content=insight.description,
                summary=f"洞察类别: {insight.category}",
                source_debate_id=debate_id,
                extraction_method=ExtractionMethod.CONTRASTIVE,
                confidence=insight.importance,
                applicability=insight.recommendations,
                metadata={
                    "supporting_evidence": insight.supporting_evidence,
                    "actionability": insight.actionability,
                },
            )
            knowledge_list.append(knowledge)
        
        return knowledge_list
    
    def _extract_evidence_knowledge(
        self,
        arguments: List[Argument],
        verdict: Verdict,
        debate_id: str
    ) -> List[ExtractedKnowledge]:
        """提取证据知识"""
        knowledge_list = []
        
        # 收集胜方证据
        winning_args = [
            a for a in arguments 
            if a.stance == verdict.winning_stance
        ]
        
        for arg in winning_args:
            for evidence in arg.evidence_list:
                knowledge = ExtractedKnowledge(
                    knowledge_type=KnowledgeType.EVIDENTIAL,
                    content=evidence.content,
                    summary=f"证据来源: {evidence.source}",
                    source_arguments=[arg.argument_id],
                    source_debate_id=debate_id,
                    extraction_method=ExtractionMethod.EVIDENCE_SYNTHESIS,
                    confidence=evidence.quality_score(),
                    applicability=[f"可信度: {evidence.credibility:.2f}"],
                )
                knowledge_list.append(knowledge)
        
        return knowledge_list
    
    def _map_argument_type_to_knowledge(
        self,
        arg_type: ArgumentType
    ) -> KnowledgeType:
        """映射论点类型到知识类型"""
        mapping = {
            ArgumentType.FACTUAL: KnowledgeType.FACTUAL,
            ArgumentType.NORMATIVE: KnowledgeType.STRATEGIC,
            ArgumentType.CAUSAL: KnowledgeType.PROCEDURAL,
            ArgumentType.ANALOGICAL: KnowledgeType.STRATEGIC,
            ArgumentType.AUTHORITY: KnowledgeType.EVIDENTIAL,
            ArgumentType.PRAGMATIC: KnowledgeType.PROCEDURAL,
        }
        return mapping.get(arg_type, KnowledgeType.FACTUAL)
    
    def get_knowledge_by_type(
        self,
        knowledge_type: KnowledgeType
    ) -> List[ExtractedKnowledge]:
        """按类型获取知识"""
        return [
            k for k in self.extracted_knowledge
            if k.knowledge_type == knowledge_type
        ]
    
    def get_top_knowledge(
        self,
        limit: int = 10,
        min_confidence: float = 0.5
    ) -> List[ExtractedKnowledge]:
        """获取高质量知识"""
        filtered = [
            k for k in self.extracted_knowledge
            if k.confidence >= min_confidence
        ]
        
        sorted_knowledge = sorted(
            filtered,
            key=lambda k: k.confidence,
            reverse=True
        )
        
        return sorted_knowledge[:limit]
    
    def synthesize_knowledge(
        self,
        knowledge_ids: List[str]
    ) -> Optional[ExtractedKnowledge]:
        """综合指定知识"""
        items = [
            self.knowledge_synthesizer.knowledge_base.get(kid)
            for kid in knowledge_ids
        ]
        items = [k for k in items if k is not None]
        
        if not items:
            return None
        
        return self.knowledge_synthesizer.synthesize(items)
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习摘要"""
        if not self.extracted_knowledge:
            return {"message": "暂无提取的知识"}
        
        # 按类型统计
        type_counts = defaultdict(int)
        for k in self.extracted_knowledge:
            type_counts[k.knowledge_type.value] += 1
        
        # 按提取方法统计
        method_counts = defaultdict(int)
        for k in self.extracted_knowledge:
            method_counts[k.extraction_method.value] += 1
        
        # 置信度统计
        confidences = [k.confidence for k in self.extracted_knowledge]
        
        return {
            "total_knowledge": len(self.extracted_knowledge),
            "debates_analyzed": len(self.learning_history),
            "type_distribution": dict(type_counts),
            "method_distribution": dict(method_counts),
            "confidence_stats": {
                "mean": statistics.mean(confidences),
                "median": statistics.median(confidences),
                "min": min(confidences),
                "max": max(confidences),
            },
        }


__all__ = [
    "KnowledgeType",
    "ExtractionMethod",
    "KnowledgePattern",
    "ExtractedKnowledge",
    "LearningInsight",
    "WinnerAnalyzer",
    "PatternMiner",
    "ContrastiveAnalyzer",
    "KnowledgeSynthesizer",
    "LearningFromDebate",
]
