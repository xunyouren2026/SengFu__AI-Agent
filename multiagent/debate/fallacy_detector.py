"""
逻辑谬误检测模块
识别循环论证、稻草人等错误
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Pattern
from enum import Enum, auto
import re
from collections import defaultdict

from .protocol import Argument, Rebuttal


class FallacyType(Enum):
    """逻辑谬误类型枚举"""
    CIRCULAR_REASONING = "circular_reasoning"           # 循环论证
    STRAW_MAN = "straw_man"                             # 稻草人谬误
    AD_HOMINEM = "ad_hominem"                           # 人身攻击
    FALSE_DICHOTOMY = "false_dichotomy"                 # 虚假两难
    SLIPPERY_SLOPE = "slippery_slope"                   # 滑坡谬误
    APPEAL_TO_AUTHORITY = "appeal_to_authority"         # 诉诸权威
    APPEAL_TO_EMOTION = "appeal_to_emotion"             # 诉诸情感
    HASTY_GENERALIZATION = "hasty_generalization"       # 草率概括
    POST_HOC = "post_hoc"                               # 后此谬误
    BEGGING_THE_QUESTION = "begging_the_question"       # 乞求论点
    FALSE_CAUSE = "false_cause"                         # 虚假因果
    RED_HERRING = "red_herring"                         # 红鲱鱼
    TU_QUOQUE = "tu_quoque"                             # 你也一样
    NO_TRUE_SCOTSMAN = "no_true_scotsman"               # 没有真正的苏格兰人
    GENETIC_FALLACY = "genetic_fallacy"                 # 基因谬误


@dataclass
class FallacyDetection:
    """谬误检测结果"""
    fallacy_type: FallacyType
    confidence: float                              # 检测置信度
    matched_text: str                              # 匹配的文本
    explanation: str                               # 解释
    suggestion: str                                # 改进建议
    severity: str = "medium"                       # 严重程度: low/medium/high


@dataclass
class ArgumentAnalysis:
    """论点分析结果"""
    argument_id: str
    fallacies_detected: List[FallacyDetection] = field(default_factory=list)
    overall_quality_score: float = 1.0
    risk_level: str = "low"                        # low/medium/high
    
    def has_critical_fallacies(self) -> bool:
        """检查是否有严重谬误"""
        return any(
            f.severity == "high" or f.confidence > 0.8
            for f in self.fallacies_detected
        )


class FallacyPattern:
    """谬误检测模式"""
    
    def __init__(
        self,
        fallacy_type: FallacyType,
        patterns: List[Pattern[str]],
        keywords: List[str],
        description: str,
        suggestion: str
    ) -> None:
        self.fallacy_type = fallacy_type
        self.patterns = patterns
        self.keywords = keywords
        self.description = description
        self.suggestion = suggestion


class FallacyPatternDatabase:
    """
    谬误模式数据库
    存储各种逻辑谬误的检测模式
    """
    
    def __init__(self) -> None:
        self.patterns: List[FallacyPattern] = []
        self._initialize_patterns()
    
    def _initialize_patterns(self) -> None:
        """初始化检测模式"""
        # 循环论证模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.CIRCULAR_REASONING,
            patterns=[
                re.compile(r'因为.*所以.*因为', re.IGNORECASE),
                re.compile(r'显然.*因为.*显然', re.IGNORECASE),
            ],
            keywords=["因为A所以A", "循环", "显然", "毫无疑问"],
            description="结论被用作前提，形成循环论证",
            suggestion="提供独立于结论的证据支持您的主张"
        ))
        
        # 稻草人谬误模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.STRAW_MAN,
            patterns=[
                re.compile(r'你的意思.*其实', re.IGNORECASE),
                re.compile(r'实际上你是在说', re.IGNORECASE),
            ],
            keywords=["歪曲", "曲解", "实际上", "你的意思是"],
            description="歪曲对方的论点，然后攻击歪曲后的版本",
            suggestion="准确理解并回应对方的真实论点"
        ))
        
        # 人身攻击模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.AD_HOMINEM,
            patterns=[
                re.compile(r'你(这个人|自己|们).*还', re.IGNORECASE),
                re.compile(r'像你这样.*的人', re.IGNORECASE),
            ],
            keywords=["你这个人", "你自己", "像你这样", "无知", "愚蠢"],
            description="攻击提出论点的人，而非论点本身",
            suggestion="专注于论点的内容，而非提出者的个人特征"
        ))
        
        # 虚假两难模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.FALSE_DICHOTOMY,
            patterns=[
                re.compile(r'要么.*要么.*没有', re.IGNORECASE),
                re.compile(r'只有两种选择.*要么', re.IGNORECASE),
                re.compile(r'不是.*就是', re.IGNORECASE),
            ],
            keywords=["要么", "要么", "只有两种", "不是就是", "非此即彼"],
            description="将复杂问题简化为只有两种极端选择",
            suggestion="考虑是否存在中间立场或其他选择"
        ))
        
        # 滑坡谬误模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.SLIPPERY_SLOPE,
            patterns=[
                re.compile(r'一旦.*就会.*然后.*最终', re.IGNORECASE),
                re.compile(r'如果.*那么.*接着.*最后', re.IGNORECASE),
            ],
            keywords=["一旦", "就会", "然后", "最终", "导致", "不可避免地"],
            description="假设一个事件将不可避免地导致一系列负面结果",
            suggestion="证明每个环节之间的因果联系，而非假设连锁反应"
        ))
        
        # 诉诸权威模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.APPEAL_TO_AUTHORITY,
            patterns=[
                re.compile(r'专家说.*所以', re.IGNORECASE),
                re.compile(r'权威.*认为.*因此', re.IGNORECASE),
            ],
            keywords=["专家说", "权威", "某某认为", "大师说"],
            description="仅仅因为权威说过就认为是正确的",
            suggestion="提供权威观点的具体证据和推理过程"
        ))
        
        # 诉诸情感模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.APPEAL_TO_EMOTION,
            patterns=[
                re.compile(r'想想.*多么.*难道', re.IGNORECASE),
                re.compile(r'难道你不觉得.*吗', re.IGNORECASE),
            ],
            keywords=["想想", "多么", "难道", "令人发指", "无法接受"],
            description="使用情感诉求代替逻辑论证",
            suggestion="用逻辑和证据支持论点，而非仅仅诉诸情感"
        ))
        
        # 草率概括模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.HASTY_GENERALIZATION,
            patterns=[
                re.compile(r'所有.*都.*因为.*我', re.IGNORECASE),
                re.compile(r'每次.*总是.*有一次', re.IGNORECASE),
            ],
            keywords=["所有", "都", "总是", "每次", "从来", "绝对"],
            description="基于少量样本做出普遍性结论",
            suggestion="提供更多样化的证据支持普遍性主张"
        ))
        
        # 后此谬误模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.POST_HOC,
            patterns=[
                re.compile(r'之后.*所以.*因为', re.IGNORECASE),
                re.compile(r'自从.*发生了.*因此', re.IGNORECASE),
            ],
            keywords=["之后", "所以", "自从", "因此", "导致"],
            description="仅仅因为A发生在B之前，就认为A导致B",
            suggestion="证明因果关系，而非仅仅时间顺序"
        ))
        
        # 乞求论点模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.BEGGING_THE_QUESTION,
            patterns=[
                re.compile(r'显然.*因为.*显然', re.IGNORECASE),
                re.compile(r'毫无疑问.*因为.*毫无疑问', re.IGNORECASE),
            ],
            keywords=["显然", "毫无疑问", "众所周知", "明显"],
            description="假设了需要证明的结论",
            suggestion="明确证明您的前提，而非假设其正确性"
        ))
        
        # 虚假因果模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.FALSE_CAUSE,
            patterns=[
                re.compile(r'导致.*因为.*相关', re.IGNORECASE),
                re.compile(r'引起.*由于.*同时', re.IGNORECASE),
            ],
            keywords=["导致", "引起", "造成", "相关", "有关"],
            description="将相关性误认为因果性",
            suggestion="证明因果机制，而非仅仅指出相关性"
        ))
        
        # 红鲱鱼模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.RED_HERRING,
            patterns=[
                re.compile(r'且不说.*更重要的是', re.IGNORECASE),
                re.compile(r'这个问题不重要.*关键是', re.IGNORECASE),
            ],
            keywords=["且不说", "更重要的是", "关键是", "问题在于"],
            description="引入无关话题转移注意力",
            suggestion="保持讨论焦点，回应核心论点"
        ))
        
        # 你也一样模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.TU_QUOQUE,
            patterns=[
                re.compile(r'你自己也.*凭什么', re.IGNORECASE),
                re.compile(r'你不也.*有什么资格', re.IGNORECASE),
            ],
            keywords=["你自己也", "你不也", "凭什么", "有什么资格"],
            description="指出对方也犯同样的错误来反驳",
            suggestion="回应批评的内容，而非批评者的行为"
        ))
        
        # 没有真正的苏格兰人模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.NO_TRUE_SCOTSMAN,
            patterns=[
                re.compile(r'真正的.*不会.*那些', re.IGNORECASE),
                re.compile(r'真正的.*应该.*而不是', re.IGNORECASE),
            ],
            keywords=["真正的", "真正的", "不是真正的", "应该"],
            description="通过重新定义排除反例",
            suggestion="承认反例的存在，调整您的普遍性主张"
        ))
        
        # 基因谬误模式
        self.patterns.append(FallacyPattern(
            fallacy_type=FallacyType.GENETIC_FALLACY,
            patterns=[
                re.compile(r'因为.*来自.*所以', re.IGNORECASE),
                re.compile(r'源于.*因此.*不好', re.IGNORECASE),
            ],
            keywords=["来自", "源于", "出身", "背景", "来源"],
            description="根据事物的来源判断其价值",
            suggestion="评估论点本身的优劣，而非其来源"
        ))


class FallacyDetector:
    """
    逻辑谬误检测器
    识别论证中的逻辑谬误
    """
    
    def __init__(self) -> None:
        self.pattern_db = FallacyPatternDatabase()
        self.detection_history: List[ArgumentAnalysis] = []
    
    def analyze_argument(self, argument: Argument) -> ArgumentAnalysis:
        """
        分析论点中的谬误
        
        Args:
            argument: 要分析的论点
            
        Returns:
            分析结果
        """
        analysis = ArgumentAnalysis(argument_id=argument.argument_id)
        content = argument.content
        
        # 应用所有模式进行检测
        for pattern in self.pattern_db.patterns:
            detection = self._apply_pattern(content, pattern)
            if detection:
                analysis.fallacies_detected.append(detection)
        
        # 计算整体质量分数
        if analysis.fallacies_detected:
            total_penalty = sum(
                d.confidence * (1.5 if d.severity == "high" else 1.0 if d.severity == "medium" else 0.5)
                for d in analysis.fallacies_detected
            )
            analysis.overall_quality_score = max(0.0, 1.0 - total_penalty / len(analysis.fallacies_detected))
        
        # 确定风险等级
        if analysis.has_critical_fallacies():
            analysis.risk_level = "high"
        elif len(analysis.fallacies_detected) >= 2:
            analysis.risk_level = "medium"
        
        self.detection_history.append(analysis)
        return analysis
    
    def _apply_pattern(
        self,
        content: str,
        pattern: FallacyPattern
    ) -> Optional[FallacyDetection]:
        """应用单个模式进行检测"""
        max_confidence = 0.0
        matched_text = ""
        
        # 检查正则模式
        for regex in pattern.patterns:
            match = regex.search(content)
            if match:
                confidence = 0.7  # 模式匹配的基础置信度
                if confidence > max_confidence:
                    max_confidence = confidence
                    matched_text = match.group(0)
        
        # 检查关键词
        keyword_matches = sum(1 for kw in pattern.keywords if kw in content)
        if keyword_matches > 0:
            keyword_confidence = min(0.9, 0.4 + keyword_matches * 0.15)
            if keyword_confidence > max_confidence:
                max_confidence = keyword_confidence
                # 找到匹配的关键词上下文
                for kw in pattern.keywords:
                    if kw in content:
                        idx = content.find(kw)
                        matched_text = content[max(0, idx-10):min(len(content), idx+20)]
                        break
        
        if max_confidence > 0.5:  # 置信度阈值
            severity = self._determine_severity(max_confidence, pattern.fallacy_type)
            return FallacyDetection(
                fallacy_type=pattern.fallacy_type,
                confidence=max_confidence,
                matched_text=matched_text,
                explanation=pattern.description,
                suggestion=pattern.suggestion,
                severity=severity
            )
        
        return None
    
    def _determine_severity(
        self,
        confidence: float,
        fallacy_type: FallacyType
    ) -> str:
        """确定谬误严重程度"""
        # 某些谬误类型天然更严重
        critical_fallacies = {
            FallacyType.CIRCULAR_REASONING,
            FallacyType.FALSE_DICHOTOMY,
            FallacyType.AD_HOMINEM,
        }
        
        if fallacy_type in critical_fallacies and confidence > 0.7:
            return "high"
        elif confidence > 0.8:
            return "high"
        elif confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def analyze_rebuttal(
        self,
        rebuttal: Rebuttal,
        target_argument: Argument
    ) -> ArgumentAnalysis:
        """
        分析反驳中的谬误
        
        Args:
            rebuttal: 反驳
            target_argument: 目标论点
            
        Returns:
            分析结果
        """
        # 创建临时论点进行分析
        temp_argument = Argument(
            argument_id=rebuttal.rebuttal_id,
            speaker_id=rebuttal.speaker_id,
            content=rebuttal.content,
        )
        
        analysis = self.analyze_argument(temp_argument)
        analysis.argument_id = rebuttal.rebuttal_id
        
        # 额外检查：稻草人检测
        straw_man_check = self._detect_straw_man(rebuttal, target_argument)
        if straw_man_check:
            analysis.fallacies_detected.append(straw_man_check)
        
        return analysis
    
    def _detect_straw_man(
        self,
        rebuttal: Rebuttal,
        target_argument: Argument
    ) -> Optional[FallacyDetection]:
        """检测稻草人谬误"""
        # 检查反驳内容是否准确反映目标论点
        target_content = target_argument.content.lower()
        rebuttal_content = rebuttal.content.lower()
        
        # 简单的相似度检查
        target_words = set(target_content.split())
        rebuttal_words = set(rebuttal_content.split())
        
        if not target_words:
            return None
        
        overlap = len(target_words & rebuttal_words)
        similarity = overlap / len(target_words)
        
        # 如果相似度很低，可能是稻草人
        if similarity < 0.3 and len(rebuttal_content) > 20:
            return FallacyDetection(
                fallacy_type=FallacyType.STRAW_MAN,
                confidence=0.6 + (0.3 - similarity),
                matched_text=rebuttal.content[:50],
                explanation="反驳可能没有准确反映对方的论点",
                suggestion="确保您理解并准确回应对方的真实论点",
                severity="medium"
            )
        
        return None
    
    def batch_analyze(
        self,
        arguments: List[Argument]
    ) -> Dict[str, ArgumentAnalysis]:
        """
        批量分析论点
        
        Args:
            arguments: 论点列表
            
        Returns:
            论点ID到分析结果的映射
        """
        results = {}
        for argument in arguments:
            analysis = self.analyze_argument(argument)
            results[argument.argument_id] = analysis
        return results
    
    def get_debate_fallacy_report(
        self,
        debate_id: str
    ) -> Dict[str, Any]:
        """获取辩论谬误报告"""
        relevant_analyses = [
            a for a in self.detection_history
            if a.argument_id.startswith(debate_id)
        ]
        
        if not relevant_analyses:
            return {"message": "该辩论暂无谬误检测记录"}
        
        # 统计各类谬误
        fallacy_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for analysis in relevant_analyses:
            for detection in analysis.fallacies_detected:
                fallacy_counts[detection.fallacy_type.value] += 1
                severity_counts[detection.severity] += 1
        
        # 找出问题最严重的论点
        problematic_args = sorted(
            relevant_analyses,
            key=lambda a: len(a.fallacies_detected),
            reverse=True
        )[:5]
        
        return {
            "debate_id": debate_id,
            "total_arguments_analyzed": len(relevant_analyses),
            "arguments_with_fallacies": sum(
                1 for a in relevant_analyses if a.fallacies_detected
            ),
            "fallacy_type_distribution": dict(fallacy_counts),
            "severity_distribution": dict(severity_counts),
            "average_quality_score": sum(
                a.overall_quality_score for a in relevant_analyses
            ) / len(relevant_analyses),
            "most_problematic_arguments": [
                {
                    "argument_id": a.argument_id,
                    "fallacy_count": len(a.fallacies_detected),
                    "risk_level": a.risk_level,
                }
                for a in problematic_args
            ],
        }
    
    def generate_feedback(
        self,
        analysis: ArgumentAnalysis
    ) -> str:
        """生成改进反馈"""
        if not analysis.fallacies_detected:
            return "您的论证逻辑清晰，未发现明显谬误。"
        
        feedback_parts = ["检测到以下逻辑问题："]
        
        for i, detection in enumerate(analysis.fallacies_detected[:3], 1):
            feedback_parts.append(
                f"{i}. {detection.fallacy_type.value}: {detection.explanation}\n"
                f"   建议：{detection.suggestion}"
            )
        
        if len(analysis.fallacies_detected) > 3:
            feedback_parts.append(
                f"... 还有 {len(analysis.fallacies_detected) - 3} 个问题"
            )
        
        return "\n\n".join(feedback_parts)


class FallacyPreventionAdvisor:
    """
    谬误预防顾问
    提供避免逻辑谬误的建议
    """
    
    PREVENTION_TIPS: Dict[FallacyType, List[str]] = {
        FallacyType.CIRCULAR_REASONING: [
            "确保您的前提独立于结论",
            "尝试用不同的方式表达您的前提和结论",
            "检查您是否假设了需要证明的内容",
        ],
        FallacyType.STRAW_MAN: [
            "在反驳前，用自己的话复述对方的论点",
            "请对方确认您的理解是否正确",
            "针对对方最强的论点进行反驳",
        ],
        FallacyType.AD_HOMINEM: [
            "专注于论点本身，而非提出者",
            "即使对方有缺陷，其论点仍可能正确",
            "区分对人评价和对事评价",
        ],
        FallacyType.FALSE_DICHOTOMY: [
            "考虑是否存在第三种选择",
            "探索中间立场的可能性",
            "承认问题的复杂性",
        ],
        FallacyType.HASTY_GENERALIZATION: [
            "使用限定词（如'某些'、'可能'）",
            "提供更多样化的证据",
            "承认例外情况的存在",
        ],
    }
    
    @classmethod
    def get_prevention_tips(cls, fallacy_type: FallacyType) -> List[str]:
        """获取预防建议"""
        return cls.PREVENTION_TIPS.get(fallacy_type, ["仔细检查您的论证逻辑"])
    
    @classmethod
    def get_general_guidelines(cls) -> List[str]:
        """获取一般性指导原则"""
        return [
            "明确区分事实和观点",
            "为每个主张提供证据支持",
            "考虑反方观点并予以回应",
            "使用精确的语言，避免绝对化表述",
            "检查因果关系的有效性",
            "确保结论严格遵循前提",
        ]


__all__ = [
    "FallacyType",
    "FallacyDetection",
    "ArgumentAnalysis",
    "FallacyPattern",
    "FallacyPatternDatabase",
    "FallacyDetector",
    "FallacyPreventionAdvisor",
]
