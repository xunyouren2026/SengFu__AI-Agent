"""
辩论参与者Agent模块
接收论点并生成反驳或支持
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import random
import re

from .protocol import (
    Argument, Rebuttal, Evidence, Revision,
    Stance, ArgumentType, DebatePhase, DebateState
)


class ParticipantType(Enum):
    """参与者类型"""
    PROPONENT = "proponent"      # 支持者
    OPPONENT = "opponent"        # 反对者
    NEUTRAL = "neutral"          # 中立者
    MODERATOR = "moderator"      # 主持人
    EXPERT = "expert"            # 专家


class StrategyType(Enum):
    """辩论策略类型"""
    AGGRESSIVE = "aggressive"    # 攻击性策略
    DEFENSIVE = "defensive"      # 防御性策略
    BALANCED = "balanced"        # 平衡策略
    EVIDENCE_BASED = "evidence_based"  # 证据导向
    RHETORICAL = "rhetorical"    # 修辞导向


@dataclass
class ParticipantProfile:
    """参与者档案"""
    participant_id: str
    name: str = ""
    participant_type: ParticipantType = ParticipantType.NEUTRAL
    expertise_domains: Set[str] = field(default_factory=set)
    debate_style: StrategyType = StrategyType.BALANCED
    personality_traits: Dict[str, float] = field(default_factory=dict)
    # 性格特质：开放性、尽责性、外向性、宜人性、神经质
    credibility_score: float = 0.5
    past_performance: List[float] = field(default_factory=list)
    
    def get_average_performance(self) -> float:
        """获取平均表现分数"""
        if not self.past_performance:
            return 0.5
        return sum(self.past_performance) / len(self.past_performance)


class ArgumentGenerator:
    """
    论点生成器
    基于不同策略生成论点
    """
    
    def __init__(self, profile: ParticipantProfile) -> None:
        self.profile = profile
        self.argument_templates: Dict[ArgumentType, List[str]] = {
            ArgumentType.FACTUAL: [
                "根据数据显示，{claim}",
                "事实表明，{claim}",
                "实证研究证明，{claim}",
            ],
            ArgumentType.NORMATIVE: [
                "我们应该认为，{claim}",
                "从价值角度看，{claim}",
                "基于道德原则，{claim}",
            ],
            ArgumentType.CAUSAL: [
                "这导致了{claim}",
                "原因是{claim}",
                "因此可以推断，{claim}",
            ],
            ArgumentType.ANALOGICAL: [
                "类似于{example}，{claim}",
                "这就好比{example}，{claim}",
            ],
            ArgumentType.AUTHORITY: [
                "根据{authority}的研究，{claim}",
                "{authority}指出，{claim}",
            ],
            ArgumentType.PRAGMATIC: [
                "实际应用表明，{claim}",
                "从实用角度，{claim}",
            ],
        }
    
    def generate_argument(
        self,
        topic: str,
        stance: Stance,
        target_argument: Optional[Argument] = None,
        context: Optional[List[Argument]] = None
    ) -> Argument:
        """
        生成论点
        
        Args:
            topic: 辩论主题
            stance: 立场
            target_argument: 目标论点（用于反驳）
            context: 上下文论点列表
        """
        # 根据策略选择论点类型
        argument_type = self._select_argument_type()
        
        # 生成论点内容
        content = self._construct_content(topic, stance, argument_type, target_argument)
        
        argument = Argument(
            speaker_id=self.profile.participant_id,
            content=content,
            stance=stance,
            argument_type=argument_type,
            target_argument_id=target_argument.argument_id if target_argument else None,
            confidence=self._calculate_confidence()
        )
        
        # 根据策略决定是否添加证据
        if self.profile.debate_style in [StrategyType.EVIDENCE_BASED, StrategyType.BALANCED]:
            evidence = self._generate_evidence(topic, content)
            if evidence:
                argument.add_evidence(evidence)
        
        return argument
    
    def _select_argument_type(self) -> ArgumentType:
        """根据参与者特征选择论点类型"""
        weights = {
            StrategyType.AGGRESSIVE: [0.3, 0.2, 0.3, 0.1, 0.05, 0.05],
            StrategyType.DEFENSIVE: [0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
            StrategyType.BALANCED: [0.25, 0.2, 0.2, 0.15, 0.1, 0.1],
            StrategyType.EVIDENCE_BASED: [0.5, 0.1, 0.2, 0.05, 0.1, 0.05],
            StrategyType.RHETORICAL: [0.1, 0.3, 0.1, 0.3, 0.1, 0.1],
        }
        
        style_weights = weights.get(self.profile.debate_style, weights[StrategyType.BALANCED])
        types = list(ArgumentType)
        return random.choices(types, weights=style_weights)[0]
    
    def _construct_content(
        self,
        topic: str,
        stance: Stance,
        argument_type: ArgumentType,
        target_argument: Optional[Argument] = None
    ) -> str:
        """构建论点内容"""
        templates = self.argument_templates.get(argument_type, ["{claim}"])
        template = random.choice(templates)
        
        # 根据立场构建主张
        if stance == Stance.PRO:
            claim = f"{topic} 是正确的，因为..."
        elif stance == Stance.CON:
            claim = f"{topic} 存在问题，因为..."
        else:
            claim = f"关于 {topic} 需要更全面的考虑..."
        
        # 如果是反驳，针对目标论点
        if target_argument:
            if stance == Stance.CON:
                claim = f"对方关于 '{target_argument.content[:50]}...' 的观点不成立，因为..."
            else:
                claim = f"我支持 '{target_argument.content[:50]}...' 的观点，并补充..."
        
        # 填充模板
        if "{example}" in template:
            example = self._get_analogy_example(topic)
            content = template.format(example=example, claim=claim)
        elif "{authority}" in template:
            authority = self._get_authority_reference()
            content = template.format(authority=authority, claim=claim)
        else:
            content = template.format(claim=claim)
        
        return content
    
    def _generate_evidence(self, topic: str, claim: str) -> Optional[Evidence]:
        """生成证据"""
        # 模拟证据生成
        evidence = Evidence(
            source=f"研究文献_{random.randint(1000, 9999)}",
            content=f"关于{topic}的实证数据支持该论点",
            credibility=random.uniform(0.6, 0.95),
            relevance=random.uniform(0.5, 0.9)
        )
        return evidence
    
    def _calculate_confidence(self) -> float:
        """计算置信度"""
        base_confidence = self.profile.credibility_score
        performance_factor = self.profile.get_average_performance()
        
        # 根据策略调整
        adjustments = {
            StrategyType.AGGRESSIVE: 0.1,
            StrategyType.DEFENSIVE: -0.05,
            StrategyType.BALANCED: 0.0,
            StrategyType.EVIDENCE_BASED: 0.05,
            StrategyType.RHETORICAL: 0.08,
        }
        
        adjustment = adjustments.get(self.profile.debate_style, 0.0)
        confidence = (base_confidence + performance_factor) / 2 + adjustment
        return max(0.1, min(0.95, confidence))
    
    def _get_analogy_example(self, topic: str) -> str:
        """获取类比例子"""
        examples = [
            "建造房屋需要稳固的基础",
            "生态系统需要平衡",
            "学习语言需要持续练习",
            "市场经济需要调控",
        ]
        return random.choice(examples)
    
    def _get_authority_reference(self) -> str:
        """获取权威引用"""
        authorities = [
            "相关领域专家",
            "权威研究机构",
            "历史数据",
            "同行评议文献",
        ]
        return random.choice(authorities)


class RebuttalGenerator:
    """
    反驳生成器
    生成针对论点的反驳
    """
    
    REBUTTAL_PATTERNS = {
        "logical": [
            "对方的论证存在逻辑漏洞：{reason}",
            "这一推理忽略了关键因素：{reason}",
            "从逻辑上讲，这无法成立，因为{reason}",
        ],
        "evidence": [
            "对方提供的证据不足：{reason}",
            "有相反证据表明{reason}",
            "该证据的可信度存疑，因为{reason}",
        ],
        "relevance": [
            "这与当前议题关联不大，{reason}",
            "对方偏离了核心问题：{reason}",
        ],
        "assumption": [
            "对方的论证基于未经证实的假设：{reason}",
            "这一观点隐含了错误的预设：{reason}",
        ],
    }
    
    def __init__(self, profile: ParticipantProfile) -> None:
        self.profile = profile
    
    def generate_rebuttal(
        self,
        target_argument: Argument,
        debate_context: Optional[DebateState] = None
    ) -> Rebuttal:
        """
        生成反驳
        
        Args:
            target_argument: 要反驳的目标论点
            debate_context: 辩论上下文
        """
        # 选择反驳类型
        rebuttal_type = self._select_rebuttal_type(target_argument)
        
        # 生成反驳理由
        reason = self._generate_reason(target_argument, rebuttal_type)
        
        # 构建反驳内容
        templates = self.REBUTTAL_PATTERNS.get(rebuttal_type, self.REBUTTAL_PATTERNS["logical"])
        template = random.choice(templates)
        content = template.format(reason=reason)
        
        # 计算有效性分数
        effectiveness = self._calculate_effectiveness(target_argument, rebuttal_type)
        
        return Rebuttal(
            speaker_id=self.profile.participant_id,
            target_argument_id=target_argument.argument_id,
            content=content,
            rebuttal_type=rebuttal_type,
            effectiveness_score=effectiveness
        )
    
    def _select_rebuttal_type(self, target_argument: Argument) -> str:
        """选择反驳类型"""
        # 根据目标论点特征选择反驳策略
        if not target_argument.evidence_list:
            return "evidence"
        
        if target_argument.argument_type == ArgumentType.CAUSAL:
            return "logical"
        
        if target_argument.argument_type == ArgumentType.ANALOGICAL:
            return "assumption"
        
        # 根据参与者风格
        weights = {
            StrategyType.AGGRESSIVE: {"logical": 0.4, "evidence": 0.2, "relevance": 0.2, "assumption": 0.2},
            StrategyType.DEFENSIVE: {"logical": 0.2, "evidence": 0.4, "relevance": 0.2, "assumption": 0.2},
            StrategyType.BALANCED: {"logical": 0.3, "evidence": 0.3, "relevance": 0.2, "assumption": 0.2},
            StrategyType.EVIDENCE_BASED: {"logical": 0.2, "evidence": 0.5, "relevance": 0.15, "assumption": 0.15},
            StrategyType.RHETORICAL: {"logical": 0.25, "evidence": 0.15, "relevance": 0.3, "assumption": 0.3},
        }
        
        type_weights = weights.get(self.profile.debate_style, weights[StrategyType.BALANCED])
        types = list(type_weights.keys())
        weights_list = list(type_weights.values())
        return random.choices(types, weights=weights_list)[0]
    
    def _generate_reason(self, target_argument: Argument, rebuttal_type: str) -> str:
        """生成反驳理由"""
        reasons = {
            "logical": [
                "前提与结论之间缺乏必然联系",
                "存在因果倒置的可能性",
                "推理链条存在断裂",
            ],
            "evidence": [
                "样本量不足以支撑结论",
                "数据来源的权威性不足",
                "存在选择性引用的问题",
            ],
            "relevance": [
                "讨论的是次要问题而非核心议题",
                "引入的案例与当前情境差异较大",
            ],
            "assumption": [
                "预设了所有人都有相同的条件",
                "忽略了情境的特殊性",
            ],
        }
        
        reason_list = reasons.get(rebuttal_type, reasons["logical"])
        return random.choice(reason_list)
    
    def _calculate_effectiveness(self, target_argument: Argument, rebuttal_type: str) -> float:
        """计算反驳有效性"""
        base_effectiveness = 0.5
        
        # 根据目标论点的证据强度调整
        evidence_strength = target_argument.get_evidence_strength()
        if rebuttal_type == "evidence" and evidence_strength < 0.5:
            base_effectiveness += 0.2
        
        # 根据参与者能力调整
        skill_bonus = self.profile.get_average_performance() * 0.2
        
        # 根据策略调整
        strategy_bonus = 0.1 if self.profile.debate_style == StrategyType.AGGRESSIVE else 0.0
        
        return min(0.95, base_effectiveness + skill_bonus + strategy_bonus)


class DebateParticipant:
    """
    辩论参与者Agent
    完整的辩论参与者实现
    """
    
    def __init__(self, profile: ParticipantProfile) -> None:
        self.profile = profile
        self.argument_generator = ArgumentGenerator(profile)
        self.rebuttal_generator = RebuttalGenerator(profile)
        self.participation_history: List[str] = []
        self.current_stance: Stance = Stance.NEUTRAL
        self.memory: Dict[str, Any] = {
            "arguments_made": [],
            "rebuttals_made": [],
            "topics_debated": set(),
        }
    
    def set_stance(self, stance: Stance) -> None:
        """设置当前立场"""
        self.current_stance = stance
    
    def make_argument(
        self,
        topic: str,
        debate_state: Optional[DebateState] = None
    ) -> Argument:
        """
        发表论点
        
        Args:
            topic: 辩论主题
            debate_state: 当前辩论状态
        """
        # 获取上下文
        context = None
        if debate_state:
            context = list(debate_state.arguments.values())
        
        argument = self.argument_generator.generate_argument(
            topic=topic,
            stance=self.current_stance,
            context=context
        )
        
        # 记录历史
        self.memory["arguments_made"].append(argument.argument_id)
        self.memory["topics_debated"].add(topic)
        self.participation_history.append(f"argument:{argument.argument_id}")
        
        return argument
    
    def rebut_argument(
        self,
        target_argument: Argument,
        debate_state: Optional[DebateState] = None
    ) -> Rebuttal:
        """
        反驳论点
        
        Args:
            target_argument: 要反驳的论点
            debate_state: 当前辩论状态
        """
        rebuttal = self.rebuttal_generator.generate_rebuttal(
            target_argument=target_argument,
            debate_context=debate_state
        )
        
        # 记录历史
        self.memory["rebuttals_made"].append(rebuttal.rebuttal_id)
        self.participation_history.append(f"rebuttal:{rebuttal.rebuttal_id}")
        
        return rebuttal
    
    def support_argument(
        self,
        target_argument: Argument,
        debate_state: Optional[DebateState] = None
    ) -> Argument:
        """
        支持论点（生成补充论证）
        
        Args:
            target_argument: 要支持的论点
            debate_state: 当前辩论状态
        """
        # 生成支持性论点
        support_argument = self.argument_generator.generate_argument(
            topic=target_argument.content[:50],
            stance=target_argument.stance,
            target_argument=target_argument
        )
        
        self.participation_history.append(f"support:{support_argument.argument_id}")
        
        return support_argument
    
    def revise_argument(
        self,
        original_argument: Argument,
        feedback: str
    ) -> Revision:
        """
        修正自己的论点
        
        Args:
            original_argument: 原始论点
            feedback: 反馈信息
        """
        # 生成修正内容
        revised_content = self._revise_content(
            original_argument.content,
            feedback
        )
        
        revision = Revision(
            original_argument_id=original_argument.argument_id,
            speaker_id=self.profile.participant_id,
            revised_content=revised_content,
            reason=f"基于反馈进行修正: {feedback[:50]}"
        )
        
        self.participation_history.append(f"revision:{revision.revision_id}")
        
        return revision
    
    def _revise_content(self, original: str, feedback: str) -> str:
        """根据反馈修正内容"""
        # 简单实现：添加澄清说明
        revisions = [
            f"{original} (澄清: 我的意思是更准确地表述)",
            f"修正后的观点：{original}，需要补充说明的是...",
            f"基于反馈，我重新表述：{original[:len(original)//2]}...",
        ]
        return random.choice(revisions)
    
    def evaluate_arguments(
        self,
        arguments: List[Argument]
    ) -> Dict[str, float]:
        """
        评估一组论点
        
        Args:
            arguments: 要评估的论点列表
            
        Returns:
            论点ID到评分的映射
        """
        scores = {}
        for arg in arguments:
            # 基于多个维度评分
            evidence_score = arg.get_evidence_strength()
            clarity_score = self._assess_clarity(arg.content)
            relevance_score = 0.7  # 默认相关度
            
            # 综合评分
            total_score = (evidence_score * 0.4 + clarity_score * 0.3 + relevance_score * 0.3)
            scores[arg.argument_id] = total_score
        
        return scores
    
    def _assess_clarity(self, content: str) -> float:
        """评估论点清晰度"""
        # 简单启发式：长度适中和结构清晰度
        length_score = 1.0 - abs(len(content) - 200) / 400
        length_score = max(0.3, min(1.0, length_score))
        
        # 检查是否有清晰的结构标记
        structure_markers = ["因为", "所以", "首先", "其次", "因此", "然而"]
        structure_score = sum(1 for m in structure_markers if m in content) / len(structure_markers)
        
        return (length_score + structure_score) / 2
    
    def get_stats(self) -> Dict[str, Any]:
        """获取参与者统计信息"""
        return {
            "participant_id": self.profile.participant_id,
            "arguments_made": len(self.memory["arguments_made"]),
            "rebuttals_made": len(self.memory["rebuttals_made"]),
            "topics_debated": len(self.memory["topics_debated"]),
            "participation_count": len(self.participation_history),
            "credibility": self.profile.credibility_score,
            "average_performance": self.profile.get_average_performance(),
        }


# 工厂函数
def create_participant(
    participant_id: str,
    name: str,
    participant_type: ParticipantType = ParticipantType.NEUTRAL,
    debate_style: StrategyType = StrategyType.BALANCED,
    expertise: Optional[Set[str]] = None
) -> DebateParticipant:
    """
    创建辩论参与者
    
    Args:
        participant_id: 参与者ID
        name: 参与者名称
        participant_type: 参与者类型
        debate_style: 辩论风格
        expertise: 专业领域
    """
    profile = ParticipantProfile(
        participant_id=participant_id,
        name=name,
        participant_type=participant_type,
        expertise_domains=expertise or set(),
        debate_style=debate_style,
        personality_traits={
            "openness": random.uniform(0.3, 0.9),
            "conscientiousness": random.uniform(0.3, 0.9),
            "extraversion": random.uniform(0.3, 0.9),
            "agreeableness": random.uniform(0.3, 0.9),
            "neuroticism": random.uniform(0.1, 0.6),
        }
    )
    
    return DebateParticipant(profile)


__all__ = [
    "ParticipantType",
    "StrategyType",
    "ParticipantProfile",
    "ArgumentGenerator",
    "RebuttalGenerator",
    "DebateParticipant",
    "create_participant",
]
