"""
标准辩论协议模块
定义主张、反驳、修正、裁决各阶段的数据结构和协议规范
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4


class DebatePhase(Enum):
    """辩论阶段枚举"""
    CLAIM = auto()          # 主张阶段
    REBUTTAL = auto()       # 反驳阶段
    COUNTER_CLAIM = auto()  # 反主张阶段
    EVIDENCE = auto()       # 证据提交阶段
    CROSS_EXAMINATION = auto()  # 交叉质询阶段
    REVISION = auto()       # 修正阶段
    DELIBERATION = auto()   # 审议阶段
    VERDICT = auto()        # 裁决阶段
    CONSENSUS = auto()      # 共识阶段


class ArgumentType(Enum):
    """论点类型枚举"""
    FACTUAL = auto()        # 事实性论点
    NORMATIVE = auto()      # 规范性论点
    CAUSAL = auto()         # 因果性论点
    ANALOGICAL = auto()     # 类比性论点
    AUTHORITY = auto()      # 权威性论点
    PRAGMATIC = auto()      # 实用性论点


class Stance(Enum):
    """立场枚举"""
    PRO = auto()            # 支持
    CON = auto()            # 反对
    NEUTRAL = auto()        # 中立
    UNDECIDED = auto()      # 未决定


@dataclass
class Evidence:
    """证据数据结构"""
    evidence_id: str = field(default_factory=lambda: str(uuid4())[:8])
    source: str = ""
    content: str = ""
    credibility: float = 0.5  # 0-1可信度评分
    relevance: float = 0.5    # 0-1相关度评分
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def quality_score(self) -> float:
        """计算证据质量分数"""
        return (self.credibility * 0.6 + self.relevance * 0.4)


@dataclass
class Argument:
    """论点数据结构"""
    argument_id: str = field(default_factory=lambda: str(uuid4())[:8])
    speaker_id: str = ""
    content: str = ""
    argument_type: ArgumentType = ArgumentType.FACTUAL
    stance: Stance = Stance.NEUTRAL
    target_argument_id: Optional[str] = None  # 针对的论点ID
    evidence_list: List[Evidence] = field(default_factory=list)
    phase: DebatePhase = DebatePhase.CLAIM
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5  # 发言者对此论点的置信度
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_evidence(self, evidence: Evidence) -> None:
        """添加证据"""
        self.evidence_list.append(evidence)
    
    def get_evidence_strength(self) -> float:
        """获取证据总强度"""
        if not self.evidence_list:
            return 0.0
        return sum(e.quality_score() for e in self.evidence_list) / len(self.evidence_list)


@dataclass
class Rebuttal:
    """反驳数据结构"""
    rebuttal_id: str = field(default_factory=lambda: str(uuid4())[:8])
    speaker_id: str = ""
    target_argument_id: str = ""
    content: str = ""
    rebuttal_type: str = ""  # 反驳类型：逻辑/证据/相关性
    timestamp: datetime = field(default_factory=datetime.now)
    fallacies_detected: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.0


@dataclass
class Revision:
    """修正数据结构"""
    revision_id: str = field(default_factory=lambda: str(uuid4())[:8])
    original_argument_id: str = ""
    speaker_id: str = ""
    revised_content: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    accepted: bool = False


@dataclass
class Verdict:
    """裁决数据结构"""
    verdict_id: str = field(default_factory=lambda: str(uuid4())[:8])
    arbitrator_id: str = ""
    topic: str = ""
    winning_stance: Optional[Stance] = None
    reasoning: str = ""
    confidence: float = 0.0
    argument_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DebateState:
    """辩论状态数据结构"""
    debate_id: str = field(default_factory=lambda: str(uuid4())[:8])
    topic: str = ""
    current_phase: DebatePhase = DebatePhase.CLAIM
    phase_history: List[Tuple[DebatePhase, datetime]] = field(default_factory=list)
    arguments: Dict[str, Argument] = field(default_factory=dict)
    rebuttals: Dict[str, Rebuttal] = field(default_factory=dict)
    revisions: Dict[str, Revision] = field(default_factory=dict)
    participants: Set[str] = field(default_factory=set)
    round_number: int = 0
    max_rounds: int = 5
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    verdict: Optional[Verdict] = None
    consensus_reached: bool = False
    consensus_threshold: float = 0.75
    
    def transition_to(self, new_phase: DebatePhase) -> None:
        """转换到新的辩论阶段"""
        self.phase_history.append((self.current_phase, datetime.now()))
        self.current_phase = new_phase
        if new_phase == DebatePhase.VERDICT:
            self.end_time = datetime.now()
    
    def add_argument(self, argument: Argument) -> None:
        """添加论点"""
        self.arguments[argument.argument_id] = argument
        self.participants.add(argument.speaker_id)
    
    def add_rebuttal(self, rebuttal: Rebuttal) -> None:
        """添加反驳"""
        self.rebuttals[rebuttal.rebuttal_id] = rebuttal
        self.participants.add(rebuttal.speaker_id)
    
    def get_arguments_by_stance(self, stance: Stance) -> List[Argument]:
        """获取特定立场的所有论点"""
        return [arg for arg in self.arguments.values() if arg.stance == stance]
    
    def get_rebuttals_for_argument(self, argument_id: str) -> List[Rebuttal]:
        """获取针对特定论点的所有反驳"""
        return [reb for reb in self.rebuttals.values() if reb.target_argument_id == argument_id]


class DebateProtocol:
    """
    辩论协议管理器
    管理辩论流程和阶段转换规则
    """
    
    # 定义有效的阶段转换
    VALID_TRANSITIONS: Dict[DebatePhase, Set[DebatePhase]] = {
        DebatePhase.CLAIM: {DebatePhase.REBUTTAL, DebatePhase.EVIDENCE},
        DebatePhase.REBUTTAL: {DebatePhase.COUNTER_CLAIM, DebatePhase.CROSS_EXAMINATION},
        DebatePhase.COUNTER_CLAIM: {DebatePhase.EVIDENCE, DebatePhase.REBUTTAL},
        DebatePhase.EVIDENCE: {DebatePhase.CROSS_EXAMINATION, DebatePhase.REVISION},
        DebatePhase.CROSS_EXAMINATION: {DebatePhase.REVISION, DebatePhase.DELIBERATION},
        DebatePhase.REVISION: {DebatePhase.DELIBERATION, DebatePhase.REBUTTAL},
        DebatePhase.DELIBERATION: {DebatePhase.VERDICT, DebatePhase.CONSENSUS},
        DebatePhase.VERDICT: set(),  # 终态
        DebatePhase.CONSENSUS: set(),  # 终态
    }
    
    def __init__(self, debate_id: Optional[str] = None) -> None:
        self.state = DebateState(debate_id=debate_id or str(uuid4())[:8])
        self.protocol_rules: Dict[str, Any] = {
            "min_arguments_per_phase": 1,
            "max_speech_length": 1000,  # 字符数
            "require_evidence": True,
            "allow_self_correction": True,
        }
    
    def can_transition(self, from_phase: DebatePhase, to_phase: DebatePhase) -> bool:
        """检查阶段转换是否有效"""
        return to_phase in self.VALID_TRANSITIONS.get(from_phase, set())
    
    def transition(self, new_phase: DebatePhase) -> bool:
        """
        尝试转换到新的阶段
        返回是否成功
        """
        if self.can_transition(self.state.current_phase, new_phase):
            self.state.transition_to(new_phase)
            return True
        return False
    
    def create_argument(
        self,
        speaker_id: str,
        content: str,
        stance: Stance,
        argument_type: ArgumentType = ArgumentType.FACTUAL,
        target_argument_id: Optional[str] = None
    ) -> Argument:
        """创建新论点"""
        return Argument(
            speaker_id=speaker_id,
            content=content,
            stance=stance,
            argument_type=argument_type,
            target_argument_id=target_argument_id,
            phase=self.state.current_phase
        )
    
    def create_rebuttal(
        self,
        speaker_id: str,
        target_argument_id: str,
        content: str,
        rebuttal_type: str = "general"
    ) -> Rebuttal:
        """创建新反驳"""
        return Rebuttal(
            speaker_id=speaker_id,
            target_argument_id=target_argument_id,
            content=content,
            rebuttal_type=rebuttal_type
        )
    
    def create_revision(
        self,
        speaker_id: str,
        original_argument_id: str,
        revised_content: str,
        reason: str
    ) -> Revision:
        """创建修正"""
        return Revision(
            speaker_id=speaker_id,
            original_argument_id=original_argument_id,
            revised_content=revised_content,
            reason=reason
        )
    
    def create_verdict(
        self,
        arbitrator_id: str,
        winning_stance: Optional[Stance],
        reasoning: str,
        confidence: float
    ) -> Verdict:
        """创建裁决"""
        return Verdict(
            arbitrator_id=arbitrator_id,
            topic=self.state.topic,
            winning_stance=winning_stance,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def set_topic(self, topic: str) -> None:
        """设置辩论主题"""
        self.state.topic = topic
    
    def get_phase_requirements(self, phase: DebatePhase) -> Dict[str, Any]:
        """获取特定阶段的要求"""
        requirements = {
            DebatePhase.CLAIM: {"min_participants": 2, "requires_stance": True},
            DebatePhase.REBUTTAL: {"min_arguments": 1, "requires_target": True},
            DebatePhase.EVIDENCE: {"requires_source": True},
            DebatePhase.CROSS_EXAMINATION: {"max_questions": 3},
            DebatePhase.DELIBERATION: {"min_consideration_time": 60},
            DebatePhase.VERDICT: {"requires_reasoning": True},
        }
        return requirements.get(phase, {})


# 协议验证器
class ProtocolValidator:
    """辩论协议验证器"""
    
    @staticmethod
    def validate_argument(argument: Argument, phase: DebatePhase) -> Tuple[bool, List[str]]:
        """验证论点是否符合阶段要求"""
        errors = []
        
        if not argument.content or len(argument.content.strip()) < 10:
            errors.append("论点内容过短")
        
        if phase == DebatePhase.EVIDENCE and not argument.evidence_list:
            errors.append("证据阶段需要提供证据")
        
        if phase == DebatePhase.REBUTTAL and not argument.target_argument_id:
            errors.append("反驳需要指定目标论点")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_evidence(evidence: Evidence) -> Tuple[bool, List[str]]:
        """验证证据有效性"""
        errors = []
        
        if not evidence.source:
            errors.append("证据需要来源")
        
        if not evidence.content:
            errors.append("证据内容不能为空")
        
        if evidence.credibility < 0 or evidence.credibility > 1:
            errors.append("可信度必须在0-1之间")
        
        return len(errors) == 0, errors


# 导出主要类
__all__ = [
    "DebatePhase",
    "ArgumentType",
    "Stance",
    "Evidence",
    "Argument",
    "Rebuttal",
    "Revision",
    "Verdict",
    "DebateState",
    "DebateProtocol",
    "ProtocolValidator",
]
