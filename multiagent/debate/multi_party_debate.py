"""
多方辩论主持模块
支持2v2、多方自由辩论
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum, auto
import random
from collections import defaultdict

from .protocol import (
    DebateProtocol, DebateState, DebatePhase, Argument, Rebuttal,
    Stance, Verdict
)
from .participant import DebateParticipant, ParticipantType
from .arbitrator import Arbitrator
from .rules_of_order import RulesOfOrder, Moderator
from .consensus_reach import ConsensusDetector


class DebateFormat(Enum):
    """辩论形式枚举"""
    ONE_ON_ONE = "one_on_one"              # 一对一
    TWO_VS_TWO = "two_vs_two"              # 二对二
    FREE_FOR_ALL = "free_for_all"          # 多方自由辩论
    PANEL_DISCUSSION = "panel_discussion"  # 小组讨论
    LINCOLN_DOUGLAS = "lincoln_douglas"    # 林肯-道格拉斯式


class TeamAlignment(Enum):
    """队伍对齐方式"""
    FIXED = "fixed"                        # 固定队伍
    DYNAMIC = "dynamic"                    # 动态结盟
    INDIVIDUAL = "individual"              # 个人战


@dataclass
class Team:
    """辩论队伍"""
    team_id: str
    name: str
    members: Set[str] = field(default_factory=set)
    stance: Stance = Stance.NEUTRAL
    score: float = 0.0
    arguments_made: int = 0
    rebuttals_made: int = 0
    
    def add_member(self, participant_id: str) -> None:
        """添加成员"""
        self.members.add(participant_id)
    
    def remove_member(self, participant_id: str) -> None:
        """移除成员"""
        self.members.discard(participant_id)


@dataclass
class DebateConfiguration:
    """辩论配置"""
    format: DebateFormat = DebateFormat.FREE_FOR_ALL
    team_alignment: TeamAlignment = TeamAlignment.INDIVIDUAL
    num_rounds: int = 3
    time_limit_per_speech: int = 180
    allow_switching_sides: bool = False
    require_cross_examination: bool = False
    scoring_method: str = "weighted"       # weighted/cumulative/average


class MultiPartyDebate:
    """
    多方辩论主持人
    管理多方参与的复杂辩论
    """
    
    def __init__(
        self,
        debate_id: str,
        topic: str,
        config: Optional[DebateConfiguration] = None
    ) -> None:
        self.debate_id = debate_id
        self.topic = topic
        self.config = config or DebateConfiguration()
        
        self.protocol = DebateProtocol(debate_id)
        self.protocol.set_topic(topic)
        
        self.participants: Dict[str, DebateParticipant] = {}
        self.teams: Dict[str, Team] = {}
        self.arbitrator: Optional[Arbitrator] = None
        self.rules = RulesOfOrder()
        self.consensus_detector = ConsensusDetector()
        
        self.current_round = 0
        self.current_speaker_index = 0
        self.speaking_order: List[str] = []
        self.is_active = False
        self.result: Optional[Verdict] = None
        
        self.history: List[Dict[str, Any]] = []
    
    def add_participant(
        self,
        participant: DebateParticipant,
        team_id: Optional[str] = None
    ) -> None:
        """添加参与者"""
        self.participants[participant.profile.participant_id] = participant
        
        if team_id and team_id in self.teams:
            self.teams[team_id].add_member(participant.profile.participant_id)
        
        self.protocol.state.participants.add(participant.profile.participant_id)
    
    def create_team(
        self,
        team_id: str,
        name: str,
        stance: Stance = Stance.NEUTRAL
    ) -> Team:
        """创建队伍"""
        team = Team(team_id=team_id, name=name, stance=stance)
        self.teams[team_id] = team
        return team
    
    def set_arbitrator(self, arbitrator: Arbitrator) -> None:
        """设置仲裁者"""
        self.arbitrator = arbitrator
    
    def start(self) -> None:
        """开始辩论"""
        self.is_active = True
        self.current_round = 1
        self.speaking_order = list(self.participants.keys())
        random.shuffle(self.speaking_order)
        
        self.history.append({
            "event": "debate_started",
            "round": self.current_round,
            "participants": list(self.participants.keys()),
        })
    
    def get_current_speaker(self) -> Optional[DebateParticipant]:
        """获取当前发言者"""
        if not self.speaking_order:
            return None
        
        speaker_id = self.speaking_order[self.current_speaker_index]
        return self.participants.get(speaker_id)
    
    def advance_speaker(self) -> Optional[DebateParticipant]:
        """推进到下一个发言者"""
        if not self.speaking_order:
            return None
        
        self.current_speaker_index = (self.current_speaker_index + 1) % len(self.speaking_order)
        
        # 如果回到第一个发言者，增加轮数
        if self.current_speaker_index == 0:
            self.current_round += 1
            
            if self.current_round > self.config.num_rounds:
                self.end()
                return None
        
        return self.get_current_speaker()
    
    def submit_argument(
        self,
        participant_id: str,
        content: str,
        stance: Stance
    ) -> Optional[Argument]:
        """提交论点"""
        if not self.is_active:
            return None
        
        participant = self.participants.get(participant_id)
        if not participant:
            return None
        
        argument = participant.make_argument(
            topic=self.topic,
            debate_state=self.protocol.state
        )
        
        # 覆盖内容和立场
        argument.content = content
        argument.stance = stance
        
        self.protocol.state.add_argument(argument)
        
        self.history.append({
            "event": "argument_submitted",
            "participant_id": participant_id,
            "argument_id": argument.argument_id,
            "stance": stance.name,
        })
        
        return argument
    
    def submit_rebuttal(
        self,
        participant_id: str,
        target_argument_id: str,
        content: str
    ) -> Optional[Rebuttal]:
        """提交反驳"""
        if not self.is_active:
            return None
        
        participant = self.participants.get(participant_id)
        target_argument = self.protocol.state.arguments.get(target_argument_id)
        
        if not participant or not target_argument:
            return None
        
        rebuttal = participant.rebut_argument(
            target_argument=target_argument,
            debate_state=self.protocol.state
        )
        
        # 覆盖内容
        rebuttal.content = content
        
        self.protocol.state.add_rebuttal(rebuttal)
        
        self.history.append({
            "event": "rebuttal_submitted",
            "participant_id": participant_id,
            "rebuttal_id": rebuttal.rebuttal_id,
            "target_argument_id": target_argument_id,
        })
        
        return rebuttal
    
    def check_consensus(self) -> Tuple[bool, float]:
        """检查是否达成共识"""
        consensus_state = self.consensus_detector.analyze_debate_state(
            self.protocol.state
        )
        
        return (
            consensus_state.is_consensus_reached,
            consensus_state.current_consensus_level
        )
    
    def end(self) -> Verdict:
        """结束辩论"""
        self.is_active = False
        
        # 如果有仲裁者，进行裁决
        if self.arbitrator:
            self.result = self.arbitrator.arbitrate(self.protocol.state)
        else:
            # 创建默认裁决
            self.result = Verdict(
                arbitrator_id="system",
                topic=self.topic,
                winning_stance=None,
                reasoning="辩论结束，无仲裁者",
                confidence=0.0
            )
        
        self.history.append({
            "event": "debate_ended",
            "result": self.result.winning_stance.name if self.result.winning_stance else None,
        })
        
        return self.result
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "is_active": self.is_active,
            "current_round": self.current_round,
            "total_rounds": self.config.num_rounds,
            "participants": len(self.participants),
            "teams": len(self.teams),
            "arguments": len(self.protocol.state.arguments),
            "rebuttals": len(self.protocol.state.rebuttals),
            "current_phase": self.protocol.state.current_phase.name,
        }
    
    def get_team_scores(self) -> Dict[str, float]:
        """获取队伍分数"""
        return {team_id: team.score for team_id, team in self.teams.items()}
    
    def get_participant_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取参与者统计"""
        stats = {}
        for pid, participant in self.participants.items():
            stats[pid] = participant.get_stats()
        return stats


__all__ = [
    "DebateFormat",
    "TeamAlignment",
    "Team",
    "DebateConfiguration",
    "MultiPartyDebate",
]
