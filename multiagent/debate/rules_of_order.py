"""
议事规则模块
限定发言轮次、发言时间
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
import time

from .protocol import DebatePhase, DebateState, Argument, Rebuttal


class SpeakingRight(Enum):
    """发言权枚举"""
    ACTIVE = auto()        # 当前有发言权
    WAITING = auto()       # 等待中
    EXPIRED = auto()       # 已过期
    SUSPENDED = auto()     # 被暂停
    FORFEITED = auto()     # 已放弃


class RuleViolationType(Enum):
    """规则违反类型"""
    TIME_EXCEEDED = "time_exceeded"          # 超时
    INTERRUPTION = "interruption"            # 打断他人
    OFF_TOPIC = "off_topic"                  # 离题
    PERSONAL_ATTACK = "personal_attack"      # 人身攻击
    REPETITION = "repetition"                # 重复发言
    PROCEDURAL = "procedural"                # 程序违规


@dataclass
class SpeakingSlot:
    """发言时段"""
    slot_id: str
    participant_id: str
    phase: DebatePhase
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_duration: int = 180  # 最大持续时间（秒）
    status: SpeakingRight = SpeakingRight.WAITING
    content: Optional[str] = None
    
    def start(self) -> None:
        """开始发言"""
        self.start_time = datetime.now()
        self.status = SpeakingRight.ACTIVE
    
    def end(self) -> None:
        """结束发言"""
        self.end_time = datetime.now()
        self.status = SpeakingRight.EXPIRED
    
    def get_duration(self) -> float:
        """获取发言持续时间"""
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def is_expired(self) -> bool:
        """检查是否超时"""
        return self.get_duration() > self.max_duration
    
    def get_remaining_time(self) -> float:
        """获取剩余时间"""
        return max(0.0, self.max_duration - self.get_duration())


@dataclass
class SpeakingRecord:
    """发言记录"""
    participant_id: str
    slot_id: str
    phase: DebatePhase
    start_time: datetime
    end_time: datetime
    duration: float
    content_length: int
    violation_flags: List[RuleViolationType] = field(default_factory=list)


@dataclass
class RulesConfig:
    """议事规则配置"""
    # 发言时间限制
    max_speech_duration: int = 180           # 单次发言最大秒数
    min_speech_duration: int = 10            # 单次发言最小秒数
    
    # 轮次限制
    max_rounds: int = 5                      # 最大轮数
    max_speeches_per_round: int = 2          # 每轮每方最多发言次数
    
    # 发言顺序
    rotation_mode: str = "alternating"       # 轮换模式: alternating/sequential/free
    allow_interruptions: bool = False        # 是否允许打断
    require_acknowledgment: bool = True      # 是否需要确认
    
    # 冷却时间
    cooldown_period: int = 5                 # 发言间隔秒数
    
    # 违规处理
    max_violations: int = 3                  # 最大违规次数
    violation_penalty: str = "warning"       # 惩罚方式: warning/time_deduction/removal


class RulesOfOrder:
    """
    议事规则管理器
    管理辩论的发言顺序、时间和规则执行
    """
    
    def __init__(self, config: Optional[RulesConfig] = None) -> None:
        self.config = config or RulesConfig()
        self.speaking_queue: List[SpeakingSlot] = []
        self.speaking_history: List[SpeakingRecord] = []
        self.current_slot: Optional[SpeakingSlot] = None
        self.violation_counts: Dict[str, int] = defaultdict(int)
        self.speech_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.last_speech_time: Dict[str, datetime] = {}
        self.suspended_participants: Set[str] = set()
    
    def initialize_debate(
        self,
        participants: List[str],
        phases: List[DebatePhase]
    ) -> None:
        """
        初始化辩论
        
        Args:
            participants: 参与者ID列表
            phases: 辩论阶段列表
        """
        self.speaking_queue = []
        self.speaking_history = []
        
        # 根据轮换模式创建发言队列
        for phase in phases:
            if self.config.rotation_mode == "alternating":
                # 交替发言
                for round_num in range(self.config.max_rounds):
                    for participant in participants:
                        slot = SpeakingSlot(
                            slot_id=f"{phase.name}_{round_num}_{participant}",
                            participant_id=participant,
                            phase=phase,
                            max_duration=self.config.max_speech_duration
                        )
                        self.speaking_queue.append(slot)
            elif self.config.rotation_mode == "sequential":
                # 顺序发言
                for participant in participants:
                    for round_num in range(self.config.max_rounds):
                        slot = SpeakingSlot(
                            slot_id=f"{phase.name}_{round_num}_{participant}",
                            participant_id=participant,
                            phase=phase,
                            max_duration=self.config.max_speech_duration
                        )
                        self.speaking_queue.append(slot)
    
    def get_next_speaker(self, phase: DebatePhase) -> Optional[SpeakingSlot]:
        """
        获取下一个发言者
        
        Args:
            phase: 当前阶段
            
        Returns:
            发言时段或None
        """
        # 结束当前发言
        if self.current_slot:
            self.current_slot.end()
            self._record_speech(self.current_slot)
        
        # 查找下一个有效发言者
        for slot in self.speaking_queue:
            if (slot.phase == phase and 
                slot.status == SpeakingRight.WAITING and
                slot.participant_id not in self.suspended_participants):
                
                # 检查冷却时间
                if not self._check_cooldown(slot.participant_id):
                    continue
                
                # 检查发言次数
                if not self._check_speech_limit(slot.participant_id, phase):
                    continue
                
                self.current_slot = slot
                slot.start()
                return slot
        
        return None
    
    def _check_cooldown(self, participant_id: str) -> bool:
        """检查冷却时间"""
        if participant_id not in self.last_speech_time:
            return True
        
        elapsed = (datetime.now() - self.last_speech_time[participant_id]).total_seconds()
        return elapsed >= self.config.cooldown_period
    
    def _check_speech_limit(self, participant_id: str, phase: DebatePhase) -> bool:
        """检查发言次数限制"""
        count = self.speech_counts[participant_id][hash(phase)]
        return count < self.config.max_speeches_per_round
    
    def _record_speech(self, slot: SpeakingSlot) -> None:
        """记录发言"""
        record = SpeakingRecord(
            participant_id=slot.participant_id,
            slot_id=slot.slot_id,
            phase=slot.phase,
            start_time=slot.start_time or datetime.now(),
            end_time=slot.end_time or datetime.now(),
            duration=slot.get_duration(),
            content_length=len(slot.content) if slot.content else 0
        )
        self.speaking_history.append(record)
        self.last_speech_time[slot.participant_id] = datetime.now()
        self.speech_counts[slot.participant_id][hash(slot.phase)] += 1
    
    def check_time_limit(self) -> Tuple[bool, Optional[str]]:
        """
        检查时间限制
        
        Returns:
            (是否超时, 警告信息)
        """
        if not self.current_slot:
            return False, None
        
        remaining = self.current_slot.get_remaining_time()
        
        if remaining <= 0:
            return True, "发言时间已用完"
        elif remaining <= 10:
            return False, f"剩余时间：{remaining:.0f}秒"
        
        return False, None
    
    def register_violation(
        self,
        participant_id: str,
        violation_type: RuleViolationType,
        details: str = ""
    ) -> Dict[str, Any]:
        """
        注册违规行为
        
        Args:
            participant_id: 违规者ID
            violation_type: 违规类型
            details: 详细信息
            
        Returns:
            处理结果
        """
        self.violation_counts[participant_id] += 1
        count = self.violation_counts[participant_id]
        
        result = {
            "participant_id": participant_id,
            "violation_type": violation_type.value,
            "violation_count": count,
            "action_taken": "warning",
            "details": details,
        }
        
        # 根据违规次数采取不同措施
        if count >= self.config.max_violations:
            if self.config.violation_penalty == "removal":
                self.suspended_participants.add(participant_id)
                result["action_taken"] = "removal"
                result["message"] = f"参与者 {participant_id} 因多次违规被暂停发言"
            elif self.config.violation_penalty == "time_deduction":
                result["action_taken"] = "time_deduction"
                result["time_deducted"] = 30
                result["message"] = f"扣除 {participant_id} 30秒发言时间"
        else:
            result["message"] = f"警告：{participant_id} 违规 ({count}/{self.config.max_violations})"
        
        return result
    
    def validate_argument(
        self,
        argument: Argument,
        previous_arguments: List[Argument]
    ) -> Tuple[bool, List[RuleViolationType]]:
        """
        验证论点是否符合规则
        
        Args:
            argument: 要验证的论点
            previous_arguments: 之前的论点列表
            
        Returns:
            (是否有效, 违规类型列表)
        """
        violations = []
        
        # 检查重复
        if self._is_repetitive(argument, previous_arguments):
            violations.append(RuleViolationType.REPETITION)
        
        # 检查人身攻击（简化检测）
        if self._contains_personal_attack(argument.content):
            violations.append(RuleViolationType.PERSONAL_ATTACK)
        
        # 检查离题（简化检测）
        # 实际实现需要更复杂的主题相关性分析
        
        return len(violations) == 0, violations
    
    def _is_repetitive(
        self,
        argument: Argument,
        previous_arguments: List[Argument]
    ) -> bool:
        """检查是否重复"""
        current_content = argument.content.lower()
        
        for prev in previous_arguments[-3:]:  # 检查最近3个论点
            prev_content = prev.content.lower()
            # 简单的相似度检查
            similarity = self._calculate_similarity(current_content, prev_content)
            if similarity > 0.8:  # 80%相似度阈值
                return True
        
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _contains_personal_attack(self, content: str) -> bool:
        """检查是否包含人身攻击（简化检测）"""
        attack_keywords = [
            "愚蠢", "无知", "白痴", "笨蛋", "你这个人",
            "stupid", "idiot", "fool", "ignorant"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in attack_keywords)
    
    def get_speaking_stats(self, participant_id: str) -> Dict[str, Any]:
        """获取发言统计"""
        records = [r for r in self.speaking_history if r.participant_id == participant_id]
        
        if not records:
            return {
                "participant_id": participant_id,
                "total_speeches": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
            }
        
        total_duration = sum(r.duration for r in records)
        violations = sum(len(r.violation_flags) for r in records)
        
        return {
            "participant_id": participant_id,
            "total_speeches": len(records),
            "total_duration": total_duration,
            "average_duration": total_duration / len(records),
            "violations": violations,
            "violation_count": self.violation_counts[participant_id],
        }
    
    def get_phase_summary(self, phase: DebatePhase) -> Dict[str, Any]:
        """获取阶段总结"""
        phase_records = [r for r in self.speaking_history if r.phase == phase]
        
        participant_counts = defaultdict(int)
        for r in phase_records:
            participant_counts[r.participant_id] += 1
        
        return {
            "phase": phase.name,
            "total_speeches": len(phase_records),
            "total_duration": sum(r.duration for r in phase_records),
            "participant_distribution": dict(participant_counts),
            "average_speech_duration": (
                sum(r.duration for r in phase_records) / len(phase_records)
                if phase_records else 0
            ),
        }


class Moderator:
    """
    主持人
    执行议事规则，管理辩论流程
    """
    
    def __init__(self, moderator_id: str, rules: Optional[RulesOfOrder] = None) -> None:
        self.moderator_id = moderator_id
        self.rules = rules or RulesOfOrder()
        self.active_debates: Set[str] = set()
        self.notifications: List[Dict[str, Any]] = []
    
    def start_debate(
        self,
        debate_id: str,
        participants: List[str],
        phases: List[DebatePhase]
    ) -> None:
        """开始辩论"""
        self.rules.initialize_debate(participants, phases)
        self.active_debates.add(debate_id)
        self._notify_all(f"辩论 {debate_id} 开始，参与者：{', '.join(participants)}")
    
    def request_speaking_right(
        self,
        participant_id: str,
        phase: DebatePhase
    ) -> Tuple[bool, Optional[SpeakingSlot]]:
        """
        请求发言权
        
        Returns:
            (是否允许, 发言时段)
        """
        # 检查是否被暂停
        if participant_id in self.rules.suspended_participants:
            return False, None
        
        # 获取下一个发言者
        slot = self.rules.get_next_speaker(phase)
        
        if slot and slot.participant_id == participant_id:
            return True, slot
        
        return False, None
    
    def end_speech(self, slot_id: str) -> SpeakingRecord:
        """结束发言"""
        if self.rules.current_slot and self.rules.current_slot.slot_id == slot_id:
            self.rules.current_slot.end()
            self.rules._record_speech(self.rules.current_slot)
            
            # 创建记录副本
            slot = self.rules.current_slot
            record = SpeakingRecord(
                participant_id=slot.participant_id,
                slot_id=slot.slot_id,
                phase=slot.phase,
                start_time=slot.start_time or datetime.now(),
                end_time=datetime.now(),
                duration=slot.get_duration(),
                content_length=len(slot.content) if slot.content else 0
            )
            
            self.rules.current_slot = None
            return record
        
        raise ValueError(f"无效的发言时段ID: {slot_id}")
    
    def intervene(self, reason: str, action: str) -> Dict[str, Any]:
        """
        干预辩论
        
        Args:
            reason: 干预原因
            action: 干预动作
        """
        intervention = {
            "moderator_id": self.moderator_id,
            "timestamp": datetime.now(),
            "reason": reason,
            "action": action,
        }
        
        if action == "cut_off":
            # 切断当前发言
            if self.rules.current_slot:
                self.rules.current_slot.end()
                intervention["affected_participant"] = self.rules.current_slot.participant_id
        
        self.notifications.append(intervention)
        return intervention
    
    def _notify_all(self, message: str) -> None:
        """通知所有参与者"""
        self.notifications.append({
            "timestamp": datetime.now(),
            "message": message,
        })
    
    def get_moderator_report(self) -> Dict[str, Any]:
        """获取主持人报告"""
        return {
            "moderator_id": self.moderator_id,
            "active_debates": len(self.active_debates),
            "total_speeches_managed": len(self.rules.speaking_history),
            "violations_recorded": sum(self.rules.violation_counts.values()),
            "suspended_participants": list(self.rules.suspended_participants),
            "notifications": len(self.notifications),
        }


__all__ = [
    "SpeakingRight",
    "RuleViolationType",
    "SpeakingSlot",
    "SpeakingRecord",
    "RulesConfig",
    "RulesOfOrder",
    "Moderator",
]
