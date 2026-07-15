"""
罚没机制模块 - Slashing Mechanism

实现作恶或失败时的质押代币罚没机制，
维护网络安全和激励机制。
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time
import math


class SlashingReason(Enum):
    """罚没原因"""
    MALICIOUS_BEHAVIOR = "malicious_behavior"      # 恶意行为
    TASK_FAILURE = "task_failure"                  # 任务失败
    SLA_VIOLATION = "sla_violation"                # SLA违约
    DOWNTIME = "downtime"                          # 宕机
    MISCONDUCT = "misconduct"                      # 不当行为
    COLLUSION = "collusion"                        # 串通
    DOUBLE_SPENDING = "double_spending"           # 双花
    INVALID_RESPONSE = "invalid_response"         # 无效响应
    TIMEOUT = "timeout"                            # 超时
    DATA_TAMPERING = "data_tampering"             # 数据篡改


class SlashingSeverity(Enum):
    """罚没严重程度"""
    MINOR = "minor"          # 轻微: 1-5%
    MODERATE = "moderate"    # 中等: 5-20%
    MAJOR = "major"          # 严重: 20-50%
    CRITICAL = "critical"    # 关键: 50-100%
    MAXIMUM = "maximum"      # 最大: 100%


@dataclass
class SlashingEvent:
    """罚没事件"""
    event_id: str
    agent_id: str
    reason: SlashingReason
    severity: SlashingSeverity
    timestamp: float
    stake_ids: List[str] = field(default_factory=list)
    slash_percentage: float = 0.0
    slash_amount: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    reporter_id: Optional[str] = None
    appealable: bool = True
    appealed: bool = False
    appeal_result: Optional[bool] = None


@dataclass
class SlashingPolicy:
    """罚没策略"""
    reason: SlashingReason
    base_percentage: float       # 基础罚没比例
    max_percentage: float        # 最大罚没比例
    escalation_factor: float     # 递增因子（重复违规）
    cooldown_period: float       # 冷却期（秒）
    requires_evidence: bool      # 是否需要证据
    appeal_allowed: bool         # 是否允许申诉


@dataclass
class AgentSlashingRecord:
    """Agent罚没记录"""
    agent_id: str
    total_slashed: float = 0.0
    event_count: int = 0
    last_slash_time: float = 0.0
    reason_counts: Dict[SlashingReason, int] = field(default_factory=dict)
    is_blacklisted: bool = False
    blacklist_reason: Optional[str] = None


class SlashingManager:
    """
    罚没管理器
    
    实现完整的罚没机制:
    1. 定义罚没规则和比例
    2. 处理罚没事件
    3. 管理申诉流程
    4. 跟踪罚没历史
    """
    
    # 默认罚没策略
    DEFAULT_POLICIES: Dict[SlashingReason, SlashingPolicy] = {
        SlashingReason.MALICIOUS_BEHAVIOR: SlashingPolicy(
            reason=SlashingReason.MALICIOUS_BEHAVIOR,
            base_percentage=0.5,
            max_percentage=1.0,
            escalation_factor=2.0,
            cooldown_period=86400,
            requires_evidence=True,
            appeal_allowed=True
        ),
        SlashingReason.TASK_FAILURE: SlashingPolicy(
            reason=SlashingReason.TASK_FAILURE,
            base_percentage=0.05,
            max_percentage=0.3,
            escalation_factor=1.5,
            cooldown_period=3600,
            requires_evidence=False,
            appeal_allowed=True
        ),
        SlashingReason.SLA_VIOLATION: SlashingPolicy(
            reason=SlashingReason.SLA_VIOLATION,
            base_percentage=0.1,
            max_percentage=0.5,
            escalation_factor=1.3,
            cooldown_period=7200,
            requires_evidence=True,
            appeal_allowed=True
        ),
        SlashingReason.DOWNTIME: SlashingPolicy(
            reason=SlashingReason.DOWNTIME,
            base_percentage=0.02,
            max_percentage=0.2,
            escalation_factor=1.2,
            cooldown_period=1800,
            requires_evidence=False,
            appeal_allowed=True
        ),
        SlashingReason.MISCONDUCT: SlashingPolicy(
            reason=SlashingReason.MISCONDUCT,
            base_percentage=0.15,
            max_percentage=0.6,
            escalation_factor=1.8,
            cooldown_period=43200,
            requires_evidence=True,
            appeal_allowed=True
        ),
        SlashingReason.COLLUSION: SlashingPolicy(
            reason=SlashingReason.COLLUSION,
            base_percentage=0.4,
            max_percentage=1.0,
            escalation_factor=2.0,
            cooldown_period=172800,
            requires_evidence=True,
            appeal_allowed=False
        ),
        SlashingReason.DOUBLE_SPENDING: SlashingPolicy(
            reason=SlashingReason.DOUBLE_SPENDING,
            base_percentage=1.0,
            max_percentage=1.0,
            escalation_factor=1.0,
            cooldown_period=0,
            requires_evidence=True,
            appeal_allowed=False
        ),
        SlashingReason.INVALID_RESPONSE: SlashingPolicy(
            reason=SlashingReason.INVALID_RESPONSE,
            base_percentage=0.03,
            max_percentage=0.15,
            escalation_factor=1.2,
            cooldown_period=600,
            requires_evidence=False,
            appeal_allowed=True
        ),
        SlashingReason.TIMEOUT: SlashingPolicy(
            reason=SlashingReason.TIMEOUT,
            base_percentage=0.01,
            max_percentage=0.1,
            escalation_factor=1.1,
            cooldown_period=300,
            requires_evidence=False,
            appeal_allowed=True
        ),
        SlashingReason.DATA_TAMPERING: SlashingPolicy(
            reason=SlashingReason.DATA_TAMPERING,
            base_percentage=0.3,
            max_percentage=0.8,
            escalation_factor=2.0,
            cooldown_period=86400,
            requires_evidence=True,
            appeal_allowed=True
        ),
    }
    
    # 黑名单阈值
    BLACKLIST_THRESHOLD = 3  # 严重违规次数
    BLACKLIST_REASONS = {
        SlashingReason.MALICIOUS_BEHAVIOR,
        SlashingReason.COLLUSION,
        SlashingReason.DOUBLE_SPENDING,
        SlashingReason.DATA_TAMPERING
    }
    
    def __init__(
        self,
        policies: Optional[Dict[SlashingReason, SlashingPolicy]] = None,
        stake_manager: Any = None
    ):
        self.policies = policies or self.DEFAULT_POLICIES.copy()
        self.stake_manager = stake_manager
        
        # 罚没事件记录
        self.events: Dict[str, SlashingEvent] = {}
        
        # Agent罚没记录
        self.agent_records: Dict[str, AgentSlashingRecord] = {}
        
        # 事件计数器
        self._event_counter = 0
        
        # 回调函数
        self._slash_callbacks: List[Callable[[SlashingEvent], None]] = []
        self._blacklist_callbacks: List[Callable[[str, str], None]] = []
    
    def register_slash_callback(
        self, 
        callback: Callable[[SlashingEvent], None]
    ) -> None:
        """注册罚没回调"""
        self._slash_callbacks.append(callback)
    
    def register_blacklist_callback(
        self, 
        callback: Callable[[str, str], None]
    ) -> None:
        """注册黑名单回调"""
        self._blacklist_callbacks.append(callback)
    
    def _generate_event_id(self) -> str:
        """生成事件ID"""
        self._event_counter += 1
        return f"slash_{int(time.time())}_{self._event_counter}"
    
    def _get_or_create_record(self, agent_id: str) -> AgentSlashingRecord:
        """获取或创建Agent记录"""
        if agent_id not in self.agent_records:
            self.agent_records[agent_id] = AgentSlashingRecord(agent_id=agent_id)
        return self.agent_records[agent_id]
    
    def compute_slash_percentage(
        self,
        agent_id: str,
        reason: SlashingReason,
        severity: Optional[SlashingSeverity] = None
    ) -> float:
        """
        计算罚没比例
        
        考虑历史违规记录进行递增
        """
        policy = self.policies.get(reason)
        if policy is None:
            return 0.0
        
        record = self._get_or_create_record(agent_id)
        
        # 基础比例
        base = policy.base_percentage
        
        # 严重程度调整
        severity_multiplier = self._get_severity_multiplier(
            severity or SlashingSeverity.MODERATE
        )
        
        # 历史违规递增
        reason_count = record.reason_counts.get(reason, 0)
        escalation = policy.escalation_factor ** reason_count
        
        # 计算最终比例
        final_percentage = base * severity_multiplier * escalation
        
        # 限制在最大比例内
        return min(final_percentage, policy.max_percentage)
    
    def _get_severity_multiplier(self, severity: SlashingSeverity) -> float:
        """获取严重程度乘数"""
        multipliers = {
            SlashingSeverity.MINOR: 0.2,
            SlashingSeverity.MODERATE: 1.0,
            SlashingSeverity.MAJOR: 2.0,
            SlashingSeverity.CRITICAL: 3.0,
            SlashingSeverity.MAXIMUM: 5.0
        }
        return multipliers.get(severity, 1.0)
    
    def slash(
        self,
        agent_id: str,
        reason: SlashingReason,
        stake_ids: List[str],
        stake_amounts: Dict[str, float],
        severity: Optional[SlashingSeverity] = None,
        evidence: Optional[Dict[str, Any]] = None,
        reporter_id: Optional[str] = None
    ) -> Tuple[bool, SlashingEvent]:
        """
        执行罚没
        
        Args:
            agent_id: 被罚没的Agent
            reason: 罚没原因
            stake_ids: 质押ID列表
            stake_amounts: 质押金额映射
            severity: 严重程度
            evidence: 证据
            reporter_id: 举报者ID
            
        Returns:
            (是否成功, 罚没事件)
        """
        policy = self.policies.get(reason)
        if policy is None:
            return False, self._create_failed_event(agent_id, reason, "无对应罚没策略")
        
        # 检查是否需要证据
        if policy.requires_evidence and not evidence:
            return False, self._create_failed_event(agent_id, reason, "缺少必要证据")
        
        # 计算罚没比例
        slash_percentage = self.compute_slash_percentage(agent_id, reason, severity)
        
        # 计算总质押金额
        total_stake = sum(stake_amounts.get(sid, 0.0) for sid in stake_ids)
        
        # 计算罚没金额
        slash_amount = total_stake * slash_percentage
        
        # 创建罚没事件
        event = SlashingEvent(
            event_id=self._generate_event_id(),
            agent_id=agent_id,
            reason=reason,
            severity=severity or SlashingSeverity.MODERATE,
            timestamp=time.time(),
            stake_ids=stake_ids,
            slash_percentage=slash_percentage,
            slash_amount=slash_amount,
            evidence=evidence or {},
            reporter_id=reporter_id,
            appealable=policy.appeal_allowed
        )
        
        # 执行质押罚没
        if self.stake_manager is not None:
            for stake_id in stake_ids:
                stake_amount = stake_amounts.get(stake_id, 0.0)
                if stake_amount > 0:
                    # 按比例罚没每个质押
                    individual_percentage = slash_amount / total_stake if total_stake > 0 else 0
                    try:
                        self.stake_manager.slash_stake(stake_id, individual_percentage)
                    except Exception:
                        pass  # 质押罚没失败不阻止记录
        
        # 更新Agent记录
        record = self._get_or_create_record(agent_id)
        record.total_slashed += slash_amount
        record.event_count += 1
        record.last_slash_time = event.timestamp
        record.reason_counts[reason] = record.reason_counts.get(reason, 0) + 1
        
        # 检查是否应该加入黑名单
        self._check_blacklist(agent_id, reason, record)
        
        # 保存事件
        self.events[event.event_id] = event
        
        # 触发回调
        for callback in self._slash_callbacks:
            try:
                callback(event)
            except Exception:
                pass
        
        return True, event
    
    def _create_failed_event(
        self, 
        agent_id: str, 
        reason: SlashingReason,
        error_msg: str
    ) -> SlashingEvent:
        """创建失败事件"""
        return SlashingEvent(
            event_id=self._generate_event_id(),
            agent_id=agent_id,
            reason=reason,
            severity=SlashingSeverity.MINOR,
            timestamp=time.time(),
            evidence={'error': error_msg}
        )
    
    def _check_blacklist(
        self,
        agent_id: str,
        reason: SlashingReason,
        record: AgentSlashingRecord
    ) -> None:
        """检查是否应该加入黑名单"""
        if record.is_blacklisted:
            return
        
        # 计算严重违规次数
        severe_violations = sum(
            record.reason_counts.get(r, 0)
            for r in self.BLACKLIST_REASONS
        )
        
        if severe_violations >= self.BLACKLIST_THRESHOLD:
            record.is_blacklisted = True
            record.blacklist_reason = f"累计{severe_violations}次严重违规"
            
            # 触发黑名单回调
            for callback in self._blacklist_callbacks:
                try:
                    callback(agent_id, record.blacklist_reason)
                except Exception:
                    pass
    
    def appeal(
        self,
        event_id: str,
        appeal_reason: str,
        new_evidence: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        申诉罚没
        
        Args:
            event_id: 罚没事件ID
            appeal_reason: 申诉理由
            new_evidence: 新证据
            
        Returns:
            (是否接受申诉, 消息)
        """
        if event_id not in self.events:
            return False, "罚没事件不存在"
        
        event = self.events[event_id]
        
        if not event.appealable:
            return False, "该罚没事件不允许申诉"
        
        if event.appealed:
            return False, "该罚没事件已申诉"
        
        # 标记为已申诉
        event.appealed = True
        
        # 简化的申诉判断逻辑
        # 实际应该有人工审核或更复杂的判断
        appeal_accepted = False
        
        # 检查新证据
        if new_evidence:
            # 如果有强有力的新证据，可能接受申诉
            if new_evidence.get('exculpatory', False):
                appeal_accepted = True
        
        # 检查申诉理由
        if not appeal_accepted and appeal_reason:
            # 如果是首次违规且理由充分，可能减轻
            record = self.agent_records.get(event.agent_id)
            if record and record.event_count == 1:
                if 'mistake' in appeal_reason.lower() or 'error' in appeal_reason.lower():
                    appeal_accepted = True
        
        event.appeal_result = appeal_accepted
        
        if appeal_accepted:
            # 退还罚没金额（如果有质押管理器）
            if self.stake_manager:
                # 这里简化处理，实际需要更复杂的退还逻辑
                pass
            
            return True, "申诉成功，罚没已撤销"
        else:
            return False, "申诉被拒绝"
    
    def get_agent_slashing_history(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """获取Agent罚没历史"""
        record = self.agent_records.get(agent_id)
        
        if record is None:
            return {
                'agent_id': agent_id,
                'total_slashed': 0.0,
                'event_count': 0,
                'is_blacklisted': False,
                'events': []
            }
        
        # 获取该Agent的所有罚没事件
        agent_events = [
            e for e in self.events.values()
            if e.agent_id == agent_id
        ]
        agent_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return {
            'agent_id': agent_id,
            'total_slashed': round(record.total_slashed, 4),
            'event_count': record.event_count,
            'last_slash_time': record.last_slash_time,
            'is_blacklisted': record.is_blacklisted,
            'blacklist_reason': record.blacklist_reason,
            'reason_breakdown': {
                reason.value: count
                for reason, count in record.reason_counts.items()
            },
            'events': [
                {
                    'event_id': e.event_id,
                    'reason': e.reason.value,
                    'severity': e.severity.value,
                    'timestamp': e.timestamp,
                    'slash_percentage': round(e.slash_percentage, 4),
                    'slash_amount': round(e.slash_amount, 4),
                    'appealed': e.appealed,
                    'appeal_result': e.appeal_result
                }
                for e in agent_events
            ]
        }
    
    def get_slashing_statistics(self) -> Dict[str, Any]:
        """获取罚没统计"""
        total_slashed = sum(
            record.total_slashed
            for record in self.agent_records.values()
        )
        
        reason_counts = defaultdict(int)
        reason_amounts = defaultdict(float)
        
        for event in self.events.values():
            reason_counts[event.reason] += 1
            reason_amounts[event.reason] += event.slash_amount
        
        blacklisted_count = sum(
            1 for record in self.agent_records.values()
            if record.is_blacklisted
        )
        
        appealed_count = sum(
            1 for event in self.events.values()
            if event.appealed
        )
        
        successful_appeals = sum(
            1 for event in self.events.values()
            if event.appeal_result is True
        )
        
        return {
            'total_events': len(self.events),
            'total_agents_slashed': len(self.agent_records),
            'total_amount_slashed': round(total_slashed, 4),
            'blacklisted_agents': blacklisted_count,
            'appeal_statistics': {
                'total_appeals': appealed_count,
                'successful_appeals': successful_appeals,
                'success_rate': successful_appeals / appealed_count if appealed_count > 0 else 0.0
            },
            'by_reason': {
                reason.value: {
                    'count': count,
                    'total_amount': round(reason_amounts[reason], 4)
                }
                for reason, count in reason_counts.items()
            }
        }
    
    def is_blacklisted(self, agent_id: str) -> bool:
        """检查Agent是否在黑名单"""
        record = self.agent_records.get(agent_id)
        return record.is_blacklisted if record else False
    
    def get_blacklist(self) -> List[Dict[str, Any]]:
        """获取黑名单"""
        return [
            {
                'agent_id': record.agent_id,
                'reason': record.blacklist_reason,
                'total_slashed': round(record.total_slashed, 4),
                'event_count': record.event_count,
                'last_slash_time': record.last_slash_time
            }
            for record in self.agent_records.values()
            if record.is_blacklisted
        ]
    
    def remove_from_blacklist(
        self,
        agent_id: str,
        reason: str
    ) -> bool:
        """
        从黑名单移除
        
        需要管理员权限和充分理由
        """
        record = self.agent_records.get(agent_id)
        
        if record is None or not record.is_blacklisted:
            return False
        
        record.is_blacklisted = False
        record.blacklist_reason = None
        
        return True
    
    def calculate_reputation_penalty(
        self,
        agent_id: str
    ) -> float:
        """
        计算信誉惩罚
        
        基于罚没历史计算信誉分惩罚
        """
        record = self.agent_records.get(agent_id)
        
        if record is None or record.event_count == 0:
            return 0.0
        
        # 基础惩罚
        base_penalty = min(50.0, record.event_count * 5.0)
        
        # 严重违规额外惩罚
        severe_penalty = 0.0
        for reason in self.BLACKLIST_REASONS:
            count = record.reason_counts.get(reason, 0)
            severe_penalty += count * 10.0
        
        # 时间衰减
        if record.last_slash_time > 0:
            days_since_last = (time.time() - record.last_slash_time) / 86400
            decay = math.exp(-days_since_last / 30)  # 30天半衰期
        else:
            decay = 1.0
        
        total_penalty = (base_penalty + severe_penalty) * decay
        
        return min(100.0, total_penalty)


class ProgressiveSlashingManager(SlashingManager):
    """
    渐进式罚没管理器
    
    实现更温和的渐进式罚没策略，
    给予初犯者更多宽容
    """
    
    def __init__(
        self,
        policies: Optional[Dict[SlashingReason, SlashingPolicy]] = None,
        stake_manager: Any = None,
        grace_period: float = 86400,  # 宽限期（秒）
        warning_threshold: int = 2     # 警告阈值
    ):
        super().__init__(policies, stake_manager)
        self.grace_period = grace_period
        self.warning_threshold = warning_threshold
        self.warnings: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def issue_warning(
        self,
        agent_id: str,
        reason: SlashingReason,
        message: str
    ) -> Dict[str, Any]:
        """发出警告"""
        warning = {
            'timestamp': time.time(),
            'reason': reason,
            'message': message
        }
        self.warnings[agent_id].append(warning)
        
        return {
            'agent_id': agent_id,
            'warning_count': len(self.warnings[agent_id]),
            'should_slash': len(self.warnings[agent_id]) >= self.warning_threshold
        }
    
    def compute_slash_percentage(
        self,
        agent_id: str,
        reason: SlashingReason,
        severity: Optional[SlashingSeverity] = None
    ) -> float:
        """计算渐进式罚没比例"""
        record = self._get_or_create_record(agent_id)
        
        # 检查是否在宽限期内且为初犯
        if record.event_count == 0:
            # 初犯：检查警告次数
            warning_count = len(self.warnings.get(agent_id, []))
            if warning_count < self.warning_threshold:
                # 还在警告阶段，不罚没
                return 0.0
            elif warning_count == self.warning_threshold:
                # 刚达到阈值，轻微罚没
                return super().compute_slash_percentage(
                    agent_id, reason, SlashingSeverity.MINOR
                ) * 0.5
        
        # 非初犯：正常计算
        return super().compute_slash_percentage(agent_id, reason, severity)
    
    def slash(
        self,
        agent_id: str,
        reason: SlashingReason,
        stake_ids: List[str],
        stake_amounts: Dict[str, float],
        severity: Optional[SlashingSeverity] = None,
        evidence: Optional[Dict[str, Any]] = None,
        reporter_id: Optional[str] = None
    ) -> Tuple[bool, SlashingEvent]:
        """渐进式罚没"""
        record = self._get_or_create_record(agent_id)
        
        # 检查是否应该先发警告
        if record.event_count == 0 and severity != SlashingSeverity.CRITICAL:
            warning_count = len(self.warnings.get(agent_id, []))
            
            if warning_count < self.warning_threshold:
                # 发出警告而不是直接罚没
                warning = self.issue_warning(
                    agent_id, reason,
                    f"检测到{reason.value}行为，请立即纠正"
                )
                
                # 返回一个特殊的"警告"事件
                event = SlashingEvent(
                    event_id=self._generate_event_id(),
                    agent_id=agent_id,
                    reason=reason,
                    severity=SlashingSeverity.MINOR,
                    timestamp=time.time(),
                    evidence={'type': 'warning', 'warning_count': warning_count + 1}
                )
                return True, event
        
        # 执行实际罚没
        return super().slash(
            agent_id, reason, stake_ids, stake_amounts,
            severity, evidence, reporter_id
        )
