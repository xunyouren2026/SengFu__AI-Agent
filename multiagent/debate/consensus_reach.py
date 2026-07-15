"""
共识达成检测模块
当意见分歧缩小时终止辩论
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import math
import statistics
from collections import defaultdict

from .protocol import DebateState, Argument, Stance, DebatePhase


class ConsensusMetric(Enum):
    """共识度量指标枚举"""
    STANCE_ALIGNMENT = "stance_alignment"          # 立场对齐度
    OPINION_CONVERGENCE = "opinion_convergence"    # 意见收敛度
    AGREEMENT_RATIO = "agreement_ratio"            # 同意比例
    VARIANCE_REDUCTION = "variance_reduction"      # 方差缩减
    ENTROPY_DECREASE = "entropy_decrease"          # 熵减


@dataclass
class ConsensusState:
    """共识状态数据结构"""
    debate_id: str
    current_consensus_level: float = 0.0           # 当前共识水平
    consensus_history: List[Tuple[int, float]] = field(default_factory=list)
    stance_distribution: Dict[Stance, float] = field(default_factory=dict)
    opinion_vectors: Dict[str, List[float]] = field(default_factory=dict)
    convergence_rate: float = 0.0                  # 收敛速率
    is_consensus_reached: bool = False
    consensus_stance: Optional[Stance] = None
    confidence: float = 0.0


@dataclass
class ConsensusConfig:
    """共识检测配置"""
    threshold: float = 0.75                        # 共识阈值
    min_rounds: int = 2                            # 最小辩论轮数
    max_rounds: int = 10                           # 最大辩论轮数
    stability_rounds: int = 2                      # 稳定轮数要求
    metric: ConsensusMetric = ConsensusMetric.STANCE_ALIGNMENT
    early_termination: bool = True                 # 是否允许提前终止
    convergence_tolerance: float = 0.05            # 收敛容差


class OpinionAnalyzer:
    """
    意见分析器
    分析参与者意见分布和变化
    """
    
    def __init__(self) -> None:
        self.opinion_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    
    def record_opinion(
        self,
        participant_id: str,
        round_num: int,
        opinion_score: float
    ) -> None:
        """
        记录参与者意见
        
        Args:
            participant_id: 参与者ID
            round_num: 轮次
            opinion_score: 意见分数 (-1 到 1，-1为强烈反对，1为强烈支持)
        """
        self.opinion_history[participant_id].append((round_num, opinion_score))
    
    def get_opinion_trajectory(self, participant_id: str) -> List[Tuple[int, float]]:
        """获取参与者的意见轨迹"""
        return self.opinion_history.get(participant_id, [])
    
    def calculate_convergence(self, round_num: int) -> float:
        """
        计算当前轮次的意见收敛度
        
        返回值：0-1，越高表示意见越集中
        """
        current_opinions = []
        for participant_id, history in self.opinion_history.items():
            # 找到该轮或最近轮的意见
            for r, o in reversed(history):
                if r <= round_num:
                    current_opinions.append(o)
                    break
        
        if len(current_opinions) < 2:
            return 0.0
        
        # 使用标准差计算分散度，然后转换为收敛度
        std_dev = statistics.stdev(current_opinions)
        # 标准差越小，收敛度越高
        convergence = max(0.0, 1.0 - std_dev)
        return convergence
    
    def calculate_variance_reduction(self, current_round: int) -> float:
        """计算方差缩减程度"""
        if current_round < 2:
            return 0.0
        
        # 获取前两轮的意见
        opinions_prev = []
        opinions_curr = []
        
        for history in self.opinion_history.values():
            # 上一轮
            for r, o in reversed(history):
                if r <= current_round - 1:
                    opinions_prev.append(o)
                    break
            # 当前轮
            for r, o in reversed(history):
                if r <= current_round:
                    opinions_curr.append(o)
                    break
        
        if len(opinions_prev) < 2 or len(opinions_curr) < 2:
            return 0.0
        
        var_prev = statistics.variance(opinions_prev)
        var_curr = statistics.variance(opinions_curr)
        
        if var_prev == 0:
            return 1.0 if var_curr == 0 else 0.0
        
        reduction = (var_prev - var_curr) / var_prev
        return max(0.0, reduction)


class ConsensusDetector:
    """
    共识检测器
    检测辩论是否达成共识
    """
    
    def __init__(self, config: Optional[ConsensusConfig] = None) -> None:
        self.config = config or ConsensusConfig()
        self.opinion_analyzer = OpinionAnalyzer()
        self.consensus_states: Dict[str, ConsensusState] = {}
        self.stability_counter: Dict[str, int] = defaultdict(int)
    
    def analyze_debate_state(
        self,
        debate_state: DebateState
    ) -> ConsensusState:
        """
        分析辩论状态并计算共识水平
        
        Args:
            debate_state: 辩论状态
            
        Returns:
            共识状态
        """
        debate_id = debate_state.debate_id
        
        # 获取或创建共识状态
        if debate_id not in self.consensus_states:
            self.consensus_states[debate_id] = ConsensusState(debate_id=debate_id)
        
        consensus_state = self.consensus_states[debate_id]
        
        # 计算立场分布
        stance_dist = self._calculate_stance_distribution(debate_state)
        consensus_state.stance_distribution = stance_dist
        
        # 根据配置的指标计算共识水平
        if self.config.metric == ConsensusMetric.STANCE_ALIGNMENT:
            consensus_level = self._calculate_stance_alignment(stance_dist)
        elif self.config.metric == ConsensusMetric.OPINION_CONVERGENCE:
            consensus_level = self.opinion_analyzer.calculate_convergence(
                debate_state.round_number
            )
        elif self.config.metric == ConsensusMetric.AGREEMENT_RATIO:
            consensus_level = self._calculate_agreement_ratio(debate_state)
        elif self.config.metric == ConsensusMetric.VARIANCE_REDUCTION:
            consensus_level = self.opinion_analyzer.calculate_variance_reduction(
                debate_state.round_number
            )
        elif self.config.metric == ConsensusMetric.ENTROPY_DECREASE:
            consensus_level = self._calculate_entropy_decrease(debate_state)
        else:
            consensus_level = self._calculate_stance_alignment(stance_dist)
        
        consensus_state.current_consensus_level = consensus_level
        consensus_state.consensus_history.append(
            (debate_state.round_number, consensus_level)
        )
        
        # 计算收敛速率
        if len(consensus_state.consensus_history) >= 2:
            prev_level = consensus_state.consensus_history[-2][1]
            consensus_state.convergence_rate = consensus_level - prev_level
        
        # 检测是否达成共识
        consensus_state.is_consensus_reached = self._check_consensus_reached(
            consensus_state, debate_state
        )
        
        if consensus_state.is_consensus_reached:
            consensus_state.consensus_stance = self._determine_consensus_stance(
                stance_dist
            )
            consensus_state.confidence = consensus_level
        
        return consensus_state
    
    def _calculate_stance_distribution(
        self,
        debate_state: DebateState
    ) -> Dict[Stance, float]:
        """计算立场分布"""
        stance_counts = defaultdict(int)
        total = 0
        
        for argument in debate_state.arguments.values():
            stance_counts[argument.stance] += 1
            total += 1
        
        if total == 0:
            return {Stance.NEUTRAL: 1.0}
        
        return {
            stance: count / total
            for stance, count in stance_counts.items()
        }
    
    def _calculate_stance_alignment(
        self,
        stance_distribution: Dict[Stance, float]
    ) -> float:
        """
        计算立场对齐度
        
        使用最大比例作为对齐度指标
        """
        if not stance_distribution:
            return 0.0
        
        max_proportion = max(stance_distribution.values())
        return max_proportion
    
    def _calculate_agreement_ratio(self, debate_state: DebateState) -> float:
        """计算同意比例（基于论点间的相似性）"""
        arguments = list(debate_state.arguments.values())
        if len(arguments) < 2:
            return 0.0
        
        # 计算立场一致性
        stance_agreements = 0
        total_pairs = 0
        
        for i, arg1 in enumerate(arguments):
            for arg2 in arguments[i+1:]:
                total_pairs += 1
                if arg1.stance == arg2.stance:
                    stance_agreements += 1
        
        if total_pairs == 0:
            return 0.0
        
        return stance_agreements / total_pairs
    
    def _calculate_entropy_decrease(self, debate_state: DebateState) -> float:
        """计算熵减（意见不确定性降低）"""
        stance_dist = self._calculate_stance_distribution(debate_state)
        
        # 计算香农熵
        entropy = 0.0
        for proportion in stance_dist.values():
            if proportion > 0:
                entropy -= proportion * math.log2(proportion)
        
        # 最大熵（均匀分布）
        max_entropy = math.log2(len(stance_dist)) if stance_dist else 1.0
        
        # 归一化熵减
        if max_entropy == 0:
            return 1.0
        
        normalized_entropy = entropy / max_entropy
        return 1.0 - normalized_entropy  # 熵减 = 1 - 归一化熵
    
    def _check_consensus_reached(
        self,
        consensus_state: ConsensusState,
        debate_state: DebateState
    ) -> bool:
        """检查是否达成共识"""
        # 检查最小轮数
        if debate_state.round_number < self.config.min_rounds:
            return False
        
        # 检查最大轮数
        if debate_state.round_number >= self.config.max_rounds:
            return True  # 强制终止
        
        # 检查共识水平
        if consensus_state.current_consensus_level < self.config.threshold:
            return False
        
        # 检查稳定性
        if len(consensus_state.consensus_history) >= self.config.stability_rounds:
            recent_levels = [
                level for _, level in 
                consensus_state.consensus_history[-self.config.stability_rounds:]
            ]
            # 检查共识水平是否稳定
            if max(recent_levels) - min(recent_levels) > self.config.convergence_tolerance:
                return False
        
        # 增加稳定性计数
        self.stability_counter[debate_state.debate_id] += 1
        
        # 需要连续多轮稳定
        if self.stability_counter[debate_state.debate_id] < self.config.stability_rounds:
            return False
        
        return True
    
    def _determine_consensus_stance(
        self,
        stance_distribution: Dict[Stance, float]
    ) -> Optional[Stance]:
        """确定共识立场"""
        if not stance_distribution:
            return None
        
        # 找出比例最高的立场
        max_stance = max(stance_distribution.keys(), key=lambda s: stance_distribution[s])
        max_proportion = stance_distribution[max_stance]
        
        # 需要超过阈值
        if max_proportion >= self.config.threshold:
            return max_stance
        
        return None
    
    def should_terminate(self, debate_state: DebateState) -> Tuple[bool, str]:
        """
        判断是否应该终止辩论
        
        Returns:
            (是否终止, 原因)
        """
        consensus_state = self.analyze_debate_state(debate_state)
        
        if consensus_state.is_consensus_reached:
            return True, f"达成共识（水平={consensus_state.current_consensus_level:.2f}）"
        
        if debate_state.round_number >= self.config.max_rounds:
            return True, "达到最大轮数限制"
        
        if self.config.early_termination:
            # 检查是否意见发散（共识水平下降且持续）
            if len(consensus_state.consensus_history) >= 3:
                recent = [level for _, level in consensus_state.consensus_history[-3:]]
                if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    if recent[-1] < 0.4:  # 共识水平过低
                        return True, "意见持续发散，难以达成共识"
        
        return False, "继续辩论"
    
    def get_consensus_report(self, debate_id: str) -> Dict[str, Any]:
        """获取共识检测报告"""
        if debate_id not in self.consensus_states:
            return {"error": "未找到该辩论的共识记录"}
        
        state = self.consensus_states[debate_id]
        return {
            "debate_id": debate_id,
            "current_consensus_level": state.current_consensus_level,
            "is_consensus_reached": state.is_consensus_reached,
            "consensus_stance": state.consensus_stance.name if state.consensus_stance else None,
            "confidence": state.confidence,
            "convergence_rate": state.convergence_rate,
            "consensus_history": state.consensus_history,
            "stance_distribution": {
                stance.name: prop 
                for stance, prop in state.stance_distribution.items()
            },
        }


class ConsensusBuilder:
    """
    共识构建器
    主动促进共识形成
    """
    
    def __init__(self) -> None:
        self.bridge_proposals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def identify_disagreement_points(
        self,
        debate_state: DebateState
    ) -> List[Dict[str, Any]]:
        """
        识别分歧点
        
        Returns:
            分歧点列表
        """
        disagreements = []
        
        # 按立场分组论点
        pro_args = debate_state.get_arguments_by_stance(Stance.PRO)
        con_args = debate_state.get_arguments_by_stance(Stance.CON)
        
        # 识别核心分歧
        if pro_args and con_args:
            disagreements.append({
                "type": "fundamental",
                "description": "对议题的基本立场存在分歧",
                "pro_count": len(pro_args),
                "con_count": len(con_args),
            })
        
        # 识别证据分歧
        pro_evidence = sum(len(arg.evidence_list) for arg in pro_args)
        con_evidence = sum(len(arg.evidence_list) for arg in con_args)
        
        if abs(pro_evidence - con_evidence) > 2:
            disagreements.append({
                "type": "evidence",
                "description": "证据数量存在显著差异",
                "pro_evidence": pro_evidence,
                "con_evidence": con_evidence,
            })
        
        return disagreements
    
    def generate_bridge_proposal(
        self,
        debate_state: DebateState
    ) -> Optional[Dict[str, Any]]:
        """
        生成折中提案
        
        Returns:
            折中提案或None
        """
        disagreements = self.identify_disagreement_points(debate_state)
        
        if not disagreements:
            return None
        
        # 基于分歧生成折中方案
        proposal = {
            "proposal_id": f"bridge_{len(self.bridge_proposals[debate_state.debate_id])}",
            "description": "基于当前讨论，建议采取以下折中方案：",
            "key_points": [],
            "supporting_arguments": [],
        }
        
        # 提取双方的高质量论点
        all_args = list(debate_state.arguments.values())
        high_quality_args = [
            arg for arg in all_args 
            if arg.confidence > 0.7 or len(arg.evidence_list) >= 2
        ]
        
        for arg in high_quality_args[:3]:  # 取前3个
            proposal["supporting_arguments"].append({
                "argument_id": arg.argument_id,
                "stance": arg.stance.name,
                "content_preview": arg.content[:100],
            })
        
        self.bridge_proposals[debate_state.debate_id].append(proposal)
        
        return proposal
    
    def suggest_compromise_position(
        self,
        debate_state: DebateState
    ) -> Optional[Dict[str, Any]]:
        """
        建议妥协立场
        
        Returns:
            妥协立场建议
        """
        # 分析立场分布
        stance_dist = {}
        for arg in debate_state.arguments.values():
            stance_dist[arg.stance] = stance_dist.get(arg.stance, 0) + 1
        
        total = sum(stance_dist.values())
        if total == 0:
            return None
        
        # 计算支持度
        pro_ratio = stance_dist.get(Stance.PRO, 0) / total
        con_ratio = stance_dist.get(Stance.CON, 0) / total
        
        # 建议妥协立场
        if abs(pro_ratio - con_ratio) < 0.2:
            # 势均力敌，建议中立
            suggestion = {
                "position": "neutral",
                "description": "双方观点势均力敌，建议采取审慎的中立立场",
                "pro_ratio": pro_ratio,
                "con_ratio": con_ratio,
            }
        elif pro_ratio > con_ratio:
            # 支持方占优，但承认反对方部分观点
            suggestion = {
                "position": "qualified_pro",
                "description": "总体支持，但需考虑反对方提出的合理关切",
                "pro_ratio": pro_ratio,
                "con_ratio": con_ratio,
            }
        else:
            suggestion = {
                "position": "qualified_con",
                "description": "总体反对，但承认支持方提出的部分价值",
                "pro_ratio": pro_ratio,
                "con_ratio": con_ratio,
            }
        
        return suggestion


__all__ = [
    "ConsensusMetric",
    "ConsensusState",
    "ConsensusConfig",
    "OpinionAnalyzer",
    "ConsensusDetector",
    "ConsensusBuilder",
]
