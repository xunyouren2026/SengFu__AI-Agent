"""
仲裁者模块
综合各方论点做出最终判断
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import statistics
from collections import defaultdict

from .protocol import (
    Argument, Rebuttal, Verdict, Revision,
    Stance, DebateState, DebatePhase
)


class ArbitrationMethod(Enum):
    """仲裁方法枚举"""
    WEIGHTED_SCORING = "weighted_scoring"      # 加权评分法
    DELIBERATIVE = "deliberative"              # 审议式
    EVIDENCE_BASED = "evidence_based"          # 证据导向
    CONSENSUS_ORIENTED = "consensus_oriented"  # 共识导向
    MAJORITY_RULE = "majority_rule"            # 多数决


@dataclass
class ArgumentEvaluation:
    """论点评估结果"""
    argument_id: str
    logical_strength: float = 0.0      # 逻辑强度
    evidence_quality: float = 0.0      # 证据质量
    relevance: float = 0.0             # 相关性
    clarity: float = 0.0               # 清晰度
    originality: float = 0.0           # 原创性
    rebuttal_resistance: float = 0.0   # 抗反驳能力
    overall_score: float = 0.0         # 综合得分
    
    def calculate_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """计算综合得分"""
        if weights is None:
            weights = {
                "logical_strength": 0.25,
                "evidence_quality": 0.25,
                "relevance": 0.15,
                "clarity": 0.15,
                "originality": 0.1,
                "rebuttal_resistance": 0.1,
            }
        
        score = (
            self.logical_strength * weights["logical_strength"] +
            self.evidence_quality * weights["evidence_quality"] +
            self.relevance * weights["relevance"] +
            self.clarity * weights["clarity"] +
            self.originality * weights["originality"] +
            self.rebuttal_resistance * weights["rebuttal_resistance"]
        )
        self.overall_score = score
        return score


@dataclass
class ArbitrationConfig:
    """仲裁配置"""
    method: ArbitrationMethod = ArbitrationMethod.WEIGHTED_SCORING
    min_confidence_threshold: float = 0.6
    require_unanimity: bool = False
    consider_revisions: bool = True
    weight_by_credibility: bool = True
    weight_by_expertise: bool = True
    transparency_level: str = "high"  # high/medium/low


class ArgumentEvaluator:
    """
    论点评估器
    评估单个论点的质量
    """
    
    def __init__(self, config: ArbitrationConfig) -> None:
        self.config = config
    
    def evaluate(
        self,
        argument: Argument,
        rebuttals: List[Rebuttal],
        all_arguments: Dict[str, Argument]
    ) -> ArgumentEvaluation:
        """
        评估论点
        
        Args:
            argument: 要评估的论点
            rebuttals: 针对该论点的反驳
            all_arguments: 所有论点字典
        """
        evaluation = ArgumentEvaluation(argument_id=argument.argument_id)
        
        # 评估逻辑强度
        evaluation.logical_strength = self._assess_logical_strength(argument)
        
        # 评估证据质量
        evaluation.evidence_quality = argument.get_evidence_strength()
        
        # 评估相关性
        evaluation.relevance = self._assess_relevance(argument)
        
        # 评估清晰度
        evaluation.clarity = self._assess_clarity(argument.content)
        
        # 评估原创性
        evaluation.originality = self._assess_originality(argument, all_arguments)
        
        # 评估抗反驳能力
        evaluation.rebuttal_resistance = self._assess_rebuttal_resistance(
            argument, rebuttals
        )
        
        # 计算综合得分
        evaluation.calculate_overall()
        
        return evaluation
    
    def _assess_logical_strength(self, argument: Argument) -> float:
        """评估逻辑强度"""
        # 检查逻辑连接词
        logical_markers = ["因为", "所以", "因此", "如果", "那么", "由此", "从而"]
        marker_count = sum(1 for m in logical_markers if m in argument.content)
        
        # 检查逻辑结构完整性
        has_premise = any(w in argument.content for w in ["首先", "第一", "前提"])
        has_conclusion = any(w in argument.content for w in ["结论", "因此", "所以"])
        
        structure_score = 0.3 if has_premise else 0.0
        structure_score += 0.3 if has_conclusion else 0.0
        
        # 逻辑标记分数
        marker_score = min(0.4, marker_count * 0.1)
        
        return min(1.0, structure_score + marker_score + 0.3)  # 基础分0.3
    
    def _assess_relevance(self, argument: Argument) -> float:
        """评估相关性"""
        # 基于论点类型的相关性基线
        base_relevance = {
            "FACTUAL": 0.8,
            "NORMATIVE": 0.7,
            "CAUSAL": 0.75,
            "ANALOGICAL": 0.6,
            "AUTHORITY": 0.7,
            "PRAGMATIC": 0.75,
        }
        
        type_name = argument.argument_type.name
        return base_relevance.get(type_name, 0.6)
    
    def _assess_clarity(self, content: str) -> float:
        """评估清晰度"""
        # 长度适中性
        length = len(content)
        if 100 <= length <= 500:
            length_score = 1.0
        elif length < 100:
            length_score = 0.5 + length / 200
        else:
            length_score = max(0.5, 1.0 - (length - 500) / 1000)
        
        # 结构清晰度
        structure_markers = ["首先", "其次", "最后", "第一", "第二", "总结"]
        structure_score = min(1.0, sum(1 for m in structure_markers if m in content) * 0.25 + 0.5)
        
        return (length_score + structure_score) / 2
    
    def _assess_originality(
        self,
        argument: Argument,
        all_arguments: Dict[str, Argument]
    ) -> float:
        """评估原创性"""
        if not all_arguments:
            return 0.8  # 第一个论点默认为原创
        
        # 计算与其他论点的相似度（简化版）
        similarities = []
        for other_id, other in all_arguments.items():
            if other_id != argument.argument_id:
                # 简单的词重叠计算
                words1 = set(argument.content.lower().split())
                words2 = set(other.content.lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    similarities.append(overlap)
        
        if not similarities:
            return 0.8
        
        avg_similarity = sum(similarities) / len(similarities)
        originality = 1.0 - avg_similarity
        return max(0.3, originality)
    
    def _assess_rebuttal_resistance(
        self,
        argument: Argument,
        rebuttals: List[Rebuttal]
    ) -> float:
        """评估抗反驳能力"""
        if not rebuttals:
            return 0.8  # 未被反驳的论点
        
        # 计算反驳的平均有效性
        avg_effectiveness = sum(r.effectiveness_score for r in rebuttals) / len(rebuttals)
        
        # 抗反驳能力与反驳有效性负相关
        resistance = 1.0 - avg_effectiveness * 0.8
        return max(0.2, resistance)


class Arbitrator:
    """
    仲裁者
    综合各方论点做出最终判断
    """
    
    def __init__(
        self,
        arbitrator_id: str,
        config: Optional[ArbitrationConfig] = None
    ) -> None:
        self.arbitrator_id = arbitrator_id
        self.config = config or ArbitrationConfig()
        self.evaluator = ArgumentEvaluator(self.config)
        self.evaluation_history: List[ArgumentEvaluation] = []
    
    def arbitrate(self, debate_state: DebateState) -> Verdict:
        """
        对辩论进行仲裁
        
        Args:
            debate_state: 辩论状态
            
        Returns:
            裁决结果
        """
        # 评估所有论点
        evaluations = self._evaluate_all_arguments(debate_state)
        self.evaluation_history.extend(evaluations)
        
        # 根据仲裁方法做出裁决
        if self.config.method == ArbitrationMethod.WEIGHTED_SCORING:
            return self._weighted_scoring_arbitration(debate_state, evaluations)
        elif self.config.method == ArbitrationMethod.EVIDENCE_BASED:
            return self._evidence_based_arbitration(debate_state, evaluations)
        elif self.config.method == ArbitrationMethod.CONSENSUS_ORIENTED:
            return self._consensus_oriented_arbitration(debate_state, evaluations)
        elif self.config.method == ArbitrationMethod.MAJORITY_RULE:
            return self._majority_rule_arbitration(debate_state, evaluations)
        else:
            return self._deliberative_arbitration(debate_state, evaluations)
    
    def _evaluate_all_arguments(
        self,
        debate_state: DebateState
    ) -> List[ArgumentEvaluation]:
        """评估所有论点"""
        evaluations = []
        
        for argument_id, argument in debate_state.arguments.items():
            rebuttals = debate_state.get_rebuttals_for_argument(argument_id)
            evaluation = self.evaluator.evaluate(
                argument, rebuttals, debate_state.arguments
            )
            evaluations.append(evaluation)
        
        return evaluations
    
    def _weighted_scoring_arbitration(
        self,
        debate_state: DebateState,
        evaluations: List[ArgumentEvaluation]
    ) -> Verdict:
        """加权评分仲裁"""
        # 按立场分组计算总分
        pro_scores = []
        con_scores = []
        
        for eval_result in evaluations:
            argument = debate_state.arguments.get(eval_result.argument_id)
            if not argument:
                continue
            
            score = eval_result.overall_score
            
            # 根据发言者可信度加权
            if self.config.weight_by_credibility:
                score *= argument.confidence
            
            if argument.stance == Stance.PRO:
                pro_scores.append(score)
            elif argument.stance == Stance.CON:
                con_scores.append(score)
        
        # 计算总分
        pro_total = sum(pro_scores) if pro_scores else 0
        con_total = sum(con_scores) if con_scores else 0
        
        # 确定获胜方
        if pro_total > con_total * 1.1:  # 需要10%的优势
            winning_stance = Stance.PRO
            confidence = min(0.95, pro_total / (pro_total + con_total + 0.01))
        elif con_total > pro_total * 1.1:
            winning_stance = Stance.CON
            confidence = min(0.95, con_total / (pro_total + con_total + 0.01))
        else:
            winning_stance = None  # 平局或无法确定
            confidence = 0.5
        
        # 生成裁决理由
        reasoning = self._generate_reasoning(
            evaluations, winning_stance, pro_total, con_total
        )
        
        # 构建论点分数映射
        argument_scores = {
            e.argument_id: e.overall_score for e in evaluations
        }
        
        return Verdict(
            arbitrator_id=self.arbitrator_id,
            topic=debate_state.topic,
            winning_stance=winning_stance,
            reasoning=reasoning,
            confidence=confidence,
            argument_scores=argument_scores
        )
    
    def _evidence_based_arbitration(
        self,
        debate_state: DebateState,
        evaluations: List[ArgumentEvaluation]
    ) -> Verdict:
        """证据导向仲裁"""
        # 主要基于证据质量评分
        pro_evidence = []
        con_evidence = []
        
        for eval_result in evaluations:
            argument = debate_state.arguments.get(eval_result.argument_id)
            if not argument:
                continue
            
            evidence_score = eval_result.evidence_quality
            
            if argument.stance == Stance.PRO:
                pro_evidence.append(evidence_score)
            elif argument.stance == Stance.CON:
                con_evidence.append(evidence_score)
        
        pro_avg = statistics.mean(pro_evidence) if pro_evidence else 0
        con_avg = statistics.mean(con_evidence) if con_evidence else 0
        
        if pro_avg > con_avg + 0.15:
            winning_stance = Stance.PRO
            confidence = min(0.9, 0.5 + (pro_avg - con_avg))
        elif con_avg > pro_avg + 0.15:
            winning_stance = Stance.CON
            confidence = min(0.9, 0.5 + (con_avg - pro_avg))
        else:
            winning_stance = None
            confidence = 0.5
        
        reasoning = (
            f"基于证据质量评估：支持方平均证据质量={pro_avg:.2f}, "
            f"反对方平均证据质量={con_avg:.2f}"
        )
        
        argument_scores = {
            e.argument_id: e.evidence_quality for e in evaluations
        }
        
        return Verdict(
            arbitrator_id=self.arbitrator_id,
            topic=debate_state.topic,
            winning_stance=winning_stance,
            reasoning=reasoning,
            confidence=confidence,
            argument_scores=argument_scores
        )
    
    def _consensus_oriented_arbitration(
        self,
        debate_state: DebateState,
        evaluations: List[ArgumentEvaluation]
    ) -> Verdict:
        """共识导向仲裁"""
        # 寻找双方都能接受的中间立场
        pro_args = []
        con_args = []
        
        for eval_result in evaluations:
            argument = debate_state.arguments.get(eval_result.argument_id)
            if not argument:
                continue
            
            if argument.stance == Stance.PRO:
                pro_args.append(eval_result)
            elif argument.stance == Stance.CON:
                con_args.append(eval_result)
        
        # 找出双方的高质量论点
        strong_pro = [e for e in pro_args if e.overall_score > 0.7]
        strong_con = [e for e in con_args if e.overall_score > 0.7]
        
        # 如果双方都有强论点，倾向于折中
        if strong_pro and strong_con:
            winning_stance = None  # 建议折中
            confidence = 0.6
            reasoning = (
                "双方均提出了有说服力的论点。建议采取折中方案，"
                f"支持方强论点数：{len(strong_pro)}, 反对方强论点数：{len(strong_con)}"
            )
        elif strong_pro:
            winning_stance = Stance.PRO
            confidence = 0.75
            reasoning = f"支持方的论点质量明显更高，强论点数：{len(strong_pro)}"
        elif strong_con:
            winning_stance = Stance.CON
            confidence = 0.75
            reasoning = f"反对方的论点质量明显更高，强论点数：{len(strong_con)}"
        else:
            winning_stance = None
            confidence = 0.5
            reasoning = "双方论点质量相当，无法做出明确裁决"
        
        argument_scores = {
            e.argument_id: e.overall_score for e in evaluations
        }
        
        return Verdict(
            arbitrator_id=self.arbitrator_id,
            topic=debate_state.topic,
            winning_stance=winning_stance,
            reasoning=reasoning,
            confidence=confidence,
            argument_scores=argument_scores
        )
    
    def _majority_rule_arbitration(
        self,
        debate_state: DebateState,
        evaluations: List[ArgumentEvaluation]
    ) -> Verdict:
        """多数决仲裁"""
        # 简单计算论点数量
        pro_count = sum(
            1 for e in evaluations
            if debate_state.arguments.get(e.argument_id, Argument()).stance == Stance.PRO
        )
        con_count = sum(
            1 for e in evaluations
            if debate_state.arguments.get(e.argument_id, Argument()).stance == Stance.CON
        )
        
        total = pro_count + con_count
        if total == 0:
            winning_stance = None
            confidence = 0.0
        elif pro_count > con_count:
            winning_stance = Stance.PRO
            confidence = pro_count / total
        elif con_count > pro_count:
            winning_stance = Stance.CON
            confidence = con_count / total
        else:
            winning_stance = None
            confidence = 0.5
        
        reasoning = f"多数决结果：支持方论点数={pro_count}, 反对方论点数={con_count}"
        
        argument_scores = {
            e.argument_id: float(
                debate_state.arguments.get(e.argument_id, Argument()).stance == Stance.PRO
            ) for e in evaluations
        }
        
        return Verdict(
            arbitrator_id=self.arbitrator_id,
            topic=debate_state.topic,
            winning_stance=winning_stance,
            reasoning=reasoning,
            confidence=confidence,
            argument_scores=argument_scores
        )
    
    def _deliberative_arbitration(
        self,
        debate_state: DebateState,
        evaluations: List[ArgumentEvaluation]
    ) -> Verdict:
        """审议式仲裁"""
        # 综合考虑多个维度
        return self._weighted_scoring_arbitration(debate_state, evaluations)
    
    def _generate_reasoning(
        self,
        evaluations: List[ArgumentEvaluation],
        winning_stance: Optional[Stance],
        pro_total: float,
        con_total: float
    ) -> str:
        """生成裁决理由"""
        parts = []
        
        # 总体评估
        parts.append(f"综合评分：支持方总分={pro_total:.2f}, 反对方总分={con_total:.2f}")
        
        # 最佳论点
        if evaluations:
            best = max(evaluations, key=lambda e: e.overall_score)
            parts.append(f"最高质量论点得分：{best.overall_score:.2f}")
        
        # 裁决结论
        if winning_stance == Stance.PRO:
            parts.append("裁决：支持方获胜")
        elif winning_stance == Stance.CON:
            parts.append("裁决：反对方获胜")
        else:
            parts.append("裁决：无法明确判定胜负，建议进一步讨论")
        
        return "；".join(parts)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.evaluation_history:
            return {"message": "暂无评估记录"}
        
        scores = [e.overall_score for e in self.evaluation_history]
        return {
            "total_evaluated": len(self.evaluation_history),
            "average_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0,
        }


class ArbitrationPanel:
    """
    仲裁小组
    多个仲裁者共同裁决
    """
    
    def __init__(self, panel_id: str) -> None:
        self.panel_id = panel_id
        self.arbitrators: List[Arbitrator] = []
        self.arbitration_history: List[Verdict] = []
    
    def add_arbitrator(self, arbitrator: Arbitrator) -> None:
        """添加仲裁者"""
        self.arbitrators.append(arbitrator)
    
    def collective_arbitration(
        self,
        debate_state: DebateState,
        aggregation_method: str = "majority"
    ) -> Verdict:
        """
        集体仲裁
        
        Args:
            debate_state: 辩论状态
            aggregation_method: 聚合方法 (majority/average/weighted)
        """
        if not self.arbitrators:
            raise ValueError("仲裁小组为空")
        
        # 收集各仲裁者的裁决
        verdicts = []
        for arbitrator in self.arbitrators:
            verdict = arbitrator.arbitrate(debate_state)
            verdicts.append(verdict)
        
        self.arbitration_history.extend(verdicts)
        
        # 聚合裁决
        if aggregation_method == "majority":
            return self._majority_aggregation(verdicts, debate_state)
        elif aggregation_method == "average":
            return self._average_aggregation(verdicts, debate_state)
        else:
            return self._weighted_aggregation(verdicts, debate_state)
    
    def _majority_aggregation(
        self,
        verdicts: List[Verdict],
        debate_state: DebateState
    ) -> Verdict:
        """多数聚合"""
        stance_counts = defaultdict(int)
        for v in verdicts:
            if v.winning_stance:
                stance_counts[v.winning_stance] += 1
        
        if stance_counts:
            winning_stance = max(stance_counts.keys(), key=lambda s: stance_counts[s])
            confidence = stance_counts[winning_stance] / len(verdicts)
        else:
            winning_stance = None
            confidence = 0.0
        
        # 合并理由
        all_reasonings = " | ".join(set(v.reasoning for v in verdicts))
        
        # 合并分数
        merged_scores: Dict[str, float] = {}
        for v in verdicts:
            for arg_id, score in v.argument_scores.items():
                if arg_id not in merged_scores:
                    merged_scores[arg_id] = []
                merged_scores[arg_id] = merged_scores.get(arg_id, 0) + score / len(verdicts)
        
        return Verdict(
            arbitrator_id=f"panel:{self.panel_id}",
            topic=debate_state.topic,
            winning_stance=winning_stance,
            reasoning=f"集体裁决（多数决）：{all_reasonings}",
            confidence=confidence,
            argument_scores=merged_scores
        )
    
    def _average_aggregation(
        self,
        verdicts: List[Verdict],
        debate_state: DebateState
    ) -> Verdict:
        """平均聚合"""
        # 计算平均置信度
        avg_confidence = sum(v.confidence for v in verdicts) / len(verdicts)
        
        # 确定多数立场
        pro_count = sum(1 for v in verdicts if v.winning_stance == Stance.PRO)
        con_count = sum(1 for v in verdicts if v.winning_stance == Stance.CON)
        
        if pro_count > con_count:
            winning_stance = Stance.PRO
        elif con_count > pro_count:
            winning_stance = Stance.CON
        else:
            winning_stance = None
        
        return Verdict(
            arbitrator_id=f"panel:{self.panel_id}",
            topic=debate_state.topic,
            winning_stance=winning_stance,
            reasoning="集体裁决（平均聚合）",
            confidence=avg_confidence,
            argument_scores={}
        )
    
    def _weighted_aggregation(
        self,
        verdicts: List[Verdict],
        debate_state: DebateState
    ) -> Verdict:
        """加权聚合"""
        # 简化实现：与平均聚合相同
        return self._average_aggregation(verdicts, debate_state)


__all__ = [
    "ArbitrationMethod",
    "ArgumentEvaluation",
    "ArbitrationConfig",
    "ArgumentEvaluator",
    "Arbitrator",
    "ArbitrationPanel",
]
