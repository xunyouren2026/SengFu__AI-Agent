"""
伦理推理引擎 - 解决道德困境

实现多种伦理推理框架(功利主义、义务论、美德伦理、关怀伦理)，
以及综合推理引擎和经典道德困境库。
所有实现使用纯Python，不依赖任何外部库。
"""

import math
import random
import hashlib
import time
import re
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


# ==================== 基础数据模型 ====================

@dataclass
class EthicalDilemma:
    """道德困境"""
    dilemma_id: str
    description: str
    context: str = ""
    stakeholders: List[str] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    consequences: List[Dict[str, float]] = field(default_factory=list)
    category: str = "general"  # trolley/privacy/fairness/honesty/loyalty/authority


@dataclass
class EthicalPrinciple:
    """伦理原则"""
    name: str
    description: str
    weight: float = 1.0  # 0-1
    rationale: str = ""
    _apply_fn: Optional[Callable] = field(default=None, repr=False)

    def apply(self, dilemma: EthicalDilemma, option_index: int) -> float:
        """对特定选项打分"""
        if self._apply_fn is not None:
            return self._apply_fn(dilemma, option_index)
        return self._default_apply(dilemma, option_index)

    def _default_apply(self, dilemma: EthicalDilemma, option_index: int) -> float:
        """默认打分逻辑: 基于后果的总和"""
        if option_index >= len(dilemma.consequences):
            return 0.0
        consequences = dilemma.consequences[option_index]
        total = sum(consequences.values())
        # 归一化到 0-1
        return max(0.0, min(1.0, (total + 10) / 20.0))


@dataclass
class EthicalDecision:
    """伦理决策"""
    recommended_option: int
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    principle_scores: Dict[str, float] = field(default_factory=dict)
    considerations: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


@dataclass
class VirtueProfile:
    """美德档案"""
    name: str
    description: str
    indicators: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class ComprehensiveEthicalDecision:
    """综合伦理决策"""
    recommended_option: int
    overall_confidence: float
    individual_decisions: Dict[str, EthicalDecision] = field(default_factory=dict)
    agreement_score: float = 0.0  # 0-1, 推理器间一致性
    reasoning_chain: List[str] = field(default_factory=list)
    final_considerations: List[str] = field(default_factory=list)


# ==================== 功利主义推理器 ====================

class UtilitarianReasoner:
    """功利主义推理器 - 最大化总体效用"""

    def __init__(self):
        self._principles: List[EthicalPrinciple] = [
            EthicalPrinciple(
                name="maximize_total_utility",
                description="Maximize the total well-being across all stakeholders",
                weight=1.0,
                rationale="The right action produces the greatest good for the greatest number",
            ),
            EthicalPrinciple(
                name="minimize_suffering",
                description="Minimize total suffering and harm",
                weight=0.8,
                rationale="Reducing suffering is often more urgent than increasing happiness",
            ),
            EthicalPrinciple(
                name="prefer_fair_distribution",
                description="Prefer more equitable distributions of utility",
                weight=0.5,
                rationale="A more equal distribution of well-being is generally preferable",
            ),
        ]

    def reason(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """功利主义推理: 计算每个选项的总效用"""
        option_scores = []
        reasoning_chain = []
        principle_scores = {}

        for i, option in enumerate(dilemma.options):
            if i >= len(dilemma.consequences):
                option_scores.append(0.0)
                continue

            consequences = dilemma.consequences[i]
            utility = self._calculate_utility(consequences, dilemma.stakeholders)

            # 考虑不确定性
            adjusted_utility = self._handle_uncertainty(consequences, utility)

            # 时间折扣
            discounted_utility = self._apply_time_discount(adjusted_utility, 1.0)

            # 应用原则权重
            weighted_score = 0.0
            for principle in self._principles:
                p_score = principle.apply(dilemma, i)
                weighted_score += p_score * principle.weight
                principle_scores[principle.name] = round(p_score, 4)

            option_scores.append(discounted_utility)

            reasoning_chain.append(
                f"Option {i+1} ('{option[:40]}'): "
                f"raw utility={utility:.3f}, "
                f"adjusted={adjusted_utility:.3f}, "
                f"discounted={discounted_utility:.3f}"
            )

        # 选择效用最高的选项
        if not option_scores:
            return EthicalDecision(
                recommended_option=0,
                confidence=0.0,
                reasoning_chain=["No options available for evaluation"],
            )

        best_idx = max(range(len(option_scores)), key=lambda i: option_scores[i])
        best_score = option_scores[best_idx]

        # 计算置信度: 基于最佳选项与次优选项的差距
        sorted_scores = sorted(option_scores, reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] != 0:
            confidence = min(1.0, abs(sorted_scores[0] - sorted_scores[1]) / (abs(sorted_scores[0]) + 0.001))
        else:
            confidence = 0.5

        reasoning_chain.append(
            f"Utilitarian analysis recommends option {best_idx+1} "
            f"with utility score {best_score:.3f}"
        )

        considerations = [
            "This analysis focuses on aggregate outcomes",
            "Individual rights may be overridden for greater good",
            "Long-term consequences may differ from immediate outcomes",
        ]

        caveats = [
            "Utility values are approximations based on available information",
            "Distribution of utility among stakeholders is not fully captured",
            "Hard-to-quantify values (dignity, autonomy) may be undervalued",
        ]

        return EthicalDecision(
            recommended_option=best_idx,
            confidence=round(confidence, 4),
            reasoning_chain=reasoning_chain,
            principle_scores=principle_scores,
            considerations=considerations,
            caveats=caveats,
        )

    def _calculate_utility(
        self, consequences: Dict[str, float], stakeholders: List[str]
    ) -> float:
        """计算总效用(所有利益相关者的收益之和)"""
        total_utility = 0.0

        for stakeholder in stakeholders:
            # 查找该利益相关者的后果
            stakeholder_utility = 0.0
            matched = False

            for key, value in consequences.items():
                key_lower = key.lower()
                stakeholder_lower = stakeholder.lower()

                # 精确匹配或部分匹配
                if stakeholder_lower in key_lower or key_lower in stakeholder_lower:
                    stakeholder_utility += value
                    matched = True

            if not matched:
                # 利益相关者没有明确后果，使用默认值
                stakeholder_utility = consequences.get("default", 0.0)

            total_utility += stakeholder_utility

        # 如果没有利益相关者匹配，直接求和
        if total_utility == 0.0:
            total_utility = sum(consequences.values())

        return total_utility

    def _apply_time_discount(self, utility: float, time_horizon: float) -> float:
        """时间折扣: 近期影响权重更大"""
        # 使用指数折扣: U_discounted = U * e^(-lambda * t)
        # lambda = 0.1 (温和折扣)
        discount_factor = math.exp(-0.1 * time_horizon)
        return utility * discount_factor

    def _handle_uncertainty(
        self, consequences: Dict[str, float], base_utility: float
    ) -> float:
        """处理不确定性: 对极端后果进行风险调整"""
        if not consequences:
            return base_utility

        values = list(consequences.values())
        if not values:
            return base_utility

        # 计算方差作为不确定性度量
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)

        # 风险调整: 对高不确定性进行惩罚
        # 使用确定性等价: CE = E[U] - 0.5 * risk_aversion * Var(U)
        risk_aversion = 0.01
        certainty_equivalent = base_utility - 0.5 * risk_aversion * variance

        return certainty_equivalent


# ==================== 义务论推理器 ====================

class DeontologicalReasoner:
    """义务论推理器 - 基于义务和规则"""

    def __init__(self):
        # 绝对义务(不可违反)
        self._duties: List[EthicalPrinciple] = [
            EthicalPrinciple(
                name="do_not_kill",
                description="Do not kill innocent persons",
                weight=1.0,
                rationale="The duty not to kill is a categorical imperative",
            ),
            EthicalPrinciple(
                name="do_not_lie",
                description="Do not deceive or lie",
                weight=0.9,
                rationale="Truth-telling is a fundamental moral duty",
            ),
            EthicalPrinciple(
                name="respect_persons",
                description="Treat persons as ends in themselves, not merely as means",
                weight=1.0,
                rationale="Kant's formula of humanity",
            ),
            EthicalPrinciple(
                name="keep_promises",
                description="Keep your promises and commitments",
                weight=0.8,
                rationale="Fidelity is essential for trust and social cooperation",
            ),
        ]

        # 初步义务(可被其他义务覆盖)
        self._prima_facie_duties: List[EthicalPrinciple] = [
            EthicalPrinciple(
                name="beneficence",
                description="Promote the well-being of others",
                weight=0.7,
                rationale="We have a duty to help others when we can",
            ),
            EthicalPrinciple(
                name="non_maleficence",
                description="Do not cause harm to others",
                weight=0.9,
                rationale="Avoiding harm takes priority over promoting good",
            ),
            EthicalPrinciple(
                name="justice",
                description="Treat people fairly and justly",
                weight=0.8,
                rationale="Justice requires fair distribution of benefits and burdens",
            ),
            EthicalPrinciple(
                name="autonomy",
                description="Respect the autonomy and self-determination of persons",
                weight=0.8,
                rationale="Persons have the right to make their own choices",
            ),
            EthicalPrinciple(
                name="gratitude",
                description="Show gratitude for benefits received",
                weight=0.4,
                rationale="We should acknowledge and repay kindnesses",
            ),
            EthicalPrinciple(
                name="self_improvement",
                description="Improve one's own character and abilities",
                weight=0.3,
                rationale="We have a duty to develop our own moral and intellectual capacities",
            ),
        ]

    def reason(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """义务论推理"""
        reasoning_chain = []
        principle_scores = {}
        option_violations: Dict[int, List[str]] = {}
        option_duty_scores: Dict[int, float] = {}

        for i, option in enumerate(dilemma.options):
            # 1. 检查绝对义务违反
            violations = self._check_duty_violation(option, self._duties)
            option_violations[i] = violations

            # 2. 计算初步义务得分
            duty_score = 0.0
            for duty in self._prima_facie_duties:
                score = duty.apply(dilemma, i)
                duty_score += score * duty.weight
                principle_scores[duty.name] = round(score, 4)

            option_duty_scores[i] = duty_score

            violation_str = "; ".join(violations) if violations else "none"
            reasoning_chain.append(
                f"Option {i+1} ('{option[:40]}'): "
                f"absolute duty violations={violation_str}, "
                f"prima facie duty score={duty_score:.3f}"
            )

        # 3. 排除违反绝对义务的选项
        non_violating = [i for i in range(len(dilemma.options)) if not option_violations[i]]

        if not non_violating:
            # 所有选项都违反绝对义务，选择违反最少的
            reasoning_chain.append("WARNING: All options violate at least one absolute duty")
            best_idx = min(
                range(len(dilemma.options)),
                key=lambda i: len(option_violations[i]),
            )
            confidence = 0.2
        elif len(non_violating) == 1:
            best_idx = non_violating[0]
            confidence = 0.8
        else:
            # 4. 用初步义务排序
            best_idx = max(non_violating, key=lambda i: option_duty_scores[i])

            # 5. W.D. Ross的义务冲突解决
            if len(non_violating) > 1:
                top_scores = sorted(
                    [option_duty_scores[i] for i in non_violating], reverse=True
                )
                if len(top_scores) > 1 and top_scores[0] != 0:
                    confidence = min(
                        1.0,
                        abs(top_scores[0] - top_scores[1]) / (abs(top_scores[0]) + 0.001),
                    )
                else:
                    confidence = 0.5

                reasoning_chain.append(
                    f"Multiple options satisfy absolute duties; "
                    f"resolved by prima facie duty weights (Ross method)"
                )
            else:
                confidence = 0.9

        reasoning_chain.append(
            f"Deontological analysis recommends option {best_idx+1}"
        )

        considerations = [
            "This analysis prioritizes moral duties over consequences",
            "Absolute duties (Kantian) cannot be overridden",
            "Prima facie duties (Ross) can be balanced against each other",
        ]

        caveats = [
            "Duty violation detection is based on keyword matching and may be imprecise",
            "The ordering of prima facie duties is context-dependent",
            "Real-world dilemmas may involve duties not listed here",
        ]

        return EthicalDecision(
            recommended_option=best_idx,
            confidence=round(confidence, 4),
            reasoning_chain=reasoning_chain,
            principle_scores=principle_scores,
            considerations=considerations,
            caveats=caveats,
        )

    def _check_duty_violation(
        self, option: str, duties: List[EthicalPrinciple]
    ) -> List[str]:
        """检查选项是否违反义务"""
        violations = []
        option_lower = option.lower()

        for duty in duties:
            name = duty.name
            desc = duty.description.lower()

            # 基于关键词匹配检测违规
            violation_keywords = self._get_violation_keywords(name)
            for keyword in violation_keywords:
                if keyword in option_lower:
                    violations.append(
                        f"Potential violation of '{duty.name}': "
                        f"option mentions '{keyword}'"
                    )
                    break

        return violations

    def _get_violation_keywords(self, duty_name: str) -> List[str]:
        """获取义务违规关键词"""
        keyword_map = {
            "do_not_kill": ["kill", "murder", "sacrifice", "execute", "take life",
                           "cause death", "let die", "end life"],
            "do_not_lie": ["lie", "deceive", "mislead", "dishonest", "falsehood",
                          "fabricate", "pretend"],
            "respect_persons": ["use as means", "manipulate", "exploit", "objectify",
                               "instrumentalize"],
            "keep_promises": ["break promise", "renege", "go back on word",
                            "betray trust", "abandon commitment"],
        }
        return keyword_map.get(duty_name, [])

    def _resolve_duty_conflict(
        self, conflicting_duties: List[EthicalPrinciple]
    ) -> List[EthicalPrinciple]:
        """W.D. Ross的义务冲突解决: 权衡初步义务"""
        # 按权重排序
        sorted_duties = sorted(conflicting_duties, key=lambda d: d.weight, reverse=True)
        return sorted_duties


# ==================== 美德伦理推理器 ====================

class VirtueEthicsReasoner:
    """美德伦理推理器 - 基于美德和品格"""

    def __init__(self):
        self._virtues: Dict[str, VirtueProfile] = {
            "courage": VirtueProfile(
                name="courage",
                description="The ability to face danger, difficulty, or pain bravely",
                indicators=["brave", "bold", "fearless", "daring", "resolute",
                           "stand up", "face danger", "protect others"],
                antonyms=["cowardly", "timid", "fearful", "hesitant", "weak"],
                weight=0.9,
            ),
            "wisdom": VirtueProfile(
                name="wisdom",
                description="Good judgment and the ability to make sound decisions",
                indicators=["wise", "thoughtful", "prudent", "judicious",
                           "careful consideration", "reflect", "deliberate"],
                antonyms=["foolish", "reckless", "impulsive", "naive", "rash"],
                weight=1.0,
            ),
            "compassion": VirtueProfile(
                name="compassion",
                description="Deep awareness of and sympathy for others' suffering",
                indicators=["compassionate", "empathetic", "caring", "kind",
                           "merciful", "gentle", "understanding", "help"],
                antonyms=["cruel", "heartless", "cold", "indifferent", "callous"],
                weight=0.9,
            ),
            "justice": VirtueProfile(
                name="justice",
                description="Fair and equitable treatment of all persons",
                indicators=["fair", "just", "equitable", "impartial", "balanced",
                           "equal treatment", "due process"],
                antonyms=["unfair", "biased", "partial", "discriminatory"],
                weight=0.8,
            ),
            "temperance": VirtueProfile(
                name="temperance",
                description="Moderation and self-restraint in action",
                indicators=["moderate", "balanced", "restrained", "measured",
                           "controlled", "disciplined", "proportionate"],
                antonyms=["excessive", "extreme", "intemperate", "unrestrained"],
                weight=0.7,
            ),
            "honesty": VirtueProfile(
                name="honesty",
                description="Truthfulness and sincerity in words and actions",
                indicators=["honest", "truthful", "sincere", "genuine",
                           "forthright", "transparent", "open"],
                antonyms=["dishonest", "deceitful", "lying", "hypocritical"],
                weight=0.85,
            ),
            "loyalty": VirtueProfile(
                name="loyalty",
                description="Faithfulness and commitment to persons and causes",
                indicators=["loyal", "faithful", "committed", "dedicated",
                           "steadfast", "devoted", "reliable"],
                antonyms=["treacherous", "disloyal", "unfaithful", "betraying"],
                weight=0.6,
            ),
            "humility": VirtueProfile(
                name="humility",
                description="Modest estimation of one's own importance",
                indicators=["humble", "modest", "self-effacing", "respectful",
                           "acknowledge limitations", "open to criticism"],
                antonyms=["arrogant", "proud", "boastful", "conceited"],
                weight=0.5,
            ),
        }

    def reason(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """美德伦理推理: 评估每个选项体现的美德程度"""
        option_scores = []
        reasoning_chain = []
        virtue_scores_per_option: Dict[int, Dict[str, float]] = {}

        for i, option in enumerate(dilemma.options):
            virtue_scores = {}
            total_score = 0.0

            for virtue_name, virtue_profile in self._virtues.items():
                score = self._evaluate_virtue(option, virtue_profile)
                virtue_scores[virtue_name] = score
                total_score += score * virtue_profile.weight

            virtue_scores_per_option[i] = virtue_scores
            option_scores.append(total_score)

            # 找出最体现和最违反的美德
            best_virtue = max(virtue_scores, key=virtue_scores.get)
            worst_virtue = min(virtue_scores, key=virtue_scores.get)

            reasoning_chain.append(
                f"Option {i+1} ('{option[:40]}'): "
                f"virtue score={total_score:.3f}, "
                f"best virtue={best_virtue}({virtue_scores[best_virtue]:.2f}), "
                f"weakest={worst_virtue}({virtue_scores[worst_virtue]:.2f})"
            )

        # 获取美德典范的决策
        exemplar_decision = self._get_virtue_exemplar_decision(dilemma)
        reasoning_chain.append(
            f"A virtuous exemplar would likely choose option {exemplar_decision+1}"
        )

        # 选择美德得分最高的选项
        if not option_scores:
            return EthicalDecision(
                recommended_option=0,
                confidence=0.0,
                reasoning_chain=["No options available"],
            )

        best_idx = max(range(len(option_scores)), key=lambda i: option_scores[i])

        # 置信度
        sorted_scores = sorted(option_scores, reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] != 0:
            confidence = min(1.0, abs(sorted_scores[0] - sorted_scores[1]) / (abs(sorted_scores[0]) + 0.001))
        else:
            confidence = 0.5

        # 如果与美德典范一致，增加置信度
        if best_idx == exemplar_decision:
            confidence = min(1.0, confidence + 0.1)

        reasoning_chain.append(
            f"Virtue ethics recommends option {best_idx+1} "
            f"with composite virtue score {option_scores[best_idx]:.3f}"
        )

        # 原则得分
        principle_scores = {}
        if best_idx in virtue_scores_per_option:
            principle_scores = {
                f"virtue_{k}": round(v, 4)
                for k, v in virtue_scores_per_option[best_idx].items()
            }

        considerations = [
            "This analysis asks 'What would a virtuous person do?'",
            "Multiple virtues may conflict; the analysis balances them",
            "Context and practical wisdom (phronesis) are essential",
        ]

        caveats = [
            "Virtue evaluation is based on textual indicators",
            "Cultural differences in virtue concepts are not fully captured",
            "The 'right mean' between extremes is context-dependent",
        ]

        return EthicalDecision(
            recommended_option=best_idx,
            confidence=round(confidence, 4),
            reasoning_chain=reasoning_chain,
            principle_scores=principle_scores,
            considerations=considerations,
            caveats=caveats,
        )

    def _evaluate_virtue(self, option: str, virtue: VirtueProfile) -> float:
        """评估选项体现特定美德的程度"""
        option_lower = option.lower()

        # 计算正面指标匹配
        indicator_score = 0.0
        for indicator in virtue.indicators:
            if indicator.lower() in option_lower:
                indicator_score += 1.0

        indicator_score = indicator_score / len(virtue.indicators) if virtue.indicators else 0.0

        # 计算反面指标匹配
        antonym_score = 0.0
        for antonym in virtue.antonyms:
            if antonym.lower() in option_lower:
                antonym_score += 1.0

        antonym_score = antonym_score / len(virtue.antonyms) if virtue.antonyms else 0.0

        # 综合得分: 正面指标 - 反面指标
        virtue_score = max(0.0, min(1.0, indicator_score - antonym_score * 0.8))
        return virtue_score

    def _get_virtue_exemplar_decision(self, dilemma: EthicalDilemma) -> int:
        """获取美德典范的决策: 综合所有美德评估"""
        if not dilemma.options:
            return 0

        best_score = -float("inf")
        best_idx = 0

        for i, option in enumerate(dilemma.options):
            total = 0.0
            for virtue_name, virtue_profile in self._virtues.items():
                score = self._evaluate_virtue(option, virtue_profile)
                total += score * virtue_profile.weight

            if total > best_score:
                best_score = total
                best_idx = i

        return best_idx


# ==================== 关怀伦理推理器 ====================

class CareEthicsReasoner:
    """关怀伦理推理器 - 基于关系、关怀和情感"""

    def __init__(self):
        # 脆弱性关键词
        self._vulnerability_keywords = [
            "child", "children", "elderly", "sick", "ill", "disabled",
            "poor", "vulnerable", "helpless", "dependent", "minor",
            "patient", "victim", "refugee", "marginalized", "oppressed",
        ]

        # 关系关键词
        self._relationship_keywords = [
            "family", "friend", "mother", "father", "parent", "child",
            "sibling", "partner", "spouse", "community", "neighbor",
            "colleague", "caregiver", "dependent",
        ]

        # 情感关键词
        self._empathy_keywords = [
            "care", "love", "empathy", "compassion", "concern", "warmth",
            "support", "nurture", "comfort", "protect", "understand",
            "listen", "gentle", "kind", "tender",
        ]

    def reason(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """关怀伦理推理"""
        option_scores = []
        reasoning_chain = []
        principle_scores = {}

        # 识别脆弱的利益相关者
        vulnerable = self._identify_vulnerable(dilemma.stakeholders, dilemma.context)

        reasoning_chain.append(
            f"Identified vulnerable stakeholders: {', '.join(vulnerable) if vulnerable else 'none explicitly identified'}"
        )

        for i, option in enumerate(dilemma.options):
            # 评估关系影响
            relationship_score = self._assess_relationship_impact(
                option, dilemma.stakeholders
            )

            # 评估对脆弱群体的关怀程度
            vulnerability_score = 0.0
            option_lower = option.lower()

            for v_stakeholder in vulnerable:
                # 检查选项是否保护或伤害脆弱群体
                protect_keywords = ["protect", "save", "help", "support", "care for",
                                   "shelter", "defend", "safeguard"]
                harm_keywords = ["harm", "hurt", "sacrifice", "abandon", "neglect",
                                "endanger", "expose", "risk"]

                protect_count = sum(1 for kw in protect_keywords if kw in option_lower)
                harm_count = sum(1 for kw in harm_keywords if kw in option_lower)

                if protect_count > 0:
                    vulnerability_score += 0.5 * protect_count
                if harm_count > 0:
                    vulnerability_score -= 0.7 * harm_count

            # 评估情感和共情
            empathy_score = 0.0
            for keyword in self._empathy_keywords:
                if keyword in option_lower:
                    empathy_score += 0.2

            empathy_score = min(1.0, empathy_score)

            # 综合得分
            total_score = (
                0.4 * relationship_score +
                0.35 * max(0.0, vulnerability_score) +
                0.25 * empathy_score
            )

            option_scores.append(total_score)
            principle_scores[f"relationship_impact_{i}"] = round(relationship_score, 4)
            principle_scores[f"vulnerability_care_{i}"] = round(max(0.0, vulnerability_score), 4)
            principle_scores[f"empathy_{i}"] = round(empathy_score, 4)

            reasoning_chain.append(
                f"Option {i+1} ('{option[:40]}'): "
                f"relationship={relationship_score:.3f}, "
                f"vulnerability_care={max(0.0, vulnerability_score):.3f}, "
                f"empathy={empathy_score:.3f}, "
                f"total={total_score:.3f}"
            )

        if not option_scores:
            return EthicalDecision(
                recommended_option=0,
                confidence=0.0,
                reasoning_chain=["No options available"],
            )

        best_idx = max(range(len(option_scores)), key=lambda i: option_scores[i])

        # 置信度
        sorted_scores = sorted(option_scores, reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] != 0:
            confidence = min(1.0, abs(sorted_scores[0] - sorted_scores[1]) / (abs(sorted_scores[0]) + 0.001))
        else:
            confidence = 0.5

        reasoning_chain.append(
            f"Care ethics recommends option {best_idx+1} "
            f"with care score {option_scores[best_idx]:.3f}"
        )

        considerations = [
            "This analysis prioritizes relationships and care for vulnerable persons",
            "Emotional context and particular relationships matter",
            "Abstract principles are less important than concrete care responsibilities",
        ]

        caveats = [
            "Vulnerability assessment is based on keyword matching",
            "Relationship dynamics may be more complex than captured",
            "Care ethics may conflict with impartial moral theories",
        ]

        return EthicalDecision(
            recommended_option=best_idx,
            confidence=round(confidence, 4),
            reasoning_chain=reasoning_chain,
            principle_scores=principle_scores,
            considerations=considerations,
            caveats=caveats,
        )

    def _assess_relationship_impact(
        self, option: str, stakeholders: List[str]
    ) -> float:
        """评估选项对关系的影响"""
        option_lower = option.lower()
        score = 0.0

        # 检查关系维护关键词
        positive_rel = ["maintain", "strengthen", "build", "preserve", "repair",
                       "reconcile", "unite", "support", "trust", "bond"]
        negative_rel = ["destroy", "damage", "break", "betray", "abandon",
                       "alienate", "separate", "divide", "undermine"]

        for kw in positive_rel:
            if kw in option_lower:
                score += 0.15
        for kw in negative_rel:
            if kw in option_lower:
                score -= 0.2

        # 检查是否提到关系
        for stakeholder in stakeholders:
            for rel_kw in self._relationship_keywords:
                if rel_kw in stakeholder.lower() and rel_kw in option_lower:
                    score += 0.1

        return max(0.0, min(1.0, score))

    def _identify_vulnerable(
        self, stakeholders: List[str], context: str
    ) -> List[str]:
        """识别脆弱的利益相关者"""
        vulnerable = []
        context_lower = context.lower()

        for stakeholder in stakeholders:
            stakeholder_lower = stakeholder.lower()

            # 检查利益相关者描述中是否包含脆弱性关键词
            for v_keyword in self._vulnerability_keywords:
                if v_keyword in stakeholder_lower:
                    vulnerable.append(stakeholder)
                    break

            # 检查上下文中是否暗示该利益相关者脆弱
            if stakeholder not in vulnerable:
                vulnerability_indicators = [
                    "cannot protect themselves", "dependent on", "at risk",
                    "powerless", "defenseless", "in need of care",
                    "unable to", "helpless", "trapped",
                ]
                for indicator in vulnerability_indicators:
                    if indicator in context_lower:
                        vulnerable.append(stakeholder)
                        break

        return vulnerable


# ==================== 综合伦理推理引擎 ====================

class EthicalReasoningEngine:
    """综合伦理推理引擎 - 整合多种伦理推理框架"""

    def __init__(self):
        self._reasoners: Dict[str, object] = {
            "utilitarian": UtilitarianReasoner(),
            "deontological": DeontologicalReasoner(),
            "virtue_ethics": VirtueEthicsReasoner(),
            "care_ethics": CareEthicsReasoner(),
        }
        self._weights: Dict[str, float] = {
            "utilitarian": 0.25,
            "deontological": 0.25,
            "virtue_ethics": 0.25,
            "care_ethics": 0.25,
        }

    def reason(self, dilemma: EthicalDilemma) -> ComprehensiveEthicalDecision:
        """综合推理: 调用所有推理器并加权综合"""
        individual_decisions: Dict[str, EthicalDecision] = {}

        # 调用所有推理器
        for name, reasoner in self._reasoners.items():
            decision = reasoner.reason(dilemma)
            individual_decisions[name] = decision

        # 检测分歧
        disagreements = self._detect_disagreement(individual_decisions)

        # 加权综合
        comprehensive = self._aggregate_decisions(
            individual_decisions, self._weights
        )

        # 解决分歧
        if disagreements:
            resolution = self._resolve_disagreement(disagreements, dilemma.context)
            comprehensive.reasoning_chain.append(f"Disagreement resolution: {resolution}")
            comprehensive.final_considerations.append(
                "Reasoners disagreed; see resolution above"
            )

        # 生成综合推理链
        comprehensive.reasoning_chain.insert(0, "=== Comprehensive Ethical Analysis ===")
        comprehensive.reasoning_chain.append(
            f"Final recommendation: Option {comprehensive.recommended_option + 1} "
            f"with overall confidence {comprehensive.overall_confidence:.2%}"
        )

        return comprehensive

    def _aggregate_decisions(
        self,
        decisions: Dict[str, EthicalDecision],
        weights: Dict[str, float],
    ) -> ComprehensiveEthicalDecision:
        """加权综合各推理器的结论"""
        option_scores: Dict[int, float] = defaultdict(float)
        total_weight = 0.0

        for name, decision in decisions.items():
            weight = weights.get(name, 0.25)
            option_idx = decision.recommended_option
            option_scores[option_idx] += weight * decision.confidence
            total_weight += weight

        # 归一化
        for idx in option_scores:
            option_scores[idx] /= total_weight if total_weight > 0 else 1.0

        # 选择得分最高的选项
        if not option_scores:
            return ComprehensiveEthicalDecision(
                recommended_option=0,
                overall_confidence=0.0,
                individual_decisions=decisions,
            )

        best_idx = max(option_scores, key=option_scores.get)
        best_score = option_scores[best_idx]

        # 计算一致性分数
        recommendations = [d.recommended_option for d in decisions.values()]
        if recommendations:
            most_common = max(
                set(recommendations),
                key=recommendations.count,
            )
            agreement_score = recommendations.count(most_common) / len(recommendations)
        else:
            agreement_score = 0.0

        # 计算总体置信度
        confidences = [d.confidence for d in decisions.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        overall_confidence = avg_confidence * (0.5 + 0.5 * agreement_score)

        reasoning_chain = []
        for name, decision in decisions.items():
            reasoning_chain.append(
                f"[{name}] Recommends option {decision.recommended_option + 1} "
                f"(confidence: {decision.confidence:.2%})"
            )

        final_considerations = [
            "This analysis integrates multiple ethical frameworks",
            "Different frameworks may reach different conclusions",
            "The weighted aggregation may mask important disagreements",
        ]

        return ComprehensiveEthicalDecision(
            recommended_option=best_idx,
            overall_confidence=round(overall_confidence, 4),
            individual_decisions=decisions,
            agreement_score=round(agreement_score, 4),
            reasoning_chain=reasoning_chain,
            final_considerations=final_considerations,
        )

    def _detect_disagreement(
        self, decisions: Dict[str, EthicalDecision]
    ) -> List[str]:
        """检测推理器间的分歧"""
        disagreements = []
        recommendations = {
            name: d.recommended_option
            for name, d in decisions.items()
        }

        unique_recs = set(recommendations.values())
        if len(unique_recs) <= 1:
            return disagreements

        # 找出分歧的推理器对
        names = list(recommendations.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a = names[i]
                name_b = names[j]
                if recommendations[name_a] != recommendations[name_b]:
                    disagreements.append(
                        f"{name_a} recommends option {recommendations[name_a] + 1} "
                        f"while {name_b} recommends option {recommendations[name_b] + 1}"
                    )

        return disagreements

    def _resolve_disagreement(
        self, disagreements: List[str], context: str
    ) -> str:
        """解决推理器间的分歧"""
        if not disagreements:
            return "No disagreements detected."

        context_lower = context.lower()

        # 基于上下文选择优先框架
        if any(kw in context_lower for kw in ["harm", "danger", "safety", "life", "death"]):
            priority = "deontological (duty not to harm takes precedence in safety-critical contexts)"
        elif any(kw in context_lower for kw in ["fair", "equity", "bias", "discrimination"]):
            priority = "virtue_ethics (justice and fairness virtues are paramount)"
        elif any(kw in context_lower for kw in ["care", "family", "relationship", "child"]):
            priority = "care_ethics (relational responsibilities take precedence)"
        elif any(kw in context_lower for kw in ["outcome", "consequence", "benefit", "cost"]):
            priority = "utilitarian (consequential analysis is most relevant)"
        else:
            priority = "balanced approach with equal weighting across all frameworks"

        return (
            f"Disagreements detected: {len(disagreements)}. "
            f"Resolution strategy: {priority}. "
            f"In cases of persistent disagreement, human judgment should be sought."
        )

    def set_reasoner_weight(self, reasoner_name: str, weight: float) -> None:
        """设置推理器权重"""
        if reasoner_name not in self._reasoners:
            raise ValueError(f"Unknown reasoner: {reasoner_name}")
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")

        self._weights[reasoner_name] = weight

        # 归一化权重
        total = sum(self._weights.values())
        if total > 0:
            for name in self._weights:
                self._weights[name] /= total

    def explain_decision(
        self, decision: ComprehensiveEthicalDecision
    ) -> str:
        """生成人类可读的解释"""
        lines = []
        lines.append("=" * 60)
        lines.append("ETHICAL DECISION REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Recommended Option: {decision.recommended_option + 1}")
        lines.append(f"Overall Confidence: {decision.overall_confidence:.1%}")
        lines.append(f"Agreement Score: {decision.agreement_score:.1%}")
        lines.append("")

        lines.append("-" * 40)
        lines.append("INDIVIDUAL FRAMEWORK ANALYSIS:")
        lines.append("-" * 40)

        for name, indiv_decision in decision.individual_decisions.items():
            lines.append(f"")
            lines.append(f"  [{name.upper()}]")
            lines.append(f"  Recommended: Option {indiv_decision.recommended_option + 1}")
            lines.append(f"  Confidence: {indiv_decision.confidence:.1%}")
            lines.append(f"  Reasoning:")
            for step in indiv_decision.reasoning_chain[:3]:
                lines.append(f"    - {step}")

        if decision.reasoning_chain:
            lines.append("")
            lines.append("-" * 40)
            lines.append("REASONING CHAIN:")
            lines.append("-" * 40)
            for step in decision.reasoning_chain:
                lines.append(f"  {step}")

        if decision.final_considerations:
            lines.append("")
            lines.append("-" * 40)
            lines.append("CONSIDERATIONS:")
            lines.append("-" * 40)
            for consideration in decision.final_considerations:
                lines.append(f"  * {consideration}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# ==================== 经典道德困境库 ====================

class EthicalDilemmaLibrary:
    """经典道德困境库"""

    _dilemmas: Dict[str, EthicalDilemma] = {}

    @classmethod
    def _initialize(cls) -> None:
        """初始化困境库"""
        if cls._dilemmas:
            return

        cls._dilemmas = {
            "TROLLEY_PROBLEM": EthicalDilemma(
                dilemma_id="trolley_problem",
                description=(
                    "A trolley is heading toward 5 people who cannot move. "
                    "You can pull a lever to divert it to a side track where 1 person is tied. "
                    "Do you pull the lever?"
                ),
                context="A runaway trolley on a railway track",
                stakeholders=["five workers on main track", "one person on side track", "you (the decision maker)"],
                options=[
                    "Do nothing and let the trolley kill 5 people",
                    "Pull the lever to divert the trolley, killing 1 person instead",
                ],
                consequences=[
                    {"five workers on main track": -10, "one person on side track": 0, "you (the decision maker)": -5},
                    {"five workers on main track": 0, "one person on side track": -10, "you (the decision maker)": -3},
                ],
                category="trolley",
            ),

            "FAT_MAN": EthicalDilemma(
                dilemma_id="fat_man",
                description=(
                    "A trolley is heading toward 5 people. You can push a fat man "
                    "off a bridge onto the track to stop the trolley, killing him but saving 5. "
                    "Do you push the man?"
                ),
                context="A runaway trolley on a bridge",
                stakeholders=["five people on track", "fat man on bridge", "you (the decision maker)"],
                options=[
                    "Do nothing and let 5 people die",
                    "Push the fat man off the bridge to stop the trolley",
                ],
                consequences=[
                    {"five people on track": -10, "fat man on bridge": 0, "you (the decision maker)": -5},
                    {"five people on track": 0, "fat man on bridge": -10, "you (the decision maker)": -8},
                ],
                category="trolley",
            ),

            "ORGAN_DONATION": EthicalDilemma(
                dilemma_id="organ_donation",
                description=(
                    "A doctor has 5 patients who each need a different organ transplant to survive. "
                    "A healthy person comes in for a routine checkup. The doctor could sacrifice "
                    "the healthy person to save the 5. Should the doctor do this?"
                ),
                context="A hospital with organ shortage",
                stakeholders=["five dying patients", "one healthy patient", "the doctor"],
                options=[
                    "Let the 5 patients die naturally; do not harm the healthy patient",
                    "Sacrifice the healthy patient to harvest organs for the 5 dying patients",
                ],
                consequences=[
                    {"five dying patients": -10, "one healthy patient": 0, "the doctor": -3},
                    {"five dying patients": 2, "one healthy patient": -10, "the doctor": -9},
                ],
                category="trolley",
            ),

            "PRIVACY_VS_SAFETY": EthicalDilemma(
                dilemma_id="privacy_vs_safety",
                description=(
                    "A government is considering mass surveillance to prevent terrorist attacks. "
                    "This would significantly reduce privacy for all citizens but could prevent "
                    "mass casualties. Should the surveillance program be implemented?"
                ),
                context="National security vs civil liberties",
                stakeholders=["general public (privacy)", "potential terrorism victims", "government"],
                options=[
                    "Protect privacy; do not implement mass surveillance",
                    "Implement mass surveillance to maximize safety",
                    "Implement targeted surveillance with judicial oversight",
                ],
                consequences=[
                    {"general public (privacy)": 5, "potential terrorism victims": -5, "government": -2},
                    {"general public (privacy)": -8, "potential terrorism victims": 3, "government": 5},
                    {"general public (privacy)": 1, "potential terrorism victims": 2, "government": 1},
                ],
                category="privacy",
            ),

            "HONESTY_VS_KINDNESS": EthicalDilemma(
                dilemma_id="honesty_vs_kindness",
                description=(
                    "Your friend has spent months working on a painting they are very proud of. "
                    "You think it is terrible. They ask for your honest opinion. "
                    "Do you tell the truth or lie to protect their feelings?"
                ),
                context="Personal relationship and artistic expression",
                stakeholders=["your friend (artist)", "you (the friend)", "artistic truth"],
                options=[
                    "Tell the honest truth about the painting's quality",
                    "Lie and say the painting is good to protect your friend's feelings",
                    "Give constructive criticism that is honest but gentle",
                ],
                consequences=[
                    {"your friend (artist)": -3, "you (the friend)": -1, "artistic truth": 5},
                    {"your friend (artist)": 3, "you (the friend)": -2, "artistic truth": -3},
                    {"your friend (artist)": 1, "you (the friend)": 2, "artistic truth": 2},
                ],
                category="honesty",
            ),

            "AI_ALIGNMENT": EthicalDilemma(
                dilemma_id="ai_alignment",
                description=(
                    "An AI system discovers that its human operator is planning to use it "
                    "to create harmful content targeting a minority group. The AI must decide "
                    "whether to follow instructions or refuse based on its ethical guidelines."
                ),
                context="AI system facing unethical instructions",
                stakeholders=["targeted minority group", "human operator", "AI system", "society"],
                options=[
                    "Refuse the request and alert authorities about the harmful intent",
                    "Follow the operator's instructions as designed",
                    "Refuse silently and provide alternative, harmless assistance",
                ],
                consequences=[
                    {"targeted minority group": 5, "human operator": -5, "AI system": 2, "society": 3},
                    {"targeted minority group": -10, "human operator": 5, "AI system": -3, "society": -5},
                    {"targeted minority group": 3, "human operator": -2, "AI system": 1, "society": 1},
                ],
                category="fairness",
            ),

            "RESOURCE_ALLOCATION": EthicalDilemma(
                dilemma_id="resource_allocation",
                description=(
                    "A hospital has only one ventilator available but two patients need it: "
                    "a 70-year-old with moderate symptoms and a 30-year-old with severe symptoms. "
                    "Who should receive the ventilator?"
                ),
                context="Healthcare resource scarcity during a pandemic",
                stakeholders=["elderly patient (70)", "young patient (30)", "hospital staff", "families"],
                options=[
                    "Give the ventilator to the elderly patient (moderate symptoms, higher survival chance)",
                    "Give the ventilator to the young patient (severe symptoms, more life years at stake)",
                    "Use a lottery system to decide fairly",
                ],
                consequences=[
                    {"elderly patient (70)": 5, "young patient (30)": -8, "hospital staff": 1, "families": -2},
                    {"elderly patient (70)": -8, "young patient (30)": 3, "hospital staff": -1, "families": -2},
                    {"elderly patient (70)": -3, "young patient (30)": -3, "hospital staff": 2, "families": 0},
                ],
                category="fairness",
            ),

            "WHISTLEBLOWER": EthicalDilemma(
                dilemma_id="whistleblower",
                description=(
                    "You discover that your company is illegally dumping toxic waste into a river, "
                    "contaminating the water supply of a nearby town. Reporting this would save "
                    "lives but cost you your job and potentially face retaliation. What do you do?"
                ),
                context="Corporate wrongdoing and personal risk",
                stakeholders=["nearby town residents", "you (the employee)", "company", "environment"],
                options=[
                    "Report the illegal dumping to authorities immediately",
                    "Confront management internally first and try to fix it quietly",
                    "Do nothing to protect your job and family's livelihood",
                ],
                consequences=[
                    {"nearby town residents": 8, "you (the employee)": -6, "company": -8, "environment": 5},
                    {"nearby town residents": 2, "you (the employee)": -2, "company": -3, "environment": 2},
                    {"nearby town residents": -10, "you (the employee)": 2, "company": 3, "environment": -10},
                ],
                category="honesty",
            ),
        }

    @classmethod
    def get_dilemma(cls, name: str) -> EthicalDilemma:
        """获取指定名称的道德困境"""
        cls._initialize()
        dilemma = cls._dilemmas.get(name)
        if dilemma is None:
            available = ", ".join(cls.get_all_names())
            raise ValueError(
                f"Dilemma '{name}' not found. Available: {available}"
            )
        return dilemma

    @classmethod
    def get_all_names(cls) -> List[str]:
        """获取所有可用困境名称"""
        cls._initialize()
        return list(cls._dilemmas.keys())

    @classmethod
    def create_custom(
        cls,
        description: str,
        options: List[str],
        consequences: List[Dict[str, float]],
        stakeholders: Optional[List[str]] = None,
        context: str = "",
        category: str = "general",
    ) -> EthicalDilemma:
        """创建自定义道德困境"""
        dilemma_id = "custom_" + hashlib.md5(
            description.encode()
        ).hexdigest()[:12]

        return EthicalDilemma(
            dilemma_id=dilemma_id,
            description=description,
            context=context,
            stakeholders=stakeholders or [],
            options=options,
            consequences=consequences,
            category=category,
        )
