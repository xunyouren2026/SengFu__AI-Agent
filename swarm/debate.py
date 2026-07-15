"""
辩论共识系统 - Debate Consensus System

提供多Agent辩论框架，包含论点评估、逻辑谬误检测和共识达成机制。
仅使用Python标准库。
"""

import uuid
import time
import math
import enum
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict


# ============================================================
# 枚举类型
# ============================================================

class DebateStatus(enum.Enum):
    """辩论状态"""
    PENDING = "pending"          # 等待开始
    IN_PROGRESS = "in_progress"  # 进行中
    CONSENSUS_REACHED = "consensus_reached"  # 已达成共识
    DEADLOCK = "deadlock"        # 僵局
    FINISHED = "finished"        # 已结束


class FallacyType(enum.Enum):
    """逻辑谬误类型"""
    STRAW_MAN = "straw_man"                     # 稻草人谬误
    AD_HOMINEM = "ad_hominem"                   # 人身攻击
    APPEAL_TO_AUTHORITY = "appeal_to_authority" # 诉诸权威
    SLIPPERY_SLOPE = "slippery_slope"           # 滑坡谬误
    RED_HERRING = "red_herring"                 # 红鲱鱼谬误
    CIRCULAR_REASONING = "circular_reasoning"   # 循环论证
    FALSE_DICHOTOMY = "false_dichotomy"         # 虚假二分法
    BANDWAGON = "bandwagon"                     # 从众谬误


# ============================================================
# 数据模型
# ============================================================

@dataclass
class DebateTopic:
    """辩论主题"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    context: str = ""
    criteria: List[str] = field(default_factory=list)
    max_rounds: int = 5

    def __post_init__(self):
        if not self.criteria:
            self.criteria = [
                "logical_coherence",      # 逻辑连贯性
                "evidence_quality",       # 证据质量
                "relevance",              # 相关性
                "completeness",           # 完整性
            ]


@dataclass
class DebateArgument:
    """辩论论点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    round: int = 0
    content: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    supporting_args: List[str] = field(default_factory=list)
    refuting_args: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    fallacies: List[Tuple[FallacyType, float]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateSession:
    """辩论会话"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: Optional[DebateTopic] = None
    participants: List[str] = field(default_factory=list)
    arguments: List[DebateArgument] = field(default_factory=list)
    round: int = 0
    status: DebateStatus = DebateStatus.PENDING
    consensus_value: Optional[str] = None
    consensus_confidence: float = 0.0
    winner: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ============================================================
# 谬误检测器
# ============================================================

class FallacyDetector:
    """逻辑谬误检测器 - 基于关键词和结构模式检测"""

    # 各类谬误的关键词模式
    FALLACY_PATTERNS: Dict[FallacyType, List[str]] = {
        FallacyType.STRAW_MAN: [
            r"you're\s+saying\s+that",
            r"so\s+you\s+believe\s+that",
            r"your\s+argument\s+is\s+basically",
            r"what\s+you're\s+really\s+saying",
            r"misrepresent",
            r"distort",
        ],
        FallacyType.AD_HOMINEM: [
            r"you\s+(are|is)\s+(stupid|ignorant|wrong|crazy|idiot|hypocrite)",
            r"people\s+like\s+you",
            r"your\s+kind",
            r"typical\s+of\s+you",
            r"instead\s+of\s+listening\s+to\s+(\w+)",
        ],
        FallacyType.APPEAL_TO_AUTHORITY: [
            r"expert\s+says",
            r"according\s+to\s+the\s+expert",
            r"famous\s+(scientist|professor|doctor)",
            r"authority\s+says",
            r"well-known\s+expert",
            r"studies\s+show\s+(without|no)\s+(citing|reference)",
        ],
        FallacyType.SLIPPERY_SLOPE: [
            r"if\s+we\s+allow\s+this",
            r"next\s+thing\s+you\s+know",
            r"it\s+will\s+inevitably\s+lead\s+to",
            r"domino\s+effect",
            r"slippery\s+slope",
            r"cascade\s+of",
            r"eventually\s+(result|lead)\s+in\s+disaster",
        ],
        FallacyType.RED_HERRING: [
            r"but\s+what\s+about",
            r"let's\s+not\s+forget\s+that",
            r"the\s+real\s+issue\s+is",
            r"that's\s+beside\s+the\s+point",
            r"changing\s+the\s+subject",
            r"instead\s+let's\s+focus\s+on",
        ],
        FallacyType.CIRCULAR_REASONING: [
            r"it\s+is\s+true\s+because\s+it\s+is",
            r"obviously\s+(true|false|correct|wrong)",
            r"self-evident",
            r"it\s+goes\s+without\s+saying",
            r"everyone\s+knows\s+that",
            r"because\s+I\s+said\s+so",
        ],
        FallacyType.FALSE_DICHOTOMY: [
            r"either\s+\.\.\.\s+or\s+\.\.\.",
            r"you're\s+either\s+with\s+us\s+or\s+against\s+us",
            r"only\s+two\s+options?",
            r"no\s+other\s+choice",
            r"must\s+choose\s+between",
            r"black\s+and\s+white",
        ],
        FallacyType.BANDWAGON: [
            r"everyone\s+(believes|thinks|agrees)",
            r"most\s+people\s+(say|think|agree)",
            r"popular\s+opinion",
            r"mainstream\s+view",
            r"the\s+majority\s+believes",
            r"nobody\s+(believes|thinks|agrees)\s+otherwise",
        ],
    }

    def detect_patterns(self, content: str) -> List[Tuple[FallacyType, float]]:
        """
        基于关键词和结构模式检测谬误。

        对文本内容进行正则匹配，返回检测到的谬误类型及其置信度。
        置信度基于匹配到的模式数量和匹配强度计算。
        """
        detected = []
        content_lower = content.lower()

        for fallacy_type, patterns in self.FALLACY_PATTERNS.items():
            match_count = 0
            total_pattern_strength = 0.0

            for pattern in patterns:
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                if matches:
                    match_count += len(matches)
                    # 模式越长越具体，置信度越高
                    pattern_strength = min(len(pattern.split(r"\s+")) / 5.0, 1.0)
                    total_pattern_strength += pattern_strength * len(matches)

            if match_count > 0:
                # 置信度 = 匹配强度 * 模式覆盖率
                coverage = min(match_count / 3.0, 1.0)
                confidence = min(total_pattern_strength * coverage, 1.0)
                # 基础置信度下限
                confidence = max(confidence, 0.2 * match_count)
                confidence = min(confidence, 1.0)
                detected.append((fallacy_type, round(confidence, 4)))

        # 按置信度降序排列
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected

    def confidence_score(self, content: str) -> float:
        """
        计算文本的整体谬误置信度。

        返回值范围 [0, 1]，越高表示文本中谬误越严重。
        使用加权平均：检测到的谬误数量和各自置信度综合计算。
        """
        detected = self.detect_patterns(content)
        if not detected:
            return 0.0

        # 加权平均置信度，数量越多惩罚越大
        total_confidence = sum(conf for _, conf in detected)
        count_penalty = 1.0 - 0.1 * (len(detected) - 1)
        count_penalty = max(count_penalty, 0.5)

        return min(total_confidence * count_penalty, 1.0)


# ============================================================
# 辩论引擎
# ============================================================

class DebateEngine:
    """辩论引擎 - 管理辩论生命周期"""

    def __init__(self):
        self._debates: Dict[str, DebateSession] = {}
        self._fallacy_detector = FallacyDetector()
        self._argument_quality_weights = {
            "logical_coherence": 0.30,
            "evidence_quality": 0.25,
            "relevance": 0.20,
            "completeness": 0.15,
            "novelty": 0.10,
        }

    def create_debate(self, topic: DebateTopic, participants: List[str]) -> str:
        """
        创建一场新的辩论。

        Args:
            topic: 辩论主题
            participants: 参与者ID列表

        Returns:
            辩论会话ID
        """
        session = DebateSession(
            topic=topic,
            participants=list(participants),
            status=DebateStatus.PENDING,
        )
        self._debates[session.id] = session
        return session.id

    def submit_argument(
        self,
        debate_id: str,
        agent_id: str,
        content: str,
        evidence: Optional[List[str]] = None,
        supporting_args: Optional[List[str]] = None,
        refuting_args: Optional[List[str]] = None,
    ) -> DebateArgument:
        """
        提交论点到指定辩论。

        自动评估论点质量并检测逻辑谬误。
        """
        session = self._get_debate(debate_id)
        if session.status not in (DebateStatus.PENDING, DebateStatus.IN_PROGRESS):
            raise ValueError(f"辩论状态不允许提交论点: {session.status.value}")
        if agent_id not in session.participants:
            raise ValueError(f"Agent {agent_id} 不是辩论参与者")

        if session.status == DebateStatus.PENDING:
            session.status = DebateStatus.IN_PROGRESS

        evidence = evidence or []
        supporting_args = supporting_args or []
        refuting_args = refuting_args or []

        # 检测谬误
        fallacies = self._fallacy_detector.detect_patterns(content)

        # 评估论点质量
        quality_score = self.evaluate_argument(
            DebateArgument(
                agent_id=agent_id,
                round=session.round,
                content=content,
                evidence=evidence,
                supporting_args=supporting_args,
                refuting_args=refuting_args,
                fallacies=fallacies,
            )
        )

        # 计算置信度：基于质量分数和谬误惩罚
        fallacy_penalty = sum(conf for _, conf in fallacies)
        confidence = max(0.0, quality_score - fallacy_penalty * 0.3)

        argument = DebateArgument(
            agent_id=agent_id,
            round=session.round,
            content=content,
            evidence=evidence,
            supporting_args=supporting_args,
            refuting_args=refuting_args,
            confidence=round(confidence, 4),
            quality_score=round(quality_score, 4),
            fallacies=fallacies,
        )

        session.arguments.append(argument)
        session.updated_at = time.time()
        return argument

    def evaluate_argument(self, argument: DebateArgument) -> float:
        """
        评估论点质量，返回 [0, 1] 范围的质量分数。

        评估维度：
        - 逻辑连贯性：内容长度和结构完整性
        - 证据质量：提供的证据数量和质量
        - 相关性：支持/反驳论点的关联度
        - 完整性：论点的全面程度
        - 新颖性：内容的独特性
        """
        scores = {}

        # 1. 逻辑连贯性 - 基于内容结构分析
        content = argument.content
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        coherence = 0.0
        if sentences:
            # 句子数量适中得分更高
            sentence_score = min(len(sentences) / 8.0, 1.0)
            # 检查逻辑连接词
            connectors = [
                "therefore", "because", "however", "moreover",
                "furthermore", "consequently", "nevertheless",
                "thus", "hence", "accordingly", "additionally",
            ]
            connector_count = sum(
                1 for s in sentences
                if any(c in s.lower() for c in connectors)
            )
            connector_score = min(connector_count / max(len(sentences) * 0.3, 1), 1.0)
            coherence = 0.6 * sentence_score + 0.4 * connector_score
        scores["logical_coherence"] = coherence

        # 2. 证据质量
        evidence = argument.evidence
        evidence_score = 0.0
        if evidence:
            evidence_score = min(len(evidence) / 3.0, 1.0)
            # 证据长度加分
            avg_evidence_len = sum(len(e) for e in evidence) / len(evidence)
            length_bonus = min(avg_evidence_len / 100.0, 0.3)
            evidence_score = min(evidence_score + length_bonus, 1.0)
        scores["evidence_quality"] = evidence_score

        # 3. 相关性 - 基于支持/反驳论点数量
        total_refs = len(argument.supporting_args) + len(argument.refuting_args)
        relevance = min(total_refs / 4.0, 1.0)
        scores["relevance"] = relevance

        # 4. 完整性 - 综合评估
        has_content = len(content) > 20
        has_evidence = len(evidence) > 0
        has_refs = total_refs > 0
        completeness = sum([has_content, has_evidence, has_refs]) / 3.0
        scores["completeness"] = completeness

        # 5. 新颖性 - 内容长度和词汇多样性
        words = content.lower().split()
        if len(words) > 0:
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            novelty = min(diversity * 1.5, 1.0)
        else:
            novelty = 0.0
        scores["novelty"] = novelty

        # 谬误惩罚
        fallacy_penalty = sum(conf for _, conf in argument.fallacies) * 0.15

        # 加权求和
        total_score = 0.0
        for criterion, weight in self._argument_quality_weights.items():
            total_score += weight * scores.get(criterion, 0.0)

        total_score = max(0.0, total_score - fallacy_penalty)
        return round(min(total_score, 1.0), 4)

    def detect_fallacy(self, argument: DebateArgument) -> List[Tuple[FallacyType, float]]:
        """检测论点中的逻辑谬误"""
        return self._fallacy_detector.detect_patterns(argument.content)

    def next_round(self, debate_id: str) -> int:
        """
        推进辩论到下一轮。

        Returns:
            新的轮次编号
        """
        session = self._get_debate(debate_id)
        if session.status != DebateStatus.IN_PROGRESS:
            raise ValueError(f"辩论未在进行中: {session.status.value}")

        session.round += 1
        session.updated_at = time.time()

        # 检查是否达到最大轮次
        if session.round >= session.topic.max_rounds:
            session.status = DebateStatus.FINISHED

        return session.round

    def reach_consensus(self, debate_id: str, threshold: float = 0.7) -> Tuple[bool, float]:
        """
        检测是否达成共识。

        共识判定逻辑：
        1. 计算各参与者的立场相似度
        2. 基于论点质量加权计算共识度
        3. 共识度超过阈值则达成共识

        Args:
            debate_id: 辩论ID
            threshold: 共识阈值，默认0.7

        Returns:
            (是否达成共识, 共识度)
        """
        session = self._get_debate(debate_id)
        if not session.arguments:
            return False, 0.0

        # 按参与者分组论点
        agent_args: Dict[str, List[DebateArgument]] = defaultdict(list)
        for arg in session.arguments:
            agent_args[arg.agent_id].append(arg)

        # 计算每个参与者的平均立场（基于论点置信度）
        agent_stances: Dict[str, float] = {}
        for agent_id, args in agent_args.items():
            if args:
                avg_confidence = sum(a.confidence for a in args) / len(args)
                # 加上质量权重
                avg_quality = sum(a.quality_score for a in args) / len(args)
                agent_stances[agent_id] = 0.6 * avg_confidence + 0.4 * avg_quality

        if len(agent_stances) < 2:
            return True, 1.0

        # 计算立场收敛度
        stance_values = list(agent_stances.values())
        mean_stance = sum(stance_values) / len(stance_values)

        # 标准差作为分歧度量
        variance = sum((s - mean_stance) ** 2 for s in stance_values) / len(stance_values)
        std_dev = math.sqrt(variance)

        # 收敛度：标准差越小，收敛度越高
        # 使用高斯函数映射: convergence = exp(-std_dev^2 / (2 * sigma^2))
        sigma = 0.3  # 控制收敛敏感度
        convergence = math.exp(-(std_dev ** 2) / (2 * sigma ** 2))

        # 加权因子：论点数量越多，共识越可靠
        total_args = len(session.arguments)
        volume_factor = min(total_args / (len(session.participants) * 2), 1.0)

        consensus_score = convergence * (0.5 + 0.5 * volume_factor)
        consensus_score = round(min(consensus_score, 1.0), 4)

        if consensus_score >= threshold:
            session.status = DebateStatus.CONSENSUS_REACHED
            session.consensus_value = self._derive_consensus_value(session)
            session.consensus_confidence = consensus_score
            session.updated_at = time.time()
            return True, consensus_score

        # 检查是否僵局（收敛度极低且已进行多轮）
        if convergence < 0.2 and session.round >= 3:
            session.status = DebateStatus.DEADLOCK

        return False, consensus_score

    def _derive_consensus_value(self, session: DebateSession) -> str:
        """从辩论论点中推导共识值"""
        if not session.arguments:
            return ""

        # 选择质量最高的论点作为共识基础
        best_args = sorted(
            session.arguments,
            key=lambda a: a.quality_score * a.confidence,
            reverse=True,
        )

        # 合并前3个最佳论点
        top_args = best_args[:3]
        consensus_parts = []
        for arg in top_args:
            # 提取关键句子（简化处理：取前两句）
            sentences = re.split(r'[.!?]+', arg.content)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:2]
            consensus_parts.extend(key_sentences)

        return " ".join(consensus_parts)

    def get_debate_summary(self, debate_id: str) -> Dict[str, Any]:
        """
        获取辩论摘要。

        包含各轮次论点统计、参与者表现、谬误检测报告等。
        """
        session = self._get_debate(debate_id)

        # 按轮次分组
        rounds: Dict[int, List[DebateArgument]] = defaultdict(list)
        for arg in session.arguments:
            rounds[arg.round].append(arg)

        # 参与者统计
        participant_stats: Dict[str, Dict[str, Any]] = {}
        for agent_id in session.participants:
            agent_arguments = [a for a in session.arguments if a.agent_id == agent_id]
            if agent_arguments:
                avg_quality = sum(a.quality_score for a in agent_arguments) / len(agent_arguments)
                avg_confidence = sum(a.confidence for a in agent_arguments) / len(agent_arguments)
                total_fallacies = sum(len(a.fallacies) for a in agent_arguments)
            else:
                avg_quality = 0.0
                avg_confidence = 0.0
                total_fallacies = 0

            participant_stats[agent_id] = {
                "arguments_count": len(agent_arguments),
                "avg_quality": round(avg_quality, 4),
                "avg_confidence": round(avg_confidence, 4),
                "total_fallacies": total_fallacies,
            }

        # 轮次统计
        round_summaries = {}
        for rnd, args in rounds.items():
            round_summaries[str(rnd)] = {
                "argument_count": len(args),
                "avg_quality": round(
                    sum(a.quality_score for a in args) / len(args), 4
                ) if args else 0.0,
                "total_fallacies": sum(len(a.fallacies) for a in args),
            }

        # 谬误统计
        fallacy_counts: Dict[str, int] = defaultdict(int)
        for arg in session.arguments:
            for ftype, _ in arg.fallacies:
                fallacy_counts[ftype.value] += 1

        return {
            "debate_id": session.id,
            "topic": session.topic.description,
            "status": session.status.value,
            "current_round": session.round,
            "max_rounds": session.topic.max_rounds,
            "total_arguments": len(session.arguments),
            "participants": participant_stats,
            "rounds": round_summaries,
            "fallacy_distribution": dict(fallacy_counts),
            "consensus": {
                "reached": session.status == DebateStatus.CONSENSUS_REACHED,
                "value": session.consensus_value,
                "confidence": session.consensus_confidence,
            },
            "winner": session.winner,
        }

    def declare_winner(self, debate_id: str) -> Optional[str]:
        """
        宣布辩论胜者。

        胜者判定基于综合得分 = 加权平均质量 * 论点数量因子 * 谬误惩罚
        """
        session = self._get_debate(debate_id)
        if not session.arguments:
            return None

        # 计算每个参与者的综合得分
        scores: Dict[str, float] = {}
        for agent_id in session.participants:
            agent_arguments = [a for a in session.arguments if a.agent_id == agent_id]
            if not agent_arguments:
                scores[agent_id] = 0.0
                continue

            avg_quality = sum(a.quality_score for a in agent_arguments) / len(agent_arguments)
            avg_confidence = sum(a.confidence for a in agent_arguments) / len(agent_arguments)
            total_fallacies = sum(len(a.fallacies) for a in agent_arguments)
            fallacy_penalty = total_fallacies * 0.05

            # 数量因子：鼓励积极参与，但有上限
            volume_factor = min(len(agent_arguments) / 3.0, 1.5)

            # 综合得分
            composite = (0.5 * avg_quality + 0.3 * avg_confidence) * volume_factor
            composite = max(0.0, composite - fallacy_penalty)
            scores[agent_id] = round(composite, 4)

        if not scores:
            return None

        winner = max(scores, key=scores.get)
        session.winner = winner
        session.status = DebateStatus.FINISHED
        session.updated_at = time.time()

        return winner

    def _get_debate(self, debate_id: str) -> DebateSession:
        """获取辩论会话，不存在则抛出异常"""
        if debate_id not in self._debates:
            raise KeyError(f"辩论不存在: {debate_id}")
        return self._debates[debate_id]


# ============================================================
# 辩论仲裁者
# ============================================================

class DebateMediator:
    """辩论仲裁者 - 主持辩论流程，解决僵局"""

    def __init__(self, engine: Optional[DebateEngine] = None):
        self.engine = engine or DebateEngine()
        self._mediation_history: List[Dict[str, Any]] = []

    def moderate(self, debate_id: str) -> Dict[str, Any]:
        """
        主持辩论，执行一轮完整的辩论流程。

        流程：
        1. 检查辩论状态
        2. 评估当前论点
        3. 检测共识
        4. 决定是否推进到下一轮
        5. 必要时解决僵局
        """
        session = self.engine._get_debate(debate_id)

        result = {
            "debate_id": debate_id,
            "actions_taken": [],
            "recommendations": [],
        }

        if session.status == DebateStatus.PENDING:
            # 首次主持，启动辩论
            session.status = DebateStatus.IN_PROGRESS
            result["actions_taken"].append("辩论已启动")

        if session.status == DebateStatus.IN_PROGRESS:
            # 检查当前轮次是否有足够的论点
            current_round_args = [
                a for a in session.arguments if a.round == session.round
            ]

            if len(current_round_args) >= len(session.participants):
                # 所有参与者都已发言，尝试达成共识
                consensus_reached, consensus_score = self.engine.reach_consensus(debate_id)

                if consensus_reached:
                    result["actions_taken"].append(
                        f"共识已达成，置信度: {consensus_score:.4f}"
                    )
                    winner = self.engine.declare_winner(debate_id)
                    if winner:
                        result["actions_taken"].append(f"胜者: {winner}")
                else:
                    # 检查是否僵局
                    if session.status == DebateStatus.DEADLOCK:
                        resolution = self.resolve_deadlock(debate_id)
                        result["actions_taken"].append(f"僵局已解决: {resolution}")
                    else:
                        # 推进到下一轮
                        new_round = self.engine.next_round(debate_id)
                        result["actions_taken"].append(f"进入第 {new_round} 轮")
                        result["recommendations"].append(
                            "建议参与者关注对方论点中的薄弱环节"
                        )
            else:
                missing = set(session.participants) - set(
                    a.agent_id for a in current_round_args
                )
                result["recommendations"].append(
                    f"等待以下参与者发言: {', '.join(missing)}"
                )

        elif session.status == DebateStatus.DEADLOCK:
            resolution = self.resolve_deadlock(debate_id)
            result["actions_taken"].append(f"僵局已解决: {resolution}")

        elif session.status in (DebateStatus.CONSENSUS_REACHED, DebateStatus.FINISHED):
            result["actions_taken"].append("辩论已结束")

        self._mediation_history.append(result)
        return result

    def resolve_deadlock(self, debate_id: str) -> str:
        """
        解决辩论僵局。

        策略：
        1. 分析各方立场差距
        2. 提出妥协方案
        3. 基于论点质量给出仲裁建议
        """
        session = self.engine._get_debate(debate_id)

        # 分析各方立场
        agent_scores: Dict[str, float] = {}
        for agent_id in session.participants:
            agent_args = [a for a in session.arguments if a.agent_id == agent_id]
            if agent_args:
                avg_quality = sum(a.quality_score for a in agent_args) / len(agent_args)
                agent_scores[agent_id] = avg_quality
            else:
                agent_scores[agent_id] = 0.0

        # 提出妥协方案
        compromise = self.propose_compromise(debate_id)

        # 基于质量给出仲裁建议
        if agent_scores:
            best_agent = max(agent_scores, key=agent_scores.get)
            best_score = agent_scores[best_agent]
            second_best = sorted(agent_scores.values(), reverse=True)[1] if len(agent_scores) > 1 else 0

            score_gap = best_score - second_best

            if score_gap > 0.2:
                # 分差明显，建议采用质量较高方的观点
                resolution = (
                    f"仲裁建议: 采纳 {best_agent} 的主要论点 "
                    f"(质量优势: {score_gap:.4f})，同时纳入妥协方案"
                )
            else:
                # 分差不大，建议融合
                resolution = (
                    f"仲裁建议: 双方论点质量接近 "
                    f"(差距: {score_gap:.4f})，建议采用融合方案"
                )
        else:
            resolution = "仲裁建议: 论据不足，建议补充证据后重新辩论"

        session.status = DebateStatus.IN_PROGRESS
        session.updated_at = time.time()

        return resolution

    def propose_compromise(self, debate_id: str) -> str:
        """
        提出妥协方案。

        基于各参与者的论点，提取共同点和分歧点，
        生成一个折中的共识建议。
        """
        session = self.engine._get_debate(debate_id)
        if not session.arguments:
            return "无法提出妥协方案：暂无论点"

        # 按参与者分组
        agent_args: Dict[str, List[DebateArgument]] = defaultdict(list)
        for arg in session.arguments:
            agent_args[arg.agent_id].append(arg)

        # 提取每个参与者的核心论点（质量最高的）
        core_arguments = {}
        for agent_id, args in agent_args.items():
            best_arg = max(args, key=lambda a: a.quality_score)
            # 提取关键句子
            sentences = re.split(r'[.!?]+', best_arg.content)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
            core_arguments[agent_id] = key_sentences[:3]

        # 寻找共同点（简化处理：基于词汇重叠）
        all_words_sets = {}
        for agent_id, sentences in core_arguments.items():
            words = set()
            for s in sentences:
                words.update(s.lower().split())
            # 过滤停用词
            stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "can", "shall",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "as", "into", "through", "during", "before", "after", "and",
                "but", "or", "nor", "not", "so", "yet", "both", "either",
                "neither", "each", "every", "all", "any", "few", "more",
                "most", "other", "some", "such", "no", "only", "own",
                "same", "than", "too", "very", "just", "because", "if",
                "that", "this", "these", "those", "it", "its", "we",
                "our", "they", "their", "what", "which", "who", "whom",
            }
            words -= stop_words
            all_words_sets[agent_id] = words

        # 找出所有参与者共有的词汇
        if all_words_sets:
            common_words = set.intersection(*all_words_sets.values())
            shared_concepts = list(common_words)[:10]
        else:
            shared_concepts = []

        # 构建妥协方案
        compromise_parts = ["妥协方案："]
        if shared_concepts:
            compromise_parts.append(
                f"各方共同关注的关键概念: {', '.join(shared_concepts)}"
            )

        for agent_id, sentences in core_arguments.items():
            compromise_parts.append(f"  - {agent_id} 核心观点: {'; '.join(sentences[:2])}")

        compromise_parts.append(
            "建议：各方在共同关注点基础上，适当调整各自立场，"
            "寻求最大公约数。"
        )

        return "\n".join(compromise_parts)

    def get_mediation_history(self) -> List[Dict[str, Any]]:
        """获取仲裁历史记录"""
        return list(self._mediation_history)
