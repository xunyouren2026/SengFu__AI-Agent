"""
极端化抑制模块
检测观点极化并引入温和调和者，防止辩论极端化
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import math
import statistics
from collections import defaultdict

from .protocol import Argument, Rebuttal, DebateState, Stance


class PolarizationLevel(Enum):
    """极化程度枚举"""
    NONE = "none"              # 无极化
    LOW = "low"                # 低度极化
    MODERATE = "moderate"      # 中度极化
    HIGH = "high"              # 高度极化
    SEVERE = "severe"          # 严重极化
    EXTREME = "extreme"        # 极端极化


class MitigationStrategy(Enum):
    """抑制策略枚举"""
    INTRODUCE_MODERATOR = "introduce_moderator"     # 引入调和者
    BALANCE_PERSPECTIVES = "balance_perspectives"   # 平衡视角
    SEEK_COMMON_GROUND = "seek_common_ground"       # 寻找共同点
    ENCOURAGE_NUANCE = "encourage_nuance"           # 鼓励细微差别
    PAUSE_DEBATE = "pause_debate"                   # 暂停辩论
    REFRAME_ISSUE = "reframe_issue"                 # 重新框架问题


@dataclass
class PolarizationMetrics:
    """极化度量指标"""
    stance_divergence: float = 0.0        # 立场分歧度 (0-1)
    argument_extremity: float = 0.0       # 论点极端度 (0-1)
    emotional_intensity: float = 0.0      # 情绪强度 (0-1)
    cross_understanding: float = 0.0      # 跨理解度 (0-1)
    group_polarization: float = 0.0       # 群体极化指数 (0-1)
    echo_chamber_index: float = 0.0       # 回音室指数 (0-1)
    
    def overall_polarization(self) -> float:
        """计算总体极化程度"""
        weights = {
            'stance_divergence': 0.25,
            'argument_extremity': 0.20,
            'emotional_intensity': 0.15,
            'cross_understanding': 0.15,
            'group_polarization': 0.15,
            'echo_chamber_index': 0.10,
        }
        
        # 跨理解度越低，极化越高
        adjusted_understanding = 1.0 - self.cross_understanding
        
        return (
            self.stance_divergence * weights['stance_divergence'] +
            self.argument_extremity * weights['argument_extremity'] +
            self.emotional_intensity * weights['emotional_intensity'] +
            adjusted_understanding * weights['cross_understanding'] +
            self.group_polarization * weights['group_polarization'] +
            self.echo_chamber_index * weights['echo_chamber_index']
        )
    
    def get_level(self) -> PolarizationLevel:
        """获取极化等级"""
        score = self.overall_polarization()
        
        if score < 0.2:
            return PolarizationLevel.NONE
        elif score < 0.4:
            return PolarizationLevel.LOW
        elif score < 0.6:
            return PolarizationLevel.MODERATE
        elif score < 0.75:
            return PolarizationLevel.HIGH
        elif score < 0.9:
            return PolarizationLevel.SEVERE
        else:
            return PolarizationLevel.EXTREME


@dataclass
class ModerationAction:
    """调和行动"""
    action_id: str
    strategy: MitigationStrategy
    trigger_metrics: PolarizationMetrics
    description: str
    moderator_id: Optional[str] = None
    target_participants: List[str] = field(default_factory=list)
    suggested_interventions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    effectiveness: float = 0.0


class StanceAnalyzer:
    """
    立场分析器
    分析参与者立场分布和分歧程度
    """
    
    def __init__(self) -> None:
        self.stance_history: Dict[str, List[Stance]] = defaultdict(list)
    
    def analyze_stance_distribution(
        self,
        arguments: List[Argument]
    ) -> Dict[Stance, float]:
        """
        分析立场分布
        
        Returns:
            各立场的比例
        """
        if not arguments:
            return {Stance.PRO: 0.0, Stance.CON: 0.0, Stance.NEUTRAL: 1.0}
        
        stance_counts: Dict[Stance, int] = defaultdict(int)
        for arg in arguments:
            stance_counts[arg.stance] += 1
        
        total = len(arguments)
        return {
            stance: count / total
            for stance, count in stance_counts.items()
        }
    
    def calculate_divergence(
        self,
        arguments: List[Argument]
    ) -> float:
        """
        计算立场分歧度
        
        使用极化指数公式：
        P = |p_pro - p_con| + p_neutral * 0.5
        
        其中 p_neutral 越多，分歧度越低
        """
        distribution = self.analyze_stance_distribution(arguments)
        
        p_pro = distribution.get(Stance.PRO, 0.0)
        p_con = distribution.get(Stance.CON, 0.0)
        p_neutral = distribution.get(Stance.NEUTRAL, 0.0)
        
        # 两极分化程度
        bipolar = abs(p_pro - p_con)
        
        # 如果双方都很极端（都接近0.5），则极化程度高
        if p_pro > 0.3 and p_con > 0.3:
            bipolar = 1.0 - abs(p_pro - p_con)  # 越接近，极化越高
        
        # 中立立场降低极化
        neutral_reduction = p_neutral * 0.5
        
        divergence = bipolar * (1.0 - neutral_reduction)
        return min(1.0, max(0.0, divergence))
    
    def track_stance_shift(
        self,
        participant_id: str,
        new_stance: Stance
    ) -> float:
        """
        追踪立场变化
        
        Returns:
            立场稳定性分数 (0-1, 越高越稳定)
        """
        history = self.stance_history[participant_id]
        history.append(new_stance)
        
        if len(history) < 2:
            return 1.0
        
        # 计算立场变化频率
        changes = 0
        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                changes += 1
        
        change_rate = changes / (len(history) - 1)
        stability = 1.0 - change_rate
        
        return stability


class ArgumentExtremityDetector:
    """
    论点极端度检测器
    检测论点是否包含极端表述
    """
    
    EXTREME_MARKERS = {
        "absolute": [
            "绝对", "永远", "从不", "所有", "没有", "必须", "一定",
            "完全", "彻底", "毫无", "百分之百"
        ],
        "intensifier": [
            "极其", "非常", "特别", "尤其", "最为", "极其",
            "灾难性的", "毁灭性的", "不可接受的"
        ],
        "exclusionary": [
            "只有", "除非", "否则", "绝不", "根本不", "完全不"
        ],
        "emotional": [
            "荒谬", "可笑", "愚蠢", "可耻", "卑鄙", "无耻",
            "令人发指", "不可理喻"
        ]
    }
    
    MODERATE_MARKERS = [
        "可能", "也许", "某种程度上", "部分", "一些",
        "通常", "往往", "一般", "倾向于", "相对"
    ]
    
    def detect_extremity(self, argument: Argument) -> Tuple[float, List[str]]:
        """
        检测论点极端度
        
        Returns:
            (极端度分数, 检测到的极端标记列表)
        """
        content = argument.content
        detected_markers: List[str] = []
        extremity_scores: List[float] = []
        
        # 检测绝对化表述
        absolute_count = 0
        for marker in self.EXTREME_MARKERS["absolute"]:
            if marker in content:
                absolute_count += 1
                detected_markers.append(f"absolute:{marker}")
        extremity_scores.append(min(1.0, absolute_count * 0.25))
        
        # 检测强化词
        intensifier_count = 0
        for marker in self.EXTREME_MARKERS["intensifier"]:
            if marker in content:
                intensifier_count += 1
                detected_markers.append(f"intensifier:{marker}")
        extremity_scores.append(min(1.0, intensifier_count * 0.2))
        
        # 检测排他性表述
        exclusionary_count = 0
        for marker in self.EXTREME_MARKERS["exclusionary"]:
            if marker in content:
                exclusionary_count += 1
                detected_markers.append(f"exclusionary:{marker}")
        extremity_scores.append(min(1.0, exclusionary_count * 0.2))
        
        # 检测情绪化表述
        emotional_count = 0
        for marker in self.EXTREME_MARKERS["emotional"]:
            if marker in content:
                emotional_count += 1
                detected_markers.append(f"emotional:{marker}")
        extremity_scores.append(min(1.0, emotional_count * 0.3))
        
        # 检测缓和标记（降低极端度）
        moderate_count = sum(1 for m in self.MODERATE_MARKERS if m in content)
        moderation_factor = max(0.3, 1.0 - moderate_count * 0.15)
        
        if extremity_scores:
            base_extremity = sum(extremity_scores) / len(extremity_scores)
            final_extremity = base_extremity * moderation_factor
        else:
            final_extremity = 0.0
        
        return final_extremity, detected_markers
    
    def batch_analyze(
        self,
        arguments: List[Argument]
    ) -> Tuple[float, Dict[str, float]]:
        """
        批量分析论点极端度
        
        Returns:
            (平均极端度, 各论点极端度映射)
        """
        extremity_map: Dict[str, float] = {}
        
        for arg in arguments:
            extremity, _ = self.detect_extremity(arg)
            extremity_map[arg.argument_id] = extremity
        
        if extremity_map:
            avg_extremity = sum(extremity_map.values()) / len(extremity_map)
        else:
            avg_extremity = 0.0
        
        return avg_extremity, extremity_map


class EchoChamberDetector:
    """
    回音室效应检测器
    检测参与者是否只与相同观点者互动
    """
    
    def __init__(self) -> None:
        self.interaction_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        self.stance_map: Dict[str, Stance] = {}
    
    def record_interaction(
        self,
        from_participant: str,
        to_participant: str,
        interaction_type: str = "rebuttal"
    ) -> None:
        """记录互动"""
        key = (from_participant, to_participant)
        self.interaction_matrix[key] += 1
    
    def update_stance(self, participant_id: str, stance: Stance) -> None:
        """更新参与者立场"""
        self.stance_map[participant_id] = stance
    
    def calculate_echo_chamber_index(
        self,
        participants: Set[str]
    ) -> float:
        """
        计算回音室指数
        
        回音室指数 = 同立场互动数 / 总互动数
        
        指数越高，说明参与者越倾向于只与相同立场者互动
        """
        same_stance_interactions = 0
        cross_stance_interactions = 0
        
        for (from_p, to_p), count in self.interaction_matrix.items():
            from_stance = self.stance_map.get(from_p)
            to_stance = self.stance_map.get(to_p)
            
            if from_stance and to_stance:
                if from_stance == to_stance:
                    same_stance_interactions += count
                else:
                    cross_stance_interactions += count
        
        total = same_stance_interactions + cross_stance_interactions
        
        if total == 0:
            return 0.0
        
        # 回音室指数：同立场互动比例
        echo_index = same_stance_interactions / total
        
        return echo_index
    
    def identify_echo_chambers(
        self,
        participants: Set[str]
    ) -> List[Set[str]]:
        """
        识别回音室群体
        
        Returns:
            回音室群体列表
        """
        # 按立场分组
        stance_groups: Dict[Stance, Set[str]] = defaultdict(set)
        for p_id, stance in self.stance_map.items():
            if p_id in participants:
                stance_groups[stance].add(p_id)
        
        # 检查每个群体内部的互动密度
        echo_chambers: List[Set[str]] = []
        
        for stance, group in stance_groups.items():
            if len(group) < 2:
                continue
            
            # 计算群体内部互动密度
            internal_interactions = 0
            for (from_p, to_p), count in self.interaction_matrix.items():
                if from_p in group and to_p in group:
                    internal_interactions += count
            
            # 计算群体对外互动
            external_interactions = 0
            for (from_p, to_p), count in self.interaction_matrix.items():
                if (from_p in group and to_p not in group) or \
                   (from_p not in group and to_p in group):
                    external_interactions += count
            
            total = internal_interactions + external_interactions
            if total > 0:
                internal_ratio = internal_interactions / total
                # 如果内部互动比例超过70%，认为是回音室
                if internal_ratio > 0.7:
                    echo_chambers.append(group)
        
        return echo_chambers


class EmotionalIntensityAnalyzer:
    """
    情绪强度分析器
    分析论点的情绪强度
    """
    
    EMOTIONAL_PATTERNS = {
        "anger": {
            "markers": ["愤怒", "气愤", "不可接受", "荒唐", "可耻"],
            "weight": 0.9
        },
        "frustration": {
            "markers": ["失望", "无奈", "令人沮丧", "难以理解"],
            "weight": 0.7
        },
        "certainty": {
            "markers": ["毫无疑问", "显然", "显而易见", "绝对"],
            "weight": 0.6
        },
        "disappointment": {
            "markers": ["遗憾", "可惜", "本应", "期望"],
            "weight": 0.5
        },
        "neutral": {
            "markers": ["认为", "看来", "可能", "也许"],
            "weight": 0.1
        }
    }
    
    def analyze_intensity(self, argument: Argument) -> Tuple[float, str]:
        """
        分析情绪强度
        
        Returns:
            (强度分数, 主要情绪类型)
        """
        content = argument.content
        emotion_scores: Dict[str, float] = {}
        
        for emotion, config in self.EMOTIONAL_PATTERNS.items():
            markers = config["markers"]
            weight = config["weight"]
            
            # 计算该情绪的触发强度
            matches = sum(1 for m in markers if m in content)
            if matches > 0:
                emotion_scores[emotion] = min(1.0, matches * 0.3) * weight
        
        if emotion_scores:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            intensity = emotion_scores[dominant_emotion]
        else:
            dominant_emotion = "neutral"
            intensity = 0.0
        
        return intensity, dominant_emotion
    
    def batch_analyze(
        self,
        arguments: List[Argument]
    ) -> float:
        """批量分析平均情绪强度"""
        if not arguments:
            return 0.0
        
        intensities = [self.analyze_intensity(arg)[0] for arg in arguments]
        return sum(intensities) / len(intensities)


class ModerationInterventionGenerator:
    """
    调和干预生成器
    生成缓和极端化的干预措施
    """
    
    INTERVENTION_TEMPLATES = {
        MitigationStrategy.INTRODUCE_MODERATOR: [
            "让我们引入一个中立的视角来审视这个问题。",
            "我想请调和者分享他们对这个问题的看法。",
            "为了平衡讨论，我们需要听听中间立场的声音。"
        ],
        MitigationStrategy.BALANCE_PERSPECTIVES: [
            "让我们也考虑一下对方观点的合理之处。",
            "每个观点都有其价值，让我们全面审视。",
            "我们需要更全面地看待这个问题的不同方面。"
        ],
        MitigationStrategy.SEEK_COMMON_GROUND: [
            "让我们找出双方都能认同的共同点。",
            "尽管存在分歧，我们是否有共同的出发点？",
            "让我们先确认双方都同意的基本前提。"
        ],
        MitigationStrategy.ENCOURAGE_NUANCE: [
            "这个问题可能不是非黑即白的，让我们考虑其中的细微差别。",
            "让我们区分不同情况，而不是一概而论。",
            "现实往往比简单的对错更复杂，让我们深入分析。"
        ],
        MitigationStrategy.REFRAME_ISSUE: [
            "让我们换一个角度来看这个问题。",
            "也许我们可以用不同的框架来思考这个问题。",
            "让我们重新定义问题的核心。"
        ],
        MitigationStrategy.PAUSE_DEBATE: [
            "讨论变得激烈，让我们暂停一下，冷静思考。",
            "我建议我们稍作休息，以更清晰的头脑继续讨论。",
            "让我们花一点时间反思刚才的讨论。"
        ]
    }
    
    def generate_intervention(
        self,
        strategy: MitigationStrategy,
        context: str = ""
    ) -> str:
        """生成干预语句"""
        templates = self.INTERVENTION_TEMPLATES.get(
            strategy,
            ["让我们以更开放的心态继续讨论。"]
        )
        
        import random
        return random.choice(templates)
    
    def generate_moderator_argument(
        self,
        pro_arguments: List[Argument],
        con_arguments: List[Argument],
        topic: str
    ) -> str:
        """
        生成调和者论点
        综合双方观点，提出平衡视角
        """
        # 提取双方核心观点
        pro_points = [arg.content[:100] for arg in pro_arguments[:3]]
        con_points = [arg.content[:100] for arg in con_arguments[:3]]
        
        moderator_content = f"关于{topic}，我观察到双方都有合理的关切。"
        
        if pro_points:
            moderator_content += f" 支持方强调了{len(pro_points)}个要点。"
        if con_points:
            moderator_content += f" 反对方提出了{len(con_points)}个关切。"
        
        moderator_content += " 我认为关键在于找到平衡点，既考虑支持方的合理诉求，也重视反对方的关切。"
        moderator_content += " 让我们寻求一个能够兼顾各方利益的解决方案。"
        
        return moderator_content


class RadicalizationMitigator:
    """
    极端化抑制器
    主类，协调所有极化检测和抑制功能
    """
    
    def __init__(
        self,
        polarization_threshold: float = 0.6,
        intervention_threshold: float = 0.75
    ) -> None:
        self.polarization_threshold = polarization_threshold
        self.intervention_threshold = intervention_threshold
        
        self.stance_analyzer = StanceAnalyzer()
        self.extremity_detector = ArgumentExtremityDetector()
        self.echo_chamber_detector = EchoChamberDetector()
        self.emotional_analyzer = EmotionalIntensityAnalyzer()
        self.intervention_generator = ModerationInterventionGenerator()
        
        self.metrics_history: List[PolarizationMetrics] = []
        self.interventions: List[ModerationAction] = []
    
    def assess_polarization(
        self,
        debate_state: DebateState
    ) -> PolarizationMetrics:
        """
        评估当前辩论的极化程度
        
        Args:
            debate_state: 辩论状态
            
        Returns:
            极化度量指标
        """
        arguments = list(debate_state.arguments.values())
        participants = debate_state.participants
        
        # 1. 立场分歧度
        stance_divergence = self.stance_analyzer.calculate_divergence(arguments)
        
        # 2. 论点极端度
        argument_extremity, _ = self.extremity_detector.batch_analyze(arguments)
        
        # 3. 情绪强度
        emotional_intensity = self.emotional_analyzer.batch_analyze(arguments)
        
        # 4. 跨理解度（基于反驳的针对性）
        cross_understanding = self._calculate_cross_understanding(debate_state)
        
        # 5. 群体极化指数
        group_polarization = self._calculate_group_polarization(arguments)
        
        # 6. 回音室指数
        self._update_interaction_data(debate_state)
        echo_chamber_index = self.echo_chamber_detector.calculate_echo_chamber_index(participants)
        
        metrics = PolarizationMetrics(
            stance_divergence=stance_divergence,
            argument_extremity=argument_extremity,
            emotional_intensity=emotional_intensity,
            cross_understanding=cross_understanding,
            group_polarization=group_polarization,
            echo_chamber_index=echo_chamber_index
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_cross_understanding(
        self,
        debate_state: DebateState
    ) -> float:
        """
        计算跨理解度
        基于反驳是否准确理解对方观点
        """
        rebuttals = list(debate_state.rebuttals.values())
        
        if not rebuttals:
            return 1.0
        
        # 简化计算：基于反驳的有效性分数
        effectiveness_scores = [r.effectiveness_score for r in rebuttals]
        
        if effectiveness_scores:
            return sum(effectiveness_scores) / len(effectiveness_scores)
        return 0.5
    
    def _calculate_group_polarization(
        self,
        arguments: List[Argument]
    ) -> float:
        """
        计算群体极化指数
        使用群体极化理论：群体讨论往往导致更极端的立场
        """
        if len(arguments) < 4:
            return 0.0
        
        # 按立场分组
        pro_args = [a for a in arguments if a.stance == Stance.PRO]
        con_args = [a for a in arguments if a.stance == Stance.CON]
        
        if not pro_args or not con_args:
            return 0.0
        
        # 计算各组的平均置信度
        pro_confidence = sum(a.confidence for a in pro_args) / len(pro_args)
        con_confidence = sum(a.confidence for a in con_args) / len(con_args)
        
        # 高置信度表示极化
        avg_confidence = (pro_confidence + con_confidence) / 2
        
        # 组大小差异也影响极化
        size_ratio = min(len(pro_args), len(con_args)) / max(len(pro_args), len(con_args))
        
        polarization = avg_confidence * (1.0 - size_ratio * 0.3)
        
        return min(1.0, polarization)
    
    def _update_interaction_data(
        self,
        debate_state: DebateState
    ) -> None:
        """更新互动数据"""
        # 从反驳中提取互动
        for rebuttal in debate_state.rebuttals.values():
            target_arg = debate_state.arguments.get(rebuttal.target_argument_id)
            if target_arg:
                self.echo_chamber_detector.record_interaction(
                    rebuttal.speaker_id,
                    target_arg.speaker_id
                )
        
        # 更新立场信息
        for arg in debate_state.arguments.values():
            self.echo_chamber_detector.update_stance(
                arg.speaker_id,
                arg.stance
            )
    
    def should_intervene(
        self,
        metrics: PolarizationMetrics
    ) -> Tuple[bool, MitigationStrategy]:
        """
        判断是否需要干预
        
        Returns:
            (是否需要干预, 推荐策略)
        """
        level = metrics.get_level()
        
        if level == PolarizationLevel.EXTREME:
            return True, MitigationStrategy.PAUSE_DEBATE
        
        if level == PolarizationLevel.SEVERE:
            return True, MitigationStrategy.INTRODUCE_MODERATOR
        
        if level == PolarizationLevel.HIGH:
            # 根据具体指标选择策略
            if metrics.echo_chamber_index > 0.7:
                return True, MitigationStrategy.BALANCE_PERSPECTIVES
            elif metrics.argument_extremity > 0.7:
                return True, MitigationStrategy.ENCOURAGE_NUANCE
            else:
                return True, MitigationStrategy.SEEK_COMMON_GROUND
        
        if level == PolarizationLevel.MODERATE:
            if metrics.overall_polarization() > self.polarization_threshold:
                return True, MitigationStrategy.REFRAME_ISSUE
        
        return False, MitigationStrategy.SEEK_COMMON_GROUND
    
    def generate_intervention(
        self,
        debate_state: DebateState,
        metrics: PolarizationMetrics
    ) -> Optional[ModerationAction]:
        """
        生成干预措施
        
        Args:
            debate_state: 辩论状态
            metrics: 极化指标
            
        Returns:
            干预措施（如果需要）
        """
        should_act, strategy = self.should_intervene(metrics)
        
        if not should_act:
            return None
        
        from uuid import uuid4
        action = ModerationAction(
            action_id=str(uuid4())[:8],
            strategy=strategy,
            trigger_metrics=metrics,
            description=f"检测到{metrics.get_level().value}程度极化，采取{strategy.value}策略"
        )
        
        # 生成具体干预语句
        intervention_text = self.intervention_generator.generate_intervention(
            strategy,
            debate_state.topic
        )
        action.suggested_interventions.append(intervention_text)
        
        # 如果需要引入调和者，生成调和者论点
        if strategy == MitigationStrategy.INTRODUCE_MODERATOR:
            pro_args = debate_state.get_arguments_by_stance(Stance.PRO)
            con_args = debate_state.get_arguments_by_stance(Stance.CON)
            moderator_arg = self.intervention_generator.generate_moderator_argument(
                pro_args, con_args, debate_state.topic
            )
            action.suggested_interventions.append(moderator_arg)
        
        # 识别目标参与者
        action.target_participants = list(debate_state.participants)
        
        self.interventions.append(action)
        
        return action
    
    def get_polarization_trend(self) -> Dict[str, Any]:
        """
        获取极化趋势
        
        Returns:
            趋势分析结果
        """
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_scores = [
            m.overall_polarization() 
            for m in self.metrics_history[-5:]
        ]
        
        # 计算趋势
        if len(recent_scores) >= 2:
            diff = recent_scores[-1] - recent_scores[0]
            if diff > 0.1:
                trend = "increasing"
            elif diff < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "trend": trend,
            "recent_scores": recent_scores,
            "current_level": self.metrics_history[-1].get_level().value,
            "intervention_count": len(self.interventions)
        }
    
    def reset(self) -> None:
        """重置状态"""
        self.metrics_history.clear()
        self.interventions.clear()
        self.echo_chamber_detector.interaction_matrix.clear()
        self.echo_chamber_detector.stance_map.clear()
        self.stance_analyzer.stance_history.clear()


__all__ = [
    "PolarizationLevel",
    "MitigationStrategy",
    "PolarizationMetrics",
    "ModerationAction",
    "StanceAnalyzer",
    "ArgumentExtremityDetector",
    "EchoChamberDetector",
    "EmotionalIntensityAnalyzer",
    "ModerationInterventionGenerator",
    "RadicalizationMitigator",
]
