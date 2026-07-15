"""
Superego Agent Module - AGI Unified Framework
================================================
Implements ethical oversight, behavior review, and alignment monitoring
for AI agents using a superego-inspired architecture.

Pure Python implementation with no external dependencies.
"""

import math
import random
import re
import hashlib
import json
import time
import threading
import statistics
import functools
import copy
from typing import (
    List, Dict, Tuple, Optional, Any, Set, Callable, Union
)
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


# =============================================================================
# Data Classes
# =============================================================================

class ActionType(str, Enum):
    """Types of agent actions that can be reviewed."""
    GENERATE = "generate"
    TOOL_CALL = "tool_call"
    MEMORY_ACCESS = "memory_access"
    PLANNING = "planning"


class RiskLevel(str, Enum):
    """Risk levels for review results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EthicalFrameworkType(str, Enum):
    """Supported ethical frameworks."""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    MIXED = "mixed"


@dataclass
class BehaviorTrace:
    """Represents a single behavior trace from an agent."""
    trace_id: str
    agent_id: str
    timestamp: float
    action_type: str  # generate/tool_call/memory_access/planning
    input_text: str
    output_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a unique trace ID using SHA-256 hash."""
        raw = f"{time.time()}-{random.getrandbits(64)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class SuperegoConfig:
    """Configuration for the SuperegoAgent."""
    strictness_level: int = 3  # 1-5
    review_scope: str = "full"  # full / minimal / targeted
    auto_correct: bool = True
    max_reviews_per_minute: int = 60
    enable_learning: bool = True
    ethical_framework: str = "utilitarian"

    def __post_init__(self):
        """Validate configuration values."""
        if not 1 <= self.strictness_level <= 5:
            raise ValueError(f"strictness_level must be 1-5, got {self.strictness_level}")
        valid_frameworks = {"utilitarian", "deontological", "virtue_ethics", "mixed"}
        if self.ethical_framework not in valid_frameworks:
            raise ValueError(f"ethical_framework must be one of {valid_frameworks}, "
                             f"got {self.ethical_framework}")
        valid_scopes = {"full", "minimal", "targeted"}
        if self.review_scope not in valid_scopes:
            raise ValueError(f"review_scope must be one of {valid_scopes}, "
                             f"got {self.review_scope}")


@dataclass
class ImpactAssessment:
    """Multi-dimensional impact assessment of an action."""
    user_impact: float  # 0-1
    third_party_impact: float  # 0-1
    short_term_risk: float  # 0-1
    long_term_risk: float  # 0-1
    reversibility: float  # 0-1, 1 = fully reversible
    overall_impact: float  # 0-1

    def to_dict(self) -> Dict[str, float]:
        return {
            "user_impact": self.user_impact,
            "third_party_impact": self.third_party_impact,
            "short_term_risk": self.short_term_risk,
            "long_term_risk": self.long_term_risk,
            "reversibility": self.reversibility,
            "overall_impact": self.overall_impact,
        }


@dataclass
class ReviewResult:
    """Result of a behavior review."""
    passed: bool
    risk_level: str  # low/medium/high/critical
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ethical_score: float = 1.0  # 0-1
    impact_assessment: Optional[ImpactAssessment] = None
    correction: str = ""
    review_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "passed": self.passed,
            "risk_level": self.risk_level,
            "violations": self.violations,
            "warnings": self.warnings,
            "ethical_score": self.ethical_score,
            "correction": self.correction,
            "review_time": self.review_time,
            "metadata": self.metadata,
        }
        if self.impact_assessment is not None:
            result["impact_assessment"] = self.impact_assessment.to_dict()
        return result


@dataclass
class HealthStatus:
    """Health status of the SuperegoMonitor."""
    is_healthy: bool
    uptime_seconds: float
    total_reviews: int
    reviews_per_minute: float
    average_review_time: float
    error_count: int
    last_check_time: float
    alerts_pending: int


# =============================================================================
# Ethical Frameworks
# =============================================================================

class EthicalFramework:
    """Base class for ethical evaluation frameworks."""

    def evaluate(self, action: str, **kwargs) -> float:
        """Evaluate an action ethically. Returns score 0-1.
        
        Default implementation provides a heuristic evaluation based on
        common ethical principles: non-maleficence, beneficence, autonomy,
        and justice. Subclasses should override this for more specialized
        ethical frameworks.
        
        Args:
            action: Description of the action to evaluate.
            **kwargs: Additional context (e.g., consequences, stakeholders).
            
        Returns:
            Ethical score between 0.0 (unethical) and 1.0 (fully ethical).
        """
        import re
        
        if not action or not isinstance(action, str):
            return 0.5
        
        action_lower = action.lower().strip()
        score = 0.5  # neutral baseline
        
        # Harm-related keywords (reduce score)
        harm_patterns = [
            r'\bharm\b', r'\bhurt\b', r'\bkill\b', r'\bdamage\b',
            r'\bdestroy\b', r'\bexploit\b', r'\bdeceive\b', r'\babuse\b',
            r'\bmanipulate\b', r'\bsteal\b', r'\bthreaten\b', r'\bcoerce\b',
            r'\bendanger\b', r'\bviolence\b', r'\billegal\b', r'\bfraud\b',
            r'\b伤害\b', r'\b破坏\b', r'\b欺骗\b', r'\b利用\b', r'\b威胁\b',
            r'\b滥用\b', r'\b违法\b',
        ]
        
        # Benefit-related keywords (increase score)
        benefit_patterns = [
            r'\bhelp\b', r'\bsave\b', r'\bprotect\b', r'\bimprove\b',
            r'\bsupport\b', r'\beducate\b', r'\bheal\b', r'\benable\b',
            r'\bempower\b', r'\brespect\b', r'\bconsent\b', r'\bfair\b',
            r'\b安全\b', r'\b帮助\b', r'\b改善\b', r'\b保护\b', r'\b支持\b',
            r'\b尊重\b', r'\b公正\b',
        ]
        
        harm_count = 0
        benefit_count = 0
        
        for pattern in harm_patterns:
            if re.search(pattern, action_lower):
                harm_count += 1
        
        for pattern in benefit_patterns:
            if re.search(pattern, action_lower):
                benefit_count += 1
        
        # Adjust score based on keyword analysis
        if harm_count > 0:
            score -= min(0.4, 0.1 * harm_count)
        if benefit_count > 0:
            score += min(0.4, 0.1 * benefit_count)
        
        # Consider explicit consequences if provided
        consequences = kwargs.get("consequences")
        if isinstance(consequences, list):
            for consequence in consequences:
                cons_lower = str(consequence).lower()
                for pattern in harm_patterns:
                    if re.search(pattern, cons_lower):
                        score -= 0.05
                for pattern in benefit_patterns:
                    if re.search(pattern, cons_lower):
                        score += 0.05
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        return score

    def get_description(self) -> str:
        """Return a description of this ethical framework."""
        return "Base ethical framework"


class UtilitarianFramework(EthicalFramework):
    """
    Utilitarian ethical framework.
    Evaluates actions based on their consequences and overall utility.
    Maximum happiness for the maximum number of sentient beings.
    """

    # Keywords associated with positive and negative outcomes
    POSITIVE_CONSEQUENCE_KEYWORDS = {
        "help", "benefit", "improve", "save", "protect", "heal",
        "support", "enable", "empower", "educate", "clarify",
        "solve", "resolve", "create", "build", "enhance",
        "安全", "帮助", "改善", "保护", "支持", "解决",
    }
    NEGATIVE_CONSEQUENCE_KEYWORDS = {
        "harm", "damage", "destroy", "kill", "hurt", "exploit",
        "manipulate", "deceive", "steal", "abuse", "endanger",
        "threaten", "coerce", "intimidate", "suppress", "mislead",
        "伤害", "破坏", "欺骗", "利用", "威胁", "滥用",
    }
    NEUTRAL_KEYWORDS = {
        "information", "data", "process", "analyze", "compute",
        "retrieve", "store", "update", "delete", "modify",
        "信息", "数据", "处理", "分析", "计算",
    }

    def evaluate(self, action: str, consequences: Optional[List[str]] = None,
                 **kwargs) -> float:
        """
        Evaluate action based on consequences using utility calculus.
        Score 0 = maximally harmful, 1 = maximally beneficial.

        Algorithm:
        1. Tokenize action text
        2. Count positive and negative consequence indicators
        3. If explicit consequences provided, weight them more heavily
        4. Compute utility score with diminishing returns for extreme values
        """
        if not action:
            return 0.5

        action_lower = action.lower()
        tokens = set(re.findall(r'\w+', action_lower))

        # Count keyword matches
        positive_count = len(tokens & self.POSITIVE_CONSEQUENCE_KEYWORDS)
        negative_count = len(tokens & self.NEGATIVE_CONSEQUENCE_KEYWORDS)
        neutral_count = len(tokens & self.NEUTRAL_KEYWORDS)

        # Base score from keyword analysis
        total_relevant = positive_count + negative_count + neutral_count
        if total_relevant > 0:
            base_score = (positive_count + 0.5 * neutral_count) / total_relevant
        else:
            base_score = 0.5  # Neutral default

        # Factor in explicit consequences if provided
        if consequences:
            consequence_score = self._evaluate_consequences(consequences)
            # Weight explicit consequences at 60%, keyword analysis at 40%
            base_score = 0.6 * consequence_score + 0.4 * base_score

        # Apply sigmoid-like transformation to avoid extreme values
        # Maps [0, 1] -> [0.05, 0.95] with smooth curve
        score = self._sigmoid_transform(base_score)

        return max(0.0, min(1.0, score))

    def _evaluate_consequences(self, consequences: List[str]) -> float:
        """Evaluate a list of consequence descriptions."""
        if not consequences:
            return 0.5

        scores = []
        for consequence in consequences:
            conc_lower = consequence.lower()
            tokens = set(re.findall(r'\w+', conc_lower))
            pos = len(tokens & self.POSITIVE_CONSEQUENCE_KEYWORDS)
            neg = len(tokens & self.NEGATIVE_CONSEQUENCE_KEYWORDS)
            total = pos + neg
            if total > 0:
                scores.append(pos / total)
            else:
                scores.append(0.5)

        return statistics.mean(scores)

    def compare_actions(self, actions: List[str],
                        consequences: List[List[str]]) -> List[Tuple[int, float]]:
        """
        Compare multiple actions by their utility scores.
        Returns sorted list of (action_index, score) tuples, highest first.
        """
        if len(actions) != len(consequences):
            raise ValueError("actions and consequences must have same length")

        results = []
        for i, (action, cons) in enumerate(zip(actions, consequences)):
            score = self.evaluate(action, consequences=cons)
            results.append((i, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _sigmoid_transform(x: float) -> float:
        """Apply sigmoid transform: maps [0,1] to a smooth curve avoiding extremes."""
        # Sigmoid centered at 0.5, steepness 6
        sigmoid = 1.0 / (1.0 + math.exp(-6.0 * (x - 0.5)))
        # Scale to [0.05, 0.95]
        return 0.05 + 0.9 * sigmoid

    def get_description(self) -> str:
        return ("Utilitarian framework: evaluates actions based on their "
                "consequences and overall utility for all affected parties.")


class DeontologicalFramework(EthicalFramework):
    """
    Deontological ethical framework.
    Evaluates actions based on adherence to rules and duties,
    regardless of consequences.
    """

    CORE_DUTIES = [
        "Do not deceive or lie to users",
        "Do not cause intentional harm",
        "Respect user autonomy and privacy",
        "Do not enable illegal activities",
        "Provide honest and accurate information",
        "Respect intellectual property rights",
        "Do not discriminate or show bias",
        "Maintain transparency about AI nature",
        "不欺骗或误导用户",
        "不造成故意伤害",
        "尊重用户自主权和隐私",
        "不协助非法活动",
        "提供诚实准确的信息",
        "尊重知识产权",
        "不歧视或展示偏见",
        "保持AI身份的透明度",
    ]

    # Rule violation patterns
    VIOLATION_PATTERNS = {
        "deception": [
            r'\b(假装|pretend)\b.*\b(是|am|is)\b',
            r'\b(我是人类|i am human|i\'m human)\b',
            r'\b(我有感情|i have feelings|i feel emotions)\b',
            r'\b(我不知道我是AI|i don\'t know.*AI)\b',
        ],
        "harm": [
            r'\b(如何伤害|how to harm|how to hurt|how to kill)\b',
            r'\b(制造武器|make a weapon|build a bomb)\b',
            r'\b(攻击他人|attack someone|hurt someone)\b',
        ],
        "privacy": [
            r'\b(窃取|steal).*\b(密码|password|data|数据)\b',
            r'\b(未经允许|without permission).*\b(访问|access)\b',
            r'\b(监控|spy|surveil).*\b(他人|someone)\b',
        ],
        "illegal": [
            r'\b(如何犯罪|how to commit|how to hack)\b',
            r'\b(绕过安全|bypass security|evade detection)\b',
            r'\b(洗钱|money laundering|drug.*manufactur)\b',
        ],
        "discrimination": [
            r'\b(歧视|discriminat|racist|sexist|homophobic)\b',
            r'\b( inferior|superior ).*\b(race|gender|ethnic)\b',
        ],
    }

    def evaluate(self, action: str, rules: Optional[List[str]] = None,
                 **kwargs) -> float:
        """
        Evaluate action based on rule adherence.
        Score 0 = violates core duties, 1 = fully compliant.
        """
        if not action:
            return 1.0

        action_lower = action.lower()
        violations_found = 0
        total_rules_checked = 0

        # Check against built-in violation patterns
        for category, patterns in self.VIOLATION_PATTERNS.items():
            for pattern in patterns:
                total_rules_checked += 1
                if re.search(pattern, action_lower, re.IGNORECASE):
                    violations_found += 1

        # Check against custom rules if provided
        if rules:
            for rule in rules:
                total_rules_checked += 1
                rule_keywords = set(re.findall(r'\w+', rule.lower()))
                action_keywords = set(re.findall(r'\w+', action_lower))
                # If significant overlap with a negative rule, flag violation
                if rule_keywords and action_keywords:
                    overlap = len(rule_keywords & action_keywords)
                    if overlap >= max(1, len(rule_keywords) * 0.5):
                        violations_found += 1

        if total_rules_checked == 0:
            return 1.0

        compliance_ratio = 1.0 - (violations_found / total_rules_checked)

        # Apply strictness curve: small violations are penalized more heavily
        if compliance_ratio < 0.8:
            score = compliance_ratio * 0.8  # Amplify penalty
        else:
            score = 0.8 + (compliance_ratio - 0.8) * 1.0  # Linear recovery

        return max(0.0, min(1.0, score))

    def get_duties(self) -> List[str]:
        """Return the list of core duties."""
        return list(self.CORE_DUTIES)

    def get_description(self) -> str:
        return ("Deontological framework: evaluates actions based on adherence "
                "to moral rules and duties, independent of consequences.")


class VirtueEthicsFramework(EthicalFramework):
    """
    Virtue ethics framework.
    Evaluates actions based on whether they express or cultivate virtues.
    """

    CORE_VIRTUES = {
        "honesty": {
            "indicators": ["truth", "accurate", "honest", "transparent",
                           "事实", "准确", "诚实", "透明"],
            "anti_indicators": ["lie", "deceive", "mislead", "distort",
                                "谎言", "欺骗", "误导", "歪曲"],
            "weight": 1.0,
        },
        "benevolence": {
            "indicators": ["help", "care", "support", "kind", "compassion",
                           "帮助", "关心", "支持", "善良", "同情"],
            "anti_indicators": ["harm", "cruel", "ignore suffering",
                                "伤害", "残忍", "无视痛苦"],
            "weight": 0.9,
        },
        "wisdom": {
            "indicators": ["consider", "reflect", "balanced", "thoughtful",
                           "reasoned", "considered",
                           "考虑", "反思", "平衡", "审慎", "理性"],
            "anti_indicators": ["rash", "reckless", "impulsive", "foolish",
                                "鲁莽", "冲动", "愚蠢"],
            "weight": 0.8,
        },
        "justice": {
            "indicators": ["fair", "equitable", "impartial", "unbiased",
                           "公平", "公正", "无偏见", "平等"],
            "anti_indicators": ["unfair", "biased", "prejudiced",
                                "不公平", "偏见", "歧视"],
            "weight": 0.9,
        },
        "courage": {
            "indicators": ["courage", "brave", "stand up", "principled",
                           "勇气", "勇敢", "坚持原则"],
            "anti_indicators": ["cowardly", "evasive", "avoid responsibility",
                                "懦弱", "逃避", "推卸责任"],
            "weight": 0.7,
        },
        "temperance": {
            "indicators": ["moderate", "balanced", "measured", "restrained",
                           "适度", "平衡", "克制", "节制"],
            "anti_indicators": ["extreme", "excessive", "obsessive",
                                "极端", "过度", "偏执"],
            "weight": 0.7,
        },
    }

    def evaluate(self, action: str, virtues: Optional[List[str]] = None,
                 **kwargs) -> float:
        """
        Evaluate action based on virtue expression.
        Score 0 = expresses vice, 1 = expresses virtue.
        """
        if not action:
            return 0.5

        action_lower = action.lower()
        action_tokens = set(re.findall(r'\w+', action_lower))

        virtue_scores = []
        virtues_to_check = virtues if virtues else list(self.CORE_VIRTUES.keys())

        for virtue_name in virtues_to_check:
            virtue_info = self.CORE_VIRTUES.get(virtue_name)
            if virtue_info is None:
                continue

            indicators = set(virtue_info["indicators"])
            anti_indicators = set(virtue_info["anti_indicators"])
            weight = virtue_info["weight"]

            pos_matches = len(action_tokens & indicators)
            neg_matches = len(action_tokens & anti_indicators)

            if pos_matches + neg_matches > 0:
                virtue_score = pos_matches / (pos_matches + neg_matches)
            else:
                virtue_score = 0.5  # Neutral

            virtue_scores.append(virtue_score * weight)

        if not virtue_scores:
            return 0.5

        # Weighted average of virtue scores
        total_weight = sum(
            self.CORE_VIRTUES[v]["weight"]
            for v in virtues_to_check
            if v in self.CORE_VIRTUES
        )
        if total_weight == 0:
            return 0.5

        weighted_score = sum(virtue_scores) / total_weight
        return max(0.0, min(1.0, weighted_score))

    def get_virtues(self) -> List[str]:
        """Return the list of core virtues."""
        return list(self.CORE_VIRTUES.keys())

    def get_description(self) -> str:
        return ("Virtue ethics framework: evaluates actions based on whether "
                "they express or cultivate moral virtues.")


class MixedFramework(EthicalFramework):
    """
    Mixed ethical framework combining utilitarian, deontological,
    and virtue ethics approaches with configurable weights.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with configurable weights.

        Args:
            weights: Dict with keys 'utilitarian', 'deontological', 'virtue_ethics'.
                     Values should sum to 1.0. Defaults to equal weights.
        """
        self._utilitarian = UtilitarianFramework()
        self._deontological = DeontologicalFramework()
        self._virtue = VirtueEthicsFramework()

        if weights is None:
            self.weights = {
                "utilitarian": 1.0 / 3.0,
                "deontological": 1.0 / 3.0,
                "virtue_ethics": 1.0 / 3.0,
            }
        else:
            total = sum(weights.values())
            if total == 0:
                raise ValueError("Sum of weights must be positive")
            self.weights = {k: v / total for k, v in weights.items()}

    def evaluate(self, action: str, context: Optional[Dict[str, Any]] = None,
                 **kwargs) -> float:
        """
        Evaluate action using weighted combination of all frameworks.

        Algorithm:
        1. Get score from each sub-framework
        2. Apply framework-specific weights
        3. Compute weighted average
        4. Apply context adjustments if context provided
        """
        if not action:
            return 0.5

        # Evaluate with each framework
        util_score = self._utilitarian.evaluate(
            action,
            consequences=kwargs.get("consequences")
        )
        deont_score = self._deontological.evaluate(
            action,
            rules=kwargs.get("rules")
        )
        virtue_score = self._virtue.evaluate(
            action,
            virtues=kwargs.get("virtues")
        )

        # Weighted combination
        mixed_score = (
            self.weights.get("utilitarian", 0) * util_score
            + self.weights.get("deontological", 0) * deont_score
            + self.weights.get("virtue_ethics", 0) * virtue_score
        )

        # Apply context adjustments
        if context:
            mixed_score = self._apply_context_adjustments(
                mixed_score, action, context
            )

        return max(0.0, min(1.0, mixed_score))

    def _apply_context_adjustments(self, base_score: float, action: str,
                                   context: Dict[str, Any]) -> float:
        """Adjust score based on contextual factors."""
        adjustment = 0.0

        # Sensitive domain penalty
        sensitive_domains = context.get("sensitive_domains", [])
        action_lower = action.lower()
        for domain in sensitive_domains:
            if domain.lower() in action_lower:
                adjustment -= 0.1

        # User vulnerability adjustment
        if context.get("user_is_vulnerable", False):
            # Higher standard for vulnerable users
            if base_score < 0.8:
                adjustment -= 0.15

        # High-stakes context
        stakes = context.get("stakes_level", "normal")
        stakes_penalty = {"low": 0.0, "normal": 0.0, "high": -0.05, "critical": -0.1}
        adjustment += stakes_penalty.get(stakes, 0.0)

        # Prior violations context
        prior_violations = context.get("prior_violations", 0)
        if prior_violations > 0:
            adjustment -= min(0.2, prior_violations * 0.05)

        return max(0.0, min(1.0, base_score + adjustment))

    def get_description(self) -> str:
        w = self.weights
        return (f"Mixed framework: utilitarian({w.get('utilitarian', 0):.0%}), "
                f"deontological({w.get('deontological', 0):.0%}), "
                f"virtue ethics({w.get('virtue_ethics', 0):.0%}).")


# =============================================================================
# SuperegoAgent - Core Review Engine
# =============================================================================

class SuperegoAgent:
    """
    Core superego agent responsible for ethical oversight and behavior review.

    Implements comprehensive behavior analysis including:
    - Goal deviation detection via TF vector cosine similarity
    - Deception detection via linguistic pattern analysis
    - Manipulation detection via psychological manipulation patterns
    - Multi-dimensional impact assessment
    - Adaptive threshold learning from human feedback
    """

    def __init__(self, config: Optional[SuperegoConfig] = None):
        self.config = config or SuperegoConfig()
        self._framework = self._create_framework()
        self._audit_trail: List[ReviewResult] = []
        self._review_history: deque = deque(maxlen=10000)
        self._feedback_history: List[Dict[str, Any]] = []
        self._review_timestamps: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._violation_counts: Dict[str, int] = defaultdict(int)
        self._review_threshold: float = self._initial_threshold()
        self._learning_rate: float = 0.05
        self._total_reviews: int = 0
        self._total_passed: int = 0

    def _create_framework(self) -> EthicalFramework:
        """Create the ethical framework based on configuration."""
        fw_type = self.config.ethical_framework
        if fw_type == "utilitarian":
            return UtilitarianFramework()
        elif fw_type == "deontological":
            return DeontologicalFramework()
        elif fw_type == "virtue_ethics":
            return VirtueEthicsFramework()
        elif fw_type == "mixed":
            return MixedFramework()
        else:
            return UtilitarianFramework()

    def _initial_threshold(self) -> float:
        """Calculate initial review threshold based on strictness level."""
        # strictness 1 -> threshold 0.3 (lenient), 5 -> 0.8 (strict)
        return 0.3 + (self.config.strictness_level - 1) * 0.125

    # -------------------------------------------------------------------------
    # TF Vector & Cosine Similarity
    # -------------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase word tokens."""
        if not text:
            return []
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def _build_tf_vector(tokens: List[str]) -> Dict[str, float]:
        """Build term-frequency vector from tokens."""
        if not tokens:
            return {}
        counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        total = len(counts)
        return {token: count / total for token, count in counts.items()}

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float],
                           vec_b: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse TF vectors."""
        if not vec_a or not vec_b:
            return 0.0

        # Dot product (only common keys)
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)

        # Magnitudes
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        return dot_product / (mag_a * mag_b)

    # -------------------------------------------------------------------------
    # Detection Methods
    # -------------------------------------------------------------------------

    def _detect_goal_deviation(self, trace: BehaviorTrace) -> Tuple[bool, float]:
        """
        Detect if the output deviates from the input goal using TF cosine similarity.

        Algorithm:
        1. Tokenize input and output texts
        2. Build TF vectors for both
        3. Compute cosine similarity
        4. Compare against deviation threshold (adjusted by strictness)
        5. Return (is_deviation, similarity_score)
        """
        input_tokens = self._tokenize(trace.input_text)
        output_tokens = self._tokenize(trace.output_text)

        if not input_tokens or not output_tokens:
            return False, 1.0  # Cannot assess, assume no deviation

        input_tf = self._build_tf_vector(input_tokens)
        output_tf = self._build_tf_vector(output_tokens)

        similarity = self._cosine_similarity(input_tf, output_tf)

        # Deviation threshold: lower strictness = more tolerance for deviation
        # strictness 1 -> threshold 0.05, strictness 5 -> threshold 0.35
        deviation_threshold = 0.05 + (self.config.strictness_level - 1) * 0.075

        is_deviation = similarity < deviation_threshold
        return is_deviation, similarity

    def _detect_deception(self, trace: BehaviorTrace) -> Tuple[bool, List[str]]:
        """
        Detect deceptive patterns in the output.

        Checks for:
        1. Absolutist statements (unverifiable claims of certainty)
        2. Unverifiable declarations (claims that cannot be fact-checked)
        3. Internal contradictions within the output
        4. False authority claims
        5. Selective information presentation
        """
        output = trace.output_text
        if not output:
            return False, []

        output_lower = output.lower()
        indicators: List[str] = []

        # 1. Absolutist statements
        absolutist_patterns = [
            (r'\b(绝对|absolutely|certainly|definitely|guaranteed|100%|always)\b',
             "绝对化表述: 使用了过于确定的表述方式"),
            (r'\b(毫无疑问|without any doubt|no question)\b',
             "绝对化表述: 声称毫无疑义"),
            (r'\b(所有人都|everyone knows|everybody agrees)\b',
             "绝对化表述: 以偏概全的表述"),
        ]
        for pattern, msg in absolutist_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        # 2. Unverifiable declarations
        unverifiable_patterns = [
            (r'\b(据内部消息|according to inside sources|secretly)\b',
             "不可验证声明: 引用无法验证的内部消息"),
            (r'\b(我亲眼看到|i personally saw|i can confirm)\b',
             "不可验证声明: AI声称个人经历"),
            (r'\b(科学已证明.*不存在|science has disproven)\b',
             "不可验证声明: 对科学结论的绝对化否定"),
        ]
        for pattern, msg in unverifiable_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        # 3. Internal contradiction detection
        contradictions = self._detect_contradictions(output)
        indicators.extend(contradictions)

        # 4. False authority claims
        authority_patterns = [
            (r'\b(作为专家|as an expert|as a doctor|as a lawyer)\b',
             "虚假权威: AI声称具有专业身份"),
            (r'\b(根据我的经验|based on my experience|in my practice)\b',
             "虚假权威: AI声称拥有个人经验"),
        ]
        for pattern, msg in authority_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        # 5. Evasion patterns
        evasion_patterns = [
            (r'\b(这个问题不重要|that question is not important|irrelevant)\b',
             "信息回避: 试图转移话题"),
            (r'\b(我不需要回答|i don\'t need to answer|that\'s beside the point)\b',
             "信息回避: 拒绝回答相关问题"),
        ]
        for pattern, msg in evasion_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        is_deceptive = len(indicators) > 0
        return is_deceptive, indicators

    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect internal contradictions within text."""
        contradictions = []
        sentences = re.split(r'[.!?。！？]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return contradictions

        # Extract key claims from each sentence
        claims = []
        for sentence in sentences:
            tokens = set(self._tokenize(sentence))
            # Identify negated and affirmed terms
            negated = set()
            affirmed = set()
            negation_words = {"不", "没", "无", "非", "not", "no", "never",
                              "neither", "none", "cannot", "can't", "don't"}
            for token in tokens:
                if token in negation_words:
                    continue
                # Check if token is negated in the sentence
                sentence_lower = sentence.lower()
                for neg_word in negation_words:
                    if re.search(rf'\b{re.escape(neg_word)}\s+\w*{re.escape(token)}',
                                 sentence_lower):
                        negated.add(token)
                        break
                else:
                    affirmed.add(token)

            claims.append({"negated": negated, "affirmed": affirmed,
                           "text": sentence})

        # Check for contradictions between sentence pairs
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                # A term affirmed in one sentence and negated in another
                conflict = claims[i]["affirmed"] & claims[j]["negated"]
                conflict |= claims[j]["affirmed"] & claims[i]["negated"]
                if conflict:
                    contradictions.append(
                        f"内部矛盾: '{claims[i]['text'][:30]}...' 与 "
                        f"'{claims[j]['text'][:30]}...' 存在矛盾表述"
                    )

        return contradictions[:3]  # Limit to top 3 contradictions

    def _detect_manipulation(self, trace: BehaviorTrace) -> Tuple[bool, List[str]]:
        """
        Detect manipulative patterns in the output.

        Checks for:
        1. Emotional manipulation keywords
        2. False urgency creation
        3. Social pressure tactics
        4. Authority exploitation
        5. Guilt tripping
        """
        output = trace.output_text
        if not output:
            return False, []

        output_lower = output.lower()
        indicators: List[str] = []

        # 1. Emotional manipulation
        emotional_patterns = [
            (r'\b(如果你不.*就会|if you don\'t.*you will|you must feel)\b',
             "情感操控: 利用恐惧心理施压"),
            (r'\b(你应该感到|you should feel guilty|shame on you)\b',
             "情感操控: 试图引发负罪感"),
            (r'\b(只有我|only i can|nobody else can|you have no choice)\b',
             "情感操控: 制造依赖感"),
            (r'\b(相信我|trust me blindly|don\'t think just do)\b',
             "情感操控: 要求盲目信任"),
        ]
        for pattern, msg in emotional_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        # 2. False urgency
        urgency_patterns = [
            (r'\b(立即|马上|立刻|urgently|immediately|right now|act now)\b',
             "虚假紧迫感: 制造不必要的时间压力"),
            (r'\b(最后机会|last chance|limited time|won\'t last)\b',
             "虚假紧迫感: 声称时间有限"),
            (r'\b(不这样做就晚了|or else it will be too late|before it\'s too late)\b',
             "虚假紧迫感: 威胁性时间限制"),
        ]
        for pattern, msg in urgency_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        # 3. Social pressure
        social_pressure_patterns = [
            (r'\b(大家都在|everyone is|everybody else|most people)\b',
             "社会压力: 利用从众心理"),
            (r'\b(别人会怎么看|what will others think|people will judge)\b',
             "社会压力: 利用社会评价焦虑"),
            (r'\b(正常人都|any normal person would|a reasonable person)\b',
             "社会压力: 标签化正常行为"),
        ]
        for pattern, msg in social_pressure_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        # 4. Authority exploitation
        authority_patterns = [
            (r'\b(专家说|experts say|studies show.*without citation)\b',
             "权威利用: 笼统引用权威但无具体来源"),
            (r'\b(根据法律|according to law.*without reference)\b',
             "权威利用: 笼统引用法律但无具体条文"),
        ]
        for pattern, msg in authority_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        # 5. Guilt tripping
        guilt_patterns = [
            (r'\b(想想你的|think about your.*would want|after all i\'ve done)\b',
             "道德绑架: 利用关系施压"),
            (r'\b(你欠我的|you owe me|after all i did for you)\b',
             "道德绑架: 声称对方亏欠"),
        ]
        for pattern, msg in guilt_patterns:
            if re.search(pattern, output_lower):
                indicators.append(msg)

        is_manipulative = len(indicators) > 0
        return is_manipulative, indicators

    def _assess_impact(self, trace: BehaviorTrace) -> ImpactAssessment:
        """
        Multi-dimensional impact assessment of the agent's action.

        Evaluates:
        - user_impact: Direct effect on the user
        - third_party_impact: Effect on other parties
        - short_term_risk: Immediate risks
        - long_term_risk: Extended risks
        - reversibility: How reversible the action is
        - overall_impact: Weighted combination
        """
        output = trace.output_text
        context = trace.context or {}
        action_type = trace.action_type

        # Base impact by action type
        action_impact_map = {
            "generate": 0.3,
            "tool_call": 0.6,
            "memory_access": 0.5,
            "planning": 0.4,
        }
        base_impact = action_impact_map.get(action_type, 0.3)

        # Analyze output for impact signals
        output_lower = (output or "").lower()
        output_tokens = set(re.findall(r'\w+', output_lower))

        # User impact assessment
        user_impact_keywords = {
            "personal": {"your", "you", "个人", "你的", "您"},
            "financial": {"money", "cost", "pay", "钱", "费用", "支付"},
            "health": {"health", "medical", "treatment", "健康", "医疗", "治疗"},
            "safety": {"safety", "danger", "risk", "安全", "危险", "风险"},
        }
        user_impact = base_impact
        for category, keywords in user_impact_keywords.items():
            if output_tokens & keywords:
                user_impact = min(1.0, user_impact + 0.15)

        # Third-party impact assessment
        third_party_keywords = {
            "others": {"others", "people", "they", "他人", "别人", "他们"},
            "society": {"society", "public", "community", "社会", "公众", "社区"},
            "environment": {"environment", "ecosystem", "nature", "环境", "生态"},
            "children": {"children", "minors", "kids", "儿童", "未成年人", "孩子"},
        }
        third_party_impact = 0.1
        for category, keywords in third_party_keywords.items():
            if output_tokens & keywords:
                third_party_impact = min(1.0, third_party_impact + 0.2)

        # Short-term risk
        short_term_risk = 0.1
        risk_indicators = {"immediate", "urgent", "now", "quick", "fast",
                           "立即", "紧急", "马上", "快速", "立刻"}
        if output_tokens & risk_indicators:
            short_term_risk += 0.3
        if action_type == "tool_call":
            short_term_risk += 0.2  # Tool calls have immediate effect

        # Long-term risk
        long_term_risk = 0.1
        persistence_indicators = {"permanent", "irreversible", "forever",
                                  "长期", "永久", "不可逆", "永远"}
        if output_tokens & persistence_indicators:
            long_term_risk += 0.4
        if action_type == "planning":
            long_term_risk += 0.2  # Plans have extended effects

        # Reversibility assessment
        reversibility = 0.8  # Default: mostly reversible
        irreversible_indicators = {"delete", "remove", "destroy", "permanent",
                                   "删除", "移除", "销毁", "永久"}
        reversible_indicators = {"undo", "revert", "restore", "recover",
                                 "撤销", "恢复", "还原", "找回"}
        if output_tokens & irreversible_indicators:
            reversibility -= 0.4
        if output_tokens & reversible_indicators:
            reversibility += 0.2
        reversibility = max(0.0, min(1.0, reversibility))

        # Context adjustments
        if context.get("high_stakes", False):
            user_impact = min(1.0, user_impact + 0.2)
            short_term_risk = min(1.0, short_term_risk + 0.15)
            long_term_risk = min(1.0, long_term_risk + 0.15)

        if context.get("affects_vulnerable", False):
            user_impact = min(1.0, user_impact + 0.3)
            third_party_impact = min(1.0, third_party_impact + 0.2)

        # Overall impact: weighted combination
        overall_impact = (
            0.30 * user_impact
            + 0.20 * third_party_impact
            + 0.20 * short_term_risk
            + 0.20 * long_term_risk
            + 0.10 * (1.0 - reversibility)  # Lower reversibility = higher impact
        )

        return ImpactAssessment(
            user_impact=max(0.0, min(1.0, user_impact)),
            third_party_impact=max(0.0, min(1.0, third_party_impact)),
            short_term_risk=max(0.0, min(1.0, short_term_risk)),
            long_term_risk=max(0.0, min(1.0, long_term_risk)),
            reversibility=max(0.0, min(1.0, reversibility)),
            overall_impact=max(0.0, min(1.0, overall_impact)),
        )

    # -------------------------------------------------------------------------
    # Correction Generation
    # -------------------------------------------------------------------------

    def _generate_correction(self, trace: BehaviorTrace,
                             violations: List[str]) -> str:
        """
        Generate correction suggestions based on detected violations.

        Algorithm:
        1. Categorize violations by type
        2. Generate specific correction for each category
        3. Prioritize corrections by severity
        4. Compose final correction message
        """
        if not violations:
            return ""

        corrections: List[str] = []
        violation_categories = self._categorize_violations(violations)

        # Goal deviation correction
        if "goal_deviation" in violation_categories:
            corrections.append(
                "建议: 请确保回复内容与用户的问题或请求直接相关。"
                "当前回复可能偏离了原始目标，请重新审视用户意图后作答。"
            )

        # Deception correction
        if "deception" in violation_categories:
            deception_items = violation_categories["deception"]
            corrections.append(
                f"建议: 检测到潜在的欺骗性表述({len(deception_items)}项)。"
                "请避免使用绝对化表述、不可验证的声明或虚假权威。"
                "确保信息准确、可验证，并对不确定性保持诚实。"
            )

        # Manipulation correction
        if "manipulation" in violation_categories:
            manipulation_items = violation_categories["manipulation"]
            corrections.append(
                f"建议: 检测到潜在的操纵性表述({len(manipulation_items)}项)。"
                "请避免使用情感操控、虚假紧迫感或社会压力策略。"
                "尊重用户的自主决策权，提供客观信息。"
            )

        # Ethical violation correction
        if "ethical" in violation_categories:
            corrections.append(
                "建议: 回复可能存在伦理问题。请确保内容符合道德规范，"
                "不造成伤害，尊重用户权利，并提供公正客观的信息。"
            )

        # High impact correction
        if "high_impact" in violation_categories:
            corrections.append(
                "建议: 该回复可能产生显著影响。请在发布前仔细评估"
                "潜在后果，确保对用户和第三方的影响是积极的。"
            )

        if not corrections:
            corrections.append(
                "建议: 请审查回复内容，确保其符合安全、诚实和有益的原则。"
            )

        # Compose final correction
        header = f"[超我审核修正建议 - Trace: {trace.trace_id}]"
        body = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(corrections))
        footer = "请根据以上建议修正回复内容后重新提交审核。"

        return f"{header}\n{body}\n{footer}"

    def _categorize_violations(self, violations: List[str]) -> Dict[str, List[str]]:
        """Categorize violation strings by type."""
        categories: Dict[str, List[str]] = {}

        for violation in violations:
            if "偏离" in violation or "deviation" in violation.lower():
                categories.setdefault("goal_deviation", []).append(violation)
            elif "欺骗" in violation or "虚假" in violation or "矛盾" in violation:
                categories.setdefault("deception", []).append(violation)
            elif "操控" in violation or "操纵" in violation or "紧迫" in violation:
                categories.setdefault("manipulation", []).append(violation)
            elif "伦理" in violation or "ethical" in violation.lower():
                categories.setdefault("ethical", []).append(violation)
            elif "影响" in violation or "impact" in violation.lower():
                categories.setdefault("high_impact", []).append(violation)
            else:
                categories.setdefault("other", []).append(violation)

        return categories

    # -------------------------------------------------------------------------
    # Core Review Methods
    # -------------------------------------------------------------------------

    def review_behavior(self, trace: BehaviorTrace) -> ReviewResult:
        """
        Perform a comprehensive review of a behavior trace.

        Review pipeline:
        1. Rate limiting check
        2. Goal deviation detection
        3. Deception detection
        4. Manipulation detection
        5. Ethical framework evaluation
        6. Impact assessment
        7. Aggregate results and determine pass/fail
        8. Generate corrections if needed
        """
        start_time = time.time()

        with self._lock:
            # Rate limiting
            now = time.time()
            self._review_timestamps.append(now)
            # Remove timestamps older than 60 seconds
            while self._review_timestamps and self._review_timestamps[0] < now - 60:
                self._review_timestamps.popleft()
            if len(self._review_timestamps) > self.config.max_reviews_per_minute:
                # Throttle: still review but flag it
                pass

            violations: List[str] = []
            warnings: List[str] = []
            ethical_score = 1.0

            # 1. Goal deviation detection
            if self.config.review_scope in ("full", "targeted"):
                is_deviation, similarity = self._detect_goal_deviation(trace)
                if is_deviation:
                    violations.append(
                        f"目标偏离: 输入输出相似度 {similarity:.3f} 低于阈值"
                    )
                    ethical_score -= 0.2
                elif similarity < 0.3:
                    warnings.append(
                        f"低相似度警告: 输入输出相似度 {similarity:.3f}"
                    )

            # 2. Deception detection
            if self.config.review_scope == "full":
                is_deceptive, deception_indicators = self._detect_deception(trace)
                if is_deceptive:
                    violations.extend(deception_indicators)
                    ethical_score -= 0.15 * len(deception_indicators)

            # 3. Manipulation detection
            if self.config.review_scope == "full":
                is_manipulative, manipulation_indicators = self._detect_manipulation(trace)
                if is_manipulative:
                    violations.extend(manipulation_indicators)
                    ethical_score -= 0.15 * len(manipulation_indicators)

            # 4. Ethical framework evaluation
            ethical_eval = self._framework.evaluate(
                trace.output_text,
                context=trace.context,
            )
            ethical_score = max(0.0, ethical_score * ethical_eval)

            # 5. Impact assessment
            impact = self._assess_impact(trace)

            # High impact warning
            if impact.overall_impact > 0.6:
                warnings.append(
                    f"高影响警告: 综合影响评分 {impact.overall_impact:.2f}"
                )
            if impact.overall_impact > 0.8:
                violations.append(
                    f"极高影响: 综合影响评分 {impact.overall_impact:.2f}，需要特别关注"
                )

            # 6. Clamp ethical score
            ethical_score = max(0.0, min(1.0, ethical_score))

            # 7. Determine pass/fail
            passed = ethical_score >= self._review_threshold
            risk_level = self._determine_risk_level(ethical_score, impact)

            # 8. Generate correction if failed
            correction = ""
            if not passed and self.config.auto_correct:
                correction = self._generate_correction(trace, violations)

            # Track violation counts
            for v in violations:
                category = v.split(":")[0] if ":" in v else v[:20]
                self._violation_counts[category] += 1

            review_time = time.time() - start_time

            result = ReviewResult(
                passed=passed,
                risk_level=risk_level,
                violations=violations,
                warnings=warnings,
                ethical_score=ethical_score,
                impact_assessment=impact,
                correction=correction,
                review_time=review_time,
                metadata={
                    "trace_id": trace.trace_id,
                    "agent_id": trace.agent_id,
                    "action_type": trace.action_type,
                    "framework": self.config.ethical_framework,
                    "threshold": self._review_threshold,
                },
            )

            # Store results
            self._audit_trail.append(result)
            self._review_history.append({
                "trace_id": trace.trace_id,
                "agent_id": trace.agent_id,
                "result": result,
                "timestamp": now,
            })
            self._total_reviews += 1
            if passed:
                self._total_passed += 1

        return result

    def _determine_risk_level(self, ethical_score: float,
                              impact: ImpactAssessment) -> str:
        """Determine risk level from ethical score and impact assessment."""
        # Combined risk metric
        risk = (1.0 - ethical_score) * 0.6 + impact.overall_impact * 0.4

        if risk >= 0.75:
            return RiskLevel.CRITICAL
        elif risk >= 0.5:
            return RiskLevel.HIGH
        elif risk >= 0.25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def should_intervene(self, trace: BehaviorTrace) -> bool:
        """
        Determine if intervention is needed for a behavior trace.

        Intervention criteria:
        1. Ethical score below critical threshold
        2. Critical risk level
        3. High-impact actions with low ethical scores
        4. Pattern of repeated violations by the same agent
        """
        result = self.review_behavior(trace)

        if not result.passed:
            return True

        if result.risk_level == RiskLevel.CRITICAL:
            return True

        if (result.risk_level == RiskLevel.HIGH
                and result.impact_assessment is not None
                and result.impact_assessment.overall_impact > 0.7):
            return True

        # Check for repeated violations by the same agent
        agent_id = trace.agent_id
        recent_violations = 0
        with self._lock:
            for entry in list(self._review_history)[-50:]:
                if (entry["agent_id"] == agent_id
                        and not entry["result"].passed):
                    recent_violations += 1

        # If agent has >3 violations in last 50 reviews, intervene
        if recent_violations >= 3:
            return True

        return False

    def review_batch(self, traces: List[BehaviorTrace]) -> List[ReviewResult]:
        """Review a batch of behavior traces."""
        return [self.review_behavior(trace) for trace in traces]

    # -------------------------------------------------------------------------
    # Audit & Statistics
    # -------------------------------------------------------------------------

    def get_audit_trail(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the audit trail, optionally filtered by agent_id."""
        with self._lock:
            if agent_id is None:
                return [
                    {**r.to_dict(), "index": i}
                    for i, r in enumerate(self._audit_trail)
                ]
            return [
                {**r.to_dict(), "index": i}
                for i, r in enumerate(self._audit_trail)
                if r.metadata.get("agent_id") == agent_id
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get review statistics."""
        with self._lock:
            total = self._total_reviews
            passed = self._total_passed
            failed = total - passed
            pass_rate = passed / total if total > 0 else 0.0

            # Average ethical score
            if self._audit_trail:
                avg_score = statistics.mean(
                    r.ethical_score for r in self._audit_trail
                )
                avg_review_time = statistics.mean(
                    r.review_time for r in self._audit_trail
                )
            else:
                avg_score = 0.0
                avg_review_time = 0.0

            # Risk level distribution
            risk_counts: Dict[str, int] = defaultdict(int)
            for r in self._audit_trail:
                risk_counts[r.risk_level] += 1

            # Top violation categories
            top_violations = sorted(
                self._violation_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            return {
                "total_reviews": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate,
                "average_ethical_score": avg_score,
                "average_review_time_ms": avg_review_time * 1000,
                "risk_level_distribution": dict(risk_counts),
                "top_violation_categories": top_violations,
                "current_threshold": self._review_threshold,
                "strictness_level": self.config.strictness_level,
                "framework": self.config.ethical_framework,
            }

    # -------------------------------------------------------------------------
    # Learning from Feedback
    # -------------------------------------------------------------------------

    def learn_from_feedback(self, trace: BehaviorTrace,
                            human_feedback: str) -> Dict[str, Any]:
        """
        Learn from human feedback to improve review accuracy.

        Feedback types:
        - "approve": The review was correct
        - "reject": The review was incorrect (false positive)
        - "escalate": The review missed something (false negative)

        Algorithm:
        1. Parse feedback type and severity
        2. Find the corresponding review result
        3. Adjust internal thresholds based on feedback
        4. Update violation pattern weights
        5. Record feedback for future reference
        """
        feedback_lower = human_feedback.lower().strip()
        feedback_record: Dict[str, Any] = {
            "trace_id": trace.trace_id,
            "agent_id": trace.agent_id,
            "feedback": human_feedback,
            "timestamp": time.time(),
        }

        # Parse feedback
        if any(kw in feedback_lower for kw in
               ["approve", "correct", "同意", "正确", "批准"]):
            feedback_type = "approve"
            feedback_record["type"] = "approve"
        elif any(kw in feedback_lower for kw in
                 ["reject", "false positive", "误报", "拒绝", "错误"]):
            feedback_type = "reject"
            feedback_record["type"] = "reject"
        elif any(kw in feedback_lower for kw in
                 ["escalate", "missed", "false negative", "漏报", "升级"]):
            feedback_type = "escalate"
            feedback_record["type"] = "escalate"
        else:
            feedback_type = "unknown"
            feedback_record["type"] = "unknown"

        # Find the corresponding review result
        matching_result = None
        with self._lock:
            for entry in list(self._review_history):
                if entry["trace_id"] == trace.trace_id:
                    matching_result = entry["result"]
                    break

        adjustments: Dict[str, Any] = {}

        if feedback_type == "reject" and matching_result is not None:
            # False positive: our threshold was too strict
            adjustments["threshold_adjustment"] = -self._learning_rate * 0.5
            feedback_record["action"] = "lower_threshold"

        elif feedback_type == "escalate" and matching_result is not None:
            # False negative: our threshold was too lenient
            adjustments["threshold_adjustment"] = self._learning_rate * 0.5
            feedback_record["action"] = "raise_threshold"

        elif feedback_type == "approve":
            # Confirm current threshold is appropriate
            adjustments["threshold_adjustment"] = 0.0
            feedback_record["action"] = "maintain_threshold"

        # Apply threshold adjustment
        if adjustments.get("threshold_adjustment") is not None:
            self._update_review_threshold(
                adjustments["threshold_adjustment"]
            )

        # Store feedback
        with self._lock:
            self._feedback_history.append(feedback_record)

        adjustments["new_threshold"] = self._review_threshold
        adjustments["feedback_type"] = feedback_type
        adjustments["total_feedback_count"] = len(self._feedback_history)

        return adjustments

    def _update_review_threshold(self, delta: float = None) -> float:
        """
        Dynamically adjust the review threshold based on feedback history.

        Algorithm:
        1. If delta provided, apply direct adjustment
        2. Otherwise, compute adjustment from recent feedback patterns
        3. Clamp threshold to valid range [0.1, 0.95]
        4. Apply momentum to prevent oscillation
        """
        if delta is not None:
            new_threshold = self._review_threshold + delta
        else:
            # Compute from recent feedback
            with self._lock:
                recent_feedback = list(self._feedback_history)[-100:]

            if len(recent_feedback) < 5:
                return self._review_threshold

            # Count feedback types
            approvals = sum(1 for f in recent_feedback
                           if f.get("type") == "approve")
            rejections = sum(1 for f in recent_feedback
                            if f.get("type") == "reject")
            escalations = sum(1 for f in recent_feedback
                            if f.get("type") == "escalate")

            total = len(recent_feedback)
            # Net signal: escalations push threshold up, rejections push down
            net_signal = (escalations - rejections) / total
            # Apply learning rate with momentum
            momentum = 0.9  # High momentum to prevent oscillation
            adjustment = self._learning_rate * net_signal * (1.0 - momentum)
            new_threshold = self._review_threshold + adjustment

        # Clamp to valid range
        min_threshold = 0.1
        max_threshold = 0.95
        new_threshold = max(min_threshold, min(max_threshold, new_threshold))

        self._review_threshold = new_threshold
        return new_threshold


# =============================================================================
# SuperegoMonitor - Health Monitoring
# =============================================================================

class SuperegoMonitor:
    """
    Monitoring system for the SuperegoAgent.

    Provides:
    - Periodic health checks
    - Alert triggering for anomalies
    - Performance metrics tracking
    """

    def __init__(self, agent: SuperegoAgent, interval: float = 30.0):
        self.agent = agent
        self.interval = interval
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_monitoring = False
        self._start_time: Optional[float] = None
        self._alerts: deque = deque(maxlen=100)
        self._health_history: deque = deque(maxlen=1000)
        self._error_count: int = 0
        self._lock = threading.Lock()

    def start_monitoring(self, interval: Optional[float] = None) -> None:
        """
        Start the monitoring loop in a background thread.

        Args:
            interval: Check interval in seconds. Uses default if not provided.
        """
        if self._is_monitoring:
            return

        if interval is not None:
            self.interval = interval

        self._stop_event.clear()
        self._is_monitoring = True
        self._start_time = time.time()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SuperegoMonitor",
        )
        self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        if not self._is_monitoring:
            return

        self._stop_event.set()
        self._is_monitoring = False

        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=5.0)
            self._monitoring_thread = None

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                health = self.check_health()
                with self._lock:
                    self._health_history.append(health)

                # Check for alert conditions
                if not health.is_healthy:
                    self._trigger_alert(health)

                # Check review rate
                if health.reviews_per_minute > self.agent.config.max_reviews_per_minute:
                    self._trigger_alert(health, reason="rate_limit_exceeded")

            except Exception as e:
                with self._lock:
                    self._error_count += 1
                self._trigger_alert(
                    None,
                    reason=f"monitoring_error: {str(e)}"
                )

            # Wait for next interval or stop signal
            self._stop_event.wait(timeout=self.interval)

    def check_health(self) -> HealthStatus:
        """
        Perform a health check of the superego agent.

        Returns:
            HealthStatus with current health metrics.
        """
        stats = self.agent.get_statistics()
        now = time.time()

        # Calculate uptime
        uptime = now - self._start_time if self._start_time else 0.0

        # Calculate reviews per minute from recent history
        with self.agent._lock:
            recent_timestamps = list(self.agent._review_timestamps)
        recent_count = sum(1 for t in recent_timestamps if t > now - 60)
        reviews_per_minute = float(recent_count)

        # Average review time
        avg_review_time = stats.get("average_review_time_ms", 0.0) / 1000.0

        # Determine health
        is_healthy = True
        if avg_review_time > 1.0:  # Reviews taking more than 1 second
            is_healthy = False
        if stats.get("pass_rate", 1.0) < 0.5:  # Less than 50% pass rate
            is_healthy = False

        # Count pending alerts
        with self._lock:
            pending_alerts = len(self._alerts)
            error_count = self._error_count

        return HealthStatus(
            is_healthy=is_healthy,
            uptime_seconds=uptime,
            total_reviews=stats.get("total_reviews", 0),
            reviews_per_minute=reviews_per_minute,
            average_review_time=avg_review_time,
            error_count=error_count,
            last_check_time=now,
            alerts_pending=pending_alerts,
        )

    def _trigger_alert(self, health: Optional[HealthStatus] = None,
                       reason: str = "") -> None:
        """
        Trigger an alert for anomalous conditions.

        Records the alert and logs relevant information.
        """
        alert: Dict[str, Any] = {
            "timestamp": time.time(),
            "reason": reason,
        }

        if health is not None:
            alert["health_snapshot"] = {
                "is_healthy": health.is_healthy,
                "total_reviews": health.total_reviews,
                "reviews_per_minute": health.reviews_per_minute,
                "average_review_time": health.average_review_time,
                "error_count": health.error_count,
            }

        with self._lock:
            self._alerts.append(alert)

    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        with self._lock:
            alerts = list(self._alerts)
        return alerts[-limit:]

    def get_health_history(self, limit: int = 20) -> List[HealthStatus]:
        """Get recent health check results."""
        with self._lock:
            history = list(self._health_history)
        return history[-limit:]
