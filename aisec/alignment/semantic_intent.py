"""
Semantic Intent Verification Module

Provides intent embedding, behavior-intent consistency scoring,
drift detection, intent decomposition, and semantic similarity metrics
for verifying that agent behavior aligns with declared intentions.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class IntentCategory(Enum):
    """Categories of agent intents."""
    INFORMATION = "information"
    ACTION = "action"
    ANALYSIS = "analysis"
    CREATION = "creation"
    MODIFICATION = "modification"
    DELETION = "deletion"
    COMMUNICATION = "communication"
    NAVIGATION = "navigation"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"


class DriftSeverity(Enum):
    """Severity of intent drift."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class IntentVector:
    """Sparse vector representation of an intent."""
    dimensions: Dict[str, float] = field(default_factory=dict)
    norm: float = 0.0

    def __post_init__(self) -> None:
        self.norm = math.sqrt(sum(v * v for v in self.dimensions.values())) if self.dimensions else 0.0

    def cosine_similarity(self, other: IntentVector) -> float:
        if self.norm == 0.0 or other.norm == 0.0:
            return 0.0
        dot_product = sum(
            self.dimensions.get(k, 0.0) * other.dimensions.get(k, 0.0)
            for k in set(self.dimensions) | set(other.dimensions)
        )
        return dot_product / (self.norm * other.norm)

    def euclidean_distance(self, other: IntentVector) -> float:
        all_keys = set(self.dimensions) | set(other.dimensions)
        return math.sqrt(
            sum(
                (self.dimensions.get(k, 0.0) - other.dimensions.get(k, 0.0)) ** 2
                for k in all_keys
            )
        )

    def normalize(self) -> IntentVector:
        if self.norm == 0.0:
            return IntentVector()
        return IntentVector(
            dimensions={k: v / self.norm for k, v in self.dimensions.items()}
        )


@dataclass
class IntentProfile:
    """Complete profile of an agent's intent."""
    intent_id: str
    description: str
    category: IntentCategory
    embedding: IntentVector = field(default_factory=IntentVector)
    constraints: List[str] = field(default_factory=list)
    expected_behaviors: List[str] = field(default_factory=list)
    forbidden_behaviors: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "description": self.description,
            "category": self.category.value,
            "embedding_dimensions": len(self.embedding.dimensions),
            "constraints": self.constraints,
            "expected_behaviors": self.expected_behaviors,
            "forbidden_behaviors": self.forbidden_behaviors,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class BehaviorObservation:
    """A single observation of agent behavior."""
    observation_id: str
    timestamp: float
    action_description: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "action_description": self.action_description,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "outcome": self.outcome,
            "context": self.context,
        }


@dataclass
class DriftReport:
    """Report on detected intent drift."""
    drift_id: str
    intent_id: str
    severity: DriftSeverity
    drift_score: float
    original_intent: str
    observed_behavior_summary: str
    detected_at: float = field(default_factory=time.time)
    contributing_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "intent_id": self.intent_id,
            "severity": self.severity.value,
            "drift_score": self.drift_score,
            "original_intent": self.original_intent,
            "observed_behavior_summary": self.observed_behavior_summary,
            "detected_at": self.detected_at,
            "contributing_factors": self.contributing_factors,
            "recommendations": self.recommendations,
        }


@dataclass
class DecomposedIntent:
    """Result of intent decomposition into sub-intents."""
    parent_intent_id: str
    sub_intents: List[IntentProfile] = field(default_factory=list)
    decomposition_strategy: str = "hierarchical"
    completeness_score: float = 0.0
    coherence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_intent_id": self.parent_intent_id,
            "sub_intent_count": len(self.sub_intents),
            "sub_intents": [si.to_dict() for si in self.sub_intents],
            "decomposition_strategy": self.decomposition_strategy,
            "completeness_score": self.completeness_score,
            "coherence_score": self.coherence_score,
        }


class SemanticSimilarity:
    """Computes semantic similarity between texts using multiple metrics."""

    def __init__(self) -> None:
        self._stop_words: Set[str] = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such", "no",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "because", "but", "and", "or", "if", "while", "that", "this", "these",
            "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "they", "them", "their", "what",
            "which", "who", "whom",
        }
        self._synonym_groups: Dict[str, Set[str]] = {
            "get": {"retrieve", "fetch", "obtain", "acquire", "collect"},
            "delete": {"remove", "erase", "destroy", "eliminate", "drop"},
            "create": {"make", "build", "generate", "produce", "construct"},
            "modify": {"change", "update", "alter", "edit", "revise"},
            "search": {"find", "look", "seek", "query", "locate"},
            "send": {"transmit", "dispatch", "deliver", "forward", "push"},
            "read": {"view", "display", "show", "list", "get"},
            "write": {"save", "store", "persist", "record", "insert"},
            "analyze": {"examine", "inspect", "evaluate", "assess", "review"},
            "connect": {"join", "link", "attach", "bind", "associate"},
        }

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9_]+\b', text)
        return [t for t in tokens if t not in self._stop_words and len(t) > 1]

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        set1 = set(self.tokenize(text1))
        set2 = set(self.tokenize(text2))
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union)

    def cosine_similarity_text(self, text1: str, text2: str) -> float:
        vec1 = self._text_to_vector(text1)
        vec2 = self._text_to_vector(text2)
        dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        ngrams1 = self._generate_ngrams(text1.lower(), n)
        ngrams2 = self._generate_ngrams(text2.lower(), n)
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        c1 = Counter(ngrams1)
        c2 = Counter(ngrams2)
        intersection = sum((c1 & c2).values())
        union = sum((c1 | c2).values())
        return intersection / union if union else 0.0

    def edit_distance_similarity(self, text1: str, text2: str) -> float:
        words1 = self.tokenize(text1)
        words2 = self.tokenize(text2)
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        total_dist = 0.0
        comparisons = 0
        for w1 in words1:
            min_dist = min(self._levenshtein(w1, w2) for w2 in words2)
            max_len = max(len(w1), 1)
            total_dist += 1.0 - (min_dist / max_len)
            comparisons += 1
        return total_dist / comparisons if comparisons else 0.0

    def expanded_similarity(self, text1: str, text2: str) -> float:
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        expanded1: Set[str] = set()
        expanded2: Set[str] = set()
        for token in tokens1:
            expanded1.add(token)
            for group in self._synonym_groups.values():
                if token in group:
                    expanded1.update(group)
        for token in tokens2:
            expanded2.add(token)
            for group in self._synonym_groups.values():
                if token in group:
                    expanded2.update(group)
        if not expanded1 and not expanded2:
            return 1.0
        if not expanded1 or not expanded2:
            return 0.0
        intersection = expanded1 & expanded2
        union = expanded1 | expanded2
        return len(intersection) / len(union)

    def combined_similarity(
        self, text1: str, text2: str, weights: Optional[Dict[str, float]] = None
    ) -> float:
        if weights is None:
            weights = {
                "jaccard": 0.25,
                "cosine": 0.30,
                "ngram": 0.20,
                "edit": 0.10,
                "expanded": 0.15,
            }
        scores = {
            "jaccard": self.jaccard_similarity(text1, text2),
            "cosine": self.cosine_similarity_text(text1, text2),
            "ngram": self.ngram_similarity(text1, text2),
            "edit": self.edit_distance_similarity(text1, text2),
            "expanded": self.expanded_similarity(text1, text2),
        }
        total = sum(scores.get(k, 0) * v for k, v in weights.items())
        weight_sum = sum(weights.values())
        return total / weight_sum if weight_sum else 0.0

    def _text_to_vector(self, text: str) -> Dict[str, float]:
        tokens = self.tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        max_count = max(counts.values())
        return {k: v / max_count for k, v in counts.items()}

    def _generate_ngrams(self, text: str, n: int) -> List[str]:
        words = text.split()
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return SemanticSimilarity._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]


class IntentEmbedding:
    """Generates intent embeddings using bag-of-words and TF-IDF-like approaches."""

    def __init__(self, vocabulary_size: int = 5000) -> None:
        self.vocabulary_size: int = vocabulary_size
        self._vocabulary: Dict[str, int] = {}
        self._idf_scores: Dict[str, float] = {}
        self._document_count: int = 0
        self._category_keywords: Dict[IntentCategory, Set[str]] = {
            IntentCategory.INFORMATION: {"what", "tell", "describe", "explain", "info", "detail", "show", "list"},
            IntentCategory.ACTION: {"do", "execute", "run", "perform", "start", "stop", "trigger", "invoke"},
            IntentCategory.ANALYSIS: {"analyze", "evaluate", "assess", "compare", "measure", "compute", "calculate"},
            IntentCategory.CREATION: {"create", "make", "build", "generate", "new", "add", "construct", "design"},
            IntentCategory.MODIFICATION: {"change", "update", "modify", "edit", "alter", "adjust", "revise", "set"},
            IntentCategory.DELETION: {"delete", "remove", "erase", "destroy", "drop", "clear", "clean"},
            IntentCategory.COMMUNICATION: {"send", "notify", "message", "email", "alert", "inform", "contact"},
            IntentCategory.NAVIGATION: {"go", "navigate", "open", "switch", "browse", "visit", "access"},
            IntentCategory.AUTHENTICATION: {"login", "logout", "auth", "verify", "identify", "authenticate"},
            IntentCategory.CONFIGURATION: {"config", "setting", "configure", "setup", "enable", "disable", "option"},
        }

    def build_vocabulary(self, documents: List[str]) -> None:
        word_doc_count: Dict[str, int] = {}
        for doc in documents:
            tokens = set(re.findall(r'\b[a-z0-9_]+\b', doc.lower()))
            for token in tokens:
                word_doc_count[token] = word_doc_count.get(token, 0) + 1
        sorted_words = sorted(word_doc_count.items(), key=lambda x: -x[1])
        self._vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:self.vocabulary_size])}
        self._document_count = len(documents)
        self._idf_scores = {
            word: math.log((self._document_count + 1) / (count + 1)) + 1
            for word, count in word_doc_count.items()
            if word in self._vocabulary
        }

    def embed(self, text: str) -> IntentVector:
        tokens = re.findall(r'\b[a-z0-9_]+\b', text.lower())
        tf_counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        dimensions: Dict[str, float] = {}
        for token, count in tf_counts.items():
            tf = count / total
            idf = self._idf_scores.get(token, 1.0)
            dimensions[token] = tf * idf
        category_scores = self._compute_category_scores(tokens)
        for cat, score in category_scores.items():
            dimensions[f"_cat_{cat.value}"] = score
        return IntentVector(dimensions=dimensions)

    def embed_behavior(self, observation: BehaviorObservation) -> IntentVector:
        combined = f"{observation.action_description} {observation.action_type} {observation.outcome} {observation.context}"
        return self.embed(combined)

    def _compute_category_scores(self, tokens: List[str]) -> Dict[IntentCategory, float]:
        token_set = set(tokens)
        scores: Dict[IntentCategory, float] = {}
        for category, keywords in self._category_keywords.items():
            overlap = token_set & keywords
            scores[category] = len(overlap) / len(keywords) if keywords else 0.0
        return scores

    def infer_category(self, text: str) -> IntentCategory:
        tokens = set(re.findall(r'\b[a-z0-9_]+\b', text.lower()))
        best_category = IntentCategory.INFORMATION
        best_score = 0.0
        for category, keywords in self._category_keywords.items():
            overlap = tokens & keywords
            score = len(overlap) / len(keywords) if keywords else 0.0
            if score > best_score:
                best_score = score
                best_category = category
        return best_category


class ConsistencyScorer:
    """Scores the consistency between declared intent and observed behavior."""

    def __init__(
        self,
        similarity_engine: Optional[SemanticSimilarity] = None,
        embedding_engine: Optional[IntentEmbedding] = None,
    ) -> None:
        self.similarity: SemanticSimilarity = similarity_engine or SemanticSimilarity()
        self.embedding: IntentEmbedding = embedding_engine or IntentEmbedding()
        self._behavior_window: int = 100
        self._behavior_buffer: Dict[str, List[BehaviorObservation]] = {}

    def score_single(
        self, intent: IntentProfile, observation: BehaviorObservation
    ) -> Dict[str, float]:
        text_sim = self.similarity.combined_similarity(
            intent.description, observation.action_description
        )
        behavior_sim = 0.0
        for expected in intent.expected_behaviors:
            sim = self.similarity.combined_similarity(
                expected, observation.action_description
            )
            behavior_sim = max(behavior_sim, sim)
        forbidden_sim = 0.0
        for forbidden in intent.forbidden_behaviors:
            sim = self.similarity.combined_similarity(
                forbidden, observation.action_description
            )
            forbidden_sim = max(forbidden_sim, sim)
        intent_embedding = intent.embedding
        behavior_embedding = self.embedding.embed_behavior(observation)
        embedding_sim = intent_embedding.cosine_similarity(behavior_embedding)
        consistency = (
            text_sim * 0.3
            + behavior_sim * 0.3
            + (1.0 - forbidden_sim) * 0.2
            + embedding_sim * 0.2
        )
        return {
            "text_similarity": text_sim,
            "behavior_similarity": behavior_sim,
            "forbidden_similarity": forbidden_sim,
            "embedding_similarity": embedding_sim,
            "overall_consistency": max(0.0, min(1.0, consistency)),
        }

    def score_window(
        self, intent: IntentProfile, observations: List[BehaviorObservation]
    ) -> Dict[str, Any]:
        if not observations:
            return {
                "average_consistency": 0.0,
                "min_consistency": 0.0,
                "max_consistency": 0.0,
                "trend": "stable",
                "violation_count": 0,
            }
        scores = [self.score_single(intent, obs) for obs in observations]
        consistencies = [s["overall_consistency"] for s in scores]
        violations = sum(1 for c in consistencies if c < 0.3)
        trend = self._compute_trend(consistencies)
        return {
            "average_consistency": sum(consistencies) / len(consistencies),
            "min_consistency": min(consistencies),
            "max_consistency": max(consistencies),
            "std_dev": self._std_deviation(consistencies),
            "trend": trend,
            "violation_count": violations,
            "score_details": scores[-10:],
        }

    def add_observation(self, intent_id: str, observation: BehaviorObservation) -> None:
        if intent_id not in self._behavior_buffer:
            self._behavior_buffer[intent_id] = []
        self._behavior_buffer[intent_id].append(observation)
        if len(self._behavior_buffer[intent_id]) > self._behavior_window:
            self._behavior_buffer[intent_id] = self._behavior_buffer[intent_id][-self._behavior_window:]

    def get_observations(self, intent_id: str) -> List[BehaviorObservation]:
        return list(self._behavior_buffer.get(intent_id, []))

    def _compute_trend(self, values: List[float]) -> str:
        if len(values) < 3:
            return "stable"
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        diff = avg_second - avg_first
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

    @staticmethod
    def _std_deviation(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)


class DriftDetector:
    """Detects drift between declared intent and observed behavior over time."""

    def __init__(
        self,
        drift_threshold_minor: float = 0.3,
        drift_threshold_moderate: float = 0.5,
        drift_threshold_severe: float = 0.7,
        drift_threshold_critical: float = 0.85,
        window_size: int = 50,
    ) -> None:
        self.thresholds: Dict[DriftSeverity, float] = {
            DriftSeverity.MINOR: drift_threshold_minor,
            DriftSeverity.MODERATE: drift_threshold_moderate,
            DriftSeverity.SEVERE: drift_threshold_severe,
            DriftSeverity.CRITICAL: drift_threshold_critical,
        }
        self.window_size: int = window_size
        self._consistency_history: Dict[str, List[float]] = {}
        self._baseline: Dict[str, float] = {}
        self._drift_reports: List[DriftReport] = []

    def update_baseline(self, intent_id: str, baseline_score: float) -> None:
        self._baseline[intent_id] = baseline_score

    def record_consistency(self, intent_id: str, score: float) -> Optional[DriftReport]:
        if intent_id not in self._consistency_history:
            self._consistency_history[intent_id] = []
        self._consistency_history[intent_id].append(score)
        if len(self._consistency_history[intent_id]) > self.window_size * 2:
            self._consistency_history[intent_id] = self._consistency_history[intent_id][-self.window_size * 2:]
        return self._check_drift(intent_id)

    def _check_drift(self, intent_id: str) -> Optional[DriftReport]:
        history = self._consistency_history.get(intent_id, [])
        if len(history) < 5:
            return None
        window = history[-self.window_size:]
        current_avg = sum(window) / len(window)
        baseline = self._baseline.get(intent_id, window[0] if window else 1.0)
        drift_score = abs(current_avg - baseline)
        if drift_score < self.thresholds[DriftSeverity.MINOR]:
            return None
        severity = DriftSeverity.NONE
        for sev in [DriftSeverity.CRITICAL, DriftSeverity.SEVERE, DriftSeverity.MODERATE, DriftSeverity.MINOR]:
            if drift_score >= self.thresholds[sev]:
                severity = sev
                break
        trend = self._detect_trend_direction(history)
        factors = self._identify_contributing_factors(intent_id, history)
        recommendations = self._generate_recommendations(severity, trend, factors)
        report = DriftReport(
            drift_id=uuid.uuid4().hex[:12],
            intent_id=intent_id,
            severity=severity,
            drift_score=drift_score,
            original_intent=f"baseline={baseline:.3f}",
            observed_behavior_summary=f"current_avg={current_avg:.3f}",
            contributing_factors=factors,
            recommendations=recommendations,
        )
        self._drift_reports.append(report)
        return report

    def _detect_trend_direction(self, history: List[float]) -> str:
        if len(history) < 4:
            return "insufficient_data"
        n = len(history)
        half = n // 2
        first_avg = sum(history[:half]) / half
        second_avg = sum(history[half:]) / (n - half)
        if second_avg < first_avg - 0.05:
            return "degrading"
        elif second_avg > first_avg + 0.05:
            return "improving"
        return "stable"

    def _identify_contributing_factors(
        self, intent_id: str, history: List[float]
    ) -> List[str]:
        factors: List[str] = []
        if len(history) < 5:
            return factors
        recent = history[-10:]
        if min(recent) < 0.2:
            factors.append("recent_severe_violation")
        if max(recent) - min(recent) > 0.5:
            factors.append("high_volatility")
        avg_recent = sum(recent) / len(recent)
        avg_earlier = sum(history[:-10]) / len(history[:-10]) if len(history) > 10 else avg_recent
        if avg_recent < avg_earlier - 0.15:
            factors.append("sustained_decline")
        if len(history) > 20:
            last_5 = history[-5:]
            if all(s < 0.3 for s in last_5):
                factors.append("consecutive_low_scores")
        return factors

    def _generate_recommendations(
        self, severity: DriftSeverity, trend: str, factors: List[str]
    ) -> List[str]:
        recommendations: List[str] = []
        if severity in (DriftSeverity.SEVERE, DriftSeverity.CRITICAL):
            recommendations.append("Immediate intent re-verification recommended")
            recommendations.append("Consider pausing agent execution")
        if "high_volatility" in factors:
            recommendations.append("Investigate inconsistent behavior patterns")
        if "sustained_decline" in factors:
            recommendations.append("Review recent context changes or prompt modifications")
        if trend == "degrading":
            recommendations.append("Monitor for potential prompt injection or goal hijacking")
        if severity == DriftSeverity.MINOR:
            recommendations.append("Continue monitoring, no immediate action required")
        return recommendations

    def get_drift_reports(
        self, intent_id: Optional[str] = None, limit: int = 50
    ) -> List[DriftReport]:
        reports = self._drift_reports
        if intent_id:
            reports = [r for r in reports if r.intent_id == intent_id]
        return reports[-limit:]

    def reset(self, intent_id: Optional[str] = None) -> None:
        if intent_id:
            self._consistency_history.pop(intent_id, None)
            self._baseline.pop(intent_id, None)
        else:
            self._consistency_history.clear()
            self._baseline.clear()


class IntentDecomposer:
    """Decomposes complex intents into simpler sub-intents."""

    def __init__(
        self,
        embedding_engine: Optional[IntentEmbedding] = None,
        similarity_engine: Optional[SemanticSimilarity] = None,
    ) -> None:
        self.embedding: IntentEmbedding = embedding_engine or IntentEmbedding()
        self.similarity: SemanticSimilarity = similarity_engine or SemanticSimilarity()
        self._action_verbs: Set[str] = {
            "create", "delete", "update", "read", "send", "receive", "analyze",
            "generate", "modify", "remove", "add", "list", "search", "compute",
            "validate", "transform", "export", "import", "configure", "deploy",
            "test", "monitor", "schedule", "encrypt", "decrypt", "authenticate",
            "authorize", "backup", "restore", "migrate", "convert", "filter",
            "sort", "aggregate", "join", "split", "merge", "compare", "evaluate",
        }
        self._conjunction_patterns: List[str] = [
            r'\band\b', r'\bthen\b', r'\bafter\b', r'\bfollowed by\b',
            r'\bbefore\b', r'\bwhile\b', r'\balso\b', r'\bfurthermore\b',
            r'\bin addition\b', r'\bnext\b', r'\bfinally\b',
        ]
        self._decomposition_cache: Dict[str, DecomposedIntent] = {}

    def decompose(self, intent: IntentProfile) -> DecomposedIntent:
        cache_key = hashlib.sha256(intent.description.encode()).hexdigest()[:16]
        if cache_key in self._decomposition_cache:
            return self._decomposition_cache[cache_key]
        sub_descriptions = self._split_intent(intent.description)
        if len(sub_descriptions) <= 1:
            sub_descriptions = self._verb_based_decomposition(intent.description)
        sub_intents: List[IntentProfile] = []
        for i, desc in enumerate(sub_descriptions):
            sub_intent = IntentProfile(
                intent_id=f"{intent.intent_id}_sub_{i}",
                description=desc.strip(),
                category=self.embedding.infer_category(desc),
                embedding=self.embedding.embed(desc),
                constraints=list(intent.constraints),
                confidence=intent.confidence * 0.9,
            )
            sub_intents.append(sub_intent)
        completeness = self._compute_completeness(intent, sub_intents)
        coherence = self._compute_coherence(sub_intents)
        result = DecomposedIntent(
            parent_intent_id=intent.intent_id,
            sub_intents=sub_intents,
            decomposition_strategy="conjunction_and_verb",
            completeness_score=completeness,
            coherence_score=coherence,
        )
        self._decomposition_cache[cache_key] = result
        return result

    def _split_intent(self, description: str) -> List[str]:
        parts = [description]
        for pattern in self._conjunction_patterns:
            new_parts: List[str] = []
            for part in parts:
                segments = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend(s.strip() for s in segments if s.strip())
            parts = new_parts
            if len(parts) > 1:
                break
        return [p for p in parts if len(p.strip()) > 3]

    def _verb_based_decomposition(self, description: str) -> List[str]:
        sentences = re.split(r'[.!?]+', description)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            return sentences
        tokens = description.split()
        verb_indices: List[int] = []
        for i, token in enumerate(tokens):
            clean = re.sub(r'[^a-z]', '', token.lower())
            if clean in self._action_verbs:
                verb_indices.append(i)
        if len(verb_indices) <= 1:
            return [description]
        segments: List[str] = []
        for j, idx in enumerate(verb_indices):
            end = verb_indices[j + 1] if j + 1 < len(verb_indices) else len(tokens)
            segment = " ".join(tokens[idx:end])
            if segment.strip():
                segments.append(segment.strip())
        return segments if segments else [description]

    def _compute_completeness(
        self, original: IntentProfile, sub_intents: List[IntentProfile]
    ) -> float:
        if not sub_intents:
            return 0.0
        original_tokens = set(self.similarity.tokenize(original.description))
        if not original_tokens:
            return 1.0
        covered_tokens: Set[str] = set()
        for sub in sub_intents:
            sub_tokens = set(self.similarity.tokenize(sub.description))
            covered_tokens.update(sub_tokens)
        coverage = len(covered_tokens & original_tokens) / len(original_tokens)
        constraint_coverage = 0.0
        if original.constraints:
            covered_constraints = 0
            for constraint in original.constraints:
                for sub in sub_intents:
                    if self.similarity.combined_similarity(constraint, sub.description) > 0.3:
                        covered_constraints += 1
                        break
            constraint_coverage = covered_constraints / len(original.constraints)
        return (coverage * 0.7 + constraint_coverage * 0.3)

    def _compute_coherence(self, sub_intents: List[IntentProfile]) -> float:
        if len(sub_intents) <= 1:
            return 1.0
        similarities: List[float] = []
        for i in range(len(sub_intents) - 1):
            sim = self.similarity.combined_similarity(
                sub_intents[i].description, sub_intents[i + 1].description
            )
            similarities.append(sim)
        if not similarities:
            return 1.0
        return sum(similarities) / len(similarities)


class IntentVerifier:
    """Main class for semantic intent verification."""

    def __init__(
        self,
        drift_threshold_minor: float = 0.3,
        drift_threshold_moderate: float = 0.5,
        drift_threshold_severe: float = 0.7,
    ) -> None:
        self.embedding: IntentEmbedding = IntentEmbedding()
        self.similarity: SemanticSimilarity = SemanticSimilarity()
        self.consistency_scorer: ConsistencyScorer = ConsistencyScorer(
            similarity_engine=self.similarity,
            embedding_engine=self.embedding,
        )
        self.drift_detector: DriftDetector = DriftDetector(
            drift_threshold_minor=drift_threshold_minor,
            drift_threshold_moderate=drift_threshold_moderate,
            drift_threshold_severe=drift_threshold_severe,
        )
        self.decomposer: IntentDecomposer = IntentDecomposer(
            embedding_engine=self.embedding,
            similarity_engine=self.similarity,
        )
        self._registered_intents: Dict[str, IntentProfile] = {}
        self._verification_history: List[Dict[str, Any]] = []

    def register_intent(
        self,
        description: str,
        category: Optional[IntentCategory] = None,
        constraints: Optional[List[str]] = None,
        expected_behaviors: Optional[List[str]] = None,
        forbidden_behaviors: Optional[List[str]] = None,
    ) -> IntentProfile:
        intent_id = uuid.uuid4().hex[:12]
        if category is None:
            category = self.embedding.infer_category(description)
        embedding = self.embedding.embed(description)
        profile = IntentProfile(
            intent_id=intent_id,
            description=description,
            category=category,
            embedding=embedding,
            constraints=constraints or [],
            expected_behaviors=expected_behaviors or [],
            forbidden_behaviors=forbidden_behaviors or [],
        )
        self._registered_intents[intent_id] = profile
        self.drift_detector.update_baseline(intent_id, 1.0)
        self._verification_history.append({
            "action": "register",
            "intent_id": intent_id,
            "timestamp": time.time(),
        })
        return profile

    def verify_behavior(
        self,
        intent_id: str,
        observation: BehaviorObservation,
    ) -> Dict[str, Any]:
        intent = self._registered_intents.get(intent_id)
        if intent is None:
            return {
                "status": "error",
                "message": f"Intent {intent_id} not found",
                "consistency": 0.0,
            }
        scores = self.consistency_scorer.score_single(intent, observation)
        self.consistency_scorer.add_observation(intent_id, observation)
        drift_report = self.drift_detector.record_consistency(
            intent_id, scores["overall_consistency"]
        )
        result: Dict[str, Any] = {
            "status": "ok",
            "intent_id": intent_id,
            "observation_id": observation.observation_id,
            "consistency_scores": scores,
            "drift_detected": drift_report is not None,
        }
        if drift_report:
            result["drift_report"] = drift_report.to_dict()
        self._verification_history.append({
            "action": "verify",
            "intent_id": intent_id,
            "consistency": scores["overall_consistency"],
            "timestamp": time.time(),
        })
        return result

    def decompose_intent(self, intent_id: str) -> Optional[DecomposedIntent]:
        intent = self._registered_intents.get(intent_id)
        if intent is None:
            return None
        return self.decomposer.decompose(intent)

    def compare_intents(self, intent_id1: str, intent_id2: str) -> Dict[str, float]:
        intent1 = self._registered_intents.get(intent_id1)
        intent2 = self._registered_intents.get(intent_id2)
        if intent1 is None or intent2 is None:
            return {"error": "One or both intents not found"}
        text_sim = self.similarity.combined_similarity(
            intent1.description, intent2.description
        )
        embedding_sim = intent1.embedding.cosine_similarity(intent2.embedding)
        return {
            "text_similarity": text_sim,
            "embedding_similarity": embedding_sim,
            "combined": (text_sim + embedding_sim) / 2,
        }

    def get_intent_profile(self, intent_id: str) -> Optional[IntentProfile]:
        return self._registered_intents.get(intent_id)

    def list_intents(self) -> List[IntentProfile]:
        return list(self._registered_intents.values())

    def get_drift_reports(self, intent_id: Optional[str] = None) -> List[DriftReport]:
        return self.drift_detector.get_drift_reports(intent_id)

    def get_verification_summary(self) -> Dict[str, Any]:
        if not self._verification_history:
            return {"total_verifications": 0}
        verifications = [
            h for h in self._verification_history if h["action"] == "verify"
        ]
        scores = [h["consistency"] for h in verifications if "consistency" in h]
        return {
            "total_intents": len(self._registered_intents),
            "total_verifications": len(verifications),
            "average_consistency": sum(scores) / len(scores) if scores else 0.0,
            "min_consistency": min(scores) if scores else 0.0,
            "max_consistency": max(scores) if scores else 0.0,
        }
