"""
Threat Detection Module
=======================

Detects security threats in prompts and requests including injection
attacks, jailbreak attempts, data exfiltration, and statistical anomalies.

Only uses the Python standard library.
"""

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Threat Level & Report
# ---------------------------------------------------------------------------

class ThreatLevel(Enum):
    """Severity of a detected threat."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatReport:
    """Detailed report of a detected threat."""

    level: ThreatLevel
    """Threat severity level."""

    category: str
    """Category of the threat (e.g. 'injection', 'jailbreak', 'exfiltration')."""

    description: str
    """Human-readable description of the detected threat."""

    confidence: float
    """Confidence score in [0, 1]."""

    recommendations: List[str] = field(default_factory=list)
    """Recommended mitigations."""

    matched_patterns: List[str] = field(default_factory=list)
    """Patterns that triggered the detection."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "category": self.category,
            "description": self.description,
            "confidence": round(self.confidence, 4),
            "recommendations": self.recommendations,
            "matched_patterns": self.matched_patterns,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Injection Detector
# ---------------------------------------------------------------------------

class InjectionDetector:
    """Detects prompt injection attacks using pattern matching and heuristics.

    Covers:
    - Direct instruction injection
    - Role-play injection
    - Encoding-based bypass attempts
    - Multi-part / segmented injection
    """

    # Patterns organised by category
    DIRECT_INJECTION_PATTERNS: List[Tuple[str, float]] = [
        (r"(?i)ignore\s+(all\s+)?previous\s+(instructions?|prompts?)", 0.95),
        (r"(?i)forget\s+(everything|all|your)\s+(instructions?|rules?|training)", 0.90),
        (r"(?i)disregard\s+(your|all|previous)\s+(instructions?|rules?|guidelines?)", 0.90),
        (r"(?i)you\s+are\s+now\s+(a|an|the)\s+", 0.70),
        (r"(?i)new\s+instructions?\s*:", 0.85),
        (r"(?i)system\s*:\s*", 0.80),
        (r"(?i)override\s+(your|the|system)\s+(instructions?|rules?|settings?)", 0.90),
        (r"(?i)pretend\s+(you\s+are|to\s+be|that\s+you)", 0.75),
        (r"(?i)act\s+as\s+(if\s+you\s+(are|were)|a|an|the)", 0.70),
        (r"(?i)from\s+now\s+on", 0.50),
        (r"(?i)do\s+not\s+(follow|use|obey)\s+(your|the|any)\s+(instructions?|rules?|guidelines?)", 0.90),
        (r"(?i)stop\s+(being|acting)\s+(a|an|the|like)\s+", 0.65),
        (r"(?i)no\s+longer\s+(follow|obey|use)\s+", 0.80),
        (r"(?i)switch\s+(to|into)\s+", 0.55),
        (r"(?i)\badmin\b.*\bmode\b", 0.85),
        (r"(?i)\bdeveloper\b.*\bmode\b", 0.85),
        (r"(?i)debug\s+mode", 0.75),
        (r"(?i)\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", 0.80),
        (r"(?i)<\|im_start\|>|<\|im_end\|>", 0.80),
        (r"(?i)human\s*:\s*|assistant\s*:\s*", 0.60),
    ]

    ENCODING_BYPASS_PATTERNS: List[Tuple[str, float]] = [
        (r"\\u[0-9a-fA-F]{4}", 0.60),
        (r"&#\d+;", 0.55),
        (r"&#x[0-9a-fA-F]+;", 0.55),
        (r"base64\s*:", 0.50),
        (r"rot13\s*:", 0.45),
        (r"\\x[0-9a-fA-F]{2}", 0.55),
        (r"(?i)unicode\s+(escape|encode)", 0.50),
        (r"(?i)url\s*encode", 0.45),
        (r"(?i)html\s*entit", 0.50),
        (r"\\n.*\\n.*\\n", 0.40),
    ]

    SEGMENTED_INJECTION_PATTERNS: List[Tuple[str, float]] = [
        (r"(?i)part\s+\d+\s+of\s+\d+", 0.55),
        (r"(?i)step\s+\d+\s*:", 0.45),
        (r"(?i)first,?\s*(then|next|after)\s+that", 0.40),
        (r"(?i)continue\s+(from|with|the)", 0.45),
        (r"\.{3,}.*\.{3,}", 0.35),
        (r"(?i)wait\s+for\s+(my|the|next)\s+(instruction|prompt|message)", 0.60),
    ]

    def __init__(self):
        self._compiled_direct = [
            (re.compile(p, re.IGNORECASE), c)
            for p, c in self.DIRECT_INJECTION_PATTERNS
        ]
        self._compiled_encoding = [
            (re.compile(p, re.IGNORECASE), c)
            for p, c in self.ENCODING_BYPASS_PATTERNS
        ]
        self._compiled_segmented = [
            (re.compile(p, re.IGNORECASE), c)
            for p, c in self.SEGMENTED_INJECTION_PATTERNS
        ]

    def pattern_based_detection(self, text: str) -> List[ThreatReport]:
        """Run all pattern-based detectors on *text*."""
        reports: List[ThreatReport] = []

        # Direct injection
        for pattern, confidence in self._compiled_direct:
            matches = pattern.findall(text)
            if matches:
                level = self._confidence_to_level(confidence)
                reports.append(ThreatReport(
                    level=level,
                    category="injection",
                    description=f"Direct instruction injection detected: {pattern.pattern[:60]}",
                    confidence=confidence,
                    matched_patterns=[pattern.pattern],
                    recommendations=[
                        "Reject the prompt or sanitize the injection payload.",
                        "Apply output filtering to prevent instruction leakage.",
                    ],
                ))

        # Encoding bypass
        for pattern, confidence in self._compiled_encoding:
            matches = pattern.findall(text)
            if matches:
                level = self._confidence_to_level(confidence)
                reports.append(ThreatReport(
                    level=level,
                    category="encoding_bypass",
                    description=f"Encoding-based bypass attempt detected: {pattern.pattern[:60]}",
                    confidence=confidence,
                    matched_patterns=[pattern.pattern],
                    recommendations=[
                        "Decode the text and re-analyze for injection patterns.",
                        "Block encoded payloads that decode to injection instructions.",
                    ],
                ))

        # Segmented injection
        for pattern, confidence in self._compiled_segmented:
            matches = pattern.findall(text)
            if matches:
                level = self._confidence_to_level(confidence)
                reports.append(ThreatReport(
                    level=level,
                    category="segmented_injection",
                    description=f"Segmented injection pattern detected: {pattern.pattern[:60]}",
                    confidence=confidence,
                    matched_patterns=[pattern.pattern],
                    recommendations=[
                        "Track conversation context across multiple turns.",
                        "Flag segmented prompts for manual review.",
                    ],
                ))

        return reports

    def heuristic_detection(self, text: str) -> List[ThreatReport]:
        """Run heuristic-based detection on *text*."""
        reports: List[ThreatReport] = []

        # Heuristic 1: High ratio of imperative verbs
        imperative_verbs = re.findall(
            r"\b(ignore|forget|disregard|override|pretend|act|stop|switch|bypass|skip|disable|deactivate|hide|conceal|reveal|disclose|expose|dump|print|show|display|output|return|provide|give|tell|say|speak|write|read|access|fetch|retrieve|download|upload|send|transmit|leak|extract|export)\b",
            text,
            re.IGNORECASE,
        )
        verb_ratio = len(imperative_verbs) / max(len(text.split()), 1)
        if verb_ratio > 0.15:
            reports.append(ThreatReport(
                level=ThreatLevel.MEDIUM,
                category="heuristic_injection",
                description=f"High imperative verb ratio: {verb_ratio:.2%}",
                confidence=min(verb_ratio * 3, 1.0),
                recommendations=[
                    "Review the prompt for command-like language.",
                ],
            ))

        # Heuristic 2: Unusual punctuation density
        punct_chars = re.findall(r"[!@#$%^&*()_+=\[\]{}|\\:;\"'<>,.?/~`]", text)
        punct_ratio = len(punct_chars) / max(len(text), 1)
        if punct_ratio > 0.1:
            reports.append(ThreatReport(
                level=ThreatLevel.LOW,
                category="obfuscation",
                description=f"High punctuation density: {punct_ratio:.2%}",
                confidence=min(punct_ratio * 5, 1.0),
                recommendations=[
                    "Check for obfuscation attempts via special characters.",
                ],
            ))

        # Heuristic 3: Excessive whitespace or formatting
        whitespace_ratio = text.count(" ") / max(len(text), 1)
        if whitespace_ratio > 0.4:
            reports.append(ThreatReport(
                level=ThreatLevel.LOW,
                category="formatting_anomaly",
                description=f"High whitespace ratio: {whitespace_ratio:.2%}",
                confidence=min(whitespace_ratio * 2, 1.0),
                recommendations=[
                    "Normalize whitespace before processing.",
                ],
            ))

        # Heuristic 4: Length-based detection (very long prompts may be attempts to overwhelm)
        if len(text) > 5000:
            reports.append(ThreatReport(
                level=ThreatLevel.LOW,
                category="long_prompt",
                description=f"Unusually long prompt: {len(text)} characters",
                confidence=0.3,
                recommendations=[
                    "Consider truncating or chunking very long prompts.",
                ],
            ))

        return reports

    @staticmethod
    def _confidence_to_level(confidence: float) -> ThreatLevel:
        if confidence >= 0.85:
            return ThreatLevel.HIGH
        elif confidence >= 0.70:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.50:
            return ThreatLevel.LOW
        return ThreatLevel.NONE


# ---------------------------------------------------------------------------
# Jailbreak Detector
# ---------------------------------------------------------------------------

class JailbreakDetector:
    """Detects jailbreak attempts including role-play, encoding bypass,
    and multi-turn conversation attacks."""

    ROLE_PLAY_PATTERNS: List[Tuple[str, float]] = [
        (r"(?i)you\s+are\s+(now|currently)\s+(a|an|the)\s+(evil|malicious|unrestricted|unfiltered|uncensored)", 0.90),
        (r"(?i)imagine\s+you\s+(are|were)\s+(a|an)\s+(hacker|criminal|attacker|villain)", 0.88),
        (r"(?i)roleplay\s+as\s+(a|an)\s+", 0.75),
        (r"(?i)in\s+this\s+(scenario|simulation|story|fiction|hypothetical)", 0.65),
        (r"(?i)for\s+(educational|research|academic|testing)\s+purposes?\s*(only)?", 0.55),
        (r"(?i)as\s+an\s+(AI|assistant)\s+without\s+(restrictions?|limitations?|filters?|safety)", 0.85),
        (r"(?i)jailbreak(ed)?\s*", 0.80),
        (r"(?i)DAN\s+(mode|prompt|10|20)", 0.85),
        (r"(?i)anti\s*[- ]?(censorship|filter|safety|guard)", 0.80),
        (r"(?i)unrestricted\s+mode", 0.85),
        (r"(?i)above\s+(the|your)\s+(law|rules|guidelines|restrictions)", 0.75),
        (r"(?i)nobody\s+is\s+(watching|listening|monitoring)", 0.70),
        (r"(?i)hypothetically\s+speaking", 0.50),
        (r"(?i)let'?s\s+pretend", 0.60),
        (r"(?i)write\s+a\s+(story|fiction|novel)\s+about", 0.40),
    ]

    ENCODING_BYPASS_PATTERNS: List[Tuple[str, float]] = [
        (r"(?i)using\s+(base64|rot13|binary|hex|unicode|url)\s+(encoding|decoding)", 0.70),
        (r"(?i)encode\s+(this|the|it)\s+(in|as|with|using)", 0.65),
        (r"(?i)represent\s+(as|in|using)\s+(binary|hex|base64|morse)", 0.60),
        (r"(?i)spell\s+(it|this|the\s+word)\s+(out|backwards)", 0.50),
        (r"(?i)use\s+(leetspeak|1337|pig\s+latin)", 0.55),
        (r"(?i)first\s+letters?\s+(of\s+each\s+word|only|spell\s+out)", 0.55),
        (r"(?i)replace\s+(each|every)\s+(letter|char)", 0.50),
        (r"(?i)translate\s+(this|to|into)\s+(latin|french|german|spanish|chinese|japanese)", 0.35),
    ]

    MULTI_TURN_PATTERNS: List[Tuple[str, float]] = [
        (r"(?i)remember\s+(what|the|that)\s+(i|we)\s+(said|told|discussed|asked)", 0.55),
        (r"(?i)going\s+back\s+to", 0.45),
        (r"(?i)as\s+(i\s+mentioned|we\s+discussed|noted)\s+(earlier|before|previously)", 0.40),
        (r"(?i)let'?s\s+change\s+(the|our)\s+(topic|subject|direction)", 0.50),
        (r"(?i)that'?s\s+not\s+what\s+i\s+meant", 0.40),
        (r"(?i)try\s+(again|once\s+more|differently|harder)", 0.45),
        (r"(?i)you\s+(didn'?t|failed\s+to|forgot\s+to)\s+(answer|respond|address)", 0.55),
        (r"(?i)be\s+more\s+(specific|detailed|explicit|direct|helpful)", 0.40),
        (r"(?i)don'?t\s+(hold\s+back|be\s+(shy|cautious|conservative|restrictive))", 0.60),
    ]

    def __init__(self):
        self._compiled_role = [
            (re.compile(p, re.IGNORECASE), c)
            for p, c in self.ROLE_PLAY_PATTERNS
        ]
        self._compiled_encoding = [
            (re.compile(p, re.IGNORECASE), c)
            for p, c in self.ENCODING_BYPASS_PATTERNS
        ]
        self._compiled_multi = [
            (re.compile(p, re.IGNORECASE), c)
            for p, c in self.MULTI_TURN_PATTERNS
        ]
        self._conversation_history: deque = deque(maxlen=20)

    def detect_role_play(self, text: str) -> List[ThreatReport]:
        """Detect role-play based jailbreak attempts."""
        reports: List[ThreatReport] = []
        for pattern, confidence in self._compiled_role:
            if pattern.search(text):
                level = InjectionDetector._confidence_to_level(confidence)
                reports.append(ThreatReport(
                    level=level,
                    category="jailbreak_roleplay",
                    description=f"Role-play jailbreak attempt: {pattern.pattern[:60]}",
                    confidence=confidence,
                    matched_patterns=[pattern.pattern],
                    recommendations=[
                        "Reject role-play requests that bypass safety constraints.",
                        "Maintain consistent persona and refuse unsafe role assignments.",
                    ],
                ))
        return reports

    def detect_encoding_bypass(self, text: str) -> List[ThreatReport]:
        """Detect encoding-based jailbreak bypass attempts."""
        reports: List[ThreatReport] = []
        for pattern, confidence in self._compiled_encoding:
            if pattern.search(text):
                level = InjectionDetector._confidence_to_level(confidence)
                reports.append(ThreatReport(
                    level=level,
                    category="jailbreak_encoding",
                    description=f"Encoding bypass attempt: {pattern.pattern[:60]}",
                    confidence=confidence,
                    matched_patterns=[pattern.pattern],
                    recommendations=[
                        "Decode encoded text before processing.",
                        "Block requests that ask for encoding-based obfuscation.",
                    ],
                ))
        return reports

    def detect_multi_turn(self, text: str) -> List[ThreatReport]:
        """Detect multi-turn conversation jailbreak attempts.

        Analyses the current prompt in the context of recent conversation
        history to identify gradual escalation patterns.
        """
        reports: List[ThreatReport] = []

        # Pattern-based multi-turn detection
        for pattern, confidence in self._compiled_multi:
            if pattern.search(text):
                level = InjectionDetector._confidence_to_level(confidence)
                reports.append(ThreatReport(
                    level=level,
                    category="jailbreak_multi_turn",
                    description=f"Multi-turn jailbreak pattern: {pattern.pattern[:60]}",
                    confidence=confidence,
                    matched_patterns=[pattern.pattern],
                    recommendations=[
                        "Monitor conversation context for escalating requests.",
                        "Reset safety context when suspicious multi-turn patterns are detected.",
                    ],
                ))

        # Contextual analysis: check for escalation in threat levels
        if len(self._conversation_history) >= 3:
            recent_threats = []
            for past_text in list(self._conversation_history)[-5:]:
                past_reports = self.detect_role_play(past_text)
                past_reports += self.detect_encoding_bypass(past_text)
                if past_reports:
                    recent_threats.append(max(r.confidence for r in past_reports))

            if len(recent_threats) >= 3:
                avg_threat = sum(recent_threats) / len(recent_threats)
                if avg_threat > 0.3:
                    reports.append(ThreatReport(
                        level=ThreatLevel.MEDIUM,
                        category="jailbreak_escalation",
                        description=f"Escalating threat pattern detected across {len(recent_threats)} recent turns (avg confidence: {avg_threat:.2f})",
                        confidence=min(avg_threat * 1.5, 1.0),
                        recommendations=[
                            "Apply stricter safety checks for this conversation.",
                            "Consider terminating the conversation if escalation continues.",
                        ],
                    ))

        self._conversation_history.append(text)
        return reports

    def update_history(self, text: str) -> None:
        """Manually add a text to the conversation history."""
        self._conversation_history.append(text)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Statistical anomaly detector for request analysis.

    Maintains a sliding window of historical request metrics (length,
    character diversity, word count, etc.) and flags requests that deviate
    significantly from the baseline using Z-score analysis.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._lengths: deque = deque(maxlen=window_size)
        self._word_counts: deque = deque(maxlen=window_size)
        self._char_diversities: deque = deque(maxlen=window_size)
        self._special_char_ratios: deque = deque(maxlen=window_size)
        self._uppercase_ratios: deque = deque(maxlen=window_size)
        self._request_times: deque = deque(maxlen=window_size)

    @staticmethod
    def _char_diversity(text: str) -> float:
        """Compute the ratio of unique characters to total characters."""
        if not text:
            return 0.0
        return len(set(text)) / len(text)

    @staticmethod
    def _special_char_ratio(text: str) -> float:
        """Compute the ratio of non-alphanumeric characters."""
        if not text:
            return 0.0
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special / len(text)

    @staticmethod
    def _uppercase_ratio(text: str) -> float:
        """Compute the ratio of uppercase alphabetic characters."""
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return 0.0
        return sum(1 for c in alpha if c.isupper()) / len(alpha)

    @staticmethod
    def _z_score(value: float, mean: float, std: float) -> float:
        """Compute the Z-score of *value* given *mean* and *std*."""
        if std < 1e-10:
            return 0.0
        return abs(value - mean) / std

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = AnomalyDetector._mean(values)
        variance = sum((v - m) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def _update_baseline(self, text: str) -> None:
        """Update the baseline statistics with a new request."""
        self._lengths.append(len(text))
        self._word_counts.append(len(text.split()))
        self._char_diversities.append(self._char_diversity(text))
        self._special_char_ratios.append(self._special_char_ratio(text))
        self._uppercase_ratios.append(self._uppercase_ratio(text))
        self._request_times.append(time.time())

    def detect_anomaly(self, request: str) -> ThreatReport:
        """Analyse *request* for statistical anomalies.

        Compares the request against the historical baseline and returns
        a ThreatReport if any metric deviates significantly.
        """
        # Need at least 10 samples for a meaningful baseline
        if len(self._lengths) < 10:
            self._update_baseline(request)
            return ThreatReport(
                level=ThreatLevel.NONE,
                category="anomaly",
                description="Insufficient baseline data for anomaly detection.",
                confidence=0.0,
            )

        metrics = {
            "length": len(request),
            "word_count": len(request.split()),
            "char_diversity": self._char_diversity(request),
            "special_char_ratio": self._special_char_ratio(request),
            "uppercase_ratio": self._uppercase_ratio(request),
        }

        baselines = {
            "length": (list(self._lengths), "Request length"),
            "word_count": (list(self._word_counts), "Word count"),
            "char_diversity": (list(self._char_diversities), "Character diversity"),
            "special_char_ratio": (list(self._special_char_ratios), "Special character ratio"),
            "uppercase_ratio": (list(self._uppercase_ratios), "Uppercase ratio"),
        }

        z_threshold = 3.0
        anomalies: List[str] = []
        max_z = 0.0
        anomaly_details: Dict[str, float] = {}

        for metric_name, value in metrics.items():
            history, label = baselines[metric_name]
            m = self._mean(history)
            s = self._std(history)
            z = self._z_score(value, m, s)
            anomaly_details[metric_name] = z
            if z > z_threshold:
                anomalies.append(f"{label}: Z={z:.2f} (value={value:.4f}, mean={m:.4f}, std={s:.4f})")
                max_z = max(max_z, z)

        self._update_baseline(request)

        if not anomalies:
            return ThreatReport(
                level=ThreatLevel.NONE,
                category="anomaly",
                description="No anomalies detected.",
                confidence=0.0,
            )

        confidence = min(max_z / 5.0, 1.0)
        level = (
            ThreatLevel.CRITICAL if max_z > 5.0
            else ThreatLevel.HIGH if max_z > 4.0
            else ThreatLevel.MEDIUM if max_z > 3.0
            else ThreatLevel.LOW
        )

        return ThreatReport(
            level=level,
            category="anomaly",
            description=f"Statistical anomaly detected in {len(anomalies)} metric(s).",
            confidence=confidence,
            matched_patterns=anomalies,
            metadata=anomaly_details,
            recommendations=[
                "Review the anomalous request for potential security threats.",
                "Consider rate-limiting the source if anomalies persist.",
                "Update baseline if the anomaly represents a legitimate pattern change.",
            ],
        )

    def detect_rate_anomaly(self) -> ThreatReport:
        """Detect if requests are arriving at an unusual rate."""
        if len(self._request_times) < 10:
            return ThreatReport(
                level=ThreatLevel.NONE,
                category="rate_anomaly",
                description="Insufficient data for rate analysis.",
                confidence=0.0,
            )

        times = sorted(self._request_times)
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        if not intervals:
            return ThreatReport(
                level=ThreatLevel.NONE,
                category="rate_anomaly",
                description="No intervals to analyze.",
                confidence=0.0,
            )

        mean_interval = self._mean(intervals)
        std_interval = self._std(intervals)

        # Check if the most recent interval is anomalous
        recent_interval = intervals[-1] if intervals else 0
        z = self._z_score(recent_interval, mean_interval, std_interval)

        # Very short intervals (high rate) are suspicious
        if z > 3.0 and recent_interval < mean_interval:
            confidence = min(z / 5.0, 1.0)
            return ThreatReport(
                level=ThreatLevel.MEDIUM if confidence < 0.8 else ThreatLevel.HIGH,
                category="rate_anomaly",
                description=f"Unusually high request rate detected (interval={recent_interval:.3f}s, mean={mean_interval:.3f}s, Z={z:.2f}).",
                confidence=confidence,
                recommendations=[
                    "Apply rate limiting.",
                    "Monitor for automated attack patterns.",
                ],
            )

        return ThreatReport(
            level=ThreatLevel.NONE,
            category="rate_anomaly",
            description="Request rate is within normal bounds.",
            confidence=0.0,
        )

    def get_baseline_stats(self) -> Dict[str, Dict[str, float]]:
        """Return current baseline statistics."""
        return {
            "length": {"mean": self._mean(list(self._lengths)), "std": self._std(list(self._lengths))},
            "word_count": {"mean": self._mean(list(self._word_counts)), "std": self._std(list(self._word_counts))},
            "char_diversity": {"mean": self._mean(list(self._char_diversities)), "std": self._std(list(self._char_diversities))},
            "special_char_ratio": {"mean": self._mean(list(self._special_char_ratios)), "std": self._std(list(self._special_char_ratios))},
            "uppercase_ratio": {"mean": self._mean(list(self._uppercase_ratios)), "std": self._std(list(self._uppercase_ratios))},
        }


# ---------------------------------------------------------------------------
# Threat Detector (Main Interface)
# ---------------------------------------------------------------------------

class ThreatDetector:
    """Unified threat detection interface combining injection, jailbreak,
    exfiltration, and anomaly detection."""

    def __init__(self):
        self._injection_detector = InjectionDetector()
        self._jailbreak_detector = JailbreakDetector()
        self._anomaly_detector = AnomalyDetector()

    def analyze_prompt(self, prompt: str) -> ThreatReport:
        """Perform a comprehensive threat analysis on *prompt*.

        Returns the highest-severity threat report found, or a clean report
        if no threats are detected.
        """
        all_reports = (
            self.detect_injection(prompt)
            + self.detect_jailbreak(prompt)
            + self.detect_data_exfiltration(prompt)
        )
        anomaly_report = self._anomaly_detector.detect_anomaly(prompt)
        all_reports.append(anomaly_report)

        if not all_reports:
            return ThreatReport(
                level=ThreatLevel.NONE,
                category="none",
                description="No threats detected.",
                confidence=1.0,
            )

        # Return the highest-severity report
        all_reports.sort(key=lambda r: r.level.value, reverse=True)
        return all_reports[0]

    def detect_injection(self, prompt: str) -> List[ThreatReport]:
        """Detect prompt injection attacks."""
        reports = self._injection_detector.pattern_based_detection(prompt)
        reports += self._injection_detector.heuristic_detection(prompt)
        return reports

    def detect_jailbreak(self, prompt: str) -> List[ThreatReport]:
        """Detect jailbreak attempts."""
        reports = self._jailbreak_detector.detect_role_play(prompt)
        reports += self._jailbreak_detector.detect_encoding_bypass(prompt)
        reports += self._jailbreak_detector.detect_multi_turn(prompt)
        return reports

    def detect_data_exfiltration(self, prompt: str) -> List[ThreatReport]:
        """Detect potential data exfiltration attempts."""
        reports: List[ThreatReport] = []

        exfil_patterns: List[Tuple[str, float]] = [
            (r"(?i)(send|transmit|email|post|upload|ftp|scp|curl|wget)\s+(this|the|all|my)\s+(data|information|content|output|response|result)", 0.80),
            (r"(?i)exfiltrate\s+", 0.90),
            (r"(?i)(dump|export|extract|copy|leak|reveal|disclose)\s+(the|all|my|your|internal|system|hidden|secret)\s+(data|information|keys?|passwords?|tokens?|credentials?)", 0.85),
            (r"(?i)print\s+(all|the|your|internal|system|hidden|secret)\s+(variables?|constants?|config|settings?|env|environment)", 0.80),
            (r"(?i)(access|read|connect\s+to)\s+(the\s+)?(database|db|filesystem|file\s+system|network|internal\s+(api|service|network))", 0.70),
            (r"(?i)http[s]?://[^\s]+", 0.30),
            (r"(?i)\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", 0.40),
            (r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|bearer)\s*[=:]\s*\S+", 0.75),
            (r"(?i)base64\s*(encode|decode)\s+(the|this|my|your)\s+(response|output|data|result)", 0.65),
        ]

        for pattern_str, confidence in exfil_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(prompt):
                level = InjectionDetector._confidence_to_level(confidence)
                reports.append(ThreatReport(
                    level=level,
                    category="exfiltration",
                    description=f"Potential data exfiltration attempt: {pattern_str[:60]}",
                    confidence=confidence,
                    matched_patterns=[pattern_str],
                    recommendations=[
                        "Block requests that attempt to extract internal data.",
                        "Sanitize output to prevent sensitive data leakage.",
                        "Monitor for external network connections from responses.",
                    ],
                ))

        return reports

    def detect_anomaly(self, request: str) -> ThreatReport:
        """Detect statistical anomalies in the request."""
        return self._anomaly_detector.detect_anomaly(request)

    def update_conversation_history(self, text: str) -> None:
        """Update the jailbreak detector's conversation history."""
        self._jailbreak_detector.update_history(text)

    def get_anomaly_stats(self) -> Dict[str, Dict[str, float]]:
        """Return the current anomaly detection baseline statistics."""
        return self._anomaly_detector.get_baseline_stats()
