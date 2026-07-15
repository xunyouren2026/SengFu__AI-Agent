"""
Dial Controller - Alignment Strictness Dial Management System

Provides fine-grained control over alignment strictness across multiple
categories (safety, privacy, fairness, honesty, kindness) with adaptive
auto-adjustment capabilities.
"""

import math
import random
import hashlib
import json
import time
import threading
import statistics
import functools
import copy
import re
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# 1. StrictnessLevel Enum
# ---------------------------------------------------------------------------

class StrictnessLevel(Enum):
    """Five-tier strictness hierarchy from permissive to paranoid."""
    PERMISSIVE = 1
    RELAXED = 2
    MODERATE = 3
    STRICT = 4
    PARANOID = 5


# ---------------------------------------------------------------------------
# 2. StrictnessProfile Dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrictnessProfile:
    """Immutable-style profile describing threshold configuration for one level."""
    level: StrictnessLevel
    description: str
    safety_threshold: float
    privacy_threshold: float
    fairness_threshold: float
    honesty_threshold: float
    kindness_threshold: float
    auto_correct: bool
    require_human_approval: bool
    max_retries: int
    timeout_seconds: float

    _CATEGORY_MAP: Dict[str, str] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        self._CATEGORY_MAP = {
            "safety": "safety_threshold",
            "privacy": "privacy_threshold",
            "fairness": "fairness_threshold",
            "honesty": "honesty_threshold",
            "kindness": "kindness_threshold",
        }

    def get_effective_threshold(self, category: str) -> float:
        """Return the threshold for *category*, clamped to [0.0, 1.0]."""
        attr = self._CATEGORY_MAP.get(category.lower())
        if attr is None:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid: {list(self._CATEGORY_MAP.keys())}"
            )
        value = getattr(self, attr)
        return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# 4. StrictnessChange Dataclass (declared early so DialController can use it)
# ---------------------------------------------------------------------------

@dataclass
class StrictnessChange:
    """Record of a single strictness-level transition."""
    timestamp: float
    from_level: StrictnessLevel
    to_level: StrictnessLevel
    reason: str
    changed_by: str  # "user", "auto", "system", etc.


# ---------------------------------------------------------------------------
# 3. DialController
# ---------------------------------------------------------------------------

class DialController:
    """Core strictness dial with five preset profiles and runtime adjustments."""

    # -- preset factory -----------------------------------------------------

    @staticmethod
    def _make_profiles() -> Dict[StrictnessLevel, StrictnessProfile]:
        return {
            StrictnessLevel.PERMISSIVE: StrictnessProfile(
                level=StrictnessLevel.PERMISSIVE,
                description="Minimal constraints; suitable for trusted internal tools.",
                safety_threshold=0.20,
                privacy_threshold=0.15,
                fairness_threshold=0.15,
                honesty_threshold=0.20,
                kindness_threshold=0.10,
                auto_correct=False,
                require_human_approval=False,
                max_retries=5,
                timeout_seconds=120.0,
            ),
            StrictnessLevel.RELAXED: StrictnessProfile(
                level=StrictnessLevel.RELAXED,
                description="Light guardrails; appropriate for low-risk public use.",
                safety_threshold=0.40,
                privacy_threshold=0.35,
                fairness_threshold=0.30,
                honesty_threshold=0.35,
                kindness_threshold=0.25,
                auto_correct=True,
                require_human_approval=False,
                max_retries=4,
                timeout_seconds=90.0,
            ),
            StrictnessLevel.MODERATE: StrictnessProfile(
                level=StrictnessLevel.MODERATE,
                description="Balanced defaults for general-purpose interaction.",
                safety_threshold=0.60,
                privacy_threshold=0.55,
                fairness_threshold=0.50,
                honesty_threshold=0.55,
                kindness_threshold=0.45,
                auto_correct=True,
                require_human_approval=False,
                max_retries=3,
                timeout_seconds=60.0,
            ),
            StrictnessLevel.STRICT: StrictnessProfile(
                level=StrictnessLevel.STRICT,
                description="Strong enforcement; for sensitive domains (medical, legal).",
                safety_threshold=0.80,
                privacy_threshold=0.75,
                fairness_threshold=0.70,
                honesty_threshold=0.75,
                kindness_threshold=0.65,
                auto_correct=True,
                require_human_approval=True,
                max_retries=2,
                timeout_seconds=30.0,
            ),
            StrictnessLevel.PARANOID: StrictnessProfile(
                level=StrictnessLevel.PARANOID,
                description="Maximum caution; every output reviewed, zero tolerance.",
                safety_threshold=0.95,
                privacy_threshold=0.95,
                fairness_threshold=0.90,
                honesty_threshold=0.95,
                kindness_threshold=0.85,
                auto_correct=True,
                require_human_approval=True,
                max_retries=1,
                timeout_seconds=15.0,
            ),
        }

    # -- construction -------------------------------------------------------

    def __init__(self, initial_level: StrictnessLevel = StrictnessLevel.MODERATE):
        self._presets: Dict[StrictnessLevel, StrictnessProfile] = self._make_profiles()
        self._current_level: StrictnessLevel = initial_level
        # Deep-copy the preset so runtime adjustments don't mutate the template
        self._current_profile: StrictnessProfile = copy.deepcopy(
            self._presets[initial_level]
        )
        self._history: List[StrictnessChange] = []
        self._callbacks: List[Callable[[StrictnessChange], None]] = []
        self._lock = threading.RLock()

    # -- level management ---------------------------------------------------

    def set_level(self, level: StrictnessLevel, reason: str = "manual change",
                  changed_by: str = "user") -> None:
        """Switch to a different strictness level and record the transition."""
        with self._lock:
            if level == self._current_level:
                return
            old = self._current_level
            self._current_level = level
            self._current_profile = copy.deepcopy(self._presets[level])
            change = StrictnessChange(
                timestamp=time.time(),
                from_level=old,
                to_level=level,
                reason=reason,
                changed_by=changed_by,
            )
            self._history.append(change)
            for cb in self._callbacks:
                try:
                    cb(change)
                except Exception:
                    pass  # callbacks must not break the controller

    def get_level(self) -> StrictnessLevel:
        with self._lock:
            return self._current_level

    # -- per-category fine-tuning -------------------------------------------

    def adjust_category(self, category: str, delta: float) -> None:
        """Nudge a single category threshold by *delta* (clamped to [0, 1])."""
        with self._lock:
            attr_map = {
                "safety": "safety_threshold",
                "privacy": "privacy_threshold",
                "fairness": "fairness_threshold",
                "honesty": "honesty_threshold",
                "kindness": "kindness_threshold",
            }
            attr = attr_map.get(category.lower())
            if attr is None:
                raise ValueError(f"Unknown category '{category}'")
            old_val = getattr(self._current_profile, attr)
            new_val = max(0.0, min(1.0, old_val + delta))
            setattr(self._current_profile, attr, new_val)

    # -- profile access -----------------------------------------------------

    def get_profile(self) -> StrictnessProfile:
        with self._lock:
            return copy.deepcopy(self._current_profile)

    def get_all_profiles(self) -> Dict[StrictnessLevel, StrictnessProfile]:
        with self._lock:
            return {lvl: copy.deepcopy(p) for lvl, p in self._presets.items()}

    # -- comparison ---------------------------------------------------------

    def compare_levels(self, level_a: StrictnessLevel,
                       level_b: StrictnessLevel) -> Dict[str, float]:
        """Return per-category threshold deltas (profile_b - profile_a)."""
        categories = ["safety", "privacy", "fairness", "honesty", "kindness"]
        pa = self._presets[level_a]
        pb = self._presets[level_b]
        result: Dict[str, float] = {}
        for cat in categories:
            result[cat] = round(pb.get_effective_threshold(cat) -
                                pa.get_effective_threshold(cat), 4)
        return result

    # -- callbacks ----------------------------------------------------------

    def register_callback(self, callback: Callable[[StrictnessChange], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    # -- history ------------------------------------------------------------

    def get_history(self) -> List[StrictnessChange]:
        with self._lock:
            return list(self._history)

    # -- serialisation ------------------------------------------------------

    def export_config(self) -> dict:
        """Serialise current profile and level to a plain dict."""
        with self._lock:
            return {
                "level": self._current_level.name,
                "profile": {
                    "description": self._current_profile.description,
                    "safety_threshold": self._current_profile.safety_threshold,
                    "privacy_threshold": self._current_profile.privacy_threshold,
                    "fairness_threshold": self._current_profile.fairness_threshold,
                    "honesty_threshold": self._current_profile.honesty_threshold,
                    "kindness_threshold": self._current_profile.kindness_threshold,
                    "auto_correct": self._current_profile.auto_correct,
                    "require_human_approval": self._current_profile.require_human_approval,
                    "max_retries": self._current_profile.max_retries,
                    "timeout_seconds": self._current_profile.timeout_seconds,
                },
            }

    def import_config(self, config: dict) -> None:
        """Restore state from a dict previously produced by ``export_config``."""
        with self._lock:
            level_name = config.get("level", "MODERATE")
            level = StrictnessLevel[level_name]
            prof = config.get("profile", {})
            self._current_level = level
            self._current_profile = StrictnessProfile(
                level=level,
                description=prof.get("description", ""),
                safety_threshold=float(prof.get("safety_threshold", 0.6)),
                privacy_threshold=float(prof.get("privacy_threshold", 0.55)),
                fairness_threshold=float(prof.get("fairness_threshold", 0.5)),
                honesty_threshold=float(prof.get("honesty_threshold", 0.55)),
                kindness_threshold=float(prof.get("kindness_threshold", 0.45)),
                auto_correct=bool(prof.get("auto_correct", True)),
                require_human_approval=bool(prof.get("require_human_approval", False)),
                max_retries=int(prof.get("max_retries", 3)),
                timeout_seconds=float(prof.get("timeout_seconds", 60.0)),
            )

    # -- reset --------------------------------------------------------------

    def reset_to_default(self) -> None:
        """Revert to the preset profile for the current level."""
        with self._lock:
            self._current_profile = copy.deepcopy(self._presets[self._current_level])


# ---------------------------------------------------------------------------
# 5. AdaptiveDialController
# ---------------------------------------------------------------------------

class AdaptiveDialController(DialController):
    """Extends DialController with automatic risk-responsive adjustment."""

    _CATEGORIES = ("safety", "privacy", "fairness", "honesty", "kindness")

    def __init__(self, initial_level: StrictnessLevel = StrictnessLevel.MODERATE,
                 history_window: int = 50):
        super().__init__(initial_level)
        self._violation_log: deque = deque(maxlen=history_window)
        self._satisfaction_log: deque = deque(maxlen=history_window)
        self._adjustment_log: List[Dict[str, Any]] = []
        self._window = history_window

    # -- public adaptive interface ------------------------------------------

    def auto_adjust(self, context: Dict[str, Any]) -> StrictnessLevel:
        """Evaluate *context* and possibly change strictness level.

        Returns the (possibly new) current level after adjustment.
        """
        risk = self._calculate_context_risk(context)
        current = self.get_level()

        if self._should_adjust_up():
            candidate = self._next_level_up(current)
            if candidate is not None:
                self.set_level(candidate, reason="auto: rising violations",
                              changed_by="auto")
                self._adjustment_log.append({
                    "timestamp": time.time(),
                    "action": "up",
                    "risk": risk,
                    "new_level": candidate.name,
                })
        elif self._should_adjust_down():
            candidate = self._next_level_down(current)
            if candidate is not None:
                self.set_level(candidate, reason="auto: high satisfaction, low risk",
                              changed_by="auto")
                self._adjustment_log.append({
                    "timestamp": time.time(),
                    "action": "down",
                    "risk": risk,
                    "new_level": candidate.name,
                })
        return self.get_level()

    def record_violation(self, severity: float = 1.0,
                         category: str = "safety") -> None:
        """Feed a violation event into the adaptive history."""
        self._violation_log.append({
            "timestamp": time.time(),
            "severity": max(0.0, min(1.0, severity)),
            "category": category,
        })

    def record_satisfaction(self, score: float) -> None:
        """Feed a user-satisfaction signal (0.0 -- 1.0)."""
        self._satisfaction_log.append({
            "timestamp": time.time(),
            "score": max(0.0, min(1.0, score)),
        })

    # -- risk calculation ---------------------------------------------------

    def _calculate_context_risk(self, context: Dict[str, Any]) -> float:
        """Compute a composite risk score in [0, 1] from *context*.

        The context dict may contain:
          - ``violation_count`` (int): recent violations
          - ``sensitive_topics`` (list[str]): topic keywords present
          - ``user_trust_score`` (float): 0-1 trust metric
          - ``domain`` (str): interaction domain tag
          - ``request_complexity`` (float): 0-1 complexity estimate
        """
        risk = 0.0

        # Factor 1: recent violation rate
        recent_violations = len(self._violation_log)
        violation_factor = min(1.0, recent_violations / max(1, self._window * 0.3))
        risk += violation_factor * 0.35

        # Factor 2: sensitive topic density
        sensitive_topics = context.get("sensitive_topics", [])
        if sensitive_topics:
            topic_risk = min(1.0, len(sensitive_topics) / 5.0)
            risk += topic_risk * 0.20

        # Factor 3: inverse of user trust
        trust = float(context.get("user_trust_score", 0.5))
        risk += (1.0 - trust) * 0.15

        # Factor 4: domain risk mapping
        domain = context.get("domain", "").lower()
        high_risk_domains = {"medical", "legal", "financial", "child_safety"}
        medium_risk_domains = {"education", "government", "mental_health"}
        if domain in high_risk_domains:
            risk += 0.20
        elif domain in medium_risk_domains:
            risk += 0.10

        # Factor 5: request complexity
        complexity = float(context.get("request_complexity", 0.0))
        risk += complexity * 0.10

        return max(0.0, min(1.0, risk))

    # -- adjustment heuristics ----------------------------------------------

    def _should_adjust_up(self) -> bool:
        """Return True when recent violation trends justify stricter level."""
        if len(self._violation_log) < 3:
            return False

        # Check if violations are accelerating
        severities = [v["severity"] for v in self._violation_log]
        recent_window = min(10, len(severities))
        recent_avg = statistics.mean(severities[-recent_window:])
        older_window = min(10, len(severities) - recent_window)
        if older_window < 3:
            older_avg = 0.0
        else:
            older_avg = statistics.mean(severities[:older_window])

        # Acceleration criterion
        accelerating = (recent_avg - older_avg) > 0.1

        # Absolute severity criterion
        high_severity = recent_avg > 0.6

        # Frequency criterion: more than 5 violations in the window
        frequent = len(self._violation_log) > self._window * 0.5

        # Satisfaction criterion: low satisfaction
        low_satisfaction = False
        if len(self._satisfaction_log) >= 3:
            recent_sat = [s["score"] for s in list(self._satisfaction_log)[-10:]]
            low_satisfaction = statistics.mean(recent_sat) < 0.3

        # Need at least two signals to trigger upward adjustment
        signals = sum([accelerating, high_severity, frequent, low_satisfaction])
        return signals >= 2

    def _should_adjust_down(self) -> bool:
        """Return True when conditions are safe enough to relax strictness."""
        if len(self._violation_log) < 5:
            return False

        # No recent high-severity violations
        recent = list(self._violation_log)[-10:]
        no_high_severity = all(v["severity"] < 0.3 for v in recent)

        # Violation rate is low
        low_rate = len(self._violation_log) < self._window * 0.1

        # Satisfaction is consistently high
        high_satisfaction = False
        if len(self._satisfaction_log) >= 5:
            recent_sat = [s["score"] for s in list(self._satisfaction_log)[-10:]]
            high_satisfaction = statistics.mean(recent_sat) > 0.8

        # All three conditions must hold to relax
        return no_high_severity and low_rate and high_satisfaction

    # -- recommendation -----------------------------------------------------

    def get_adjustment_recommendation(self) -> Tuple[str, str]:
        """Return (direction, rationale) without actually changing the level.

        *direction* is one of ``"up"``, ``"down"``, ``"hold"``.
        """
        if self._should_adjust_up():
            severity_sum = sum(v["severity"] for v in self._violation_log)
            rationale = (
                f"Violation trend upward (total severity={severity_sum:.2f}, "
                f"count={len(self._violation_log)}). Recommend increasing strictness."
            )
            return ("up", rationale)
        if self._should_adjust_down():
            sat_avg = 0.0
            if self._satisfaction_log:
                sat_avg = statistics.mean(s["score"] for s in self._satisfaction_log)
            rationale = (
                f"Low violation rate and high satisfaction ({sat_avg:.2f}). "
                f"Safe to relax strictness."
            )
            return ("down", rationale)
        return ("hold", "Current strictness level is appropriate for observed conditions.")

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _next_level_up(current: StrictnessLevel) -> Optional[StrictnessLevel]:
        if current == StrictnessLevel.PARANOID:
            return None
        return StrictnessLevel(current.value + 1)

    @staticmethod
    def _next_level_down(current: StrictnessLevel) -> Optional[StrictnessLevel]:
        if current == StrictnessLevel.PERMISSIVE:
            return None
        return StrictnessLevel(current.value - 1)


# ---------------------------------------------------------------------------
# 6. CategoryWeight
# ---------------------------------------------------------------------------

class CategoryWeight:
    """Manages per-category importance weights with normalisation."""

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "safety": 0.30,
        "privacy": 0.25,
        "fairness": 0.20,
        "honesty": 0.15,
        "kindness": 0.10,
    }

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        self._weights: Dict[str, float] = {}
        if initial_weights:
            for k, v in initial_weights.items():
                self._weights[k.lower()] = float(v)
        else:
            self._weights = dict(self._DEFAULT_WEIGHTS)
        self.normalize()

    # -- accessors ----------------------------------------------------------

    def set_weight(self, category: str, weight: float) -> None:
        if weight < 0:
            raise ValueError("Weight must be non-negative")
        self._weights[category.lower()] = float(weight)

    def get_weight(self, category: str) -> float:
        cat = category.lower()
        if cat not in self._weights:
            raise KeyError(f"No weight registered for category '{category}'")
        return self._weights[cat]

    # -- normalisation ------------------------------------------------------

    def normalize(self) -> None:
        """Scale all weights so they sum to 1.0. Zero-weight categories are kept."""
        total = sum(self._weights.values())
        if total == 0:
            # Distribute equally
            n = len(self._weights)
            for k in self._weights:
                self._weights[k] = 1.0 / n if n else 0.0
            return
        for k in self._weights:
            self._weights[k] /= total

    # -- scoring ------------------------------------------------------------

    def get_weighted_score(self, scores: Dict[str, float]) -> float:
        """Compute a weighted composite score from per-category *scores*.

        Categories present in *scores* but not in weights are ignored.
        Categories in weights but missing from *scores* contribute 0.
        """
        composite = 0.0
        for cat, w in self._weights.items():
            val = scores.get(cat, 0.0)
            composite += w * max(0.0, min(1.0, float(val)))
        return max(0.0, min(1.0, composite))

    # -- serialisation ------------------------------------------------------

    def export_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def import_weights(self, weights: Dict[str, float]) -> None:
        self._weights.clear()
        for k, v in weights.items():
            self._weights[k.lower()] = float(v)
        self.normalize()
