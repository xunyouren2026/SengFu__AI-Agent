"""
Guardrails module for content filtering and classification.

This module provides topic-based content guardrails including:
- Political content detection
- Violence and harmful content classification
- Drug and illegal substance detection
- Keyword and regex pattern matching
- Confidence scoring and violation levels

Author: AGI Unified Framework Team
"""

from .topics import (
    ViolationLevel,
    TopicCategory,
    GuardrailConfig,
    GuardrailResult,
    KeywordMatcher,
    RegexMatcher,
    TopicClassifier,
    create_default_classifier,
    create_strict_classifier,
    create_permissive_classifier,
)

__all__ = [
    # Enums
    "ViolationLevel",
    "TopicCategory",
    # Config and Result
    "GuardrailConfig",
    "GuardrailResult",
    # Matchers
    "KeywordMatcher",
    "RegexMatcher",
    # Main Classifier
    "TopicClassifier",
    # Factory functions
    "create_default_classifier",
    "create_strict_classifier",
    "create_permissive_classifier",
]
