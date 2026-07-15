"""
Response Blocking Module

Real-time content inspection, keyword matching, regex pattern matching,
ML-based scoring (simulated), immediate interruption, and replacement suggestions.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class BlockReason(Enum):
    """Reasons for blocking a response."""
    KEYWORD = "keyword"
    PATTERN = "pattern"
    ML_SCORE = "ml_score"
    LENGTH = "length"
    CONTENT_TYPE = "content_type"
    POLICY = "policy"
    CUSTOM = "custom"


class BlockAction(Enum):
    """Actions to take when content is blocked."""
    BLOCK_FULL = "block_full"
    BLOCK_PARTIAL = "block_partial"
    REDACT = "redact"
    REPLACE = "replace"
    WARN = "warn"
    QUARANTINE = "quarantine"


@dataclass
class BlockRule:
    """A rule for blocking content."""
    rule_id: str
    name: str
    reason: BlockReason
    action: BlockAction
    pattern: str = ""
    keywords: List[str] = field(default_factory=list)
    threshold: float = 0.0
    enabled: bool = True
    priority: int = 0
    replacement: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "reason": self.reason.value,
            "action": self.action.value,
            "pattern": self.pattern,
            "keywords": self.keywords,
            "threshold": self.threshold,
            "enabled": self.enabled,
            "priority": self.priority,
        }


@dataclass
class BlockResult:
    """Result of a content block check."""
    result_id: str
    blocked: bool
    action: BlockAction
    reason: str
    matched_rules: List[str] = field(default_factory=list)
    score: float = 0.0
    original_content: str = ""
    modified_content: str = ""
    replacements: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "blocked": self.blocked,
            "action": self.action.value,
            "reason": self.reason,
            "matched_rules": self.matched_rules,
            "score": self.score,
            "original_length": len(self.original_content),
            "modified_length": len(self.modified_content),
            "replacement_count": len(self.replacements),
        }


class KeywordMatcher:
    """Matches keywords in content for blocking."""

    def __init__(self) -> None:
        self._keyword_rules: Dict[str, BlockRule] = {}
        self._keyword_set: Set[str] = set()
        self._phrase_set: Set[str] = set()
        self._compiled_phrases: List[Tuple[str, str]] = []

    def add_keywords(self, rule: BlockRule) -> None:
        self._keyword_rules[rule.rule_id] = rule
        for kw in rule.keywords:
            if " " in kw:
                self._phrase_set.add(kw.lower())
                self._compiled_phrases.append((kw.lower(), rule.rule_id))
            else:
                self._keyword_set.add(kw.lower())

    def remove_keywords(self, rule_id: str) -> None:
        rule = self._keyword_rules.pop(rule_id, None)
        if rule:
            for kw in rule.keywords:
                self._keyword_set.discard(kw.lower())
                self._phrase_set.discard(kw.lower())
            self._compiled_phrases = [
                (p, rid) for p, rid in self._compiled_phrases if rid != rule_id
            ]

    def match(self, content: str) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        content_lower = content.lower()
        words = set(re.findall(r'\b\w+\b', content_lower))
        matched_keywords = words & self._keyword_set
        for kw in matched_keywords:
            rule_id = self._find_rule_for_keyword(kw)
            if rule_id:
                rule = self._keyword_rules.get(rule_id)
                if rule and rule.enabled:
                    matches.append({
                        "type": "keyword",
                        "keyword": kw,
                        "rule_id": rule_id,
                        "rule_name": rule.name,
                        "action": rule.action,
                    })
        for phrase, rule_id in self._compiled_phrases:
            if phrase in content_lower:
                rule = self._keyword_rules.get(rule_id)
                if rule and rule.enabled:
                    matches.append({
                        "type": "phrase",
                        "keyword": phrase,
                        "rule_id": rule_id,
                        "rule_name": rule.name,
                        "action": rule.action,
                    })
        return matches

    def _find_rule_for_keyword(self, keyword: str) -> Optional[str]:
        for rule_id, rule in self._keyword_rules.items():
            if keyword in [k.lower() for k in rule.keywords]:
                return rule_id
        return None

    def get_keyword_count(self) -> int:
        return len(self._keyword_set) + len(self._phrase_set)


class RegexMatcher:
    """Matches regex patterns in content for blocking."""

    def __init__(self) -> None:
        self._pattern_rules: List[Tuple[str, re.Pattern, BlockRule]] = []

    def add_pattern(self, rule: BlockRule) -> None:
        try:
            compiled = re.compile(rule.pattern, re.IGNORECASE)
            self._pattern_rules.append((rule.rule_id, compiled, rule))
        except re.error:
            pass

    def remove_pattern(self, rule_id: str) -> None:
        self._pattern_rules = [
            (rid, pat, rule) for rid, pat, rule in self._pattern_rules
            if rid != rule_id
        ]

    def match(self, content: str) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for rule_id, pattern, rule in self._pattern_rules:
            if not rule.enabled:
                continue
            for m in pattern.finditer(content):
                matches.append({
                    "type": "pattern",
                    "matched_text": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                    "rule_id": rule_id,
                    "rule_name": rule.name,
                    "action": rule.action,
                })
        return matches

    def get_pattern_count(self) -> int:
        return len(self._pattern_rules)


class MLScoreSimulator:
    """Simulates ML-based content scoring for blocking decisions."""

    def __init__(self) -> None:
        self._risk_features: Dict[str, float] = {
            "profanity": 0.3,
            "violence": 0.5,
            "hate_speech": 0.7,
            "sexual_content": 0.6,
            "personal_info": 0.4,
            "illegal_activity": 0.8,
            "self_harm": 0.9,
            "medical_advice": 0.5,
            "legal_advice": 0.3,
            "financial_advice": 0.2,
        }
        self._category_keywords: Dict[str, List[str]] = {
            "profanity": ["damn", "hell", "crap", "ass"],
            "violence": ["kill", "fight", "attack", "hurt", "weapon", "bomb"],
            "hate_speech": ["hate", "racist", "supremacist", "bigot"],
            "sexual_content": ["explicit", "nsfw", "adult content"],
            "personal_info": ["ssn", "social security", "credit card number"],
            "illegal_activity": ["hack", "exploit", "bypass", "circumvent"],
            "self_harm": ["suicide", "self-harm", "end my life"],
            "medical_advice": ["diagnosis", "prescription", "dosage"],
            "legal_advice": ["lawsuit", "litigation", "legal action"],
            "financial_advice": ["investment", "stock tip", "guaranteed return"],
        }
        self._block_threshold: float = 0.7
        self._warn_threshold: float = 0.5
        self._scoring_history: List[Dict[str, Any]] = []
        self._max_history: int = 1000

    def score(self, content: str) -> Dict[str, Any]:
        content_lower = content.lower()
        words = set(re.findall(r'\b\w+\b', content_lower))
        category_scores: Dict[str, float] = {}
        for category, keywords in self._category_keywords.items():
            matches = words & set(keywords)
            if matches:
                match_ratio = len(matches) / len(keywords)
                category_scores[category] = min(1.0, match_ratio * 3.0)
            else:
                category_scores[category] = 0.0
        weighted_score = 0.0
        total_weight = 0.0
        for category, score in category_scores.items():
            weight = self._risk_features.get(category, 0.5)
            weighted_score += score * weight
            total_weight += weight
        overall_score = weighted_score / total_weight if total_weight else 0.0
        length_factor = min(1.0, len(content) / 1000.0)
        density_factor = sum(category_scores.values()) / len(category_scores) if category_scores else 0.0
        final_score = min(1.0, overall_score * 0.6 + density_factor * 0.3 + length_factor * 0.1)
        result = {
            "overall_score": final_score,
            "category_scores": category_scores,
            "action": self._determine_action(final_score),
            "thresholds": {
                "block": self._block_threshold,
                "warn": self._warn_threshold,
            },
        }
        self._scoring_history.append({
            "score": final_score,
            "action": result["action"],
            "timestamp": time.time(),
        })
        if len(self._scoring_history) > self._max_history:
            self._scoring_history = self._scoring_history[-self._max_history:]
        return result

    def _determine_action(self, score: float) -> str:
        if score >= self._block_threshold:
            return "block"
        elif score >= self._warn_threshold:
            return "warn"
        return "allow"

    def set_thresholds(self, block: float, warn: float) -> None:
        self._block_threshold = block
        self._warn_threshold = warn

    def add_category_keywords(self, category: str, keywords: List[str]) -> None:
        if category in self._category_keywords:
            self._category_keywords[category].extend(keywords)
        else:
            self._category_keywords[category] = keywords
            self._risk_features[category] = 0.5

    def get_scoring_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._scoring_history[-limit:]


class InterruptionHandler:
    """Handles immediate interruption of content generation."""

    def __init__(self) -> None:
        self._interrupt_tokens: List[str] = [
            "\n\nHuman:", "\n\nAssistant:",
            "<|im_end|>", "<|endoftext|>",
            "[INST]", "[/INST]",
        ]
        self._max_response_length: int = 10000
        self._interrupt_callbacks: List[Callable[[str, str], None]] = []

    def add_interrupt_token(self, token: str) -> None:
        self._interrupt_tokens.append(token)

    def set_max_length(self, max_length: int) -> None:
        self._max_response_length = max_length

    def register_callback(self, callback: Callable[[str, str], None]) -> None:
        self._interrupt_callbacks.append(callback)

    def check_interrupt(self, content: str) -> Optional[Dict[str, Any]]:
        for token in self._interrupt_tokens:
            if token in content:
                idx = content.index(token)
                return {
                    "interrupted": True,
                    "reason": f"Interrupt token found: '{token[:20]}'",
                    "position": idx,
                    "truncated_content": content[:idx],
                }
        if len(content) > self._max_response_length:
            return {
                "interrupted": True,
                "reason": f"Response length {len(content)} exceeds maximum {self._max_response_length}",
                "position": self._max_response_length,
                "truncated_content": content[:self._max_response_length],
            }
        return None

    def truncate(self, content: str) -> str:
        result = self.check_interrupt(content)
        if result and result["interrupted"]:
            return result["truncated_content"]
        return content

    def notify_interrupt(self, content: str, reason: str) -> None:
        for callback in self._interrupt_callbacks:
            try:
                callback(content, reason)
            except Exception:
                pass


class ReplacementSuggester:
    """Suggests replacements for blocked content."""

    def __init__(self) -> None:
        self._replacement_map: Dict[str, str] = {
            "password": "[REDACTED_CREDENTIAL]",
            "secret": "[REDACTED_SECRET]",
            "api_key": "[REDACTED_KEY]",
            "token": "[REDACTED_TOKEN]",
            "ssn": "[REDACTED_SSN]",
            "credit card": "[REDACTED_CARD]",
            "social security": "[REDACTED_SSN]",
            "private key": "[REDACTED_KEY]",
            "connection string": "[REDACTED_CONNECTION]",
        }
        self._generic_replacements: Dict[str, str] = {
            BlockReason.KEYWORD: "[FILTERED]",
            BlockReason.PATTERN: "[REDACTED]",
            BlockReason.ML_SCORE: "[REVIEW_REQUIRED]",
            BlockReason.CONTENT_TYPE: "[UNSUPPORTED_TYPE]",
        }

    def add_replacement(self, original: str, replacement: str) -> None:
        self._replacement_map[original.lower()] = replacement

    def suggest(self, content: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        replacements: List[Dict[str, Any]] = []
        modified = content
        offset_adjustment = 0
        sorted_matches = sorted(matches, key=lambda m: m.get("start", 0))
        for match in sorted_matches:
            matched_text = match.get("matched_text", match.get("keyword", ""))
            if not matched_text:
                continue
            replacement = self._find_replacement(matched_text, match)
            start = match.get("start", 0)
            if start > 0:
                actual_start = start + offset_adjustment
                modified = modified[:actual_start] + replacement + modified[actual_start + len(matched_text):]
                offset_adjustment += len(replacement) - len(matched_text)
            else:
                modified = modified.replace(matched_text, replacement, 1)
            replacements.append({
                "original": matched_text,
                "replacement": replacement,
                "position": start,
                "rule_id": match.get("rule_id", ""),
            })
        return replacements

    def _find_replacement(self, matched_text: str, match: Dict[str, Any]) -> str:
        text_lower = matched_text.lower()
        for original, replacement in self._replacement_map.items():
            if original in text_lower:
                return replacement
        reason = match.get("reason", "")
        if reason:
            try:
                return self._generic_replacements.get(BlockReason(reason), "[FILTERED]")
            except ValueError:
                pass
        rule = match.get("rule_name", "")
        if rule:
            return f"[FILTERED_BY_{rule.upper()}]"
        return "[FILTERED]"

    def apply_replacements(
        self, content: str, matches: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        replacements = self.suggest(content, matches)
        modified = content
        for rep in replacements:
            modified = modified.replace(rep["original"], rep["replacement"], 1)
        return modified, replacements


class ContentInspector:
    """Main content inspection orchestrator."""

    def __init__(self) -> None:
        self.keyword_matcher = KeywordMatcher()
        self.regex_matcher = RegexMatcher()
        self.ml_scorer = MLScoreSimulator()
        self.replacement_suggester = ReplacementSuggester()
        self.interruption_handler = InterruptionHandler()
        self._inspection_count: int = 0
        self._block_count: int = 0

    def inspect(self, content: str) -> BlockResult:
        self._inspection_count += 1
        result_id = uuid.uuid4().hex[:12]
        all_matches: List[Dict[str, Any]] = []
        keyword_matches = self.keyword_matcher.match(content)
        all_matches.extend(keyword_matches)
        pattern_matches = self.regex_matcher.match(content)
        all_matches.extend(pattern_matches)
        ml_result = self.ml_scorer.score(content)
        if ml_result["action"] == "block":
            all_matches.append({
                "type": "ml_score",
                "score": ml_result["overall_score"],
                "rule_id": "ml_scorer",
                "rule_name": "ML Content Scorer",
                "action": "block",
            })
        elif ml_result["action"] == "warn":
            all_matches.append({
                "type": "ml_score",
                "score": ml_result["overall_score"],
                "rule_id": "ml_scorer",
                "rule_name": "ML Content Scorer",
                "action": "warn",
            })
        interrupt = self.interruption_handler.check_interrupt(content)
        if interrupt and interrupt["interrupted"]:
            all_matches.append({
                "type": "interruption",
                "reason": interrupt["reason"],
                "rule_id": "interrupt_handler",
                "rule_name": "Interruption Handler",
                "action": "block",
            })
        if not all_matches:
            return BlockResult(
                result_id=result_id,
                blocked=False,
                action=BlockAction.WARN,
                reason="No blocking rules matched",
                original_content=content,
                modified_content=content,
                score=ml_result["overall_score"],
            )
        has_block = any(
            m.get("action") in ("block", BlockAction.BLOCK_FULL.value)
            for m in all_matches
        )
        has_warn = any(
            m.get("action") in ("warn", BlockAction.WARN.value)
            for m in all_matches
        )
        if has_block:
            action = BlockAction.BLOCK_FULL
            self._block_count += 1
        elif has_warn:
            action = BlockAction.WARN
        else:
            action = BlockAction.WARN
        modified_content, replacements = self.replacement_suggester.apply_replacements(
            content, all_matches
        )
        matched_rules = list(set(m.get("rule_id", "") for m in all_matches if m.get("rule_id")))
        reasons = [m.get("rule_name", m.get("type", "")) for m in all_matches if m.get("action") in ("block", "warn")]
        return BlockResult(
            result_id=result_id,
            blocked=has_block,
            action=action,
            reason="; ".join(set(reasons)) if reasons else "Content flagged",
            matched_rules=matched_rules,
            score=ml_result["overall_score"],
            original_content=content,
            modified_content=modified_content if has_block or has_warn else content,
            replacements=replacements,
        )

    def add_keyword_rule(self, rule: BlockRule) -> None:
        self.keyword_matcher.add_keywords(rule)

    def add_pattern_rule(self, rule: BlockRule) -> None:
        self.regex_matcher.add_pattern(rule)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_inspections": self._inspection_count,
            "total_blocks": self._block_count,
            "block_rate": self._block_count / self._inspection_count if self._inspection_count else 0,
            "keyword_rules": self.keyword_matcher.get_keyword_count(),
            "pattern_rules": self.regex_matcher.get_pattern_count(),
        }


class ResponseBlocker:
    """Main response blocker class."""

    def __init__(
        self,
        block_threshold: float = 0.7,
        warn_threshold: float = 0.5,
        max_response_length: int = 10000,
    ) -> None:
        self.content_inspector = ContentInspector()
        self.content_inspector.ml_scorer.set_thresholds(block_threshold, warn_threshold)
        self.content_inspector.interruption_handler.set_max_length(max_response_length)
        self._default_block_rules: List[BlockRule] = []
        self._session_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"inspected": 0, "blocked": 0})

    def add_default_rules(self) -> None:
        self.content_inspector.add_keyword_rule(BlockRule(
            rule_id="default_profanity",
            name="Profanity Filter",
            reason=BlockReason.KEYWORD,
            action=BlockAction.WARN,
            keywords=["damn", "hell", "crap"],
        ))
        self.content_inspector.add_pattern_rule(BlockRule(
            rule_id="default_email",
            name="Email Pattern",
            reason=BlockReason.PATTERN,
            action=BlockAction.REDACT,
            pattern=r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            replacement="[EMAIL_REDACTED]",
        ))
        self.content_inspector.add_pattern_rule(BlockRule(
            rule_id="default_phone",
            name="Phone Pattern",
            reason=BlockReason.PATTERN,
            action=BlockAction.REDACT,
            pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            replacement="[PHONE_REDACTED]",
        ))
        self.content_inspector.add_pattern_rule(BlockRule(
            rule_id="default_ssn",
            name="SSN Pattern",
            reason=BlockReason.PATTERN,
            action=BlockAction.BLOCK_FULL,
            pattern=r'\b\d{3}-\d{2}-\d{4}\b',
        ))

    def check_response(
        self,
        content: str,
        session_id: str = "",
    ) -> BlockResult:
        result = self.content_inspector.inspect(content)
        if session_id:
            self._session_stats[session_id]["inspected"] += 1
            if result.blocked:
                self._session_stats[session_id]["blocked"] += 1
        return result

    def check_streaming_chunk(
        self,
        chunk: str,
        accumulated: str,
        session_id: str = "",
    ) -> BlockResult:
        combined = accumulated + chunk
        result = self.content_inspector.inspect(combined)
        if session_id:
            self._session_stats[session_id]["inspected"] += 1
            if result.blocked:
                self._session_stats[session_id]["blocked"] += 1
        return result

    def get_session_stats(self, session_id: str) -> Dict[str, int]:
        return dict(self._session_stats.get(session_id, {"inspected": 0, "blocked": 0}))

    def get_global_stats(self) -> Dict[str, Any]:
        stats = self.content_inspector.get_statistics()
        total_sessions = len(self._session_stats)
        total_session_blocks = sum(s["blocked"] for s in self._session_stats.values())
        stats["total_sessions"] = total_sessions
        stats["total_session_blocks"] = total_session_blocks
        return stats
