"""
Field Masking Module

Regex-based masking, format-preserving masking, partial masking,
hash-based masking, and contextual masking rules.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class MaskingMethod(Enum):
    """Methods for masking sensitive data."""
    FULL = "full"
    PARTIAL = "partial"
    HASH = "hash"
    HASH_REVERSIBLE = "hash_reversible"
    FORMAT_PRESERVING = "format_preserving"
    REGEX = "regex"
    CONTEXTUAL = "contextual"
    REPLACE = "replace"
    NULL = "null"
    FIXED = "fixed"


class FieldType(Enum):
    """Types of fields that can be masked."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE = "date"
    NAME = "name"
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


@dataclass
class MaskingRule:
    """A rule defining how to mask a specific field type."""
    rule_id: str
    name: str
    field_type: FieldType
    method: MaskingMethod
    pattern: str = ""
    replacement: str = ""
    mask_char: str = "*"
    visible_prefix: int = 0
    visible_suffix: int = 0
    hash_algorithm: str = "sha256"
    hash_prefix_length: int = 0
    context_rules: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "field_type": self.field_type.value,
            "method": self.method.value,
            "pattern": self.pattern,
            "mask_char": self.mask_char,
            "visible_prefix": self.visible_prefix,
            "visible_suffix": self.visible_suffix,
            "enabled": self.enabled,
        }


@dataclass
class MaskingPolicy:
    """A collection of masking rules."""
    policy_id: str
    name: str
    description: str
    rules: List[MaskingRule] = field(default_factory=list)
    default_method: MaskingMethod = MaskingMethod.FULL
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

    def add_rule(self, rule: MaskingRule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def get_rule(self, field_type: FieldType) -> Optional[MaskingRule]:
        for rule in self.rules:
            if rule.field_type == field_type and rule.enabled:
                return rule
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
            "default_method": self.default_method.value,
            "enabled": self.enabled,
        }


@dataclass
class MaskingResult:
    """Result of a masking operation."""
    original: str
    masked: str
    method: MaskingMethod
    field_type: FieldType
    rule_id: str = ""
    was_masked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_length": len(self.original),
            "masked_length": len(self.masked),
            "method": self.method.value,
            "field_type": self.field_type.value,
            "was_masked": self.was_masked,
        }


class RegexMasker:
    """Masks data using regex patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, MaskingMethod, Dict[str, Any]]] = []

    def add_pattern(
        self,
        name: str,
        pattern: str,
        method: MaskingMethod = MaskingMethod.PARTIAL,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        compiled = re.compile(pattern)
        self._patterns.append((name, compiled, method, options or {}))

    def mask(self, text: str) -> str:
        result = text
        for name, pattern, method, options in self._patterns:
            mask_char = options.get("mask_char", "*")
            visible_prefix = options.get("visible_prefix", 0)
            visible_suffix = options.get("visible_suffix", 0)
            replacement = options.get("replacement", "")

            def replacer(match: re.Match) -> str:
                matched = match.group(0)
                if method == MaskingMethod.FULL:
                    return mask_char * len(matched)
                elif method == MaskingMethod.PARTIAL:
                    return self._partial_mask(matched, mask_char, visible_prefix, visible_suffix)
                elif method == MaskingMethod.REPLACE:
                    return replacement
                elif method == MaskingMethod.NULL:
                    return ""
                return matched

            result = pattern.sub(replacer, result)
        return result

    @staticmethod
    def _partial_mask(text: str, mask_char: str, prefix: int, suffix: int) -> str:
        if len(text) <= prefix + suffix:
            return mask_char * len(text)
        return text[:prefix] + mask_char * (len(text) - prefix - suffix) + text[-suffix:] if suffix > 0 else text[:prefix] + mask_char * (len(text) - prefix)

    def find_matches(self, text: str) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for name, pattern, method, options in self._patterns:
            for match in pattern.finditer(text):
                matches.append({
                    "pattern_name": name,
                    "matched_text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "method": method.value,
                })
        return matches


class FormatPreservingMasker:
    """Masks data while preserving the original format."""

    def __init__(self) -> None:
        self._format_map: Dict[str, Callable[[str], str]] = {
            FieldType.EMAIL.value: self._mask_email,
            FieldType.PHONE.value: self._mask_phone,
            FieldType.SSN.value: self._mask_ssn,
            FieldType.CREDIT_CARD.value: self._mask_credit_card,
            FieldType.IP_ADDRESS.value: self._mask_ip,
            FieldType.DATE.value: self._mask_date,
            FieldType.ZIP_CODE.value: self._mask_zip,
        }
        self._digit_map: Dict[str, str] = {}

    def mask(self, value: str, field_type: FieldType) -> str:
        masker = self._format_map.get(field_type.value)
        if masker:
            return masker(value)
        return self._mask_generic(value)

    def _mask_email(self, email: str) -> str:
        parts = email.split("@")
        if len(parts) != 2:
            return "***@***"
        local = parts[0]
        domain = parts[1]
        if len(local) <= 2:
            masked_local = "**"
        else:
            masked_local = local[0] + "***" + local[-1]
        return f"{masked_local}@{domain}"

    def _mask_phone(self, phone: str) -> str:
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"***-***-{digits[-4:]}"
        elif len(digits) == 11:
            return f"*-{digits[1]}**-***-{digits[-4:]}"
        return "***-***-****"

    def _mask_ssn(self, ssn: str) -> str:
        digits = re.sub(r'\D', '', ssn)
        if len(digits) == 9:
            return f"***-**-{digits[-4:]}"
        return "***-**-****"

    def _mask_credit_card(self, card: str) -> str:
        digits = re.sub(r'\D', '', card)
        if len(digits) >= 13:
            return "*" * (len(digits) - 4) + digits[-4:]
        return "****"

    def _mask_ip(self, ip: str) -> str:
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.*.*"
        return "*.*.*.*"

    def _mask_date(self, date_str: str) -> str:
        return date_str[:4] + "-**-**"

    def _mask_zip(self, zip_code: str) -> str:
        digits = re.sub(r'\D', '', zip_code)
        if len(digits) == 5:
            return "***" + digits[-2:]
        elif len(digits) == 9:
            return "***" + digits[-6:]
        return "*****"

    def _mask_generic(self, value: str) -> str:
        if len(value) <= 2:
            return "**"
        return value[0] + "*" * (len(value) - 2) + value[-1]


class PartialMasker:
    """Applies partial masking to values."""

    def __init__(self, mask_char: str = "*", default_prefix: int = 0, default_suffix: int = 0) -> None:
        self.mask_char: str = mask_char
        self.default_prefix: int = default_prefix
        self.default_suffix: int = default_suffix

    def mask(
        self,
        value: str,
        prefix: Optional[int] = None,
        suffix: Optional[int] = None,
    ) -> str:
        pref = prefix if prefix is not None else self.default_prefix
        suff = suffix if suffix is not None else self.default_suffix
        if not value:
            return value
        if pref + suff >= len(value):
            return self.mask_char * len(value)
        masked_length = len(value) - pref - suff
        return value[:pref] + self.mask_char * masked_length + (value[-suff:] if suff > 0 else "")

    def mask_middle(self, value: str, visible_ratio: float = 0.2) -> str:
        if not value:
            return value
        visible_chars = max(1, int(len(value) * visible_ratio))
        prefix = visible_chars // 2
        suffix = visible_chars - prefix
        return self.mask(value, prefix=prefix, suffix=suffix)


class HashMasker:
    """Masks data using hash functions."""

    def __init__(self, algorithm: str = "sha256", prefix_length: int = 0) -> None:
        self.algorithm: str = algorithm
        self.prefix_length: int = prefix_length
        self._hash_cache: Dict[str, str] = {}

    def mask(self, value: str) -> str:
        if value in self._hash_cache:
            return self._hash_cache[value]
        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(value.encode("utf-8"))
        hashed = hash_obj.hexdigest()
        if self.prefix_length > 0:
            hashed = hashed[:self.prefix_length]
        self._hash_cache[value] = hashed
        return hashed

    def mask_with_prefix(self, value: str, prefix_chars: int = 4) -> str:
        prefix = value[:prefix_chars] if len(value) >= prefix_chars else value
        hashed = self.mask(value)
        return f"{prefix}_{hashed[:8]}"

    def consistent_mask(self, value: str, length: int = 16) -> str:
        full_hash = self.mask(value)
        return full_hash[:length]

    def clear_cache(self) -> None:
        self._hash_cache.clear()


class ContextualMasker:
    """Applies masking based on context rules."""

    def __init__(self) -> None:
        self._context_rules: List[Dict[str, Any]] = []
        self._field_context: Dict[str, Dict[str, Any]] = {}

    def add_context_rule(
        self,
        field_name_pattern: str,
        condition: Callable[[str, Dict[str, Any]], bool],
        masking_method: MaskingMethod,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._context_rules.append({
            "field_pattern": re.compile(field_name_pattern),
            "condition": condition,
            "method": masking_method,
            "options": options or {},
        })

    def set_field_context(self, field_name: str, context: Dict[str, Any]) -> None:
        self._field_context[field_name] = context

    def should_mask(self, field_name: str, value: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, MaskingMethod, Dict[str, Any]]:
        ctx = context or self._field_context.get(field_name, {})
        for rule in self._context_rules:
            if rule["field_pattern"].search(field_name):
                if rule["condition"](value, ctx):
                    return True, rule["method"], rule["options"]
        return False, MaskingMethod.FULL, {}

    def mask_with_context(
        self,
        field_name: str,
        value: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        should, method, options = self.should_mask(field_name, value, context)
        if not should:
            return value
        if method == MaskingMethod.FULL:
            return options.get("mask_char", "*") * len(value)
        elif method == MaskingMethod.PARTIAL:
            prefix = options.get("visible_prefix", 0)
            suffix = options.get("visible_suffix", 0)
            masker = PartialMasker(options.get("mask_char", "*"))
            return masker.mask(value, prefix=prefix, suffix=suffix)
        elif method == MaskingMethod.HASH:
            hasher = HashMasker(options.get("algorithm", "sha256"))
            return hasher.mask(value)
        elif method == MaskingMethod.REPLACE:
            return options.get("replacement", "[REDACTED]")
        return value


class FieldMasker:
    """Main field masking orchestrator."""

    def __init__(self) -> None:
        self.regex_masker = RegexMasker()
        self.format_masker = FormatPreservingMasker()
        self.partial_masker = PartialMasker()
        self.hash_masker = HashMasker()
        self.contextual_masker = ContextualMasker()
        self._policies: Dict[str, MaskingPolicy] = {}
        self._active_policy: Optional[str] = None
        self._field_type_map: Dict[str, FieldType] = {}
        self._default_field_type: FieldType = FieldType.CUSTOM
        self._masking_log: List[Dict[str, Any]] = []
        self._max_log: int = 10000
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        self.regex_masker.add_pattern(
            "email",
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            MaskingMethod.PARTIAL,
            {"mask_char": "*", "visible_prefix": 2, "visible_suffix": 0},
        )
        self.regex_masker.add_pattern(
            "phone",
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            MaskingMethod.PARTIAL,
            {"mask_char": "*", "visible_prefix": 0, "visible_suffix": 4},
        )
        self.regex_masker.add_pattern(
            "ssn",
            r'\b\d{3}-\d{2}-\d{4}\b',
            MaskingMethod.PARTIAL,
            {"mask_char": "*", "visible_prefix": 0, "visible_suffix": 4},
        )
        self.regex_masker.add_pattern(
            "credit_card",
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            MaskingMethod.PARTIAL,
            {"mask_char": "*", "visible_prefix": 0, "visible_suffix": 4},
        )
        self.regex_masker.add_pattern(
            "ip_address",
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            MaskingMethod.PARTIAL,
            {"mask_char": "*", "visible_prefix": 0, "visible_suffix": 0},
        )

    def add_policy(self, policy: MaskingPolicy) -> None:
        self._policies[policy.policy_id] = policy

    def set_active_policy(self, policy_id: str) -> bool:
        if policy_id in self._policies:
            self._active_policy = policy_id
            return True
        return False

    def register_field_type(self, field_name: str, field_type: FieldType) -> None:
        self._field_type_map[field_name] = field_type

    def mask_field(
        self,
        field_name: str,
        value: str,
        field_type: Optional[FieldType] = None,
        method: Optional[MaskingMethod] = None,
    ) -> MaskingResult:
        ft = field_type or self._field_type_map.get(field_name, self._default_field_type)
        policy = self._policies.get(self._active_policy) if self._active_policy else None
        rule = policy.get_rule(ft) if policy else None
        actual_method = method or (rule.method if rule else MaskingMethod.FULL)
        masked_value = self._apply_mask(value, ft, actual_method, rule)
        was_masked = masked_value != value
        result = MaskingResult(
            original=value,
            masked=masked_value,
            method=actual_method,
            field_type=ft,
            rule_id=rule.rule_id if rule else "",
            was_masked=was_masked,
        )
        self._log_masking(field_name, result)
        return result

    def mask_text(self, text: str) -> str:
        return self.regex_masker.mask(text)

    def mask_dict(
        self,
        data: Dict[str, Any],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.mask_field(key, value).masked
            elif isinstance(value, dict) and recursive:
                result[key] = self.mask_dict(value, recursive)
            elif isinstance(value, list) and recursive:
                result[key] = [
                    self.mask_dict(item, recursive) if isinstance(item, dict)
                    else self.mask_field(key, str(item)).masked if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def mask_list(self, items: List[str], field_type: FieldType = FieldType.CUSTOM) -> List[str]:
        return [self.mask_field(f"item_{i}", item, field_type).masked for i, item in enumerate(items)]

    def _apply_mask(
        self,
        value: str,
        field_type: FieldType,
        method: MaskingMethod,
        rule: Optional[MaskingRule],
    ) -> str:
        if not value:
            return value
        if method == MaskingMethod.FULL:
            char = rule.mask_char if rule else "*"
            return char * len(value)
        elif method == MaskingMethod.PARTIAL:
            prefix = rule.visible_prefix if rule else 0
            suffix = rule.visible_suffix if rule else 0
            char = rule.mask_char if rule else "*"
            return self.partial_masker.mask(value, prefix=prefix, suffix=suffix)
        elif method == MaskingMethod.HASH:
            algo = rule.hash_algorithm if rule else "sha256"
            hasher = HashMasker(algo)
            return hasher.mask(value)
        elif method == MaskingMethod.HASH_REVERSIBLE:
            algo = rule.hash_algorithm if rule else "sha256"
            prefix_len = rule.hash_prefix_length if rule else 4
            hasher = HashMasker(algo)
            return hasher.mask_with_prefix(value, prefix_len)
        elif method == MaskingMethod.FORMAT_PRESERVING:
            return self.format_masker.mask(value, field_type)
        elif method == MaskingMethod.REGEX:
            return self.regex_masker.mask(value)
        elif method == MaskingMethod.CONTEXTUAL:
            return self.contextual_masker.mask_with_context(field_type.value, value)
        elif method == MaskingMethod.REPLACE:
            replacement = rule.replacement if rule else "[REDACTED]"
            return replacement
        elif method == MaskingMethod.NULL:
            return ""
        elif method == MaskingMethod.FIXED:
            return rule.replacement if rule else "***"
        return value

    def _log_masking(self, field_name: str, result: MaskingResult) -> None:
        self._masking_log.append({
            "field_name": field_name,
            "method": result.method.value,
            "field_type": result.field_type.value,
            "was_masked": result.was_masked,
            "original_length": len(result.original),
            "timestamp": time.time(),
        })
        if len(self._masking_log) > self._max_log:
            self._masking_log = self._masking_log[-self._max_log:]

    def get_masking_stats(self) -> Dict[str, Any]:
        if not self._masking_log:
            return {"total_masked": 0}
        total = len(self._masking_log)
        masked = sum(1 for entry in self._masking_log if entry["was_masked"])
        method_counts: Dict[str, int] = {}
        for entry in self._masking_log:
            method = entry["method"]
            method_counts[method] = method_counts.get(method, 0) + 1
        return {
            "total_operations": total,
            "total_masked": masked,
            "masking_rate": masked / total if total else 0,
            "method_distribution": method_counts,
        }
