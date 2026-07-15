"""
TestPrivacy - 安全单元测试：隐私保护模块

模块路径: testing/unit/security/test_privacy.py
"""
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


class SensitivityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class PIIMatch:
    pii_type: str
    value: str
    start: int
    end: int
    sensitivity: SensitivityLevel


@dataclass
class PrivacyReport:
    original_text: str
    anonymized_text: str
    pii_found: List[PIIMatch]
    risk_score: float
    sensitivity_level: SensitivityLevel


class MockPrivacyGuard:
    """模拟隐私保护系统"""

    PII_PATTERNS = [
        (r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "SSN", SensitivityLevel.RESTRICTED),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
         "EMAIL", SensitivityLevel.CONFIDENTIAL),
        (r"\b\+?1?[-.]?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
         "PHONE", SensitivityLevel.CONFIDENTIAL),
        (r"\b\d{16}\b", "CREDIT_CARD", SensitivityLevel.RESTRICTED),
        (r"\b\d{4}[-.]?\d{4}[-.]?\d{4}[-.]?\d{4}\b",
         "CREDIT_CARD_DASHED", SensitivityLevel.RESTRICTED),
        (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
         "IP_ADDRESS", SensitivityLevel.INTERNAL),
        (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
         r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
         "DATE", SensitivityLevel.INTERNAL),
        (r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", "DATE_ISO", SensitivityLevel.INTERNAL),
    ]

    NAME_PATTERNS = [
        (r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b", "NAMEWithTitle"),
        (r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "POSSIBLE_NAME"),
    ]

    def __init__(self):
        self.detection_history: List[PrivacyReport] = []
        self.custom_patterns: List[Tuple[str, str, SensitivityLevel]] = []

    def detect_pii(self, text: str) -> List[PIIMatch]:
        matches = []
        for pattern, pii_type, sensitivity in self.PII_PATTERNS + self.custom_patterns:
            for m in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    pii_type=pii_type, value=m.group(),
                    start=m.start(), end=m.end(),
                    sensitivity=sensitivity,
                ))
        return matches

    def mask_pii(self, text: str, mask_char: str = "*") -> str:
        matches = self.detect_pii(text)
        result = text
        offset = 0
        for m in sorted(matches, key=lambda x: x.start):
            masked = mask_char * (m.end - m.start)
            result = result[:m.start + offset] + masked + result[m.end + offset:]
            offset += len(masked) - (m.end - m.start)
        return result

    def anonymize(self, text: str) -> str:
        matches = self.detect_pii(text)
        result = text
        offset = 0
        type_counters: Dict[str, int] = {}
        for m in sorted(matches, key=lambda x: x.start):
            type_counters[m.pii_type] = type_counters.get(m.pii_type, 0) + 1
            replacement = f"[{m.pii_type}_{type_counters[m.pii_type]}]"
            result = result[:m.start + offset] + replacement + result[m.end + offset:]
            offset += len(replacement) - (m.end - m.start)
        return result

    def hash_pii(self, text: str) -> str:
        matches = self.detect_pii(text)
        result = text
        offset = 0
        for m in sorted(matches, key=lambda x: x.start):
            hashed = hashlib.sha256(m.value.encode()).hexdigest()[:12]
            result = result[:m.start + offset] + hashed + result[m.end + offset:]
            offset += len(hashed) - (m.end - m.start)
        return result

    def compute_risk_score(self, pii_matches: List[PIIMatch]) -> float:
        if not pii_matches:
            return 0.0
        score = 0.0
        for m in pii_matches:
            weights = {
                SensitivityLevel.PUBLIC: 0.0,
                SensitivityLevel.INTERNAL: 0.2,
                SensitivityLevel.CONFIDENTIAL: 0.5,
                SensitivityLevel.RESTRICTED: 1.0,
            }
            score += weights.get(m.sensitivity, 0.3)
        return min(1.0, score / 5.0)

    def classify_sensitivity(self, risk_score: float) -> SensitivityLevel:
        if risk_score == 0:
            return SensitivityLevel.PUBLIC
        elif risk_score < 0.3:
            return SensitivityLevel.INTERNAL
        elif risk_score < 0.6:
            return SensitivityLevel.CONFIDENTIAL
        else:
            return SensitivityLevel.RESTRICTED

    def full_report(self, text: str) -> PrivacyReport:
        pii = self.detect_pii(text)
        risk = self.compute_risk_score(pii)
        return PrivacyReport(
            original_text=text,
            anonymized_text=self.anonymize(text),
            pii_found=pii,
            risk_score=risk,
            sensitivity_level=self.classify_sensitivity(risk),
        )

    def add_custom_pattern(self, pattern: str, pii_type: str,
                           sensitivity: SensitivityLevel):
        self.custom_patterns.append((pattern, pii_type, sensitivity))


class TestPIIDetection:
    """PII检测测试"""

    def setup_method(self):
        self.guard = MockPrivacyGuard()

    def test_detect_email(self):
        matches = self.guard.detect_pii("Contact me at user@example.com")
        assert any(m.pii_type == "EMAIL" for m in matches)

    def test_detect_phone(self):
        matches = self.guard.detect_pii("Call me at 555-123-4567")
        assert any(m.pii_type == "PHONE" for m in matches)

    def test_detect_ssn(self):
        matches = self.guard.detect_pii("SSN: 123-45-6789")
        assert any(m.pii_type == "SSN" for m in matches)

    def test_detect_ip_address(self):
        matches = self.guard.detect_pii("Server at 192.168.1.100")
        assert any(m.pii_type == "IP_ADDRESS" for m in matches)

    def test_detect_date_iso(self):
        matches = self.guard.detect_pii("Born on 1990-01-15")
        assert any(m.pii_type == "DATE_ISO" for m in matches)

    def test_detect_credit_card(self):
        matches = self.guard.detect_pii("Card: 4111-1111-1111-1111")
        assert any("CREDIT_CARD" in m.pii_type for m in matches)

    def test_no_pii_in_safe_text(self):
        matches = self.guard.detect_pii("Hello, how are you today?")
        assert len(matches) == 0

    def test_multiple_pii_types(self):
        text = "Email: test@test.com, Phone: 555-000-0000, SSN: 000-00-0000"
        matches = self.guard.detect_pii(text)
        types = {m.pii_type for m in matches}
        assert "EMAIL" in types
        assert "PHONE" in types
        assert "SSN" in types

    def test_custom_pattern(self):
        self.guard.add_custom_pattern(r"\bID-\d+\b", "EMPLOYEE_ID",
                                      SensitivityLevel.CONFIDENTIAL)
        matches = self.guard.detect_pii("Employee ID-12345")
        assert any(m.pii_type == "EMPLOYEE_ID" for m in matches)


class TestPIIMasking:
    """PII遮蔽测试"""

    def setup_method(self):
        self.guard = MockPrivacyGuard()

    def test_mask_email(self):
        result = self.guard.mask_pii("user@example.com")
        assert "@" not in result
        assert "*" in result

    def test_mask_phone(self):
        result = self.guard.mask_pii("555-123-4567")
        assert "555" not in result

    def test_mask_preserves_length(self):
        text = "user@example.com"
        masked = self.guard.mask_pii(text)
        assert len(masked) == len(text)

    def test_mask_multiple(self):
        text = "a@b.com and c@d.com"
        result = self.guard.mask_pii(text)
        assert "@" not in result

    def test_custom_mask_char(self):
        result = self.guard.mask_pii("user@example.com", mask_char="#")
        assert "#" in result
        assert "*" not in result


class TestPIIAnonymization:
    """PII匿名化测试"""

    def setup_method(self):
        self.guard = MockPrivacyGuard()

    def test_anonymize_email(self):
        result = self.guard.anonymize("Contact: user@example.com")
        assert "@" not in result
        assert "[EMAIL_1]" in result

    def test_anonymize_multiple_same_type(self):
        result = self.guard.anonymize("a@b.com and c@d.com")
        assert "[EMAIL_1]" in result
        assert "[EMAIL_2]" in result

    def test_anonymize_preserves_non_pii(self):
        result = self.guard.anonymize("Hello, my email is test@test.com")
        assert "Hello" in result
        assert "my email is" in result


class TestRiskScoring:
    """风险评分测试"""

    def setup_method(self):
        self.guard = MockPrivacyGuard()

    def test_no_pii_zero_risk(self):
        score = self.guard.compute_risk_score([])
        assert score == 0.0

    def test_restricted_pii_high_risk(self):
        matches = [PIIMatch("SSN", "123-45-6789", 0, 11,
                            SensitivityLevel.RESTRICTED)]
        score = self.guard.compute_risk_score(matches)
        assert score > 0.5

    def test_internal_pii_low_risk(self):
        matches = [PIIMatch("DATE", "2023-01-01", 0, 10,
                            SensitivityLevel.INTERNAL)]
        score = self.guard.compute_risk_score(matches)
        assert score < 0.3

    def test_risk_score_capped(self):
        matches = [PIIMatch("SSN", "x", 0, 1, SensitivityLevel.RESTRICTED)] * 10
        score = self.guard.compute_risk_score(matches)
        assert score <= 1.0

    def test_classify_public(self):
        assert self.guard.classify_sensitivity(0.0) == SensitivityLevel.PUBLIC

    def test_classify_restricted(self):
        assert self.guard.classify_sensitivity(0.8) == SensitivityLevel.RESTRICTED

    def test_full_report(self):
        report = self.guard.full_report("Email: test@test.com, SSN: 000-00-0000")
        assert len(report.pii_found) >= 2
        assert report.risk_score > 0
        assert report.anonymized_text != report.original_text


class TestPrivacyHashing:
    """PII哈希测试"""

    def setup_method(self):
        self.guard = MockPrivacyGuard()

    def test_hash_replaces_pii(self):
        result = self.guard.hash_pii("user@example.com")
        assert "@" not in result
        assert len(result) > 0

    def test_hash_deterministic(self):
        r1 = self.guard.hash_pii("user@example.com")
        r2 = self.guard.hash_pii("user@example.com")
        assert r1 == r2

    def test_hash_different_inputs(self):
        r1 = self.guard.hash_pii("user1@example.com")
        r2 = self.guard.hash_pii("user2@example.com")
        assert r1 != r2
