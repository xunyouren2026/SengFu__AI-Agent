"""
Secrets Scanner Module

API key patterns (AWS, GCP, Azure), private key detection,
database connection strings, JWT tokens, OAuth credentials, and custom pattern support.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SecretType(Enum):
    """Types of detectable secrets."""
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GCP_SERVICE_KEY = "gcp_service_key"
    GCP_API_KEY = "gcp_api_key"
    AZURE_CLIENT_SECRET = "azure_client_secret"
    AZURE_CONNECTION_STRING = "azure_connection_string"
    PRIVATE_KEY_RSA = "private_key_rsa"
    PRIVATE_KEY_DSA = "private_key_dsa"
    PRIVATE_KEY_EC = "private_key_ec"
    PRIVATE_KEY_GENERIC = "private_key_generic"
    DATABASE_URL = "database_url"
    MONGODB_URI = "mongodb_uri"
    POSTGRES_URI = "postgres_uri"
    MYSQL_URI = "mysql_uri"
    REDIS_URI = "redis_uri"
    JWT_TOKEN = "jwt_token"
    OAUTH_BEARER = "oauth_bearer"
    SLACK_TOKEN = "slack_token"
    SLACK_WEBHOOK = "slack_webhook"
    STRIPE_API_KEY = "stripe_api_key"
    GITHUB_TOKEN = "github_token"
    GITLAB_TOKEN = "gitlab_token"
    DOCKER_HUB_TOKEN = "docker_hub_token"
    ENCRYPTED_PRIVATE_KEY = "encrypted_private_key"
    SSH_PRIVATE_KEY = "ssh_private_key"
    CUSTOM = "custom"


class SecretSeverity(Enum):
    """Severity of detected secrets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecretMatch:
    """A detected secret match."""
    match_id: str
    secret_type: SecretType
    severity: SecretSeverity
    pattern_name: str
    matched_text: str
    redacted_text: str
    file_path: str = ""
    line_number: int = 0
    column_start: int = 0
    column_end: int = 0
    context_before: str = ""
    context_after: str = ""
    confidence: float = 1.0
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.match_id:
            self.match_id = uuid.uuid4().hex[:12]
        if not self.fingerprint:
            self.fingerprint = hashlib.sha256(
                f"{self.secret_type.value}:{self.matched_text}".encode()
            ).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "secret_type": self.secret_type.value,
            "severity": self.severity.value,
            "pattern_name": self.pattern_name,
            "matched_text": self.redacted_text,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "confidence": self.confidence,
            "fingerprint": self.fingerprint,
        }


@dataclass
class ScanResult:
    """Result of a secrets scan."""
    scan_id: str
    timestamp: float = field(default_factory=time.time)
    matches: List[SecretMatch] = field(default_factory=list)
    files_scanned: int = 0
    lines_scanned: int = 0
    scan_duration_ms: float = 0.0
    source: str = ""

    def add_match(self, match: SecretMatch) -> None:
        self.matches.append(match)

    @property
    def critical_count(self) -> int:
        return sum(1 for m in self.matches if m.severity == SecretSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for m in self.matches if m.severity == SecretSeverity.HIGH)

    @property
    def total_risk_score(self) -> float:
        weights = {
            SecretSeverity.CRITICAL: 100.0,
            SecretSeverity.HIGH: 75.0,
            SecretSeverity.MEDIUM: 50.0,
            SecretSeverity.LOW: 25.0,
        }
        return sum(weights.get(m.severity, 0) * m.confidence for m in self.matches)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "match_count": len(self.matches),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "total_risk_score": self.total_risk_score,
            "files_scanned": self.files_scanned,
            "lines_scanned": self.lines_scanned,
            "scan_duration_ms": self.scan_duration_ms,
            "source": self.source,
            "matches": [m.to_dict() for m in self.matches],
        }


class PatternRegistry:
    """Registry of secret detection patterns."""

    def __init__(self) -> None:
        self._patterns: Dict[str, Tuple[re.Pattern, SecretType, SecretSeverity, str]] = {}

    def register(
        self,
        name: str,
        pattern: str,
        secret_type: SecretType,
        severity: SecretSeverity,
        description: str = "",
        flags: int = re.IGNORECASE,
    ) -> None:
        try:
            compiled = re.compile(pattern, flags)
            self._patterns[name] = (compiled, secret_type, severity, description)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{name}': {e}")

    def get_pattern(self, name: str) -> Optional[Tuple[re.Pattern, SecretType, SecretSeverity, str]]:
        return self._patterns.get(name)

    def get_all_patterns(self) -> Dict[str, Tuple[re.Pattern, SecretType, SecretSeverity, str]]:
        return dict(self._patterns)

    def remove_pattern(self, name: str) -> bool:
        return self._patterns.pop(name, None) is not None

    def list_pattern_names(self) -> List[str]:
        return list(self._patterns.keys())


class AWSCredentialDetector:
    """Detects AWS credentials."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, SecretType, SecretSeverity]] = [
            (
                "aws_access_key_id",
                re.compile(r'(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}'),
                SecretType.AWS_ACCESS_KEY,
                SecretSeverity.CRITICAL,
            ),
            (
                "aws_secret_access_key",
                re.compile(r'(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)\s*[:=]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'),
                SecretType.AWS_SECRET_KEY,
                SecretSeverity.CRITICAL,
            ),
            (
                "aws_session_token",
                re.compile(r'(?:aws_session_token|AWS_SESSION_TOKEN)\s*[:=]\s*["\']?([A-Za-z0-9/+=]{200,})["\']?'),
                SecretType.AWS_SECRET_KEY,
                SecretSeverity.CRITICAL,
            ),
            (
                "aws_account_id",
                re.compile(r'(?:aws_account_id|AWS_ACCOUNT_ID)\s*[:=]\s*["\']?(\d{12})["\']?'),
                SecretType.AWS_ACCESS_KEY,
                SecretSeverity.MEDIUM,
            ),
        ]
        self._false_positive_patterns: List[re.Pattern] = [
            re.compile(r'(?:EXAMPLE|example|YOUR_AWS)'),
            re.compile(r'(?:123456789012|111111111111|222222222222)'),
        ]

    def scan(self, content: str, file_path: str = "") -> List[SecretMatch]:
        matches: List[SecretMatch] = []
        for name, pattern, secret_type, severity in self._patterns:
            for match in pattern.finditer(content):
                matched_text = match.group(0)
                if self._is_false_positive(matched_text):
                    continue
                line_num = content[:match.start()].count("\n") + 1
                redacted = self._redact(matched_text)
                matches.append(SecretMatch(
                    match_id=uuid.uuid4().hex[:12],
                    secret_type=secret_type,
                    severity=severity,
                    pattern_name=name,
                    matched_text=matched_text,
                    redacted_text=redacted,
                    file_path=file_path,
                    line_number=line_num,
                    column_start=match.start(),
                    column_end=match.end(),
                    confidence=0.95,
                ))
        return matches

    def _is_false_positive(self, text: str) -> bool:
        for pattern in self._false_positive_patterns:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def _redact(text: str) -> str:
        if len(text) <= 8:
            return "*" * len(text)
        return text[:4] + "*" * (len(text) - 8) + text[-4:]


class GCPCredentialDetector:
    """Detects GCP credentials."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, SecretType, SecretSeverity]] = [
            (
                "gcp_service_account_key",
                re.compile(r'"type":\s*"service_account"'),
                SecretType.GCP_SERVICE_KEY,
                SecretSeverity.CRITICAL,
            ),
            (
                "gcp_private_key",
                re.compile(r'"private_key":\s*"-----BEGIN (?:RSA )?PRIVATE KEY-----'),
                SecretType.GCP_SERVICE_KEY,
                SecretSeverity.CRITICAL,
            ),
            (
                "gcp_client_email",
                re.compile(r'"client_email":\s*"[^@]+@[^\s"]+\.iam\.gserviceaccount\.com"'),
                SecretType.GCP_SERVICE_KEY,
                SecretSeverity.HIGH,
            ),
            (
                "gcp_project_id",
                re.compile(r'"project_id":\s*"[^"]+"'),
                SecretType.GCP_SERVICE_KEY,
                SecretSeverity.LOW,
            ),
            (
                "gcp_api_key",
                re.compile(r'AIza[0-9A-Za-z_-]{35}'),
                SecretType.GCP_API_KEY,
                SecretSeverity.CRITICAL,
            ),
        ]

    def scan(self, content: str, file_path: str = "") -> List[SecretMatch]:
        matches: List[SecretMatch] = []
        for name, pattern, secret_type, severity in self._patterns:
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count("\n") + 1
                matched_text = match.group(0)
                if len(matched_text) > 50:
                    redacted = matched_text[:20] + "..." + matched_text[-10:]
                else:
                    redacted = AWSCredentialDetector._redact(matched_text)
                matches.append(SecretMatch(
                    match_id=uuid.uuid4().hex[:12],
                    secret_type=secret_type,
                    severity=severity,
                    pattern_name=name,
                    matched_text=matched_text,
                    redacted_text=redacted,
                    file_path=file_path,
                    line_number=line_num,
                    column_start=match.start(),
                    column_end=match.end(),
                    confidence=0.9,
                ))
        return matches


class AzureCredentialDetector:
    """Detects Azure credentials."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, SecretType, SecretSeverity]] = [
            (
                "azure_client_secret",
                re.compile(r'(?:azure_client_secret|AZURE_CLIENT_SECRET)\s*[:=]\s*["\']?([A-Za-z0-9_-]{30,})["\']?'),
                SecretType.AZURE_CLIENT_SECRET,
                SecretSeverity.CRITICAL,
            ),
            (
                "azure_connection_string",
                re.compile(
                    r'(?:DefaultEndpointsProtocol|AccountName|AccountKey)'
                    r'=[^\s;]{5,}',
                    re.IGNORECASE,
                ),
                SecretType.AZURE_CONNECTION_STRING,
                SecretSeverity.CRITICAL,
            ),
            (
                "azure_storage_key",
                re.compile(r'(?:azure_storage_key|AZURE_STORAGE_KEY)\s*[:=]\s*["\']?([A-Za-z0-9+/=]{80,})["\']?'),
                SecretType.AZURE_CLIENT_SECRET,
                SecretSeverity.CRITICAL,
            ),
            (
                "azure_tenant_id",
                re.compile(r'(?:azure_tenant_id|AZURE_TENANT_ID)\s*[:=]\s*["\']?([0-9a-f-]{36})["\']?'),
                SecretType.AZURE_CLIENT_SECRET,
                SecretSeverity.MEDIUM,
            ),
            (
                "azure_subscription_id",
                re.compile(r'(?:azure_subscription_id|AZURE_SUBSCRIPTION_ID)\s*[:=]\s*["\']?([0-9a-f-]{36})["\']?'),
                SecretType.AZURE_CLIENT_SECRET,
                SecretSeverity.MEDIUM,
            ),
        ]

    def scan(self, content: str, file_path: str = "") -> List[SecretMatch]:
        matches: List[SecretMatch] = []
        for name, pattern, secret_type, severity in self._patterns:
            for match in pattern.finditer(content):
                matched_text = match.group(0)
                line_num = content[:match.start()].count("\n") + 1
                redacted = AWSCredentialDetector._redact(matched_text)
                matches.append(SecretMatch(
                    match_id=uuid.uuid4().hex[:12],
                    secret_type=secret_type,
                    severity=severity,
                    pattern_name=name,
                    matched_text=matched_text,
                    redacted_text=redacted,
                    file_path=file_path,
                    line_number=line_num,
                    column_start=match.start(),
                    column_end=match.end(),
                    confidence=0.9,
                ))
        return matches


class PrivateKeyDetector:
    """Detects private keys in various formats."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, SecretType, SecretSeverity]] = [
            (
                "rsa_private_key",
                re.compile(r'-----BEGIN RSA PRIVATE KEY-----[\s\S]*?-----END RSA PRIVATE KEY-----'),
                SecretType.PRIVATE_KEY_RSA,
                SecretSeverity.CRITICAL,
            ),
            (
                "dsa_private_key",
                re.compile(r'-----BEGIN DSA PRIVATE KEY-----[\s\S]*?-----END DSA PRIVATE KEY-----'),
                SecretType.PRIVATE_KEY_DSA,
                SecretSeverity.CRITICAL,
            ),
            (
                "ec_private_key",
                re.compile(r'-----BEGIN EC PRIVATE KEY-----[\s\S]*?-----END EC PRIVATE KEY-----'),
                SecretType.PRIVATE_KEY_EC,
                SecretSeverity.CRITICAL,
            ),
            (
                "openssh_private_key",
                re.compile(r'-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----'),
                SecretType.SSH_PRIVATE_KEY,
                SecretSeverity.CRITICAL,
            ),
            (
                "pgp_private_key",
                re.compile(r'-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----'),
                SecretType.PRIVATE_KEY_GENERIC,
                SecretSeverity.CRITICAL,
            ),
            (
                "generic_private_key",
                re.compile(r'-----BEGIN PRIVATE KEY-----[\s\S]*?-----END PRIVATE KEY-----'),
                SecretType.PRIVATE_KEY_GENERIC,
                SecretSeverity.CRITICAL,
            ),
            (
                "encrypted_private_key",
                re.compile(r'-----BEGIN ENCRYPTED PRIVATE KEY-----[\s\S]*?-----END ENCRYPTED PRIVATE KEY-----'),
                SecretType.ENCRYPTED_PRIVATE_KEY,
                SecretSeverity.CRITICAL,
            ),
            (
                "pkcs8_private_key",
                re.compile(r'-----BEGIN RSA PRIVATE KEY-----[\s\S]*?-----END RSA PRIVATE KEY-----'),
                SecretType.PRIVATE_KEY_RSA,
                SecretSeverity.CRITICAL,
            ),
        ]

    def scan(self, content: str, file_path: str = "") -> List[SecretMatch]:
        matches: List[SecretMatch] = []
        for name, pattern, secret_type, severity in self._patterns:
            for match in pattern.finditer(content):
                matched_text = match.group(0)
                line_num = content[:match.start()].count("\n") + 1
                redacted = f"[{secret_type.value} REDACTED, {len(matched_text)} bytes]"
                matches.append(SecretMatch(
                    match_id=uuid.uuid4().hex[:12],
                    secret_type=secret_type,
                    severity=severity,
                    pattern_name=name,
                    matched_text=matched_text,
                    redacted_text=redacted,
                    file_path=file_path,
                    line_number=line_num,
                    column_start=match.start(),
                    column_end=match.end(),
                    confidence=0.99,
                ))
        return matches


class JWTAnalyzer:
    """Analyzes JWT tokens for embedded secrets."""

    def __init__(self) -> None:
        self._jwt_pattern = re.compile(
            r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'
        )
        self._sensitive_claims: List[str] = [
            "password", "secret", "api_key", "token", "private_key",
            "credit_card", "ssn", "social_security",
        ]

    def scan(self, content: str, file_path: str = "") -> List[SecretMatch]:
        matches: List[SecretMatch] = []
        for match in self._jwt_pattern.finditer(content):
            jwt_text = match.group(0)
            line_num = content[:match.start()].count("\n") + 1
            severity = self._analyze_jwt(jwt_text)
            parts = jwt_text.split(".")
            redacted = parts[0][:20] + "..." + parts[2][-10:] if len(parts) == 3 else "[JWT]"
            matches.append(SecretMatch(
                match_id=uuid.uuid4().hex[:12],
                secret_type=SecretType.JWT_TOKEN,
                severity=severity,
                pattern_name="jwt_token",
                matched_text=jwt_text,
                redacted_text=redacted,
                file_path=file_path,
                line_number=line_num,
                column_start=match.start(),
                column_end=match.end(),
                confidence=0.85,
            ))
        return matches

    def _analyze_jwt(self, jwt_text: str) -> SecretSeverity:
        parts = jwt_text.split(".")
        if len(parts) != 3:
            return SecretSeverity.MEDIUM
        try:
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            payload_str = payload_bytes.decode("utf-8", errors="replace")
            payload = json.loads(payload_str)
            if "exp" in payload:
                exp = payload["exp"]
                if isinstance(exp, (int, float)) and time.time() > exp:
                    return SecretSeverity.LOW
            for claim in self._sensitive_claims:
                if claim in payload:
                    return SecretSeverity.CRITICAL
            if "sub" in payload and "iss" in payload:
                return SecretSeverity.HIGH
            return SecretSeverity.MEDIUM
        except Exception:
            return SecretSeverity.MEDIUM


class CustomPatternBuilder:
    """Builds and manages custom secret detection patterns."""

    def __init__(self) -> None:
        self._custom_patterns: Dict[str, Tuple[re.Pattern, SecretSeverity]] = {}
        self._template_builders: Dict[str, Callable[[Dict[str, str]], str]] = {
            "env_assignment": self._build_env_assignment,
            "config_assignment": self._build_config_assignment,
            "url_parameter": self._build_url_parameter,
            "header_value": self._build_header_value,
            "json_field": self._build_json_field,
        }

    def add_custom_pattern(
        self,
        name: str,
        pattern: str,
        severity: SecretSeverity = SecretSeverity.HIGH,
    ) -> None:
        compiled = re.compile(pattern, re.IGNORECASE)
        self._custom_patterns[name] = (compiled, severity)

    def build_from_template(
        self,
        name: str,
        template: str,
        variables: Dict[str, str],
        severity: SecretSeverity = SecretSeverity.HIGH,
    ) -> bool:
        builder = self._template_builders.get(template)
        if builder is None:
            return False
        try:
            pattern = builder(variables)
            self.add_custom_pattern(name, pattern, severity)
            return True
        except Exception:
            return False

    def scan(self, content: str, file_path: str = "") -> List[SecretMatch]:
        matches: List[SecretMatch] = []
        for name, (pattern, severity) in self._custom_patterns.items():
            for match in pattern.finditer(content):
                matched_text = match.group(0)
                line_num = content[:match.start()].count("\n") + 1
                redacted = AWSCredentialDetector._redact(matched_text)
                matches.append(SecretMatch(
                    match_id=uuid.uuid4().hex[:12],
                    secret_type=SecretType.CUSTOM,
                    severity=severity,
                    pattern_name=f"custom_{name}",
                    matched_text=matched_text,
                    redacted_text=redacted,
                    file_path=file_path,
                    line_number=line_num,
                    column_start=match.start(),
                    column_end=match.end(),
                    confidence=0.8,
                ))
        return matches

    def list_patterns(self) -> List[Dict[str, str]]:
        return [{"name": name, "severity": sev.value} for name, (_, sev) in self._custom_patterns.items()]

    def remove_pattern(self, name: str) -> bool:
        return self._custom_patterns.pop(name, None) is not None

    @staticmethod
    def _build_env_assignment(variables: Dict[str, str]) -> str:
        key = re.escape(variables.get("key", r"\w+"))
        return rf'(?:{key})\s*[:=]\s*["\']?([^\s"\']{{8,}})["\']?'

    @staticmethod
    def _build_config_assignment(variables: Dict[str, str]) -> str:
        key = re.escape(variables.get("key", r"\w+"))
        return rf'(?:{key})\s*[:=]\s*["\']?([^\s"\']{{8,}})["\']?'

    @staticmethod
    def _build_url_parameter(variables: Dict[str, str]) -> str:
        param = re.escape(variables.get("param", r"\w+"))
        return rf'{param}=([A-Za-z0-9_-]{{8,}})'

    @staticmethod
    def _build_header_value(variables: Dict[str, str]) -> str:
        header = re.escape(variables.get("header", r"Authorization"))
        return rf'{header}:\s*([A-Za-z0-9_\-\.]{{8,}})'

    @staticmethod
    def _build_json_field(variables: Dict[str, str]) -> str:
        field_name = re.escape(variables.get("field", r"\w+"))
        return rf'"{field_name}"\s*:\s*"([^"]{{8,}})"'


class SecretsScanner:
    """Main secrets scanner orchestrating all detectors."""

    def __init__(
        self,
        enable_aws: bool = True,
        enable_gcp: bool = True,
        enable_azure: bool = True,
        enable_private_keys: bool = True,
        enable_jwt: bool = True,
        enable_custom: bool = True,
    ) -> None:
        self.registry = PatternRegistry()
        self.aws_detector: Optional[AWSCredentialDetector] = AWSCredentialDetector() if enable_aws else None
        self.gcp_detector: Optional[GCPCredentialDetector] = GCPCredentialDetector() if enable_gcp else None
        self.azure_detector: Optional[AzureCredentialDetector] = AzureCredentialDetector() if enable_azure else None
        self.private_key_detector: Optional[PrivateKeyDetector] = PrivateKeyDetector() if enable_private_keys else None
        self.jwt_analyzer: Optional[JWTAnalyzer] = JWTAnalyzer() if enable_jwt else None
        self.custom_builder: CustomPatternBuilder = CustomPatternBuilder()
        self._scan_history: List[ScanResult] = []
        self._max_history: int = 500
        self._unique_fingerprints: Set[str] = set()
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        self.registry.register(
            "database_url",
            r'(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis)://[^\s]+',
            SecretType.DATABASE_URL,
            SecretSeverity.HIGH,
        )
        self.registry.register(
            "slack_token",
            r'xox[baprs]-[0-9a-zA-Z-]{10,}',
            SecretType.SLACK_TOKEN,
            SecretSeverity.HIGH,
        )
        self.registry.register(
            "slack_webhook",
            r'https://hooks\.slack\.com/services/T[0-9A-Z]{8,}/B[0-9A-Z]{8,}/[0-9a-zA-Z]{24,}',
            SecretType.SLACK_WEBHOOK,
            SecretSeverity.HIGH,
        )
        self.registry.register(
            "stripe_api_key",
            r'(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24,}',
            SecretType.STRIPE_API_KEY,
            SecretSeverity.CRITICAL,
        )
        self.registry.register(
            "github_token",
            r'(?:ghp|gho|ghu|ghs|ghr)_[0-9a-zA-Z]{36,}',
            SecretType.GITHUB_TOKEN,
            SecretSeverity.CRITICAL,
        )
        self.registry.register(
            "gitlab_token",
            r'glpat-[0-9a-zA-Z_-]{20,}',
            SecretType.GITLAB_TOKEN,
            SecretSeverity.CRITICAL,
        )
        self.registry.register(
            "docker_hub_token",
            r'dockerhub[._-]?(?:token|password|key)\s*[:=]\s*["\']?([^\s"\']{8,})',
            SecretType.DOCKER_HUB_TOKEN,
            SecretSeverity.HIGH,
        )
        self.registry.register(
            "oauth_bearer",
            r'Bearer\s+[A-Za-z0-9_\-.]{20,}',
            SecretType.OAUTH_BEARER,
            SecretSeverity.HIGH,
        )
        self.registry.register(
            "generic_password",
            r'(?:password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']{4,})["\']?',
            SecretType.CUSTOM,
            SecretSeverity.MEDIUM,
        )
        self.registry.register(
            "generic_secret",
            r'(?:secret|token|api[_-]?key)\s*[:=]\s*["\']?([^\s"\']{8,})["\']?',
            SecretType.CUSTOM,
            SecretSeverity.HIGH,
        )

    def scan_text(
        self,
        content: str,
        file_path: str = "",
        scan_id: Optional[str] = None,
    ) -> ScanResult:
        start_time = time.time()
        if scan_id is None:
            scan_id = uuid.uuid4().hex[:12]
        result = ScanResult(scan_id=scan_id, source=file_path or "inline")
        lines = content.split("\n")
        result.lines_scanned = len(lines)
        result.files_scanned = 1
        detectors = [
            (self.aws_detector, "AWS"),
            (self.gcp_detector, "GCP"),
            (self.azure_detector, "Azure"),
            (self.private_key_detector, "PrivateKey"),
            (self.jwt_analyzer, "JWT"),
        ]
        for detector, name in detectors:
            if detector is not None:
                matches = detector.scan(content, file_path)
                for match in matches:
                    if match.fingerprint not in self._unique_fingerprints:
                        self._unique_fingerprints.add(match.fingerprint)
                        result.add_match(match)
        for pattern_name, (pattern, secret_type, severity, desc) in self.registry.get_all_patterns().items():
            for match in pattern.finditer(content):
                matched_text = match.group(0)
                line_num = content[:match.start()].count("\n") + 1
                redacted = AWSCredentialDetector._redact(matched_text)
                secret_match = SecretMatch(
                    match_id=uuid.uuid4().hex[:12],
                    secret_type=secret_type,
                    severity=severity,
                    pattern_name=pattern_name,
                    matched_text=matched_text,
                    redacted_text=redacted,
                    file_path=file_path,
                    line_number=line_num,
                    confidence=0.8,
                )
                if secret_match.fingerprint not in self._unique_fingerprints:
                    self._unique_fingerprints.add(secret_match.fingerprint)
                    result.add_match(secret_match)
        custom_matches = self.custom_builder.scan(content, file_path)
        for match in custom_matches:
            if match.fingerprint not in self._unique_fingerprints:
                self._unique_fingerprints.add(match.fingerprint)
                result.add_match(match)
        result.scan_duration_ms = (time.time() - start_time) * 1000
        self._scan_history.append(result)
        if len(self._scan_history) > self._max_history:
            self._scan_history = self._scan_history[-self._max_history:]
        return result

    def scan_dict(
        self,
        data: Dict[str, Any],
        source: str = "dict",
    ) -> ScanResult:
        content = json.dumps(data, indent=2, default=str)
        return self.scan_text(content, file_path=source)

    def get_scan_history(self, limit: int = 50) -> List[ScanResult]:
        return self._scan_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        if not self._scan_history:
            return {"total_scans": 0, "total_matches": 0}
        total_matches = sum(len(r.matches) for r in self._scan_history)
        type_counts: Dict[str, int] = {}
        for result in self._scan_history:
            for match in result.matches:
                key = match.secret_type.value
                type_counts[key] = type_counts.get(key, 0) + 1
        return {
            "total_scans": len(self._scan_history),
            "total_matches": total_matches,
            "unique_secrets": len(self._unique_fingerprints),
            "type_distribution": type_counts,
        }
