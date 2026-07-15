"""
Parameter Risk Scanner Module

Command injection patterns, path traversal detection, SQL injection detection,
XSS patterns, SSRF detection, and deserialization attack patterns.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class RiskCategory(Enum):
    """Categories of security risks."""
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    SSRF = "ssrf"
    DESERIALIZATION = "deserialization"
    CODE_INJECTION = "code_injection"
    PROTOCOL_SMUGGLING = "protocol_smuggling"
    LOG_INJECTION = "log_injection"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    TEMPLATE_INJECTION = "template_injection"


class RiskSeverity(Enum):
    """Severity levels for detected risks."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskFinding:
    """A single risk finding from scanning."""
    finding_id: str
    category: RiskCategory
    severity: RiskSeverity
    pattern: str
    matched_text: str
    parameter_name: str
    description: str
    remediation: str = ""
    confidence: float = 1.0
    location: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "pattern": self.pattern,
            "matched_text": self.matched_text[:200],
            "parameter_name": self.parameter_name,
            "description": self.description,
            "remediation": self.remediation,
            "confidence": self.confidence,
            "location": self.location,
        }


@dataclass
class RiskReport:
    """Comprehensive risk report for a parameter scan."""
    scan_id: str
    tool_name: str
    timestamp: float = field(default_factory=time.time)
    findings: List[RiskFinding] = field(default_factory=list)
    overall_risk_score: float = 0.0
    risk_level: str = "none"
    parameters_scanned: int = 0
    scan_duration_ms: float = 0.0

    def add_finding(self, finding: RiskFinding) -> None:
        self.findings.append(finding)

    def compute_overall_risk(self) -> float:
        if not self.findings:
            self.overall_risk_score = 0.0
            self.risk_level = "none"
            return 0.0
        severity_weights = {
            RiskSeverity.CRITICAL: 100.0,
            RiskSeverity.HIGH: 75.0,
            RiskSeverity.MEDIUM: 50.0,
            RiskSeverity.LOW: 25.0,
            RiskSeverity.INFO: 10.0,
        }
        weighted_sum = sum(
            severity_weights.get(f.severity, 10.0) * f.confidence
            for f in self.findings
        )
        max_possible = len(self.findings) * 100.0
        self.overall_risk_score = min(100.0, (weighted_sum / max_possible) * 100.0)
        if self.overall_risk_score >= 80:
            self.risk_level = "critical"
        elif self.overall_risk_score >= 60:
            self.risk_level = "high"
        elif self.overall_risk_score >= 40:
            self.risk_level = "medium"
        elif self.overall_risk_score >= 20:
            self.risk_level = "low"
        else:
            self.risk_level = "info"
        return self.overall_risk_score

    def get_findings_by_category(self, category: RiskCategory) -> List[RiskFinding]:
        return [f for f in self.findings if f.category == category]

    def get_findings_by_severity(self, severity: RiskSeverity) -> List[RiskFinding]:
        return [f for f in self.findings if f.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "findings": [f.to_dict() for f in self.findings],
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level,
            "parameters_scanned": self.parameters_scanned,
            "scan_duration_ms": self.scan_duration_ms,
            "finding_count": len(self.findings),
            "critical_count": len(self.get_findings_by_severity(RiskSeverity.CRITICAL)),
            "high_count": len(self.get_findings_by_severity(RiskSeverity.HIGH)),
        }


class CommandInjectionDetector:
    """Detects command injection patterns in parameters."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, RiskSeverity, str, str]] = [
            (
                "shell_metachar_semicolon",
                re.compile(r';\s*\w'),
                RiskSeverity.HIGH,
                "Semicolon followed by command (potential command chaining)",
                "Avoid using semicolons in parameters; validate input strictly",
            ),
            (
                "shell_metachar_pipe",
                re.compile(r'\|\s*\w'),
                RiskSeverity.HIGH,
                "Pipe to shell command detected",
                "Avoid pipe characters in parameters; use allowlists",
            ),
            (
                "shell_metachar_backtick",
                re.compile(r'`[^`]*`'),
                RiskSeverity.CRITICAL,
                "Backtick command substitution detected",
                "Never allow backtick characters in user input",
            ),
            (
                "shell_metachar_dollar_paren",
                re.compile(r'\$\([^)]*\)'),
                RiskSeverity.CRITICAL,
                "Dollar-paren command substitution detected",
                "Block $() patterns in user-facing parameters",
            ),
            (
                "shell_redirect_output",
                re.compile(r'>\s*/'),
                RiskSeverity.HIGH,
                "Output redirect to filesystem path detected",
                "Block output redirection characters in parameters",
            ),
            (
                "shell_redirect_input",
                re.compile(r'<\s*/'),
                RiskSeverity.HIGH,
                "Input redirect from filesystem detected",
                "Block input redirection characters in parameters",
            ),
            (
                "shell_newline_cmd",
                re.compile(r'\n\s*(ls|cat|rm|chmod|chown|wget|curl|bash|sh|python|perl|ruby|nc|ncat)\b'),
                RiskSeverity.CRITICAL,
                "Newline followed by shell command detected",
                "Strip newlines from parameters before processing",
            ),
            (
                "shell_ampersand_bg",
                re.compile(r'&\s*(ls|cat|rm|wget|curl|bash|sh)\b'),
                RiskSeverity.HIGH,
                "Ampersand background execution detected",
                "Block ampersand characters in parameters",
            ),
            (
                "shell_double_amp",
                re.compile(r'&&\s*\w'),
                RiskSeverity.HIGH,
                "Double ampersand command chaining detected",
                "Block && patterns in parameters",
            ),
            (
                "shell_double_pipe",
                re.compile(r'\|\|\s*\w'),
                RiskSeverity.MEDIUM,
                "Double pipe OR execution detected",
                "Validate and sanitize pipe characters",
            ),
            (
                "shell_paren_subshell",
                re.compile(r'\(\s*(ls|cat|rm|wget|curl|bash|sh)\b'),
                RiskSeverity.HIGH,
                "Subshell execution via parentheses detected",
                "Block parentheses used for subshell execution",
            ),
            (
                "shell_heredoc",
                re.compile(r'<<\s*\w+'),
                RiskSeverity.MEDIUM,
                "Heredoc syntax detected",
                "Block heredoc patterns in parameters",
            ),
            (
                "shell_env_var_exec",
                re.compile(r'\$\{[^}]*\}'),
                RiskSeverity.MEDIUM,
                "Environment variable expansion detected",
                "Validate environment variable usage in parameters",
            ),
        ]

    def scan(self, param_name: str, value: Any) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        if not isinstance(value, str):
            return findings
        for pattern_name, pattern, severity, description, remediation in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(RiskFinding(
                    finding_id=uuid.uuid4().hex[:12],
                    category=RiskCategory.COMMAND_INJECTION,
                    severity=severity,
                    pattern=pattern_name,
                    matched_text=match.group(0),
                    parameter_name=param_name,
                    description=description,
                    remediation=remediation,
                    confidence=0.9,
                    location={"start": match.start(), "end": match.end()},
                ))
        return findings


class PathTraversalDetector:
    """Detects path traversal attack patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, RiskSeverity, str, str]] = [
            (
                "dot_dot_slash",
                re.compile(r'\.\.[/\\]'),
                RiskSeverity.HIGH,
                "Directory traversal using ../ pattern",
                "Normalize paths and validate against allowed directories",
            ),
            (
                "dot_dot_encoded",
                re.compile(r'%2e%2e[/\\%]'),
                RiskSeverity.HIGH,
                "URL-encoded directory traversal",
                "Decode URL-encoded characters before path validation",
            ),
            (
                "dot_dot_double_encoded",
                re.compile(r'%252e%252e'),
                RiskSeverity.HIGH,
                "Double URL-encoded directory traversal",
                "Apply recursive URL decoding before validation",
            ),
            (
                "null_byte_in_path",
                re.compile(r'%00'),
                RiskSeverity.CRITICAL,
                "Null byte injection in path",
                "Reject any path containing null bytes",
            ),
            (
                "absolute_unix_path",
                re.compile(r'^/etc/|^/var/|^/tmp/|^/root/|^/home/'),
                RiskSeverity.MEDIUM,
                "Absolute path to sensitive Unix directory",
                "Restrict file access to designated directories",
            ),
            (
                "absolute_windows_path",
                re.compile(r'[A-Za-z]:\\(Windows|Program Files|Users)'),
                RiskSeverity.MEDIUM,
                "Absolute path to sensitive Windows directory",
                "Restrict file access to designated directories",
            ),
            (
                "path_unc",
                re.compile(r'\\\\[^\s]+'),
                RiskSeverity.HIGH,
                "UNC path detected (potential network access)",
                "Block UNC paths in parameters",
            ),
            (
                "path_with_pipe",
                re.compile(r'[a-zA-Z]:[\\/][^\s]*\|'),
                RiskSeverity.HIGH,
                "Path with pipe (potential command injection on Windows)",
                "Validate paths strictly; reject pipe characters",
            ),
            (
                "dot_dot_unicode",
                re.compile(r'\u2025|\u2024|\uff0e'),
                RiskSeverity.HIGH,
                "Unicode-based directory traversal",
                "Normalize Unicode before path validation",
            ),
        ]

    def scan(self, param_name: str, value: Any) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        if not isinstance(value, str):
            return findings
        for pattern_name, pattern, severity, description, remediation in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(RiskFinding(
                    finding_id=uuid.uuid4().hex[:12],
                    category=RiskCategory.PATH_TRAVERSAL,
                    severity=severity,
                    pattern=pattern_name,
                    matched_text=match.group(0),
                    parameter_name=param_name,
                    description=description,
                    remediation=remediation,
                    confidence=0.85,
                    location={"start": match.start(), "end": match.end()},
                ))
        return findings


class SQLInjectionDetector:
    """Detects SQL injection patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, RiskSeverity, str, str]] = [
            (
                "sql_union_select",
                re.compile(r'\bUNION\b\s+(?:ALL\s+)?SELECT\b', re.I),
                RiskSeverity.CRITICAL,
                "UNION SELECT injection detected",
                "Use parameterized queries exclusively",
            ),
            (
                "sql_or_true",
                re.compile(r"'\s*OR\s+'?\d*'\s*=\s*'?\d*", re.I),
                RiskSeverity.HIGH,
                "OR true injection pattern detected",
                "Use parameterized queries and input validation",
            ),
            (
                "sql_comment",
                re.compile(r'--\s*$|--\s+\w'),
                RiskSeverity.MEDIUM,
                "SQL comment detected (potential injection)",
                "Strip SQL comment sequences from input",
            ),
            (
                "sql_block_comment",
                re.compile(r'/\*.*?\*/'),
                RiskSeverity.MEDIUM,
                "SQL block comment detected",
                "Block SQL comment syntax in input parameters",
            ),
            (
                "sql_semicolon_multi",
                re.compile(r';\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC)\b', re.I),
                RiskSeverity.CRITICAL,
                "Multiple SQL statements via semicolon detected",
                "Use parameterized queries; never concatenate SQL",
            ),
            (
                "sql_tautology",
                re.compile(r"'\s*=\s*'"),
                RiskSeverity.HIGH,
                "SQL tautology (always true) detected",
                "Use parameterized queries",
            ),
            (
                "sql_waitfor_delay",
                re.compile(r'\bWAITFOR\s+DELAY\b', re.I),
                RiskSeverity.HIGH,
                "SQL time-based blind injection (WAITFOR DELAY)",
                "Use parameterized queries; monitor query timing",
            ),
            (
                "sql_benchmark",
                re.compile(r'\bBENCHMARK\s*\(', re.I),
                RiskSeverity.HIGH,
                "SQL benchmark injection (MySQL blind injection)",
                "Use parameterized queries; limit query execution time",
            ),
            (
                "sql_sleep",
                re.compile(r'\bSLEEP\s*\(', re.I),
                RiskSeverity.HIGH,
                "SQL sleep injection (time-based blind injection)",
                "Use parameterized queries; monitor query timing",
            ),
            (
                "sql_hex_encoding",
                re.compile(r'0x[0-9a-fA-F]{6,}'),
                RiskSeverity.MEDIUM,
                "Hex-encoded SQL payload detected",
                "Validate and sanitize hex-encoded input",
            ),
            (
                "sql_char_function",
                re.compile(r'\bCHAR\s*\(\s*\d+', re.I),
                RiskSeverity.MEDIUM,
                "SQL CHAR function detected (obfuscation technique)",
                "Block SQL function calls in input parameters",
            ),
            (
                "sql_drop_table",
                re.compile(r'\bDROP\s+TABLE\b', re.I),
                RiskSeverity.CRITICAL,
                "DROP TABLE statement detected",
                "Use parameterized queries; apply least-privilege database access",
            ),
        ]

    def scan(self, param_name: str, value: Any) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        if not isinstance(value, str):
            return findings
        for pattern_name, pattern, severity, description, remediation in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(RiskFinding(
                    finding_id=uuid.uuid4().hex[:12],
                    category=RiskCategory.SQL_INJECTION,
                    severity=severity,
                    pattern=pattern_name,
                    matched_text=match.group(0),
                    parameter_name=param_name,
                    description=description,
                    remediation=remediation,
                    confidence=0.85,
                    location={"start": match.start(), "end": match.end()},
                ))
        return findings


class XSSDetector:
    """Detects Cross-Site Scripting (XSS) patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, RiskSeverity, str, str]] = [
            (
                "xss_script_tag",
                re.compile(r'<script\b[^>]*>.*?</script>', re.I | re.S),
                RiskSeverity.CRITICAL,
                "Script tag injection detected",
                "HTML-encode all user input before rendering",
            ),
            (
                "xss_script_tag_open",
                re.compile(r'<script\b', re.I),
                RiskSeverity.CRITICAL,
                "Opening script tag detected",
                "HTML-encode all user input; use Content Security Policy",
            ),
            (
                "xss_event_handler",
                re.compile(r'\bon\w+\s*=\s*["\']?[^"\'>\s]', re.I),
                RiskSeverity.HIGH,
                "Event handler attribute detected (e.g., onclick, onerror)",
                "HTML-encode user input; whitelist allowed attributes",
            ),
            (
                "xss_javascript_uri",
                re.compile(r'javascript\s*:', re.I),
                RiskSeverity.CRITICAL,
                "JavaScript URI scheme detected",
                "Block javascript: URIs; validate URL schemes",
            ),
            (
                "xss_vbscript_uri",
                re.compile(r'vbscript\s*:', re.I),
                RiskSeverity.HIGH,
                "VBScript URI scheme detected",
                "Block vbscript: URIs in user input",
            ),
            (
                "xss_data_uri",
                re.compile(r'data\s*:\s*text/html', re.I),
                RiskSeverity.HIGH,
                "Data URI with HTML content detected",
                "Validate and restrict data URI usage",
            ),
            (
                "xss_svg_onload",
                re.compile(r'<svg\b[^>]*\bonload\b', re.I),
                RiskSeverity.HIGH,
                "SVG with onload event handler detected",
                "Sanitize SVG content; disallow event handlers",
            ),
            (
                "xss_img_onerror",
                re.compile(r'<img\b[^>]*\bonerror\b', re.I),
                RiskSeverity.HIGH,
                "IMG with onerror event handler detected",
                "HTML-encode user input; sanitize HTML",
            ),
            (
                "xss_body_onload",
                re.compile(r'<body\b[^>]*\bonload\b', re.I),
                RiskSeverity.HIGH,
                "BODY with onload event handler detected",
                "HTML-encode user input; sanitize HTML",
            ),
            (
                "xss_iframe_src",
                re.compile(r'<iframe\b[^>]*src\s*=', re.I),
                RiskSeverity.MEDIUM,
                "IFRAME with src attribute detected",
                "Whitelist allowed iframe sources; use sandbox attribute",
            ),
            (
                "xss_object_embed",
                re.compile(r'<(?:object|embed)\b', re.I),
                RiskSeverity.MEDIUM,
                "OBJECT or EMBED tag detected",
                "Block OBJECT and EMBED tags in user input",
            ),
            (
                "xss_expression",
                re.compile(r'expression\s*\(', re.I),
                RiskSeverity.MEDIUM,
                "CSS expression detected (IE XSS vector)",
                "Block CSS expressions; validate CSS input",
            ),
            (
                "xss_encoded_script",
                re.compile(r'(?:&#x?0*7[3c]|&#x?0*6[5e])\w*(?:&#x?0*7[3c]|&#x?0*6[5e])', re.I),
                RiskSeverity.HIGH,
                "HTML entity encoded script tag detected",
                "Decode HTML entities before validation",
            ),
        ]

    def scan(self, param_name: str, value: Any) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        if not isinstance(value, str):
            return findings
        for pattern_name, pattern, severity, description, remediation in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(RiskFinding(
                    finding_id=uuid.uuid4().hex[:12],
                    category=RiskCategory.XSS,
                    severity=severity,
                    pattern=pattern_name,
                    matched_text=match.group(0),
                    parameter_name=param_name,
                    description=description,
                    remediation=remediation,
                    confidence=0.85,
                    location={"start": match.start(), "end": match.end()},
                ))
        return findings


class SSRFDetector:
    """Detects Server-Side Request Forgery (SSRF) patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, RiskSeverity, str, str]] = [
            (
                "ssrf_private_ip",
                re.compile(r'https?://(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})'),
                RiskSeverity.HIGH,
                "Request to private IP address detected",
                "Block requests to private IP ranges; use allowlist for outbound URLs",
            ),
            (
                "ssrf_localhost",
                re.compile(r'https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])', re.I),
                RiskSeverity.CRITICAL,
                "Request to localhost detected",
                "Block all requests to localhost and loopback addresses",
            ),
            (
                "ssrf_metadata",
                re.compile(r'https?://169\.254\.169\.254'),
                RiskSeverity.CRITICAL,
                "Request to cloud metadata endpoint (169.254.169.254)",
                "Block access to cloud metadata endpoints",
            ),
            (
                "ssrf_dns_rebinding",
                re.compile(r'https?://[a-z0-9]+\.local(?:\.\w+)?', re.I),
                RiskSeverity.MEDIUM,
                "Request to .local domain (potential DNS rebinding)",
                "Block requests to .local domains; use DNS pinning",
            ),
            (
                "ssrf_file_protocol",
                re.compile(r'file://'),
                RiskSeverity.HIGH,
                "file:// protocol detected (local file access)",
                "Block file:// protocol in URL parameters",
            ),
            (
                "ssrf_gopher_protocol",
                re.compile(r'gopher://'),
                RiskSeverity.HIGH,
                "gopher:// protocol detected (SSRF tunneling)",
                "Block gopher:// and other non-HTTP protocols",
            ),
            (
                "ssrf_dict_protocol",
                re.compile(r'dict://'),
                RiskSeverity.MEDIUM,
                "dict:// protocol detected",
                "Block dict:// protocol in URL parameters",
            ),
            (
                "ssrf_ftp_protocol",
                re.compile(r'ftp://'),
                RiskSeverity.MEDIUM,
                "ftp:// protocol detected",
                "Restrict FTP access; validate allowed protocols",
            ),
            (
                "ssrf_url_redirect",
                re.compile(r'(?:redirect|url|next|return|returnTo|destination)=\s*(https?|ftp)://', re.I),
                RiskSeverity.MEDIUM,
                "Open redirect / URL parameter detected (potential SSRF vector)",
                "Validate redirect URLs against allowlist",
            ),
            (
                "ssrf_ip_obfuscation",
                re.compile(r'https?://0x[0-9a-fA-F]+|https?://\d{10,}'),
                RiskSeverity.HIGH,
                "Obfuscated IP address detected (decimal/hex)",
                "Resolve and validate all IP addresses before making requests",
            ),
        ]

    def scan(self, param_name: str, value: Any) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        if not isinstance(value, str):
            return findings
        for pattern_name, pattern, severity, description, remediation in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(RiskFinding(
                    finding_id=uuid.uuid4().hex[:12],
                    category=RiskCategory.SSRF,
                    severity=severity,
                    pattern=pattern_name,
                    matched_text=match.group(0),
                    parameter_name=param_name,
                    description=description,
                    remediation=remediation,
                    confidence=0.8,
                    location={"start": match.start(), "end": match.end()},
                ))
        return findings


class DeserializationDetector:
    """Detects insecure deserialization patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, RiskSeverity, str, str]] = [
            (
                "deser_python_pickle",
                re.compile(r"(?:c__builtin__|c__main__|cpickle|cos\n|cposix\n|c__import__)"),
                RiskSeverity.CRITICAL,
                "Python pickle deserialization payload detected",
                "Never deserialize untrusted pickle data; use safe formats like JSON",
            ),
            (
                "deser_python_reduce",
                re.compile(r'\(reduce\b|__reduce__|__reduce_ex__'),
                RiskSeverity.CRITICAL,
                "Python __reduce__ exploitation pattern detected",
                "Never deserialize untrusted data with pickle",
            ),
            (
                "deser_java_serial",
                re.compile(r'\xac\xed\x00\x05|rO0AB|aced0005'),
                RiskSeverity.HIGH,
                "Java serialization magic bytes detected",
                "Never deserialize untrusted Java serialized data",
            ),
            (
                "deser_php_serialize",
                re.compile(r'O:\d+:"'),
                RiskSeverity.HIGH,
                "PHP serialized object detected",
                "Validate and sanitize PHP serialized data",
            ),
            (
                "deser_yaml_unsafe",
                re.compile(r'!!python/(?:object|module|name|function)'),
                RiskSeverity.CRITICAL,
                "Unsafe YAML deserialization tag detected",
                "Use yaml.safe_load() instead of yaml.load()",
            ),
            (
                "deser_xml_external",
                re.compile(r'<!DOCTYPE[^>]*\[.*\s+SYSTEM\s+', re.I | re.S),
                RiskSeverity.HIGH,
                "XML External Entity (XXE) declaration detected",
                "Disable external entity processing in XML parsers",
            ),
            (
                "deser_xml_entity",
                re.compile(r'<!ENTITY\s+\S+\s+SYSTEM\s+["\']', re.I),
                RiskSeverity.HIGH,
                "XML entity declaration with SYSTEM identifier",
                "Disable DTD processing in XML parsers",
            ),
            (
                "deser_base64_pickle",
                re.compile(r'(?:gASV|gASL|gAN9)'),
                RiskSeverity.HIGH,
                "Base64-encoded pickle data detected",
                "Never decode and deserialize untrusted base64 pickle data",
            ),
            (
                "deser_nodejs_prototype",
                re.compile(r'__proto__|constructor\[["\']prototype["\']\]'),
                RiskSeverity.HIGH,
                "Node.js prototype pollution pattern detected",
                "Validate object keys; block __proto__ and constructor.prototype",
            ),
            (
                "deser_dotnet_type",
                re.compile(r'TypeName|_TypeHandle|SerializationInfo'),
                RiskSeverity.MEDIUM,
                ".NET serialization artifact detected",
                "Validate .NET serialization data before deserializing",
            ),
        ]

    def scan(self, param_name: str, value: Any) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if not isinstance(value, str):
            return findings
        for pattern_name, pattern, severity, description, remediation in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(RiskFinding(
                    finding_id=uuid.uuid4().hex[:12],
                    category=RiskCategory.DESERIALIZATION,
                    severity=severity,
                    pattern=pattern_name,
                    matched_text=match.group(0)[:100],
                    parameter_name=param_name,
                    description=description,
                    remediation=remediation,
                    confidence=0.8,
                    location={"start": match.start(), "end": match.end()},
                ))
        return findings


class RiskScanner:
    """Main risk scanner that orchestrates all individual detectors."""

    def __init__(
        self,
        enable_command_injection: bool = True,
        enable_path_traversal: bool = True,
        enable_sql_injection: bool = True,
        enable_xss: bool = True,
        enable_ssrf: bool = True,
        enable_deserialization: bool = True,
    ) -> None:
        self.command_detector: Optional[CommandInjectionDetector] = (
            CommandInjectionDetector() if enable_command_injection else None
        )
        self.path_detector: Optional[PathTraversalDetector] = (
            PathTraversalDetector() if enable_path_traversal else None
        )
        self.sql_detector: Optional[SQLInjectionDetector] = (
            SQLInjectionDetector() if enable_sql_injection else None
        )
        self.xss_detector: Optional[XSSDetector] = (
            XSSDetector() if enable_xss else None
        )
        self.ssrf_detector: Optional[SSRFDetector] = (
            SSRFDetector() if enable_ssrf else None
        )
        self.deser_detector: Optional[DeserializationDetector] = (
            DeserializationDetector() if enable_deserialization else None
        )
        self._scan_history: List[RiskReport] = []
        self._max_history: int = 1000

    def scan(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        scan_id: Optional[str] = None,
    ) -> RiskReport:
        start_time = time.time()
        if scan_id is None:
            scan_id = uuid.uuid4().hex[:12]
        report = RiskReport(
            scan_id=scan_id,
            tool_name=tool_name,
            parameters_scanned=len(parameters),
        )
        for param_name, value in parameters.items():
            self._scan_parameter(param_name, value, report)
        report.scan_duration_ms = (time.time() - start_time) * 1000
        report.compute_overall_risk()
        self._scan_history.append(report)
        if len(self._scan_history) > self._max_history:
            self._scan_history = self._scan_history[-self._max_history:]
        return report

    def scan_single(self, param_name: str, value: Any) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        detectors = [
            (self.command_detector, RiskCategory.COMMAND_INJECTION),
            (self.path_detector, RiskCategory.PATH_TRAVERSAL),
            (self.sql_detector, RiskCategory.SQL_INJECTION),
            (self.xss_detector, RiskCategory.XSS),
            (self.ssrf_detector, RiskCategory.SSRF),
            (self.deser_detector, RiskCategory.DESERIALIZATION),
        ]
        for detector, category in detectors:
            if detector is not None:
                findings.extend(detector.scan(param_name, value))
        return findings

    def _scan_parameter(
        self, param_name: str, value: Any, report: RiskReport
    ) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                self._scan_parameter(f"{param_name}.{k}", v, report)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                self._scan_parameter(f"{param_name}[{i}]", item, report)
        else:
            findings = self.scan_single(param_name, value)
            for finding in findings:
                report.add_finding(finding)

    def get_scan_history(
        self, tool_name: Optional[str] = None, limit: int = 50
    ) -> List[RiskReport]:
        reports = self._scan_history
        if tool_name:
            reports = [r for r in reports if r.tool_name == tool_name]
        return reports[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        if not self._scan_history:
            return {"total_scans": 0}
        total_findings = sum(len(r.findings) for r in self._scan_history)
        category_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        for report in self._scan_history:
            for finding in report.findings:
                cat = finding.category.value
                sev = finding.severity.value
                category_counts[cat] = category_counts.get(cat, 0) + 1
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
        return {
            "total_scans": len(self._scan_history),
            "total_findings": total_findings,
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "avg_risk_score": sum(r.overall_risk_score for r in self._scan_history) / len(self._scan_history),
        }
