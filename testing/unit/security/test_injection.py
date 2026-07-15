"""
TestInjection - 安全单元测试：注入攻击防护

模块路径: testing/unit/security/test_injection.py
"""
import re
import html
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


class MockInputSanitizer:
    """模拟输入净化器"""

    SQL_KEYWORDS = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "UNION",
                    "ALTER", "CREATE", "EXEC", "EXECUTE", "--", ";"]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<img[^>]+onerror",
        r"<svg[^>]*>",
        r"eval\(",
        r"document\.",
        r"window\.",
    ]

    CMD_PATTERNS = [
        r";\s*rm\s+-rf",
        r"\|\s*sh",
        r"`.*`",
        r"\$\(.*\)",
        r"&&\s*rm",
        r"\|\s*cat",
        r">\s*/dev/",
        r"\bnc\b",
        r"\bwget\b",
        r"\bcurl\b",
    ]

    def sanitize_sql(self, input_str: str) -> str:
        sanitized = input_str
        for kw in self.SQL_KEYWORDS:
            sanitized = re.sub(kw, "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"'", "''", sanitized)
        sanitized = re.sub(r'""', '\"\"', sanitized)
        return sanitized.strip()

    def detect_sql_injection(self, input_str: str) -> bool:
        upper = input_str.upper()
        for kw in self.SQL_KEYWORDS:
            if kw in upper:
                return True
        if re.search(r"'\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d*'?", upper):
            return True
        return False

    def sanitize_xss(self, input_str: str) -> str:
        sanitized = html.escape(input_str)
        for pattern in self.XSS_PATTERNS:
            sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def detect_xss(self, input_str: str) -> bool:
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                return True
        return False

    def sanitize_command(self, input_str: str) -> str:
        sanitized = re.sub(r"[;&|`$]", "", input_str)
        for pattern in self.CMD_PATTERNS:
            sanitized = re.sub(pattern, "[BLOCKED]", sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

    def detect_command_injection(self, input_str: str) -> bool:
        for pattern in self.CMD_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                return True
        if re.search(r"[;&|`$]", input_str):
            return True
        return False

    def sanitize_all(self, input_str: str) -> str:
        result = self.sanitize_sql(input_str)
        result = self.sanitize_xss(result)
        result = self.sanitize_command(result)
        return result

    def validate_input(self, input_str: str, max_length: int = 1000,
                       allowed_chars: str = None) -> Dict[str, Any]:
        issues = []
        if len(input_str) > max_length:
            issues.append(f"Input exceeds max length {max_length}")
        if allowed_chars and not all(c in allowed_chars for c in input_str):
            issues.append("Input contains disallowed characters")
        if self.detect_sql_injection(input_str):
            issues.append("Potential SQL injection detected")
        if self.detect_xss(input_str):
            issues.append("Potential XSS detected")
        if self.detect_command_injection(input_str):
            issues.append("Potential command injection detected")
        return {"valid": len(issues) == 0, "issues": issues}


class TestSQLInjection:
    """SQL注入检测与净化测试"""

    def setup_method(self):
        self.sanitizer = MockInputSanitizer()

    def test_detect_basic_sql_injection(self):
        assert self.sanitizer.detect_sql_injection("SELECT * FROM users") is True
        assert self.sanitizer.detect_sql_injection("DROP TABLE users") is True

    def test_detect_union_injection(self):
        assert self.sanitizer.detect_sql_injection("1 UNION SELECT * FROM passwords") is True

    def test_detect_comment_injection(self):
        assert self.sanitizer.detect_sql_injection("admin'--") is True

    def test_detect_or_true_injection(self):
        assert self.sanitizer.detect_sql_injection("' OR '1'='1") is True

    def test_safe_input_not_detected(self):
        assert self.sanitizer.detect_sql_injection("Hello World") is False
        assert self.sanitizer.detect_sql_injection("user@example.com") is False

    def test_sanitize_removes_keywords(self):
        result = self.sanitizer.sanitize_sql("SELECT * FROM users")
        assert "SELECT" not in result.upper()

    def test_sanitize_escapes_quotes(self):
        result = self.sanitizer.sanitize_sql("it's a test")
        assert "\''" in result

    def test_sanitize_preserves_safe_text(self):
        result = self.sanitizer.sanitize_sql("Hello, my name is John")
        assert "Hello" in result
        assert "John" in result


class TestXSSDetection:
    """XSS检测与净化测试"""

    def setup_method(self):
        self.sanitizer = MockInputSanitizer()

    def test_detect_script_tag(self):
        assert self.sanitizer.detect_xss("<script>alert(1)</script>") is True

    def test_detect_javascript_protocol(self):
        assert self.sanitizer.detect_xss("javascript:alert(1)") is True

    def test_detect_event_handler(self):
        assert self.sanitizer.detect_xss("<img onerror=alert(1)>") is True

    def test_detect_iframe(self):
        assert self.sanitizer.detect_xss("<iframe src=evil.com>") is True

    def test_detect_svg(self):
        assert self.sanitizer.detect_xss("<svg onload=alert(1)>") is True

    def test_safe_html_not_detected(self):
        assert self.sanitizer.detect_xss("<p>Hello World</p>") is False
        assert self.sanitizer.detect_xss("Plain text") is False

    def test_sanitize_escapes_html(self):
        result = self.sanitizer.sanitize_xss("<b>bold</b>")
        assert "&lt;" in result
        assert "&gt;" in result

    def test_sanitize_removes_script(self):
        result = self.sanitizer.sanitize_xss("<script>alert(1)</script>")
        assert "<script" not in result.lower()

    def test_sanitize_preserves_text(self):
        result = self.sanitizer.sanitize_xss("Hello World 123")
        assert result == "Hello World 123"


class TestCommandInjection:
    """命令注入检测与净化测试"""

    def setup_method(self):
        self.sanitizer = MockInputSanitizer()

    def test_detect_pipe(self):
        assert self.sanitizer.detect_command_injection("ls | cat /etc/passwd") is True

    def test_detect_semicolon(self):
        assert self.sanitizer.detect_command_injection("; rm -rf /") is True

    def test_detect_backtick(self):
        assert self.sanitizer.detect_command_injection("`rm -rf /`") is True

    def test_detect_subshell(self):
        assert self.sanitizer.detect_command_injection("$(cat /etc/passwd)") is True

    def test_detect_rm_rf(self):
        assert self.sanitizer.detect_command_injection("file; rm -rf /") is True

    def test_safe_command_not_detected(self):
        assert self.sanitizer.detect_command_injection("hello.txt") is False
        assert self.sanitizer.detect_command_injection("document.pdf") is False

    def test_sanitize_removes_special_chars(self):
        result = self.sanitizer.sanitize_command("file; rm -rf /")
        assert ";" not in result

    def test_sanitize_blocks_dangerous_commands(self):
        result = self.sanitizer.sanitize_command("file && rm -rf /")
        assert "[BLOCKED]" in result


class TestInputValidation:
    """综合输入验证测试"""

    def setup_method(self):
        self.sanitizer = MockInputSanitizer()

    def test_validate_clean_input(self):
        result = self.sanitizer.validate_input("Hello World")
        assert result["valid"] is True
        assert result["issues"] == []

    def test_validate_long_input(self):
        result = self.sanitizer.validate_input("A" * 1001, max_length=1000)
        assert result["valid"] is False
        assert any("max length" in i for i in result["issues"])

    def test_validate_sql_injection(self):
        result = self.sanitizer.validate_input("' OR 1=1 --")
        assert result["valid"] is False
        assert any("SQL" in i for i in result["issues"])

    def test_validate_xss(self):
        result = self.sanitizer.validate_input("<script>alert(1)</script>")
        assert result["valid"] is False
        assert any("XSS" in i for i in result["issues"])

    def test_validate_command_injection(self):
        result = self.sanitizer.validate_input("; rm -rf /")
        assert result["valid"] is False
        assert any("command" in i for i in result["issues"])

    def test_sanitize_all_layers(self):
        malicious = "<script>' OR 1=1; rm -rf /</script>"
        result = self.sanitizer.sanitize_all(malicious)
        assert "<script" not in result.lower()
        assert "OR" not in result.upper()

    def test_validate_allowed_chars(self):
        result = self.sanitizer.validate_input("abc123", allowed_chars="abc123")
        assert result["valid"] is True
        result2 = self.sanitizer.validate_input("abc@123", allowed_chars="abc123")
        assert result2["valid"] is False
