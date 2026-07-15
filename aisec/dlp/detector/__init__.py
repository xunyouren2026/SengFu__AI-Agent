"""
DLP Detector Subpackage - DLP 检测器子包

整合 PII 扫描器、密钥扫描器和内容检测器，提供统一的数据泄露检测接口。
"""

# PII 扫描器 - 从父级 dlp 包导入
from ..pii_scanner import (
    PIIScanner,
    PIIMatch,
    PIIPattern,
    Severity,
    PIICategory,
)

# 密钥扫描器 - 从父级 dlp 包导入
from ..secrets_scanner import (
    SecretsScanner,
    SecretType,
    SecretSeverity,
    SecretMatch,
    ScanResult,
)

# 内容检测器 - 从本地 base 模块导入
from .base import (
    DetectionType,
    RiskLevel,
    DetectionResult,
    AccessPattern,
    ContentDetector,
    ExfiltrationDetector,
    AnomalyDetector,
)

__all__ = [
    "PIIScanner", "PIIMatch", "PIIPattern", "Severity", "PIICategory",
    "SecretsScanner", "SecretType", "SecretSeverity", "SecretMatch", "ScanResult",
    "DetectionType", "RiskLevel", "DetectionResult", "AccessPattern",
    "ContentDetector", "ExfiltrationDetector", "AnomalyDetector",
]
