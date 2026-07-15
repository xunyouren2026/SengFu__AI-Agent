"""
DLP (Data Loss Prevention) Module - 数据防泄漏模块

提供完整的数据防泄漏能力，包括：
- PII扫描与识别
- 数据分类与标记
- 数据脱敏与加密
- 数据血缘追踪
- 实时监控与告警
- 内容检测与外泄防护
"""

__version__ = "1.0.0"
__author__ = "AGI Unified Framework Team"

# PII扫描器
from .pii_scanner import (
    PIIScanner,
    PIIMatch,
    PIIPattern,
    Severity,
    PIICategory,
)

# 数据分类器
from .data_classifier import (
    DataClassifier,
    ClassificationRule,
    ClassificationResult,
    DataLabel,
    LabelManager,
    ClassificationLevel,
)

# 数据脱敏器
from .sanitizer import (
    DataSanitizer,
    MaskingStrategy,
    SanitizationRule,
    SanitizationPipeline,
    ReversibleSanitizer,
    sanitizer_for_type,
)

# 数据血缘追踪
from .lineage_tracker import (
    DataLineageTracker,
    LineageNode,
    LineageEdge,
    LineageGraph,
    DataOrigin,
    TransformationRecord,
    AccessRecord,
)

# DLP监控器
from .monitor import (
    DLPMonitor,
    DLPPolicy,
    PolicyEngine,
    DLPAlert,
    DLPReport,
)

# 内容检测器
from .detector import (
    ContentDetector,
    ExfiltrationDetector,
    AnomalyDetector,
)

__all__ = [
    # PII扫描器
    "PIIScanner",
    "PIIMatch",
    "PIIPattern",
    "Severity",
    "PIICategory",
    # 数据分类器
    "DataClassifier",
    "ClassificationRule",
    "ClassificationResult",
    "DataLabel",
    "LabelManager",
    "ClassificationLevel",
    # 数据脱敏器
    "DataSanitizer",
    "MaskingStrategy",
    "SanitizationRule",
    "SanitizationPipeline",
    "ReversibleSanitizer",
    "sanitizer_for_type",
    # 数据血缘追踪
    "DataLineageTracker",
    "LineageNode",
    "LineageEdge",
    "LineageGraph",
    "DataOrigin",
    "TransformationRecord",
    "AccessRecord",
    # DLP监控器
    "DLPMonitor",
    "DLPPolicy",
    "PolicyEngine",
    "DLPAlert",
    "DLPReport",
    # 内容检测器
    "ContentDetector",
    "ExfiltrationDetector",
    "AnomalyDetector",
]
