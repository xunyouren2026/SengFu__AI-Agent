"""
AISec Audit Module

提供全面的安全审计功能，包括操作审计、访问日志、告警引擎、
合规报告和审计分析。

Usage:
    from agi_unified_framework.aisec.audit import (
        OperationAuditor,
        AccessLogger,
        AuditAlertEngine,
        ComplianceReporter,
        AuditAnalyzer
    )
"""

# Operation Audit
from .operation_audit import (
    OperationAuditor,
    OperationRecord,
    AuditLogStore,
    AuditLogIndexer,
    OperationType,
    OperationResult,
)

# Access Log
from .access_log import (
    AccessLogger,
    AccessLogEntry,
    AccessLogAggregator,
    AccessPattern,
    SessionTracker,
    AccessType,
    AccessDecision,
)

# Alert Engine
from .alert_engine import (
    AuditAlertEngine,
    AlertRule,
    AlertCondition,
    Alert,
    AlertManager,
    NotificationChannel,
    AlertSeverity,
    AlertStatus,
)

# Compliance Report
from .compliance_report import (
    ComplianceReporter,
    ComplianceFramework,
    ComplianceControl,
    ComplianceEvidence,
    ComplianceGap,
    ComplianceReport,
)

# Analyzer
from .analyzer import (
    AuditAnalyzer,
    BehaviorProfile,
    RiskScorer,
    ThreatIndicator,
)

# Trail Logger
from .trail_logger import (
    TrailLogger,
    AuditTrail,
    CausalChain,
    CausalLink,
    SessionReconstructor,
    Session,
    TimelineBuilder,
    TimelineEvent,
    EvidenceIntegrity,
    ComplianceExporter,
    TrailQuery,
    TrailEntry,
    EntrySeverity,
    EntryCategory,
    ComplianceFormat,
)

__version__ = "1.0.0"

__all__ = [
    # Operation Audit
    "OperationAuditor",
    "OperationRecord",
    "AuditLogStore",
    "AuditLogIndexer",
    "OperationType",
    "OperationResult",
    
    # Access Log
    "AccessLogger",
    "AccessLogEntry",
    "AccessLogAggregator",
    "AccessPattern",
    "SessionTracker",
    "AccessType",
    "AccessDecision",
    
    # Alert Engine
    "AuditAlertEngine",
    "AlertRule",
    "AlertCondition",
    "Alert",
    "AlertManager",
    "NotificationChannel",
    "AlertSeverity",
    "AlertStatus",
    
    # Compliance Report
    "ComplianceReporter",
    "ComplianceFramework",
    "ComplianceControl",
    "ComplianceEvidence",
    "ComplianceGap",
    "ComplianceReport",
    
    # Analyzer
    "AuditAnalyzer",
    "BehaviorProfile",
    "RiskScorer",
    "ThreatIndicator",

    # Trail Logger
    "TrailLogger",
    "AuditTrail",
    "CausalChain",
    "CausalLink",
    "SessionReconstructor",
    "Session",
    "TimelineBuilder",
    "TimelineEvent",
    "EvidenceIntegrity",
    "ComplianceExporter",
    "TrailQuery",
    "TrailEntry",
    "EntrySeverity",
    "EntryCategory",
    "ComplianceFormat",
]
