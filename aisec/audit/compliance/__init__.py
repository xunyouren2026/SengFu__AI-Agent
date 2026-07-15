"""
AISEC Audit Compliance Module
==============================
GDPR and SOC2 compliance checking, reporting, and evidence collection.
"""

from .gdpr import (
    GDPRChecker,
    DataMinimizationChecker,
    ConsentTracker,
    ErasureHandler,
    PortabilityHandler,
    RetentionPolicy,
    BreachNotifier,
    ComplianceReport,
)
from .soc2 import (
    SOC2Reporter,
    ControlMapper,
    EvidenceCollector,
    TrustServiceCriteria,
    RiskAssessor,
    ComplianceScore,
)

__all__ = [
    "GDPRChecker",
    "DataMinimizationChecker",
    "ConsentTracker",
    "ErasureHandler",
    "PortabilityHandler",
    "RetentionPolicy",
    "BreachNotifier",
    "ComplianceReport",
    "SOC2Reporter",
    "ControlMapper",
    "EvidenceCollector",
    "TrustServiceCriteria",
    "RiskAssessor",
    "ComplianceScore",
]
