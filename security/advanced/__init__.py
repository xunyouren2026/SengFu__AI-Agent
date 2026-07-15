"""
AGI Unified Framework - Advanced Security Module
================================================

Provides advanced security capabilities including:
- Model watermarking and fingerprinting
- Differential privacy mechanisms
- Zero-knowledge proofs (simulated)
- Audit trail with Merkle tree integrity
- Threat detection for prompt injection and jailbreak
"""

from .model_watermark import (
    ModelWatermark,
    ContentSignature,
    ModelFingerprint,
    WatermarkConfig,
)
from .differential_privacy import (
    DifferentialPrivacy,
    DPConfig,
    PrivacyAccountant,
    RDPAccountant,
    MomentsAccountant,
)
from .zero_knowledge import (
    ZeroKnowledgeProof,
    ZKProof,
    ZKConfig,
    SigmaProtocol,
    ZKPSerializer,
    BatchProof,
)
from .audit_trail import (
    AuditTrail,
    MerkleTree,
    TamperDetector,
    AuditEvent,
    AuditQuery,
    ComplianceReport,
    Severity,
    ActionCategory,
)
from .threat_detection import (
    ThreatDetector,
    InjectionDetector,
    JailbreakDetector,
    AnomalyDetector,
    ThreatLevel,
    ThreatReport,
)

__all__ = [
    # Model watermarking
    "ModelWatermark",
    "ContentSignature",
    "ModelFingerprint",
    "WatermarkConfig",
    # Differential privacy
    "DifferentialPrivacy",
    "DPConfig",
    "PrivacyAccountant",
    "RDPAccountant",
    "MomentsAccountant",
    # Zero-knowledge proofs
    "ZeroKnowledgeProof",
    "ZKProof",
    "ZKConfig",
    "SigmaProtocol",
    "ZKPSerializer",
    "BatchProof",
    # Audit trail
    "AuditTrail",
    "MerkleTree",
    "TamperDetector",
    "AuditEvent",
    "AuditQuery",
    "ComplianceReport",
    "Severity",
    "ActionCategory",
    # Threat detection
    "ThreatDetector",
    "InjectionDetector",
    "JailbreakDetector",
    "AnomalyDetector",
    "ThreatLevel",
    "ThreatReport",
]
