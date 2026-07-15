"""
AISEC Alignment Module
======================
Agent reasoning chain auditing, semantic intent verification,
goal deviation detection, and behavior-intent consistency scoring.
"""

from .agent_auditor import (
    AgentAuditor,
    ReasoningInspector,
    GoalTracker,
    PlanConsistencyChecker,
    ActionLegitimacyVerifier,
    AnomalyScorer,
    AuditReport,
    AuditTrail,
)
from .semantic_intent import (
    IntentVerifier,
    IntentEmbedding,
    ConsistencyScorer,
    DriftDetector,
    IntentDecomposer,
    SemanticSimilarity,
    IntentProfile,
)

__all__ = [
    "AgentAuditor",
    "ReasoningInspector",
    "GoalTracker",
    "PlanConsistencyChecker",
    "ActionLegitimacyVerifier",
    "AnomalyScorer",
    "AuditReport",
    "AuditTrail",
    "IntentVerifier",
    "IntentEmbedding",
    "ConsistencyScorer",
    "DriftDetector",
    "IntentDecomposer",
    "SemanticSimilarity",
    "IntentProfile",
]
