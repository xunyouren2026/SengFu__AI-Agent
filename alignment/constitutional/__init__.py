"""
Constitutional AI - 宪章AI模块
包含宪章加载、审查引擎、自我批评和内容修订功能。
"""

from .engine import (
    ConstitutionalRule,
    Constitution,
    RuleViolation,
    ConstitutionalReviewResult,
    ConstitutionalEngine,
    ConstitutionalAuditLog,
    ConstitutionalEvaluator,
)
from .loader import (
    ConstitutionLoader,
    BuiltinConstitutions,
)
from .critic import (
    ConstitutionCritic,
    RuleEvaluator,
    ViolationScorer,
    SuggestionGenerator,
    MultiPerspectiveCritic,
    CritiqueChain,
    CritiqueReport,
    Severity,
    CritiquePerspective,
    Violation,
    ImprovementSuggestion,
    PerspectiveCritique,
)
from .reviser import (
    ContentReviser,
    ViolationReviser,
    IterativeRefiner,
    StylePreserver,
    MinimalEditStrategy,
    RevisionScorer,
    RevisionHistory,
    RevisionReport,
    RevisionStrategy,
)

__all__ = [
    "ConstitutionalRule",
    "Constitution",
    "RuleViolation",
    "ConstitutionalReviewResult",
    "ConstitutionalEngine",
    "ConstitutionalAuditLog",
    "ConstitutionalEvaluator",
    "ConstitutionLoader",
    "BuiltinConstitutions",
    "ConstitutionCritic",
    "RuleEvaluator",
    "ViolationScorer",
    "SuggestionGenerator",
    "MultiPerspectiveCritic",
    "CritiqueChain",
    "CritiqueReport",
    "Severity",
    "CritiquePerspective",
    "Violation",
    "ImprovementSuggestion",
    "PerspectiveCritique",
    "ContentReviser",
    "ViolationReviser",
    "IterativeRefiner",
    "StylePreserver",
    "MinimalEditStrategy",
    "RevisionScorer",
    "RevisionHistory",
    "RevisionReport",
    "RevisionStrategy",
]
