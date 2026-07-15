"""
ClawHub 插件市场核心模块

提供插件注册表、安全验证、评分系统、版本控制、全文搜索和依赖解析功能。
"""

from .registry import (
    PluginRegistry,
    PluginMetadata,
    PluginVersion,
    DependencyGraph,
    RegistrySearchResult,
)
from .verifier import (
    SecurityVerifier,
    StaticAnalyzer,
    MalwareDetector,
    SandboxValidator,
    SignatureVerifier,
    VerificationResult,
    SecurityReport,
)
from .ratings import (
    RatingSystem,
    UserRating,
    Review,
    ReputationAlgorithm,
    RankingAlgorithm,
    RatingSummary,
)
from .versioning import (
    SemVer,
    VersionConstraint,
    VersionCompatibility,
    UpgradePath,
    RollbackManager,
    VersionResolver,
    VersionManager,
)
from .indexer import (
    FullTextIndexer,
    InvertedIndex,
    FuzzySearcher,
    TagClassifier,
    RecommendationEngine,
    SearchQuery,
    SearchResult,
)
from .dependency import (
    DependencyResolver,
    DependencyGraphBuilder,
    ConflictDetector,
    AutoResolver,
    CycleDetector,
    DependencyNode,
    ResolutionResult,
)

__all__ = [
    # Registry
    "PluginRegistry",
    "PluginMetadata",
    "PluginVersion",
    "DependencyGraph",
    "RegistrySearchResult",
    # Verifier
    "SecurityVerifier",
    "StaticAnalyzer",
    "MalwareDetector",
    "SandboxValidator",
    "SignatureVerifier",
    "VerificationResult",
    "SecurityReport",
    # Ratings
    "RatingSystem",
    "UserRating",
    "Review",
    "ReputationAlgorithm",
    "RankingAlgorithm",
    "RatingSummary",
    # Versioning
    "SemVer",
    "VersionConstraint",
    "VersionCompatibility",
    "UpgradePath",
    "RollbackManager",
    "VersionResolver",
    "VersionManager",
    # Indexer
    "FullTextIndexer",
    "InvertedIndex",
    "FuzzySearcher",
    "TagClassifier",
    "RecommendationEngine",
    "SearchQuery",
    "SearchResult",
    # Dependency
    "DependencyResolver",
    "DependencyGraphBuilder",
    "ConflictDetector",
    "AutoResolver",
    "CycleDetector",
    "DependencyNode",
    "ResolutionResult",
]

__version__ = "1.0.0"
__author__ = "ClawHub Team"
