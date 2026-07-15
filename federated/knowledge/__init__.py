"""
联邦学习知识模块
"""
from .distillation import (
    DistillationType,
    KnowledgeBuffer,
    FederatedDistillation,
    FeatureDistillation,
    AttentionDistillation
)
from .ensemble import (
    EnsembleMethod,
    ModelPrediction,
    KnowledgeEnsemble,
    AdaptiveEnsemble
)
from .conflict_resolver import (
    ConflictType,
    Conflict,
    ConflictDetector,
    ConflictResolver,
    LabelConflictResolver
)

__all__ = [
    # distillation
    'DistillationType',
    'KnowledgeBuffer',
    'FederatedDistillation',
    'FeatureDistillation',
    'AttentionDistillation',
    # ensemble
    'EnsembleMethod',
    'ModelPrediction',
    'KnowledgeEnsemble',
    'AdaptiveEnsemble',
    # conflict_resolver
    'ConflictType',
    'Conflict',
    'ConflictDetector',
    'ConflictResolver',
    'LabelConflictResolver'
]
