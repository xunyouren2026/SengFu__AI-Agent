"""
联邦学习迁移模块
"""
from .domain_adapter import (
    AdaptationMethod,
    DomainInfo,
    DomainAdapter,
    MultiDomainAdapter
)
from .meta_learner import (
    MetaLearningMethod,
    Task,
    MetaLearner,
    FederatedMetaLearner
)
from .feature_alignment import (
    AlignmentMethod,
    FeatureStatistics,
    FeatureAligner,
    MultiSourceAligner
)

__all__ = [
    # domain_adapter
    'AdaptationMethod',
    'DomainInfo',
    'DomainAdapter',
    'MultiDomainAdapter',
    # meta_learner
    'MetaLearningMethod',
    'Task',
    'MetaLearner',
    'FederatedMetaLearner',
    # feature_alignment
    'AlignmentMethod',
    'FeatureStatistics',
    'FeatureAligner',
    'MultiSourceAligner'
]
