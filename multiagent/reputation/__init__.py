"""
信誉模块

提供多Agent系统的信誉管理:
- 信誉分计算
- 贝叶斯平均
- 信誉衰减
- 信誉背书
- 评价收集
- 女巫攻击防御
"""

from .calculator import (
    ReputationCalculator,
    ReputationScore,
    ReputationLevel,
    TaskMetrics
)

from .bayesian_average import (
    BayesianAverageCalculator,
    BayesianConfig,
    RatingData,
    DynamicBayesianAverage
)

from .decay import (
    ReputationDecay,
    DecayConfig,
    AgentActivity,
    AdaptiveDecay
)

from .endorsement import (
    EndorsementManager,
    Endorsement,
    EndorsementStatus,
    EndorsementType,
    EndorsementMetrics
)

from .review_collector import (
    ReviewCollector,
    Review,
    ReviewerRole,
    ReviewDimension,
    ReviewSummary
)

from .sybil_resistant import (
    SybilResistantManager,
    IdentityType,
    Identity,
    PoWProof
)


__all__ = [
    # 信誉计算
    'ReputationCalculator',
    'ReputationScore',
    'ReputationLevel',
    'TaskMetrics',
    
    # 贝叶斯平均
    'BayesianAverageCalculator',
    'BayesianConfig',
    'RatingData',
    'DynamicBayesianAverage',
    
    # 信誉衰减
    'ReputationDecay',
    'DecayConfig',
    'AgentActivity',
    'AdaptiveDecay',
    
    # 信誉背书
    'EndorsementManager',
    'Endorsement',
    'EndorsementStatus',
    'EndorsementType',
    'EndorsementMetrics',
    
    # 评价收集
    'ReviewCollector',
    'Review',
    'ReviewerRole',
    'ReviewDimension',
    'ReviewSummary',
    
    # 女巫攻击防御
    'SybilResistantManager',
    'IdentityType',
    'Identity',
    'PoWProof'
]
