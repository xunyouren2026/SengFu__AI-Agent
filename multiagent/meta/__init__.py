"""
元智能体与自组织系统 (Meta-Agent & Self-Organization System)

该模块提供了多智能体系统的元层管理能力，包括：
- 自组织规则系统：Agent基于规则自主组队
- 分布式选主算法：Bully算法和Raft选举
- 动态Agent工厂：根据任务描述生成专用Agent
- 智能体退役机制：评估并下线低效Agent
- 智能体克隆机制：复制高负载Agent分担任务
- 智能体合并机制：合并相似功能Agent减少冗余
- 教练智能体：分析失败并给出优化建议
"""

# 自组织系统
from .self_organization import (
    OrganizationStrategy,
    Capability,
    TaskRequirement,
    AgentProfile,
    Team,
    TeamFormationRule,
    CapabilityCoverageRule,
    LoadBalancingRule,
    TeamAffinityRule,
    DiversityRule,
    SelfOrganizationEngine,
    EmergentBehaviorDetector,
)

# 分布式选主
from .leader_election import (
    NodeState,
    ElectionState,
    NodeInfo,
    VoteRecord,
    ElectionMessage,
    ElectionAlgorithm,
    BullyElection,
    RaftElection,
    LeaderElectionManager,
    create_node_cluster,
)

# Agent工厂
from .agent_factory import (
    AgentType,
    CapabilityLevel,
    TaskFeature,
    AgentConfiguration,
    AgentTemplate,
    TaskFeatureExtractor,
    DynamicAgentFactory,
    quick_create_agent,
)

# 智能体退役
from .retirement import (
    RetirementStatus,
    RetirementReason,
    PerformanceMetrics,
    UsageMetrics,
    AgentHealthRecord,
    RetirementPolicy,
    PerformanceBasedPolicy,
    InactivityBasedPolicy,
    RedundancyBasedPolicy,
    CompositeRetirementPolicy,
    RetirementManager,
    ResourceOptimizer,
    create_default_retirement_manager,
)

# 智能体克隆
from .cloning import (
    CloneStatus,
    CloneStrategy,
    LoadMetrics,
    CloneInstance,
    ParentAgent,
    CloneDecisionPolicy,
    ThresholdBasedPolicy,
    PredictivePolicy,
    StateSynchronizer,
    IncrementalStateSynchronizer,
    TrafficDistributor,
    CloningManager,
    create_cloning_manager,
)

# 智能体合并
from .merge import (
    MergeStatus,
    SimilarityDimension,
    AgentProfile as MergeAgentProfile,
    SimilarityScore,
    MergePlan,
    SimilarityCalculator,
    CosineSimilarityCalculator,
    MergeStrategy,
    UnionMergeStrategy,
    IntersectionMergeStrategy,
    WeightedMergeStrategy,
    MergeManager,
    create_default_merge_manager,
)

# 教练智能体
from .agent_coach import (
    FailureType,
    SeverityLevel,
    FailureRecord,
    PerformanceSnapshot,
    DiagnosisReport,
    TrainingPlan,
    FailureAnalyzer,
    PatternBasedAnalyzer,
    RootCauseAnalyzer,
    RecommendationEngine,
    TrainingPlanGenerator,
    AgentCoach,
    create_agent_coach,
)

__version__ = "1.0.0"

__all__ = [
    # 自组织系统
    "OrganizationStrategy",
    "Capability",
    "TaskRequirement",
    "AgentProfile",
    "Team",
    "TeamFormationRule",
    "CapabilityCoverageRule",
    "LoadBalancingRule",
    "TeamAffinityRule",
    "DiversityRule",
    "SelfOrganizationEngine",
    "EmergentBehaviorDetector",
    
    # 分布式选主
    "NodeState",
    "ElectionState",
    "NodeInfo",
    "VoteRecord",
    "ElectionMessage",
    "ElectionAlgorithm",
    "BullyElection",
    "RaftElection",
    "LeaderElectionManager",
    "create_node_cluster",
    
    # Agent工厂
    "AgentType",
    "CapabilityLevel",
    "TaskFeature",
    "AgentConfiguration",
    "AgentTemplate",
    "TaskFeatureExtractor",
    "DynamicAgentFactory",
    "quick_create_agent",
    
    # 智能体退役
    "RetirementStatus",
    "RetirementReason",
    "PerformanceMetrics",
    "UsageMetrics",
    "AgentHealthRecord",
    "RetirementPolicy",
    "PerformanceBasedPolicy",
    "InactivityBasedPolicy",
    "RedundancyBasedPolicy",
    "CompositeRetirementPolicy",
    "RetirementManager",
    "ResourceOptimizer",
    "create_default_retirement_manager",
    
    # 智能体克隆
    "CloneStatus",
    "CloneStrategy",
    "LoadMetrics",
    "CloneInstance",
    "ParentAgent",
    "CloneDecisionPolicy",
    "ThresholdBasedPolicy",
    "PredictivePolicy",
    "StateSynchronizer",
    "IncrementalStateSynchronizer",
    "TrafficDistributor",
    "CloningManager",
    "create_cloning_manager",
    
    # 智能体合并
    "MergeStatus",
    "SimilarityDimension",
    "MergeAgentProfile",
    "SimilarityScore",
    "MergePlan",
    "SimilarityCalculator",
    "CosineSimilarityCalculator",
    "MergeStrategy",
    "UnionMergeStrategy",
    "IntersectionMergeStrategy",
    "WeightedMergeStrategy",
    "MergeManager",
    "create_default_merge_manager",
    
    # 教练智能体
    "FailureType",
    "SeverityLevel",
    "FailureRecord",
    "PerformanceSnapshot",
    "DiagnosisReport",
    "TrainingPlan",
    "FailureAnalyzer",
    "PatternBasedAnalyzer",
    "RootCauseAnalyzer",
    "RecommendationEngine",
    "TrainingPlanGenerator",
    "AgentCoach",
    "create_agent_coach",
]
