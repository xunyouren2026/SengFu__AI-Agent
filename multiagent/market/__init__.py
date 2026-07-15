"""
智能体市场与生态系统

提供完整的Agent服务市场功能，包括服务上架、发现、撮合、定价、支付、争议仲裁、评分和SLA监控。
"""

from .listings import (
    ServiceListing,
    ListingManager,
    Capability,
    PricingTier,
    ServiceType,
    PricingModel,
    ListingStatus,
)

from .discovery import (
    DiscoveryEngine,
    SearchQuery,
    SearchResult,
    RecommendationContext,
    SemanticMatcher,
    SortCriteria,
    FilterCondition,
    FilterOperator,
)

from .order_matching import (
    MatchingEngine,
    TaskRequest,
    TaskRequirement,
    MatchResult,
    MatchCandidate,
    AgentLoad,
    TaskPriority,
    TaskStatus,
    MatchingStrategy,
    BatchMatcher,
)

from .pricing_oracle import (
    PricingOracle,
    PriceSignal,
    MarketDemand,
    MarketSupply,
    HistoricalPrice,
    PriceElasticity,
    PricingStrategy,
    PriceDirection,
)

from .escrow import (
    EscrowManager,
    EscrowAccount,
    ReleaseStage,
    PaymentReceipt,
    EscrowStatus,
    ReleaseCondition,
)

from .dispute_resolution import (
    DisputeResolutionManager,
    DisputeCase,
    Evidence,
    Arbitrator,
    ArbitrationVote,
    AutoArbitrationEngine,
    DisputeType,
    DisputeStatus,
    ResolutionType,
    ArbitratorType,
)

from .rating_aggregator import (
    RatingAggregator,
    Rating,
    AgentReputation,
    RatingCategory,
)

from .composite_agent import (
    CompositeAgentManager,
    CompositeDefinition,
    CompositeListing,
    AgentNode,
    CompositionEdge,
    CompositionType,
    CompositionStatus,
)

from .sla_monitor import (
    SLAMonitor,
    SLAMetric,
    MetricReading,
    SLAReport,
    SLAAlert,
    SLAMetricType,
    SLAStatus,
    AlertSeverity,
)


__version__ = "1.0.0"

__all__ = [
    # listings
    "ServiceListing",
    "ListingManager",
    "Capability",
    "PricingTier",
    "ServiceType",
    "PricingModel",
    "ListingStatus",
    # discovery
    "DiscoveryEngine",
    "SearchQuery",
    "SearchResult",
    "RecommendationContext",
    "SemanticMatcher",
    "SortCriteria",
    "FilterCondition",
    "FilterOperator",
    # order_matching
    "MatchingEngine",
    "TaskRequest",
    "TaskRequirement",
    "MatchResult",
    "MatchCandidate",
    "AgentLoad",
    "TaskPriority",
    "TaskStatus",
    "MatchingStrategy",
    "BatchMatcher",
    # pricing_oracle
    "PricingOracle",
    "PriceSignal",
    "MarketDemand",
    "MarketSupply",
    "HistoricalPrice",
    "PriceElasticity",
    "PricingStrategy",
    "PriceDirection",
    # escrow
    "EscrowManager",
    "EscrowAccount",
    "ReleaseStage",
    "PaymentReceipt",
    "EscrowStatus",
    "ReleaseCondition",
    # dispute_resolution
    "DisputeResolutionManager",
    "DisputeCase",
    "Evidence",
    "Arbitrator",
    "ArbitrationVote",
    "AutoArbitrationEngine",
    "DisputeType",
    "DisputeStatus",
    "ResolutionType",
    "ArbitratorType",
    # rating_aggregator
    "RatingAggregator",
    "Rating",
    "AgentReputation",
    "RatingCategory",
    # composite_agent
    "CompositeAgentManager",
    "CompositeDefinition",
    "CompositeListing",
    "AgentNode",
    "CompositionEdge",
    "CompositionType",
    "CompositionStatus",
    # sla_monitor
    "SLAMonitor",
    "SLAMetric",
    "MetricReading",
    "SLAReport",
    "SLAAlert",
    "SLAMetricType",
    "SLAStatus",
    "AlertSeverity",
]
