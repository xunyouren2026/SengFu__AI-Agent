"""
LLM路由模块 - 智能模型选择与路由

该模块提供了多模型编排的核心功能，包括：
- 智能模型选择器：根据任务类型、复杂度、语言选择最佳模型
- 渠道-模型映射：支持不同渠道的专属模型配置
- 多语言智能路由：支持中/英/日/韩等语言的自动检测和路由
- 多模型聚合推理：同时调用多个模型并融合结果
- 模型负载均衡：支持轮询/加权/最少连接等策略
- 成本感知优化：在性能和成本之间找到最佳平衡
- 模型Fallback链：故障自动切换和恢复
- 模型熔断器：防止故障模型影响整体服务
- 语义缓存引擎：基于向量相似度的智能缓存
- Token配额管理：用户/渠道级别的用量控制
- 优先级调度器：延迟敏感任务的优先处理

Author: AGI Team
Version: 1.0.0
"""

from .selector import (
    ModelSelector,
    TaskType,
    TaskComplexity,
    ModelCapability,
    ModelProfile,
    SelectionCriteria,
    SelectionResult,
)
from .channel_mapper import (
    ChannelMapper,
    ChannelConfig,
    ChannelType,
    ChannelCapability,
    ModelBinding,
)
from .lang_router import (
    LanguageRouter,
    Language,
    LanguageDetection,
    LanguageConfig,
    TranslationOption,
)
from .aggregator import (
    ModelAggregator,
    AggregationStrategy,
    FusionResult,
    ModelResponse,
    ConflictResolution,
)
from .load_balancer import (
    LoadBalancer,
    BalanceStrategy,
    ModelInstance,
    HealthStatus,
    LoadBalanceResult,
)
from .cost_optimizer import (
    CostOptimizer,
    CostModel,
    TokenCost,
    BudgetLimit,
    CostReport,
)
from .fallback_chain import (
    FallbackChain,
    FallbackConfig,
    FallbackResult,
    RecoveryConfig,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitConfig,
    CircuitMetrics,
)
from .response_cache import (
    SemanticCache,
    CacheEntry,
    CacheConfig,
    CacheResult,
    SimilarityMatch,
)
from .quota_manager import (
    QuotaManager,
    QuotaConfig,
    QuotaUsage,
    QuotaLimit,
)
from .priority_scheduler import (
    PriorityScheduler,
    TaskPriority,
    ScheduledTask,
    SchedulingResult,
    QueueConfig,
)

__all__ = [
    # 智能模型选择器
    "ModelSelector",
    "TaskType",
    "TaskComplexity",
    "ModelCapability",
    "ModelProfile",
    "SelectionCriteria",
    "SelectionResult",
    # 渠道映射
    "ChannelMapper",
    "ChannelConfig",
    "ChannelType",
    "ChannelCapability",
    "ModelBinding",
    # 多语言路由
    "LanguageRouter",
    "Language",
    "LanguageDetection",
    "LanguageConfig",
    "TranslationOption",
    # 多模型聚合
    "ModelAggregator",
    "AggregationStrategy",
    "FusionResult",
    "ModelResponse",
    "ConflictResolution",
    # 负载均衡
    "LoadBalancer",
    "BalanceStrategy",
    "ModelInstance",
    "HealthStatus",
    "LoadBalanceResult",
    # 成本优化
    "CostOptimizer",
    "CostModel",
    "TokenCost",
    "BudgetLimit",
    "CostReport",
    # Fallback链
    "FallbackChain",
    "FallbackConfig",
    "FallbackResult",
    "RecoveryConfig",
    # 熔断器
    "CircuitBreaker",
    "CircuitState",
    "CircuitConfig",
    "CircuitMetrics",
    # 语义缓存
    "SemanticCache",
    "CacheEntry",
    "CacheConfig",
    "CacheResult",
    "SimilarityMatch",
    # 配额管理
    "QuotaManager",
    "QuotaConfig",
    "QuotaUsage",
    "QuotaLimit",
    # 优先级调度
    "PriorityScheduler",
    "TaskPriority",
    "ScheduledTask",
    "SchedulingResult",
    "QueueConfig",
]
