"""
统一核心算法系统

为AGI和视频生成系统提供通用的算法组件。
所有组件都使用泛型设计，支持任意类型的数据。

模块结构：
- unified_config: 统一配置管理
- unified_memory: 统一记忆系统
- unified_attention: 统一注意力机制
- unified_chunking: 统一分块/上下文管理
- unified_moe: 统一混合专家系统
- unified_constraints: 统一约束系统

使用示例：
    >>> from agi_unified_framework.core.unified_algorithms import (
    ...     UnifiedAlgorithmConfig,
    ...     UnifiedMemoryBank,
    ...     UnifiedSlidingWindowAttention,
    ...     MixtureOfExperts,
    ...     ConstraintManager
    ... )
    
    >>> # 创建配置
    >>> config = UnifiedAlgorithmConfig.default_config()
    
    >>> # 使用记忆系统
    >>> memory = UnifiedMemoryBank(config)
    >>> memory.store(MemoryEntry(data="hello", importance=0.8))
    
    >>> # 使用注意力机制
    >>> attention = UnifiedSlidingWindowAttention(config=config)
    >>> output = attention.compute(query, key, value)
    
    >>> # 使用MoE
    >>> moe = MixtureOfExperts(config=config)
    >>> result = moe.process(data)
"""

__version__ = "1.0.0"
__author__ = "AGI Unified Framework Team"

# ============================================================================
# 配置模块
# ============================================================================

from .unified_config import (
    # 泛型类型
    T, S, K, V,
    
    # 枚举类型
    MemoryRetrievalMode,
    CompressionStrategy,
    AttentionPattern,
    ChunkingStrategy,
    BoundaryType,
    ExpertType,
    ConstraintPriority,
    
    # 配置类
    UnifiedAlgorithmConfig,
    
    # 配置工具函数
    merge_configs,
    config_from_dict,
    config_to_dict,
)

# ============================================================================
# 记忆系统模块
# ============================================================================

from .unified_memory import (
    # 数据结构
    MemoryEntry,
    MemoryQuery,
    
    # 记忆系统
    BaseMemory,
    UnifiedMemoryBank,
    UnifiedLightweightMemory,
    UnifiedHierarchicalMemory,
    MemoryLevel,
    UnifiedAdaptiveCompressor,
)

# ============================================================================
# 注意力机制模块
# ============================================================================

from .unified_attention import (
    # 数据结构
    AttentionContext,
    CacheZone,
    RoutingDecision,
    
    # 注意力机制
    UnifiedAttention,
    UnifiedSlidingWindowAttention,
    UnifiedDynamicRouting,
    UnifiedSparseAttention,
)

# ============================================================================
# 分块/上下文管理模块
# ============================================================================

from .unified_chunking import (
    # 数据结构
    Chunk,
    ChunkingResult,
    Boundary,
    
    # 分块组件
    UnifiedChunker,
    UnifiedOverlapFusion,
    UnifiedBoundaryDetector,
    UnifiedProgressiveLoader,
)

# ============================================================================
# 混合专家系统模块
# ============================================================================

from .unified_moe import (
    # 数据结构
    ExpertOutput,
    RoutingInfo,
    MoEStats,
    
    # 专家类
    Expert,
    PhysicalExpert,
    ReasoningExpert,
    MemoryExpert,
    PerceptionExpert,
    GenerationExpert,
    
    # MoE层
    MixtureOfExperts,
)

# ============================================================================
# 约束系统模块
# ============================================================================

from .unified_constraints import (
    # 数据结构
    ConstraintViolation,
    ConstraintCheckResult,
    ConstraintStats,
    
    # 约束类
    Constraint,
    PhysicsConstraint,
    LogicalConstraint,
    ConsistencyConstraint,
    
    # 约束管理器
    ConstraintManager,
)


# ============================================================================
# 便捷导入列表
# ============================================================================

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 配置
    "T", "S", "K", "V",
    "MemoryRetrievalMode",
    "CompressionStrategy",
    "AttentionPattern",
    "ChunkingStrategy",
    "BoundaryType",
    "ExpertType",
    "ConstraintPriority",
    "UnifiedAlgorithmConfig",
    "merge_configs",
    "config_from_dict",
    "config_to_dict",
    
    # 记忆系统
    "MemoryEntry",
    "MemoryQuery",
    "BaseMemory",
    "UnifiedMemoryBank",
    "UnifiedLightweightMemory",
    "UnifiedHierarchicalMemory",
    "MemoryLevel",
    "UnifiedAdaptiveCompressor",
    
    # 注意力机制
    "AttentionContext",
    "CacheZone",
    "RoutingDecision",
    "UnifiedAttention",
    "UnifiedSlidingWindowAttention",
    "UnifiedDynamicRouting",
    "UnifiedSparseAttention",
    
    # 分块管理
    "Chunk",
    "ChunkingResult",
    "Boundary",
    "UnifiedChunker",
    "UnifiedOverlapFusion",
    "UnifiedBoundaryDetector",
    "UnifiedProgressiveLoader",
    
    # 混合专家系统
    "ExpertOutput",
    "RoutingInfo",
    "MoEStats",
    "Expert",
    "PhysicalExpert",
    "ReasoningExpert",
    "MemoryExpert",
    "PerceptionExpert",
    "GenerationExpert",
    "MixtureOfExperts",
    
    # 约束系统
    "ConstraintViolation",
    "ConstraintCheckResult",
    "ConstraintStats",
    "Constraint",
    "PhysicsConstraint",
    "LogicalConstraint",
    "ConsistencyConstraint",
    "ConstraintManager",
]


def get_version() -> str:
    """获取版本信息"""
    return __version__


def create_default_system():
    """
    创建默认的统一算法系统实例
    
    返回包含所有核心组件的字典，方便快速开始使用。
    
    Returns:
        包含所有核心组件的字典
    """
    config = UnifiedAlgorithmConfig.default_config()
    
    return {
        'config': config,
        'memory': UnifiedMemoryBank(config),
        'attention': UnifiedSlidingWindowAttention(config=config),
        'chunker': UnifiedChunker(config=config),
        'moe': MixtureOfExperts(config=config),
        'constraints': ConstraintManager(),
    }


def create_video_optimized_system():
    """
    创建视频优化的统一算法系统实例
    
    Returns:
        视频优化的组件字典
    """
    config = UnifiedAlgorithmConfig.video_optimized_config()
    
    return {
        'config': config,
        'memory': UnifiedMemoryBank(config),
        'attention': UnifiedSlidingWindowAttention(config=config),
        'chunker': UnifiedChunker(
            strategy=ChunkingStrategy.TEMPORAL,
            config=config
        ),
        'moe': MixtureOfExperts(config=config),
        'constraints': ConstraintManager(),
        'boundary_detector': UnifiedBoundaryDetector(
            boundary_type=BoundaryType.SHOT
        ),
    }


def create_agi_optimized_system():
    """
    创建AGI优化的统一算法系统实例
    
    Returns:
        AGI优化的组件字典
    """
    config = UnifiedAlgorithmConfig.agi_optimized_config()
    
    return {
        'config': config,
        'memory': UnifiedHierarchicalMemory(config=config),
        'attention': UnifiedSparseAttention(config=config),
        'chunker': UnifiedChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            config=config
        ),
        'moe': MixtureOfExperts(config=config),
        'constraints': ConstraintManager(),
        'boundary_detector': UnifiedBoundaryDetector(
            boundary_type=BoundaryType.TOPIC
        ),
    }
