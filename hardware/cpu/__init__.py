"""
AGI Unified Framework - CPU硬件模块

提供CPU硬件层面的优化和管理功能，包括：
- 缓存优化 (CacheOptimization)
- CPU性能调优 (CpuOptimizer)
- 内存对齐 (MemoryAlignment)
- NUMA节点绑定 (NumaBinding)
- SIMD指令集检测 (SimdDetection)
- 线程池管理 (ThreadPool)

模块路径: hardware/cpu/
"""

from .cache_optimization import (
    CacheOptimization,
    CacheLevel,
    CacheInfo,
    CacheTopology,
    CacheAccessPattern,
)

from .cpu_optimizer import (
    CpuOptimizer,
    PowerProfile,
    GovernorType,
    CpuCoreInfo,
    CpuSnapshot,
)

from .memory_alignment import (
    MemoryAlignment,
    AlignedBuffer,
    AlignmentSize,
    AlignmentStats,
)

from .numa_binding import (
    NumaBinding,
    NumaPolicy,
    NumaNode,
    NumaTopology,
    NumaAllocation,
)

from .simd_detection import (
    SimdDetection,
    SimdArchitecture,
    SimdExtension,
    SimdCapability,
    SimdProfile,
)

from .thread_pool import (
    ThreadPool,
    TaskPriority,
    WorkerState,
    ThreadPoolState,
    TaskItem,
    WorkerInfo,
    ThreadPoolStats,
)

__all__ = [
    # 缓存优化
    "CacheOptimization",
    "CacheLevel",
    "CacheInfo",
    "CacheTopology",
    "CacheAccessPattern",
    # CPU性能调优
    "CpuOptimizer",
    "PowerProfile",
    "GovernorType",
    "CpuCoreInfo",
    "CpuSnapshot",
    # 内存对齐
    "MemoryAlignment",
    "AlignedBuffer",
    "AlignmentSize",
    "AlignmentStats",
    # NUMA绑定
    "NumaBinding",
    "NumaPolicy",
    "NumaNode",
    "NumaTopology",
    "NumaAllocation",
    # SIMD检测
    "SimdDetection",
    "SimdArchitecture",
    "SimdExtension",
    "SimdCapability",
    "SimdProfile",
    # 线程池
    "ThreadPool",
    "TaskPriority",
    "WorkerState",
    "ThreadPoolState",
    "TaskItem",
    "WorkerInfo",
    "ThreadPoolStats",
]
