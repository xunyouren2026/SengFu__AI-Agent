"""
Hardware module for GPU memory planning and quantization.

This module provides:
- Model memory requirement estimation
- KV cache allocation strategies
- Dynamic batch size planning
- Out-of-memory prevention
- GPU detection and capability analysis
- Quantization strategy recommendation

Author: AGI Unified Framework Team
"""

# Memory planner exports
from .memory_planner import (
    MemoryUnit,
    AllocationStrategy,
    GPUInfo,
    ModelMemoryRequirements,
    KVCacheConfig,
    BatchSizePlan,
    MemoryPlan,
    MemoryEstimator,
    KVCacheAllocator,
    BatchSizePlanner,
    OOMPrevention,
    MemoryPlanner,
    create_memory_planner,
    estimate_model_memory,
)

# Quantization selector exports
from .quant_selector import (
    QuantStrategy,
    GPUArchitecture,
    GPUCapability,
    QuantizationResult,
    QuantizationProfile,
    AccuracyBenchmark,
    GPUDetector,
    AccuracyComparator,
    PerformanceEstimator,
    QuantSelector,
    detect_gpu,
    select_quantization,
    get_quantization_profile,
)

__all__ = [
    # Memory Planner
    "MemoryUnit",
    "AllocationStrategy",
    "GPUInfo",
    "ModelMemoryRequirements",
    "KVCacheConfig",
    "BatchSizePlan",
    "MemoryPlan",
    "MemoryEstimator",
    "KVCacheAllocator",
    "BatchSizePlanner",
    "OOMPrevention",
    "MemoryPlanner",
    "create_memory_planner",
    "estimate_model_memory",
    # Quantization Selector
    "QuantStrategy",
    "GPUArchitecture",
    "GPUCapability",
    "QuantizationResult",
    "QuantizationProfile",
    "AccuracyBenchmark",
    "GPUDetector",
    "AccuracyComparator",
    "PerformanceEstimator",
    "QuantSelector",
    "detect_gpu",
    "select_quantization",
    "get_quantization_profile",
]
