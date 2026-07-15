"""
AGI Unified Framework - Memory Management Module

提供统一的GPU/CPU内存管理功能，包括：
- MemoryManager: 统一管理GPU/CPU内存分配和释放
- MemoryAllocator: 内存分配器，支持内存池和预分配
- MemoryProfiler: 内存使用分析，记录峰值和趋势
- GarbageCollector: 垃圾回收触发和优化
- OOMHandler: 内存溢出处理，自动清理和降级
- SwapManager: CPU-GPU内存交换管理

模块路径: hardware/memory/__init__.py
"""

import warnings
from typing import Optional, Dict, Any

# 导入所有子模块
from .memory_manager import (
    MemoryManager,
    MemoryConfig,
    MemoryStats,
    MemoryType,
    MemoryStatus,
    get_memory_manager,
    get_gpu_memory_summary
)

from .allocator import (
    MemoryAllocator,
    MemoryPool,
    MemoryBlock,
    PoolConfig,
    AllocationStrategy,
    BlockStatus,
    create_allocator,
    aligned_size
)

from .memory_profiler import (
    MemoryProfiler,
    MemoryEvent,
    MemorySnapshot,
    MemoryTrend,
    ProfileLevel,
    MemoryEventType,
    create_profiler,
    profile_memory
)

from .garbage_collector import (
    GarbageCollector,
    GCConfig,
    GCStats,
    GCStrategy,
    GCPhase,
    create_gc,
    force_gc,
    get_gc_stats
)

from .oom_handler import (
    OOMHandler,
    OOMConfig,
    OOMEvent,
    RecoveryResult,
    OOMSeverity,
    RecoveryAction,
    create_oom_handler,
    handle_oom
)

from .swap_manager import (
    SwapManager,
    SwapConfig,
    SwappedTensor,
    SwapStatus,
    SwapStrategy,
    create_swap_manager,
    auto_swap
)

# 版本信息
__version__ = "1.0.0"
__author__ = "AGI Unified Framework Team"

# 导出列表
__all__ = [
    # Memory Manager
    'MemoryManager',
    'MemoryConfig',
    'MemoryStats',
    'MemoryType',
    'MemoryStatus',
    'get_memory_manager',
    'get_gpu_memory_summary',
    
    # Allocator
    'MemoryAllocator',
    'MemoryPool',
    'MemoryBlock',
    'PoolConfig',
    'AllocationStrategy',
    'BlockStatus',
    'create_allocator',
    'aligned_size',
    
    # Profiler
    'MemoryProfiler',
    'MemoryEvent',
    'MemorySnapshot',
    'MemoryTrend',
    'ProfileLevel',
    'MemoryEventType',
    'create_profiler',
    'profile_memory',
    
    # Garbage Collector
    'GarbageCollector',
    'GCConfig',
    'GCStats',
    'GCStrategy',
    'GCPhase',
    'create_gc',
    'force_gc',
    'get_gc_stats',
    
    # OOM Handler
    'OOMHandler',
    'OOMConfig',
    'OOMEvent',
    'RecoveryResult',
    'OOMSeverity',
    'RecoveryAction',
    'create_oom_handler',
    'handle_oom',
    
    # Swap Manager
    'SwapManager',
    'SwapConfig',
    'SwappedTensor',
    'SwapStatus',
    'SwapStrategy',
    'create_swap_manager',
    'auto_swap',
]


class UnifiedMemorySystem:
    """
    统一内存系统
    
    整合所有内存管理组件，提供一站式内存管理解决方案。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化统一内存系统
        
        Args:
            config: 配置字典，包含各组件的配置
        """
        config = config or {}
        
        # 初始化各组件
        self.memory_manager = MemoryManager(config.get('memory_manager'))
        self.allocator = MemoryAllocator(config.get('allocator'))
        self.profiler = MemoryProfiler(config.get('profiler'))
        self.garbage_collector = GarbageCollector(config.get('garbage_collector'))
        self.oom_handler = OOMHandler(config.get('oom_handler'))
        self.swap_manager = SwapManager(config.get('swap_manager'))
        
        self._initialized = False
    
    def initialize(self):
        """初始化所有组件"""
        if self._initialized:
            return
        
        self.memory_manager.initialize()
        self.allocator.initialize()
        self.profiler.initialize()
        self.garbage_collector.initialize()
        self.oom_handler.initialize()
        self.swap_manager.initialize()
        
        self._initialized = True
    
    def shutdown(self):
        """关闭所有组件"""
        self.swap_manager.shutdown()
        self.oom_handler.shutdown()
        self.garbage_collector.shutdown()
        self.profiler.shutdown()
        self.allocator.shutdown()
        self.memory_manager.shutdown()
        
        self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有组件的统计信息"""
        return {
            'memory_manager': self.memory_manager.get_memory_stats(),
            'allocator': self.allocator.get_stats(),
            'profiler': self.profiler.get_statistics(),
            'garbage_collector': self.garbage_collector.get_stats(),
            'oom_handler': self.oom_handler.get_stats(),
            'swap_manager': self.swap_manager.get_memory_stats()
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False


def create_memory_system(config: Optional[Dict[str, Any]] = None) -> UnifiedMemorySystem:
    """
    创建统一内存系统
    
    Args:
        config: 配置字典
        
    Returns:
        UnifiedMemorySystem实例
    """
    return UnifiedMemorySystem(config)
