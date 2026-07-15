"""
MemoryAllocator - 内存分配器模块

支持内存池和预分配策略，提供高效的内存管理机制。
支持多种分配策略和内存块复用。

模块路径: hardware/memory/allocator.py
"""

import os
import sys
import math
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import warnings
import weakref

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. MemoryAllocator will run in limited mode.")


class AllocationStrategy(Enum):
    """内存分配策略"""
    FIRST_FIT = auto()
    BEST_FIT = auto()
    WORST_FIT = auto()
    BUDDY_SYSTEM = auto()
    SLAB_ALLOCATION = auto()


class BlockStatus(Enum):
    """内存块状态"""
    FREE = auto()
    ALLOCATED = auto()
    RESERVED = auto()
    PINNED = auto()


@dataclass
class MemoryBlock:
    """内存块数据结构"""
    size: int
    address: int
    status: BlockStatus
    device: Union[str, int]
    tensor: Optional[Any] = None
    allocation_time: Optional[float] = None
    last_access_time: Optional[float] = None
    access_count: int = 0
    tag: Optional[str] = None
    
    def __post_init__(self):
        if self.allocation_time is None:
            self.allocation_time = time.time()
        if self.last_access_time is None:
            self.last_access_time = time.time()
    
    @property
    def is_free(self) -> bool:
        return self.status == BlockStatus.FREE
    
    @property
    def age(self) -> float:
        """获取内存块年龄（秒）"""
        return time.time() - self.allocation_time
    
    def mark_accessed(self):
        """标记访问"""
        self.last_access_time = time.time()
        self.access_count += 1


@dataclass
class PoolConfig:
    """内存池配置"""
    # 基本配置
    initial_pool_size: int = 1024 * 1024 * 1024  # 1GB
    max_pool_size: int = 8 * 1024 * 1024 * 1024  # 8GB
    min_block_size: int = 1024  # 1KB
    alignment: int = 512  # 对齐要求
    
    # 扩展策略
    growth_factor: float = 2.0
    max_growth_steps: int = 10
    
    # 回收策略
    enable_gc: bool = True
    gc_threshold: float = 0.8
    idle_timeout: float = 300.0  # 5分钟
    
    # 预分配策略
    enable_preallocation: bool = True
    preallocation_sizes: List[int] = field(default_factory=lambda: [
        1024 * 1024,      # 1MB
        4 * 1024 * 1024,  # 4MB
        16 * 1024 * 1024, # 16MB
        64 * 1024 * 1024, # 64MB
        256 * 1024 * 1024 # 256MB
    ])
    
    # 分配策略
    strategy: AllocationStrategy = AllocationStrategy.BEST_FIT


class MemoryPool:
    """
    内存池实现
    
    管理预分配的内存块，支持多种分配策略和内存复用。
    """
    
    def __init__(self, device: Union[str, int], config: PoolConfig = None):
        """
        初始化内存池
        
        Args:
            device: 设备标识 ('cpu' 或 GPU ID)
            config: 内存池配置
        """
        self.device = device
        self.config = config or PoolConfig()
        self.logger = logging.getLogger(f"{__name__}.MemoryPool.{device}")
        
        # 内存块管理
        self._blocks: List[MemoryBlock] = []
        self._free_blocks: Dict[int, List[MemoryBlock]] = defaultdict(list)
        self._allocated_blocks: Dict[int, MemoryBlock] = {}  # id -> block
        
        # 统计信息
        self._total_allocated = 0
        self._total_requested = 0
        self._allocation_count = 0
        self._hit_count = 0
        self._miss_count = 0
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 预分配
        if self.config.enable_preallocation:
            self._preallocate()
    
    def _preallocate(self):
        """预分配内存块"""
        if not TORCH_AVAILABLE:
            return
        
        for size in self.config.preallocation_sizes:
            try:
                if self.device == 'cpu':
                    tensor = torch.empty(size // 4, dtype=torch.float32)
                else:
                    tensor = torch.empty(
                        size // 4,
                        dtype=torch.float32,
                        device=f'cuda:{self.device}'
                    )
                
                block = MemoryBlock(
                    size=size,
                    address=id(tensor),
                    status=BlockStatus.FREE,
                    device=self.device,
                    tensor=tensor,
                    tag='preallocated'
                )
                
                self._blocks.append(block)
                self._free_blocks[size].append(block)
                self._total_allocated += size
                
                self.logger.debug(f"Preallocated {size} bytes on {self.device}")
                
            except Exception as e:
                self.logger.warning(f"Failed to preallocate {size} bytes: {e}")
                break
    
    def allocate(self, size: int, dtype: Any = None, tag: str = None) -> Optional[Any]:
        """
        从内存池分配内存
        
        Args:
            size: 需要的字节数
            dtype: 数据类型
            tag: 分配标签（用于调试）
            
        Returns:
            分配的张量，失败返回None
        """
        with self._lock:
            # 对齐大小
            aligned_size = self._align_size(size)
            self._total_requested += aligned_size
            self._allocation_count += 1
            
            # 尝试从内存池分配
            block = self._find_free_block(aligned_size)
            
            if block is not None:
                # 内存池命中
                self._hit_count += 1
                block.status = BlockStatus.ALLOCATED
                block.tag = tag
                block.mark_accessed()
                
                # 如果块大小大于请求大小，分割块
                if block.size > aligned_size * 1.5:
                    remaining = self._split_block(block, aligned_size)
                    if remaining:
                        self._free_blocks[remaining.size].append(remaining)
                
                self._allocated_blocks[id(block.tensor)] = block
                
                # 调整张量大小
                if dtype is not None and block.tensor.dtype != dtype:
                    block.tensor = block.tensor.to(dtype)
                
                return block.tensor
            
            # 内存池未命中，直接分配
            self._miss_count += 1
            return self._allocate_new(aligned_size, dtype, tag)
    
    def _align_size(self, size: int) -> int:
        """对齐大小"""
        alignment = self.config.alignment
        return ((size + alignment - 1) // alignment) * alignment
    
    def _find_free_block(self, size: int) -> Optional[MemoryBlock]:
        """
        查找空闲内存块
        
        根据配置的策略选择合适的块。
        """
        strategy = self.config.strategy
        
        if strategy == AllocationStrategy.FIRST_FIT:
            return self._first_fit(size)
        elif strategy == AllocationStrategy.BEST_FIT:
            return self._best_fit(size)
        elif strategy == AllocationStrategy.WORST_FIT:
            return self._worst_fit(size)
        elif strategy == AllocationStrategy.BUDDY_SYSTEM:
            return self._buddy_allocate(size)
        else:
            return self._first_fit(size)
    
    def _first_fit(self, size: int) -> Optional[MemoryBlock]:
        """首次适应算法"""
        for block_size, blocks in self._free_blocks.items():
            if block_size >= size and blocks:
                return blocks.pop(0)
        return None
    
    def _best_fit(self, size: int) -> Optional[MemoryBlock]:
        """最佳适应算法"""
        best_block = None
        best_size = float('inf')
        
        for block_size, blocks in self._free_blocks.items():
            if size <= block_size < best_size and blocks:
                best_block = blocks[0]
                best_size = block_size
        
        if best_block:
            self._free_blocks[best_size].remove(best_block)
        
        return best_block
    
    def _worst_fit(self, size: int) -> Optional[MemoryBlock]:
        """最坏适应算法"""
        worst_block = None
        worst_size = 0
        
        for block_size, blocks in self._free_blocks.items():
            if block_size > worst_size and blocks:
                worst_block = blocks[0]
                worst_size = block_size
        
        if worst_block and worst_size >= size:
            self._free_blocks[worst_size].remove(worst_block)
            return worst_block
        
        return None
    
    def _buddy_allocate(self, size: int) -> Optional[MemoryBlock]:
        """伙伴系统分配算法"""
        # 找到最小的2的幂次方大小
        buddy_size = 1
        while buddy_size < size:
            buddy_size *= 2
        
        return self._best_fit(buddy_size)
    
    def _split_block(self, block: MemoryBlock, size: int) -> Optional[MemoryBlock]:
        """分割内存块"""
        if block.size - size < self.config.min_block_size:
            return None
        
        remaining_size = block.size - size
        
        # 创建新块
        if block.tensor is not None:
            # 分割张量
            try:
                remaining_tensor = block.tensor[remaining_size // 4:]
                block.tensor = block.tensor[:size // 4]
            except:
                return None
        else:
            remaining_tensor = None
        
        block.size = size
        
        remaining_block = MemoryBlock(
            size=remaining_size,
            address=id(remaining_tensor) if remaining_tensor else 0,
            status=BlockStatus.FREE,
            device=self.device,
            tensor=remaining_tensor
        )
        
        self._blocks.append(remaining_block)
        return remaining_block
    
    def _allocate_new(self, size: int, dtype: Any, tag: str) -> Optional[Any]:
        """分配新内存"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            if self.device == 'cpu':
                tensor = torch.empty(size // 4, dtype=dtype or torch.float32)
            else:
                tensor = torch.empty(
                    size // 4,
                    dtype=dtype or torch.float32,
                    device=f'cuda:{self.device}'
                )
            
            block = MemoryBlock(
                size=size,
                address=id(tensor),
                status=BlockStatus.ALLOCATED,
                device=self.device,
                tensor=tensor,
                tag=tag
            )
            
            self._blocks.append(block)
            self._allocated_blocks[id(tensor)] = block
            self._total_allocated += size
            
            return tensor
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error(f"OOM when allocating {size} bytes: {e}")
            return None
    
    def free(self, tensor: Any) -> bool:
        """
        释放内存到内存池
        
        Args:
            tensor: 要释放的张量
            
        Returns:
            bool: 释放是否成功
        """
        with self._lock:
            tensor_id = id(tensor)
            
            if tensor_id not in self._allocated_blocks:
                return False
            
            block = self._allocated_blocks.pop(tensor_id)
            block.status = BlockStatus.FREE
            block.tag = None
            block.mark_accessed()
            
            # 尝试合并相邻的空闲块
            self._coalesce_blocks()
            
            # 如果启用了GC且内存使用率过高，执行清理
            if self.config.enable_gc and self._should_gc():
                self._gc()
            
            return True
    
    def _coalesce_blocks(self):
        """合并相邻的空闲块"""
        # 简化的合并策略：合并相同大小的块
        for size in list(self._free_blocks.keys()):
            blocks = self._free_blocks[size]
            if len(blocks) > 1:
                # 保留较新的块
                blocks.sort(key=lambda b: b.last_access_time, reverse=True)
                self._free_blocks[size] = blocks[:max(1, len(blocks) // 2)]
    
    def _should_gc(self) -> bool:
        """检查是否需要垃圾回收"""
        if self._total_allocated == 0:
            return False
        
        utilization = self._get_utilization()
        return utilization > self.config.gc_threshold
    
    def _gc(self):
        """执行垃圾回收"""
        self.logger.debug("Running garbage collection")
        
        current_time = time.time()
        freed = 0
        
        # 清理长时间未使用的空闲块
        for size in list(self._free_blocks.keys()):
            blocks = self._free_blocks[size]
            keep_blocks = []
            
            for block in blocks:
                if current_time - block.last_access_time > self.config.idle_timeout:
                    freed += block.size
                    block.status = BlockStatus.RESERVED
                    block.tensor = None
                else:
                    keep_blocks.append(block)
            
            self._free_blocks[size] = keep_blocks
        
        self._total_allocated -= freed
        self.logger.debug(f"GC freed {freed} bytes")
    
    def _get_utilization(self) -> float:
        """获取内存使用率"""
        if self.config.max_pool_size == 0:
            return 0.0
        return self._total_allocated / self.config.max_pool_size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息"""
        with self._lock:
            total_free = sum(
                sum(b.size for b in blocks)
                for blocks in self._free_blocks.values()
            )
            
            return {
                'device': self.device,
                'total_allocated': self._total_allocated,
                'total_free': total_free,
                'allocated_blocks': len(self._allocated_blocks),
                'free_blocks': sum(len(blocks) for blocks in self._free_blocks.values()),
                'utilization': self._get_utilization(),
                'hit_rate': self._hit_count / max(1, self._allocation_count),
                'allocation_count': self._allocation_count,
                'cache_hits': self._hit_count,
                'cache_misses': self._miss_count
            }
    
    def clear(self):
        """清空内存池"""
        with self._lock:
            self._blocks.clear()
            self._free_blocks.clear()
            self._allocated_blocks.clear()
            self._total_allocated = 0
            self._total_requested = 0
            self._allocation_count = 0
            self._hit_count = 0
            self._miss_count = 0
            
            if TORCH_AVAILABLE and self.device != 'cpu':
                torch.cuda.empty_cache()


class MemoryAllocator:
    """
    内存分配器
    
    提供高级的内存分配功能，支持多设备内存池和预分配策略。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化内存分配器
        
        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 配置
        self.config = config or {}
        self._pool_configs: Dict[Union[str, int], PoolConfig] = {}
        
        # 内存池
        self._pools: Dict[Union[str, int], MemoryPool] = {}
        
        # 设备列表
        self._devices: Set[Union[str, int]] = set()
        
        # 全局统计
        self._global_stats = {
            'total_allocations': 0,
            'total_bytes_allocated': 0,
            'peak_memory_usage': 0
        }
        
        # 线程安全
        self._lock = threading.RLock()
        
        self._initialized = False
        self.logger.info("MemoryAllocator initialized")
    
    def _setup_logging(self):
        """设置日志"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def initialize(self, devices: Optional[List[Union[str, int]]] = None):
        """
        初始化分配器
        
        Args:
            devices: 要管理的设备列表
        """
        if self._initialized:
            return
        
        if devices is None:
            devices = self._detect_devices()
        
        for device in devices:
            self._add_device(device)
        
        self._initialized = True
        self.logger.info(f"MemoryAllocator initialized with devices: {devices}")
    
    def _detect_devices(self) -> List[Union[str, int]]:
        """检测可用设备"""
        devices = ['cpu']
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            devices.extend(range(gpu_count))
        
        return devices
    
    def _add_device(self, device: Union[str, int], config: Optional[PoolConfig] = None):
        """添加设备"""
        with self._lock:
            if device in self._devices:
                return
            
            pool_config = config or self._pool_configs.get(device, PoolConfig())
            self._pools[device] = MemoryPool(device, pool_config)
            self._devices.add(device)
            
            self.logger.info(f"Added device {device} to allocator")
    
    def allocate(self, size: int, device: Optional[Union[str, int]] = None,
                 dtype: Any = None, tag: str = None) -> Optional[Any]:
        """
        分配内存
        
        Args:
            size: 字节数
            device: 目标设备
            dtype: 数据类型
            tag: 分配标签
            
        Returns:
            分配的张量
        """
        if not self._initialized:
            self.initialize()
        
        # 确定目标设备
        if device is None:
            device = self._select_best_device(size)
        
        if device not in self._pools:
            self._add_device(device)
        
        # 从内存池分配
        tensor = self._pools[device].allocate(size, dtype, tag)
        
        if tensor is not None:
            with self._lock:
                self._global_stats['total_allocations'] += 1
                self._global_stats['total_bytes_allocated'] += size
                self._global_stats['peak_memory_usage'] = max(
                    self._global_stats['peak_memory_usage'],
                    self._global_stats['total_bytes_allocated']
                )
        
        return tensor
    
    def _select_best_device(self, size: int) -> Union[str, int]:
        """选择最佳设备"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 'cpu'
        
        # 简单策略：选择可用内存最多的GPU
        best_device = 'cpu'
        best_memory = 0
        
        for device in self._devices:
            if device == 'cpu':
                continue
            
            try:
                torch.cuda.synchronize(device)
                free_memory = (torch.cuda.get_device_properties(device).total_memory -
                              torch.cuda.memory_allocated(device))
                
                if free_memory > best_memory:
                    best_memory = free_memory
                    best_device = device
            except:
                continue
        
        return best_device
    
    def free(self, tensor: Any) -> bool:
        """
        释放内存
        
        Args:
            tensor: 要释放的张量
            
        Returns:
            bool: 释放是否成功
        """
        if tensor is None:
            return False
        
        # 确定张量所属设备
        device = self._get_tensor_device(tensor)
        
        if device in self._pools:
            return self._pools[device].free(tensor)
        
        return False
    
    def _get_tensor_device(self, tensor: Any) -> Union[str, int]:
        """获取张量所在设备"""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        if hasattr(tensor, 'device'):
            if tensor.device.type == 'cuda':
                return tensor.device.index or 0
            return 'cpu'
        
        return 'cpu'
    
    def preallocate(self, sizes: List[int], device: Union[str, int] = None):
        """
        预分配内存
        
        Args:
            sizes: 要预分配的大小列表
            device: 目标设备
        """
        if device is None:
            device = 0 if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        if device not in self._pools:
            self._add_device(device)
        
        pool = self._pools[device]
        for size in sizes:
            pool.allocate(size, tag='preallocated')
    
    def get_stats(self) -> Dict[str, Any]:
        """获取分配器统计信息"""
        with self._lock:
            stats = {
                'global': self._global_stats.copy(),
                'pools': {}
            }
            
            for device, pool in self._pools.items():
                stats['pools'][str(device)] = pool.get_stats()
            
            return stats
    
    def clear_cache(self, device: Optional[Union[str, int]] = None):
        """
        清空缓存
        
        Args:
            device: 指定设备，None则清空所有
        """
        if device is None:
            for pool in self._pools.values():
                pool.clear()
        elif device in self._pools:
            self._pools[device].clear()
    
    def shutdown(self):
        """关闭分配器"""
        self.logger.info("Shutting down MemoryAllocator...")
        self.clear_cache()
        self._pools.clear()
        self._devices.clear()
        self._initialized = False
        self.logger.info("MemoryAllocator shutdown complete")


# 便捷函数
def create_allocator(config: Optional[Dict] = None) -> MemoryAllocator:
    """创建内存分配器"""
    return MemoryAllocator(config)


def aligned_size(size: int, alignment: int = 512) -> int:
    """计算对齐后的大小"""
    return ((size + alignment - 1) // alignment) * alignment
