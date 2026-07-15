"""
GPU Memory Pool - GPU内存池管理

模块路径: hardware/gpu/memory_pool.py

提供高效的GPU内存池管理，减少内存分配/释放开销，
支持多GPU场景下的内存管理。
"""

import logging
import threading
import weakref
from typing import Optional, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """内存块"""
    tensor: torch.Tensor
    size: int
    device: int
    in_use: bool = False
    
    def __post_init__(self):
        self._ref_count = 1
    
    def acquire(self):
        """获取内存块"""
        self.in_use = True
        self._ref_count += 1
    
    def release(self):
        """释放内存块"""
        self._ref_count -= 1
        if self._ref_count <= 0:
            self.in_use = False
            return True
        return False


@dataclass
class MemoryPoolConfig:
    """内存池配置"""
    initial_pool_size: int = 1024**3  # 1GB初始池大小
    max_pool_size: int = 8 * 1024**3  # 8GB最大池大小
    block_alignment: int = 512  # 内存块对齐（字节）
    enable_gc: bool = True  # 启用垃圾回收
    gc_threshold: float = 0.9  # GC触发阈值
    device_id: int = 0


class GPUMemoryPool:
    """
    GPU内存池管理器
    
    管理GPU内存分配，减少频繁的cudaMalloc/cudaFree调用。
    支持多GPU场景。
    """
    
    def __init__(self, config: Optional[MemoryPoolConfig] = None, device_id: int = 0):
        """
        初始化内存池
        
        Args:
            config: 内存池配置
            device_id: GPU设备ID
        """
        self.config = config or MemoryPoolConfig()
        self.config.device_id = device_id
        self._device_id = device_id
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        self._pools: Dict[int, List[MemoryBlock]] = defaultdict(list)
        self._allocated: Dict[int, Set[int]] = defaultdict(set)  # 已分配的块ID
        self._block_id_counter = 0
        self._lock = threading.Lock()
        self._initialized = False
        
        # 统计信息
        self._stats = {
            "allocations": 0,
            "deallocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_allocated": 0,
            "total_freed": 0
        }
    
    def initialize(self) -> None:
        """初始化内存池"""
        if self._initialized:
            return
        
        with torch.cuda.device(self._device_id):
            # 预分配初始内存池
            initial_size = self.config.initial_pool_size
            try:
                # 尝试分配初始内存
                dummy = torch.empty(initial_size // 4, dtype=torch.float32, device=f"cuda:{self._device_id}")
                del dummy
                torch.cuda.empty_cache()
                logger.info(f"Memory pool initialized on device {self._device_id} with {initial_size / 1024**2:.1f}MB")
            except RuntimeError as e:
                logger.warning(f"Failed to pre-allocate initial pool: {e}")
        
        self._initialized = True
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[int] = None
    ) -> torch.Tensor:
        """
        分配内存
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            device: 设备ID
            
        Returns:
            分配的张量
        """
        device = device if device is not None else self._device_id
        
        if not self._initialized:
            self.initialize()
        
        # 计算所需内存大小
        element_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        required_size = num_elements * element_size
        
        with self._lock:
            # 尝试从池中获取合适的块
            block = self._find_free_block(required_size, device)
            
            if block is not None:
                # 使用池中的块
                block.acquire()
                self._allocated[device].add(id(block.tensor))
                self._stats["cache_hits"] += 1
                
                # 调整形状
                tensor = block.tensor.view(shape)
                tensor.fill_(0)  # 清零
                return tensor
            
            # 池中没有合适的块，分配新的
            self._stats["cache_misses"] += 1
        
        # 在锁外分配新内存
        with torch.cuda.device(device):
            try:
                tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{device}")
                
                with self._lock:
                    block = MemoryBlock(
                        tensor=tensor,
                        size=required_size,
                        device=device,
                        in_use=True
                    )
                    self._pools[device].append(block)
                    self._allocated[device].add(id(tensor))
                    self._stats["allocations"] += 1
                    self._stats["total_allocated"] += required_size
                
                return tensor
                
            except RuntimeError as e:
                # 内存不足，尝试清理并重新分配
                self._garbage_collect(device)
                
                with torch.cuda.device(device):
                    tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{device}")
                    return tensor
    
    def _find_free_block(self, size: int, device: int) -> Optional[MemoryBlock]:
        """
        查找空闲内存块
        
        Args:
            size: 所需大小
            device: 设备ID
            
        Returns:
            空闲块或None
        """
        best_fit = None
        best_fit_size = float('inf')
        
        for block in self._pools[device]:
            if not block.in_use and block.size >= size:
                if block.size < best_fit_size:
                    best_fit = block
                    best_fit_size = block.size
        
        return best_fit
    
    def free(self, tensor: torch.Tensor) -> None:
        """
        释放张量回内存池
        
        Args:
            tensor: 要释放的张量
        """
        if tensor is None:
            return
        
        tensor_id = id(tensor)
        device = tensor.device.index if tensor.is_cuda else self._device_id
        
        with self._lock:
            if tensor_id in self._allocated[device]:
                self._allocated[device].remove(tensor_id)
                self._stats["deallocations"] += 1
                
                # 查找对应的块并释放
                for block in self._pools[device]:
                    if id(block.tensor) == tensor_id:
                        if block.release():
                            self._stats["total_freed"] += block.size
                        break
    
    def _garbage_collect(self, device: Optional[int] = None) -> int:
        """
        垃圾回收
        
        Args:
            device: 设备ID，如果为None则清理所有设备
            
        Returns:
            释放的内存大小
        """
        freed = 0
        devices = [device] if device is not None else list(self._pools.keys())
        
        for dev in devices:
            with self._lock:
                # 移除未使用的块
                new_pool = []
                for block in self._pools[dev]:
                    if block.in_use:
                        new_pool.append(block)
                    else:
                        freed += block.size
                
                self._pools[dev] = new_pool
        
        if freed > 0:
            torch.cuda.empty_cache()
            logger.debug(f"Garbage collected {freed / 1024**2:.1f}MB")
        
        return freed
    
    def get_memory_stats(self, device: Optional[int] = None) -> Dict[str, Any]:
        """
        获取内存统计
        
        Args:
            device: 设备ID
            
        Returns:
            内存统计字典
        """
        stats = self._stats.copy()
        
        if torch.cuda.is_available():
            dev = device if device is not None else self._device_id
            stats["cuda_allocated"] = torch.cuda.memory_allocated(dev)
            stats["cuda_reserved"] = torch.cuda.memory_reserved(dev)
            stats["cuda_max_allocated"] = torch.cuda.max_memory_allocated(dev)
        
        # 池统计
        total_pool_size = 0
        used_pool_size = 0
        
        for dev, blocks in self._pools.items():
            if device is None or dev == device:
                for block in blocks:
                    total_pool_size += block.size
                    if block.in_use:
                        used_pool_size += block.size
        
        stats["pool_total_size"] = total_pool_size
        stats["pool_used_size"] = used_pool_size
        stats["pool_free_size"] = total_pool_size - used_pool_size
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "allocations": 0,
            "deallocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_allocated": 0,
            "total_freed": 0
        }
    
    def clear(self) -> None:
        """清空内存池"""
        with self._lock:
            self._pools.clear()
            self._allocated.clear()
        
        torch.cuda.empty_cache()
        logger.info("Memory pool cleared")
    
    def __del__(self):
        """析构函数"""
        self.clear()


class PooledTensor:
    """
    内存池张量包装器
    
    自动管理张量的生命周期，返回内存池时自动释放。
    """
    
    def __init__(self, tensor: torch.Tensor, pool: GPUMemoryPool):
        self.tensor = tensor
        self._pool = pool
        self._released = False
    
    def __enter__(self):
        return self.tensor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def release(self):
        """释放张量回内存池"""
        if not self._released:
            self._pool.free(self.tensor)
            self._released = True
    
    def __del__(self):
        self.release()


class MemoryPoolContext:
    """
    内存池上下文管理器
    
    在上下文中使用内存池管理张量分配。
    """
    
    def __init__(self, pool: Optional[GPUMemoryPool] = None, device_id: int = 0):
        self.pool = pool or GPUMemoryPool(device_id=device_id)
        self._original_empty = torch.empty
        self._original_zeros = torch.zeros
        self._original_ones = torch.ones
    
    def _pooled_empty(self, *args, **kwargs):
        """包装torch.empty"""
        if 'device' in kwargs and 'cuda' in str(kwargs['device']):
            return self.pool.allocate(args, kwargs.get('dtype', torch.float32))
        return self._original_empty(*args, **kwargs)
    
    def __enter__(self):
        self.pool.initialize()
        # 可以在这里hook torch函数
        return self.pool
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始函数
        pass


class MultiDeviceMemoryPool:
    """
    多设备内存池
    
    管理多个GPU设备的内存池。
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        初始化多设备内存池
        
        Args:
            device_ids: 设备ID列表，如果为None则使用所有可用GPU
        """
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        self._device_ids = device_ids
        self._pools: Dict[int, GPUMemoryPool] = {}
        
        for device_id in device_ids:
            self._pools[device_id] = GPUMemoryPool(device_id=device_id)
    
    def initialize(self) -> None:
        """初始化所有内存池"""
        for pool in self._pools.values():
            pool.initialize()
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[int] = None
    ) -> torch.Tensor:
        """
        在指定设备上分配内存
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            device: 设备ID
            
        Returns:
            分配的张量
        """
        if device is None:
            device = self._device_ids[0]
        
        if device not in self._pools:
            raise ValueError(f"Device {device} not managed by this pool")
        
        return self._pools[device].allocate(shape, dtype, device)
    
    def free(self, tensor: torch.Tensor) -> None:
        """
        释放张量
        
        Args:
            tensor: 要释放的张量
        """
        if not tensor.is_cuda:
            return
        
        device = tensor.device.index
        if device in self._pools:
            self._pools[device].free(tensor)
    
    def get_memory_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        获取所有设备的内存统计
        
        Returns:
            各设备的内存统计
        """
        return {device_id: pool.get_memory_stats() 
                for device_id, pool in self._pools.items()}
    
    def clear(self) -> None:
        """清空所有内存池"""
        for pool in self._pools.values():
            pool.clear()
    
    def garbage_collect(self) -> int:
        """
        对所有设备进行垃圾回收
        
        Returns:
            释放的总内存大小
        """
        total_freed = 0
        for pool in self._pools.values():
            total_freed += pool._garbage_collect()
        return total_freed


# 便捷的上下文管理器
@contextmanager
def memory_pool_context(device_id: int = 0):
    """
    内存池上下文管理器
    
    Args:
        device_id: 设备ID
    """
    pool = GPUMemoryPool(device_id=device_id)
    pool.initialize()
    try:
        yield pool
    finally:
        pool.clear()


# 全局内存池实例
_global_memory_pool: Optional[GPUMemoryPool] = None
_global_pool_lock = threading.Lock()


def get_global_memory_pool(device_id: int = 0) -> GPUMemoryPool:
    """
    获取全局内存池实例
    
    Args:
        device_id: 设备ID
        
    Returns:
        全局内存池
    """
    global _global_memory_pool
    
    with _global_pool_lock:
        if _global_memory_pool is None:
            _global_memory_pool = GPUMemoryPool(device_id=device_id)
            _global_memory_pool.initialize()
        return _global_memory_pool


def allocate_pooled(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: int = 0
) -> torch.Tensor:
    """
    从全局内存池分配张量
    
    Args:
        shape: 张量形状
        dtype: 数据类型
        device: 设备ID
        
    Returns:
        分配的张量
    """
    pool = get_global_memory_pool(device)
    return pool.allocate(shape, dtype, device)


def free_pooled(tensor: torch.Tensor) -> None:
    """
    释放张量回全局内存池
    
    Args:
        tensor: 要释放的张量
    """
    global _global_memory_pool
    if _global_memory_pool is not None:
        _global_memory_pool.free(tensor)


def get_memory_info(device: Optional[int] = None) -> Dict[str, Any]:
    """
    获取内存信息
    
    Args:
        device: 设备ID
        
    Returns:
        内存信息字典
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    device = device if device is not None else torch.cuda.current_device()
    
    return {
        "cuda_available": True,
        "device": device,
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "total_mb": torch.cuda.get_device_properties(device).total_memory / 1024**2
    }


def print_memory_summary(device: Optional[int] = None) -> None:
    """
    打印内存摘要
    
    Args:
        device: 设备ID
    """
    info = get_memory_info(device)
    
    if not info["cuda_available"]:
        print("CUDA not available")
        return
    
    print(f"GPU Memory Summary (Device {info['device']}):")
    print(f"  Allocated: {info['allocated_mb']:.1f} MB")
    print(f"  Reserved:  {info['reserved_mb']:.1f} MB")
    print(f"  Max Allocated: {info['max_allocated_mb']:.1f} MB")
    print(f"  Total: {info['total_mb']:.1f} MB")
    print(f"  Utilization: {info['allocated_mb'] / info['total_mb'] * 100:.1f}%")
