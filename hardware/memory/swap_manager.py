"""
SwapManager - CPU-GPU内存交换管理模块

管理CPU和GPU之间的内存交换，支持异步传输和预取。
提供高效的内存分层管理。

模块路径: hardware/memory/swap_manager.py
"""

import os
import sys
import time
import threading
import asyncio
from collections import deque, OrderedDict
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import warnings
import weakref

try:
    import torch
    from torch.cuda import Stream
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. SwapManager will run in limited mode.")


class SwapStatus(Enum):
    """交换状态"""
    CPU = auto()
    GPU = auto()
    SWAPPING_IN = auto()
    SWAPPING_OUT = auto()
    PINNED = auto()


class SwapStrategy(Enum):
    """交换策略"""
    LRU = auto()          # 最近最少使用
    LFU = auto()          # 最少使用频率
    FIFO = auto()         # 先进先出
    PRIORITY = auto()     # 优先级
    ADAPTIVE = auto()     # 自适应


@dataclass
class SwapConfig:
    """交换管理器配置"""
    # 容量配置
    max_gpu_memory: int = 8 * 1024 * 1024 * 1024  # 8GB
    max_cpu_memory: int = 32 * 1024 * 1024 * 1024  # 32GB
    
    # 阈值配置
    gpu_high_watermark: float = 0.85  # GPU内存高水位线
    gpu_low_watermark: float = 0.60   # GPU内存低水位线
    
    # 策略配置
    strategy: SwapStrategy = SwapStrategy.LRU
    prefetch_enabled: bool = True
    prefetch_lookahead: int = 3
    
    # 异步配置
    async_transfer: bool = True
    num_transfer_streams: int = 2
    
    # 性能配置
    pin_memory: bool = True
    non_blocking: bool = True
    
    # 多GPU配置
    device_ids: List[int] = field(default_factory=lambda: [0])
    primary_device: int = 0


@dataclass
class SwappedTensor:
    """被交换的张量信息"""
    tensor_id: int
    cpu_tensor: Optional[torch.Tensor] = None
    gpu_tensor: Optional[torch.Tensor] = None
    status: SwapStatus = SwapStatus.CPU
    size_bytes: int = 0
    device: int = 0
    
    # 统计信息
    access_count: int = 0
    last_access_time: float = 0.0
    swap_in_count: int = 0
    swap_out_count: int = 0
    
    # 优先级
    priority: int = 0
    pinned: bool = False
    
    def __post_init__(self):
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()
    
    @property
    def is_on_gpu(self) -> bool:
        return self.status == SwapStatus.GPU
    
    @property
    def is_on_cpu(self) -> bool:
        return self.status == SwapStatus.CPU
    
    def mark_accessed(self):
        """标记访问"""
        self.access_count += 1
        self.last_access_time = time.time()


class SwapManager:
    """
    CPU-GPU内存交换管理器
    
    提供高效的内存分层管理：
    - 自动GPU-CPU内存交换
    - 异步传输支持
    - 预取优化
    - 多种交换策略
    """
    
    def __init__(self, config: Optional[Union[Dict, SwapConfig]] = None):
        """
        初始化交换管理器
        
        Args:
            config: 配置字典或SwapConfig对象
        """
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 解析配置
        if config is None:
            self.config = SwapConfig()
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        else:
            self.config = config
        
        # 状态
        self._initialized = False
        self._monitoring = False
        
        # 张量存储
        self._tensors: Dict[int, SwappedTensor] = {}
        self._access_order: deque = deque()  # 用于LRU
        
        # 内存统计
        self._gpu_memory_used: Dict[int, int] = defaultdict(int)
        self._cpu_memory_used: int = 0
        
        # 异步传输
        self._transfer_streams: Dict[int, Any] = {}
        self._pending_transfers: Dict[int, Any] = {}
        
        # 预取队列
        self._prefetch_queue: deque = deque()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()
        
        # 回调
        self._swap_in_callbacks: List[Callable] = []
        self._swap_out_callbacks: List[Callable] = []
        
        # 线程安全
        self._lock = threading.RLock()
        
        self.logger.info("SwapManager initialized")
    
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
    
    def _parse_config(self, config_dict: Dict) -> SwapConfig:
        """解析配置字典"""
        strategy = SwapStrategy[
            config_dict.get('strategy', 'LRU').upper()
        ]
        
        return SwapConfig(
            max_gpu_memory=config_dict.get('max_gpu_memory', 8 * 1024 * 1024 * 1024),
            max_cpu_memory=config_dict.get('max_cpu_memory', 32 * 1024 * 1024 * 1024),
            gpu_high_watermark=config_dict.get('gpu_high_watermark', 0.85),
            gpu_low_watermark=config_dict.get('gpu_low_watermark', 0.60),
            strategy=strategy,
            prefetch_enabled=config_dict.get('prefetch_enabled', True),
            prefetch_lookahead=config_dict.get('prefetch_lookahead', 3),
            async_transfer=config_dict.get('async_transfer', True),
            num_transfer_streams=config_dict.get('num_transfer_streams', 2),
            pin_memory=config_dict.get('pin_memory', True),
            non_blocking=config_dict.get('non_blocking', True),
            device_ids=config_dict.get('device_ids', [0]),
            primary_device=config_dict.get('primary_device', 0)
        )
    
    def initialize(self):
        """初始化交换管理器"""
        if self._initialized:
            return
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, SwapManager disabled")
            return
        
        # 创建传输流
        if self.config.async_transfer and torch.cuda.is_available():
            for device_id in self.config.device_ids:
                self._transfer_streams[device_id] = [
                    torch.cuda.Stream(device=device_id)
                    for _ in range(self.config.num_transfer_streams)
                ]
        
        # 启动预取线程
        if self.config.prefetch_enabled:
            self._start_prefetch()
        
        self._initialized = True
        self.logger.info("SwapManager initialized successfully")
    
    def _start_prefetch(self):
        """启动预取线程"""
        if self._prefetch_thread is not None:
            return
        
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True
        )
        self._prefetch_thread.start()
        self.logger.info("Prefetch thread started")
    
    def _prefetch_loop(self):
        """预取循环"""
        while not self._stop_prefetch.is_set():
            try:
                if self._prefetch_queue:
                    with self._lock:
                        if self._prefetch_queue:
                            tensor_id = self._prefetch_queue.popleft()
                            self._prefetch_tensor(tensor_id)
                
                self._stop_prefetch.wait(0.1)
            except Exception as e:
                self.logger.error(f"Error in prefetch loop: {e}")
                self._stop_prefetch.wait(1.0)
    
    def _prefetch_tensor(self, tensor_id: int):
        """预取张量到GPU"""
        if tensor_id not in self._tensors:
            return
        
        tensor_info = self._tensors[tensor_id]
        
        # 检查是否已经在GPU
        if tensor_info.is_on_gpu:
            return
        
        # 检查GPU内存
        if not self._can_allocate_on_gpu(tensor_info.size_bytes, tensor_info.device):
            return
        
        # 执行预取
        try:
            self._swap_in(tensor_id, async_transfer=True)
        except Exception as e:
            self.logger.warning(f"Prefetch failed for tensor {tensor_id}: {e}")
    
    def register_tensor(self, tensor: torch.Tensor, device: int = None,
                       priority: int = 0, pin: bool = False) -> int:
        """
        注册张量到交换管理器
        
        Args:
            tensor: 要管理的张量
            device: 目标GPU设备
            priority: 优先级（越高越不容易被交换）
            pin: 是否固定（不被交换）
            
        Returns:
            张量ID
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        if device is None:
            device = self.config.primary_device
        
        tensor_id = id(tensor)
        size_bytes = tensor.numel() * tensor.element_size()
        
        with self._lock:
            # 确定初始位置
            if tensor.device.type == 'cuda':
                status = SwapStatus.GPU
                self._gpu_memory_used[device] += size_bytes
                gpu_tensor = tensor
                cpu_tensor = None
            else:
                status = SwapStatus.CPU
                self._cpu_memory_used += size_bytes
                gpu_tensor = None
                cpu_tensor = tensor
            
            swapped_tensor = SwappedTensor(
                tensor_id=tensor_id,
                cpu_tensor=cpu_tensor,
                gpu_tensor=gpu_tensor,
                status=status,
                size_bytes=size_bytes,
                device=device,
                priority=priority,
                pinned=pin
            )
            
            self._tensors[tensor_id] = swapped_tensor
            self._access_order.append(tensor_id)
            
            self.logger.debug(f"Registered tensor {tensor_id} ({size_bytes / 1e6:.2f} MB) on {status.name}")
            
            return tensor_id
    
    def access(self, tensor_id: int, async_transfer: bool = False) -> Optional[torch.Tensor]:
        """
        访问张量，确保在GPU上
        
        Args:
            tensor_id: 张量ID
            async_transfer: 是否使用异步传输
            
        Returns:
            GPU上的张量
        """
        with self._lock:
            if tensor_id not in self._tensors:
                return None
            
            tensor_info = self._tensors[tensor_id]
            tensor_info.mark_accessed()
            
            # 更新访问顺序
            if tensor_id in self._access_order:
                self._access_order.remove(tensor_id)
            self._access_order.append(tensor_id)
            
            # 如果已经在GPU，直接返回
            if tensor_info.is_on_gpu:
                return tensor_info.gpu_tensor
            
            # 检查是否正在传输
            if tensor_info.status == SwapStatus.SWAPPING_IN:
                # 等待传输完成
                if tensor_id in self._pending_transfers:
                    self._wait_for_transfer(tensor_id)
                return tensor_info.gpu_tensor
            
            # 需要交换到GPU
            self._ensure_gpu_memory(tensor_info.size_bytes, tensor_info.device)
            self._swap_in(tensor_id, async_transfer)
            
            return tensor_info.gpu_tensor
    
    def _swap_in(self, tensor_id: int, async_transfer: bool = False):
        """
        将张量交换到GPU
        
        Args:
            tensor_id: 张量ID
            async_transfer: 是否异步传输
        """
        tensor_info = self._tensors[tensor_id]
        
        if tensor_info.cpu_tensor is None:
            raise RuntimeError(f"CPU tensor not found for {tensor_id}")
        
        tensor_info.status = SwapStatus.SWAPPING_IN
        
        try:
            # 执行传输
            if async_transfer and self.config.async_transfer:
                # 异步传输
                stream = self._get_transfer_stream(tensor_info.device)
                with torch.cuda.stream(stream):
                    gpu_tensor = tensor_info.cpu_tensor.cuda(
                        device=tensor_info.device,
                        non_blocking=self.config.non_blocking
                    )
                
                self._pending_transfers[tensor_id] = stream
            else:
                # 同步传输
                gpu_tensor = tensor_info.cpu_tensor.cuda(device=tensor_info.device)
            
            tensor_info.gpu_tensor = gpu_tensor
            tensor_info.status = SwapStatus.GPU
            tensor_info.swap_in_count += 1
            
            # 更新内存统计
            self._gpu_memory_used[tensor_info.device] += tensor_info.size_bytes
            self._cpu_memory_used -= tensor_info.size_bytes
            
            # 触发回调
            self._trigger_swap_in_callbacks(tensor_id)
            
            self.logger.debug(f"Swapped in tensor {tensor_id} ({tensor_info.size_bytes / 1e6:.2f} MB)")
            
        except Exception as e:
            tensor_info.status = SwapStatus.CPU
            raise RuntimeError(f"Failed to swap in tensor {tensor_id}: {e}")
    
    def _swap_out(self, tensor_id: int, async_transfer: bool = False):
        """
        将张量交换到CPU
        
        Args:
            tensor_id: 张量ID
            async_transfer: 是否异步传输
        """
        tensor_info = self._tensors[tensor_id]
        
        if tensor_info.gpu_tensor is None:
            return
        
        if tensor_info.pinned:
            self.logger.debug(f"Skipping swap out for pinned tensor {tensor_id}")
            return
        
        tensor_info.status = SwapStatus.SWAPPING_OUT
        
        try:
            # 执行传输
            cpu_tensor = tensor_info.gpu_tensor.cpu()
            
            # 如果使用固定内存
            if self.config.pin_memory:
                cpu_tensor = cpu_tensor.pin_memory()
            
            tensor_info.cpu_tensor = cpu_tensor
            tensor_info.gpu_tensor = None
            tensor_info.status = SwapStatus.CPU
            tensor_info.swap_out_count += 1
            
            # 更新内存统计
            self._gpu_memory_used[tensor_info.device] -= tensor_info.size_bytes
            self._cpu_memory_used += tensor_info.size_bytes
            
            # 触发回调
            self._trigger_swap_out_callbacks(tensor_id)
            
            self.logger.debug(f"Swapped out tensor {tensor_id} ({tensor_info.size_bytes / 1e6:.2f} MB)")
            
        except Exception as e:
            tensor_info.status = SwapStatus.GPU
            raise RuntimeError(f"Failed to swap out tensor {tensor_id}: {e}")
    
    def _ensure_gpu_memory(self, size_bytes: int, device: int):
        """
        确保GPU有足够内存
        
        Args:
            size_bytes: 需要的字节数
            device: 设备ID
        """
        # 检查当前GPU内存使用
        current_usage = self._gpu_memory_used.get(device, 0)
        max_usage = self.config.max_gpu_memory
        
        if current_usage + size_bytes <= max_usage * self.config.gpu_high_watermark:
            return
        
        # 需要释放内存
        self._evict_to_target(device, max_usage * self.config.gpu_low_watermark)
    
    def _evict_to_target(self, device: int, target_usage: float):
        """
        将GPU内存使用降至目标值
        
        Args:
            device: 设备ID
            target_usage: 目标内存使用量
        """
        current_usage = self._gpu_memory_used.get(device, 0)
        
        if current_usage <= target_usage:
            return
        
        # 获取可交换的张量
        evictable = [
            (tid, info) for tid, info in self._tensors.items()
            if info.device == device and info.is_on_gpu and not info.pinned
        ]
        
        # 根据策略排序
        if self.config.strategy == SwapStrategy.LRU:
            evictable.sort(key=lambda x: x[1].last_access_time)
        elif self.config.strategy == SwapStrategy.LFU:
            evictable.sort(key=lambda x: x[1].access_count)
        elif self.config.strategy == SwapStrategy.PRIORITY:
            evictable.sort(key=lambda x: x[1].priority)
        
        # 交换出张量直到达到目标
        for tensor_id, tensor_info in evictable:
            if current_usage <= target_usage:
                break
            
            self._swap_out(tensor_id)
            current_usage -= tensor_info.size_bytes
    
    def _can_allocate_on_gpu(self, size_bytes: int, device: int) -> bool:
        """检查是否可以在GPU上分配"""
        current_usage = self._gpu_memory_used.get(device, 0)
        max_usage = self.config.max_gpu_memory
        
        return current_usage + size_bytes <= max_usage * self.config.gpu_high_watermark
    
    def _get_transfer_stream(self, device: int) -> Any:
        """获取传输流"""
        if device not in self._transfer_streams:
            return None
        
        streams = self._transfer_streams[device]
        # 轮询选择流
        idx = int(time.time() * 1000) % len(streams)
        return streams[idx]
    
    def _wait_for_transfer(self, tensor_id: int):
        """等待传输完成"""
        if tensor_id not in self._pending_transfers:
            return
        
        stream = self._pending_transfers.pop(tensor_id)
        if stream is not None:
            stream.synchronize()
    
    def prefetch(self, tensor_ids: List[int]):
        """
        预取张量到GPU
        
        Args:
            tensor_ids: 张量ID列表
        """
        if not self.config.prefetch_enabled:
            return
        
        with self._lock:
            for tensor_id in tensor_ids:
                if tensor_id not in self._prefetch_queue:
                    self._prefetch_queue.append(tensor_id)
    
    def unregister_tensor(self, tensor_id: int):
        """
        注销张量
        
        Args:
            tensor_id: 张量ID
        """
        with self._lock:
            if tensor_id not in self._tensors:
                return
            
            tensor_info = self._tensors[tensor_id]
            
            # 更新内存统计
            if tensor_info.is_on_gpu:
                self._gpu_memory_used[tensor_info.device] -= tensor_info.size_bytes
            else:
                self._cpu_memory_used -= tensor_info.size_bytes
            
            # 从队列中移除
            if tensor_id in self._access_order:
                self._access_order.remove(tensor_id)
            
            # 删除张量信息
            del self._tensors[tensor_id]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计
        
        Returns:
            内存统计字典
        """
        with self._lock:
            gpu_tensors = sum(1 for info in self._tensors.values() if info.is_on_gpu)
            cpu_tensors = sum(1 for info in self._tensors.values() if info.is_on_cpu)
            
            return {
                'total_tensors': len(self._tensors),
                'gpu_tensors': gpu_tensors,
                'cpu_tensors': cpu_tensors,
                'gpu_memory_used': dict(self._gpu_memory_used),
                'cpu_memory_used': self._cpu_memory_used,
                'gpu_memory_limit': self.config.max_gpu_memory,
                'cpu_memory_limit': self.config.max_cpu_memory,
                'prefetch_queue_size': len(self._prefetch_queue)
            }
    
    def get_tensor_stats(self, tensor_id: int) -> Optional[Dict]:
        """
        获取张量统计
        
        Args:
            tensor_id: 张量ID
            
        Returns:
            张量统计字典
        """
        with self._lock:
            if tensor_id not in self._tensors:
                return None
            
            info = self._tensors[tensor_id]
            return {
                'tensor_id': tensor_id,
                'status': info.status.name,
                'size_bytes': info.size_bytes,
                'size_mb': info.size_bytes / 1e6,
                'device': info.device,
                'access_count': info.access_count,
                'swap_in_count': info.swap_in_count,
                'swap_out_count': info.swap_out_count,
                'priority': info.priority,
                'pinned': info.pinned
            }
    
    def _trigger_swap_in_callbacks(self, tensor_id: int):
        """触发交换入回调"""
        for callback in self._swap_in_callbacks:
            try:
                callback(tensor_id)
            except Exception as e:
                self.logger.error(f"Swap in callback error: {e}")
    
    def _trigger_swap_out_callbacks(self, tensor_id: int):
        """触发交换出回调"""
        for callback in self._swap_out_callbacks:
            try:
                callback(tensor_id)
            except Exception as e:
                self.logger.error(f"Swap out callback error: {e}")
    
    def register_swap_in_callback(self, callback: Callable):
        """注册交换入回调"""
        self._swap_in_callbacks.append(callback)
    
    def register_swap_out_callback(self, callback: Callable):
        """注册交换出回调"""
        self._swap_out_callbacks.append(callback)
    
    def synchronize(self):
        """同步所有传输"""
        if not TORCH_AVAILABLE:
            return
        
        # 等待所有待处理的传输
        for tensor_id in list(self._pending_transfers.keys()):
            self._wait_for_transfer(tensor_id)
        
        # 同步所有设备
        for device_id in self.config.device_ids:
            try:
                torch.cuda.synchronize(device_id)
            except:
                pass
    
    def shutdown(self):
        """关闭交换管理器"""
        self.logger.info("Shutting down SwapManager...")
        
        # 停止预取
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._stop_prefetch.set()
            self._prefetch_thread.join(timeout=5.0)
        
        # 同步所有传输
        self.synchronize()
        
        # 清理所有张量
        with self._lock:
            self._tensors.clear()
            self._access_order.clear()
            self._prefetch_queue.clear()
        
        self._initialized = False
        self.logger.info("SwapManager shutdown complete")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False


# 便捷函数
def create_swap_manager(config: Optional[Dict] = None) -> SwapManager:
    """创建交换管理器"""
    return SwapManager(config)


def auto_swap(tensor: torch.Tensor, manager: Optional[SwapManager] = None) -> torch.Tensor:
    """
    自动交换装饰器
    
    Args:
        tensor: 输入张量
        manager: 交换管理器，None则创建新实例
        
    Returns:
        GPU上的张量
    """
    if manager is None:
        manager = SwapManager()
        manager.initialize()
    
    tensor_id = manager.register_tensor(tensor)
    return manager.access(tensor_id)
