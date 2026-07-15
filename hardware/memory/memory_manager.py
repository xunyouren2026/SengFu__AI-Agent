"""
MemoryManager - 统一内存管理模块

统一管理GPU/CPU内存分配和释放，支持多GPU场景。
提供内存监控、自动优化和异常处理功能。

模块路径: hardware/memory/memory_manager.py
"""

import os
import sys
import json
import time
import gc
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import logging
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. MemoryManager will run in CPU-only mode.")


class MemoryType(Enum):
    """内存类型枚举"""
    CPU = auto()
    GPU = auto()
    PINNED = auto()
    SHARED = auto()


class MemoryStatus(Enum):
    """内存状态枚举"""
    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()
    OOM = auto()


@dataclass
class MemoryConfig:
    """内存管理配置"""
    # GPU内存配置
    gpu_memory_fraction: float = 0.95
    gpu_reserved_fraction: float = 0.05
    enable_memory_fraction: bool = True
    
    # CPU内存配置
    cpu_memory_limit_gb: Optional[float] = None
    enable_cpu_memory_limit: bool = False
    
    # 多GPU配置
    device_ids: List[int] = field(default_factory=lambda: [0])
    primary_device: int = 0
    
    # 内存优化配置
    enable_memory_optimization: bool = True
    auto_empty_cache: bool = True
    empty_cache_interval: int = 60
    
    # 监控配置
    enable_monitoring: bool = True
    monitor_interval: float = 5.0
    log_memory_stats: bool = True
    
    # OOM防护
    oom_threshold: float = 0.95
    enable_oom_protection: bool = True


@dataclass
class MemoryStats:
    """内存统计信息"""
    device_id: int
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    total_bytes: int
    peak_allocated_bytes: int
    timestamp: float
    
    @property
    def utilization_rate(self) -> float:
        """计算内存利用率"""
        if self.total_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes
    
    @property
    def available_bytes(self) -> int:
        """获取可用内存"""
        return self.total_bytes - self.allocated_bytes


class MemoryManager:
    """
    统一内存管理器
    
    负责管理GPU和CPU内存的分配、释放和监控。
    支持多GPU场景，提供内存优化和OOM防护功能。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Union[Dict, MemoryConfig]] = None):
        """
        初始化内存管理器
        
        Args:
            config: 配置字典或MemoryConfig对象
        """
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 解析配置
        if config is None:
            self.config = MemoryConfig()
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        else:
            self.config = config
        
        # 初始化状态
        self._initialized = False
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        
        # 内存统计
        self._memory_stats: Dict[int, List[MemoryStats]] = defaultdict(list)
        self._peak_memory: Dict[int, int] = {}
        self._allocation_history: List[Dict] = []
        
        # 回调函数
        self._oom_callbacks: List[Callable] = []
        self._warning_callbacks: List[Callable] = []
        
        # 设备信息
        self._device_count = 0
        self._available_devices: List[int] = []
        
        self.logger.info("MemoryManager initialized")
    
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
    
    def _parse_config(self, config_dict: Dict) -> MemoryConfig:
        """解析配置字典"""
        return MemoryConfig(
            gpu_memory_fraction=config_dict.get('gpu_memory_fraction', 0.95),
            gpu_reserved_fraction=config_dict.get('gpu_reserved_fraction', 0.05),
            enable_memory_fraction=config_dict.get('enable_memory_fraction', True),
            cpu_memory_limit_gb=config_dict.get('cpu_memory_limit_gb'),
            enable_cpu_memory_limit=config_dict.get('enable_cpu_memory_limit', False),
            device_ids=config_dict.get('device_ids', [0]),
            primary_device=config_dict.get('primary_device', 0),
            enable_memory_optimization=config_dict.get('enable_memory_optimization', True),
            auto_empty_cache=config_dict.get('auto_empty_cache', True),
            empty_cache_interval=config_dict.get('empty_cache_interval', 60),
            enable_monitoring=config_dict.get('enable_monitoring', True),
            monitor_interval=config_dict.get('monitor_interval', 5.0),
            log_memory_stats=config_dict.get('log_memory_stats', True),
            oom_threshold=config_dict.get('oom_threshold', 0.95),
            enable_oom_protection=config_dict.get('enable_oom_protection', True)
        )
    
    def initialize(self) -> bool:
        """
        初始化内存管理器
        
        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            self.logger.warning("MemoryManager already initialized")
            return True
        
        try:
            # 检测可用设备
            self._detect_devices()
            
            # 配置GPU内存限制
            if TORCH_AVAILABLE and self.config.enable_memory_fraction:
                self._configure_gpu_memory()
            
            # 启动监控线程
            if self.config.enable_monitoring:
                self._start_monitoring()
            
            self._initialized = True
            self.logger.info(f"MemoryManager initialized successfully. "
                           f"Available devices: {self._available_devices}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MemoryManager: {e}")
            return False
    
    def _detect_devices(self):
        """检测可用设备"""
        if not TORCH_AVAILABLE:
            self._device_count = 0
            self._available_devices = []
            return
        
        if torch.cuda.is_available():
            self._device_count = torch.cuda.device_count()
            self._available_devices = list(range(self._device_count))
            
            # 验证配置的device_ids
            valid_devices = [d for d in self.config.device_ids if d < self._device_count]
            if valid_devices:
                self.config.device_ids = valid_devices
            else:
                self.config.device_ids = [0] if self._device_count > 0 else []
            
            # 验证primary_device
            if self.config.primary_device >= self._device_count:
                self.config.primary_device = self.config.device_ids[0] if self.config.device_ids else 0
        else:
            self._device_count = 0
            self._available_devices = []
            self.config.device_ids = []
    
    def _configure_gpu_memory(self):
        """配置GPU内存限制"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        for device_id in self.config.device_ids:
            try:
                # 设置内存分配器
                torch.cuda.set_per_process_memory_fraction(
                    self.config.gpu_memory_fraction,
                    device_id
                )
                
                # 设置内存分配器的配置
                if hasattr(torch.cuda, 'memory_stats'):
                    torch.cuda.reset_peak_memory_stats(device_id)
                
                self.logger.info(
                    f"Configured GPU {device_id} memory fraction: "
                    f"{self.config.gpu_memory_fraction:.2%}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to configure GPU {device_id}: {e}")
    
    def _start_monitoring(self):
        """启动内存监控线程"""
        if self._monitoring:
            return
        
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self._monitoring = True
        self.logger.info("Memory monitoring started")
    
    def _monitor_loop(self):
        """监控循环"""
        last_empty_cache = time.time()
        
        while not self._stop_monitor.is_set():
            try:
                # 收集内存统计
                self._collect_memory_stats()
                
                # 检查内存状态
                self._check_memory_status()
                
                # 自动清空缓存
                if (self.config.auto_empty_cache and 
                    time.time() - last_empty_cache > self.config.empty_cache_interval):
                    self.empty_cache()
                    last_empty_cache = time.time()
                
                # 等待下一次检查
                self._stop_monitor.wait(self.config.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                self._stop_monitor.wait(1.0)
    
    def _collect_memory_stats(self):
        """收集内存统计信息"""
        if not TORCH_AVAILABLE:
            return
        
        timestamp = time.time()
        
        for device_id in self.config.device_ids:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_id)
                    
                    allocated = torch.cuda.memory_allocated(device_id)
                    reserved = torch.cuda.memory_reserved(device_id)
                    total = torch.cuda.get_device_properties(device_id).total_memory
                    free = total - allocated
                    
                    # 获取峰值内存
                    peak = torch.cuda.max_memory_allocated(device_id)
                    
                    stats = MemoryStats(
                        device_id=device_id,
                        allocated_bytes=allocated,
                        reserved_bytes=reserved,
                        free_bytes=free,
                        total_bytes=total,
                        peak_allocated_bytes=peak,
                        timestamp=timestamp
                    )
                    
                    self._memory_stats[device_id].append(stats)
                    
                    # 限制历史记录长度
                    max_history = 1000
                    if len(self._memory_stats[device_id]) > max_history:
                        self._memory_stats[device_id] = self._memory_stats[device_id][-max_history:]
                    
                    # 更新峰值
                    if device_id not in self._peak_memory or peak > self._peak_memory[device_id]:
                        self._peak_memory[device_id] = peak
                    
                    if self.config.log_memory_stats:
                        self.logger.debug(
                            f"GPU {device_id}: Allocated={allocated/1e9:.2f}GB, "
                            f"Reserved={reserved/1e9:.2f}GB, "
                            f"Utilization={stats.utilization_rate:.2%}"
                        )
                        
            except Exception as e:
                self.logger.warning(f"Failed to collect stats for GPU {device_id}: {e}")
    
    def _check_memory_status(self):
        """检查内存状态并触发相应处理"""
        for device_id in self.config.device_ids:
            if not self._memory_stats[device_id]:
                continue
            
            latest = self._memory_stats[device_id][-1]
            utilization = latest.utilization_rate
            
            if utilization >= self.config.oom_threshold:
                status = MemoryStatus.OOM
                self._handle_oom(device_id, latest)
            elif utilization >= 0.85:
                status = MemoryStatus.CRITICAL
                self._handle_critical(device_id, latest)
            elif utilization >= 0.70:
                status = MemoryStatus.WARNING
                self._handle_warning(device_id, latest)
            else:
                status = MemoryStatus.HEALTHY
            
            if status != MemoryStatus.HEALTHY and self.config.log_memory_stats:
                self.logger.warning(
                    f"GPU {device_id} memory status: {status.name} "
                    f"({utilization:.2%})"
                )
    
    def _handle_oom(self, device_id: int, stats: MemoryStats):
        """处理OOM情况"""
        self.logger.error(f"OOM detected on GPU {device_id}")
        
        # 触发OOM回调
        for callback in self._oom_callbacks:
            try:
                callback(device_id, stats)
            except Exception as e:
                self.logger.error(f"OOM callback error: {e}")
        
        if self.config.enable_oom_protection:
            self.empty_cache()
            gc.collect()
    
    def _handle_critical(self, device_id: int, stats: MemoryStats):
        """处理临界内存状态"""
        self.logger.warning(f"Critical memory on GPU {device_id}")
        
        # 尝试释放缓存
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
    
    def _handle_warning(self, device_id: int, stats: MemoryStats):
        """处理警告内存状态"""
        # 触发警告回调
        for callback in self._warning_callbacks:
            try:
                callback(device_id, stats)
            except Exception as e:
                self.logger.error(f"Warning callback error: {e}")
    
    def allocate(self, size_bytes: int, device: Optional[Union[str, int]] = None,
                 dtype: Any = None) -> Optional[Any]:
        """
        分配内存
        
        Args:
            size_bytes: 需要分配的字节数
            device: 目标设备 ('cpu', 'cuda', 或设备ID)
            dtype: 数据类型
            
        Returns:
            分配的内存对象，失败返回None
        """
        if not self._initialized:
            self.initialize()
        
        # 解析设备
        device_id = self._parse_device(device)
        
        # 检查可用内存
        if not self._check_available_memory(size_bytes, device_id):
            self.logger.warning(f"Insufficient memory for {size_bytes} bytes on device {device_id}")
            return None
        
        try:
            if device_id == 'cpu' or not TORCH_AVAILABLE:
                # CPU内存分配
                tensor = torch.empty(size_bytes // 4, dtype=dtype or torch.float32)
            else:
                # GPU内存分配
                tensor = torch.empty(
                    size_bytes // 4,
                    dtype=dtype or torch.float32,
                    device=f'cuda:{device_id}'
                )
            
            # 记录分配历史
            self._allocation_history.append({
                'size': size_bytes,
                'device': device_id,
                'timestamp': time.time()
            })
            
            return tensor
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error(f"OOM during allocation: {e}")
                self._handle_oom(device_id if device_id != 'cpu' else 0, None)
            raise
    
    def _parse_device(self, device: Optional[Union[str, int]]) -> Union[str, int]:
        """解析设备参数"""
        if device is None:
            return self.config.primary_device if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        if isinstance(device, str):
            if device.lower() == 'cpu':
                return 'cpu'
            elif device.lower().startswith('cuda'):
                try:
                    return int(device.split(':')[1]) if ':' in device else 0
                except (IndexError, ValueError):
                    return 0
        
        return device
    
    def _check_available_memory(self, size_bytes: int, device: Union[str, int]) -> bool:
        """检查是否有足够的可用内存"""
        if device == 'cpu' or not TORCH_AVAILABLE:
            # CPU内存检查
            import psutil
            available = psutil.virtual_memory().available
            return available > size_bytes * 1.2  # 预留20%缓冲
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            free_memory = torch.cuda.memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            available = total - free_memory
            return available > size_bytes * 1.1  # 预留10%缓冲
        
        return False
    
    def free(self, tensor: Any) -> bool:
        """
        释放内存
        
        Args:
            tensor: 要释放的张量
            
        Returns:
            bool: 释放是否成功
        """
        try:
            if tensor is not None and hasattr(tensor, 'detach'):
                tensor.detach()
            if tensor is not None and hasattr(tensor, 'cpu'):
                tensor.cpu()
            del tensor
            return True
        except Exception as e:
            self.logger.warning(f"Error freeing memory: {e}")
            return False
    
    def empty_cache(self):
        """清空GPU缓存"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for device_id in self.config.device_ids:
                try:
                    torch.cuda.synchronize(device_id)
                except:
                    pass
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache emptied")
    
    def get_memory_stats(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """
        获取内存统计信息
        
        Args:
            device_id: 指定设备ID，None则返回所有设备
            
        Returns:
            内存统计字典
        """
        if not TORCH_AVAILABLE:
            return {}
        
        stats = {}
        devices = [device_id] if device_id is not None else self.config.device_ids
        
        for dev in devices:
            if dev >= self._device_count:
                continue
                
            try:
                torch.cuda.synchronize(dev)
                stats[dev] = {
                    'allocated': torch.cuda.memory_allocated(dev),
                    'reserved': torch.cuda.memory_reserved(dev),
                    'total': torch.cuda.get_device_properties(dev).total_memory,
                    'peak': torch.cuda.max_memory_allocated(dev),
                    'utilization': torch.cuda.memory_allocated(dev) / 
                                  torch.cuda.get_device_properties(dev).total_memory
                }
            except Exception as e:
                self.logger.warning(f"Failed to get stats for GPU {dev}: {e}")
        
        return stats
    
    def get_best_device(self, size_bytes: int = 0) -> int:
        """
        获取最适合分配内存的设备
        
        Args:
            size_bytes: 需要的内存大小
            
        Returns:
            最佳设备ID
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 'cpu'
        
        best_device = self.config.primary_device
        best_free = 0
        
        for device_id in self.config.device_ids:
            try:
                torch.cuda.synchronize(device_id)
                free = (torch.cuda.get_device_properties(device_id).total_memory - 
                       torch.cuda.memory_allocated(device_id))
                
                if free > best_free:
                    best_free = free
                    best_device = device_id
                    
            except Exception:
                continue
        
        return best_device
    
    def register_oom_callback(self, callback: Callable):
        """注册OOM回调函数"""
        self._oom_callbacks.append(callback)
    
    def register_warning_callback(self, callback: Callable):
        """注册警告回调函数"""
        self._warning_callbacks.append(callback)
    
    def shutdown(self):
        """关闭内存管理器"""
        self.logger.info("Shutting down MemoryManager...")
        
        # 停止监控
        if self._monitoring:
            self._stop_monitor.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            self._monitoring = False
        
        # 清空缓存
        self.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        self._initialized = False
        self.logger.info("MemoryManager shutdown complete")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, '_initialized') and self._initialized:
            self.shutdown()
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False


# 便捷函数
def get_memory_manager(config: Optional[Dict] = None) -> MemoryManager:
    """获取内存管理器实例"""
    return MemoryManager(config)


def get_gpu_memory_summary(device_id: Optional[int] = None) -> str:
    """获取GPU内存摘要"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return "GPU not available"
    
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    torch.cuda.synchronize(device_id)
    allocated = torch.cuda.memory_allocated(device_id) / 1e9
    reserved = torch.cuda.memory_reserved(device_id) / 1e9
    total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    
    return (f"GPU {device_id}: Allocated={allocated:.2f}GB, "
            f"Reserved={reserved:.2f}GB, Total={total:.2f}GB")
