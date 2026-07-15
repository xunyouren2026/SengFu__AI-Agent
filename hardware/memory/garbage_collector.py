"""
GarbageCollector - 垃圾回收模块

提供智能垃圾回收触发和优化功能。
支持多种回收策略和内存碎片整理。

模块路径: hardware/memory/garbage_collector.py
"""

import os
import sys
import gc
import time
import threading
import weakref
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GarbageCollector will run in limited mode.")


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class GCStrategy(Enum):
    """垃圾回收策略"""
    PASSIVE = auto()      # 被动回收，仅在内存不足时触发
    AGGRESSIVE = auto()   # 积极回收，定期执行
    ADAPTIVE = auto()     # 自适应回收，根据内存压力调整
    MANUAL = auto()       # 手动回收


class GCPhase(Enum):
    """垃圾回收阶段"""
    IDLE = auto()
    MARK = auto()
    SWEEP = auto()
    COMPACT = auto()
    COMPLETE = auto()


@dataclass
class GCConfig:
    """垃圾回收配置"""
    # 策略配置
    strategy: GCStrategy = GCStrategy.ADAPTIVE
    enabled: bool = True
    
    # 触发阈值
    memory_threshold: float = 0.80  # 内存使用率达到80%触发
    cpu_memory_threshold: float = 0.85
    
    # 定时配置
    gc_interval: float = 60.0  # 最小回收间隔（秒）
    idle_timeout: float = 300.0  # 空闲超时（秒）
    
    # 优化配置
    enable_compaction: bool = True
    compaction_threshold: float = 0.3  # 碎片率超过30%时整理
    
    # 多GPU配置
    device_ids: List[int] = field(default_factory=lambda: [0])
    
    # 调试配置
    verbose: bool = False
    track_objects: bool = False


@dataclass
class GCStats:
    """垃圾回收统计"""
    total_collections: int = 0
    total_freed_bytes: int = 0
    total_time_ms: float = 0.0
    last_collection_time: Optional[float] = None
    avg_collection_time_ms: float = 0.0
    peak_memory_before: int = 0
    peak_memory_after: int = 0
    
    @property
    def efficiency(self) -> float:
        """回收效率（MB/s）"""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.total_freed_bytes / 1e6) / (self.total_time_ms / 1000)


class GarbageCollector:
    """
    垃圾回收器
    
    提供智能的垃圾回收功能：
    - 多种回收策略（被动、积极、自适应）
    - 内存碎片整理
    - GPU缓存管理
    - 对象生命周期追踪
    """
    
    def __init__(self, config: Optional[Union[Dict, GCConfig]] = None):
        """
        初始化垃圾回收器
        
        Args:
            config: 配置字典或GCConfig对象
        """
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 解析配置
        if config is None:
            self.config = GCConfig()
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        else:
            self.config = config
        
        # 状态
        self._initialized = False
        self._running = False
        self._current_phase = GCPhase.IDLE
        
        # 统计
        self._stats = GCStats()
        self._device_stats: Dict[Union[str, int], GCStats] = defaultdict(GCStats)
        
        # 监控
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 对象追踪
        self._tracked_objects: Dict[int, weakref.ref] = {}
        self._object_callbacks: List[Callable] = []
        
        # 回调
        self._pre_gc_callbacks: List[Callable] = []
        self._post_gc_callbacks: List[Callable] = []
        
        # 线程安全
        self._lock = threading.RLock()
        
        self.logger.info("GarbageCollector initialized")
    
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
    
    def _parse_config(self, config_dict: Dict) -> GCConfig:
        """解析配置字典"""
        strategy = GCStrategy[
            config_dict.get('strategy', 'ADAPTIVE').upper()
        ]
        
        return GCConfig(
            strategy=strategy,
            enabled=config_dict.get('enabled', True),
            memory_threshold=config_dict.get('memory_threshold', 0.80),
            cpu_memory_threshold=config_dict.get('cpu_memory_threshold', 0.85),
            gc_interval=config_dict.get('gc_interval', 60.0),
            idle_timeout=config_dict.get('idle_timeout', 300.0),
            enable_compaction=config_dict.get('enable_compaction', True),
            compaction_threshold=config_dict.get('compaction_threshold', 0.3),
            device_ids=config_dict.get('device_ids', [0]),
            verbose=config_dict.get('verbose', False),
            track_objects=config_dict.get('track_objects', False)
        )
    
    def initialize(self):
        """初始化垃圾回收器"""
        if self._initialized:
            return
        
        if not self.config.enabled:
            self.logger.info("GarbageCollector is disabled")
            return
        
        # 启动监控线程（如果是自适应策略）
        if self.config.strategy == GCStrategy.ADAPTIVE:
            self._start_monitoring()
        
        self._initialized = True
        self.logger.info(f"GarbageCollector initialized with strategy: {self.config.strategy.name}")
    
    def _start_monitoring(self):
        """启动监控线程"""
        if self._running:
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self._running = True
        self.logger.info("GC monitoring started")
    
    def _monitor_loop(self):
        """监控循环"""
        last_gc_time = 0
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # 检查是否需要回收
                if current_time - last_gc_time >= self.config.gc_interval:
                    memory_pressure = self._check_memory_pressure()
                    
                    if memory_pressure >= self.config.memory_threshold:
                        self.logger.debug(f"Memory pressure {memory_pressure:.2%}, triggering GC")
                        self.collect()
                        last_gc_time = current_time
                
                self._stop_event.wait(5.0)
                
            except Exception as e:
                self.logger.error(f"Error in GC monitor loop: {e}")
                self._stop_event.wait(1.0)
    
    def _check_memory_pressure(self) -> float:
        """
        检查内存压力
        
        Returns:
            内存压力值 (0-1)
        """
        max_pressure = 0.0
        
        # 检查GPU内存
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for device_id in self.config.device_ids:
                try:
                    torch.cuda.synchronize(device_id)
                    allocated = torch.cuda.memory_allocated(device_id)
                    total = torch.cuda.get_device_properties(device_id).total_memory
                    pressure = allocated / total
                    max_pressure = max(max_pressure, pressure)
                except:
                    continue
        
        # 检查CPU内存
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                cpu_pressure = mem.percent / 100.0
                max_pressure = max(max_pressure, cpu_pressure)
            except:
                pass
        
        return max_pressure
    
    def collect(self, device: Optional[Union[str, int]] = None,
                force: bool = False) -> Dict[str, Any]:
        """
        执行垃圾回收
        
        Args:
            device: 指定设备，None则回收所有
            force: 是否强制回收
            
        Returns:
            回收结果统计
        """
        if not self._initialized and not force:
            self.initialize()
        
        with self._lock:
            start_time = time.time()
            
            # 触发前置回调
            self._trigger_pre_gc_callbacks()
            
            # 设置阶段
            self._current_phase = GCPhase.MARK
            
            # 收集回收前内存状态
            memory_before = self._get_memory_stats()
            
            # 执行Python垃圾回收
            gc.collect()
            
            # GPU内存回收
            self._current_phase = GCPhase.SWEEP
            gpu_freed = self._collect_gpu_memory(device)
            
            # 内存整理
            if self.config.enable_compaction:
                self._current_phase = GCPhase.COMPACT
                self._compact_memory(device)
            
            # 收集回收后内存状态
            memory_after = self._get_memory_stats()
            
            # 计算释放的内存
            freed_bytes = self._calculate_freed_memory(memory_before, memory_after)
            
            # 更新统计
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_stats(freed_bytes, elapsed_ms, memory_before, memory_after)
            
            self._current_phase = GCPhase.COMPLETE
            
            # 触发后置回调
            self._trigger_post_gc_callbacks(freed_bytes, elapsed_ms)
            
            result = {
                'freed_bytes': freed_bytes,
                'freed_mb': freed_bytes / 1e6,
                'elapsed_ms': elapsed_ms,
                'phase': self._current_phase.name,
                'memory_before': memory_before,
                'memory_after': memory_after
            }
            
            if self.config.verbose:
                self.logger.info(f"GC completed: freed {result['freed_mb']:.2f} MB in {elapsed_ms:.2f} ms")
            
            self._current_phase = GCPhase.IDLE
            return result
    
    def _get_memory_stats(self) -> Dict[Union[str, int], int]:
        """获取内存统计"""
        stats = {}
        
        # GPU内存
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for device_id in self.config.device_ids:
                try:
                    torch.cuda.synchronize(device_id)
                    stats[device_id] = torch.cuda.memory_allocated(device_id)
                except:
                    stats[device_id] = 0
        
        # CPU内存
        if PSUTIL_AVAILABLE:
            try:
                stats['cpu'] = psutil.virtual_memory().used
            except:
                stats['cpu'] = 0
        
        return stats
    
    def _collect_gpu_memory(self, device: Optional[Union[str, int]] = None) -> int:
        """
        回收GPU内存
        
        Args:
            device: 指定设备，None则回收所有
            
        Returns:
            释放的字节数
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0
        
        freed = 0
        devices = [device] if device is not None else self.config.device_ids
        
        for dev in devices:
            try:
                torch.cuda.synchronize(dev)
                before = torch.cuda.memory_allocated(dev)
                torch.cuda.empty_cache()
                after = torch.cuda.memory_allocated(dev)
                freed += max(0, before - after)
            except Exception as e:
                self.logger.warning(f"Failed to collect GPU memory on device {dev}: {e}")
        
        return freed
    
    def _compact_memory(self, device: Optional[Union[str, int]] = None):
        """
        整理内存碎片
        
        Args:
            device: 指定设备
        """
        # 检查碎片率
        fragmentation = self._calculate_fragmentation(device)
        
        if fragmentation < self.config.compaction_threshold:
            return
        
        self.logger.debug(f"Memory fragmentation: {fragmentation:.2%}, compacting...")
        
        # 对于GPU，尝试通过重新分配来整理
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # 强制同步所有流
            for dev in self.config.device_ids:
                try:
                    torch.cuda.synchronize(dev)
                except:
                    pass
    
    def _calculate_fragmentation(self, device: Optional[Union[str, int]] = None) -> float:
        """
        计算内存碎片率
        
        Args:
            device: 指定设备
            
        Returns:
            碎片率 (0-1)
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        
        try:
            if device is None:
                device = self.config.device_ids[0] if self.config.device_ids else 0
            
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            
            if reserved == 0:
                return 0.0
            
            # 简单的碎片率估算
            return (reserved - allocated) / reserved
        except:
            return 0.0
    
    def _calculate_freed_memory(self, before: Dict, after: Dict) -> int:
        """计算释放的内存"""
        total_freed = 0
        
        for device, before_bytes in before.items():
            after_bytes = after.get(device, before_bytes)
            freed = max(0, before_bytes - after_bytes)
            total_freed += freed
            
            # 更新设备统计
            if device in self._device_stats:
                self._device_stats[device].total_freed_bytes += freed
        
        return total_freed
    
    def _update_stats(self, freed_bytes: int, elapsed_ms: float,
                     before: Dict, after: Dict):
        """更新统计信息"""
        self._stats.total_collections += 1
        self._stats.total_freed_bytes += freed_bytes
        self._stats.total_time_ms += elapsed_ms
        self._stats.last_collection_time = time.time()
        
        # 计算平均时间
        self._stats.avg_collection_time_ms = (
            self._stats.total_time_ms / self._stats.total_collections
        )
        
        # 更新峰值
        self._stats.peak_memory_before = max(
            self._stats.peak_memory_before,
            sum(before.values())
        )
        self._stats.peak_memory_after = max(
            self._stats.peak_memory_after,
            sum(after.values())
        )
    
    def _trigger_pre_gc_callbacks(self):
        """触发前置回调"""
        for callback in self._pre_gc_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Pre-GC callback error: {e}")
    
    def _trigger_post_gc_callbacks(self, freed_bytes: int, elapsed_ms: float):
        """触发后置回调"""
        for callback in self._post_gc_callbacks:
            try:
                callback(freed_bytes, elapsed_ms)
            except Exception as e:
                self.logger.error(f"Post-GC callback error: {e}")
    
    def register_pre_gc_callback(self, callback: Callable):
        """注册前置回调"""
        self._pre_gc_callbacks.append(callback)
    
    def register_post_gc_callback(self, callback: Callable):
        """注册后置回调"""
        self._post_gc_callbacks.append(callback)
    
    def track_object(self, obj: Any, callback: Optional[Callable] = None):
        """
        追踪对象生命周期
        
        Args:
            obj: 要追踪的对象
            callback: 对象被回收时的回调
        """
        if not self.config.track_objects:
            return
        
        obj_id = id(obj)
        
        def on_destroyed(ref):
            if callback:
                callback(obj_id)
            self._tracked_objects.pop(obj_id, None)
        
        self._tracked_objects[obj_id] = weakref.ref(obj, on_destroyed)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取垃圾回收统计
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'global': {
                    'total_collections': self._stats.total_collections,
                    'total_freed_bytes': self._stats.total_freed_bytes,
                    'total_freed_mb': self._stats.total_freed_bytes / 1e6,
                    'total_time_ms': self._stats.total_time_ms,
                    'avg_collection_time_ms': self._stats.avg_collection_time_ms,
                    'efficiency_mb_per_sec': self._stats.efficiency,
                    'last_collection_time': self._stats.last_collection_time,
                    'current_phase': self._current_phase.name
                },
                'device_stats': {
                    str(device): {
                        'collections': stats.total_collections,
                        'freed_bytes': stats.total_freed_bytes
                    }
                    for device, stats in self._device_stats.items()
                },
                'tracked_objects': len(self._tracked_objects)
            }
    
    def get_memory_summary(self) -> str:
        """
        获取内存摘要
        
        Returns:
            格式化的内存摘要字符串
        """
        stats = self.get_stats()
        
        lines = [
            "=" * 50,
            "Garbage Collector Statistics",
            "=" * 50,
            f"Total Collections: {stats['global']['total_collections']}",
            f"Total Freed: {stats['global']['total_freed_mb']:.2f} MB",
            f"Average Time: {stats['global']['avg_collection_time_ms']:.2f} ms",
            f"Efficiency: {stats['global']['efficiency_mb_per_sec']:.2f} MB/s",
            f"Tracked Objects: {stats['tracked_objects']}",
            "=" * 50
        ]
        
        return "\n".join(lines)
    
    def set_strategy(self, strategy: Union[str, GCStrategy]):
        """
        设置回收策略
        
        Args:
            strategy: 新策略
        """
        if isinstance(strategy, str):
            strategy = GCStrategy[strategy.upper()]
        
        self.config.strategy = strategy
        
        # 根据策略调整监控
        if strategy == GCStrategy.ADAPTIVE and not self._running:
            self._start_monitoring()
        elif strategy != GCStrategy.ADAPTIVE and self._running:
            self._stop_monitoring()
        
        self.logger.info(f"GC strategy changed to: {strategy.name}")
    
    def _stop_monitoring(self):
        """停止监控"""
        if not self._running:
            return
        
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self._running = False
        self.logger.info("GC monitoring stopped")
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._stats = GCStats()
            self._device_stats.clear()
        
        self.logger.info("GC statistics reset")
    
    def shutdown(self):
        """关闭垃圾回收器"""
        self.logger.info("Shutting down GarbageCollector...")
        
        self._stop_monitoring()
        
        # 执行最后一次回收
        if self.config.enabled:
            self.collect(force=True)
        
        self._initialized = False
        self.logger.info("GarbageCollector shutdown complete")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False


# 便捷函数
def create_gc(config: Optional[Dict] = None) -> GarbageCollector:
    """创建垃圾回收器"""
    return GarbageCollector(config)


def force_gc(device: Optional[Union[str, int]] = None) -> Dict[str, Any]:
    """强制垃圾回收"""
    gc = GarbageCollector()
    return gc.collect(device=device, force=True)


def get_gc_stats() -> Dict[str, Any]:
    """获取垃圾回收统计"""
    gc = GarbageCollector()
    return gc.get_stats()
