"""
MemoryProfiler - 内存分析器模块

提供内存使用分析、峰值记录和趋势预测功能。
支持详细的内存统计和可视化报告。

模块路径: hardware/memory/memory_profiler.py
"""

import os
import sys
import json
import time
import threading
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import logging
import warnings
import traceback

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. MemoryProfiler will run in limited mode.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ProfileLevel(Enum):
    """分析级别"""
    BASIC = auto()
    DETAILED = auto()
    DEBUG = auto()


class MemoryEventType(Enum):
    """内存事件类型"""
    ALLOCATION = auto()
    DEALLOCATION = auto()
    PEAK_REACHED = auto()
    THRESHOLD_EXCEEDED = auto()
    OOM_WARNING = auto()
    CACHE_CLEARED = auto()


@dataclass
class MemoryEvent:
    """内存事件记录"""
    event_type: MemoryEventType
    timestamp: float
    device: Union[str, int]
    size_bytes: int
    description: str = ""
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type.name,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'device': str(self.device),
            'size_bytes': self.size_bytes,
            'description': self.description,
            'stack_trace': self.stack_trace
        }


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    device_id: Union[str, int]
    allocated_bytes: int
    reserved_bytes: int
    total_bytes: int
    process_memory_bytes: Optional[int] = None
    
    @property
    def utilization_rate(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes
    
    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.allocated_bytes
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'device_id': str(self.device_id),
            'allocated_bytes': self.allocated_bytes,
            'reserved_bytes': self.reserved_bytes,
            'total_bytes': self.total_bytes,
            'free_bytes': self.free_bytes,
            'utilization_rate': self.utilization_rate,
            'process_memory_bytes': self.process_memory_bytes
        }


@dataclass
class MemoryTrend:
    """内存趋势数据"""
    device_id: Union[str, int]
    start_time: float
    end_time: float
    avg_utilization: float
    peak_utilization: float
    growth_rate: float  # bytes per second
    volatility: float  # 标准差
    
    def to_dict(self) -> Dict:
        return {
            'device_id': str(self.device_id),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.end_time - self.start_time,
            'avg_utilization': self.avg_utilization,
            'peak_utilization': self.peak_utilization,
            'growth_rate': self.growth_rate,
            'volatility': self.volatility
        }


class MemoryProfiler:
    """
    内存分析器
    
    提供全面的内存使用分析，包括：
    - 实时内存监控
    - 峰值记录和分析
    - 内存趋势预测
    - 详细的统计报告
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化内存分析器
        
        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.config = config or {}
        self.profile_level = ProfileLevel[
            self.config.get('profile_level', 'DETAILED').upper()
        ]
        
        # 历史数据
        self._snapshots: Dict[Union[str, int], deque] = defaultdict(
            lambda: deque(maxlen=self.config.get('max_snapshots', 10000))
        )
        self._events: deque = deque(maxlen=self.config.get('max_events', 1000))
        
        # 峰值记录
        self._peak_memory: Dict[Union[str, int], int] = {}
        self._peak_timestamps: Dict[Union[str, int], float] = {}
        
        # 统计信息
        self._allocation_stats: Dict[str, Any] = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'total_allocated_bytes': 0,
            'total_freed_bytes': 0
        }
        
        # 监控线程
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        self._monitor_interval = self.config.get('monitor_interval', 1.0)
        
        # 回调函数
        self._threshold_callbacks: List[Tuple[float, Callable]] = []
        
        # 线程安全
        self._lock = threading.RLock()
        
        self._initialized = False
        self.logger.info("MemoryProfiler initialized")
    
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
    
    def initialize(self):
        """初始化分析器"""
        if self._initialized:
            return
        
        # 启动监控
        if self.config.get('enable_monitoring', True):
            self._start_monitoring()
        
        self._initialized = True
        self.logger.info("MemoryProfiler initialized successfully")
    
    def _start_monitoring(self):
        """启动监控线程"""
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
        while not self._stop_monitor.is_set():
            try:
                self.capture_snapshot()
                self._check_thresholds()
                self._stop_monitor.wait(self._monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                self._stop_monitor.wait(1.0)
    
    def capture_snapshot(self, device_id: Optional[Union[str, int]] = None) -> Optional[MemorySnapshot]:
        """
        捕获内存快照
        
        Args:
            device_id: 指定设备，None则捕获所有设备
            
        Returns:
            内存快照
        """
        if device_id is not None:
            return self._capture_device_snapshot(device_id)
        
        # 捕获所有设备
        snapshots = []
        
        # CPU内存
        if PSUTIL_AVAILABLE:
            snapshots.append(self._capture_device_snapshot('cpu'))
        
        # GPU内存
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                snapshots.append(self._capture_device_snapshot(i))
        
        return snapshots[0] if snapshots else None
    
    def _capture_device_snapshot(self, device_id: Union[str, int]) -> Optional[MemorySnapshot]:
        """捕获指定设备的内存快照"""
        try:
            timestamp = time.time()
            
            if device_id == 'cpu':
                if not PSUTIL_AVAILABLE:
                    return None
                
                mem = psutil.virtual_memory()
                snapshot = MemorySnapshot(
                    timestamp=timestamp,
                    device_id=device_id,
                    allocated_bytes=mem.used,
                    reserved_bytes=mem.used,
                    total_bytes=mem.total,
                    process_memory_bytes=self._get_process_memory()
                )
            else:
                if not TORCH_AVAILABLE or not torch.cuda.is_available():
                    return None
                
                torch.cuda.synchronize(device_id)
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                
                snapshot = MemorySnapshot(
                    timestamp=timestamp,
                    device_id=device_id,
                    allocated_bytes=allocated,
                    reserved_bytes=reserved,
                    total_bytes=total,
                    process_memory_bytes=None
                )
                
                # 更新峰值
                if (device_id not in self._peak_memory or 
                    allocated > self._peak_memory[device_id]):
                    self._peak_memory[device_id] = allocated
                    self._peak_timestamps[device_id] = timestamp
                    
                    self._record_event(
                        MemoryEventType.PEAK_REACHED,
                        device_id,
                        allocated,
                        f"New peak memory: {allocated / 1e9:.2f} GB"
                    )
            
            # 保存快照
            with self._lock:
                self._snapshots[device_id].append(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.warning(f"Failed to capture snapshot for {device_id}: {e}")
            return None
    
    def _get_process_memory(self) -> Optional[int]:
        """获取当前进程内存使用"""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return None
    
    def _check_thresholds(self):
        """检查阈值"""
        for threshold, callback in self._threshold_callbacks:
            for device_id, snapshots in self._snapshots.items():
                if not snapshots:
                    continue
                
                latest = snapshots[-1]
                if latest.utilization_rate >= threshold:
                    try:
                        callback(device_id, latest.utilization_rate, latest)
                    except Exception as e:
                        self.logger.error(f"Threshold callback error: {e}")
    
    def record_allocation(self, size_bytes: int, device: Union[str, int] = 'cpu',
                         description: str = ""):
        """
        记录内存分配
        
        Args:
            size_bytes: 分配大小
            device: 设备
            description: 描述
        """
        with self._lock:
            self._allocation_stats['total_allocations'] += 1
            self._allocation_stats['total_allocated_bytes'] += size_bytes
        
        if self.profile_level in (ProfileLevel.DETAILED, ProfileLevel.DEBUG):
            stack_trace = traceback.format_stack() if self.profile_level == ProfileLevel.DEBUG else None
            
            self._record_event(
                MemoryEventType.ALLOCATION,
                device,
                size_bytes,
                description,
                stack_trace
            )
    
    def record_deallocation(self, size_bytes: int, device: Union[str, int] = 'cpu',
                           description: str = ""):
        """
        记录内存释放
        
        Args:
            size_bytes: 释放大小
            device: 设备
            description: 描述
        """
        with self._lock:
            self._allocation_stats['total_deallocations'] += 1
            self._allocation_stats['total_freed_bytes'] += size_bytes
        
        if self.profile_level in (ProfileLevel.DETAILED, ProfileLevel.DEBUG):
            self._record_event(
                MemoryEventType.DEALLOCATION,
                device,
                size_bytes,
                description
            )
    
    def _record_event(self, event_type: MemoryEventType, device: Union[str, int],
                     size_bytes: int, description: str = "",
                     stack_trace: Optional[str] = None):
        """记录事件"""
        event = MemoryEvent(
            event_type=event_type,
            timestamp=time.time(),
            device=device,
            size_bytes=size_bytes,
            description=description,
            stack_trace=stack_trace
        )
        
        with self._lock:
            self._events.append(event)
    
    def register_threshold_callback(self, threshold: float, callback: Callable):
        """
        注册阈值回调
        
        Args:
            threshold: 内存利用率阈值 (0-1)
            callback: 回调函数(device_id, utilization, snapshot)
        """
        self._threshold_callbacks.append((threshold, callback))
    
    def get_peak_memory(self, device_id: Optional[Union[str, int]] = None) -> Dict:
        """
        获取峰值内存信息
        
        Args:
            device_id: 指定设备，None则返回所有设备
            
        Returns:
            峰值内存信息
        """
        if device_id is not None:
            return {
                'device_id': str(device_id),
                'peak_bytes': self._peak_memory.get(device_id, 0),
                'peak_timestamp': self._peak_timestamps.get(device_id),
                'peak_datetime': datetime.fromtimestamp(
                    self._peak_timestamps.get(device_id, 0)
                ).isoformat() if device_id in self._peak_timestamps else None
            }
        
        return {
            str(device): {
                'peak_bytes': peak,
                'peak_timestamp': self._peak_timestamps.get(device),
                'peak_datetime': datetime.fromtimestamp(
                    self._peak_timestamps.get(device, 0)
                ).isoformat() if device in self._peak_timestamps else None
            }
            for device, peak in self._peak_memory.items()
        }
    
    def get_memory_trend(self, device_id: Union[str, int],
                        duration: Optional[float] = None) -> Optional[MemoryTrend]:
        """
        获取内存趋势
        
        Args:
            device_id: 设备ID
            duration: 时间窗口（秒），None则使用所有历史数据
            
        Returns:
            内存趋势
        """
        with self._lock:
            snapshots = list(self._snapshots.get(device_id, []))
        
        if len(snapshots) < 2:
            return None
        
        # 过滤时间窗口
        if duration is not None:
            cutoff_time = time.time() - duration
            snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        if len(snapshots) < 2:
            return None
        
        # 计算统计值
        utilizations = [s.utilization_rate for s in snapshots]
        avg_utilization = sum(utilizations) / len(utilizations)
        peak_utilization = max(utilizations)
        
        # 计算增长率
        first = snapshots[0]
        last = snapshots[-1]
        time_diff = last.timestamp - first.timestamp
        
        if time_diff > 0:
            growth_rate = (last.allocated_bytes - first.allocated_bytes) / time_diff
        else:
            growth_rate = 0.0
        
        # 计算波动率（标准差）
        if len(utilizations) > 1:
            mean = avg_utilization
            variance = sum((u - mean) ** 2 for u in utilizations) / len(utilizations)
            volatility = variance ** 0.5
        else:
            volatility = 0.0
        
        return MemoryTrend(
            device_id=device_id,
            start_time=first.timestamp,
            end_time=last.timestamp,
            avg_utilization=avg_utilization,
            peak_utilization=peak_utilization,
            growth_rate=growth_rate,
            volatility=volatility
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取完整统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = {
                'allocation_stats': self._allocation_stats.copy(),
                'peak_memory': self.get_peak_memory(),
                'device_stats': {},
                'trends': {}
            }
            
            # 每个设备的统计
            for device_id in self._snapshots.keys():
                snapshots = list(self._snapshots[device_id])
                if snapshots:
                    latest = snapshots[-1]
                    stats['device_stats'][str(device_id)] = {
                        'current_allocated': latest.allocated_bytes,
                        'current_utilization': latest.utilization_rate,
                        'snapshot_count': len(snapshots)
                    }
                
                trend = self.get_memory_trend(device_id)
                if trend:
                    stats['trends'][str(device_id)] = trend.to_dict()
            
            return stats
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        生成内存分析报告
        
        Args:
            output_path: 输出文件路径，None则返回字符串
            
        Returns:
            报告内容
        """
        stats = self.get_statistics()
        
        report_lines = [
            "=" * 60,
            "Memory Profiler Report",
            "=" * 60,
            f"Generated at: {datetime.now().isoformat()}",
            "",
            "Allocation Statistics:",
            f"  Total Allocations: {stats['allocation_stats']['total_allocations']}",
            f"  Total Deallocations: {stats['allocation_stats']['total_deallocations']}",
            f"  Total Allocated: {stats['allocation_stats']['total_allocated_bytes'] / 1e9:.2f} GB",
            f"  Total Freed: {stats['allocation_stats']['total_freed_bytes'] / 1e9:.2f} GB",
            "",
            "Peak Memory Usage:",
        ]
        
        for device_id, peak_info in stats['peak_memory'].items():
            if isinstance(peak_info, dict):
                report_lines.append(
                    f"  {device_id}: {peak_info.get('peak_bytes', 0) / 1e9:.2f} GB"
                )
        
        report_lines.extend([
            "",
            "Device Statistics:",
        ])
        
        for device_id, device_stats in stats['device_stats'].items():
            report_lines.extend([
                f"  {device_id}:",
                f"    Current Allocated: {device_stats['current_allocated'] / 1e9:.2f} GB",
                f"    Current Utilization: {device_stats['current_utilization']:.2%}",
            ])
        
        report_lines.extend([
            "",
            "Memory Trends:",
        ])
        
        for device_id, trend in stats['trends'].items():
            report_lines.extend([
                f"  {device_id}:",
                f"    Average Utilization: {trend['avg_utilization']:.2%}",
                f"    Peak Utilization: {trend['peak_utilization']:.2%}",
                f"    Growth Rate: {trend['growth_rate']:.2f} bytes/s",
                f"    Volatility: {trend['volatility']:.4f}",
            ])
        
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
    
    def export_data(self, output_path: str, format: str = 'json'):
        """
        导出分析数据
        
        Args:
            output_path: 输出路径
            format: 格式 ('json' 或 'csv')
        """
        if format == 'json':
            data = {
                'snapshots': {
                    str(device): [s.to_dict() for s in snapshots]
                    for device, snapshots in self._snapshots.items()
                },
                'events': [e.to_dict() for e in self._events],
                'statistics': self.get_statistics()
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'csv':
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'device_id', 'allocated_bytes',
                    'reserved_bytes', 'total_bytes', 'utilization_rate'
                ])
                
                for device_id, snapshots in self._snapshots.items():
                    for snapshot in snapshots:
                        writer.writerow([
                            snapshot.timestamp,
                            device_id,
                            snapshot.allocated_bytes,
                            snapshot.reserved_bytes,
                            snapshot.total_bytes,
                            snapshot.utilization_rate
                        ])
        
        self.logger.info(f"Data exported to {output_path}")
    
    def reset(self):
        """重置分析器状态"""
        with self._lock:
            self._snapshots.clear()
            self._events.clear()
            self._peak_memory.clear()
            self._peak_timestamps.clear()
            self._allocation_stats = {
                'total_allocations': 0,
                'total_deallocations': 0,
                'total_allocated_bytes': 0,
                'total_freed_bytes': 0
            }
        
        self.logger.info("MemoryProfiler reset")
    
    def shutdown(self):
        """关闭分析器"""
        self.logger.info("Shutting down MemoryProfiler...")
        
        if self._monitoring:
            self._stop_monitor.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            self._monitoring = False
        
        self._initialized = False
        self.logger.info("MemoryProfiler shutdown complete")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False


# 便捷函数
def create_profiler(config: Optional[Dict] = None) -> MemoryProfiler:
    """创建内存分析器"""
    return MemoryProfiler(config)


def profile_memory(func: Callable) -> Callable:
    """内存分析装饰器"""
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.initialize()
        
        try:
            result = func(*args, **kwargs)
        finally:
            report = profiler.generate_report()
            print(report)
            profiler.shutdown()
        
        return result
    
    return wrapper
