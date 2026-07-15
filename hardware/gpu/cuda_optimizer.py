"""
CUDA Optimizer - CUDA流和事件管理

模块路径: hardware/gpu/cuda_optimizer.py

提供CUDA流管理、事件同步、异步执行等高级CUDA优化功能。
支持多GPU场景下的流管理和任务调度。
"""

import os
import sys
import time
import logging
import threading
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from queue import Queue

import torch
import torch.cuda

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """CUDA流配置"""
    priority: int = 0  # 流优先级，数值越小优先级越高
    non_blocking: bool = True
    device_id: int = 0


class CUDAStream:
    """
    CUDA流包装类
    
    封装PyTorch CUDA流，提供更友好的API和自动资源管理。
    """
    
    def __init__(self, config: Optional[StreamConfig] = None, stream: Optional[torch.cuda.Stream] = None):
        """
        初始化CUDA流
        
        Args:
            config: 流配置
            stream: 现有的PyTorch流，如果提供则直接使用
        """
        self.config = config or StreamConfig()
        self._device_id = self.config.device_id
        
        if stream is not None:
            self._stream = stream
        else:
            with torch.cuda.device(self._device_id):
                self._stream = torch.cuda.Stream(
                    priority=self.config.priority
                )
        
        self._events: List[torch.cuda.Event] = []
        self._lock = threading.Lock()
    
    @property
    def stream(self) -> torch.cuda.Stream:
        """获取底层PyTorch流"""
        return self._stream
    
    @property
    def device_id(self) -> int:
        """获取流所属设备ID"""
        return self._device_id
    
    def record_event(self, enable_timing: bool = False) -> torch.cuda.Event:
        """
        在流中记录事件
        
        Args:
            enable_timing: 是否启用计时
            
        Returns:
            CUDA事件对象
        """
        event = torch.cuda.Event(enable_timing=enable_timing)
        event.record(self._stream)
        with self._lock:
            self._events.append(event)
        return event
    
    def wait_event(self, event: torch.cuda.Event) -> None:
        """
        等待指定事件完成
        
        Args:
            event: 要等待的CUDA事件
        """
        self._stream.wait_event(event)
    
    def synchronize(self) -> None:
        """同步流，等待所有操作完成"""
        self._stream.synchronize()
    
    def query(self) -> bool:
        """查询流是否完成所有操作"""
        return self._stream.query()
    
    def wait_stream(self, other_stream: 'CUDAStream') -> None:
        """
        等待另一个流完成
        
        Args:
            other_stream: 要等待的其他流
        """
        event = other_stream.record_event()
        self.wait_event(event)
    
    @contextmanager
    def stream_context(self):
        """流上下文管理器"""
        with torch.cuda.device(self._device_id):
            with torch.cuda.stream(self._stream):
                yield self
    
    def __enter__(self):
        """进入流上下文"""
        self._stream_context = self.stream_context()
        return self._stream_context.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出流上下文"""
        return self._stream_context.__exit__(exc_type, exc_val, exc_tb)


class CUDAEvent:
    """
    CUDA事件管理类
    
    提供事件创建、记录、等待和计时功能。
    """
    
    def __init__(self, enable_timing: bool = True, blocking: bool = False, 
                 interprocess: bool = False):
        """
        初始化CUDA事件
        
        Args:
            enable_timing: 是否启用计时
            blocking: 是否为阻塞事件
            interprocess: 是否支持进程间通信
        """
        self._event = torch.cuda.Event(
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess
        )
        self._enable_timing = enable_timing
    
    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        记录事件
        
        Args:
            stream: 要记录事件的流，默认为当前流
        """
        self._event.record(stream)
    
    def wait(self, stream: torch.cuda.Stream) -> None:
        """
        让指定流等待此事件
        
        Args:
            stream: 要等待的流
        """
        stream.wait_event(self._event)
    
    def synchronize(self) -> None:
        """同步等待事件完成"""
        self._event.synchronize()
    
    def query(self) -> bool:
        """查询事件是否已完成"""
        return self._event.query()
    
    def elapsed_time(self, end_event: 'CUDAEvent') -> float:
        """
        计算两个事件之间的时间差（毫秒）
        
        Args:
            end_event: 结束事件
            
        Returns:
            时间差（毫秒）
        """
        if not self._enable_timing:
            raise RuntimeError("Timing is not enabled for this event")
        return self._event.elapsed_time(end_event._event)
    
    @property
    def event(self) -> torch.cuda.Event:
        """获取底层PyTorch事件"""
        return self._event


class CUDAOptimizer:
    """
    CUDA优化器
    
    管理多个CUDA流和事件，提供异步执行、流池管理、性能分析等功能。
    支持多GPU场景。
    """
    
    def __init__(self, device_id: int = 0, num_streams: int = 4):
        """
        初始化CUDA优化器
        
        Args:
            device_id: 默认设备ID
            num_streams: 默认流池大小
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        self._device_id = device_id
        self._num_streams = num_streams
        self._streams: Dict[int, List[CUDAStream]] = {}
        self._default_stream: Optional[CUDAStream] = None
        self._stream_pool: Queue = Queue()
        self._lock = threading.Lock()
        self._initialized = False
        
        # 性能统计
        self._stats: Dict[str, Any] = {
            "stream_usage": {},
            "event_timings": [],
            "async_ops": 0
        }
    
    def initialize(self) -> None:
        """初始化CUDA优化器"""
        if self._initialized:
            return
        
        with torch.cuda.device(self._device_id):
            # 创建默认流
            self._default_stream = CUDAStream(
                StreamConfig(device_id=self._device_id)
            )
            
            # 创建流池
            self._streams[self._device_id] = []
            for i in range(self._num_streams):
                config = StreamConfig(
                    priority=i % 2,  # 交替优先级
                    device_id=self._device_id
                )
                stream = CUDAStream(config)
                self._streams[self._device_id].append(stream)
                self._stream_pool.put(stream)
        
        self._initialized = True
        logger.info(f"CUDAOptimizer initialized on device {self._device_id} with {self._num_streams} streams")
    
    def get_stream(self, device_id: Optional[int] = None, 
                   priority: int = 0) -> CUDAStream:
        """
        获取一个可用的流
        
        Args:
            device_id: 设备ID，默认为初始化时的设备
            priority: 流优先级
            
        Returns:
            CUDA流对象
        """
        if not self._initialized:
            self.initialize()
        
        device = device_id if device_id is not None else self._device_id
        
        with self._lock:
            if device not in self._streams:
                self._streams[device] = []
            
            # 查找可用流
            for stream in self._streams[device]:
                if stream.query():
                    return stream
            
            # 创建新流
            config = StreamConfig(priority=priority, device_id=device)
            new_stream = CUDAStream(config)
            self._streams[device].append(new_stream)
            return new_stream
    
    def get_default_stream(self) -> CUDAStream:
        """获取默认流"""
        if not self._initialized:
            self.initialize()
        return self._default_stream
    
    def execute_async(self, func: Callable, *args, 
                      device_id: Optional[int] = None,
                      stream: Optional[CUDAStream] = None,
                      **kwargs) -> CUDAEvent:
        """
        异步执行函数
        
        Args:
            func: 要执行的函数
            args: 位置参数
            device_id: 设备ID
            stream: 指定的流，如果为None则自动获取
            kwargs: 关键字参数
            
        Returns:
            完成事件
        """
        if stream is None:
            stream = self.get_stream(device_id)
        
        with stream.stream_context():
            result = func(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                result.record_stream(stream.stream)
        
        event = stream.record_event()
        self._stats["async_ops"] += 1
        return event
    
    def parallel_execute(self, tasks: List[Tuple[Callable, tuple, dict]],
                         device_ids: Optional[List[int]] = None) -> List[CUDAEvent]:
        """
        并行执行多个任务
        
        Args:
            tasks: 任务列表，每个任务是(func, args, kwargs)元组
            device_ids: 设备ID列表，如果为None则使用默认设备
            
        Returns:
            完成事件列表
        """
        events = []
        
        for i, (func, args, kwargs) in enumerate(tasks):
            device_id = device_ids[i] if device_ids else None
            stream = self.get_stream(device_id)
            event = self.execute_async(func, *args, stream=stream, **kwargs)
            events.append(event)
        
        return events
    
    def synchronize_all(self, device_id: Optional[int] = None) -> None:
        """
        同步所有流
        
        Args:
            device_id: 设备ID，如果为None则同步所有设备
        """
        if device_id is not None:
            if device_id in self._streams:
                for stream in self._streams[device_id]:
                    stream.synchronize()
        else:
            for device_streams in self._streams.values():
                for stream in device_streams:
                    stream.synchronize()
    
    def create_event(self, enable_timing: bool = True) -> CUDAEvent:
        """
        创建CUDA事件
        
        Args:
            enable_timing: 是否启用计时
            
        Returns:
            CUDA事件对象
        """
        return CUDAEvent(enable_timing=enable_timing)
    
    def benchmark_kernel(self, kernel_func: Callable, *args,
                         warmup: int = 10, iterations: int = 100,
                         device_id: Optional[int] = None) -> Dict[str, float]:
        """
        基准测试CUDA内核
        
        Args:
            kernel_func: 要测试的内核函数
            args: 函数参数
            warmup: 预热次数
            iterations: 测试迭代次数
            device_id: 设备ID
            
        Returns:
            包含平均时间、最小时间、最大时间的字典
        """
        device = device_id if device_id is not None else self._device_id
        stream = self.get_stream(device)
        
        # 预热
        for _ in range(warmup):
            with stream.stream_context():
                kernel_func(*args)
        
        stream.synchronize()
        
        # 测试
        times = []
        for _ in range(iterations):
            start_event = CUDAEvent(enable_timing=True)
            end_event = CUDAEvent(enable_timing=True)
            
            with stream.stream_context():
                start_event.record()
                kernel_func(*args)
                end_event.record()
            
            stream.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            times.append(elapsed)
        
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    def memory_copy_async(self, src: torch.Tensor, dst: torch.Tensor,
                          stream: Optional[CUDAStream] = None,
                          non_blocking: bool = True) -> CUDAEvent:
        """
        异步内存拷贝
        
        Args:
            src: 源张量
            dst: 目标张量
            stream: CUDA流
            non_blocking: 是否非阻塞
            
        Returns:
            完成事件
        """
        if stream is None:
            stream = self.get_stream()
        
        with stream.stream_context():
            dst.copy_(src, non_blocking=non_blocking)
        
        return stream.record_event()
    
    def prefetch_to_device(self, tensors: List[torch.Tensor],
                           device_id: Optional[int] = None) -> List[CUDAEvent]:
        """
        预取张量到设备
        
        Args:
            tensors: 要预取的张量列表
            device_id: 目标设备ID
            
        Returns:
            完成事件列表
        """
        device = device_id if device_id is not None else self._device_id
        events = []
        
        for tensor in tensors:
            stream = self.get_stream(device)
            with stream.stream_context():
                if not tensor.is_cuda:
                    tensor = tensor.cuda(device)
                else:
                    tensor.record_stream(stream.stream)
            events.append(stream.record_event())
        
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self._stats = {
            "stream_usage": {},
            "event_timings": [],
            "async_ops": 0
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        self.synchronize_all()
        self._streams.clear()
        self._default_stream = None
        while not self._stream_pool.empty():
            try:
                self._stream_pool.get_nowait()
            except:
                break
        self._initialized = False
        logger.info("CUDAOptimizer cleaned up")
    
    @contextmanager
    def optimized_context(self):
        """优化上下文管理器"""
        self.initialize()
        try:
            yield self
        finally:
            self.synchronize_all()
    
    @staticmethod
    def get_device_properties(device_id: int = 0) -> Dict[str, Any]:
        """获取设备属性"""
        props = torch.cuda.get_device_properties(device_id)
        return {
            "name": props.name,
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "major": props.major,
            "minor": props.minor,
            "is_integrated": props.is_integrated,
            "is_multi_gpu_board": props.is_multi_gpu_board,
            "warp_size": props.warp_size if hasattr(props, 'warp_size') else 32
        }
    
    @staticmethod
    def set_device(device_id: int) -> None:
        """设置当前设备"""
        torch.cuda.set_device(device_id)
    
    @staticmethod
    def current_stream(device_id: Optional[int] = None) -> torch.cuda.Stream:
        """获取当前流"""
        return torch.cuda.current_stream(device_id)
    
    @staticmethod
    def default_stream(device_id: Optional[int] = None) -> torch.cuda.Stream:
        """获取默认流"""
        return torch.cuda.default_stream(device_id)


# 便捷的上下文管理器
@contextmanager
def cuda_stream_context(device_id: int = 0, priority: int = 0):
    """
    CUDA流上下文管理器
    
    Args:
        device_id: 设备ID
        priority: 流优先级
    """
    config = StreamConfig(device_id=device_id, priority=priority)
    stream = CUDAStream(config)
    with stream.stream_context():
        yield stream


@contextmanager
def cuda_optimized(device_id: int = 0, num_streams: int = 4):
    """
    CUDA优化上下文
    
    Args:
        device_id: 设备ID
        num_streams: 流数量
    """
    optimizer = CUDAOptimizer(device_id=device_id, num_streams=num_streams)
    with optimizer.optimized_context():
        yield optimizer
