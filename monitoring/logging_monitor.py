"""
日志监控系统 - Logging and Monitoring System
实现训练日志、性能监控、可视化、分布式追踪等功能
"""

import torch
import numpy as np
import json
import time
import os
import threading
import queue
import socket
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import warnings

# ==================== 日志数据结构 ====================

@dataclass
class LogRecord:
    """日志记录"""
    timestamp: float
    level: str
    message: str
    logger_name: str
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'level': self.level,
            'message': self.message,
            'logger_name': self.logger_name,
            **self.extra,
        }


@dataclass
class MetricRecord:
    """指标记录"""
    name: str
    value: float
    step: int
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceRecord:
    """性能记录"""
    name: str
    duration_ms: float
    count: int = 1
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ==================== 日志管理器 ====================

class LogManager:
    """日志管理器"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        max_file_size_mb: int = 100,
        max_files: int = 10,
        console_output: bool = True,
        json_format: bool = True,
    ):
        self.log_dir = log_dir
        self.max_file_size_mb = max_file_size_mb
        self.max_files = max_files
        self.console_output = console_output
        self.json_format = json_format
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.loggers: Dict[str, 'Logger'] = {}
        self.handlers: List['LogHandler'] = []
        
        # 添加默认处理器
        self.handlers.append(FileHandler(log_dir, max_file_size_mb, max_files))
        if console_output:
            self.handlers.append(ConsoleHandler())
    
    def get_logger(self, name: str) -> 'Logger':
        """获取日志器"""
        if name not in self.loggers:
            self.loggers[name] = Logger(name, self.handlers)
        return self.loggers[name]
    
    def add_handler(self, handler: 'LogHandler'):
        """添加处理器"""
        self.handlers.append(handler)
        for logger in self.loggers.values():
            logger.handlers.append(handler)
    
    def flush(self):
        """刷新所有处理器"""
        for handler in self.handlers:
            handler.flush()


class Logger:
    """日志器"""
    
    LEVELS = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50,
    }
    
    def __init__(
        self,
        name: str,
        handlers: List['LogHandler'],
        level: str = 'INFO',
    ):
        self.name = name
        self.handlers = handlers
        self.level = level
    
    def _log(self, level: str, message: str, **kwargs):
        """内部日志方法"""
        if self.LEVELS.get(level, 0) < self.LEVELS.get(self.level, 0):
            return
        
        record = LogRecord(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=self.name,
            extra=kwargs,
        )
        
        for handler in self.handlers:
            handler.handle(record)
    
    def debug(self, message: str, **kwargs):
        self._log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log('CRITICAL', message, **kwargs)
    
    def exception(self, message: str, exc_info=True, **kwargs):
        """记录异常"""
        import traceback
        if exc_info:
            kwargs['traceback'] = traceback.format_exc()
        self._log('ERROR', message, **kwargs)


class LogHandler:
    """日志处理器基类"""
    
    def handle(self, record: LogRecord):
        """处理日志记录（基类默认实现：输出到stdout）"""
        import sys
        timestamp = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] [{record.level}] [{record.logger_name}] {record.message}"
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
    
    def flush(self):
        pass


class FileHandler(LogHandler):
    """文件处理器"""
    
    def __init__(
        self,
        log_dir: str,
        max_file_size_mb: int = 100,
        max_files: int = 10,
    ):
        self.log_dir = log_dir
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        
        self.current_file = None
        self.current_size = 0
        self._lock = threading.Lock()
        
        self._rotate_file()
    
    def _rotate_file(self):
        """轮转日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"log_{timestamp}.jsonl"
        filepath = os.path.join(self.log_dir, filename)
        
        self.current_file = open(filepath, 'a', encoding='utf-8')
        self.current_size = 0
        
        # 清理旧文件
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """清理旧日志文件"""
        files = sorted([
            f for f in os.listdir(self.log_dir)
            if f.startswith('log_') and f.endswith('.jsonl')
        ])
        
        while len(files) > self.max_files:
            os.remove(os.path.join(self.log_dir, files[0]))
            files.pop(0)
    
    def handle(self, record: LogRecord):
        """处理日志记录"""
        with self._lock:
            line = json.dumps(record.to_dict()) + '\n'
            line_bytes = len(line.encode('utf-8'))
            
            if self.current_size + line_bytes > self.max_file_size_bytes:
                self.current_file.close()
                self._rotate_file()
            
            self.current_file.write(line)
            self.current_size += line_bytes
    
    def flush(self):
        """刷新文件"""
        with self._lock:
            if self.current_file:
                self.current_file.flush()


class ConsoleHandler(LogHandler):
    """控制台处理器"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def handle(self, record: LogRecord):
        """处理日志记录"""
        color = self.COLORS.get(record.level, '')
        timestamp = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"{color}[{timestamp}] [{record.level}] [{record.logger_name}] {record.message}{self.RESET}"
        print(message)


# ==================== 指标追踪器 ====================

class MetricsTracker:
    """指标追踪器"""
    
    def __init__(
        self,
        history_size: int = 10000,
        aggregation_window: int = 100,
    ):
        self.history_size = history_size
        self.aggregation_window = aggregation_window
        
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.aggregations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()
    
    def log(
        self,
        name: str,
        value: float,
        step: int,
        tags: Optional[Dict[str, str]] = None,
    ):
        """记录指标"""
        record = MetricRecord(
            name=name,
            value=value,
            step=step,
            timestamp=time.time(),
            tags=tags or {},
        )
        
        with self._lock:
            self.metrics[name].append(record)
            self._update_aggregation(name, value)
    
    def log_dict(self, metrics: Dict[str, float], step: int, tags: Optional[Dict] = None):
        """批量记录指标"""
        for name, value in metrics.items():
            self.log(name, value, step, tags)
    
    def _update_aggregation(self, name: str, value: float):
        """更新聚合统计"""
        agg = self.aggregations[name]
        
        if 'count' not in agg:
            agg['count'] = 0
            agg['sum'] = 0.0
            agg['min'] = float('inf')
            agg['max'] = float('-inf')
        
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['mean'] = agg['sum'] / agg['count']
        
        # 滑动窗口统计
        history = list(self.metrics[name])[-self.aggregation_window:]
        if history:
            values = [r.value for r in history]
            agg['recent_mean'] = np.mean(values)
            agg['recent_std'] = np.std(values)
    
    def get(self, name: str) -> List[MetricRecord]:
        """获取指标历史"""
        with self._lock:
            return list(self.metrics.get(name, []))
    
    def get_latest(self, name: str) -> Optional[MetricRecord]:
        """获取最新指标"""
        with self._lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
        return None
    
    def get_aggregation(self, name: str) -> Dict[str, float]:
        """获取聚合统计"""
        with self._lock:
            return dict(self.aggregations.get(name, {}))
    
    def get_all_aggregations(self) -> Dict[str, Dict[str, float]]:
        """获取所有聚合统计"""
        with self._lock:
            return {k: dict(v) for k, v in self.aggregations.items()}
    
    def clear(self, name: Optional[str] = None):
        """清除指标"""
        with self._lock:
            if name:
                if name in self.metrics:
                    del self.metrics[name]
                if name in self.aggregations:
                    del self.aggregations[name]
            else:
                self.metrics.clear()
                self.aggregations.clear()


# ==================== 性能分析器 ====================

class Profiler:
    """性能分析器"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: Dict[str, PerformanceRecord] = {}
        self.active_timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start(self, name: str):
        """开始计时"""
        if not self.enabled:
            return
        
        self.active_timers[name] = time.perf_counter()
    
    def end(self, name: str) -> Optional[float]:
        """结束计时"""
        if not self.enabled:
            return None
        
        if name not in self.active_timers:
            return None
        
        start_time = self.active_timers.pop(name)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        with self._lock:
            if name not in self.records:
                self.records[name] = PerformanceRecord(name=name, duration_ms=duration_ms)
            else:
                record = self.records[name]
                record.count += 1
                record.total_ms += duration_ms
                record.min_ms = min(record.min_ms, duration_ms)
                record.max_ms = max(record.max_ms, duration_ms)
                record.duration_ms = duration_ms
        
        return duration_ms
    
    def profile(self, name: Optional[str] = None):
        """装饰器：自动计时"""
        def decorator(func):
            profile_name = name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start(profile_name)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.end(profile_name)
            
            return wrapper
        return decorator
    
    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """获取统计"""
        with self._lock:
            if name in self.records:
                record = self.records[name]
                return {
                    'count': record.count,
                    'total_ms': record.total_ms,
                    'mean_ms': record.total_ms / record.count if record.count > 0 else 0,
                    'min_ms': record.min_ms,
                    'max_ms': record.max_ms,
                    'last_ms': record.duration_ms,
                }
        return None
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计"""
        with self._lock:
            return {
                name: {
                    'count': r.count,
                    'total_ms': r.total_ms,
                    'mean_ms': r.total_ms / r.count if r.count > 0 else 0,
                    'min_ms': r.min_ms,
                    'max_ms': r.max_ms,
                    'last_ms': r.duration_ms,
                }
                for name, r in self.records.items()
            }
    
    def reset(self):
        """重置"""
        with self._lock:
            self.records.clear()
            self.active_timers.clear()


class Timer:
    """上下文管理器计时器"""
    
    def __init__(self, name: str, profiler: Optional[Profiler] = None):
        self.name = name
        self.profiler = profiler or global_profiler
        self.duration_ms = 0.0
    
    def __enter__(self):
        self.profiler.start(self.name)
        return self
    
    def __exit__(self, *args):
        self.duration_ms = self.profiler.end(self.name) or 0.0


# 全局性能分析器
global_profiler = Profiler()


# ==================== 系统监控 ====================

class SystemMonitor:
    """系统监控"""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.running = False
        self.monitor_thread = None
        
        self.cpu_history: deque = deque(maxlen=1000)
        self.memory_history: deque = deque(maxlen=1000)
        self.gpu_history: deque = deque(maxlen=1000)
        
        self._lock = threading.Lock()
    
    def start(self):
        """启动监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # CPU使用率
                cpu_percent = self._get_cpu_percent()
                
                # 内存使用
                memory_info = self._get_memory_info()
                
                # GPU使用（如果可用）
                gpu_info = self._get_gpu_info()
                
                timestamp = time.time()
                
                with self._lock:
                    self.cpu_history.append((timestamp, cpu_percent))
                    self.memory_history.append((timestamp, memory_info))
                    if gpu_info:
                        self.gpu_history.append((timestamp, gpu_info))
                
            except Exception as e:
                pass
            
            time.sleep(self.interval)
    
    def _get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except:
            return 0.0
    
    def _get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total_mb': mem.total / (1024 * 1024),
                'used_mb': mem.used / (1024 * 1024),
                'percent': mem.percent,
            }
        except:
            return {'total_mb': 0, 'used_mb': 0, 'percent': 0}
    
    def _get_gpu_info(self) -> Optional[List[Dict]]:
        """获取GPU信息"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            num_gpus = pynvml.nvmlDeviceGetCount()
            gpu_info = []
            
            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 内存
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # 使用率
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_info.append({
                    'index': i,
                    'memory_total_mb': mem_info.total / (1024 * 1024),
                    'memory_used_mb': mem_info.used / (1024 * 1024),
                    'memory_percent': mem_info.used / mem_info.total * 100,
                    'gpu_percent': util.gpu,
                    'memory_util_percent': util.memory,
                })
            
            pynvml.nvmlShutdown()
            return gpu_info
        
        except:
            return None
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计"""
        with self._lock:
            stats = {
                'cpu': self.cpu_history[-1] if self.cpu_history else None,
                'memory': self.memory_history[-1] if self.memory_history else None,
                'gpu': self.gpu_history[-1] if self.gpu_history else None,
            }
        return stats
    
    def get_history(
        self,
        duration_seconds: float = 60.0,
    ) -> Dict[str, List]:
        """获取历史数据"""
        cutoff = time.time() - duration_seconds
        
        with self._lock:
            cpu = [(t, v) for t, v in self.cpu_history if t >= cutoff]
            memory = [(t, v) for t, v in self.memory_history if t >= cutoff]
            gpu = [(t, v) for t, v in self.gpu_history if t >= cutoff]
        
        return {'cpu': cpu, 'memory': memory, 'gpu': gpu}


# ==================== 分布式追踪 ====================

@dataclass
class Span:
    """追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)
    
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class Tracer:
    """分布式追踪器"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: Dict[str, Span] = {}
        self.active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
    
    def _generate_id(self) -> str:
        """生成ID"""
        return hashlib.md5(f"{time.time()}{threading.current_thread().ident}".encode()).hexdigest()[:16]
    
    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Span:
        """开始跨度"""
        if trace_id is None:
            trace_id = self._generate_id()
        
        span_id = self._generate_id()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=time.time(),
            tags=tags or {},
        )
        
        with self._lock:
            self.spans[span_id] = span
            self.active_spans[span_id] = span
        
        return span
    
    def end_span(self, span: Span):
        """结束跨度"""
        span.end_time = time.time()
        
        with self._lock:
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
    
    def log_to_span(self, span: Span, message: str, **kwargs):
        """记录跨度日志"""
        span.logs.append({
            'timestamp': time.time(),
            'message': message,
            **kwargs,
        })
    
    def trace(self, name: str, tags: Optional[Dict] = None):
        """装饰器：自动追踪"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                span = self.start_span(name, tags=tags)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.log_to_span(span, f"Exception: {e}")
                    raise
                finally:
                    self.end_span(span)
            return wrapper
        return decorator
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """获取追踪的所有跨度"""
        with self._lock:
            return [s for s in self.spans.values() if s.trace_id == trace_id]
    
    def export_trace(self, trace_id: str) -> Dict:
        """导出追踪数据"""
        spans = self.get_trace(trace_id)
        return {
            'trace_id': trace_id,
            'service_name': self.service_name,
            'spans': [
                {
                    'span_id': s.span_id,
                    'parent_span_id': s.parent_span_id,
                    'name': s.name,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'duration_ms': s.duration_ms(),
                    'tags': s.tags,
                    'logs': s.logs,
                }
                for s in spans
            ],
        }


# ==================== 可视化 ====================

class MetricsVisualizer:
    """指标可视化器"""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        self.metrics = metrics_tracker
    
    def plot_metric(
        self,
        name: str,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
    ) -> Optional[Any]:
        """绘制指标曲线"""
        try:
            import matplotlib.pyplot as plt
            
            records = self.metrics.get(name)
            if not records:
                return None
            
            steps = [r.step for r in records]
            values = [r.value for r in records]
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(steps, values)
            ax.set_xlabel('Step')
            ax.set_ylabel(name)
            ax.set_title(title or f'{name} over time')
            ax.grid(True)
            
            return fig
        
        except ImportError:
            warnings.warn("matplotlib not available for visualization")
            return None
    
    def plot_metrics(
        self,
        names: List[str],
        figsize: Tuple[int, int] = (12, 8),
    ) -> Optional[Any]:
        """绘制多条指标曲线"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(len(names), 1, figsize=figsize)
            if len(names) == 1:
                axes = [axes]
            
            for ax, name in zip(axes, names):
                records = self.metrics.get(name)
                if records:
                    steps = [r.step for r in records]
                    values = [r.value for r in records]
                    ax.plot(steps, values)
                    ax.set_ylabel(name)
                    ax.grid(True)
            
            axes[-1].set_xlabel('Step')
            fig.tight_layout()
            
            return fig
        
        except ImportError:
            return None
    
    def plot_comparison(
        self,
        metrics_dict: Dict[str, List[Tuple[int, float]]],
        figsize: Tuple[int, int] = (10, 6),
    ) -> Optional[Any]:
        """绘制对比图"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            for name, data in metrics_dict.items():
                steps = [d[0] for d in data]
                values = [d[1] for d in data]
                ax.plot(steps, values, label=name)
            
            ax.set_xlabel('Step')
            ax.legend()
            ax.grid(True)
            
            return fig
        
        except ImportError:
            return None


# ==================== 训练监控器 ====================

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "experiment",
    ):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = LogManager(self.log_dir).get_logger("training")
        self.metrics = MetricsTracker()
        self.profiler = Profiler()
        self.tracer = Tracer(experiment_name)
        self.visualizer = MetricsVisualizer(self.metrics)
        
        self.step = 0
        self.epoch = 0
        self.best_metrics: Dict[str, float] = {}
        self.start_time = time.time()
    
    def log_step(self, metrics: Dict[str, float], prefix: str = "train"):
        """记录步骤指标"""
        self.step += 1
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.metrics.log_dict(prefixed_metrics, self.step)
        
        self.logger.info(
            f"Step {self.step}",
            **prefixed_metrics,
        )
    
    def log_epoch(
        self,
        metrics: Dict[str, float],
        prefix: str = "train",
    ):
        """记录epoch指标"""
        self.epoch += 1
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.metrics.log_dict(prefixed_metrics, self.epoch)
        
        self.logger.info(
            f"Epoch {self.epoch}",
            **prefixed_metrics,
        )
    
    def log_model_info(self, model: torch.nn.Module):
        """记录模型信息"""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(
            "Model info",
            num_params=num_params,
            num_trainable=num_trainable,
            model_type=type(model).__name__,
        )
    
    def log_gradients(self, model: torch.nn.Module):
        """记录梯度统计"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.metrics.log(f"grad_norm/{name}", grad_norm, self.step)
    
    def check_improvement(
        self,
        metric_name: str,
        value: float,
        mode: str = 'min',
    ) -> bool:
        """检查是否有改进"""
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = value
            return True
        
        if mode == 'min':
            improved = value < self.best_metrics[metric_name]
        else:
            improved = value > self.best_metrics[metric_name]
        
        if improved:
            self.best_metrics[metric_name] = value
        
        return improved
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        is_best: bool = False,
    ):
        """保存检查点"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metrics': self.best_metrics,
        }
        
        # 保存最新
        path = os.path.join(self.log_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, path)
        
        # 保存最佳
        if is_best:
            path = os.path.join(self.log_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, path)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        elapsed = time.time() - self.start_time
        
        return {
            'step': self.step,
            'epoch': self.epoch,
            'elapsed_seconds': elapsed,
            'best_metrics': self.best_metrics,
            'metrics_summary': self.metrics.get_all_aggregations(),
            'profiler_stats': self.profiler.get_all_stats(),
        }


# ==================== 主函数 ====================

def main():
    """测试日志监控系统"""
    print("日志监控系统测试")
    
    # 测试日志管理器
    print("\n测试日志管理器...")
    log_manager = LogManager(log_dir="/tmp/test_logs")
    logger = log_manager.get_logger("test")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # 测试指标追踪器
    print("\n测试指标追踪器...")
    tracker = MetricsTracker()
    
    for i in range(100):
        tracker.log("loss", 1.0 / (i + 1), i)
        tracker.log("accuracy", min(0.9, 0.5 + i * 0.005), i)
    
    agg = tracker.get_aggregation("loss")
    print(f"Loss aggregation: mean={agg['mean']:.4f}, min={agg['min']:.4f}, max={agg['max']:.4f}")
    
    # 测试性能分析器
    print("\n测试性能分析器...")
    profiler = Profiler()
    
    @profiler.profile("test_function")
    def test_function():
        time.sleep(0.01)
        return 42
    
    for _ in range(10):
        test_function()
    
    stats = profiler.get_stats("test_function")
    print(f"Profiler stats: count={stats['count']}, mean_ms={stats['mean_ms']:.2f}")
    
    # 测试计时器
    print("\n测试计时器...")
    with Timer("test_block", profiler) as t:
        time.sleep(0.05)
    print(f"Timer duration: {t.duration_ms:.2f}ms")
    
    # 测试分布式追踪
    print("\n测试分布式追踪...")
    tracer = Tracer("test_service")
    
    span = tracer.start_span("operation")
    time.sleep(0.01)
    tracer.log_to_span(span, "Processing...")
    time.sleep(0.01)
    tracer.end_span(span)
    
    trace_data = tracer.export_trace(span.trace_id)
    print(f"Trace duration: {trace_data['spans'][0]['duration_ms']:.2f}ms")
    
    # 测试训练监控器
    print("\n测试训练监控器...")
    monitor = TrainingMonitor(log_dir="/tmp/test_logs", experiment_name="test_exp")
    
    for i in range(10):
        monitor.log_step({"loss": 1.0 / (i + 1), "accuracy": 0.5 + i * 0.05})
    
    summary = monitor.get_summary()
    print(f"Training summary: step={summary['step']}, elapsed={summary['elapsed_seconds']:.2f}s")
    
    print("\n日志监控系统测试完成")


if __name__ == "__main__":
    main()
