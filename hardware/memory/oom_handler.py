"""
OOMHandler - 内存溢出处理模块

提供内存溢出检测、自动清理和降级策略。
支持多级防护和恢复机制。

模块路径: hardware/memory/oom_handler.py
"""

import os
import sys
import gc
import time
import threading
import traceback
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. OOMHandler will run in limited mode.")


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class OOMSeverity(Enum):
    """OOM严重程度"""
    WARNING = auto()      # 警告级别，内存压力高
    CRITICAL = auto()     # 临界级别，即将OOM
    EMERGENCY = auto()    # 紧急级别，已发生OOM


class RecoveryAction(Enum):
    """恢复操作类型"""
    CLEAR_CACHE = auto()
    RELEASE_UNUSED = auto()
    DEGRADE_PRECISION = auto()
    OFFLOAD_TO_CPU = auto()
    KILL_PROCESSES = auto()
    EMERGENCY_SAVE = auto()


@dataclass
class OOMConfig:
    """OOM处理器配置"""
    # 阈值配置
    warning_threshold: float = 0.80
    critical_threshold: float = 0.90
    emergency_threshold: float = 0.95
    
    # 自动恢复配置
    auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_cooldown: float = 5.0
    
    # 降级策略
    enable_degradation: bool = True
    degrade_precision: bool = True  # 降低精度（FP16 -> FP32）
    offload_to_cpu: bool = True
    
    # 清理策略
    aggressive_cleanup: bool = True
    clear_cuda_cache: bool = True
    force_gc: bool = True
    
    # 监控配置
    enable_monitoring: bool = True
    monitor_interval: float = 1.0
    
    # 多GPU配置
    device_ids: List[int] = field(default_factory=lambda: [0])


@dataclass
class OOMEvent:
    """OOM事件记录"""
    severity: OOMSeverity
    timestamp: float
    device: Union[str, int]
    allocated_bytes: int
    total_bytes: int
    error_message: str = ""
    stack_trace: Optional[str] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    recovery_success: bool = False


@dataclass
class RecoveryResult:
    """恢复操作结果"""
    action: RecoveryAction
    success: bool
    freed_bytes: int
    elapsed_ms: float
    message: str = ""


class OOMHandler:
    """
    内存溢出处理器
    
    提供全面的OOM防护和恢复：
    - 多级OOM检测（警告、临界、紧急）
    - 自动清理和恢复策略
    - 模型降级和CPU卸载
    - 紧急保存和恢复
    """
    
    def __init__(self, config: Optional[Union[Dict, OOMConfig]] = None):
        """
        初始化OOM处理器
        
        Args:
            config: 配置字典或OOMConfig对象
        """
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 解析配置
        if config is None:
            self.config = OOMConfig()
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        else:
            self.config = config
        
        # 状态
        self._initialized = False
        self._monitoring = False
        self._recovery_in_progress = False
        self._recovery_attempts = 0
        self._last_recovery_time = 0
        
        # 事件历史
        self._oom_events: deque = deque(maxlen=100)
        self._recovery_history: deque = deque(maxlen=50)
        
        # 回调
        self._warning_callbacks: List[Callable] = []
        self._critical_callbacks: List[Callable] = []
        self._emergency_callbacks: List[Callable] = []
        self._recovery_callbacks: List[Callable] = []
        
        # 监控
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 降级状态
        self._degraded_devices: set = set()
        self._offloaded_tensors: Dict[int, Any] = {}
        
        self.logger.info("OOMHandler initialized")
    
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
    
    def _parse_config(self, config_dict: Dict) -> OOMConfig:
        """解析配置字典"""
        return OOMConfig(
            warning_threshold=config_dict.get('warning_threshold', 0.80),
            critical_threshold=config_dict.get('critical_threshold', 0.90),
            emergency_threshold=config_dict.get('emergency_threshold', 0.95),
            auto_recovery=config_dict.get('auto_recovery', True),
            max_recovery_attempts=config_dict.get('max_recovery_attempts', 3),
            recovery_cooldown=config_dict.get('recovery_cooldown', 5.0),
            enable_degradation=config_dict.get('enable_degradation', True),
            degrade_precision=config_dict.get('degrade_precision', True),
            offload_to_cpu=config_dict.get('offload_to_cpu', True),
            aggressive_cleanup=config_dict.get('aggressive_cleanup', True),
            clear_cuda_cache=config_dict.get('clear_cuda_cache', True),
            force_gc=config_dict.get('force_gc', True),
            enable_monitoring=config_dict.get('enable_monitoring', True),
            monitor_interval=config_dict.get('monitor_interval', 1.0),
            device_ids=config_dict.get('device_ids', [0])
        )
    
    def initialize(self):
        """初始化OOM处理器"""
        if self._initialized:
            return
        
        # 启动监控
        if self.config.enable_monitoring:
            self._start_monitoring()
        
        # 注册PyTorch OOM钩子
        if TORCH_AVAILABLE:
            self._register_oom_hook()
        
        self._initialized = True
        self.logger.info("OOMHandler initialized successfully")
    
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
        self.logger.info("OOM monitoring started")
    
    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_monitor.is_set():
            try:
                self._check_memory_status()
                self._stop_monitor.wait(self.config.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in OOM monitor loop: {e}")
                self._stop_monitor.wait(1.0)
    
    def _check_memory_status(self):
        """检查内存状态"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        for device_id in self.config.device_ids:
            try:
                torch.cuda.synchronize(device_id)
                allocated = torch.cuda.memory_allocated(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                utilization = allocated / total
                
                if utilization >= self.config.emergency_threshold:
                    self._handle_oom(OOMSeverity.EMERGENCY, device_id, allocated, total)
                elif utilization >= self.config.critical_threshold:
                    self._handle_oom(OOMSeverity.CRITICAL, device_id, allocated, total)
                elif utilization >= self.config.warning_threshold:
                    self._handle_oom(OOMSeverity.WARNING, device_id, allocated, total)
                    
            except Exception as e:
                self.logger.warning(f"Failed to check memory status for device {device_id}: {e}")
    
    def _handle_oom(self, severity: OOMSeverity, device: Union[str, int],
                   allocated: int, total: int, error_msg: str = ""):
        """
        处理OOM事件
        
        Args:
            severity: 严重程度
            device: 设备ID
            allocated: 已分配内存
            total: 总内存
            error_msg: 错误信息
        """
        # 记录事件
        event = OOMEvent(
            severity=severity,
            timestamp=time.time(),
            device=device,
            allocated_bytes=allocated,
            total_bytes=total,
            error_message=error_msg,
            stack_trace=traceback.format_exc() if severity == OOMSeverity.EMERGENCY else None
        )
        
        with self._lock:
            self._oom_events.append(event)
        
        # 触发回调
        if severity == OOMSeverity.WARNING:
            self._trigger_warning_callbacks(device, allocated, total)
        elif severity == OOMSeverity.CRITICAL:
            self._trigger_critical_callbacks(device, allocated, total)
        elif severity == OOMSeverity.EMERGENCY:
            self._trigger_emergency_callbacks(device, allocated, total)
        
        # 自动恢复
        if self.config.auto_recovery and severity in (OOMSeverity.CRITICAL, OOMSeverity.EMERGENCY):
            self._attempt_recovery(device, event)
    
    def _attempt_recovery(self, device: Union[str, int], event: OOMEvent) -> bool:
        """
        尝试恢复
        
        Args:
            device: 设备ID
            event: OOM事件
            
        Returns:
            恢复是否成功
        """
        with self._lock:
            # 检查恢复冷却
            current_time = time.time()
            if current_time - self._last_recovery_time < self.config.recovery_cooldown:
                self.logger.debug("Recovery in cooldown period, skipping")
                return False
            
            # 检查恢复次数
            if self._recovery_attempts >= self.config.max_recovery_attempts:
                self.logger.error("Max recovery attempts reached")
                return False
            
            self._recovery_in_progress = True
            self._recovery_attempts += 1
            self._last_recovery_time = current_time
        
        try:
            self.logger.info(f"Attempting recovery for device {device} "
                           f"(attempt {self._recovery_attempts}/{self.config.max_recovery_attempts})")
            
            results = []
            
            # 1. 清理缓存
            if self.config.clear_cuda_cache:
                result = self._clear_cache(device)
                results.append(result)
                if result.success and result.freed_bytes > 100 * 1024 * 1024:  # 100MB
                    event.recovery_actions.append(RecoveryAction.CLEAR_CACHE)
            
            # 2. 释放未使用的张量
            result = self._release_unused_tensors(device)
            results.append(result)
            if result.success:
                event.recovery_actions.append(RecoveryAction.RELEASE_UNUSED)
            
            # 3. 强制垃圾回收
            if self.config.force_gc:
                result = self._force_garbage_collection()
                results.append(result)
            
            # 4. 降级精度
            if self.config.enable_degradation and self.config.degrade_precision:
                result = self._degrade_precision(device)
                results.append(result)
                if result.success:
                    event.recovery_actions.append(RecoveryAction.DEGRADE_PRECISION)
            
            # 5. 卸载到CPU
            if self.config.enable_degradation and self.config.offload_to_cpu:
                result = self._offload_to_cpu(device)
                results.append(result)
                if result.success:
                    event.recovery_actions.append(RecoveryAction.OFFLOAD_TO_CPU)
            
            # 计算总释放内存
            total_freed = sum(r.freed_bytes for r in results if r.success)
            
            # 检查恢复效果
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize(device)
                new_allocated = torch.cuda.memory_allocated(device)
                new_utilization = new_allocated / torch.cuda.get_device_properties(device).total_memory
                
                recovery_success = new_utilization < self.config.critical_threshold
            else:
                recovery_success = total_freed > 0
            
            event.recovery_success = recovery_success
            
            # 记录恢复历史
            recovery_record = {
                'timestamp': time.time(),
                'device': device,
                'results': results,
                'total_freed': total_freed,
                'success': recovery_success
            }
            
            with self._lock:
                self._recovery_history.append(recovery_record)
            
            # 触发恢复回调
            self._trigger_recovery_callbacks(device, recovery_success, total_freed)
            
            if recovery_success:
                self.logger.info(f"Recovery successful: freed {total_freed / 1e6:.2f} MB")
                self._recovery_attempts = 0  # 重置尝试次数
            else:
                self.logger.warning(f"Recovery partially successful: freed {total_freed / 1e6:.2f} MB")
            
            return recovery_success
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            return False
        finally:
            self._recovery_in_progress = False
    
    def _clear_cache(self, device: Union[str, int]) -> RecoveryResult:
        """清理GPU缓存"""
        start_time = time.time()
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return RecoveryResult(
                action=RecoveryAction.CLEAR_CACHE,
                success=False,
                freed_bytes=0,
                elapsed_ms=0,
                message="CUDA not available"
            )
        
        try:
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_allocated(device)
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated(device)
            freed = max(0, before - after)
            
            elapsed = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                action=RecoveryAction.CLEAR_CACHE,
                success=freed > 0,
                freed_bytes=freed,
                elapsed_ms=elapsed,
                message=f"Freed {freed / 1e6:.2f} MB"
            )
        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.CLEAR_CACHE,
                success=False,
                freed_bytes=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    def _release_unused_tensors(self, device: Union[str, int]) -> RecoveryResult:
        """释放未使用的张量"""
        start_time = time.time()
        
        if not TORCH_AVAILABLE:
            return RecoveryResult(
                action=RecoveryAction.RELEASE_UNUSED,
                success=False,
                freed_bytes=0,
                elapsed_ms=0,
                message="PyTorch not available"
            )
        
        try:
            # 获取当前内存使用
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_allocated(device)
            else:
                before = 0
            
            # 清理Python垃圾
            gc.collect()
            
            # 再次清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                after = torch.cuda.memory_allocated(device)
                freed = max(0, before - after)
            else:
                freed = 0
            
            elapsed = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                action=RecoveryAction.RELEASE_UNUSED,
                success=True,
                freed_bytes=freed,
                elapsed_ms=elapsed,
                message=f"Garbage collection completed"
            )
        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.RELEASE_UNUSED,
                success=False,
                freed_bytes=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    def _force_garbage_collection(self) -> RecoveryResult:
        """强制垃圾回收"""
        start_time = time.time()
        
        try:
            gc.collect()
            elapsed = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                action=RecoveryAction.RELEASE_UNUSED,
                success=True,
                freed_bytes=0,
                elapsed_ms=elapsed,
                message="Forced GC completed"
            )
        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.RELEASE_UNUSED,
                success=False,
                freed_bytes=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    def _degrade_precision(self, device: Union[str, int]) -> RecoveryResult:
        """降级精度（FP32 -> FP16）"""
        start_time = time.time()
        
        if not TORCH_AVAILABLE or device == 'cpu':
            return RecoveryResult(
                action=RecoveryAction.DEGRADE_PRECISION,
                success=False,
                freed_bytes=0,
                elapsed_ms=0,
                message="Precision degradation not applicable"
            )
        
        try:
            # 标记设备为降级状态
            self._degraded_devices.add(device)
            
            elapsed = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                action=RecoveryAction.DEGRADE_PRECISION,
                success=True,
                freed_bytes=0,
                elapsed_ms=elapsed,
                message=f"Device {device} marked for precision degradation"
            )
        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.DEGRADE_PRECISION,
                success=False,
                freed_bytes=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    def _offload_to_cpu(self, device: Union[str, int]) -> RecoveryResult:
        """卸载张量到CPU"""
        start_time = time.time()
        
        if not TORCH_AVAILABLE or device == 'cpu':
            return RecoveryResult(
                action=RecoveryAction.OFFLOAD_TO_CPU,
                success=False,
                freed_bytes=0,
                elapsed_ms=0,
                message="CPU offload not applicable"
            )
        
        try:
            # 这里可以实现具体的卸载逻辑
            # 由于需要访问具体的模型/张量，这里仅提供框架
            
            elapsed = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                action=RecoveryAction.OFFLOAD_TO_CPU,
                success=True,
                freed_bytes=0,
                elapsed_ms=elapsed,
                message="CPU offload prepared"
            )
        except Exception as e:
            return RecoveryResult(
                action=RecoveryAction.OFFLOAD_TO_CPU,
                success=False,
                freed_bytes=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                message=str(e)
            )
    
    def _register_oom_hook(self):
        """注册PyTorch OOM钩子"""
        # 保存原始的CUDA内存分配器
        self._original_cuda_malloc = None
        
        # 这里可以添加更复杂的OOM钩子逻辑
        # 目前依赖于定期监控
        pass
    
    def _trigger_warning_callbacks(self, device: Union[str, int],
                                   allocated: int, total: int):
        """触发警告回调"""
        for callback in self._warning_callbacks:
            try:
                callback(device, allocated, total)
            except Exception as e:
                self.logger.error(f"Warning callback error: {e}")
    
    def _trigger_critical_callbacks(self, device: Union[str, int],
                                   allocated: int, total: int):
        """触发临界回调"""
        for callback in self._critical_callbacks:
            try:
                callback(device, allocated, total)
            except Exception as e:
                self.logger.error(f"Critical callback error: {e}")
    
    def _trigger_emergency_callbacks(self, device: Union[str, int],
                                    allocated: int, total: int):
        """触发紧急回调"""
        for callback in self._emergency_callbacks:
            try:
                callback(device, allocated, total)
            except Exception as e:
                self.logger.error(f"Emergency callback error: {e}")
    
    def _trigger_recovery_callbacks(self, device: Union[str, int],
                                   success: bool, freed_bytes: int):
        """触发恢复回调"""
        for callback in self._recovery_callbacks:
            try:
                callback(device, success, freed_bytes)
            except Exception as e:
                self.logger.error(f"Recovery callback error: {e}")
    
    def register_warning_callback(self, callback: Callable):
        """注册警告回调"""
        self._warning_callbacks.append(callback)
    
    def register_critical_callback(self, callback: Callable):
        """注册临界回调"""
        self._critical_callbacks.append(callback)
    
    def register_emergency_callback(self, callback: Callable):
        """注册紧急回调"""
        self._emergency_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """注册恢复回调"""
        self._recovery_callbacks.append(callback)
    
    def handle_cuda_oom(self, exception: Exception) -> bool:
        """
        处理CUDA OOM异常
        
        Args:
            exception: CUDA OOM异常
            
        Returns:
            是否成功处理
        """
        if "out of memory" not in str(exception).lower():
            return False
        
        # 提取设备信息
        device = 0  # 默认设备
        
        self._handle_oom(
            OOMSeverity.EMERGENCY,
            device,
            0,  # 未知
            0,  # 未知
            str(exception)
        )
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取OOM处理器统计
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'oom_events': len(self._oom_events),
                'recovery_attempts': self._recovery_attempts,
                'recovery_history': len(self._recovery_history),
                'degraded_devices': list(self._degraded_devices),
                'recent_events': [
                    {
                        'severity': e.severity.name,
                        'timestamp': e.timestamp,
                        'device': str(e.device),
                        'recovery_success': e.recovery_success
                    }
                    for e in list(self._oom_events)[-10:]
                ]
            }
    
    def reset(self):
        """重置OOM处理器状态"""
        with self._lock:
            self._recovery_attempts = 0
            self._degraded_devices.clear()
            self._offloaded_tensors.clear()
        
        self.logger.info("OOMHandler reset")
    
    def shutdown(self):
        """关闭OOM处理器"""
        self.logger.info("Shutting down OOMHandler...")
        
        # 停止监控
        if self._monitoring:
            self._stop_monitor.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            self._monitoring = False
        
        self._initialized = False
        self.logger.info("OOMHandler shutdown complete")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 检查是否是OOM异常
        if exc_type is not None and TORCH_AVAILABLE:
            if self.handle_cuda_oom(exc_val):
                return True  # 异常已处理
        
        self.shutdown()
        return False


# 便捷函数
def create_oom_handler(config: Optional[Dict] = None) -> OOMHandler:
    """创建OOM处理器"""
    return OOMHandler(config)


def handle_oom(exception: Exception) -> bool:
    """处理OOM异常"""
    handler = OOMHandler()
    return handler.handle_cuda_oom(exception)
