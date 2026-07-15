"""
资源配额管理 - 执行资源限制
"""
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    TIME = "time"
    PROCESSES = "processes"
    FILE_DESCRIPTORS = "file_descriptors"


@dataclass
class ResourceQuota:
    """资源配额"""
    cpu_cores: float = 1.0              # CPU核心数
    memory_mb: int = 512                # 内存限制(MB)
    disk_mb: int = 1024                 # 磁盘限制(MB)
    network_bandwidth_mbps: int = 10    # 网络带宽(Mbps)
    time_seconds: int = 60              # 执行时间限制(秒)
    max_processes: int = 10             # 最大进程数
    max_file_descriptors: int = 100     # 最大文件描述符数
    max_open_files: int = 50            # 最大打开文件数
    max_threads: int = 10               # 最大线程数


@dataclass
class ResourceUsage:
    """资源使用情况"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    disk_mb: float = 0.0
    network_bytes_in: int = 0
    network_bytes_out: int = 0
    time_elapsed: float = 0.0
    process_count: int = 0
    file_descriptor_count: int = 0
    thread_count: int = 0


@dataclass
class QuotaViolation:
    """配额违规"""
    resource_type: ResourceType
    limit: float
    actual: float
    timestamp: float
    message: str


class ResourceQuotaManager:
    """资源配额管理器"""
    
    def __init__(self, default_quota: Optional[ResourceQuota] = None):
        self._default_quota = default_quota or ResourceQuota()
        self._quotas: Dict[str, ResourceQuota] = {}
        self._usage: Dict[str, ResourceUsage] = {}
        self._violations: Dict[str, List[QuotaViolation]] = defaultdict(list)
        self._callbacks: List[Callable[[QuotaViolation], None]] = []
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def set_quota(self, execution_id: str, quota: ResourceQuota) -> None:
        """设置执行配额"""
        with self._lock:
            self._quotas[execution_id] = quota
            self._usage[execution_id] = ResourceUsage()
    
    def get_quota(self, execution_id: str) -> ResourceQuota:
        """获取配额"""
        with self._lock:
            return self._quotas.get(execution_id, self._default_quota)
    
    def update_usage(self, execution_id: str, **kwargs) -> None:
        """更新资源使用"""
        with self._lock:
            if execution_id not in self._usage:
                self._usage[execution_id] = ResourceUsage()
            
            usage = self._usage[execution_id]
            for key, value in kwargs.items():
                if hasattr(usage, key):
                    setattr(usage, key, value)
    
    def check_quota(self, execution_id: str) -> List[QuotaViolation]:
        """检查配额是否超限"""
        violations = []
        
        with self._lock:
            quota = self._quotas.get(execution_id, self._default_quota)
            usage = self._usage.get(execution_id, ResourceUsage())
        
        # 检查CPU
        if usage.cpu_percent > quota.cpu_cores * 100:
            violations.append(QuotaViolation(
                resource_type=ResourceType.CPU,
                limit=quota.cpu_cores * 100,
                actual=usage.cpu_percent,
                timestamp=time.time(),
                message=f"CPU使用率超限: {usage.cpu_percent:.1f}% > {quota.cpu_cores * 100:.1f}%"
            ))
        
        # 检查内存
        if usage.memory_mb > quota.memory_mb:
            violations.append(QuotaViolation(
                resource_type=ResourceType.MEMORY,
                limit=quota.memory_mb,
                actual=usage.memory_mb,
                timestamp=time.time(),
                message=f"内存使用超限: {usage.memory_mb:.1f}MB > {quota.memory_mb}MB"
            ))
        
        # 检查磁盘
        if usage.disk_mb > quota.disk_mb:
            violations.append(QuotaViolation(
                resource_type=ResourceType.DISK,
                limit=quota.disk_mb,
                actual=usage.disk_mb,
                timestamp=time.time(),
                message=f"磁盘使用超限: {usage.disk_mb:.1f}MB > {quota.disk_mb}MB"
            ))
        
        # 检查时间
        if usage.time_elapsed > quota.time_seconds:
            violations.append(QuotaViolation(
                resource_type=ResourceType.TIME,
                limit=quota.time_seconds,
                actual=usage.time_elapsed,
                timestamp=time.time(),
                message=f"执行时间超限: {usage.time_elapsed:.1f}s > {quota.time_seconds}s"
            ))
        
        # 检查进程数
        if usage.process_count > quota.max_processes:
            violations.append(QuotaViolation(
                resource_type=ResourceType.PROCESSES,
                limit=quota.max_processes,
                actual=usage.process_count,
                timestamp=time.time(),
                message=f"进程数超限: {usage.process_count} > {quota.max_processes}"
            ))
        
        # 检查文件描述符
        if usage.file_descriptor_count > quota.max_file_descriptors:
            violations.append(QuotaViolation(
                resource_type=ResourceType.FILE_DESCRIPTORS,
                limit=quota.max_file_descriptors,
                actual=usage.file_descriptor_count,
                timestamp=time.time(),
                message=f"文件描述符超限: {usage.file_descriptor_count} > {quota.max_file_descriptors}"
            ))
        
        # 记录违规
        with self._lock:
            self._violations[execution_id].extend(violations)
        
        # 触发回调
        for violation in violations:
            for callback in self._callbacks:
                try:
                    callback(violation)
                except Exception:
                    pass
        
        return violations
    
    def add_violation_callback(self, callback: Callable[[QuotaViolation], None]) -> None:
        """添加违规回调"""
        self._callbacks.append(callback)
    
    def get_violations(self, execution_id: str) -> List[QuotaViolation]:
        """获取违规记录"""
        with self._lock:
            return self._violations.get(execution_id, []).copy()
    
    def get_usage(self, execution_id: str) -> ResourceUsage:
        """获取使用情况"""
        with self._lock:
            return self._usage.get(execution_id, ResourceUsage())
    
    def clear_execution(self, execution_id: str) -> None:
        """清理执行记录"""
        with self._lock:
            self._quotas.pop(execution_id, None)
            self._usage.pop(execution_id, None)
            self._violations.pop(execution_id, None)
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """启动监控"""
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                with self._lock:
                    execution_ids = list(self._quotas.keys())
                
                for exec_id in execution_ids:
                    self.check_quota(exec_id)
                
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "active_executions": len(self._quotas),
                "total_violations": sum(len(v) for v in self._violations.values()),
                "executions": {
                    exec_id: {
                        "usage": {
                            "cpu_percent": usage.cpu_percent,
                            "memory_mb": usage.memory_mb,
                            "time_elapsed": usage.time_elapsed
                        },
                        "violations": len(self._violations.get(exec_id, []))
                    }
                    for exec_id, usage in self._usage.items()
                }
            }


class ProcessMonitor:
    """进程监控器"""
    
    def __init__(self, pid: int):
        self._pid = pid
        self._start_time = time.time()
    
    def get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        try:
            import os
            # 读取/proc/[pid]/stat
            stat_path = f"/proc/{self._pid}/stat"
            if os.path.exists(stat_path):
                with open(stat_path, 'r') as f:
                    stat = f.read().split()
                
                utime = int(stat[13]) / 100.0
                stime = int(stat[14]) / 100.0
                total_time = utime + stime
                
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    return (total_time / elapsed) * 100
        except Exception:
            pass
        return 0.0
    
    def get_memory_mb(self) -> float:
        """获取内存使用(MB)"""
        try:
            import os
            statm_path = f"/proc/{self._pid}/statm"
            if os.path.exists(statm_path):
                with open(statm_path, 'r') as f:
                    statm = f.read().split()
                
                resident = int(statm[1])
                page_size = 4096  # 通常4KB
                return (resident * page_size) / (1024 * 1024)
        except Exception:
            pass
        return 0.0
    
    def get_file_descriptors(self) -> int:
        """获取文件描述符数量"""
        try:
            import os
            fd_path = f"/proc/{self._pid}/fd"
            if os.path.exists(fd_path):
                return len(os.listdir(fd_path))
        except Exception:
            pass
        return 0
    
    def get_thread_count(self) -> int:
        """获取线程数"""
        try:
            import os
            task_path = f"/proc/{self._pid}/task"
            if os.path.exists(task_path):
                return len(os.listdir(task_path))
        except Exception:
            pass
        return 0
    
    def get_usage(self) -> ResourceUsage:
        """获取完整使用情况"""
        return ResourceUsage(
            cpu_percent=self.get_cpu_percent(),
            memory_mb=self.get_memory_mb(),
            time_elapsed=time.time() - self._start_time,
            file_descriptor_count=self.get_file_descriptors(),
            thread_count=self.get_thread_count()
        )
