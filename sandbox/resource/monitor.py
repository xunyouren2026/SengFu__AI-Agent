"""
资源监控器
实时监控CPU/内存/磁盘使用情况
"""

import os
import time
import threading
import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"


class MonitorState(Enum):
    """监控器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class ResourceSnapshot:
    """资源快照"""
    timestamp: float
    cpu_percent: float = 0.0
    cpu_user: float = 0.0
    cpu_system: float = 0.0
    memory_used: int = 0
    memory_total: int = 0
    memory_percent: float = 0.0
    memory_rss: int = 0
    memory_vms: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    disk_read_count: int = 0
    disk_write_count: int = 0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    thread_count: int = 0
    file_descriptor_count: int = 0
    load_avg: tuple = (0.0, 0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'cpu': {
                'percent': self.cpu_percent,
                'user': self.cpu_user,
                'system': self.cpu_system
            },
            'memory': {
                'used': self.memory_used,
                'total': self.memory_total,
                'percent': self.memory_percent,
                'rss': self.memory_rss,
                'vms': self.memory_vms
            },
            'disk': {
                'read_bytes': self.disk_read_bytes,
                'write_bytes': self.disk_write_bytes,
                'read_count': self.disk_read_count,
                'write_count': self.disk_write_count
            },
            'network': {
                'bytes_sent': self.network_bytes_sent,
                'bytes_recv': self.network_bytes_recv
            },
            'process': {
                'count': self.process_count,
                'thread_count': self.thread_count,
                'fd_count': self.file_descriptor_count
            },
            'load_avg': list(self.load_avg)
        }


@dataclass
class ResourceThreshold:
    """资源阈值"""
    resource_type: ResourceType
    warning_threshold: float
    critical_threshold: float
    callback: Optional[Callable[[str, float], None]] = None


class ProcfsReader:
    """Linux /proc文件系统读取器"""
    
    def __init__(self):
        self._proc_path = Path("/proc")
        self._last_cpu_times: Optional[tuple] = None
        self._last_disk_io: Optional[tuple] = None
        self._last_net_io: Optional[tuple] = None
    
    def is_available(self) -> bool:
        """检查/proc是否可用"""
        return self._proc_path.exists()
    
    def read_cpu_times(self) -> tuple:
        """读取CPU时间"""
        try:
            with open(self._proc_path / "stat", "r") as f:
                line = f.readline()
                parts = line.split()
                # user, nice, system, idle, iowait, irq, softirq
                user = int(parts[1])
                nice = int(parts[2])
                system = int(parts[3])
                idle = int(parts[4])
                iowait = int(parts[5]) if len(parts) > 5 else 0
                irq = int(parts[6]) if len(parts) > 6 else 0
                softirq = int(parts[7]) if len(parts) > 7 else 0
                return (user, nice, system, idle, iowait, irq, softirq)
        except (IOError, IndexError):
            return (0, 0, 0, 0, 0, 0, 0)
    
    def read_memory_info(self) -> Dict[str, int]:
        """读取内存信息"""
        meminfo = {}
        try:
            with open(self._proc_path / "meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1])
                        # 转换为字节（如果单位是kB）
                        if len(parts) > 2 and parts[2] == 'kB':
                            value *= 1024
                        meminfo[key] = value
        except IOError:
            pass
        return meminfo
    
    def read_disk_io(self) -> tuple:
        """读取磁盘IO统计"""
        try:
            with open(self._proc_path / "diskstats", "r") as f:
                total_read = 0
                total_write = 0
                read_count = 0
                write_count = 0
                for line in f:
                    parts = line.split()
                    if len(parts) >= 11:
                        # skips fields: major, minor, device
                        read_count += int(parts[3])
                        total_read += int(parts[5]) * 512  # sectors to bytes
                        write_count += int(parts[7])
                        total_write += int(parts[9]) * 512
                return (total_read, total_write, read_count, write_count)
        except IOError:
            return (0, 0, 0, 0)
    
    def read_network_io(self) -> tuple:
        """读取网络IO统计"""
        try:
            with open(self._proc_path / "net" / "dev", "r") as f:
                # 跳过前两行标题
                next(f)
                next(f)
                total_recv = 0
                total_sent = 0
                for line in f:
                    parts = line.split()
                    if len(parts) >= 17:
                        # 接收字节和发送字节
                        total_recv += int(parts[1])
                        total_sent += int(parts[9])
                return (total_recv, total_sent)
        except IOError:
            return (0, 0)
    
    def read_load_avg(self) -> tuple:
        """读取负载平均值"""
        try:
            with open(self._proc_path / "loadavg", "r") as f:
                parts = f.read().split()
                return (float(parts[0]), float(parts[1]), float(parts[2]))
        except (IOError, IndexError, ValueError):
            return (0.0, 0.0, 0.0)
    
    def read_process_stats(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """读取进程统计"""
        pid = pid or os.getpid()
        stats = {}
        
        try:
            # 读取/proc/[pid]/stat
            stat_path = self._proc_path / str(pid) / "stat"
            with open(stat_path, "r") as f:
                content = f.read()
                # 处理命令名中可能包含的括号
                start = content.find('(')
                end = content.rfind(')')
                if start != -1 and end != -1:
                    parts = content[:start].split() + [content[start:end+1]] + content[end+1:].split()
                else:
                    parts = content.split()
                
                if len(parts) >= 24:
                    stats['rss'] = int(parts[23]) * 4096  # 页大小
                    stats['vms'] = int(parts[22])  # 虚拟内存
        except (IOError, IndexError, ValueError):
            pass
        
        try:
            # 读取/proc/[pid]/status
            status_path = self._proc_path / str(pid) / "status"
            with open(status_path, "r") as f:
                for line in f:
                    if line.startswith('Threads:'):
                        stats['threads'] = int(line.split()[1])
                    elif line.startswith('VmRSS:'):
                        stats['vm_rss'] = int(line.split()[1]) * 1024
                    elif line.startswith('VmSize:'):
                        stats['vm_size'] = int(line.split()[1]) * 1024
        except (IOError, IndexError, ValueError):
            pass
        
        try:
            # 统计文件描述符数量
            fd_path = self._proc_path / str(pid) / "fd"
            stats['fd_count'] = len(list(fd_path.iterdir())) if fd_path.exists() else 0
        except IOError:
            stats['fd_count'] = 0
        
        return stats
    
    def count_processes(self) -> int:
        """统计进程数量"""
        try:
            # 读取/proc目录下的数字目录
            count = 0
            for entry in self._proc_path.iterdir():
                if entry.name.isdigit():
                    count += 1
            return count
        except IOError:
            return 0


class ResourceMonitor:
    """
    资源监控器
    实时监控CPU/内存/磁盘使用情况
    """
    
    def __init__(
        self,
        sample_interval: float = 1.0,
        history_size: int = 100,
        pid: Optional[int] = None
    ):
        """
        初始化监控器
        
        Args:
            sample_interval: 采样间隔（秒）
            history_size: 历史记录大小
            pid: 要监控的进程ID（None表示系统级监控）
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.pid = pid or os.getpid()
        
        self._procfs = ProcfsReader()
        self._state = MonitorState.IDLE
        self._history: deque = deque(maxlen=history_size)
        self._thresholds: Dict[ResourceType, ResourceThreshold] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 上一次的值（用于计算增量）
        self._last_cpu_times: Optional[tuple] = None
        self._last_disk_io: Optional[tuple] = None
        self._last_net_io: Optional[tuple] = None
        self._last_snapshot_time: Optional[float] = None
    
    @property
    def state(self) -> MonitorState:
        """获取状态"""
        return self._state
    
    def set_threshold(
        self,
        resource_type: ResourceType,
        warning: float,
        critical: float,
        callback: Optional[Callable[[str, float], None]] = None
    ) -> None:
        """
        设置资源阈值
        
        Args:
            resource_type: 资源类型
            warning: 警告阈值
            critical: 严重阈值
            callback: 回调函数
        """
        self._thresholds[resource_type] = ResourceThreshold(
            resource_type=resource_type,
            warning_threshold=warning,
            critical_threshold=critical,
            callback=callback
        )
    
    def start(self) -> bool:
        """启动监控"""
        if self._state == MonitorState.RUNNING:
            return True
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self._state = MonitorState.RUNNING
        return True
    
    def stop(self) -> bool:
        """停止监控"""
        if self._state != MonitorState.RUNNING:
            return True
        
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._state = MonitorState.STOPPED
        return True
    
    def pause(self) -> bool:
        """暂停监控"""
        if self._state == MonitorState.RUNNING:
            self._state = MonitorState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """恢复监控"""
        if self._state == MonitorState.PAUSED:
            self._state = MonitorState.RUNNING
            return True
        return False
    
    def get_snapshot(self) -> ResourceSnapshot:
        """获取当前快照"""
        return self._collect_snapshot()
    
    def get_history(self) -> List[ResourceSnapshot]:
        """获取历史记录"""
        return list(self._history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._history:
            return {}
        
        snapshots = list(self._history)
        
        # 计算平均值
        avg_cpu = sum(s.cpu_percent for s in snapshots) / len(snapshots)
        avg_memory = sum(s.memory_percent for s in snapshots) / len(snapshots)
        
        # 计算最大值
        max_cpu = max(s.cpu_percent for s in snapshots)
        max_memory = max(s.memory_percent for s in snapshots)
        
        # 计算总IO
        if len(snapshots) >= 2:
            first = snapshots[0]
            last = snapshots[-1]
            total_disk_read = last.disk_read_bytes - first.disk_read_bytes
            total_disk_write = last.disk_write_bytes - first.disk_write_bytes
            total_net_sent = last.network_bytes_sent - first.network_bytes_sent
            total_net_recv = last.network_bytes_recv - first.network_bytes_recv
        else:
            total_disk_read = total_disk_write = 0
            total_net_sent = total_net_recv = 0
        
        return {
            'sample_count': len(snapshots),
            'duration': snapshots[-1].timestamp - snapshots[0].timestamp if snapshots else 0,
            'cpu': {
                'avg_percent': avg_cpu,
                'max_percent': max_cpu
            },
            'memory': {
                'avg_percent': avg_memory,
                'max_percent': max_memory
            },
            'disk_io': {
                'total_read_bytes': total_disk_read,
                'total_write_bytes': total_disk_write
            },
            'network_io': {
                'total_sent_bytes': total_net_sent,
                'total_recv_bytes': total_net_recv
            }
        }
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while not self._stop_event.is_set():
            if self._state == MonitorState.RUNNING:
                snapshot = self._collect_snapshot()
                self._history.append(snapshot)
                self._check_thresholds(snapshot)
            
            self._stop_event.wait(self.sample_interval)
    
    def _collect_snapshot(self) -> ResourceSnapshot:
        """收集资源快照"""
        timestamp = time.time()
        
        # CPU使用率
        cpu_times = self._procfs.read_cpu_times()
        cpu_percent = 0.0
        cpu_user = 0.0
        cpu_system = 0.0
        
        if self._last_cpu_times:
            # 计算CPU使用率
            last_total = sum(self._last_cpu_times)
            current_total = sum(cpu_times)
            total_diff = current_total - last_total
            
            if total_diff > 0:
                idle_diff = cpu_times[3] - self._last_cpu_times[3]
                cpu_percent = 100.0 * (1 - idle_diff / total_diff)
                cpu_user = 100.0 * (cpu_times[0] - self._last_cpu_times[0]) / total_diff
                cpu_system = 100.0 * (cpu_times[2] - self._last_cpu_times[2]) / total_diff
        
        self._last_cpu_times = cpu_times
        
        # 内存信息
        meminfo = self._procfs.read_memory_info()
        memory_total = meminfo.get('MemTotal', 0)
        memory_used = meminfo.get('MemTotal', 0) - meminfo.get('MemFree', 0) - meminfo.get('Buffers', 0) - meminfo.get('Cached', 0)
        memory_percent = 100.0 * memory_used / memory_total if memory_total > 0 else 0
        
        # 进程内存
        proc_stats = self._procfs.read_process_stats(self.pid)
        memory_rss = proc_stats.get('vm_rss', 0)
        memory_vms = proc_stats.get('vm_size', 0)
        
        # 磁盘IO
        disk_io = self._procfs.read_disk_io()
        disk_read_bytes = disk_io[0]
        disk_write_bytes = disk_io[1]
        disk_read_count = disk_io[2]
        disk_write_count = disk_io[3]
        
        # 网络IO
        net_io = self._procfs.read_network_io()
        network_bytes_recv = net_io[0]
        network_bytes_sent = net_io[1]
        
        # 进程统计
        process_count = self._procfs.count_processes()
        thread_count = proc_stats.get('threads', 1)
        file_descriptor_count = proc_stats.get('fd_count', 0)
        
        # 负载平均值
        load_avg = self._procfs.read_load_avg()
        
        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_user=cpu_user,
            cpu_system=cpu_system,
            memory_used=memory_used,
            memory_total=memory_total,
            memory_percent=memory_percent,
            memory_rss=memory_rss,
            memory_vms=memory_vms,
            disk_read_bytes=disk_read_bytes,
            disk_write_bytes=disk_write_bytes,
            disk_read_count=disk_read_count,
            disk_write_count=disk_write_count,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            process_count=process_count,
            thread_count=thread_count,
            file_descriptor_count=file_descriptor_count,
            load_avg=load_avg
        )
    
    def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """检查阈值"""
        # CPU阈值
        if ResourceType.CPU in self._thresholds:
            threshold = self._thresholds[ResourceType.CPU]
            if snapshot.cpu_percent >= threshold.critical_threshold:
                if threshold.callback:
                    threshold.callback('critical', snapshot.cpu_percent)
            elif snapshot.cpu_percent >= threshold.warning_threshold:
                if threshold.callback:
                    threshold.callback('warning', snapshot.cpu_percent)
        
        # 内存阈值
        if ResourceType.MEMORY in self._thresholds:
            threshold = self._thresholds[ResourceType.MEMORY]
            if snapshot.memory_percent >= threshold.critical_threshold:
                if threshold.callback:
                    threshold.callback('critical', snapshot.memory_percent)
            elif snapshot.memory_percent >= threshold.warning_threshold:
                if threshold.callback:
                    threshold.callback('warning', snapshot.memory_percent)
    
    def __enter__(self) -> 'ResourceMonitor':
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.stop()


class CgroupMonitor:
    """Cgroup资源监控器"""
    
    def __init__(self, cgroup_path: str = "/sys/fs/cgroup"):
        self.cgroup_path = Path(cgroup_path)
    
    def is_available(self) -> bool:
        """检查cgroup是否可用"""
        return self.cgroup_path.exists()
    
    def read_cpu_stats(self, cgroup_name: str = "") -> Dict[str, Any]:
        """读取CPU统计"""
        stats = {}
        
        # cgroup v2路径
        cpu_stat_path = self.cgroup_path / cgroup_name / "cpu.stat"
        if cpu_stat_path.exists():
            try:
                with open(cpu_stat_path, "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 2:
                            stats[parts[0]] = int(parts[1])
            except IOError:
                pass
        
        # cgroup v1路径
        cpuacct_usage_path = self.cgroup_path / "cpuacct" / cgroup_name / "cpuacct.usage"
        if cpuacct_usage_path.exists():
            try:
                with open(cpuacct_usage_path, "r") as f:
                    stats['usage'] = int(f.read().strip())
            except IOError:
                pass
        
        return stats
    
    def read_memory_stats(self, cgroup_name: str = "") -> Dict[str, Any]:
        """读取内存统计"""
        stats = {}
        
        # cgroup v2路径
        memory_stat_path = self.cgroup_path / cgroup_name / "memory.stat"
        if memory_stat_path.exists():
            try:
                with open(memory_stat_path, "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 2:
                            stats[parts[0]] = int(parts[1])
            except IOError:
                pass
        
        # 读取当前使用量
        memory_current_path = self.cgroup_path / cgroup_name / "memory.current"
        if memory_current_path.exists():
            try:
                with open(memory_current_path, "r") as f:
                    stats['current'] = int(f.read().strip())
            except IOError:
                pass
        
        return stats
    
    def read_io_stats(self, cgroup_name: str = "") -> Dict[str, Any]:
        """读取IO统计"""
        stats = {}
        
        # cgroup v2路径
        io_stat_path = self.cgroup_path / cgroup_name / "io.stat"
        if io_stat_path.exists():
            try:
                with open(io_stat_path, "r") as f:
                    for line in f:
                        # 格式: major:minor rbytes=... wbytes=...
                        parts = line.split()
                        for part in parts[1:]:  # 跳过major:minor
                            if '=' in part:
                                key, value = part.split('=')
                                stats[key] = stats.get(key, 0) + int(value)
            except IOError:
                pass
        
        return stats
