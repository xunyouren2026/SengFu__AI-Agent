"""
系统监控器模块
提供CPU/内存/磁盘/网络实时监控功能
"""

import os
import time
import threading
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import json

# 可选依赖: psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"
    SYSTEM = "system"


@dataclass
class CPUMetrics:
    """CPU指标数据类"""
    percent: float
    per_cpu: List[float]
    user: float
    system: float
    idle: float
    iowait: float
    load_avg: Tuple[float, float, float]
    cores: int
    physical_cores: int
    frequency: float


@dataclass
class MemoryMetrics:
    """内存指标数据类"""
    total: int
    available: int
    used: int
    free: int
    percent: float
    swap_total: int
    swap_used: int
    swap_free: int
    swap_percent: float
    cached: int
    buffers: int


@dataclass
class DiskMetrics:
    """磁盘指标数据类"""
    partitions: List[Dict[str, Any]]
    total_read: int
    total_write: int
    read_speed: float
    write_speed: float
    busy_percent: float


@dataclass
class NetworkMetrics:
    """网络指标数据类"""
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    send_speed: float
    recv_speed: float
    connections: int
    interfaces: List[Dict[str, Any]]


@dataclass
class SystemMetrics:
    """系统综合指标数据类"""
    timestamp: datetime
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: DiskMetrics
    network: NetworkMetrics
    uptime: float
    boot_time: datetime
    processes: int
    threads: int


@dataclass
class Alert:
    """告警数据类"""
    metric_type: MetricType
    metric_name: str
    current_value: float
    threshold: float
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: datetime


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        """初始化告警管理器"""
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._callbacks: List[Callable[[Alert], None]] = []
        self._alerts: List[Alert] = []
        self._lock = threading.Lock()
        
        # 默认阈值
        self._set_default_thresholds()
    
    def _set_default_thresholds(self) -> None:
        """设置默认阈值"""
        self._thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 80, 'critical': 95},
            'disk_percent': {'warning': 80, 'critical': 95},
            'swap_percent': {'warning': 50, 'critical': 80},
        }
    
    def set_threshold(self, metric_name: str, warning: float, critical: float) -> None:
        """设置阈值"""
        self._thresholds[metric_name] = {'warning': warning, 'critical': critical}
    
    def check_threshold(self, metric_type: MetricType, metric_name: str,
                        value: float) -> Optional[Alert]:
        """检查是否超过阈值"""
        thresholds = self._thresholds.get(metric_name)
        if not thresholds:
            return None
        
        severity = None
        threshold = None
        
        if value >= thresholds['critical']:
            severity = 'critical'
            threshold = thresholds['critical']
        elif value >= thresholds['warning']:
            severity = 'warning'
            threshold = thresholds['warning']
        
        if severity:
            alert = Alert(
                metric_type=metric_type,
                metric_name=metric_name,
                current_value=value,
                threshold=threshold,
                severity=severity,
                message=f"{metric_name} = {value:.1f}% ({severity} threshold: {threshold}%)",
                timestamp=datetime.now()
            )
            return alert
        
        return None
    
    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """注册告警回调"""
        self._callbacks.append(callback)
    
    def trigger_alert(self, alert: Alert) -> None:
        """触发告警"""
        with self._lock:
            self._alerts.append(alert)
        
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    def get_alerts(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        with self._lock:
            return self._alerts[-limit:]
    
    def clear_alerts(self) -> None:
        """清空告警"""
        with self._lock:
            self._alerts.clear()


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, sample_interval: float = 1.0,
                 history_size: int = 300,
                 alert_manager: Optional[AlertManager] = None):
        """
        初始化系统监控器
        
        Args:
            sample_interval: 采样间隔（秒）
            history_size: 历史记录大小
            alert_manager: 告警管理器
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.alert_manager = alert_manager or AlertManager()
        
        # 历史数据
        self._cpu_history: deque = deque(maxlen=history_size)
        self._memory_history: deque = deque(maxlen=history_size)
        self._disk_history: deque = deque(maxlen=history_size)
        self._network_history: deque = deque(maxlen=history_size)
        
        # 上一次的网络和磁盘IO（用于计算速度）
        self._last_net_io: Optional[Any] = None
        self._last_disk_io: Optional[Any] = None
        self._last_sample_time: float = 0
        
        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
        self._lock = threading.Lock()
        
        # 回调函数
        self._metric_callbacks: List[Callable[[SystemMetrics], None]] = []
    
    def get_cpu_metrics(self) -> CPUMetrics:
        """获取CPU指标"""
        if not PSUTIL_AVAILABLE:
            # 返回默认值
            return CPUMetrics(
                percent=0.0,
                per_cpu=[],
                user=0.0,
                system=0.0,
                idle=0.0,
                iowait=0.0,
                load_avg=(0, 0, 0),
                cores=0,
                physical_cores=0,
                frequency=0.0
            )
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        per_cpu = psutil.cpu_percent(percpu=True)
        
        # CPU时间
        cpu_times = psutil.cpu_times()
        
        # 负载平均值
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            load_avg = (0, 0, 0)
        
        # CPU核心数
        cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        # CPU频率
        try:
            freq = psutil.cpu_freq()
            frequency = freq.current if freq else 0
        except Exception:
            frequency = 0
        
        return CPUMetrics(
            percent=cpu_percent,
            per_cpu=list(per_cpu),
            user=cpu_times.user,
            system=cpu_times.system,
            idle=cpu_times.idle,
            iowait=getattr(cpu_times, 'iowait', 0),
            load_avg=load_avg,
            cores=cores or 0,
            physical_cores=physical_cores or 0,
            frequency=frequency
        )
    
    def get_memory_metrics(self) -> MemoryMetrics:
        """获取内存指标"""
        if not PSUTIL_AVAILABLE:
            # 返回默认值
            return MemoryMetrics(
                total=0,
                available=0,
                used=0,
                free=0,
                percent=0.0,
                swap_total=0,
                swap_used=0,
                swap_free=0,
                swap_percent=0.0,
                cached=0,
                buffers=0
            )
        
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return MemoryMetrics(
            total=mem.total,
            available=mem.available,
            used=mem.used,
            free=mem.free,
            percent=mem.percent,
            swap_total=swap.total,
            swap_used=swap.used,
            swap_free=swap.free,
            swap_percent=swap.percent,
            cached=getattr(mem, 'cached', 0),
            buffers=getattr(mem, 'buffers', 0)
        )
    
    def get_disk_metrics(self) -> DiskMetrics:
        """获取磁盘指标"""
        if not PSUTIL_AVAILABLE:
            # 返回默认值
            return DiskMetrics(
                partitions=[],
                total_read=0,
                total_write=0,
                read_speed=0.0,
                write_speed=0.0,
                busy_percent=0.0
            )
        
        # 分区信息
        partitions = []
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                partitions.append({
                    'device': part.device,
                    'mountpoint': part.mountpoint,
                    'fstype': part.fstype,
                    'opts': part.opts,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                })
            except (PermissionError, psutil.AccessDenied):
                continue
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        read_speed = 0.0
        write_speed = 0.0
        busy_percent = 0.0
        
        if disk_io:
            current_time = time.time()
            
            if self._last_disk_io and self._last_sample_time > 0:
                time_delta = current_time - self._last_sample_time
                if time_delta > 0:
                    read_speed = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta
                    write_speed = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta
            
            self._last_disk_io = disk_io
        
        return DiskMetrics(
            partitions=partitions,
            total_read=disk_io.read_bytes if disk_io else 0,
            total_write=disk_io.write_bytes if disk_io else 0,
            read_speed=read_speed,
            write_speed=write_speed,
            busy_percent=busy_percent
        )
    
    def get_network_metrics(self) -> NetworkMetrics:
        """获取网络指标"""
        if not PSUTIL_AVAILABLE:
            # 返回默认值
            return NetworkMetrics(
                bytes_sent=0,
                bytes_recv=0,
                packets_sent=0,
                packets_recv=0,
                send_speed=0.0,
                recv_speed=0.0,
                connections=0,
                interfaces=[]
            )
        
        net_io = psutil.net_io_counters()
        
        send_speed = 0.0
        recv_speed = 0.0
        
        if net_io:
            current_time = time.time()
            
            if self._last_net_io and self._last_sample_time > 0:
                time_delta = current_time - self._last_sample_time
                if time_delta > 0:
                    send_speed = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_delta
                    recv_speed = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_delta
            
            self._last_net_io = net_io
        
        # 网络接口
        interfaces = []
        for name, addrs in psutil.net_if_addrs().items():
            interface_info = {'name': name, 'addresses': []}
            for addr in addrs:
                interface_info['addresses'].append({
                    'family': str(addr.family),
                    'address': addr.address,
                    'netmask': addr.netmask,
                    'broadcast': addr.broadcast
                })
            interfaces.append(interface_info)
        
        # 连接数
        try:
            connections = len(psutil.net_connections())
        except (psutil.AccessDenied, PermissionError):
            connections = 0
        
        return NetworkMetrics(
            bytes_sent=net_io.bytes_sent if net_io else 0,
            bytes_recv=net_io.bytes_recv if net_io else 0,
            packets_sent=net_io.packets_sent if net_io else 0,
            packets_recv=net_io.packets_recv if net_io else 0,
            send_speed=send_speed,
            recv_speed=recv_speed,
            connections=connections,
            interfaces=interfaces
        )
    
    def get_system_metrics(self) -> SystemMetrics:
        """获取系统综合指标"""
        cpu = self.get_cpu_metrics()
        memory = self.get_memory_metrics()
        disk = self.get_disk_metrics()
        network = self.get_network_metrics()
        
        # 更新采样时间
        self._last_sample_time = time.time()
        
        if PSUTIL_AVAILABLE:
            # 系统信息
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = time.time() - psutil.boot_time()
            
            # 进程和线程数
            processes = len(psutil.pids())
            threads = sum(p.num_threads() for p in psutil.process_iter(['num_threads'])
                         if p.info.get('num_threads'))
        else:
            boot_time = datetime.now()
            uptime = 0.0
            processes = 0
            threads = 0
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu=cpu,
            memory=memory,
            disk=disk,
            network=network,
            uptime=uptime,
            boot_time=boot_time,
            processes=processes,
            threads=threads
        )
        
        # 检查告警
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """检查告警"""
        # CPU告警
        alert = self.alert_manager.check_threshold(
            MetricType.CPU, 'cpu_percent', metrics.cpu.percent
        )
        if alert:
            self.alert_manager.trigger_alert(alert)
        
        # 内存告警
        alert = self.alert_manager.check_threshold(
            MetricType.MEMORY, 'memory_percent', metrics.memory.percent
        )
        if alert:
            self.alert_manager.trigger_alert(alert)
        
        # Swap告警
        alert = self.alert_manager.check_threshold(
            MetricType.MEMORY, 'swap_percent', metrics.memory.swap_percent
        )
        if alert:
            self.alert_manager.trigger_alert(alert)
        
        # 磁盘告警
        for partition in metrics.disk.partitions:
            alert = self.alert_manager.check_threshold(
                MetricType.DISK, 'disk_percent', partition['percent']
            )
            if alert:
                alert.message = f"{partition['mountpoint']}: {alert.message}"
                self.alert_manager.trigger_alert(alert)
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitor_running:
            return
        
        self._monitor_running = True
        
        def monitor_loop():
            while self._monitor_running:
                try:
                    metrics = self.get_system_metrics()
                    
                    with self._lock:
                        self._cpu_history.append(metrics.cpu)
                        self._memory_history.append(metrics.memory)
                        self._disk_history.append(metrics.disk)
                        self._network_history.append(metrics.network)
                    
                    # 触发回调
                    for callback in self._metric_callbacks:
                        try:
                            callback(metrics)
                        except Exception as e:
                            logger.error(f"指标回调执行失败: {e}")
                    
                except Exception as e:
                    logger.error(f"监控采样失败: {e}")
                
                time.sleep(self.sample_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logger.info("系统监控已停止")
    
    def register_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """注册指标回调"""
        self._metric_callbacks.append(callback)
    
    def get_cpu_history(self, seconds: Optional[int] = None) -> List[CPUMetrics]:
        """获取CPU历史数据"""
        with self._lock:
            if seconds:
                count = int(seconds / self.sample_interval)
                return list(self._cpu_history)[-count:]
            return list(self._cpu_history)
    
    def get_memory_history(self, seconds: Optional[int] = None) -> List[MemoryMetrics]:
        """获取内存历史数据"""
        with self._lock:
            if seconds:
                count = int(seconds / self.sample_interval)
                return list(self._memory_history)[-count:]
            return list(self._memory_history)
    
    def get_disk_history(self, seconds: Optional[int] = None) -> List[DiskMetrics]:
        """获取磁盘历史数据"""
        with self._lock:
            if seconds:
                count = int(seconds / self.sample_interval)
                return list(self._disk_history)[-count:]
            return list(self._disk_history)
    
    def get_network_history(self, seconds: Optional[int] = None) -> List[NetworkMetrics]:
        """获取网络历史数据"""
        with self._lock:
            if seconds:
                count = int(seconds / self.sample_interval)
                return list(self._network_history)[-count:]
            return list(self._network_history)
    
    def get_process_stats(self, top_n: int = 10,
                          sort_by: str = 'cpu') -> List[Dict[str, Any]]:
        """
        获取进程统计
        
        Args:
            top_n: 返回前N个进程
            sort_by: 排序字段 ('cpu', 'memory', 'io')
            
        Returns:
            进程统计列表
        """
        if not PSUTIL_AVAILABLE:
            return []
        
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                processes.append({
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'username': pinfo['username'],
                    'cpu_percent': pinfo['cpu_percent'] or 0,
                    'memory_percent': pinfo['memory_percent'] or 0,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # 排序
        if sort_by == 'cpu':
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        elif sort_by == 'memory':
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
        
        return processes[:top_n]
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        
        info = {
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        if PSUTIL_AVAILABLE:
            info.update({
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_physical': psutil.cpu_count(logical=False),
                'memory_total': psutil.virtual_memory().total,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            })
        else:
            info.update({
                'cpu_count': 0,
                'cpu_physical': 0,
                'memory_total': 0,
                'boot_time': datetime.now().isoformat(),
            })
        
        return info
    
    def export_metrics(self, format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        导出指标数据
        
        Args:
            format: 导出格式 ('dict', 'json')
            
        Returns:
            导出的数据
        """
        metrics = self.get_system_metrics()
        
        data = {
            'timestamp': metrics.timestamp.isoformat(),
            'cpu': {
                'percent': metrics.cpu.percent,
                'per_cpu': metrics.cpu.per_cpu,
                'load_avg': list(metrics.cpu.load_avg),
                'cores': metrics.cpu.cores,
            },
            'memory': {
                'total': metrics.memory.total,
                'used': metrics.memory.used,
                'percent': metrics.memory.percent,
                'swap_percent': metrics.memory.swap_percent,
            },
            'disk': {
                'partitions': metrics.disk.partitions,
                'read_speed': metrics.disk.read_speed,
                'write_speed': metrics.disk.write_speed,
            },
            'network': {
                'send_speed': metrics.network.send_speed,
                'recv_speed': metrics.network.recv_speed,
                'connections': metrics.network.connections,
            },
            'system': {
                'uptime': metrics.uptime,
                'processes': metrics.processes,
                'threads': metrics.threads,
            }
        }
        
        if format == 'json':
            return json.dumps(data, indent=2)
        return data


class ProcessMonitor:
    """进程监控器"""
    
    def __init__(self, pid: int, sample_interval: float = 1.0):
        """
        初始化进程监控器
        
        Args:
            pid: 进程ID
            sample_interval: 采样间隔
        """
        self.pid = pid
        self.sample_interval = sample_interval
        self._process: Optional[Any] = None
        self._history: deque = deque(maxlen=300)
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
    
    def attach(self) -> bool:
        """附加到进程"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil 不可用，无法附加到进程")
            return False
        
        try:
            self._process = psutil.Process(self.pid)
            return True
        except psutil.NoSuchProcess:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取进程统计"""
        if not PSUTIL_AVAILABLE:
            return {}
        
        if not self._process:
            if not self.attach():
                return {}
        
        try:
            cpu_times = self._process.cpu_times()
            mem_info = self._process.memory_info()
            io_counters = self._process.io_counters() if hasattr(self._process, 'io_counters') else None
            
            return {
                'pid': self.pid,
                'name': self._process.name(),
                'status': self._process.status(),
                'cpu_percent': self._process.cpu_percent(interval=0.1),
                'memory_percent': self._process.memory_percent(),
                'memory_rss': mem_info.rss,
                'memory_vms': mem_info.vms,
                'num_threads': self._process.num_threads(),
                'num_handles': self._process.num_handles() if hasattr(self._process, 'num_handles') else 0,
                'cpu_times': {
                    'user': cpu_times.user,
                    'system': cpu_times.system,
                },
                'io': {
                    'read_bytes': io_counters.read_bytes if io_counters else 0,
                    'write_bytes': io_counters.write_bytes if io_counters else 0,
                } if io_counters else None,
                'timestamp': datetime.now().isoformat()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._monitor_running:
            return
        
        self._monitor_running = True
        
        def monitor_loop():
            while self._monitor_running:
                stats = self.get_stats()
                if stats:
                    self._history.append(stats)
                else:
                    self._monitor_running = False
                    break
                time.sleep(self.sample_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史数据"""
        return list(self._history)
