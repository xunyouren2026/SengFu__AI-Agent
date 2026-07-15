"""
进程管理器模块
提供进程启动/停止/监控/信号发送功能
"""

import os
import sys
import signal
import subprocess
import threading
import time
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
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

# psutil 常量 fallback (用于优先级设置等)
if PSUTIL_AVAILABLE:
    _BELOW_NORMAL_PRIORITY_CLASS = getattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS', 0)
    _NORMAL_PRIORITY_CLASS = getattr(psutil, 'NORMAL_PRIORITY_CLASS', 0)
    _HIGH_PRIORITY_CLASS = getattr(psutil, 'HIGH_PRIORITY_CLASS', 0)
    _REALTIME_PRIORITY_CLASS = getattr(psutil, 'REALTIME_PRIORITY_CLASS', 0)
else:
    _BELOW_NORMAL_PRIORITY_CLASS = 0
    _NORMAL_PRIORITY_CLASS = 0
    _HIGH_PRIORITY_CLASS = 0
    _REALTIME_PRIORITY_CLASS = 0


class ProcessState(Enum):
    """进程状态枚举"""
    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    ZOMBIE = "zombie"
    UNKNOWN = "unknown"


class ProcessPriority(Enum):
    """进程优先级枚举"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    REALTIME = "realtime"


@dataclass
class ProcessInfo:
    """进程信息数据类"""
    pid: int
    name: str
    command: str
    state: ProcessState
    cpu_percent: float
    memory_percent: float
    memory_info: Dict[str, int]
    create_time: datetime
    running_time: float
    num_threads: int
    num_handles: int
    username: str
    exe: Optional[str]
    cwd: Optional[str]
    environ: Dict[str, str]
    connections: List[Dict[str, Any]]
    open_files: List[str]
    parent_pid: Optional[int]
    children_pids: List[int]


@dataclass
class ProcessResult:
    """进程操作结果"""
    success: bool
    pid: Optional[int]
    operation: str
    message: str
    error: Optional[str] = None
    data: Any = None


class ProcessManager:
    """进程管理器"""
    
    def __init__(self, auto_cleanup: bool = True,
                 max_processes: int = 100):
        """
        初始化进程管理器
        
        Args:
            auto_cleanup: 是否自动清理已结束的进程
            max_processes: 最大管理进程数
        """
        self.auto_cleanup = auto_cleanup
        self.max_processes = max_processes
        self._managed_processes: Dict[int, subprocess.Popen] = {}
        self._process_callbacks: Dict[int, List[Callable]] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
        self._lock = threading.Lock()
    
    def start_process(self, command: Union[str, List[str]],
                      cwd: Optional[str] = None,
                      env: Optional[Dict[str, str]] = None,
                      shell: bool = False,
                      stdout: Optional[str] = None,
                      stderr: Optional[str] = None,
                      stdin: Optional[str] = None,
                      detached: bool = False,
                      priority: Optional[ProcessPriority] = None) -> ProcessResult:
        """
        启动新进程
        
        Args:
            command: 命令或命令列表
            cwd: 工作目录
            env: 环境变量
            shell: 是否使用shell
            stdout: 标准输出重定向
            stderr: 标准错误重定向
            stdin: 标准输入
            detached: 是否分离运行
            priority: 进程优先级
            
        Returns:
            进程操作结果
        """
        try:
            # 准备参数
            kwargs: Dict[str, Any] = {
                'shell': shell,
                'cwd': cwd,
            }
            
            if env:
                kwargs['env'] = {**os.environ, **env}
            
            if stdout:
                kwargs['stdout'] = open(stdout, 'w')
            elif not detached:
                kwargs['stdout'] = subprocess.PIPE
            
            if stderr:
                kwargs['stderr'] = open(stderr, 'w')
            elif not detached:
                kwargs['stderr'] = subprocess.PIPE
            
            if stdin:
                kwargs['stdin'] = subprocess.PIPE
            
            if detached:
                kwargs['stdout'] = subprocess.DEVNULL
                kwargs['stderr'] = subprocess.DEVNULL
                kwargs['stdin'] = subprocess.DEVNULL
            
            # 启动进程
            if isinstance(command, str) and not shell:
                command = command.split()
            
            proc = subprocess.Popen(command, **kwargs)
            
            # 设置优先级
            if priority:
                self._set_priority(proc.pid, priority)
            
            # 管理进程
            with self._lock:
                if len(self._managed_processes) >= self.max_processes:
                    self._cleanup_finished()
                
                if len(self._managed_processes) < self.max_processes:
                    self._managed_processes[proc.pid] = proc
            
            result = ProcessResult(
                success=True,
                pid=proc.pid,
                operation="start",
                message=f"进程启动成功，PID: {proc.pid}"
            )
            
            logger.info(f"启动进程: {command}, PID: {proc.pid}")
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=None,
                operation="start",
                message=f"进程启动失败: {e}",
                error=str(e)
            )
            logger.error(f"启动进程失败: {command}, 错误: {e}")
            return result
    
    def _set_priority(self, pid: int, priority: ProcessPriority) -> bool:
        """设置进程优先级"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil 不可用，无法设置进程优先级")
            return False
        
        try:
            p = psutil.Process(pid)
            
            if sys.platform == 'win32':
                priority_map = {
                    ProcessPriority.LOW: _BELOW_NORMAL_PRIORITY_CLASS,
                    ProcessPriority.NORMAL: _NORMAL_PRIORITY_CLASS,
                    ProcessPriority.HIGH: _HIGH_PRIORITY_CLASS,
                    ProcessPriority.REALTIME: _REALTIME_PRIORITY_CLASS,
                }
            else:
                priority_map = {
                    ProcessPriority.LOW: 10,
                    ProcessPriority.NORMAL: 0,
                    ProcessPriority.HIGH: -5,
                    ProcessPriority.REALTIME: -20,
                }
            
            p.nice(priority_map[priority])
            
            return True
        except Exception as e:
            logger.warning(f"设置优先级失败: {e}")
            return False
    
    def stop_process(self, pid: int, timeout: int = 10,
                     force: bool = False) -> ProcessResult:
        """
        停止进程
        
        Args:
            pid: 进程ID
            timeout: 等待超时时间
            force: 是否强制终止
            
        Returns:
            进程操作结果
        """
        try:
            # 检查是否是管理的进程
            proc = None
            with self._lock:
                proc = self._managed_processes.get(pid)
            
            if proc:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
                
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                
                with self._lock:
                    self._managed_processes.pop(pid, None)
            elif PSUTIL_AVAILABLE:
                # 使用psutil终止
                p = psutil.Process(pid)
                if force:
                    p.kill()
                else:
                    p.terminate()
                
                p.wait(timeout=timeout)
            else:
                # 无 psutil 时使用 os.kill
                try:
                    sig = signal.SIGKILL if force else signal.SIGTERM
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="stop",
                message=f"进程已停止，PID: {pid}"
            )
            
            logger.info(f"停止进程: PID {pid}")
            return result
            
        except psutil.NoSuchProcess if PSUTIL_AVAILABLE else Exception as e:
            if PSUTIL_AVAILABLE and isinstance(e, psutil.NoSuchProcess):
                result = ProcessResult(
                    success=False,
                    pid=pid,
                    operation="stop",
                    message="进程不存在",
                    error="No such process"
                )
                return result
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="stop",
                message=f"停止进程失败: {e}",
                error=str(e)
            )
            logger.error(f"停止进程失败: PID {pid}, 错误: {e}")
            return result
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="stop",
                message=f"停止进程失败: {e}",
                error=str(e)
            )
            logger.error(f"停止进程失败: PID {pid}, 错误: {e}")
            return result
    
    def pause_process(self, pid: int) -> ProcessResult:
        """
        暂停进程（发送SIGSTOP信号）
        
        Args:
            pid: 进程ID
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="pause",
                message="psutil 不可用，无法暂停进程",
                error="psutil not available"
            )
            return result
        
        try:
            if sys.platform == 'win32':
                # Windows不支持SIGSTOP
                result = ProcessResult(
                    success=False,
                    pid=pid,
                    operation="pause",
                    message="Windows不支持暂停进程",
                    error="Not supported on Windows"
                )
                return result
            
            p = psutil.Process(pid)
            p.suspend()
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="pause",
                message=f"进程已暂停，PID: {pid}"
            )
            
            logger.info(f"暂停进程: PID {pid}")
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="pause",
                message=f"暂停进程失败: {e}",
                error=str(e)
            )
            return result
    
    def resume_process(self, pid: int) -> ProcessResult:
        """
        恢复进程（发送SIGCONT信号）
        
        Args:
            pid: 进程ID
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="resume",
                message="psutil 不可用，无法恢复进程",
                error="psutil not available"
            )
            return result
        
        try:
            if sys.platform == 'win32':
                result = ProcessResult(
                    success=False,
                    pid=pid,
                    operation="resume",
                    message="Windows不支持恢复进程",
                    error="Not supported on Windows"
                )
                return result
            
            p = psutil.Process(pid)
            p.resume()
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="resume",
                message=f"进程已恢复，PID: {pid}"
            )
            
            logger.info(f"恢复进程: PID {pid}")
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="resume",
                message=f"恢复进程失败: {e}",
                error=str(e)
            )
            return result
    
    def send_signal(self, pid: int, sig: Union[int, signal.Signals]) -> ProcessResult:
        """
        发送信号给进程
        
        Args:
            pid: 进程ID
            sig: 信号
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            # 使用 os.kill 作为 fallback
            try:
                os.kill(pid, sig)
                result = ProcessResult(
                    success=True,
                    pid=pid,
                    operation="signal",
                    message=f"信号已发送，PID: {pid}, 信号: {sig}"
                )
                logger.info(f"发送信号: PID {pid}, 信号 {sig}")
                return result
            except Exception as e:
                result = ProcessResult(
                    success=False,
                    pid=pid,
                    operation="signal",
                    message=f"发送信号失败: {e}",
                    error=str(e)
                )
                return result
        
        try:
            p = psutil.Process(pid)
            p.send_signal(sig)
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="signal",
                message=f"信号已发送，PID: {pid}, 信号: {sig}"
            )
            
            logger.info(f"发送信号: PID {pid}, 信号 {sig}")
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="signal",
                message=f"发送信号失败: {e}",
                error=str(e)
            )
            return result
    
    def get_process_info(self, pid: int) -> ProcessResult:
        """
        获取进程详细信息
        
        Args:
            pid: 进程ID
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="info",
                message="psutil 不可用，无法获取进程详细信息",
                error="psutil not available"
            )
            return result
        
        try:
            p = psutil.Process(pid)
            
            # 获取进程状态
            status_map = {
                psutil.STATUS_RUNNING: ProcessState.RUNNING,
                psutil.STATUS_STOPPED: ProcessState.STOPPED,
                psutil.STATUS_ZOMBIE: ProcessState.ZOMBIE,
            }
            state = status_map.get(p.status(), ProcessState.UNKNOWN)
            
            # 获取内存信息
            mem_info = p.memory_info()
            memory_dict = {
                'rss': mem_info.rss,
                'vms': mem_info.vms,
            }
            if hasattr(mem_info, 'shared'):
                memory_dict['shared'] = mem_info.shared
            
            # 获取环境变量
            try:
                environ = p.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                environ = {}
            
            # 获取网络连接
            try:
                connections = [
                    {
                        'fd': c.fd,
                        'family': str(c.family),
                        'type': str(c.type),
                        'local_addr': f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else None,
                        'remote_addr': f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else None,
                        'status': c.status,
                    }
                    for c in p.connections()
                ]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = []
            
            # 获取打开的文件
            try:
                open_files = [f.path for f in p.open_files()]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = []
            
            # 获取父进程和子进程
            try:
                parent_pid = p.ppid()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                parent_pid = None
            
            try:
                children_pids = [c.pid for c in p.children()]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                children_pids = []
            
            # 计算运行时间
            create_time = datetime.fromtimestamp(p.create_time())
            running_time = time.time() - p.create_time()
            
            info = ProcessInfo(
                pid=p.pid,
                name=p.name(),
                command=' '.join(p.cmdline()),
                state=state,
                cpu_percent=p.cpu_percent(interval=0.1),
                memory_percent=p.memory_percent(),
                memory_info=memory_dict,
                create_time=create_time,
                running_time=running_time,
                num_threads=p.num_threads(),
                num_handles=p.num_handles() if sys.platform == 'win32' else 0,
                username=p.username(),
                exe=p.exe(),
                cwd=p.cwd(),
                environ=environ,
                connections=connections,
                open_files=open_files,
                parent_pid=parent_pid,
                children_pids=children_pids
            )
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="info",
                message="获取进程信息成功",
                data=info
            )
            return result
            
        except psutil.NoSuchProcess:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="info",
                message="进程不存在",
                error="No such process"
            )
            return result
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="info",
                message=f"获取进程信息失败: {e}",
                error=str(e)
            )
            return result
    
    def list_processes(self, name_filter: Optional[str] = None,
                       user_filter: Optional[str] = None,
                       state_filter: Optional[ProcessState] = None) -> ProcessResult:
        """
        列出所有进程
        
        Args:
            name_filter: 名称过滤器
            user_filter: 用户过滤器
            state_filter: 状态过滤器
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            result = ProcessResult(
                success=False,
                pid=None,
                operation="list",
                message="psutil 不可用，无法列出进程",
                error="psutil not available"
            )
            return result
        
        try:
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'username', 'status', 'cmdline']):
                try:
                    pinfo = proc.info
                    
                    # 应用过滤器
                    if name_filter and name_filter.lower() not in pinfo['name'].lower():
                        continue
                    
                    if user_filter and user_filter != pinfo['username']:
                        continue
                    
                    if state_filter:
                        status_map = {
                            psutil.STATUS_RUNNING: ProcessState.RUNNING,
                            psutil.STATUS_STOPPED: ProcessState.STOPPED,
                            psutil.STATUS_ZOMBIE: ProcessState.ZOMBIE,
                        }
                        if status_map.get(pinfo['status']) != state_filter:
                            continue
                    
                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'username': pinfo['username'],
                        'status': pinfo['status'],
                        'command': ' '.join(pinfo['cmdline']) if pinfo['cmdline'] else ''
                    })
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            result = ProcessResult(
                success=True,
                pid=None,
                operation="list",
                message=f"找到 {len(processes)} 个进程",
                data=processes
            )
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=None,
                operation="list",
                message=f"列出进程失败: {e}",
                error=str(e)
            )
            return result
    
    def find_process(self, name: str) -> ProcessResult:
        """
        按名称查找进程
        
        Args:
            name: 进程名称
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            result = ProcessResult(
                success=False,
                pid=None,
                operation="find",
                message="psutil 不可用，无法查找进程",
                error="psutil not available"
            )
            return result
        
        try:
            found = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if name.lower() in proc.info['name'].lower():
                        found.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'command': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            result = ProcessResult(
                success=True,
                pid=None,
                operation="find",
                message=f"找到 {len(found)} 个匹配进程",
                data=found
            )
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=None,
                operation="find",
                message=f"查找进程失败: {e}",
                error=str(e)
            )
            return result
    
    def wait_for_process(self, pid: int, timeout: Optional[int] = None) -> ProcessResult:
        """
        等待进程结束
        
        Args:
            pid: 进程ID
            timeout: 超时时间
            
        Returns:
            进程操作结果
        """
        try:
            proc = None
            with self._lock:
                proc = self._managed_processes.get(pid)
            
            if proc:
                exit_code = proc.wait(timeout=timeout)
                
                with self._lock:
                    self._managed_processes.pop(pid, None)
                
                result = ProcessResult(
                    success=True,
                    pid=pid,
                    operation="wait",
                    message=f"进程已结束，退出码: {exit_code}",
                    data=exit_code
                )
                return result
            elif PSUTIL_AVAILABLE:
                # 使用psutil等待
                p = psutil.Process(pid)
                p.wait(timeout=timeout)
                
                result = ProcessResult(
                    success=True,
                    pid=pid,
                    operation="wait",
                    message="进程已结束"
                )
                return result
            else:
                result = ProcessResult(
                    success=False,
                    pid=pid,
                    operation="wait",
                    message="psutil 不可用，无法等待非托管进程",
                    error="psutil not available"
                )
                return result
            
        except subprocess.TimeoutExpired:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="wait",
                message="等待超时",
                error="Timeout"
            )
            return result
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="wait",
                message=f"等待进程失败: {e}",
                error=str(e)
            )
            return result
    
    def get_process_output(self, pid: int, timeout: int = 5) -> ProcessResult:
        """
        获取进程输出（仅对管理的进程有效）
        
        Args:
            pid: 进程ID
            timeout: 超时时间
            
        Returns:
            进程操作结果
        """
        try:
            proc = None
            with self._lock:
                proc = self._managed_processes.get(pid)
            
            if not proc:
                result = ProcessResult(
                    success=False,
                    pid=pid,
                    operation="output",
                    message="进程不在管理列表中",
                    error="Process not managed"
                )
                return result
            
            stdout_data = None
            stderr_data = None
            
            if proc.stdout:
                try:
                    stdout_data = proc.stdout.read1(4096).decode('utf-8', errors='replace')
                except Exception:
                    pass
            
            if proc.stderr:
                try:
                    stderr_data = proc.stderr.read1(4096).decode('utf-8', errors='replace')
                except Exception:
                    pass
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="output",
                message="获取输出成功",
                data={
                    'stdout': stdout_data,
                    'stderr': stderr_data
                }
            )
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="output",
                message=f"获取输出失败: {e}",
                error=str(e)
            )
            return result
    
    def register_callback(self, pid: int, callback: Callable[[int, str], None]) -> None:
        """
        注册进程状态变化回调
        
        Args:
            pid: 进程ID
            callback: 回调函数
        """
        with self._lock:
            if pid not in self._process_callbacks:
                self._process_callbacks[pid] = []
            self._process_callbacks[pid].append(callback)
    
    def start_monitor(self, interval: float = 1.0) -> None:
        """
        启动进程监控线程
        
        Args:
            interval: 检查间隔
        """
        if self._monitor_running:
            return
        
        self._monitor_running = True
        
        def monitor_loop():
            while self._monitor_running:
                self._check_processes()
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitor(self) -> None:
        """停止进程监控线程"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
    
    def _check_processes(self) -> None:
        """检查管理的进程状态"""
        with self._lock:
            for pid, proc in list(self._managed_processes.items()):
                ret = proc.poll()
                if ret is not None:
                    # 进程已结束
                    self._managed_processes.pop(pid, None)
                    
                    # 触发回调
                    callbacks = self._process_callbacks.pop(pid, [])
                    for callback in callbacks:
                        try:
                            callback(pid, f"exited with code {ret}")
                        except Exception as e:
                            logger.error(f"回调执行失败: {e}")
    
    def _cleanup_finished(self) -> None:
        """清理已结束的进程"""
        with self._lock:
            for pid, proc in list(self._managed_processes.items()):
                if proc.poll() is not None:
                    self._managed_processes.pop(pid, None)
    
    def kill_tree(self, pid: int, include_parent: bool = True) -> ProcessResult:
        """
        终止进程树（包括所有子进程）
        
        Args:
            pid: 进程ID
            include_parent: 是否包含父进程
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="kill_tree",
                message="psutil 不可用，无法终止进程树",
                error="psutil not available"
            )
            return result
        
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            killed = []
            
            # 先终止子进程
            for child in children:
                try:
                    child.terminate()
                    killed.append(child.pid)
                except psutil.NoSuchProcess:
                    pass
            
            # 等待子进程结束
            gone, alive = psutil.wait_procs(children, timeout=5)
            
            # 强制终止仍在运行的子进程
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # 终止父进程
            if include_parent:
                try:
                    parent.terminate()
                    parent.wait(timeout=5)
                    killed.append(pid)
                except psutil.NoSuchProcess:
                    pass
                except psutil.TimeoutExpired:
                    parent.kill()
                    killed.append(pid)
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="kill_tree",
                message=f"已终止 {len(killed)} 个进程",
                data=killed
            )
            return result
            
        except psutil.NoSuchProcess:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="kill_tree",
                message="进程不存在",
                error="No such process"
            )
            return result
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="kill_tree",
                message=f"终止进程树失败: {e}",
                error=str(e)
            )
            return result
    
    def get_resource_usage(self, pid: int) -> ProcessResult:
        """
        获取进程资源使用情况
        
        Args:
            pid: 进程ID
            
        Returns:
            进程操作结果
        """
        if not PSUTIL_AVAILABLE:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="resource_usage",
                message="psutil 不可用，无法获取资源使用情况",
                error="psutil not available"
            )
            return result
        
        try:
            p = psutil.Process(pid)
            
            cpu_times = p.cpu_times()
            io_counters = p.io_counters() if hasattr(p, 'io_counters') else None
            
            usage = {
                'cpu': {
                    'percent': p.cpu_percent(interval=0.1),
                    'times': {
                        'user': cpu_times.user,
                        'system': cpu_times.system,
                    }
                },
                'memory': {
                    'percent': p.memory_percent(),
                    'rss': p.memory_info().rss,
                    'vms': p.memory_info().vms,
                },
                'threads': p.num_threads(),
            }
            
            if io_counters:
                usage['io'] = {
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                }
            
            result = ProcessResult(
                success=True,
                pid=pid,
                operation="resource_usage",
                message="获取资源使用成功",
                data=usage
            )
            return result
            
        except Exception as e:
            result = ProcessResult(
                success=False,
                pid=pid,
                operation="resource_usage",
                message=f"获取资源使用失败: {e}",
                error=str(e)
            )
            return result
