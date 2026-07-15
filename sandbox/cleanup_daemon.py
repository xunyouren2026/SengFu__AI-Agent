"""
清理守护进程
定期清理过期容器、临时文件、僵尸进程等
"""

import os
import time
import threading
import subprocess
import signal
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json


class CleanupTarget(Enum):
    """清理目标类型"""
    CONTAINER = "container"
    TEMP_FILE = "temp_file"
    TEMP_DIR = "temp_dir"
    ZOMBIE_PROCESS = "zombie_process"
    OLD_LOG = "old_log"
    CGROUP = "cgroup"
    NAMESPACE = "namespace"


class DaemonState(Enum):
    """守护进程状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class CleanupConfig:
    """清理配置"""
    interval: int = 60                      # 清理间隔（秒）
    container_max_age: int = 3600           # 容器最大存活时间（秒）
    temp_file_max_age: int = 1800           # 临时文件最大存活时间（秒）
    log_max_age: int = 86400                # 日志最大存活时间（秒）
    zombie_process_check: bool = True       # 是否检查僵尸进程
    dry_run: bool = False                   # 试运行模式
    targets: Set[CleanupTarget] = field(default_factory=lambda: {
        CleanupTarget.CONTAINER,
        CleanupTarget.TEMP_FILE,
        CleanupTarget.TEMP_DIR,
        CleanupTarget.ZOMBIE_PROCESS
    })


@dataclass
class CleanupResult:
    """清理结果"""
    target: CleanupTarget
    cleaned_count: int = 0
    failed_count: int = 0
    freed_bytes: int = 0
    items: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target.value,
            'cleaned_count': self.cleaned_count,
            'failed_count': self.failed_count,
            'freed_bytes': self.freed_bytes,
            'items': self.items,
            'errors': self.errors,
            'duration': self.duration
        }


class ContainerCleaner:
    """容器清理器"""
    
    def __init__(self, docker_path: str = "docker"):
        self.docker_path = docker_path
    
    def list_containers(
        self,
        label_filter: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出容器"""
        cmd = [self.docker_path, "ps", "-a", "--format", "{{json .}}"]
        
        if label_filter:
            cmd.extend(["--filter", f"label={label_filter}"])
        if status_filter:
            cmd.extend(["--filter", f"status={status_filter}"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        containers.append(json.loads(line))
                return containers
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return []
    
    def get_container_age(self, container_id: str) -> Optional[float]:
        """获取容器存活时间"""
        try:
            result = subprocess.run(
                [self.docker_path, "inspect", "--format", "{{.State.StartedAt}}", container_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # 解析时间戳
                started_at = result.stdout.strip()
                # Docker时间格式: 2024-01-01T00:00:00.000000000Z
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    return time.time() - dt.timestamp()
                except ValueError:
                    pass
        except subprocess.TimeoutExpired:
            pass
        return None
    
    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """删除容器"""
        try:
            cmd = [self.docker_path, "rm"]
            if force:
                cmd.append("-f")
            cmd.append(container_id)
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def cleanup_old_containers(
        self,
        max_age: int,
        prefix: Optional[str] = None,
        dry_run: bool = False
    ) -> CleanupResult:
        """
        清理过期容器
        
        Args:
            max_age: 最大存活时间
            prefix: 容器名称前缀
            dry_run: 试运行
            
        Returns:
            清理结果
        """
        result = CleanupResult(target=CleanupTarget.CONTAINER)
        start_time = time.time()
        
        containers = self.list_containers()
        
        for container in containers:
            container_id = container.get('ID', '')
            name = container.get('Names', '')
            
            # 检查前缀
            if prefix and not name.startswith(prefix):
                continue
            
            # 检查存活时间
            age = self.get_container_age(container_id)
            if age is None or age < max_age:
                continue
            
            result.items.append(container_id)
            
            if not dry_run:
                if self.remove_container(container_id, force=True):
                    result.cleaned_count += 1
                else:
                    result.failed_count += 1
                    result.errors.append(f"Failed to remove {container_id}")
            else:
                result.cleaned_count += 1
        
        result.duration = time.time() - start_time
        return result


class TempFileCleaner:
    """临时文件清理器"""
    
    def __init__(self, temp_dirs: Optional[List[str]] = None):
        self.temp_dirs = temp_dirs or [
            tempfile.gettempdir(),
            '/tmp',
            '/var/tmp'
        ]
    
    def cleanup_old_files(
        self,
        max_age: int,
        prefix: Optional[str] = None,
        dry_run: bool = False
    ) -> CleanupResult:
        """
        清理过期临时文件
        
        Args:
            max_age: 最大存活时间
            prefix: 文件名前缀
            dry_run: 试运行
            
        Returns:
            清理结果
        """
        result = CleanupResult(target=CleanupTarget.TEMP_FILE)
        start_time = time.time()
        current_time = time.time()
        
        for temp_dir in self.temp_dirs:
            if not os.path.exists(temp_dir):
                continue
            
            for entry in Path(temp_dir).iterdir():
                try:
                    # 检查前缀
                    if prefix and not entry.name.startswith(prefix):
                        continue
                    
                    # 检查存活时间
                    mtime = entry.stat().st_mtime
                    age = current_time - mtime
                    
                    if age < max_age:
                        continue
                    
                    # 计算大小
                    if entry.is_file():
                        size = entry.stat().st_size
                    elif entry.is_dir():
                        size = sum(f.stat().st_size for f in entry.rglob('*') if f.is_file())
                    else:
                        size = 0
                    
                    result.items.append(str(entry))
                    result.freed_bytes += size
                    
                    if not dry_run:
                        if entry.is_file():
                            entry.unlink()
                        elif entry.is_dir():
                            shutil.rmtree(entry)
                        result.cleaned_count += 1
                    else:
                        result.cleaned_count += 1
                        
                except (OSError, PermissionError) as e:
                    result.failed_count += 1
                    result.errors.append(f"Failed to clean {entry}: {e}")
        
        result.duration = time.time() - start_time
        return result
    
    def cleanup_temp_dirs(
        self,
        max_age: int,
        prefix: Optional[str] = None,
        dry_run: bool = False
    ) -> CleanupResult:
        """清理临时目录"""
        result = CleanupResult(target=CleanupTarget.TEMP_DIR)
        start_time = time.time()
        current_time = time.time()
        
        for temp_dir in self.temp_dirs:
            if not os.path.exists(temp_dir):
                continue
            
            for entry in Path(temp_dir).iterdir():
                if not entry.is_dir():
                    continue
                
                try:
                    if prefix and not entry.name.startswith(prefix):
                        continue
                    
                    mtime = entry.stat().st_mtime
                    age = current_time - mtime
                    
                    if age < max_age:
                        continue
                    
                    size = sum(f.stat().st_size for f in entry.rglob('*') if f.is_file())
                    
                    result.items.append(str(entry))
                    result.freed_bytes += size
                    
                    if not dry_run:
                        shutil.rmtree(entry)
                        result.cleaned_count += 1
                    else:
                        result.cleaned_count += 1
                        
                except (OSError, PermissionError) as e:
                    result.failed_count += 1
                    result.errors.append(f"Failed to clean {entry}: {e}")
        
        result.duration = time.time() - start_time
        return result


class ProcessCleaner:
    """进程清理器"""
    
    def find_zombie_processes(self) -> List[Dict[str, Any]]:
        """查找僵尸进程"""
        zombies = []
        
        try:
            # 使用ps命令查找僵尸进程
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[1:]:
                    parts = line.split()
                    if len(parts) >= 8:
                        stat = parts[7]
                        # 'Z' 表示僵尸进程
                        if 'Z' in stat:
                            zombies.append({
                                'pid': int(parts[1]),
                                'user': parts[0],
                                'command': ' '.join(parts[10:]) if len(parts) > 10 else ''
                            })
        except subprocess.TimeoutExpired:
            pass
        
        return zombies
    
    def reap_zombie(self, pid: int) -> bool:
        """回收僵尸进程"""
        try:
            # 发送SIGCHLD给父进程
            os.kill(pid, signal.SIGCHLD)
            return True
        except (OSError, ProcessLookupError):
            return False
    
    def cleanup_zombies(self, dry_run: bool = False) -> CleanupResult:
        """清理僵尸进程"""
        result = CleanupResult(target=CleanupTarget.ZOMBIE_PROCESS)
        start_time = time.time()
        
        zombies = self.find_zombie_processes()
        
        for zombie in zombies:
            pid = zombie['pid']
            result.items.append(f"PID {pid}: {zombie['command']}")
            
            if not dry_run:
                if self.reap_zombie(pid):
                    result.cleaned_count += 1
                else:
                    result.failed_count += 1
                    result.errors.append(f"Failed to reap PID {pid}")
            else:
                result.cleaned_count += 1
        
        result.duration = time.time() - start_time
        return result


class CleanupDaemon:
    """
    清理守护进程
    定期执行清理任务
    """
    
    def __init__(self, config: Optional[CleanupConfig] = None):
        self.config = config or CleanupConfig()
        self._state = DaemonState.IDLE
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # 清理器
        self._container_cleaner = ContainerCleaner()
        self._file_cleaner = TempFileCleaner()
        self._process_cleaner = ProcessCleaner()
        
        # 回调
        self._cleanup_callbacks: List[Callable[[CleanupResult], None]] = []
        
        # 统计
        self._cleanup_count = 0
        self._last_cleanup_time: Optional[float] = None
        self._cleanup_history: List[Dict[str, Any]] = []
    
    @property
    def state(self) -> DaemonState:
        return self._state
    
    def add_callback(self, callback: Callable[[CleanupResult], None]) -> None:
        """添加清理回调"""
        self._cleanup_callbacks.append(callback)
    
    def start(self) -> bool:
        """启动守护进程"""
        if self._state == DaemonState.RUNNING:
            return True
        
        self._stop_event.clear()
        self._pause_event.clear()
        
        self._daemon_thread = threading.Thread(
            target=self._daemon_loop,
            daemon=True
        )
        self._daemon_thread.start()
        
        self._state = DaemonState.RUNNING
        return True
    
    def stop(self) -> bool:
        """停止守护进程"""
        if self._state != DaemonState.RUNNING:
            return True
        
        self._stop_event.set()
        
        if self._daemon_thread:
            self._daemon_thread.join(timeout=10)
        
        self._state = DaemonState.STOPPED
        return True
    
    def pause(self) -> bool:
        """暂停清理"""
        if self._state == DaemonState.RUNNING:
            self._pause_event.set()
            self._state = DaemonState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """恢复清理"""
        if self._state == DaemonState.PAUSED:
            self._pause_event.clear()
            self._state = DaemonState.RUNNING
            return True
        return False
    
    def run_once(self) -> Dict[str, CleanupResult]:
        """执行一次清理"""
        return self._perform_cleanup()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'state': self._state.value,
            'cleanup_count': self._cleanup_count,
            'last_cleanup_time': self._last_cleanup_time,
            'config': {
                'interval': self.config.interval,
                'container_max_age': self.config.container_max_age,
                'temp_file_max_age': self.config.temp_file_max_age,
                'dry_run': self.config.dry_run
            }
        }
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取清理历史"""
        return self._cleanup_history[-limit:]
    
    def _daemon_loop(self) -> None:
        """守护进程主循环"""
        while not self._stop_event.is_set():
            # 检查暂停
            while self._pause_event.is_set() and not self._stop_event.is_set():
                time.sleep(1)
            
            if self._stop_event.is_set():
                break
            
            # 执行清理
            try:
                results = self._perform_cleanup()
                
                # 调用回调
                for result in results.values():
                    for callback in self._cleanup_callbacks:
                        try:
                            callback(result)
                        except Exception:
                            pass
                
            except Exception as e:
                pass
            
            # 等待下次清理
            self._stop_event.wait(self.config.interval)
    
    def _perform_cleanup(self) -> Dict[str, CleanupResult]:
        """执行清理"""
        results = {}
        self._cleanup_count += 1
        self._last_cleanup_time = time.time()
        
        # 清理容器
        if CleanupTarget.CONTAINER in self.config.targets:
            results['container'] = self._container_cleaner.cleanup_old_containers(
                max_age=self.config.container_max_age,
                prefix="sandbox_",
                dry_run=self.config.dry_run
            )
        
        # 清理临时文件
        if CleanupTarget.TEMP_FILE in self.config.targets:
            results['temp_file'] = self._file_cleaner.cleanup_old_files(
                max_age=self.config.temp_file_max_age,
                prefix="sandbox_",
                dry_run=self.config.dry_run
            )
        
        # 清理临时目录
        if CleanupTarget.TEMP_DIR in self.config.targets:
            results['temp_dir'] = self._file_cleaner.cleanup_temp_dirs(
                max_age=self.config.temp_file_max_age,
                prefix="sandbox_",
                dry_run=self.config.dry_run
            )
        
        # 清理僵尸进程
        if CleanupTarget.ZOMBIE_PROCESS in self.config.targets and self.config.zombie_process_check:
            results['zombie'] = self._process_cleaner.cleanup_zombies(
                dry_run=self.config.dry_run
            )
        
        # 记录历史
        history_entry = {
            'timestamp': self._last_cleanup_time,
            'results': {k: v.to_dict() for k, v in results.items()}
        }
        self._cleanup_history.append(history_entry)
        
        # 限制历史记录大小
        if len(self._cleanup_history) > 100:
            self._cleanup_history = self._cleanup_history[-100:]
        
        return results
    
    def __enter__(self) -> 'CleanupDaemon':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class ResourceTracker:
    """
    资源追踪器
    追踪沙箱创建的资源，便于清理
    """
    
    def __init__(self, tracker_file: Optional[str] = None):
        self.tracker_file = tracker_file or os.path.join(
            tempfile.gettempdir(),
            "sandbox_resource_tracker.json"
        )
        self._resources: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self) -> None:
        """加载追踪数据"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    self._resources = json.load(f)
            except (IOError, json.JSONDecodeError):
                self._resources = {}
    
    def _save(self) -> None:
        """保存追踪数据"""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self._resources, f, indent=2)
        except IOError:
            pass
    
    def track(
        self,
        resource_id: str,
        resource_type: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        追踪资源
        
        Args:
            resource_id: 资源ID
            resource_type: 资源类型
            path: 资源路径
            metadata: 元数据
        """
        self._resources[resource_id] = {
            'type': resource_type,
            'path': path,
            'created_at': time.time(),
            'metadata': metadata or {}
        }
        self._save()
    
    def untrack(self, resource_id: str) -> None:
        """取消追踪"""
        self._resources.pop(resource_id, None)
        self._save()
    
    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """获取资源"""
        return self._resources.get(resource_id)
    
    def get_by_type(self, resource_type: str) -> List[Dict[str, Any]]:
        """按类型获取资源"""
        return [
            {'id': rid, **data}
            for rid, data in self._resources.items()
            if data.get('type') == resource_type
        ]
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """获取所有资源"""
        return dict(self._resources)
    
    def cleanup_tracked(self, max_age: int) -> List[str]:
        """
        清理追踪的过期资源
        
        Args:
            max_age: 最大存活时间
            
        Returns:
            清理的资源ID列表
        """
        cleaned = []
        current_time = time.time()
        
        for resource_id, data in list(self._resources.items()):
            age = current_time - data.get('created_at', 0)
            if age > max_age:
                path = data.get('path')
                if path and os.path.exists(path):
                    try:
                        if os.path.isfile(path):
                            os.unlink(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                        cleaned.append(resource_id)
                    except OSError:
                        pass
        
        for rid in cleaned:
            self._resources.pop(rid, None)
        
        self._save()
        return cleaned
