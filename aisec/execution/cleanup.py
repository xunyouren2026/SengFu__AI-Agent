"""
执行清理 - 执行环境清理与资源回收
"""
import os
import shutil
import time
import threading
import subprocess
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CleanupType(Enum):
    """清理类型"""
    TEMP_FILES = "temp_files"
    CONTAINERS = "containers"
    PROCESSES = "processes"
    NETWORK = "network"
    ALL = "all"


@dataclass
class CleanupResult:
    """清理结果"""
    cleanup_type: CleanupType
    success: bool
    items_cleaned: int
    items_failed: int
    details: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ExecutionCleaner:
    """执行清理器"""
    
    def __init__(self):
        self._temp_dirs: List[str] = []
        self._containers: List[str] = []
        self._processes: Dict[int, float] = {}  # pid -> start_time
        self._cleanup_hooks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._auto_cleanup = True
        self._max_age_hours = 24
    
    def register_temp_dir(self, path: str) -> None:
        """注册临时目录"""
        with self._lock:
            if path not in self._temp_dirs:
                self._temp_dirs.append(path)
    
    def register_container(self, container_id: str) -> None:
        """注册容器"""
        with self._lock:
            if container_id not in self._containers:
                self._containers.append(container_id)
    
    def register_process(self, pid: int) -> None:
        """注册进程"""
        with self._lock:
            self._processes[pid] = time.time()
    
    def unregister_process(self, pid: int) -> None:
        """注销进程"""
        with self._lock:
            self._processes.pop(pid, None)
    
    def add_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """添加清理钩子"""
        self._cleanup_hooks.append(hook)
    
    def cleanup_temp_files(self) -> CleanupResult:
        """清理临时文件"""
        cleaned = 0
        failed = 0
        details = []
        errors = []
        
        with self._lock:
            dirs_to_clean = list(self._temp_dirs)
            self._temp_dirs.clear()
        
        for dir_path in dirs_to_clean:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    cleaned += 1
                    details.append(f"已删除: {dir_path}")
            except Exception as e:
                failed += 1
                errors.append(f"删除失败 {dir_path}: {e}")
        
        return CleanupResult(
            cleanup_type=CleanupType.TEMP_FILES,
            success=failed == 0,
            items_cleaned=cleaned,
            items_failed=failed,
            details=details,
            errors=errors
        )
    
    def cleanup_containers(self) -> CleanupResult:
        """清理容器"""
        cleaned = 0
        failed = 0
        details = []
        errors = []
        
        with self._lock:
            containers_to_clean = list(self._containers)
            self._containers.clear()
        
        for container_id in containers_to_clean:
            try:
                result = subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode == 0:
                    cleaned += 1
                    details.append(f"已删除容器: {container_id}")
                else:
                    failed += 1
                    errors.append(f"删除容器失败 {container_id}: {result.stderr.decode()}")
            except Exception as e:
                failed += 1
                errors.append(f"删除容器异常 {container_id}: {e}")
        
        # 清理孤立容器
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "label=sandbox=true", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                orphan_containers = result.stdout.strip().split('\n')
                for container_id in orphan_containers:
                    if container_id:
                        subprocess.run(
                            ["docker", "rm", "-f", container_id],
                            capture_output=True,
                            timeout=10
                        )
                        cleaned += 1
                        details.append(f"已删除孤立容器: {container_id}")
        except Exception:
            pass
        
        return CleanupResult(
            cleanup_type=CleanupType.CONTAINERS,
            success=failed == 0,
            items_cleaned=cleaned,
            items_failed=failed,
            details=details,
            errors=errors
        )
    
    def cleanup_processes(self, max_age_seconds: Optional[int] = None) -> CleanupResult:
        """清理进程"""
        cleaned = 0
        failed = 0
        details = []
        errors = []
        
        max_age = max_age_seconds or (self._max_age_hours * 3600)
        current_time = time.time()
        
        with self._lock:
            processes_to_check = dict(self._processes)
        
        for pid, start_time in processes_to_check.items():
            try:
                # 检查进程是否还存在
                if not os.path.exists(f"/proc/{pid}"):
                    self.unregister_process(pid)
                    continue
                
                # 检查是否超时
                if current_time - start_time > max_age:
                    os.kill(pid, 9)  # SIGKILL
                    self.unregister_process(pid)
                    cleaned += 1
                    details.append(f"已终止进程: {pid}")
            except ProcessLookupError:
                self.unregister_process(pid)
            except PermissionError:
                failed += 1
                errors.append(f"无权限终止进程: {pid}")
            except Exception as e:
                failed += 1
                errors.append(f"终止进程异常 {pid}: {e}")
        
        return CleanupResult(
            cleanup_type=CleanupType.PROCESSES,
            success=failed == 0,
            items_cleaned=cleaned,
            items_failed=failed,
            details=details,
            errors=errors
        )
    
    def cleanup_all(self) -> Dict[CleanupType, CleanupResult]:
        """清理所有"""
        results = {}
        
        # 执行清理钩子
        for hook in self._cleanup_hooks:
            try:
                hook()
            except Exception:
                pass
        
        # 清理各类资源
        results[CleanupType.TEMP_FILES] = self.cleanup_temp_files()
        results[CleanupType.CONTAINERS] = self.cleanup_containers()
        results[CleanupType.PROCESSES] = self.cleanup_processes()
        
        return results
    
    def cleanup_old_temp_files(self, base_dir: str = "/tmp", pattern: str = "sandbox_*") -> CleanupResult:
        """清理旧的临时文件"""
        import glob
        
        cleaned = 0
        failed = 0
        details = []
        errors = []
        
        search_pattern = os.path.join(base_dir, pattern)
        max_age = self._max_age_hours * 3600
        current_time = time.time()
        
        for path in glob.glob(search_pattern):
            try:
                # 检查修改时间
                mtime = os.path.getmtime(path)
                if current_time - mtime > max_age:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    cleaned += 1
                    details.append(f"已删除旧文件: {path}")
            except Exception as e:
                failed += 1
                errors.append(f"删除失败 {path}: {e}")
        
        return CleanupResult(
            cleanup_type=CleanupType.TEMP_FILES,
            success=failed == 0,
            items_cleaned=cleaned,
            items_failed=failed,
            details=details,
            errors=errors
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "temp_dirs": len(self._temp_dirs),
                "containers": len(self._containers),
                "processes": len(self._processes),
                "cleanup_hooks": len(self._cleanup_hooks)
            }


class CleanupScheduler:
    """清理调度器"""
    
    def __init__(self, cleaner: ExecutionCleaner):
        self._cleaner = cleaner
        self._running = False
        self._interval = 3600  # 默认每小时清理一次
        self._thread: Optional[threading.Thread] = None
    
    def start(self, interval: int = 3600) -> None:
        """启动定时清理"""
        self._interval = interval
        self._running = True
        
        def cleanup_loop():
            while self._running:
                time.sleep(self._interval)
                if self._running:
                    self._cleaner.cleanup_all()
        
        self._thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止定时清理"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def trigger_cleanup(self) -> Dict[CleanupType, CleanupResult]:
        """触发立即清理"""
        return self._cleaner.cleanup_all()


class ResourceTracker:
    """资源追踪器"""
    
    def __init__(self):
        self._resources: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def track(self, resource_id: str, resource_type: str, metadata: Dict[str, Any] = None) -> None:
        """追踪资源"""
        with self._lock:
            self._resources[resource_id] = {
                "type": resource_type,
                "created_at": time.time(),
                "metadata": metadata or {}
            }
    
    def untrack(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """取消追踪"""
        with self._lock:
            return self._resources.pop(resource_id, None)
    
    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """获取资源信息"""
        with self._lock:
            return self._resources.get(resource_id)
    
    def get_by_type(self, resource_type: str) -> List[str]:
        """按类型获取资源"""
        with self._lock:
            return [
                rid for rid, info in self._resources.items()
                if info["type"] == resource_type
            ]
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """获取所有资源"""
        with self._lock:
            return dict(self._resources)
    
    def cleanup_orphaned(self, cleanup_func: Callable[[str], bool]) -> int:
        """清理孤立资源"""
        cleaned = 0
        with self._lock:
            for resource_id, info in list(self._resources.items()):
                try:
                    if cleanup_func(resource_id):
                        del self._resources[resource_id]
                        cleaned += 1
                except Exception:
                    pass
        return cleaned
