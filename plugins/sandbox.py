"""
插件沙箱模块

提供插件隔离执行、资源限制和安全监控功能。
"""

import os
import resource
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """沙箱配置"""
    max_memory_mb: int = 256
    max_cpu_time_sec: int = 30
    max_wall_time_sec: int = 60
    max_file_size_mb: int = 10
    max_open_files: int = 64
    allowed_syscalls: List[str] = field(default_factory=lambda: [
        'read', 'write', 'open', 'close', 'exit', 'exit_group'
    ])
    blocked_modules: List[str] = field(default_factory=lambda: [
        'os', 'sys', 'subprocess', 'socket', 'ctypes'
    ])
    network_access: bool = False
    file_system_access: bool = True


@dataclass
class SandboxResult:
    """沙箱执行结果"""
    success: bool
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    error: str = ""
    killed: bool = False


class PluginSandbox:
    """插件沙箱
    
    提供隔离执行环境，限制资源使用。
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Args:
            config: 沙箱配置
        """
        self._config = config or SandboxConfig()
        self._lock = threading.RLock()
    
    def execute(self, func: Callable, *args, **kwargs) -> SandboxResult:
        """在沙箱中执行函数
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            执行结果
        """
        import multiprocessing
        
        start_time = time.time()
        
        # 使用进程实现隔离
        queue = multiprocessing.Queue()
        
        def wrapper():
            try:
                # 设置资源限制
                self._apply_resource_limits()
                
                # 执行函数
                result = func(*args, **kwargs)
                queue.put(('success', result, '', ''))
            except Exception as e:
                queue.put(('error', None, '', str(e)))
        
        process = multiprocessing.Process(target=wrapper)
        process.start()
        
        # 等待完成或超时
        process.join(timeout=self._config.max_wall_time_sec)
        
        execution_time = (time.time() - start_time) * 1000
        
        if process.is_alive():
            process.terminate()
            process.join()
            return SandboxResult(
                success=False,
                execution_time_ms=execution_time,
                error="Execution timed out",
                killed=True,
            )
        
        # 获取结果
        try:
            status, return_value, stdout, stderr = queue.get_nowait()
            
            return SandboxResult(
                success=status == 'success',
                return_value=return_value,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time,
            )
        except:
            return SandboxResult(
                success=False,
                execution_time_ms=execution_time,
                error="Failed to get result",
            )
    
    def _apply_resource_limits(self) -> None:
        """应用资源限制"""
        # 内存限制
        max_memory_bytes = self._config.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        
        # CPU时间限制
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (self._config.max_cpu_time_sec, self._config.max_cpu_time_sec)
        )
        
        # 文件大小限制
        max_file_size = self._config.max_file_size_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))
        
        # 打开文件数限制
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (self._config.max_open_files, self._config.max_open_files)
        )
    
    def create_restricted_globals(self) -> Dict[str, Any]:
        """创建受限的全局命名空间"""
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'max': max,
            'min': min,
            'print': print,
            'range': range,
            'round': round,
            'set': set,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }
        
        return {'__builtins__': safe_builtins}
    
    def is_safe_code(self, code: str) -> bool:
        """检查代码是否安全"""
        import ast
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        for node in ast.walk(tree):
            # 检查导入
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module] if node.module else []
                
                for module in modules:
                    base_module = module.split('.')[0]
                    if base_module in self._config.blocked_modules:
                        return False
            
            # 检查危险函数
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        return False
        
        return True
