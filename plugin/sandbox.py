#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plugin 沙箱模块

本模块提供插件沙箱功能，实现插件的隔离执行、资源限制和权限控制。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import os
import sys
import time
import signal
import threading
import resource
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Set
from contextlib import contextmanager
import logging
import tempfile
import json

# 配置日志
logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """沙箱错误基类"""
    pass


class ResourceLimitError(SandboxError):
    """资源限制错误"""
    pass


class PermissionDeniedError(SandboxError):
    """权限拒绝错误"""
    pass


class SandboxTimeoutError(SandboxError):
    """沙箱超时错误"""
    pass


class SandboxExitError(SandboxError):
    """沙箱异常退出错误"""
    pass


@dataclass
class ResourceLimits:
    """
    资源限制配置
    
    属性:
        max_cpu_time: 最大 CPU 时间（秒）
        max_memory: 最大内存（字节）
        max_processes: 最大进程数
        max_file_size: 最大文件大小（字节）
        max_open_files: 最大打开文件数
        max_network: 是否允许网络访问
        max_disk_space: 最大磁盘空间（字节）
    """
    max_cpu_time: Optional[int] = None  # 秒
    max_memory: Optional[int] = None    # 字节
    max_processes: Optional[int] = None
    max_file_size: Optional[int] = None  # 字节
    max_open_files: Optional[int] = None
    max_network: bool = False
    max_disk_space: Optional[int] = None  # 字节
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'max_cpu_time': self.max_cpu_time,
            'max_memory': self.max_memory,
            'max_processes': self.max_processes,
            'max_file_size': self.max_file_size,
            'max_open_files': self.max_open_files,
            'max_network': self.max_network,
            'max_disk_space': self.max_disk_space,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResourceLimits:
        """从字典创建"""
        return cls(**data)


@dataclass
class SandboxConfig:
    """
    沙箱配置
    
    属性:
        plugin_id: 插件 ID
        working_dir: 工作目录
        allowed_paths: 允许访问的路径列表
        blocked_paths: 禁止访问的路径列表
        allowed_syscalls: 允许的系统调用列表
        blocked_syscalls: 禁止的系统调用列表
        env_vars: 环境变量
        resource_limits: 资源限制
        timeout: 超时时间（秒）
    """
    plugin_id: str
    working_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp()))
    allowed_paths: List[Path] = field(default_factory=list)
    blocked_paths: List[Path] = field(default_factory=list)
    allowed_syscalls: List[str] = field(default_factory=list)
    blocked_syscalls: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    timeout: int = 60
    
    def __post_init__(self):
        """初始化后处理"""
        self.working_dir = Path(self.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认允许访问工作目录
        if self.working_dir not in self.allowed_paths:
            self.allowed_paths.append(self.working_dir)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'plugin_id': self.plugin_id,
            'working_dir': str(self.working_dir),
            'allowed_paths': [str(p) for p in self.allowed_paths],
            'blocked_paths': [str(p) for p in self.blocked_paths],
            'allowed_syscalls': self.allowed_syscalls,
            'blocked_syscalls': self.blocked_syscalls,
            'env_vars': self.env_vars,
            'resource_limits': self.resource_limits.to_dict(),
            'timeout': self.timeout,
        }


@dataclass
class SandboxResult:
    """
    沙箱执行结果
    
    属性:
        success: 是否成功
        return_code: 返回码
        stdout: 标准输出
        stderr: 标准错误
        execution_time: 执行时间（秒）
        memory_used: 内存使用（字节）
        error_message: 错误信息
    """
    success: bool
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    memory_used: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'return_code': self.return_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'execution_time': self.execution_time,
            'memory_used': self.memory_used,
            'error_message': self.error_message,
        }


class PathValidator:
    """路径验证器"""
    
    def __init__(self, allowed_paths: List[Path], blocked_paths: List[Path]):
        """
        初始化验证器
        
        参数:
            allowed_paths: 允许的路径列表
            blocked_paths: 禁止的路径列表
        """
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]
        self.blocked_paths = [Path(p).resolve() for p in blocked_paths]
    
    def is_allowed(self, path: Union[str, Path]) -> bool:
        """
        检查路径是否允许访问
        
        参数:
            path: 要检查的路径
            
        返回:
            是否允许
        """
        path = Path(path).resolve()
        
        # 检查是否在禁止列表中
        for blocked in self.blocked_paths:
            if path == blocked or blocked in path.parents:
                return False
        
        # 检查是否在允许列表中
        if not self.allowed_paths:
            return True
        
        for allowed in self.allowed_paths:
            if path == allowed or allowed in path.parents:
                return True
        
        return False
    
    def validate_read(self, path: Union[str, Path]) -> None:
        """
        验证读权限
        
        参数:
            path: 路径
            
        抛出:
            PermissionDeniedError: 无权限
        """
        if not self.is_allowed(path):
            raise PermissionDeniedError(f"无权限读取: {path}")
    
    def validate_write(self, path: Union[str, Path]) -> None:
        """
        验证写权限
        
        参数:
            path: 路径
            
        抛出:
            PermissionDeniedError: 无权限
        """
        if not self.is_allowed(path):
            raise PermissionDeniedError(f"无权限写入: {path}")


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, limits: ResourceLimits):
        """
        初始化监控器
        
        参数:
            limits: 资源限制
        """
        self.limits = limits
        self._peak_memory: int = 0
        self._start_time: Optional[float] = None
        self._monitoring: bool = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """开始监控"""
        self._start_time = time.time()
        self._monitoring = True
        
        if self.limits.max_memory:
            self._thread = threading.Thread(target=self._monitor_memory)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> None:
        """停止监控"""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor_memory(self) -> None:
        """监控内存使用"""
        import psutil
        process = psutil.Process()
        
        while self._monitoring:
            try:
                memory_info = process.memory_info()
                current_memory = memory_info.rss
                
                if current_memory > self._peak_memory:
                    self._peak_memory = current_memory
                
                if self.limits.max_memory and current_memory > self.limits.max_memory:
                    raise ResourceLimitError(
                        f"内存超限: {current_memory} > {self.limits.max_memory}"
                    )
                
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                break
    
    def check_cpu_time(self) -> None:
        """检查 CPU 时间"""
        if self.limits.max_cpu_time and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > self.limits.max_cpu_time:
                raise ResourceLimitError(
                    f"CPU 时间超限: {elapsed} > {self.limits.max_cpu_time}"
                )
    
    @property
    def peak_memory(self) -> int:
        """获取峰值内存使用"""
        return self._peak_memory
    
    @property
    def elapsed_time(self) -> float:
        """获取已用时间"""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0


class PluginSandbox:
    """
    插件沙箱
    
    提供插件的隔离执行环境。
    """
    
    def __init__(self, config: SandboxConfig):
        """
        初始化沙箱
        
        参数:
            config: 沙箱配置
        """
        self.config = config
        self.path_validator = PathValidator(
            config.allowed_paths,
            config.blocked_paths
        )
        self.resource_monitor = ResourceMonitor(config.resource_limits)
        self._process: Optional[subprocess.Popen] = None
    
    def execute(self, code: str, **kwargs) -> SandboxResult:
        """
        在沙箱中执行代码
        
        参数:
            code: 要执行的代码
            **kwargs: 额外参数
            
        返回:
            执行结果
        """
        start_time = time.time()
        
        try:
            # 创建临时脚本文件
            script_path = self.config.working_dir / "sandbox_script.py"
            script_path.write_text(code, encoding='utf-8')
            
            # 准备环境变量
            env = os.environ.copy()
            env.update(self.config.env_vars)
            env['SANDBOX_PLUGIN_ID'] = self.config.plugin_id
            
            # 设置资源限制
            def set_limits():
                if self.config.resource_limits.max_cpu_time:
                    resource.setrlimit(
                        resource.RLIMIT_CPU,
                        (self.config.resource_limits.max_cpu_time, 
                         self.config.resource_limits.max_cpu_time)
                    )
                if self.config.resource_limits.max_memory:
                    resource.setrlimit(
                        resource.RLIMIT_AS,
                        (self.config.resource_limits.max_memory,
                         self.config.resource_limits.max_memory)
                    )
                if self.config.resource_limits.max_file_size:
                    resource.setrlimit(
                        resource.RLIMIT_FSIZE,
                        (self.config.resource_limits.max_file_size,
                         self.config.resource_limits.max_file_size)
                    )
                if self.config.resource_limits.max_processes:
                    resource.setrlimit(
                        resource.RLIMIT_NPROC,
                        (self.config.resource_limits.max_processes,
                         self.config.resource_limits.max_processes)
                    )
            
            # 启动子进程
            self._process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.config.working_dir,
                env=env,
                preexec_fn=set_limits if os.name != 'nt' else None,
            )
            
            # 监控资源
            self.resource_monitor.start()
            
            try:
                stdout, stderr = self._process.communicate(timeout=self.config.timeout)
                execution_time = time.time() - start_time
                
                return SandboxResult(
                    success=self._process.returncode == 0,
                    return_code=self._process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    memory_used=self.resource_monitor.peak_memory,
                )
                
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                
                return SandboxResult(
                    success=False,
                    return_code=-1,
                    error_message=f"执行超时（超过 {self.config.timeout} 秒）",
                    execution_time=self.config.timeout,
                )
                
        except ResourceLimitError as e:
            if self._process:
                self._process.kill()
            return SandboxResult(
                success=False,
                return_code=-1,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            if self._process:
                self._process.kill()
            return SandboxResult(
                success=False,
                return_code=-1,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )
            
        finally:
            self.resource_monitor.stop()
            # 清理临时文件
            try:
                if script_path.exists():
                    script_path.unlink()
            except:
                pass
    
    def execute_file(self, file_path: Path, **kwargs) -> SandboxResult:
        """
        在沙箱中执行文件
        
        参数:
            file_path: 要执行的文件路径
            **kwargs: 额外参数
            
        返回:
            执行结果
        """
        # 验证路径权限
        self.path_validator.validate_read(file_path)
        
        # 读取代码
        code = file_path.read_text(encoding='utf-8')
        
        return self.execute(code, **kwargs)
    
    def terminate(self) -> None:
        """终止沙箱执行"""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


class SecureFileSystem:
    """
    安全文件系统
    
    提供受限的文件系统访问。
    """
    
    def __init__(self, validator: PathValidator, base_dir: Path):
        """
        初始化安全文件系统
        
        参数:
            validator: 路径验证器
            base_dir: 基础目录
        """
        self.validator = validator
        self.base_dir = Path(base_dir)
    
    def read_file(self, path: Union[str, Path]) -> str:
        """
        读取文件
        
        参数:
            path: 文件路径
            
        返回:
            文件内容
        """
        full_path = self._resolve_path(path)
        self.validator.validate_read(full_path)
        return full_path.read_text(encoding='utf-8')
    
    def write_file(self, path: Union[str, Path], content: str) -> None:
        """
        写入文件
        
        参数:
            path: 文件路径
            content: 文件内容
        """
        full_path = self._resolve_path(path)
        self.validator.validate_write(full_path)
        full_path.write_text(content, encoding='utf-8')
    
    def read_bytes(self, path: Union[str, Path]) -> bytes:
        """
        读取二进制文件
        
        参数:
            path: 文件路径
            
        返回:
            文件内容
        """
        full_path = self._resolve_path(path)
        self.validator.validate_read(full_path)
        return full_path.read_bytes()
    
    def write_bytes(self, path: Union[str, Path], content: bytes) -> None:
        """
        写入二进制文件
        
        参数:
            path: 文件路径
            content: 文件内容
        """
        full_path = self._resolve_path(path)
        self.validator.validate_write(full_path)
        full_path.write_bytes(content)
    
    def exists(self, path: Union[str, Path]) -> bool:
        """
        检查路径是否存在
        
        参数:
            path: 路径
            
        返回:
            是否存在
        """
        full_path = self._resolve_path(path)
        return full_path.exists()
    
    def is_file(self, path: Union[str, Path]) -> bool:
        """
        检查是否为文件
        
        参数:
            path: 路径
            
        返回:
            是否为文件
        """
        full_path = self._resolve_path(path)
        return full_path.is_file()
    
    def is_dir(self, path: Union[str, Path]) -> bool:
        """
        检查是否为目录
        
        参数:
            path: 路径
            
        返回:
            是否为目录
        """
        full_path = self._resolve_path(path)
        return full_path.is_dir()
    
    def list_dir(self, path: Union[str, Path] = ".") -> List[str]:
        """
        列出目录内容
        
        参数:
            path: 目录路径
            
        返回:
            文件名列表
        """
        full_path = self._resolve_path(path)
        self.validator.validate_read(full_path)
        return os.listdir(full_path)
    
    def mkdir(self, path: Union[str, Path], exist_ok: bool = False) -> None:
        """
        创建目录
        
        参数:
            path: 目录路径
            exist_ok: 是否允许已存在
        """
        full_path = self._resolve_path(path)
        self.validator.validate_write(full_path)
        full_path.mkdir(parents=True, exist_ok=exist_ok)
    
    def remove(self, path: Union[str, Path]) -> None:
        """
        删除文件或目录
        
        参数:
            path: 路径
        """
        full_path = self._resolve_path(path)
        self.validator.validate_write(full_path)
        
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """
        解析路径
        
        参数:
            path: 相对或绝对路径
            
        返回:
            完整路径
        """
        path = Path(path)
        
        if path.is_absolute():
            return path
        
        return self.base_dir / path


class IPCChannel:
    """
    进程间通信通道
    
    提供插件与核心框架之间的安全通信机制。
    """
    
    def __init__(self, plugin_id: str):
        """
        初始化通道
        
        参数:
            plugin_id: 插件 ID
        """
        self.plugin_id = plugin_id
        self._message_queue: List[Dict[str, Any]] = []
        self._handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
    
    def send(self, message_type: str, data: Dict[str, Any]) -> None:
        """
        发送消息
        
        参数:
            message_type: 消息类型
            data: 消息数据
        """
        message = {
            'type': message_type,
            'plugin_id': self.plugin_id,
            'timestamp': time.time(),
            'data': data,
        }
        
        with self._lock:
            self._message_queue.append(message)
    
    def receive(self) -> Optional[Dict[str, Any]]:
        """
        接收消息
        
        返回:
            消息或 None
        """
        with self._lock:
            if self._message_queue:
                return self._message_queue.pop(0)
        return None
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """
        注册消息处理器
        
        参数:
            message_type: 消息类型
            handler: 处理函数
        """
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)
    
    def process_messages(self) -> None:
        """处理消息队列"""
        while True:
            message = self.receive()
            if message is None:
                break
            
            handlers = self._handlers.get(message['type'], [])
            for handler in handlers:
                try:
                    handler(message['data'])
                except Exception as e:
                    logger.error(f"消息处理错误: {e}")


# 便捷函数
def create_sandbox(
    plugin_id: str,
    working_dir: Optional[Path] = None,
    **kwargs
) -> PluginSandbox:
    """
    创建沙箱
    
    参数:
        plugin_id: 插件 ID
        working_dir: 工作目录
        **kwargs: 其他配置参数
        
    返回:
        沙箱实例
    """
    config = SandboxConfig(
        plugin_id=plugin_id,
        working_dir=working_dir or Path(tempfile.mkdtemp()),
        **{k: v for k, v in kwargs.items() if k in [
            'allowed_paths', 'blocked_paths', 'env_vars', 
            'resource_limits', 'timeout'
        ]}
    )
    
    return PluginSandbox(config)


@contextmanager
def sandbox_context(plugin_id: str, **kwargs):
    """
    沙箱上下文管理器
    
    参数:
        plugin_id: 插件 ID
        **kwargs: 配置参数
        
    Yields:
        PluginSandbox: 沙箱实例
    """
    sandbox = create_sandbox(plugin_id, **kwargs)
    try:
        yield sandbox
    finally:
        sandbox.terminate()


# 单元测试存根
class TestSandbox:
    """Sandbox 单元测试"""
    
    def test_resource_limits(self) -> None:
        """测试资源限制"""
        limits = ResourceLimits(
            max_cpu_time=1,
            max_memory=1024 * 1024 * 100,  # 100MB
        )
        
        config = SandboxConfig(
            plugin_id="test",
            resource_limits=limits,
            timeout=5,
        )
        
        sandbox = PluginSandbox(config)
        
        # 测试正常代码
        result = sandbox.execute("print('Hello')")
        assert result.success
        
        # 测试超时
        result = sandbox.execute("import time; time.sleep(10)")
        assert not result.success
        assert "超时" in result.error_message
    
    def test_path_validator(self) -> None:
        """测试路径验证器"""
        validator = PathValidator(
            allowed_paths=[Path("/tmp/sandbox")],
            blocked_paths=[Path("/tmp/sandbox/secret")],
        )
        
        assert validator.is_allowed("/tmp/sandbox/file.txt")
        assert not validator.is_allowed("/tmp/sandbox/secret/data.txt")
        assert not validator.is_allowed("/etc/passwd")
    
    def test_secure_file_system(self, tmp_path) -> None:
        """测试安全文件系统"""
        validator = PathValidator(
            allowed_paths=[tmp_path],
            blocked_paths=[],
        )
        
        fs = SecureFileSystem(validator, tmp_path)
        
        # 测试写入和读取
        fs.write_file("test.txt", "Hello World")
        content = fs.read_file("test.txt")
        assert content == "Hello World"
        
        # 测试目录操作
        fs.mkdir("subdir")
        assert fs.is_dir("subdir")
        
        # 测试删除
        fs.remove("test.txt")
        assert not fs.exists("test.txt")
