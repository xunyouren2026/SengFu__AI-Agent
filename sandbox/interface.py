"""
沙箱执行环境接口定义
定义SandboxExecutor抽象基类、ExecutionResult、SandboxConfig等核心接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import time
import json


class SandboxState(Enum):
    """沙箱状态枚举"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    CLEANUP = "cleanup"


class ExecutionStatus(Enum):
    """执行状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"
    ERROR = "error"
    PENDING = "pending"


class IsolationLevel(Enum):
    """隔离级别枚举"""
    NONE = "none"                    # 无隔离
    VIRTUALENV = "virtualenv"        # Python虚拟环境
    DOCKER = "docker"                # Docker容器
    NSJAIL = "nsjail"                # NsJail命名空间
    HYPERVISOR = "hypervisor"        # 虚拟机级别


@dataclass
class ResourceLimits:
    """资源限制配置"""
    cpu_quota: float = 1.0           # CPU配额 (核心数)
    cpu_period: int = 100000         # CPU周期 (微秒)
    memory_limit: int = 512 * 1024 * 1024  # 内存限制 (字节), 默认512MB
    memory_swap: int = 512 * 1024 * 1024   # Swap限制 (字节)
    disk_limit: int = 1 * 1024 * 1024 * 1024  # 磁盘限制 (字节), 默认1GB
    pids_limit: int = 100            # 进程数限制
    timeout: int = 60                # 执行超时 (秒)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'cpu_quota': self.cpu_quota,
            'cpu_period': self.cpu_period,
            'memory_limit': self.memory_limit,
            'memory_swap': self.memory_swap,
            'disk_limit': self.disk_limit,
            'pids_limit': self.pids_limit,
            'timeout': self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceLimits':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class NetworkConfig:
    """网络配置"""
    enabled: bool = False            # 是否启用网络
    allow_dns: bool = False          # 是否允许DNS查询
    allowed_hosts: List[str] = field(default_factory=list)  # 允许访问的主机
    allowed_ports: List[int] = field(default_factory=list)  # 允许访问的端口
    blocked_hosts: List[str] = field(default_factory=list)  # 禁止访问的主机
    ingress_rules: List[Dict[str, Any]] = field(default_factory=list)  # 入站规则
    egress_rules: List[Dict[str, Any]] = field(default_factory=list)   # 出站规则
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'allow_dns': self.allow_dns,
            'allowed_hosts': self.allowed_hosts,
            'allowed_ports': self.allowed_ports,
            'blocked_hosts': self.blocked_hosts,
            'ingress_rules': self.ingress_rules,
            'egress_rules': self.egress_rules
        }


@dataclass
class SecurityConfig:
    """安全配置"""
    seccomp_profile: Optional[str] = None  # Seccomp配置文件路径
    apparmor_profile: Optional[str] = None  # AppArmor配置文件
    selinux_context: Optional[str] = None   # SELinux上下文
    no_new_privileges: bool = True          # 禁止提升权限
    read_only_rootfs: bool = False          # 只读根文件系统
    drop_capabilities: List[str] = field(default_factory=lambda: [
        'CAP_SYS_ADMIN', 'CAP_NET_ADMIN', 'CAP_SYS_PTRACE',
        'CAP_SYS_MODULE', 'CAP_SYS_RAWIO', 'CAP_SYS_CHROOT'
    ])  # 要丢弃的能力
    add_capabilities: List[str] = field(default_factory=list)  # 要添加的能力
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'seccomp_profile': self.seccomp_profile,
            'apparmor_profile': self.apparmor_profile,
            'selinux_context': self.selinux_context,
            'no_new_privileges': self.no_new_privileges,
            'read_only_rootfs': self.read_only_rootfs,
            'drop_capabilities': self.drop_capabilities,
            'add_capabilities': self.add_capabilities
        }


@dataclass
class SandboxConfig:
    """沙箱配置"""
    name: str = "default_sandbox"
    isolation_level: IsolationLevel = IsolationLevel.DOCKER
    image: str = "python:3.10-slim"  # Docker镜像或执行环境
    workdir: str = "/sandbox"        # 工作目录
    environment: Dict[str, str] = field(default_factory=dict)  # 环境变量
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    mount_points: List[Dict[str, str]] = field(default_factory=list)  # 挂载点
    auto_cleanup: bool = True        # 自动清理
    log_level: str = "INFO"          # 日志级别
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'isolation_level': self.isolation_level.value,
            'image': self.image,
            'workdir': self.workdir,
            'environment': self.environment,
            'resource_limits': self.resource_limits.to_dict(),
            'network': self.network.to_dict(),
            'security': self.security.to_dict(),
            'mount_points': self.mount_points,
            'auto_cleanup': self.auto_cleanup,
            'log_level': self.log_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SandboxConfig':
        """从字典创建"""
        config = cls()
        config.name = data.get('name', config.name)
        config.isolation_level = IsolationLevel(data.get('isolation_level', 'docker'))
        config.image = data.get('image', config.image)
        config.workdir = data.get('workdir', config.workdir)
        config.environment = data.get('environment', {})
        if 'resource_limits' in data:
            config.resource_limits = ResourceLimits.from_dict(data['resource_limits'])
        if 'network' in data:
            net_data = data['network']
            config.network = NetworkConfig(
                enabled=net_data.get('enabled', False),
                allow_dns=net_data.get('allow_dns', False),
                allowed_hosts=net_data.get('allowed_hosts', []),
                allowed_ports=net_data.get('allowed_ports', []),
                blocked_hosts=net_data.get('blocked_hosts', []),
                ingress_rules=net_data.get('ingress_rules', []),
                egress_rules=net_data.get('egress_rules', [])
            )
        if 'security' in data:
            sec_data = data['security']
            config.security = SecurityConfig(
                seccomp_profile=sec_data.get('seccomp_profile'),
                apparmor_profile=sec_data.get('apparmor_profile'),
                selinux_context=sec_data.get('selinux_context'),
                no_new_privileges=sec_data.get('no_new_privileges', True),
                read_only_rootfs=sec_data.get('read_only_rootfs', False),
                drop_capabilities=sec_data.get('drop_capabilities', config.security.drop_capabilities),
                add_capabilities=sec_data.get('add_capabilities', [])
            )
        config.mount_points = data.get('mount_points', [])
        config.auto_cleanup = data.get('auto_cleanup', True)
        config.log_level = data.get('log_level', 'INFO')
        return config
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SandboxConfig':
        """从JSON字符串创建"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ExecutionResult:
    """执行结果"""
    execution_id: str                 # 执行ID
    status: ExecutionStatus           # 执行状态
    exit_code: int                    # 退出码
    stdout: str                       # 标准输出
    stderr: str                       # 标准错误
    start_time: float                 # 开始时间
    end_time: float                   # 结束时间
    duration: float                   # 执行时长 (秒)
    resource_usage: Dict[str, Any] = field(default_factory=dict)  # 资源使用情况
    output_files: Dict[str, str] = field(default_factory=dict)    # 输出文件 {文件名: 内容}
    metadata: Dict[str, Any] = field(default_factory=dict)        # 元数据
    error_message: Optional[str] = None  # 错误信息
    warnings: List[str] = field(default_factory=list)  # 警告信息
    
    @property
    def success(self) -> bool:
        """是否执行成功"""
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def timed_out(self) -> bool:
        """是否超时"""
        return self.status == ExecutionStatus.TIMEOUT
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execution_id': self.execution_id,
            'status': self.status.value,
            'exit_code': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'resource_usage': self.resource_usage,
            'output_files': self.output_files,
            'metadata': self.metadata,
            'error_message': self.error_message,
            'warnings': self.warnings
        }
    
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def create_success(
        cls,
        execution_id: str,
        stdout: str,
        stderr: str,
        exit_code: int = 0,
        start_time: Optional[float] = None,
        resource_usage: Optional[Dict[str, Any]] = None
    ) -> 'ExecutionResult':
        """创建成功结果"""
        end_time = time.time()
        actual_start = start_time or (end_time - 1)
        return cls(
            execution_id=execution_id,
            status=ExecutionStatus.SUCCESS,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            start_time=actual_start,
            end_time=end_time,
            duration=end_time - actual_start,
            resource_usage=resource_usage or {}
        )
    
    @classmethod
    def create_error(
        cls,
        execution_id: str,
        error_message: str,
        stderr: str = "",
        start_time: Optional[float] = None
    ) -> 'ExecutionResult':
        """创建错误结果"""
        end_time = time.time()
        actual_start = start_time or (end_time - 1)
        return cls(
            execution_id=execution_id,
            status=ExecutionStatus.ERROR,
            exit_code=-1,
            stdout="",
            stderr=stderr,
            start_time=actual_start,
            end_time=end_time,
            duration=end_time - actual_start,
            error_message=error_message
        )
    
    @classmethod
    def create_timeout(
        cls,
        execution_id: str,
        stdout: str,
        stderr: str,
        start_time: float,
        timeout: int
    ) -> 'ExecutionResult':
        """创建超时结果"""
        end_time = time.time()
        return cls(
            execution_id=execution_id,
            status=ExecutionStatus.TIMEOUT,
            exit_code=-9,  # SIGKILL
            stdout=stdout,
            stderr=stderr,
            start_time=start_time,
            end_time=end_time,
            duration=timeout,
            error_message=f"Execution timed out after {timeout} seconds"
        )


@dataclass
class ExecutionContext:
    """执行上下文"""
    execution_id: str                 # 执行ID
    code: str                         # 要执行的代码
    language: str = "python"          # 编程语言
    args: List[str] = field(default_factory=list)  # 命令行参数
    input_data: Optional[str] = None  # 输入数据 (stdin)
    files: Dict[str, str] = field(default_factory=dict)  # 输入文件 {文件名: 内容}
    working_directory: Optional[str] = None  # 工作目录
    timeout: Optional[int] = None     # 超时覆盖
    callbacks: Dict[str, Callable] = field(default_factory=dict)  # 回调函数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execution_id': self.execution_id,
            'code': self.code,
            'language': self.language,
            'args': self.args,
            'input_data': self.input_data,
            'files': self.files,
            'working_directory': self.working_directory,
            'timeout': self.timeout
        }


class SandboxExecutor(ABC):
    """
    沙箱执行器抽象基类
    定义所有沙箱执行器必须实现的接口
    """
    
    def __init__(self, config: SandboxConfig):
        """
        初始化执行器
        
        Args:
            config: 沙箱配置
        """
        self.config = config
        self._state = SandboxState.IDLE
        self._execution_count = 0
        self._last_execution_time: Optional[float] = None
    
    @property
    def state(self) -> SandboxState:
        """获取当前状态"""
        return self._state
    
    @property
    def name(self) -> str:
        """获取执行器名称"""
        return self.config.name
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化沙箱环境
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        执行代码
        
        Args:
            context: 执行上下文
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    def execute_batch(self, contexts: List[ExecutionContext]) -> List[ExecutionResult]:
        """
        批量执行代码
        
        Args:
            contexts: 执行上下文列表
            
        Returns:
            执行结果列表
        """
        pass
    
    @abstractmethod
    def stop(self, execution_id: str) -> bool:
        """
        停止执行
        
        Args:
            execution_id: 执行ID
            
        Returns:
            是否停止成功
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """
        清理沙箱环境
        
        Returns:
            是否清理成功
        """
        pass
    
    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取资源使用情况
        
        Returns:
            资源使用字典
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            是否健康
        """
        pass
    
    def pause(self) -> bool:
        """
        暂停执行器
        
        Returns:
            是否暂停成功
        """
        if self._state == SandboxState.RUNNING:
            self._state = SandboxState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """
        恢复执行器
        
        Returns:
            是否恢复成功
        """
        if self._state == SandboxState.PAUSED:
            self._state = SandboxState.RUNNING
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取执行器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'name': self.name,
            'state': self._state.value,
            'isolation_level': self.config.isolation_level.value,
            'execution_count': self._execution_count,
            'last_execution_time': self._last_execution_time,
            'config': self.config.to_dict()
        }
    
    def validate_context(self, context: ExecutionContext) -> List[str]:
        """
        验证执行上下文
        
        Args:
            context: 执行上下文
            
        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors = []
        
        if not context.execution_id:
            errors.append("Execution ID is required")
        
        if not context.code:
            errors.append("Code is required")
        
        if context.timeout and context.timeout > self.config.resource_limits.timeout:
            errors.append(f"Timeout {context.timeout} exceeds maximum allowed {self.config.resource_limits.timeout}")
        
        return errors
    
    def __enter__(self) -> 'SandboxExecutor':
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        if self.config.auto_cleanup:
            self.cleanup()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, state={self._state.value})"


class SandboxFactory:
    """沙箱执行器工厂"""
    
    _executors: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, executor_class: type) -> None:
        """
        注册执行器类型
        
        Args:
            name: 执行器名称
            executor_class: 执行器类
        """
        cls._executors[name] = executor_class
    
    @classmethod
    def create(cls, name: str, config: SandboxConfig) -> SandboxExecutor:
        """
        创建执行器实例
        
        Args:
            name: 执行器名称
            config: 沙箱配置
            
        Returns:
            执行器实例
        """
        if name not in cls._executors:
            raise ValueError(f"Unknown executor type: {name}")
        return cls._executors[name](config)
    
    @classmethod
    def list_executors(cls) -> List[str]:
        """列出所有已注册的执行器"""
        return list(cls._executors.keys())


class SandboxManager:
    """沙箱管理器 - 管理多个沙箱实例"""
    
    def __init__(self):
        self._sandboxes: Dict[str, SandboxExecutor] = {}
        self._configs: Dict[str, SandboxConfig] = {}
    
    def create_sandbox(self, config: SandboxConfig) -> str:
        """
        创建沙箱
        
        Args:
            config: 沙箱配置
            
        Returns:
            沙箱ID
        """
        sandbox_id = config.name
        executor = SandboxFactory.create(
            config.isolation_level.value,
            config
        )
        executor.initialize()
        self._sandboxes[sandbox_id] = executor
        self._configs[sandbox_id] = config
        return sandbox_id
    
    def get_sandbox(self, sandbox_id: str) -> Optional[SandboxExecutor]:
        """获取沙箱"""
        return self._sandboxes.get(sandbox_id)
    
    def execute_in_sandbox(
        self,
        sandbox_id: str,
        context: ExecutionContext
    ) -> ExecutionResult:
        """在指定沙箱中执行"""
        sandbox = self.get_sandbox(sandbox_id)
        if sandbox is None:
            return ExecutionResult.create_error(
                context.execution_id,
                f"Sandbox not found: {sandbox_id}"
            )
        return sandbox.execute(context)
    
    def destroy_sandbox(self, sandbox_id: str) -> bool:
        """销毁沙箱"""
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox:
            sandbox.cleanup()
            self._configs.pop(sandbox_id, None)
            return True
        return False
    
    def list_sandboxes(self) -> List[str]:
        """列出所有沙箱"""
        return list(self._sandboxes.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有沙箱统计信息"""
        return {
            sandbox_id: sandbox.get_stats()
            for sandbox_id, sandbox in self._sandboxes.items()
        }
    
    def cleanup_all(self) -> None:
        """清理所有沙箱"""
        for sandbox_id in list(self._sandboxes.keys()):
            self.destroy_sandbox(sandbox_id)
