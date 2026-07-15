"""
沙箱执行环境
提供完整的代码隔离执行能力
"""

# 核心接口
from .interface import (
    SandboxExecutor,
    SandboxConfig,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    SandboxState,
    IsolationLevel,
    ResourceLimits,
    NetworkConfig,
    SecurityConfig,
    SandboxFactory,
    SandboxManager
)

# 执行器
from .docker_executor import (
    DockerClient,
    DockerExecutor
)

from .nsjail_executor import (
    NsJailConfig,
    NsJailConfigWriter,
    NsJailClient,
    NsJailExecutor,
    UnshareExecutor
)

from .virtualenv_executor import (
    VirtualEnvManager,
    VirtualEnvExecutor,
    SubprocessExecutor
)

# 安全模块
from .security.seccomp_profiles import (
    SeccompAction,
    SeccompProfile,
    SeccompProfileBuilder,
    SeccompValidator,
    SyscallRule,
    SyscallArch,
    DEFAULT_PROFILE,
    STRICT_PROFILE,
    PYTHON_RUNTIME_PROFILE
)

from .security.network_policy import (
    NetworkAction,
    NetworkPolicy,
    NetworkPolicyBuilder,
    NetworkRule,
    NetworkDirection,
    Protocol,
    DNSConfig,
    PortRange,
    ISOLATED_POLICY,
    WEB_ONLY_POLICY
)

# 资源管理
from .resource.monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    ResourceThreshold,
    ResourceType,
    MonitorState,
    ProcfsReader,
    CgroupMonitor
)

from .resource.limiter import (
    ResourceLimiter,
    CgroupController,
    CgroupConfig,
    CgroupVersion,
    ProcessResourceLimiter
)

# 文件系统
from .fs_manager import (
    FileSystemManager,
    FileSystemType,
    MountPoint,
    FilePermission,
    OverlayFS,
    QuotaManager
)

# 结果收集
from .result_collector import (
    ResultCollector,
    CollectedResult,
    OutputChunk,
    OutputType,
    StreamingResultCollector,
    ResultAggregator
)

# 漏洞检测
from .vulnerability_check import (
    VulnerabilityScanner,
    VulnerabilityChecker,
    Vulnerability,
    VulnerabilityType,
    ScanResult,
    Severity
)

# 报告生成
from .report_generator import (
    ReportGenerator,
    ReportFormat,
    ReportSection,
    ExecutionReport,
    BatchReportGenerator
)

# 清理守护进程
from .cleanup_daemon import (
    CleanupDaemon,
    CleanupConfig,
    CleanupResult,
    CleanupTarget,
    DaemonState,
    ContainerCleaner,
    TempFileCleaner,
    ProcessCleaner,
    ResourceTracker
)


# 注册默认执行器
def _register_default_executors():
    """注册默认执行器"""
    SandboxFactory.register('docker', DockerExecutor)
    SandboxFactory.register('nsjail', NsJailExecutor)
    SandboxFactory.register('virtualenv', VirtualEnvExecutor)
    SandboxFactory.register('none', SubprocessExecutor)


# 自动注册
_register_default_executors()


__all__ = [
    # 接口
    'SandboxExecutor',
    'SandboxConfig',
    'ExecutionContext',
    'ExecutionResult',
    'ExecutionStatus',
    'SandboxState',
    'IsolationLevel',
    'ResourceLimits',
    'NetworkConfig',
    'SecurityConfig',
    'SandboxFactory',
    'SandboxManager',
    
    # Docker执行器
    'DockerClient',
    'DockerExecutor',
    
    # NsJail执行器
    'NsJailConfig',
    'NsJailConfigWriter',
    'NsJailClient',
    'NsJailExecutor',
    'UnshareExecutor',
    
    # 虚拟环境执行器
    'VirtualEnvManager',
    'VirtualEnvExecutor',
    'SubprocessExecutor',
    
    # Seccomp安全
    'SeccompAction',
    'SeccompProfile',
    'SeccompProfileBuilder',
    'SeccompValidator',
    'SyscallRule',
    'SyscallArch',
    'DEFAULT_PROFILE',
    'STRICT_PROFILE',
    'PYTHON_RUNTIME_PROFILE',
    
    # 网络策略
    'NetworkAction',
    'NetworkPolicy',
    'NetworkPolicyBuilder',
    'NetworkRule',
    'NetworkDirection',
    'Protocol',
    'DNSConfig',
    'PortRange',
    'ISOLATED_POLICY',
    'WEB_ONLY_POLICY',
    
    # 资源监控
    'ResourceMonitor',
    'ResourceSnapshot',
    'ResourceThreshold',
    'ResourceType',
    'MonitorState',
    'ProcfsReader',
    'CgroupMonitor',
    
    # 资源限制
    'ResourceLimiter',
    'CgroupController',
    'CgroupConfig',
    'CgroupVersion',
    'ProcessResourceLimiter',
    
    # 文件系统
    'FileSystemManager',
    'FileSystemType',
    'MountPoint',
    'FilePermission',
    'OverlayFS',
    'QuotaManager',
    
    # 结果收集
    'ResultCollector',
    'CollectedResult',
    'OutputChunk',
    'OutputType',
    'StreamingResultCollector',
    'ResultAggregator',
    
    # 漏洞检测
    'VulnerabilityScanner',
    'VulnerabilityChecker',
    'Vulnerability',
    'VulnerabilityType',
    'ScanResult',
    'Severity',
    
    # 报告生成
    'ReportGenerator',
    'ReportFormat',
    'ReportSection',
    'ExecutionReport',
    'BatchReportGenerator',
    
    # 清理守护进程
    'CleanupDaemon',
    'CleanupConfig',
    'CleanupResult',
    'CleanupTarget',
    'DaemonState',
    'ContainerCleaner',
    'TempFileCleaner',
    'ProcessCleaner',
    'ResourceTracker',
]

__version__ = '1.0.0'
__author__ = 'AGI Unified Framework'
