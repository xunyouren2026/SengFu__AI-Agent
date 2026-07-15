"""
操作系统工具模块
导出所有OS相关工具类
"""

from .file_ops import (
    FileOperations,
    PermissionChecker,
    BatchOperations,
    FileInfo,
    OperationResult,
    safe_open,
    atomic_write,
)

from .process import (
    ProcessManager,
    ProcessInfo,
    ProcessResult,
    ProcessState,
    ProcessPriority,
)

from .shell_executor import (
    ShellExecutor,
    CommandWhitelist,
    ArgumentFilter,
    CommandRule,
    ExecutionResult,
    ExecutionStatus,
    SafeCommandBuilder,
)

from .system_monitor import (
    SystemMonitor,
    ProcessMonitor,
    AlertManager,
    SystemMetrics,
    CPUMetrics,
    MemoryMetrics,
    DiskMetrics,
    NetworkMetrics,
    Alert,
    MetricType,
)

from .network import (
    PortScanner,
    ConnectionTester,
    NetworkInfo,
    FirewallManager,
    NetworkTools,
    PortInfo,
    PortState,
    Protocol,
    ConnectionInfo,
    NetworkInterface,
    FirewallRule,
    ScanResult,
)


__all__ = [
    # 文件操作
    'FileOperations',
    'PermissionChecker',
    'BatchOperations',
    'FileInfo',
    'OperationResult',
    'safe_open',
    'atomic_write',
    
    # 进程管理
    'ProcessManager',
    'ProcessInfo',
    'ProcessResult',
    'ProcessState',
    'ProcessPriority',
    
    # Shell执行器
    'ShellExecutor',
    'CommandWhitelist',
    'ArgumentFilter',
    'CommandRule',
    'ExecutionResult',
    'ExecutionStatus',
    'SafeCommandBuilder',
    
    # 系统监控
    'SystemMonitor',
    'ProcessMonitor',
    'AlertManager',
    'SystemMetrics',
    'CPUMetrics',
    'MemoryMetrics',
    'DiskMetrics',
    'NetworkMetrics',
    'Alert',
    'MetricType',
    
    # 网络工具
    'PortScanner',
    'ConnectionTester',
    'NetworkInfo',
    'FirewallManager',
    'NetworkTools',
    'PortInfo',
    'PortState',
    'Protocol',
    'ConnectionInfo',
    'NetworkInterface',
    'FirewallRule',
    'ScanResult',
]


# 版本信息
__version__ = '1.0.0'
__author__ = 'AGI Unified Framework'
