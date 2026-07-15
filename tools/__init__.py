"""
工具模块
提供各种工具类的统一导入接口
"""

from .builtins import (
    # 文件读取
    FileReader,
    TextFileReader,
    JSONFileReader,
    CSVFileReader,
    MarkdownFileReader,
    YAMLFileReader,
    BinaryFileReader,
    EncodingDetector,
    FileMetadataExtractor,
    PathValidator,
    ReadResult,
    FileType,

    # 网页搜索
    WebSearch,
    DuckDuckGoSearch,
    BingSearch,
    SearchAggregator,
    RateLimiter,
    ResultCache,
    SearchHistory,
    SafeSearchFilter,
    SearchResult,
    SearchProvider,
    SearchResponse,
)

from .os import (
    FileOperations,
    PermissionChecker,
    BatchOperations,
    FileInfo,
    OperationResult,
    safe_open,
    atomic_write,
)

from .os import (
    ProcessManager,
    ProcessInfo,
    ProcessResult,
    ProcessState,
    ProcessPriority,
)

from .os import (
    ShellExecutor,
    CommandWhitelist,
    ArgumentFilter,
    CommandRule,
    ExecutionResult,
    ExecutionStatus,
    SafeCommandBuilder,
)

from .os import (
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

from .os import (
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
    # 内置工具 - 文件读取
    'FileReader',
    'TextFileReader',
    'JSONFileReader',
    'CSVFileReader',
    'MarkdownFileReader',
    'YAMLFileReader',
    'BinaryFileReader',
    'EncodingDetector',
    'FileMetadataExtractor',
    'PathValidator',
    'ReadResult',
    'FileType',

    # 内置工具 - 网页搜索
    'WebSearch',
    'DuckDuckGoSearch',
    'BingSearch',
    'SearchAggregator',
    'RateLimiter',
    'ResultCache',
    'SearchHistory',
    'SafeSearchFilter',
    'SearchResult',
    'SearchProvider',
    'SearchResponse',

    # 操作系统工具 - 文件操作
    'FileOperations',
    'PermissionChecker',
    'BatchOperations',
    'FileInfo',
    'OperationResult',
    'safe_open',
    'atomic_write',

    # 操作系统工具 - 进程管理
    'ProcessManager',
    'ProcessInfo',
    'ProcessResult',
    'ProcessState',
    'ProcessPriority',

    # 操作系统工具 - Shell执行器
    'ShellExecutor',
    'CommandWhitelist',
    'ArgumentFilter',
    'CommandRule',
    'ExecutionResult',
    'ExecutionStatus',
    'SafeCommandBuilder',

    # 操作系统工具 - 系统监控
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

    # 操作系统工具 - 网络工具
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


__version__ = '1.0.0'
__author__ = 'AGI Unified Framework'
