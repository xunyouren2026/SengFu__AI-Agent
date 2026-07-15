"""
ClawHub 插件系统

提供完整的插件管理功能，包括Hub核心、服务端、内置插件和插件管理。
"""

# Hub核心
from .hub import (
    PluginRegistry,
    PluginMetadata,
    SecurityVerifier,
    RatingSystem,
    VersionManager,
    FullTextIndexer,
    DependencyResolver,
)

# Hub服务端
from .hub.server import (
    HubAPIServer,
    AuthManager,
    PluginStorage,
    PublishWorkflow,
)

# 内置插件
from .builtin import (
    WebSearchPlugin,
    FileOperationsPlugin,
    CodeRunnerPlugin,
    CalendarPlugin,
    ImageGenerationPlugin,
    DocumentPlugin,
    DatabasePlugin,
    APITesterPlugin,
    GitManagerPlugin,
    ScreenshotPlugin,
)

# 插件管理
from .loader import PluginLoader
from .sandbox import PluginSandbox
from .manager import PluginManager
from .marketplace_client import MarketplaceClient

__all__ = [
    # Hub核心
    "PluginRegistry",
    "PluginMetadata",
    "SecurityVerifier",
    "RatingSystem",
    "VersionManager",
    "FullTextIndexer",
    "DependencyResolver",
    # Hub服务端
    "HubAPIServer",
    "AuthManager",
    "PluginStorage",
    "PublishWorkflow",
    # 内置插件
    "WebSearchPlugin",
    "FileOperationsPlugin",
    "CodeRunnerPlugin",
    "CalendarPlugin",
    "ImageGenerationPlugin",
    "DocumentPlugin",
    "DatabasePlugin",
    "APITesterPlugin",
    "GitManagerPlugin",
    "ScreenshotPlugin",
    # 插件管理
    "PluginLoader",
    "PluginSandbox",
    "PluginManager",
    "MarketplaceClient",
]

__version__ = "1.0.0"
