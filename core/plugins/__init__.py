"""
插件系统

提供插件规范定义、发现机制、生命周期管理、依赖解析和沙箱隔离。
"""

from .spec import PluginSpec, PluginManifest, parse_manifest
from .discovery import PluginDiscovery, PluginInfo
from .lifecycle import PluginLifecycleManager, PluginState
from .dependency_resolver import DependencyResolver, DependencyGraph
from .sandbox import PluginSandbox, SandboxContext
from .installer import PluginInstaller, PackageDownloader, InstallerDependencyResolver
from .installer import SignatureVerifier, RollbackManager, PackageInfo, InstallResult
from .marketplace_client import MarketplaceClient, PluginSearch, PluginMetadata
from .marketplace_client import RatingAggregator, PackageDownloader as MarketPackageDownloader

__all__ = [
    # Core
    "PluginSpec",
    "PluginManifest",
    "parse_manifest",
    "PluginDiscovery",
    "PluginInfo",
    "PluginLifecycleManager",
    "PluginState",
    "DependencyResolver",
    "DependencyGraph",
    "PluginSandbox",
    "SandboxContext",
    # Installer
    "PluginInstaller",
    "PackageDownloader",
    "InstallerDependencyResolver",
    "SignatureVerifier",
    "RollbackManager",
    "PackageInfo",
    "InstallResult",
    # Marketplace
    "MarketplaceClient",
    "PluginSearch",
    "PluginMetadata",
    "RatingAggregator",
    "MarketPackageDownloader",
]
