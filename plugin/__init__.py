#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Framework Plugin 系统

本模块提供插件系统的核心功能，包括插件 SDK、管理、沙箱和安全。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

# 版本信息
__version__ = "1.0.0"
__author__ = "AGI Framework Team"

# 从 sdk 模块导出
from .sdk import (
    # 核心类
    BasePlugin,
    PluginContext,
    PluginConfig,
    PluginManifest,
    PluginMetadata,
    PluginDependency,
    PluginAPIClient,
    
    # 扩展点
    PluginExtensionPoint,
    ToolProvider,
    CommandProvider,
    EventHandler,
    
    # 枚举
    PluginState,
    PluginPriority,
    
    # 异常
    PluginError,
    PluginLoadError,
    PluginConfigError,
    PluginRuntimeError,
    PluginAPIError,
    
    # 装饰器
    PluginHook,
    
    # 便捷函数
    create_plugin_manifest,
    validate_plugin_config,
)

# 从 manager 模块导出
from .manager import (
    # 核心类
    PluginManager,
    PluginLoader,
    DependencyResolver,
    InstalledPlugin,
    VersionConstraint,
    
    # 异常
    PluginNotFoundError,
    PluginConflictError,
    PluginDependencyError,
    PluginVersionError,
    
    # 便捷函数
    create_plugin_manager,
)

# 从 sandbox 模块导出
from .sandbox import (
    # 核心类
    PluginSandbox,
    SandboxConfig,
    SandboxResult,
    ResourceLimits,
    PathValidator,
    ResourceMonitor,
    SecureFileSystem,
    IPCChannel,
    
    # 异常
    SandboxError,
    ResourceLimitError,
    PermissionDeniedError,
    SandboxTimeoutError,
    SandboxExitError,
    
    # 便捷函数
    create_sandbox,
    sandbox_context,
)

# 从 security 模块导出
from .security import (
    # 核心类
    PluginSecurityManager,
    SignatureVerifier,
    IntegrityChecker,
    PermissionModel,
    AuditLogger,
    MaliciousPluginDetector,
    
    # 数据类
    PluginSignature,
    PluginChecksum,
    AuditLogEntry,
    
    # 枚举
    SignatureAlgorithm,
    
    # 异常
    SecurityError,
    SignatureVerificationError,
    IntegrityCheckError,
    PermissionDeniedError,
    MaliciousPluginDetectedError,
    
    # 便捷函数
    create_security_manager,
    verify_plugin_signature,
)

# 公开 API
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # SDK 模块
    "BasePlugin",
    "PluginContext",
    "PluginConfig",
    "PluginManifest",
    "PluginMetadata",
    "PluginDependency",
    "PluginAPIClient",
    "PluginExtensionPoint",
    "ToolProvider",
    "CommandProvider",
    "EventHandler",
    "PluginState",
    "PluginPriority",
    "PluginError",
    "PluginLoadError",
    "PluginConfigError",
    "PluginRuntimeError",
    "PluginAPIError",
    "PluginHook",
    "create_plugin_manifest",
    "validate_plugin_config",
    
    # Manager 模块
    "PluginManager",
    "PluginLoader",
    "DependencyResolver",
    "InstalledPlugin",
    "VersionConstraint",
    "PluginNotFoundError",
    "PluginConflictError",
    "PluginDependencyError",
    "PluginVersionError",
    "create_plugin_manager",
    
    # Sandbox 模块
    "PluginSandbox",
    "SandboxConfig",
    "SandboxResult",
    "ResourceLimits",
    "PathValidator",
    "ResourceMonitor",
    "SecureFileSystem",
    "IPCChannel",
    "SandboxError",
    "ResourceLimitError",
    "PermissionDeniedError",
    "SandboxTimeoutError",
    "SandboxExitError",
    "create_sandbox",
    "sandbox_context",
    
    # Security 模块
    "PluginSecurityManager",
    "SignatureVerifier",
    "IntegrityChecker",
    "PermissionModel",
    "AuditLogger",
    "MaliciousPluginDetector",
    "PluginSignature",
    "PluginChecksum",
    "AuditLogEntry",
    "SignatureAlgorithm",
    "SecurityError",
    "SignatureVerificationError",
    "IntegrityCheckError",
    "PermissionDeniedError",
    "MaliciousPluginDetectedError",
    "create_security_manager",
    "verify_plugin_signature",
]


def get_plugin_system_info() -> dict:
    """
    获取插件系统信息
    
    返回:
        包含系统信息的字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "modules": ["sdk", "manager", "sandbox", "security"],
        "features": [
            "Plugin lifecycle management",
            "Dependency resolution",
            "Sandboxed execution",
            "Security verification",
            "Permission model",
            "Audit logging",
        ],
    }


def init_plugin_system(
    plugin_dirs: list = None,
    data_dir: str = None,
    security_config_dir: str = None
) -> tuple[PluginManager, PluginSecurityManager]:
    """
    初始化插件系统
    
    参数:
        plugin_dirs: 插件目录列表
        data_dir: 数据目录
        security_config_dir: 安全配置目录
        
    返回:
        (插件管理器, 安全管理器) 元组
    """
    from pathlib import Path
    
    if data_dir is None:
        data_dir = Path.home() / ".agi_plugins" / "data"
    else:
        data_dir = Path(data_dir)
    
    if security_config_dir is None:
        security_config_dir = Path.home() / ".agi_plugins" / "security"
    else:
        security_config_dir = Path(security_config_dir)
    
    manager = create_plugin_manager(plugin_dirs, data_dir)
    security = create_security_manager(security_config_dir)
    
    return manager, security
