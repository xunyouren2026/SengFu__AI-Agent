"""
ClawHub 插件市场服务端模块

提供REST API服务、认证授权、插件存储和发布工作流功能。
"""

from .api import HubAPIServer, PluginAPI, SearchAPI, StatsAPI
from .auth import AuthManager, JWTAuth, OAuthHandler, PermissionManager, APIKeyManager
from .storage import PluginStorage, FileStorage, CDNStorage, VersionStorage, BackupManager
from .publisher import PublishWorkflow, ReviewQueue, AutoTester, PublishManager, UnpublishManager

__all__ = [
    # API
    "HubAPIServer",
    "PluginAPI",
    "SearchAPI",
    "StatsAPI",
    # Auth
    "AuthManager",
    "JWTAuth",
    "OAuthHandler",
    "PermissionManager",
    "APIKeyManager",
    # Storage
    "PluginStorage",
    "FileStorage",
    "CDNStorage",
    "VersionStorage",
    "BackupManager",
    # Publisher
    "PublishWorkflow",
    "ReviewQueue",
    "AutoTester",
    "PublishManager",
    "UnpublishManager",
]

__version__ = "1.0.0"
