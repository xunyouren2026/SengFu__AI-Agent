"""
工具注册与插件系统 - 工具包

提供工具基类、注册中心、安全管理、结果缓存和重试机制。
同时支持工具发现加载、LangChain 适配和 MCP 协议适配。
"""

from .base import (
    Tool,
    ToolResult,
    ToolParameter,
    tool,
)
from .registry import ToolRegistry
from .security import ToolSecurityManager, PermissionLevel
from .cache import ToolResultCache, CacheStats
from .retry import ToolRetryHandler, ToolRetryConfig

# 工具发现与动态加载
from .discovery_loader import (
    ToolDiscovery,
    FilesystemScanner,
    ModuleIntrospector,
    PluginDiscovery,
    LazyLoader,
    HotReloader,
    DependencyInjector,
    DiscoveredTool,
    PluginInfo,
    ToolSource,
    ToolFactory,
)

# LangChain 适配器
from .langchain_adapter import (
    LangChainAdapter,
    ToolConverter,
    ChainIntegrator,
    AgentCompatibility,
    CallbackHandler,
    MemoryIntegrator,
    ChainStep,
    AgentAction,
    AgentFinish,
    LangChainToolLike,
    LangChainCallback,
    LangChainMemory,
)

# MCP 适配器
from .mcp_adapter import (
    MCPAdapter,
    MCPServerDiscovery,
    MCPCapabilityNegotiator,
    MCPToolInvoker,
    MCPResourceAccessor,
    MCPPromptHandler,
    MCPMessage,
    MCPCapability,
    MCPToolInfo,
    MCPResource,
    MCPPrompt,
    MCPErrorCode,
)

__all__ = [
    # 基础组件
    "Tool",
    "ToolResult",
    "ToolParameter",
    "tool",
    "ToolRegistry",
    "ToolSecurityManager",
    "PermissionLevel",
    "ToolResultCache",
    "CacheStats",
    "ToolRetryHandler",
    "ToolRetryConfig",
    # 工具发现与动态加载
    "ToolDiscovery",
    "FilesystemScanner",
    "ModuleIntrospector",
    "PluginDiscovery",
    "LazyLoader",
    "HotReloader",
    "DependencyInjector",
    "DiscoveredTool",
    "PluginInfo",
    "ToolSource",
    "ToolFactory",
    # LangChain 适配器
    "LangChainAdapter",
    "ToolConverter",
    "ChainIntegrator",
    "AgentCompatibility",
    "CallbackHandler",
    "MemoryIntegrator",
    "ChainStep",
    "AgentAction",
    "AgentFinish",
    "LangChainToolLike",
    "LangChainCallback",
    "LangChainMemory",
    # MCP 适配器
    "MCPAdapter",
    "MCPServerDiscovery",
    "MCPCapabilityNegotiator",
    "MCPToolInvoker",
    "MCPResourceAccessor",
    "MCPPromptHandler",
    "MCPMessage",
    "MCPCapability",
    "MCPToolInfo",
    "MCPResource",
    "MCPPrompt",
    "MCPErrorCode",
]
