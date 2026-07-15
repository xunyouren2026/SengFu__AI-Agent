"""MCP (Model Context Protocol) 模块。

本模块实现了完整的MCP协议支持，包括：
- 消息Schema定义
- 传输层实现（stdio、WebSocket）
- 认证机制（API Key、OAuth2）
- 服务器和客户端
- 工具格式转换
- 资源管理
- 提示词模板
- 流式响应
- 服务发现
"""

from __future__ import annotations

# Schema模块
from .schema import (
    MCPErrorCode,
    MCPMethod,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    MCPToolInputSchema,
    MCPTool,
    MCPResource,
    MCPResourceContent,
    MCPPromptArgument,
    MCPPrompt,
    MCPInitializeParams,
    MCPServerCapabilities,
    MCPInitializeResult,
    MCPToolCallParams,
    MCPToolCallResult,
    MCPProgressNotification,
    MCPRequest,
    MCPResponse,
    MCPError,
)

# Stdio传输层
from .stdio_transport import (
    Transport,
    TransportMessage as StdioTransportMessage,
    StdioTransport,
    StdioServerTransport,
    StdioClientTransport,
)

# WebSocket传输层
from .websocket_transport import (
    WebSocketOpcode,
    WebSocketState,
    WebSocketFrame,
    TransportMessage as WebSocketTransportMessage,
    WebSocketTransport,
    WebSocketServerTransport,
)

# 认证模块
from .auth import (
    AuthType,
    AuthError,
    InvalidCredentialsError,
    ExpiredTokenError,
    InsufficientScopeError,
    AuthCredentials,
    APIKeyCredentials,
    BearerCredentials,
    BasicCredentials,
    OAuth2Token,
    OAuth2Credentials,
    AuthProvider,
    APIKeyAuthProvider,
    BearerAuthProvider,
    OAuth2AuthProvider,
    AuthMiddleware,
    AuthManager,
)

# 服务器模块
from .server import (
    ServerState,
    RequestContext,
    MCPServer,
    MCPServerRunner,
    ToolDecorator,
)

# 客户端模块
from .client import (
    ClientState,
    ClientConfig,
    MCPClient,
    MCPClientPool,
)

# 工具转换模块
from .tool_converter import (
    ToolFormat,
    InternalTool,
    OpenAITool,
    AnthropicTool,
    ToolConverter,
    ToolRegistry,
    tool,
)

# 资源管理模块
from .resource_manager import (
    ResourceType,
    ResourceError,
    ResourceNotFoundError,
    ResourceAccessError,
    ResourceMetadata,
    ResourceProvider,
    FileResourceProvider,
    MemoryResourceProvider,
    DatabaseResourceProvider,
    ResourceManager,
)

# 提示词注册模块
from .prompt_registry import (
    PromptCategory,
    PromptTemplate,
    TemplateEngine,
    SimpleTemplateEngine,
    PythonTemplateEngine,
    ConditionalTemplateEngine,
    PromptRegistry,
    PromptBuilder,
    prompt,
    BUILTIN_PROMPTS,
)

# 流式响应模块
from .streaming import (
    SSEEventType,
    SSEEvent,
    StreamProducer,
    StreamConsumer,
    StreamingResponse,
    StreamingToolExecutor,
    ChunkedStream,
    stream_generator,
)

# 服务发现模块
from .discovery import (
    DiscoverySource,
    ServiceStatus,
    ServiceEndpoint,
    ServiceInfo,
    ServiceDiscovery,
    ServiceRegistry,
)


__all__ = [
    # Schema
    "MCPErrorCode",
    "MCPMethod",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    "MCPToolInputSchema",
    "MCPTool",
    "MCPResource",
    "MCPResourceContent",
    "MCPPromptArgument",
    "MCPPrompt",
    "MCPInitializeParams",
    "MCPServerCapabilities",
    "MCPInitializeResult",
    "MCPToolCallParams",
    "MCPToolCallResult",
    "MCPProgressNotification",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    
    # Stdio Transport
    "Transport",
    "StdioTransportMessage",
    "StdioTransport",
    "StdioServerTransport",
    "StdioClientTransport",
    
    # WebSocket Transport
    "WebSocketOpcode",
    "WebSocketState",
    "WebSocketFrame",
    "WebSocketTransportMessage",
    "WebSocketTransport",
    "WebSocketServerTransport",
    
    # Auth
    "AuthType",
    "AuthError",
    "InvalidCredentialsError",
    "ExpiredTokenError",
    "InsufficientScopeError",
    "AuthCredentials",
    "APIKeyCredentials",
    "BearerCredentials",
    "BasicCredentials",
    "OAuth2Token",
    "OAuth2Credentials",
    "AuthProvider",
    "APIKeyAuthProvider",
    "BearerAuthProvider",
    "OAuth2AuthProvider",
    "AuthMiddleware",
    "AuthManager",
    
    # Server
    "ServerState",
    "RequestContext",
    "MCPServer",
    "MCPServerRunner",
    "ToolDecorator",
    
    # Client
    "ClientState",
    "ClientConfig",
    "MCPClient",
    "MCPClientPool",
    
    # Tool Converter
    "ToolFormat",
    "InternalTool",
    "OpenAITool",
    "AnthropicTool",
    "ToolConverter",
    "ToolRegistry",
    "tool",
    
    # Resource Manager
    "ResourceType",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceAccessError",
    "ResourceMetadata",
    "ResourceProvider",
    "FileResourceProvider",
    "MemoryResourceProvider",
    "DatabaseResourceProvider",
    "ResourceManager",
    
    # Prompt Registry
    "PromptCategory",
    "PromptTemplate",
    "TemplateEngine",
    "SimpleTemplateEngine",
    "PythonTemplateEngine",
    "ConditionalTemplateEngine",
    "PromptRegistry",
    "PromptBuilder",
    "prompt",
    "BUILTIN_PROMPTS",
    
    # Streaming
    "SSEEventType",
    "SSEEvent",
    "StreamProducer",
    "StreamConsumer",
    "StreamingResponse",
    "StreamingToolExecutor",
    "ChunkedStream",
    "stream_generator",
    
    # Discovery
    "DiscoverySource",
    "ServiceStatus",
    "ServiceEndpoint",
    "ServiceInfo",
    "ServiceDiscovery",
    "ServiceRegistry",
]


__version__ = "1.0.0"
