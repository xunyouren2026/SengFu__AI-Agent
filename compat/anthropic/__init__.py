"""
Anthropic Claude API 兼容接口

提供完整的Anthropic Claude API客户端实现，支持：
- Messages API (消息API)
- Tool Use (工具使用)
- Vision (视觉/图像理解)
- Streaming (流式响应)
- 错误处理和重试机制

模块路径: compat/anthropic/__init__.py
"""

from typing import TYPE_CHECKING

# 版本信息
__version__ = "0.1.0"
__author__ = "AGI Unified Framework"

# 导出主要类和类型
from .messages import (
    AsyncMessages,
    Messages,
    Message,
    MessageParam,
    ContentBlock,
    Usage,
)

from .completions import (
    AsyncCompletions,
    Completions,
    CompletionResponse,
    CompletionRequest,
)

from .tools import (
    Tool,
    ToolResult,
    ToolUseBlock,
    ToolChoice,
    ToolManager,
)

from .vision import (
    VisionContent,
    ImageSource,
    VisionMessage,
    VisionClient,
)

from .exceptions import (
    AnthropicError,
    APIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    NotFoundError,
    ServerError,
)

if TYPE_CHECKING:
    from .client import AnthropicClient, AsyncAnthropicClient


class Anthropic:
    """
    Anthropic Claude API 同步客户端主类
    
    提供对Anthropic API的完整访问，包括：
    - messages: Messages API 用于对话
    - completions: 文本补全API
    
    示例:
        >>> client = Anthropic(api_key="your-api-key")
        >>> response = client.messages.create(
        ...     model="claude-3-sonnet-20240229",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        version: str = "2023-06-01",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        初始化Anthropic客户端
        
        Args:
            api_key: Anthropic API密钥，如果为None则从环境变量ANTHROPIC_API_KEY读取
            base_url: API基础URL
            version: API版本
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
        """
        # 延迟导入避免循环依赖
        from .client import AnthropicClient
        
        self._client = AnthropicClient(
            api_key=api_key,
            base_url=base_url,
            version=version,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.messages = self._client.messages
        self.completions = self._client.completions
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False
    
    def close(self) -> None:
        """关闭客户端连接"""
        self._client.close()


class AsyncAnthropic:
    """
    Anthropic Claude API 异步客户端主类
    
    提供对Anthropic API的完整异步访问
    
    示例:
        >>> async with AsyncAnthropic(api_key="your-api-key") as client:
        ...     response = await client.messages.create(
        ...         model="claude-3-sonnet-20240229",
        ...         max_tokens=1024,
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        version: str = "2023-06-01",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """
        初始化异步Anthropic客户端
        
        Args:
            api_key: Anthropic API密钥，如果为None则从环境变量ANTHROPIC_API_KEY读取
            base_url: API基础URL
            version: API版本
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
        """
        from .client import AsyncAnthropicClient
        
        self._client = AsyncAnthropicClient(
            api_key=api_key,
            base_url=base_url,
            version=version,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.messages = self._client.messages
        self.completions = self._client.completions
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
        return False
    
    async def close(self) -> None:
        """关闭异步客户端连接"""
        await self._client.close()


# 便捷函数：快速创建客户端
def create_client(
    api_key: str | None = None,
    async_mode: bool = False,
    **kwargs,
):
    """
    快速创建Anthropic客户端
    
    Args:
        api_key: API密钥
        async_mode: 是否创建异步客户端
        **kwargs: 其他配置参数
    
    Returns:
        Anthropic或AsyncAnthropic实例
    """
    if async_mode:
        return AsyncAnthropic(api_key=api_key, **kwargs)
    return Anthropic(api_key=api_key, **kwargs)


# 模块级导出
__all__ = [
    # 主客户端类
    "Anthropic",
    "AsyncAnthropic",
    "create_client",
    
    # Messages API
    "Messages",
    "AsyncMessages",
    "Message",
    "MessageParam",
    "ContentBlock",
    "Usage",
    
    # Completions API
    "Completions",
    "AsyncCompletions",
    "CompletionResponse",
    "CompletionRequest",
    
    # Tools
    "Tool",
    "ToolResult",
    "ToolUseBlock",
    "ToolChoice",
    "ToolManager",
    
    # Vision
    "VisionContent",
    "ImageSource",
    "VisionMessage",
    "VisionClient",
    
    # Exceptions
    "AnthropicError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "NotFoundError",
    "ServerError",
    
    # 元信息
    "__version__",
    "__author__",
]
