"""
Anthropic API 异常定义模块

定义所有与Anthropic API交互时可能抛出的异常类型
"""

from typing import Any, Optional


class AnthropicError(Exception):
    """
    Anthropic API 基础异常类
    
    所有Anthropic相关异常的基类
    """
    
    def __init__(
        self,
        message: str,
        response: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.response = response
        self.body = body
    
    def __str__(self) -> str:
        if self.body:
            return f"{self.message}: {self.body}"
        return self.message


class APIError(AnthropicError):
    """
    API 请求错误
    
    当API返回非2xx状态码时抛出
    """
    
    def __init__(
        self,
        message: str,
        status_code: int,
        response: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> None:
        super().__init__(message, response, body)
        self.status_code = status_code


class AuthenticationError(APIError):
    """
    认证错误 (401)
    
    API密钥无效或已过期时抛出
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        response: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> None:
        super().__init__(message, 401, response, body)


class RateLimitError(APIError):
    """
    速率限制错误 (429)
    
    请求过于频繁时抛出
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        response: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> None:
        super().__init__(message, 429, response, body)
        self.retry_after = retry_after


class InvalidRequestError(APIError):
    """
    无效请求错误 (400)
    
    请求参数无效时抛出
    """
    
    def __init__(
        self,
        message: str = "Invalid request",
        response: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> None:
        super().__init__(message, 400, response, body)


class NotFoundError(APIError):
    """
    资源未找到错误 (404)
    
    请求的资源不存在时抛出
    """
    
    def __init__(
        self,
        message: str = "Resource not found",
        response: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> None:
        super().__init__(message, 404, response, body)


class ServerError(APIError):
    """
    服务器错误 (500-599)
    
    Anthropic服务器内部错误时抛出
    """
    
    def __init__(
        self,
        message: str = "Server error",
        status_code: int = 500,
        response: Optional[Any] = None,
        body: Optional[dict] = None,
    ) -> None:
        super().__init__(message, status_code, response, body)


class TimeoutError(AnthropicError):
    """
    请求超时错误
    
    请求在指定时间内未完成时抛出
    """
    
    def __init__(
        self,
        message: str = "Request timeout",
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.timeout = timeout


class StreamError(AnthropicError):
    """
    流式响应错误
    
    处理流式响应时发生错误
    """
    
    def __init__(
        self,
        message: str = "Stream processing error",
        chunk: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.chunk = chunk


class ToolError(AnthropicError):
    """
    工具调用错误
    
    工具定义或执行过程中发生错误
    """
    
    def __init__(
        self,
        message: str = "Tool error",
        tool_name: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class ValidationError(AnthropicError):
    """
    数据验证错误
    
    请求数据验证失败时抛出
    """
    pass
