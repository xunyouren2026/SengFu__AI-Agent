"""
Anthropic API HTTP 客户端模块

提供底层HTTP通信功能，包括：
- 同步和异步HTTP客户端
- 请求重试机制
- 错误处理和转换
- 认证管理

使用httpx库实现高性能HTTP通信
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Union

import httpx

from .exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from .messages import AsyncMessages, Messages
from .completions import AsyncCompletions, Completions


DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_VERSION = "2023-06-01"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3


class HTTPClient:
    """
    同步HTTP客户端基类
    
    封装httpx客户端，提供统一的请求接口
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        version: str = DEFAULT_VERSION,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        初始化HTTP客户端
        
        Args:
            api_key: Anthropic API密钥
            base_url: API基础URL
            version: API版本
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key is required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 创建httpx客户端
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers=self._default_headers(),
        )
    
    def _default_headers(self) -> Dict[str, str]:
        """
        获取默认请求头
        
        Returns:
            默认请求头字典
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "Content-Type": "application/json",
            "User-Agent": "anthropic-python/0.1.0",
        }
    
    def _handle_error(self, response: httpx.Response) -> None:
        """
        处理HTTP错误响应
        
        Args:
            response: HTTP响应对象
        
        Raises:
            根据状态码抛出相应的异常
        """
        if response.status_code < 400:
            return
        
        try:
            body = response.json()
        except Exception:
            body = {"error": response.text}
        
        error_message = body.get("error", {}).get("message", "Unknown error")
        
        if response.status_code == 401:
            raise AuthenticationError(error_message, response, body)
        elif response.status_code == 400:
            raise InvalidRequestError(error_message, response, body)
        elif response.status_code == 404:
            raise NotFoundError(error_message, response, body)
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                error_message,
                retry_after=float(retry_after) if retry_after else None,
                response=response,
                body=body,
            )
        elif 500 <= response.status_code < 600:
            raise ServerError(
                error_message,
                status_code=response.status_code,
                response=response,
                body=body,
            )
        else:
            raise APIError(
                error_message,
                status_code=response.status_code,
                response=response,
                body=body,
            )
    
    def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """
        带重试机制的请求
        
        Args:
            method: HTTP方法
            path: 请求路径
            **kwargs: 其他请求参数
        
        Returns:
            HTTP响应对象
        
        Raises:
            请求失败时抛出相应异常
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                response = self._client.request(method, path, **kwargs)
                self._handle_error(response)
                return response
                
            except (RateLimitError, ServerError) as e:
                last_exception = e
                # 可重试错误，等待后重试
                if attempt < self.max_retries - 1:
                    if isinstance(e, RateLimitError) and e.retry_after:
                        time.sleep(e.retry_after)
                    else:
                        time.sleep(2 ** attempt)  # 指数退避
                continue
                
            except httpx.TimeoutException as e:
                last_exception = TimeoutError(f"Request timeout: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
                
            except (AuthenticationError, InvalidRequestError, NotFoundError):
                # 不可重试错误，直接抛出
                raise
        
        # 重试耗尽，抛出最后一个异常
        if last_exception:
            raise last_exception
        
        raise APIError("Max retries exceeded")
    
    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], httpx.Response]:
        """
        发送GET请求
        
        Args:
            path: 请求路径
            params: 查询参数
            **kwargs: 其他参数
        
        Returns:
            响应数据或响应对象
        """
        response = self._request_with_retry("GET", path, params=params, **kwargs)
        
        if kwargs.get("stream"):
            return response
        
        return response.json()
    
    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], httpx.Response]:
        """
        发送POST请求
        
        Args:
            path: 请求路径
            json: JSON请求体
            **kwargs: 其他参数
        
        Returns:
            响应数据或响应对象
        """
        response = self._request_with_retry("POST", path, json=json, **kwargs)
        
        if kwargs.get("stream"):
            return response
        
        return response.json()
    
    def close(self) -> None:
        """关闭客户端连接"""
        self._client.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False


class AsyncHTTPClient:
    """
    异步HTTP客户端基类
    
    封装httpx异步客户端，提供统一的异步请求接口
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        version: str = DEFAULT_VERSION,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        初始化异步HTTP客户端
        
        Args:
            api_key: Anthropic API密钥
            base_url: API基础URL
            version: API版本
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key is required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 异步客户端延迟初始化
        self._client: Optional[httpx.AsyncClient] = None
    
    def _default_headers(self) -> Dict[str, str]:
        """获取默认请求头"""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "Content-Type": "application/json",
            "User-Agent": "anthropic-python/0.1.0",
        }
    
    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self._default_headers(),
            )
        return self._client
    
    def _handle_error(self, response: httpx.Response) -> None:
        """处理HTTP错误响应"""
        if response.status_code < 400:
            return
        
        try:
            body = response.json()
        except Exception:
            body = {"error": response.text}
        
        error_message = body.get("error", {}).get("message", "Unknown error")
        
        if response.status_code == 401:
            raise AuthenticationError(error_message, response, body)
        elif response.status_code == 400:
            raise InvalidRequestError(error_message, response, body)
        elif response.status_code == 404:
            raise NotFoundError(error_message, response, body)
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                error_message,
                retry_after=float(retry_after) if retry_after else None,
                response=response,
                body=body,
            )
        elif 500 <= response.status_code < 600:
            raise ServerError(
                error_message,
                status_code=response.status_code,
                response=response,
                body=body,
            )
        else:
            raise APIError(
                error_message,
                status_code=response.status_code,
                response=response,
                body=body,
            )
    
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """
        带重试机制的异步请求
        
        Args:
            method: HTTP方法
            path: 请求路径
            **kwargs: 其他请求参数
        
        Returns:
            HTTP响应对象
        """
        client = self._get_client()
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.request(method, path, **kwargs)
                self._handle_error(response)
                return response
                
            except (RateLimitError, ServerError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    if isinstance(e, RateLimitError) and e.retry_after:
                        await self._async_sleep(e.retry_after)
                    else:
                        await self._async_sleep(2 ** attempt)
                continue
                
            except httpx.TimeoutException as e:
                last_exception = TimeoutError(f"Request timeout: {e}")
                if attempt < self.max_retries - 1:
                    await self._async_sleep(2 ** attempt)
                continue
                
            except (AuthenticationError, InvalidRequestError, NotFoundError):
                raise
        
        if last_exception:
            raise last_exception
        
        raise APIError("Max retries exceeded")
    
    async def _async_sleep(self, seconds: float) -> None:
        """异步睡眠"""
        import asyncio
        await asyncio.sleep(seconds)
    
    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], httpx.Response]:
        """发送异步GET请求"""
        response = await self._request_with_retry("GET", path, params=params, **kwargs)
        
        if kwargs.get("stream"):
            return response
        
        return response.json()
    
    async def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], httpx.Response]:
        """发送异步POST请求"""
        response = await self._request_with_retry("POST", path, json=json, **kwargs)
        
        if kwargs.get("stream"):
            return response
        
        return response.json()
    
    async def close(self) -> None:
        """关闭异步客户端连接"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
        return False


class AnthropicClient:
    """
    Anthropic 同步客户端
    
    整合所有API功能的主客户端类
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        version: str = DEFAULT_VERSION,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        初始化Anthropic客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            version: API版本
            timeout: 请求超时
            max_retries: 最大重试次数
        """
        self._http = HTTPClient(
            api_key=api_key,
            base_url=base_url,
            version=version,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        # 初始化API子模块
        self.messages = Messages(self._http)
        self.completions = Completions(self._http)
    
    def close(self) -> None:
        """关闭客户端"""
        self._http.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False


class AsyncAnthropicClient:
    """
    Anthropic 异步客户端
    
    整合所有API功能的异步主客户端类
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        version: str = DEFAULT_VERSION,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        初始化异步Anthropic客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            version: API版本
            timeout: 请求超时
            max_retries: 最大重试次数
        """
        self._http = AsyncHTTPClient(
            api_key=api_key,
            base_url=base_url,
            version=version,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        # 初始化API子模块
        self.messages = AsyncMessages(self._http)
        self.completions = AsyncCompletions(self._http)
    
    async def close(self) -> None:
        """关闭异步客户端"""
        await self._http.close()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
        return False
