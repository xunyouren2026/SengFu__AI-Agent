"""
Chat Completions API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Chat Completions API
with support for streaming responses, function calling, and tool use.

Module path: compat/openai/chat_completions.py
"""

from __future__ import annotations

import json
import time
from typing import (
    Dict, List, Optional, Any, Union, Iterator, AsyncIterator,
    Callable, Literal, TypedDict, Generator
)
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class Role(str, Enum):
    """Message roles in chat conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reasons for completion finish."""
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


class ChatCompletionToolChoiceOption(str, Enum):
    """Tool choice options."""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class FunctionCall(TypedDict, total=False):
    """Function call definition."""
    name: str
    arguments: str


class ToolCall(TypedDict, total=False):
    """Tool call definition."""
    id: str
    type: Literal["function"]
    function: FunctionCall


class ChatCompletionMessage(TypedDict, total=False):
    """Chat completion message structure."""
    role: str
    content: Optional[str]
    name: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    tool_call_id: Optional[str]
    function_call: Optional[FunctionCall]


class ChatCompletionToolFunction(TypedDict):
    """Tool function definition."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ChatCompletionTool(TypedDict):
    """Chat completion tool definition."""
    type: Literal["function"]
    function: ChatCompletionToolFunction


@dataclass
class ChatCompletionRequest:
    """Chat completion request parameters."""
    model: str
    messages: List[ChatCompletionMessage]
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    stream_options: Optional[Dict[str, bool]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[ChatCompletionTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        result = asdict(self)
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class ChatCompletionTokenLogprob:
    """Token log probability information."""
    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[Dict[str, Any]]] = None


@dataclass
class ChatCompletionChoice:
    """Chat completion choice."""
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class ChatCompletionChunkChoice:
    """Chat completion streaming chunk choice."""
    index: int
    delta: ChatCompletionMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class ChatCompletionUsage:
    """Token usage information."""
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[Dict[str, Any]] = None
    prompt_tokens_details: Optional[Dict[str, Any]] = None


@dataclass
class ChatCompletion:
    """Chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None


@dataclass
class ChatCompletionChunk:
    """Chat completion streaming chunk."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    usage: Optional[ChatCompletionUsage] = None


class BaseChatCompletions:
    """Base class for chat completions."""
    
    def __init__(self, config: Any):
        self.config = config
        self._client: Optional[Any] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project
        headers.update(self.config.default_headers)
        return headers
    
    def _handle_error(self, response: Any) -> None:
        """Handle API error responses."""
        from . import (
            AuthenticationError, RateLimitError, APIError,
            BadRequestError, NotFoundError, APIConnectionError
        )
        
        status_code = response.status_code
        try:
            error_data = response.json()
            error = error_data.get("error", {})
            message = error.get("message", "Unknown error")
            code = error.get("code")
            param = error.get("param")
            error_type = error.get("type")
        except Exception:
            message = f"HTTP {status_code}: {response.text}"
            code = None
            param = None
            error_type = None
        
        if status_code == 401:
            raise AuthenticationError(message, code, param, error_type)
        elif status_code == 429:
            raise RateLimitError(message, code, param, error_type)
        elif status_code == 400:
            raise BadRequestError(message, code, param, error_type)
        elif status_code == 404:
            raise NotFoundError(message, code, param, error_type)
        elif status_code >= 500:
            raise APIConnectionError(message, code, param, error_type)
        else:
            raise APIError(message, code, param, error_type)


class ChatCompletions(BaseChatCompletions):
    """
    Synchronous Chat Completions API client.
    
    Example:
        >>> client = ChatCompletions(config)
        >>> response = client.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client = httpx.Client(
            timeout=config.timeout,
            verify=config.verify_ssl,
            proxies=config.proxies,
        )
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        from . import APIConnectionError
        
        delay = self.config.retry_delay
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    json=json_data,
                )
                if response.status_code >= 500 and attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay *= self.config.retry_backoff
                    continue
                return response
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay *= self.config.retry_backoff
                    continue
                raise APIConnectionError(f"Request failed after {self.config.max_retries} retries: {e}")
        
        raise APIConnectionError(f"Request failed: {last_error}")
    
    def create(
        self,
        model: str,
        messages: List[ChatCompletionMessage],
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        service_tier: Optional[str] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, bool]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[ChatCompletionTool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create a chat completion.
        
        Args:
            model: ID of the model to use
            messages: List of messages in the conversation
            frequency_penalty: Penalty for frequency of token usage
            logit_bias: Modify likelihood of specific tokens
            logprobs: Return log probabilities
            top_logprobs: Number of most likely tokens to return
            max_tokens: Maximum tokens in completion
            max_completion_tokens: Maximum completion tokens (o1 models)
            n: Number of completions to generate
            presence_penalty: Penalty for new tokens
            response_format: Output format (json_object, text)
            seed: Seed for deterministic sampling
            service_tier: Service tier preference
            stop: Stop sequences
            stream: Stream response
            stream_options: Streaming options
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tools: Available tools
            tool_choice: Tool choice strategy
            parallel_tool_calls: Allow parallel tool calls
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ChatCompletion or iterator of ChatCompletionChunk if streaming
        """
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            user=user,
        )
        
        url = f"{self.config.base_url}/chat/completions"
        data = request.to_dict()
        
        if stream:
            return self._stream_create(data)
        
        response = self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_completion(result)
    
    def _stream_create(
        self,
        data: Dict[str, Any]
    ) -> Iterator[ChatCompletionChunk]:
        """Create streaming chat completion."""
        url = f"{self.config.base_url}/chat/completions"
        
        with self._client.stream(
            "POST",
            url,
            headers=self._get_headers(),
            json=data,
        ) as response:
            if response.status_code != 200:
                self._handle_error(response)
            
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        yield self._parse_chunk(chunk_data)
                    except json.JSONDecodeError:
                        continue
    
    def _parse_completion(self, data: Dict[str, Any]) -> ChatCompletion:
        """Parse completion response."""
        choices = []
        for choice_data in data.get("choices", []):
            message_data = choice_data.get("message", {})
            message = ChatCompletionMessage(
                role=message_data.get("role", ""),
                content=message_data.get("content"),
                name=message_data.get("name"),
                tool_calls=message_data.get("tool_calls"),
                tool_call_id=message_data.get("tool_call_id"),
                function_call=message_data.get("function_call"),
            )
            choice = ChatCompletionChoice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)
        
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = ChatCompletionUsage(
                completion_tokens=usage_data.get("completion_tokens", 0),
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            )
        
        return ChatCompletion(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
        )
    
    def _parse_chunk(self, data: Dict[str, Any]) -> ChatCompletionChunk:
        """Parse streaming chunk."""
        choices = []
        for choice_data in data.get("choices", []):
            delta_data = choice_data.get("delta", {})
            delta = ChatCompletionMessage(
                role=delta_data.get("role", ""),
                content=delta_data.get("content"),
                name=delta_data.get("name"),
                tool_calls=delta_data.get("tool_calls"),
                tool_call_id=delta_data.get("tool_call_id"),
                function_call=delta_data.get("function_call"),
            )
            choice = ChatCompletionChunkChoice(
                index=choice_data.get("index", 0),
                delta=delta,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)
        
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = ChatCompletionUsage(
                completion_tokens=usage_data.get("completion_tokens", 0),
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            )
        
        return ChatCompletionChunk(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
            usage=usage,
        )
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class AsyncChatCompletions(BaseChatCompletions):
    """
    Asynchronous Chat Completions API client.
    
    Example:
        >>> client = AsyncChatCompletions(config)
        >>> response = await client.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client: Optional[httpx.AsyncClient] = None
        self._config = config
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout,
                verify=self._config.verify_ssl,
                proxies=self._config.proxies,
            )
        return self._client
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make async HTTP request with retry logic."""
        from . import APIConnectionError
        
        client = await self._get_client()
        delay = self.config.retry_delay
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    json=json_data,
                )
                if response.status_code >= 500 and attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff
                    continue
                return response
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff
                    continue
                raise APIConnectionError(f"Request failed after {self.config.max_retries} retries: {e}")
        
        raise APIConnectionError(f"Request failed: {last_error}")
    
    async def create(
        self,
        model: str,
        messages: List[ChatCompletionMessage],
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        service_tier: Optional[str] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        stream_options: Optional[Dict[str, bool]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[ChatCompletionTool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion asynchronously.
        
        Args:
            model: ID of the model to use
            messages: List of messages in the conversation
            frequency_penalty: Penalty for frequency of token usage
            logit_bias: Modify likelihood of specific tokens
            logprobs: Return log probabilities
            top_logprobs: Number of most likely tokens to return
            max_tokens: Maximum tokens in completion
            max_completion_tokens: Maximum completion tokens (o1 models)
            n: Number of completions to generate
            presence_penalty: Penalty for new tokens
            response_format: Output format (json_object, text)
            seed: Seed for deterministic sampling
            service_tier: Service tier preference
            stop: Stop sequences
            stream: Stream response
            stream_options: Streaming options
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tools: Available tools
            tool_choice: Tool choice strategy
            parallel_tool_calls: Allow parallel tool calls
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ChatCompletion or async iterator of ChatCompletionChunk if streaming
        """
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            user=user,
        )
        
        url = f"{self.config.base_url}/chat/completions"
        data = request.to_dict()
        
        if stream:
            return self._stream_create(data)
        
        response = await self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_completion(result)
    
    async def _stream_create(
        self,
        data: Dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Create async streaming chat completion."""
        client = await self._get_client()
        url = f"{self.config.base_url}/chat/completions"
        
        async with client.stream(
            "POST",
            url,
            headers=self._get_headers(),
            json=data,
        ) as response:
            if response.status_code != 200:
                self._handle_error(response)
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        yield self._parse_chunk(chunk_data)
                    except json.JSONDecodeError:
                        continue
    
    def _parse_completion(self, data: Dict[str, Any]) -> ChatCompletion:
        """Parse completion response."""
        choices = []
        for choice_data in data.get("choices", []):
            message_data = choice_data.get("message", {})
            message = ChatCompletionMessage(
                role=message_data.get("role", ""),
                content=message_data.get("content"),
                name=message_data.get("name"),
                tool_calls=message_data.get("tool_calls"),
                tool_call_id=message_data.get("tool_call_id"),
                function_call=message_data.get("function_call"),
            )
            choice = ChatCompletionChoice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)
        
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = ChatCompletionUsage(
                completion_tokens=usage_data.get("completion_tokens", 0),
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            )
        
        return ChatCompletion(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
        )
    
    def _parse_chunk(self, data: Dict[str, Any]) -> ChatCompletionChunk:
        """Parse streaming chunk."""
        choices = []
        for choice_data in data.get("choices", []):
            delta_data = choice_data.get("delta", {})
            delta = ChatCompletionMessage(
                role=delta_data.get("role", ""),
                content=delta_data.get("content"),
                name=delta_data.get("name"),
                tool_calls=delta_data.get("tool_calls"),
                tool_call_id=delta_data.get("tool_call_id"),
                function_call=delta_data.get("function_call"),
            )
            choice = ChatCompletionChunkChoice(
                index=choice_data.get("index", 0),
                delta=delta,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)
        
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = ChatCompletionUsage(
                completion_tokens=usage_data.get("completion_tokens", 0),
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            )
        
        return ChatCompletionChunk(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
            usage=usage,
        )
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
