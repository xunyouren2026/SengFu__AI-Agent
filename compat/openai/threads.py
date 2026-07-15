"""
Threads API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Threads API
for managing conversation threads with assistants.

Module path: compat/openai/threads.py
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Union, Iterator, AsyncIterator
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class MessageRole(str, Enum):
    """Message roles in threads."""
    USER = "user"
    ASSISTANT = "assistant"


class RunStatus(str, Enum):
    """Run status values."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    REQUIRES_ACTION = "requires_action"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class Thread:
    """Thread object."""
    id: str
    object: str
    created_at: int
    metadata: Optional[Dict[str, str]]
    tool_resources: Optional[Dict[str, Any]]


@dataclass
class ThreadMessage:
    """Message in a thread."""
    id: str
    object: str
    created_at: int
    thread_id: str
    role: str
    content: List[Dict[str, Any]]
    assistant_id: Optional[str]
    run_id: Optional[str]
    attachments: Optional[List[Dict[str, Any]]]
    metadata: Optional[Dict[str, str]]


@dataclass
class ThreadRun:
    """Run object for executing assistant on thread."""
    id: str
    object: str
    created_at: int
    assistant_id: str
    thread_id: str
    status: str
    started_at: Optional[int]
    expires_at: Optional[int]
    cancelled_at: Optional[int]
    failed_at: Optional[int]
    completed_at: Optional[int]
    last_error: Optional[Dict[str, Any]]
    model: str
    instructions: Optional[str]
    tools: List[Dict[str, Any]]
    metadata: Optional[Dict[str, str]]
    usage: Optional[Dict[str, int]]
    temperature: Optional[float]
    top_p: Optional[float]
    max_prompt_tokens: Optional[int]
    max_completion_tokens: Optional[int]
    truncation_strategy: Optional[Dict[str, Any]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]
    parallel_tool_calls: bool
    response_format: Optional[Union[str, Dict[str, str]]]


@dataclass
class ThreadRunStep:
    """Step in a run."""
    id: str
    object: str
    created_at: int
    run_id: str
    assistant_id: str
    thread_id: str
    type: str
    status: str
    cancelled_at: Optional[int]
    completed_at: Optional[int]
    expired_at: Optional[int]
    failed_at: Optional[int]
    last_error: Optional[Dict[str, Any]]
    step_details: Dict[str, Any]
    usage: Optional[Dict[str, int]]


@dataclass
class ThreadDeleted:
    """Thread deletion confirmation."""
    id: str
    object: str
    deleted: bool


@dataclass
class ThreadList:
    """List of threads (for pagination)."""
    object: str
    data: List[Thread]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool


@dataclass
class MessageList:
    """List of messages."""
    object: str
    data: List[ThreadMessage]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool


@dataclass
class RunList:
    """List of runs."""
    object: str
    data: List[ThreadRun]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool


@dataclass
class RunStepList:
    """List of run steps."""
    object: str
    data: List[ThreadRunStep]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool


class BaseThreads:
    """Base class for threads API."""
    
    def __init__(self, config: Any):
        self.config = config
        self._client: Optional[Any] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
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
    
    def _parse_thread(self, data: Dict[str, Any]) -> Thread:
        """Parse thread from API response."""
        return Thread(
            id=data.get("id", ""),
            object=data.get("object", "thread"),
            created_at=data.get("created_at", 0),
            metadata=data.get("metadata"),
            tool_resources=data.get("tool_resources"),
        )
    
    def _parse_message(self, data: Dict[str, Any]) -> ThreadMessage:
        """Parse message from API response."""
        return ThreadMessage(
            id=data.get("id", ""),
            object=data.get("object", "thread.message"),
            created_at=data.get("created_at", 0),
            thread_id=data.get("thread_id", ""),
            role=data.get("role", ""),
            content=data.get("content", []),
            assistant_id=data.get("assistant_id"),
            run_id=data.get("run_id"),
            attachments=data.get("attachments"),
            metadata=data.get("metadata"),
        )
    
    def _parse_run(self, data: Dict[str, Any]) -> ThreadRun:
        """Parse run from API response."""
        return ThreadRun(
            id=data.get("id", ""),
            object=data.get("object", "thread.run"),
            created_at=data.get("created_at", 0),
            assistant_id=data.get("assistant_id", ""),
            thread_id=data.get("thread_id", ""),
            status=data.get("status", ""),
            started_at=data.get("started_at"),
            expires_at=data.get("expires_at"),
            cancelled_at=data.get("cancelled_at"),
            failed_at=data.get("failed_at"),
            completed_at=data.get("completed_at"),
            last_error=data.get("last_error"),
            model=data.get("model", ""),
            instructions=data.get("instructions"),
            tools=data.get("tools", []),
            metadata=data.get("metadata"),
            usage=data.get("usage"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            max_prompt_tokens=data.get("max_prompt_tokens"),
            max_completion_tokens=data.get("max_completion_tokens"),
            truncation_strategy=data.get("truncation_strategy"),
            tool_choice=data.get("tool_choice"),
            parallel_tool_calls=data.get("parallel_tool_calls", True),
            response_format=data.get("response_format"),
        )
    
    def _parse_run_step(self, data: Dict[str, Any]) -> ThreadRunStep:
        """Parse run step from API response."""
        return ThreadRunStep(
            id=data.get("id", ""),
            object=data.get("object", "thread.run.step"),
            created_at=data.get("created_at", 0),
            run_id=data.get("run_id", ""),
            assistant_id=data.get("assistant_id", ""),
            thread_id=data.get("thread_id", ""),
            type=data.get("type", ""),
            status=data.get("status", ""),
            cancelled_at=data.get("cancelled_at"),
            completed_at=data.get("completed_at"),
            expired_at=data.get("expired_at"),
            failed_at=data.get("failed_at"),
            last_error=data.get("last_error"),
            step_details=data.get("step_details", {}),
            usage=data.get("usage"),
        )


class Threads(BaseThreads):
    """
    Synchronous Threads API client.
    
    Example:
        >>> client = Threads(config)
        >>> thread = client.create()
        >>> message = client.messages.create(
        ...     thread_id=thread.id,
        ...     role="user",
        ...     content="Hello!"
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
        self._messages = Messages(self)
        self._runs = Runs(self)
    
    @property
    def messages(self) -> Messages:
        """Access thread messages API."""
        return self._messages
    
    @property
    def runs(self) -> Runs:
        """Access thread runs API."""
        return self._runs
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
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
        messages: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Thread:
        """
        Create a new thread.
        
        Args:
            messages: Initial messages for the thread
            metadata: Key-value metadata
            tool_resources: Resources for tools
            **kwargs: Additional parameters
            
        Returns:
            Created Thread object
        """
        url = f"{self.config.base_url}/threads"
        
        data: Dict[str, Any] = {}
        if messages is not None:
            data["messages"] = messages
        if metadata is not None:
            data["metadata"] = metadata
        if tool_resources is not None:
            data["tool_resources"] = tool_resources
        
        response = self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_thread(result)
    
    def retrieve(self, thread_id: str, **kwargs: Any) -> Thread:
        """
        Retrieve a thread by ID.
        
        Args:
            thread_id: ID of the thread
            **kwargs: Additional parameters
            
        Returns:
            Thread object
        """
        url = f"{self.config.base_url}/threads/{thread_id}"
        
        response = self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_thread(result)
    
    def update(
        self,
        thread_id: str,
        metadata: Optional[Dict[str, str]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Thread:
        """
        Update a thread.
        
        Args:
            thread_id: ID of the thread to update
            metadata: New metadata
            tool_resources: New tool resources
            **kwargs: Additional parameters
            
        Returns:
            Updated Thread object
        """
        url = f"{self.config.base_url}/threads/{thread_id}"
        
        data: Dict[str, Any] = {}
        if metadata is not None:
            data["metadata"] = metadata
        if tool_resources is not None:
            data["tool_resources"] = tool_resources
        
        response = self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_thread(result)
    
    def delete(self, thread_id: str, **kwargs: Any) -> ThreadDeleted:
        """
        Delete a thread.
        
        Args:
            thread_id: ID of the thread to delete
            **kwargs: Additional parameters
            
        Returns:
            Deletion confirmation
        """
        url = f"{self.config.base_url}/threads/{thread_id}"
        
        response = self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return ThreadDeleted(
            id=result.get("id", ""),
            object=result.get("object", "thread.deleted"),
            deleted=result.get("deleted", True),
        )
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class Messages:
    """Thread messages API."""
    
    def __init__(self, threads: Threads):
        self._threads = threads
    
    def create(
        self,
        thread_id: str,
        role: str,
        content: Union[str, List[Dict[str, Any]]],
        attachments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ThreadMessage:
        """
        Create a message in a thread.
        
        Args:
            thread_id: ID of the thread
            role: Message role ("user" or "assistant")
            content: Message content
            attachments: File attachments
            metadata: Key-value metadata
            **kwargs: Additional parameters
            
        Returns:
            Created ThreadMessage object
        """
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages"
        
        data: Dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if attachments is not None:
            data["attachments"] = attachments
        if metadata is not None:
            data["metadata"] = metadata
        
        response = self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_message(result)
    
    def retrieve(self, thread_id: str, message_id: str, **kwargs: Any) -> ThreadMessage:
        """
        Retrieve a message by ID.
        
        Args:
            thread_id: ID of the thread
            message_id: ID of the message
            **kwargs: Additional parameters
            
        Returns:
            ThreadMessage object
        """
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages/{message_id}"
        
        response = self._threads._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_message(result)
    
    def update(
        self,
        thread_id: str,
        message_id: str,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ThreadMessage:
        """
        Update a message.
        
        Args:
            thread_id: ID of the thread
            message_id: ID of the message
            metadata: New metadata
            **kwargs: Additional parameters
            
        Returns:
            Updated ThreadMessage object
        """
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages/{message_id}"
        
        data: Dict[str, Any] = {}
        if metadata is not None:
            data["metadata"] = metadata
        
        response = self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_message(result)
    
    def list(
        self,
        thread_id: str,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        **kwargs: Any
    ) -> MessageList:
        """
        List messages in a thread.
        
        Args:
            thread_id: ID of the thread
            limit: Number of messages to return
            order: Sort order ("asc" or "desc")
            after: Cursor for pagination
            before: Cursor for pagination
            **kwargs: Additional parameters
            
        Returns:
            List of messages
        """
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages"
        
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = self._threads._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        messages = [self._threads._parse_message(item) for item in result.get("data", [])]
        
        return MessageList(
            object=result.get("object", "list"),
            data=messages,
            first_id=result.get("first_id"),
            last_id=result.get("last_id"),
            has_more=result.get("has_more", False),
        )
    
    def delete(self, thread_id: str, message_id: str, **kwargs: Any) -> ThreadDeleted:
        """
        Delete a message.
        
        Args:
            thread_id: ID of the thread
            message_id: ID of the message
            **kwargs: Additional parameters
            
        Returns:
            Deletion confirmation
        """
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages/{message_id}"
        
        response = self._threads._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return ThreadDeleted(
            id=result.get("id", ""),
            object=result.get("object", "thread.message.deleted"),
            deleted=result.get("deleted", True),
        )


class Runs:
    """Thread runs API."""
    
    def __init__(self, threads: Threads):
        self._threads = threads
    
    def create(
        self,
        thread_id: str,
        assistant_id: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        additional_messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_prompt_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        truncation_strategy: Optional[Dict[str, Any]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[ThreadRun, Iterator[Dict[str, Any]]]:
        """
        Create a run for a thread.
        
        Args:
            thread_id: ID of the thread
            assistant_id: ID of the assistant
            model: Override model to use
            instructions: Override instructions
            additional_instructions: Additional instructions
            additional_messages: Additional messages
            tools: Override tools
            metadata: Key-value metadata
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_prompt_tokens: Max prompt tokens
            max_completion_tokens: Max completion tokens
            truncation_strategy: Truncation strategy
            tool_choice: Tool choice strategy
            parallel_tool_calls: Allow parallel tool calls
            response_format: Response format
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            ThreadRun object or stream iterator
        """
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs"
        
        data: Dict[str, Any] = {"assistant_id": assistant_id}
        if model is not None:
            data["model"] = model
        if instructions is not None:
            data["instructions"] = instructions
        if additional_instructions is not None:
            data["additional_instructions"] = additional_instructions
        if additional_messages is not None:
            data["additional_messages"] = additional_messages
        if tools is not None:
            data["tools"] = tools
        if metadata is not None:
            data["metadata"] = metadata
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_prompt_tokens is not None:
            data["max_prompt_tokens"] = max_prompt_tokens
        if max_completion_tokens is not None:
            data["max_completion_tokens"] = max_completion_tokens
        if truncation_strategy is not None:
            data["truncation_strategy"] = truncation_strategy
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            data["parallel_tool_calls"] = parallel_tool_calls
        if response_format is not None:
            data["response_format"] = response_format
        if stream:
            data["stream"] = stream
        
        if stream:
            return self._stream_run(url, data)
        
        response = self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    def _stream_run(
        self,
        url: str,
        data: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """Stream run events."""
        with self._threads._client.stream(
            "POST",
            url,
            headers=self._threads._get_headers(),
            json=data,
        ) as response:
            if response.status_code != 200:
                self._threads._handle_error(response)
            
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event_data = json.loads(data_str)
                        yield event_data
                    except json.JSONDecodeError:
                        continue
    
    def retrieve(self, thread_id: str, run_id: str, **kwargs: Any) -> ThreadRun:
        """Retrieve a run by ID."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}"
        
        response = self._threads._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    def update(
        self,
        thread_id: str,
        run_id: str,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ThreadRun:
        """Update a run."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}"
        
        data: Dict[str, Any] = {}
        if metadata is not None:
            data["metadata"] = metadata
        
        response = self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    def list(
        self,
        thread_id: str,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        **kwargs: Any
    ) -> RunList:
        """List runs for a thread."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs"
        
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = self._threads._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        runs = [self._threads._parse_run(item) for item in result.get("data", [])]
        
        return RunList(
            object=result.get("object", "list"),
            data=runs,
            first_id=result.get("first_id"),
            last_id=result.get("last_id"),
            has_more=result.get("has_more", False),
        )
    
    def cancel(self, thread_id: str, run_id: str, **kwargs: Any) -> ThreadRun:
        """Cancel a run."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}/cancel"
        
        response = self._threads._request_with_retry("POST", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    def submit_tool_outputs(
        self,
        thread_id: str,
        run_id: str,
        tool_outputs: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[ThreadRun, Iterator[Dict[str, Any]]]:
        """Submit tool outputs for a run."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}/submit_tool_outputs"
        
        data: Dict[str, Any] = {"tool_outputs": tool_outputs}
        if stream:
            data["stream"] = stream
        
        if stream:
            return self._stream_run(url, data)
        
        response = self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    def create_thread_and_run(
        self,
        assistant_id: str,
        thread: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_prompt_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        truncation_strategy: Optional[Dict[str, Any]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs: Any
    ) -> ThreadRun:
        """Create a thread and run it in one request."""
        url = f"{self._threads.config.base_url}/threads/runs"
        
        data: Dict[str, Any] = {"assistant_id": assistant_id}
        if thread is not None:
            data["thread"] = thread
        if model is not None:
            data["model"] = model
        if instructions is not None:
            data["instructions"] = instructions
        if tools is not None:
            data["tools"] = tools
        if tool_resources is not None:
            data["tool_resources"] = tool_resources
        if metadata is not None:
            data["metadata"] = metadata
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_prompt_tokens is not None:
            data["max_prompt_tokens"] = max_prompt_tokens
        if max_completion_tokens is not None:
            data["max_completion_tokens"] = max_completion_tokens
        if truncation_strategy is not None:
            data["truncation_strategy"] = truncation_strategy
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            data["parallel_tool_calls"] = parallel_tool_calls
        if response_format is not None:
            data["response_format"] = response_format
        
        response = self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)


class AsyncThreads(BaseThreads):
    """Asynchronous Threads API client."""
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client: Optional[httpx.AsyncClient] = None
        self._config = config
        self._messages = AsyncMessages(self)
        self._runs = AsyncRuns(self)
    
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
    
    @property
    def messages(self) -> AsyncMessages:
        """Access async thread messages API."""
        return self._messages
    
    @property
    def runs(self) -> AsyncRuns:
        """Access async thread runs API."""
        return self._runs
    
    async def create(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Thread:
        """Create a new thread asynchronously."""
        url = f"{self.config.base_url}/threads"
        
        data: Dict[str, Any] = {}
        if messages is not None:
            data["messages"] = messages
        if metadata is not None:
            data["metadata"] = metadata
        if tool_resources is not None:
            data["tool_resources"] = tool_resources
        
        response = await self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_thread(result)
    
    async def retrieve(self, thread_id: str, **kwargs: Any) -> Thread:
        """Retrieve a thread by ID asynchronously."""
        url = f"{self.config.base_url}/threads/{thread_id}"
        
        response = await self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_thread(result)
    
    async def update(
        self,
        thread_id: str,
        metadata: Optional[Dict[str, str]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Thread:
        """Update a thread asynchronously."""
        url = f"{self.config.base_url}/threads/{thread_id}"
        
        data: Dict[str, Any] = {}
        if metadata is not None:
            data["metadata"] = metadata
        if tool_resources is not None:
            data["tool_resources"] = tool_resources
        
        response = await self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_thread(result)
    
    async def delete(self, thread_id: str, **kwargs: Any) -> ThreadDeleted:
        """Delete a thread asynchronously."""
        url = f"{self.config.base_url}/threads/{thread_id}"
        
        response = await self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return ThreadDeleted(
            id=result.get("id", ""),
            object=result.get("object", "thread.deleted"),
            deleted=result.get("deleted", True),
        )
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class AsyncMessages:
    """Async thread messages API."""
    
    def __init__(self, threads: AsyncThreads):
        self._threads = threads
    
    async def create(
        self,
        thread_id: str,
        role: str,
        content: Union[str, List[Dict[str, Any]]],
        attachments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ThreadMessage:
        """Create a message in a thread asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages"
        
        data: Dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if attachments is not None:
            data["attachments"] = attachments
        if metadata is not None:
            data["metadata"] = metadata
        
        response = await self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_message(result)
    
    async def retrieve(self, thread_id: str, message_id: str, **kwargs: Any) -> ThreadMessage:
        """Retrieve a message by ID asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages/{message_id}"
        
        response = await self._threads._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_message(result)
    
    async def update(
        self,
        thread_id: str,
        message_id: str,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ThreadMessage:
        """Update a message asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages/{message_id}"
        
        data: Dict[str, Any] = {}
        if metadata is not None:
            data["metadata"] = metadata
        
        response = await self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_message(result)
    
    async def list(
        self,
        thread_id: str,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        **kwargs: Any
    ) -> MessageList:
        """List messages in a thread asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages"
        
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = await self._threads._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        messages = [self._threads._parse_message(item) for item in result.get("data", [])]
        
        return MessageList(
            object=result.get("object", "list"),
            data=messages,
            first_id=result.get("first_id"),
            last_id=result.get("last_id"),
            has_more=result.get("has_more", False),
        )
    
    async def delete(self, thread_id: str, message_id: str, **kwargs: Any) -> ThreadDeleted:
        """Delete a message asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/messages/{message_id}"
        
        response = await self._threads._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return ThreadDeleted(
            id=result.get("id", ""),
            object=result.get("object", "thread.message.deleted"),
            deleted=result.get("deleted", True),
        )


class AsyncRuns:
    """Async thread runs API."""
    
    def __init__(self, threads: AsyncThreads):
        self._threads = threads
    
    async def create(
        self,
        thread_id: str,
        assistant_id: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        additional_messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_prompt_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        truncation_strategy: Optional[Dict[str, Any]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[ThreadRun, AsyncIterator[Dict[str, Any]]]:
        """Create a run for a thread asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs"
        
        data: Dict[str, Any] = {"assistant_id": assistant_id}
        if model is not None:
            data["model"] = model
        if instructions is not None:
            data["instructions"] = instructions
        if additional_instructions is not None:
            data["additional_instructions"] = additional_instructions
        if additional_messages is not None:
            data["additional_messages"] = additional_messages
        if tools is not None:
            data["tools"] = tools
        if metadata is not None:
            data["metadata"] = metadata
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_prompt_tokens is not None:
            data["max_prompt_tokens"] = max_prompt_tokens
        if max_completion_tokens is not None:
            data["max_completion_tokens"] = max_completion_tokens
        if truncation_strategy is not None:
            data["truncation_strategy"] = truncation_strategy
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            data["parallel_tool_calls"] = parallel_tool_calls
        if response_format is not None:
            data["response_format"] = response_format
        if stream:
            data["stream"] = stream
        
        if stream:
            return self._stream_run(url, data)
        
        response = await self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    async def _stream_run(
        self,
        url: str,
        data: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream run events asynchronously."""
        client = await self._threads._get_client()
        async with client.stream(
            "POST",
            url,
            headers=self._threads._get_headers(),
            json=data,
        ) as response:
            if response.status_code != 200:
                self._threads._handle_error(response)
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event_data = json.loads(data_str)
                        yield event_data
                    except json.JSONDecodeError:
                        continue
    
    async def retrieve(self, thread_id: str, run_id: str, **kwargs: Any) -> ThreadRun:
        """Retrieve a run by ID asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}"
        
        response = await self._threads._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    async def update(
        self,
        thread_id: str,
        run_id: str,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ThreadRun:
        """Update a run asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}"
        
        data: Dict[str, Any] = {}
        if metadata is not None:
            data["metadata"] = metadata
        
        response = await self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    async def list(
        self,
        thread_id: str,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        **kwargs: Any
    ) -> RunList:
        """List runs for a thread asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs"
        
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = await self._threads._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        runs = [self._threads._parse_run(item) for item in result.get("data", [])]
        
        return RunList(
            object=result.get("object", "list"),
            data=runs,
            first_id=result.get("first_id"),
            last_id=result.get("last_id"),
            has_more=result.get("has_more", False),
        )
    
    async def cancel(self, thread_id: str, run_id: str, **kwargs: Any) -> ThreadRun:
        """Cancel a run asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}/cancel"
        
        response = await self._threads._request_with_retry("POST", url)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    async def submit_tool_outputs(
        self,
        thread_id: str,
        run_id: str,
        tool_outputs: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[ThreadRun, AsyncIterator[Dict[str, Any]]]:
        """Submit tool outputs for a run asynchronously."""
        url = f"{self._threads.config.base_url}/threads/{thread_id}/runs/{run_id}/submit_tool_outputs"
        
        data: Dict[str, Any] = {"tool_outputs": tool_outputs}
        if stream:
            data["stream"] = stream
        
        if stream:
            return self._stream_run(url, data)
        
        response = await self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
    
    async def create_thread_and_run(
        self,
        assistant_id: str,
        thread: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_prompt_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        truncation_strategy: Optional[Dict[str, Any]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs: Any
    ) -> ThreadRun:
        """Create a thread and run it asynchronously."""
        url = f"{self._threads.config.base_url}/threads/runs"
        
        data: Dict[str, Any] = {"assistant_id": assistant_id}
        if thread is not None:
            data["thread"] = thread
        if model is not None:
            data["model"] = model
        if instructions is not None:
            data["instructions"] = instructions
        if tools is not None:
            data["tools"] = tools
        if tool_resources is not None:
            data["tool_resources"] = tool_resources
        if metadata is not None:
            data["metadata"] = metadata
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_prompt_tokens is not None:
            data["max_prompt_tokens"] = max_prompt_tokens
        if max_completion_tokens is not None:
            data["max_completion_tokens"] = max_completion_tokens
        if truncation_strategy is not None:
            data["truncation_strategy"] = truncation_strategy
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            data["parallel_tool_calls"] = parallel_tool_calls
        if response_format is not None:
            data["response_format"] = response_format
        
        response = await self._threads._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._threads._handle_error(response)
        
        result = response.json()
        return self._threads._parse_run(result)
