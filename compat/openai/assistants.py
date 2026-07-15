"""
Assistants API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Assistants API
for building AI assistants with custom instructions and tools.

Module path: compat/openai/assistants.py
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


class AssistantToolType(str, Enum):
    """Types of assistant tools."""
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"
    FUNCTION = "function"


class AssistantResponseFormat(str, Enum):
    """Response format options."""
    AUTO = "auto"
    JSON = "json"
    TEXT = "text"


@dataclass
class AssistantTool:
    """Assistant tool definition."""
    type: str
    function: Optional[Dict[str, Any]] = None


@dataclass
class AssistantToolResources:
    """Resources used by assistant tools."""
    code_interpreter: Optional[Dict[str, List[str]]] = None
    file_search: Optional[Dict[str, Any]] = None


@dataclass
class Assistant:
    """Assistant object."""
    id: str
    object: str
    created_at: int
    name: Optional[str]
    description: Optional[str]
    model: str
    instructions: Optional[str]
    tools: List[AssistantTool]
    tool_resources: Optional[AssistantToolResources]
    metadata: Optional[Dict[str, str]]
    top_p: Optional[float]
    temperature: Optional[float]
    response_format: Optional[Union[str, Dict[str, str]]]


@dataclass
class AssistantDeleted:
    """Assistant deletion confirmation."""
    id: str
    object: str
    deleted: bool


@dataclass
class AssistantList:
    """List of assistants."""
    object: str
    data: List[Assistant]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool


class BaseAssistants:
    """Base class for assistants API."""
    
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
    
    def _parse_assistant(self, data: Dict[str, Any]) -> Assistant:
        """Parse assistant from API response."""
        tools = []
        for tool_data in data.get("tools", []):
            tools.append(AssistantTool(
                type=tool_data.get("type", ""),
                function=tool_data.get("function"),
            ))
        
        tool_resources = None
        if "tool_resources" in data and data["tool_resources"]:
            tr_data = data["tool_resources"]
            tool_resources = AssistantToolResources(
                code_interpreter=tr_data.get("code_interpreter"),
                file_search=tr_data.get("file_search"),
            )
        
        return Assistant(
            id=data.get("id", ""),
            object=data.get("object", "assistant"),
            created_at=data.get("created_at", 0),
            name=data.get("name"),
            description=data.get("description"),
            model=data.get("model", ""),
            instructions=data.get("instructions"),
            tools=tools,
            tool_resources=tool_resources,
            metadata=data.get("metadata"),
            top_p=data.get("top_p"),
            temperature=data.get("temperature"),
            response_format=data.get("response_format"),
        )


class Assistants(BaseAssistants):
    """
    Synchronous Assistants API client.
    
    Example:
        >>> client = Assistants(config)
        >>> assistant = client.create(
        ...     model="gpt-4",
        ...     name="Math Tutor",
        ...     instructions="You are a helpful math tutor.",
        ...     tools=[{"type": "code_interpreter"}]
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
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Assistant:
        """
        Create a new assistant.
        
        Args:
            model: Model ID to use
            name: Assistant name
            description: Assistant description
            instructions: System instructions
            tools: List of tools (code_interpreter, file_search, function)
            tool_resources: Resources for tools
            metadata: Key-value metadata
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            response_format: Response format
            **kwargs: Additional parameters
            
        Returns:
            Created Assistant object
        """
        url = f"{self.config.base_url}/assistants"
        
        data: Dict[str, Any] = {"model": model}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
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
        if response_format is not None:
            data["response_format"] = response_format
        
        response = self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_assistant(result)
    
    def retrieve(self, assistant_id: str, **kwargs: Any) -> Assistant:
        """
        Retrieve an assistant by ID.
        
        Args:
            assistant_id: ID of the assistant
            **kwargs: Additional parameters
            
        Returns:
            Assistant object
        """
        url = f"{self.config.base_url}/assistants/{assistant_id}"
        
        response = self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_assistant(result)
    
    def update(
        self,
        assistant_id: str,
        model: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Assistant:
        """
        Update an assistant.
        
        Args:
            assistant_id: ID of the assistant to update
            model: New model ID
            name: New name
            description: New description
            instructions: New instructions
            tools: New tools list
            tool_resources: New tool resources
            metadata: New metadata
            temperature: New temperature
            top_p: New top_p
            response_format: New response format
            **kwargs: Additional parameters
            
        Returns:
            Updated Assistant object
        """
        url = f"{self.config.base_url}/assistants/{assistant_id}"
        
        data: Dict[str, Any] = {}
        if model is not None:
            data["model"] = model
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
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
        if response_format is not None:
            data["response_format"] = response_format
        
        response = self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_assistant(result)
    
    def delete(self, assistant_id: str, **kwargs: Any) -> AssistantDeleted:
        """
        Delete an assistant.
        
        Args:
            assistant_id: ID of the assistant to delete
            **kwargs: Additional parameters
            
        Returns:
            Deletion confirmation
        """
        url = f"{self.config.base_url}/assistants/{assistant_id}"
        
        response = self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return AssistantDeleted(
            id=result.get("id", ""),
            object=result.get("object", "assistant.deleted"),
            deleted=result.get("deleted", True),
        )
    
    def list(
        self,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        **kwargs: Any
    ) -> AssistantList:
        """
        List assistants.
        
        Args:
            limit: Number of assistants to return (1-100)
            order: Sort order ("asc" or "desc")
            after: Cursor for pagination
            before: Cursor for pagination
            **kwargs: Additional parameters
            
        Returns:
            List of assistants
        """
        url = f"{self.config.base_url}/assistants"
        
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        # Build query string
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = self._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        assistants = [self._parse_assistant(item) for item in result.get("data", [])]
        
        return AssistantList(
            object=result.get("object", "list"),
            data=assistants,
            first_id=result.get("first_id"),
            last_id=result.get("last_id"),
            has_more=result.get("has_more", False),
        )
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class AsyncAssistants(BaseAssistants):
    """
    Asynchronous Assistants API client.
    
    Example:
        >>> client = AsyncAssistants(config)
        >>> assistant = await client.create(
        ...     model="gpt-4",
        ...     name="Math Tutor",
        ...     instructions="You are a helpful math tutor."
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
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Assistant:
        """Create a new assistant asynchronously."""
        url = f"{self.config.base_url}/assistants"
        
        data: Dict[str, Any] = {"model": model}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
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
        if response_format is not None:
            data["response_format"] = response_format
        
        response = await self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_assistant(result)
    
    async def retrieve(self, assistant_id: str, **kwargs: Any) -> Assistant:
        """Retrieve an assistant by ID asynchronously."""
        url = f"{self.config.base_url}/assistants/{assistant_id}"
        
        response = await self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_assistant(result)
    
    async def update(
        self,
        assistant_id: str,
        model: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs: Any
    ) -> Assistant:
        """Update an assistant asynchronously."""
        url = f"{self.config.base_url}/assistants/{assistant_id}"
        
        data: Dict[str, Any] = {}
        if model is not None:
            data["model"] = model
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
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
        if response_format is not None:
            data["response_format"] = response_format
        
        response = await self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_assistant(result)
    
    async def delete(self, assistant_id: str, **kwargs: Any) -> AssistantDeleted:
        """Delete an assistant asynchronously."""
        url = f"{self.config.base_url}/assistants/{assistant_id}"
        
        response = await self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return AssistantDeleted(
            id=result.get("id", ""),
            object=result.get("object", "assistant.deleted"),
            deleted=result.get("deleted", True),
        )
    
    async def list(
        self,
        limit: int = 20,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        **kwargs: Any
    ) -> AssistantList:
        """List assistants asynchronously."""
        url = f"{self.config.base_url}/assistants"
        
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = await self._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        assistants = [self._parse_assistant(item) for item in result.get("data", [])]
        
        return AssistantList(
            object=result.get("object", "list"),
            data=assistants,
            first_id=result.get("first_id"),
            last_id=result.get("last_id"),
            has_more=result.get("has_more", False),
        )
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
