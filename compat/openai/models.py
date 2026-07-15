"""
Models API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Models API
for listing, retrieving, and deleting models.

Module path: compat/openai/models.py
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Union, Iterator, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class Model:
    """Model object."""
    id: str
    object: str
    created: int
    owned_by: str


@dataclass
class ModelDeleted:
    """Model deletion confirmation."""
    id: str
    object: str
    deleted: bool


@dataclass
class ModelList:
    """List of models."""
    object: str
    data: List[Model]


@dataclass
class Permission:
    """Model permission object."""
    id: str
    object: str
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: Optional[str]
    is_blocking: bool


class BaseModels:
    """Base class for models API."""
    
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
    
    def _parse_model(self, data: Dict[str, Any]) -> Model:
        """Parse model from API response."""
        return Model(
            id=data.get("id", ""),
            object=data.get("object", "model"),
            created=data.get("created", 0),
            owned_by=data.get("owned_by", ""),
        )


class Models(BaseModels):
    """
    Synchronous Models API client.
    
    Example:
        >>> client = Models(config)
        >>> models = client.list()
        >>> gpt4 = client.retrieve("gpt-4")
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
    
    def list(self, **kwargs: Any) -> ModelList:
        """
        List available models.
        
        Returns a list of models available for use, including both
        OpenAI models and fine-tuned models owned by the user.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            List of Model objects
            
        Example:
            >>> models = client.list()
            >>> for model in models.data:
            ...     print(model.id)
        """
        url = f"{self.config.base_url}/models"
        
        response = self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        models = [self._parse_model(item) for item in result.get("data", [])]
        
        return ModelList(
            object=result.get("object", "list"),
            data=models,
        )
    
    def retrieve(self, model: str, **kwargs: Any) -> Model:
        """
        Retrieve a model by ID.
        
        Get details about a specific model, including its capabilities
        and ownership information.
        
        Args:
            model: Model ID to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Model object
            
        Example:
            >>> gpt4 = client.retrieve("gpt-4")
            >>> print(gpt4.owned_by)
        """
        url = f"{self.config.base_url}/models/{model}"
        
        response = self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_model(result)
    
    def delete(self, model: str, **kwargs: Any) -> ModelDeleted:
        """
        Delete a fine-tuned model.
        
        Delete a fine-tuned model. This action cannot be undone.
        Only fine-tuned models can be deleted; base models cannot
        be deleted.
        
        Args:
            model: Model ID to delete
            **kwargs: Additional parameters
            
        Returns:
            Deletion confirmation
            
        Example:
            >>> result = client.delete("ft:gpt-3.5-turbo:my-org:custom:id")
            >>> print(result.deleted)
        """
        url = f"{self.config.base_url}/models/{model}"
        
        response = self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return ModelDeleted(
            id=result.get("id", ""),
            object=result.get("object", "model"),
            deleted=result.get("deleted", True),
        )
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class AsyncModels(BaseModels):
    """
    Asynchronous Models API client.
    
    Example:
        >>> client = AsyncModels(config)
        >>> models = await client.list()
        >>> gpt4 = await client.retrieve("gpt-4")
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
    
    async def list(self, **kwargs: Any) -> ModelList:
        """
        List available models asynchronously.
        
        Returns a list of models available for use, including both
        OpenAI models and fine-tuned models owned by the user.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            List of Model objects
        """
        url = f"{self.config.base_url}/models"
        
        response = await self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        models = [self._parse_model(item) for item in result.get("data", [])]
        
        return ModelList(
            object=result.get("object", "list"),
            data=models,
        )
    
    async def retrieve(self, model: str, **kwargs: Any) -> Model:
        """
        Retrieve a model by ID asynchronously.
        
        Get details about a specific model, including its capabilities
        and ownership information.
        
        Args:
            model: Model ID to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Model object
        """
        url = f"{self.config.base_url}/models/{model}"
        
        response = await self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_model(result)
    
    async def delete(self, model: str, **kwargs: Any) -> ModelDeleted:
        """
        Delete a fine-tuned model asynchronously.
        
        Delete a fine-tuned model. This action cannot be undone.
        Only fine-tuned models can be deleted; base models cannot
        be deleted.
        
        Args:
            model: Model ID to delete
            **kwargs: Additional parameters
            
        Returns:
            Deletion confirmation
        """
        url = f"{self.config.base_url}/models/{model}"
        
        response = await self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return ModelDeleted(
            id=result.get("id", ""),
            object=result.get("object", "model"),
            deleted=result.get("deleted", True),
        )
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
