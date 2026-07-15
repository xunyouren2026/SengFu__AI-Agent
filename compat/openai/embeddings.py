"""
Embeddings API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Embeddings API
for generating vector representations of text.

Module path: compat/openai/embeddings.py
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Union, Iterator, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import base64

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class EncodingFormat(str, Enum):
    """Encoding formats for embeddings."""
    FLOAT = "float"
    BASE64 = "base64"


@dataclass
class EmbeddingRequest:
    """Embedding request parameters."""
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class Embedding:
    """Single embedding result."""
    index: int
    embedding: Union[List[float], str]
    object: str = "embedding"


@dataclass
class EmbeddingUsage:
    """Token usage information for embeddings."""
    prompt_tokens: int
    total_tokens: int


@dataclass
class CreateEmbeddingResponse:
    """Embedding creation response."""
    object: str
    data: List[Embedding]
    model: str
    usage: EmbeddingUsage


class BaseEmbeddings:
    """Base class for embeddings."""
    
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
    
    def _decode_base64_embedding(self, encoded: str) -> List[float]:
        """Decode base64 encoded embedding to float list."""
        import struct
        decoded = base64.b64decode(encoded)
        # Assuming float32 format
        num_floats = len(decoded) // 4
        return list(struct.unpack(f"<{num_floats}f", decoded))
    
    def _parse_response(self, data: Dict[str, Any]) -> CreateEmbeddingResponse:
        """Parse embedding response."""
        embeddings = []
        for item in data.get("data", []):
            embedding_value = item.get("embedding")
            # Decode base64 if needed
            if isinstance(embedding_value, str):
                try:
                    embedding_value = self._decode_base64_embedding(embedding_value)
                except Exception:
                    pass  # Keep as string if decoding fails
            
            embedding = Embedding(
                index=item.get("index", 0),
                embedding=embedding_value,
                object=item.get("object", "embedding"),
            )
            embeddings.append(embedding)
        
        usage_data = data.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        
        return CreateEmbeddingResponse(
            object=data.get("object", "list"),
            data=embeddings,
            model=data.get("model", ""),
            usage=usage,
        )


class Embeddings(BaseEmbeddings):
    """
    Synchronous Embeddings API client.
    
    Example:
        >>> client = Embeddings(config)
        >>> response = client.create(
        ...     model="text-embedding-3-small",
        ...     input="The quick brown fox jumps over the lazy dog"
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
        input: Union[str, List[str]],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> CreateEmbeddingResponse:
        """
        Create embeddings for the given input.
        
        Args:
            input: Text to embed (string or list of strings)
            model: ID of the model to use (e.g., "text-embedding-3-small")
            encoding_format: Format for returned embeddings ("float" or "base64")
            dimensions: Number of dimensions for the output embeddings
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            CreateEmbeddingResponse containing embeddings and usage info
            
        Example:
            >>> response = client.create(
            ...     input="Hello, world!",
            ...     model="text-embedding-3-small"
            ... )
            >>> print(response.data[0].embedding[:5])  # First 5 dimensions
        """
        request = EmbeddingRequest(
            input=input,
            model=model,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )
        
        url = f"{self.config.base_url}/embeddings"
        data = request.to_dict()
        
        response = self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    def create_batch(
        self,
        inputs: List[str],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        batch_size: int = 100,
    ) -> Iterator[CreateEmbeddingResponse]:
        """
        Create embeddings for a large batch of inputs.
        
        Processes inputs in batches to avoid hitting API limits.
        
        Args:
            inputs: List of texts to embed
            model: ID of the model to use
            encoding_format: Format for returned embeddings
            dimensions: Number of dimensions for output
            user: End-user identifier
            batch_size: Number of inputs per batch
            
        Yields:
            CreateEmbeddingResponse for each batch
        """
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            yield self.create(
                input=batch,
                model=model,
                encoding_format=encoding_format,
                dimensions=dimensions,
                user=user,
            )
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class AsyncEmbeddings(BaseEmbeddings):
    """
    Asynchronous Embeddings API client.
    
    Example:
        >>> client = AsyncEmbeddings(config)
        >>> response = await client.create(
        ...     model="text-embedding-3-small",
        ...     input="The quick brown fox jumps over the lazy dog"
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
        input: Union[str, List[str]],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> CreateEmbeddingResponse:
        """
        Create embeddings asynchronously for the given input.
        
        Args:
            input: Text to embed (string or list of strings)
            model: ID of the model to use (e.g., "text-embedding-3-small")
            encoding_format: Format for returned embeddings ("float" or "base64")
            dimensions: Number of dimensions for the output embeddings
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            CreateEmbeddingResponse containing embeddings and usage info
        """
        request = EmbeddingRequest(
            input=input,
            model=model,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )
        
        url = f"{self.config.base_url}/embeddings"
        data = request.to_dict()
        
        response = await self._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    async def create_batch(
        self,
        inputs: List[str],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        batch_size: int = 100,
        max_concurrent: int = 5,
    ) -> AsyncIterator[CreateEmbeddingResponse]:
        """
        Create embeddings asynchronously for a large batch of inputs.
        
        Processes inputs in batches with controlled concurrency.
        
        Args:
            inputs: List of texts to embed
            model: ID of the model to use
            encoding_format: Format for returned embeddings
            dimensions: Number of dimensions for output
            user: End-user identifier
            batch_size: Number of inputs per batch
            max_concurrent: Maximum concurrent requests
            
        Yields:
            CreateEmbeddingResponse for each batch
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: List[str]) -> CreateEmbeddingResponse:
            async with semaphore:
                return await self.create(
                    input=batch,
                    model=model,
                    encoding_format=encoding_format,
                    dimensions=dimensions,
                    user=user,
                )
        
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        tasks = [process_batch(batch) for batch in batches]
        
        for task in asyncio.as_completed(tasks):
            yield await task
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
