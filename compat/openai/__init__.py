"""
OpenAI Compatible API Client

A production-grade OpenAI API client implementation with support for:
- Chat Completions
- Embeddings
- Images
- Audio
- Assistants
- Threads
- Files
- Fine-tuning
- Models

Module path: compat/openai/__init__.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass, field

from .chat_completions import ChatCompletions, AsyncChatCompletions
from .embeddings import Embeddings, AsyncEmbeddings
from .images import Images, AsyncImages
from .audio import Audio, AsyncAudio
from .assistants import Assistants, AsyncAssistants
from .threads import Threads, AsyncThreads
from .files import Files, AsyncFiles
from .fine_tuning import FineTuning, AsyncFineTuning
from .models import Models, AsyncModels


__version__ = "1.0.0"
__all__ = [
    "OpenAI",
    "AsyncOpenAI",
    "ChatCompletions",
    "AsyncChatCompletions",
    "Embeddings",
    "AsyncEmbeddings",
    "Images",
    "AsyncImages",
    "Audio",
    "AsyncAudio",
    "Assistants",
    "AsyncAssistants",
    "Threads",
    "AsyncThreads",
    "Files",
    "AsyncFiles",
    "FineTuning",
    "AsyncFineTuning",
    "Models",
    "AsyncModels",
    "OpenAIError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    "APIConnectionError",
    "BadRequestError",
    "NotFoundError",
]


class OpenAIError(Exception):
    """Base exception for OpenAI API errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        param: Optional[str] = None,
        type: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param
        self.type = type


class AuthenticationError(OpenAIError):
    """Raised when authentication fails."""
    pass


class RateLimitError(OpenAIError):
    """Raised when rate limit is exceeded."""
    pass


class APIError(OpenAIError):
    """Raised when API returns an error."""
    pass


class APIConnectionError(OpenAIError):
    """Raised when connection to API fails."""
    pass


class BadRequestError(OpenAIError):
    """Raised when request is invalid."""
    pass


class NotFoundError(OpenAIError):
    """Raised when resource is not found."""
    pass


@dataclass
class ClientConfig:
    """Configuration for OpenAI client."""
    
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    organization: Optional[str] = None
    project: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    verify_ssl: bool = True
    proxies: Optional[Dict[str, str]] = None
    default_headers: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.organization is None:
            self.organization = os.environ.get("OPENAI_ORG_ID")
        if self.project is None:
            self.project = os.environ.get("OPENAI_PROJECT_ID")


class OpenAI:
    """
    Synchronous OpenAI API client.
    
    Provides access to all OpenAI API endpoints with automatic retry,
    error handling, and streaming support.
    
    Example:
        >>> from compat.openai import OpenAI
        >>> client = OpenAI(api_key="your-api-key")
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        verify_ssl: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            base_url: API base URL. Defaults to https://api.openai.com/v1
            organization: Organization ID. Defaults to OPENAI_ORG_ID env var.
            project: Project ID. Defaults to OPENAI_PROJECT_ID env var.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay between retries in seconds.
            retry_backoff: Backoff multiplier for retry delays.
            verify_ssl: Whether to verify SSL certificates.
            proxies: Proxy configuration dictionary.
            default_headers: Default headers to include in all requests.
        """
        self.config = ClientConfig(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            organization=organization,
            project=project,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            verify_ssl=verify_ssl,
            proxies=proxies,
            default_headers=default_headers or {},
        )
        
        if not self.config.api_key:
            raise AuthenticationError(
                "API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize API resource clients
        self._chat = ChatCompletions(self.config)
        self._embeddings = Embeddings(self.config)
        self._images = Images(self.config)
        self._audio = Audio(self.config)
        self._assistants = Assistants(self.config)
        self._threads = Threads(self.config)
        self._files = Files(self.config)
        self._fine_tuning = FineTuning(self.config)
        self._models = Models(self.config)
    
    @property
    def chat(self) -> ChatCompletions:
        """Access chat completions API."""
        return self._chat
    
    @property
    def completions(self) -> ChatCompletions:
        """Alias for chat completions API."""
        return self._chat
    
    @property
    def embeddings(self) -> Embeddings:
        """Access embeddings API."""
        return self._embeddings
    
    @property
    def images(self) -> Images:
        """Access images API."""
        return self._images
    
    @property
    def audio(self) -> Audio:
        """Access audio API."""
        return self._audio
    
    @property
    def assistants(self) -> Assistants:
        """Access assistants API."""
        return self._assistants
    
    @property
    def threads(self) -> Threads:
        """Access threads API."""
        return self._threads
    
    @property
    def files(self) -> Files:
        """Access files API."""
        return self._files
    
    @property
    def fine_tuning(self) -> FineTuning:
        """Access fine-tuning API."""
        return self._fine_tuning
    
    @property
    def models(self) -> Models:
        """Access models API."""
        return self._models
    
    def close(self) -> None:
        """Close the client and release resources."""
        self._chat.close()
        self._embeddings.close()
        self._images.close()
        self._audio.close()
        self._assistants.close()
        self._threads.close()
        self._files.close()
        self._fine_tuning.close()
        self._models.close()
    
    def __enter__(self) -> OpenAI:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class AsyncOpenAI:
    """
    Asynchronous OpenAI API client.
    
    Provides async access to all OpenAI API endpoints with automatic retry,
    error handling, and streaming support.
    
    Example:
        >>> from compat.openai import AsyncOpenAI
        >>> client = AsyncOpenAI(api_key="your-api-key")
        >>> response = await client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        verify_ssl: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize async OpenAI client.
        
        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            base_url: API base URL. Defaults to https://api.openai.com/v1
            organization: Organization ID. Defaults to OPENAI_ORG_ID env var.
            project: Project ID. Defaults to OPENAI_PROJECT_ID env var.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay between retries in seconds.
            retry_backoff: Backoff multiplier for retry delays.
            verify_ssl: Whether to verify SSL certificates.
            proxies: Proxy configuration dictionary.
            default_headers: Default headers to include in all requests.
        """
        self.config = ClientConfig(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            organization=organization,
            project=project,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            verify_ssl=verify_ssl,
            proxies=proxies,
            default_headers=default_headers or {},
        )
        
        if not self.config.api_key:
            raise AuthenticationError(
                "API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize async API resource clients
        self._chat = AsyncChatCompletions(self.config)
        self._embeddings = AsyncEmbeddings(self.config)
        self._images = AsyncImages(self.config)
        self._audio = AsyncAudio(self.config)
        self._assistants = AsyncAssistants(self.config)
        self._threads = AsyncThreads(self.config)
        self._files = AsyncFiles(self.config)
        self._fine_tuning = AsyncFineTuning(self.config)
        self._models = AsyncModels(self.config)
    
    @property
    def chat(self) -> AsyncChatCompletions:
        """Access async chat completions API."""
        return self._chat
    
    @property
    def completions(self) -> AsyncChatCompletions:
        """Alias for async chat completions API."""
        return self._chat
    
    @property
    def embeddings(self) -> AsyncEmbeddings:
        """Access async embeddings API."""
        return self._embeddings
    
    @property
    def images(self) -> AsyncImages:
        """Access async images API."""
        return self._images
    
    @property
    def audio(self) -> AsyncAudio:
        """Access async audio API."""
        return self._audio
    
    @property
    def assistants(self) -> AsyncAssistants:
        """Access async assistants API."""
        return self._assistants
    
    @property
    def threads(self) -> AsyncThreads:
        """Access async threads API."""
        return self._threads
    
    @property
    def files(self) -> AsyncFiles:
        """Access async files API."""
        return self._files
    
    @property
    def fine_tuning(self) -> AsyncFineTuning:
        """Access async fine-tuning API."""
        return self._fine_tuning
    
    @property
    def models(self) -> AsyncModels:
        """Access async models API."""
        return self._models
    
    async def close(self) -> None:
        """Close the client and release resources."""
        await self._chat.close()
        await self._embeddings.close()
        await self._images.close()
        await self._audio.close()
        await self._assistants.close()
        await self._threads.close()
        await self._files.close()
        await self._fine_tuning.close()
        await self._models.close()
    
    async def __aenter__(self) -> AsyncOpenAI:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
