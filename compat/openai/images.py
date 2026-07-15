"""
Images API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Images API
for generating, editing, and creating variations of images.

Module path: compat/openai/images.py
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Union, BinaryIO, Iterator, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio
import base64
import io

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class ImageQuality(str, Enum):
    """Image quality options."""
    STANDARD = "standard"
    HD = "hd"


class ImageResponseFormat(str, Enum):
    """Image response format options."""
    URL = "url"
    B64_JSON = "b64_json"


class ImageSize(str, Enum):
    """Image size options for DALL-E models."""
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    WIDE = "1792x1024"
    TALL = "1024x1792"


class ImageStyle(str, Enum):
    """Image style options for DALL-E 3."""
    VIVID = "vivid"
    NATURAL = "natural"


@dataclass
class Image:
    """Generated image data."""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


@dataclass
class ImagesResponse:
    """Images generation response."""
    created: int
    data: List[Image]


class BaseImages:
    """Base class for images API."""
    
    def __init__(self, config: Any):
        self.config = config
        self._client: Optional[Any] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
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
    
    def _parse_response(self, data: Dict[str, Any]) -> ImagesResponse:
        """Parse images response."""
        images = []
        for item in data.get("data", []):
            image = Image(
                url=item.get("url"),
                b64_json=item.get("b64_json"),
                revised_prompt=item.get("revised_prompt"),
            )
            images.append(image)
        
        return ImagesResponse(
            created=data.get("created", int(time.time())),
            data=images,
        )
    
    def _read_file(self, file: Union[str, Path, BinaryIO, bytes]) -> bytes:
        """Read file content from various input types."""
        if isinstance(file, (str, Path)):
            with open(file, "rb") as f:
                return f.read()
        elif isinstance(file, bytes):
            return file
        else:
            return file.read()


class Images(BaseImages):
    """
    Synchronous Images API client.
    
    Example:
        >>> client = Images(config)
        >>> response = client.generate(
        ...     model="dall-e-3",
        ...     prompt="A cute cat playing with a ball of yarn"
        ... )
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client = httpx.Client(
            timeout=config.timeout * 3,  # Images take longer
            verify=config.verify_ssl,
            proxies=config.proxies,
        )
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        from . import APIConnectionError
        
        delay = self.config.retry_delay
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if files:
                    response = self._client.request(
                        method=method,
                        url=url,
                        headers={k: v for k, v in self._get_headers().items() 
                                if k.lower() != "content-type"},
                        files=files,
                        data=data,
                    )
                else:
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
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        n: Optional[int] = None,
        quality: Optional[str] = None,
        response_format: Optional[str] = None,
        size: Optional[str] = None,
        style: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> ImagesResponse:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            model: Model to use ("dall-e-2", "dall-e-3", "gpt-image-1")
            n: Number of images to generate (1-10 for DALL-E 2, 1 for DALL-E 3)
            quality: Image quality ("standard" or "hd")
            response_format: Format ("url" or "b64_json")
            size: Image size (e.g., "1024x1024")
            style: Image style for DALL-E 3 ("vivid" or "natural")
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ImagesResponse containing generated images
        """
        url = f"{self.config.base_url}/images/generations"
        
        data = {
            "prompt": prompt,
        }
        if model:
            data["model"] = model
        if n is not None:
            data["n"] = n
        if quality:
            data["quality"] = quality
        if response_format:
            data["response_format"] = response_format
        if size:
            data["size"] = size
        if style:
            data["style"] = style
        if user:
            data["user"] = user
        
        response = self._request_with_retry("POST", url, json_data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    def edit(
        self,
        image: Union[str, Path, BinaryIO, bytes],
        prompt: str,
        mask: Optional[Union[str, Path, BinaryIO, bytes]] = None,
        model: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        response_format: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> ImagesResponse:
        """
        Edit an image based on a prompt.
        
        Args:
            image: Source image to edit
            prompt: Text description of desired edit
            mask: Optional mask image (transparent areas indicate edit regions)
            model: Model to use
            n: Number of images to generate
            size: Image size
            response_format: Format ("url" or "b64_json")
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ImagesResponse containing edited images
        """
        url = f"{self.config.base_url}/images/edits"
        
        files = {
            "image": ("image.png", self._read_file(image), "image/png"),
        }
        if mask:
            files["mask"] = ("mask.png", self._read_file(mask), "image/png")
        
        data: Dict[str, Any] = {"prompt": prompt}
        if model:
            data["model"] = model
        if n is not None:
            data["n"] = n
        if size:
            data["size"] = size
        if response_format:
            data["response_format"] = response_format
        if user:
            data["user"] = user
        
        response = self._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    def create_variation(
        self,
        image: Union[str, Path, BinaryIO, bytes],
        model: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        response_format: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> ImagesResponse:
        """
        Create variations of an image.
        
        Args:
            image: Source image
            model: Model to use
            n: Number of variations to generate
            size: Image size
            response_format: Format ("url" or "b64_json")
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ImagesResponse containing image variations
        """
        url = f"{self.config.base_url}/images/variations"
        
        files = {
            "image": ("image.png", self._read_file(image), "image/png"),
        }
        
        data: Dict[str, Any] = {}
        if model:
            data["model"] = model
        if n is not None:
            data["n"] = n
        if size:
            data["size"] = size
        if response_format:
            data["response_format"] = response_format
        if user:
            data["user"] = user
        
        response = self._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class AsyncImages(BaseImages):
    """
    Asynchronous Images API client.
    
    Example:
        >>> client = AsyncImages(config)
        >>> response = await client.generate(
        ...     model="dall-e-3",
        ...     prompt="A cute cat playing with a ball of yarn"
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
                timeout=self._config.timeout * 3,
                verify=self._config.verify_ssl,
                proxies=self._config.proxies,
            )
        return self._client
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make async HTTP request with retry logic."""
        from . import APIConnectionError
        
        client = await self._get_client()
        delay = self.config.retry_delay
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if files:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers={k: v for k, v in self._get_headers().items() 
                                if k.lower() != "content-type"},
                        files=files,
                        data=data,
                    )
                else:
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
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        n: Optional[int] = None,
        quality: Optional[str] = None,
        response_format: Optional[str] = None,
        size: Optional[str] = None,
        style: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> ImagesResponse:
        """
        Generate an image asynchronously from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            model: Model to use ("dall-e-2", "dall-e-3", "gpt-image-1")
            n: Number of images to generate
            quality: Image quality ("standard" or "hd")
            response_format: Format ("url" or "b64_json")
            size: Image size
            style: Image style for DALL-E 3
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ImagesResponse containing generated images
        """
        url = f"{self.config.base_url}/images/generations"
        
        data = {"prompt": prompt}
        if model:
            data["model"] = model
        if n is not None:
            data["n"] = n
        if quality:
            data["quality"] = quality
        if response_format:
            data["response_format"] = response_format
        if size:
            data["size"] = size
        if style:
            data["style"] = style
        if user:
            data["user"] = user
        
        response = await self._request_with_retry("POST", url, json_data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    async def edit(
        self,
        image: Union[str, Path, BinaryIO, bytes],
        prompt: str,
        mask: Optional[Union[str, Path, BinaryIO, bytes]] = None,
        model: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        response_format: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> ImagesResponse:
        """
        Edit an image asynchronously based on a prompt.
        
        Args:
            image: Source image to edit
            prompt: Text description of desired edit
            mask: Optional mask image
            model: Model to use
            n: Number of images to generate
            size: Image size
            response_format: Format ("url" or "b64_json")
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ImagesResponse containing edited images
        """
        url = f"{self.config.base_url}/images/edits"
        
        files = {
            "image": ("image.png", self._read_file(image), "image/png"),
        }
        if mask:
            files["mask"] = ("mask.png", self._read_file(mask), "image/png")
        
        data: Dict[str, Any] = {"prompt": prompt}
        if model:
            data["model"] = model
        if n is not None:
            data["n"] = n
        if size:
            data["size"] = size
        if response_format:
            data["response_format"] = response_format
        if user:
            data["user"] = user
        
        response = await self._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    async def create_variation(
        self,
        image: Union[str, Path, BinaryIO, bytes],
        model: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        response_format: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs: Any
    ) -> ImagesResponse:
        """
        Create variations of an image asynchronously.
        
        Args:
            image: Source image
            model: Model to use
            n: Number of variations to generate
            size: Image size
            response_format: Format ("url" or "b64_json")
            user: End-user identifier
            **kwargs: Additional parameters
            
        Returns:
            ImagesResponse containing image variations
        """
        url = f"{self.config.base_url}/images/variations"
        
        files = {
            "image": ("image.png", self._read_file(image), "image/png"),
        }
        
        data: Dict[str, Any] = {}
        if model:
            data["model"] = model
        if n is not None:
            data["n"] = n
        if size:
            data["size"] = size
        if response_format:
            data["response_format"] = response_format
        if user:
            data["user"] = user
        
        response = await self._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_response(result)
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
