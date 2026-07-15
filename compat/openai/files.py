"""
Files API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Files API
for uploading and managing files used across various endpoints.

Module path: compat/openai/files.py
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Union, BinaryIO, Iterator, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio
import mimetypes

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class FilePurpose(str, Enum):
    """File purpose options."""
    ASSISTANTS = "assistants"
    ASSISTANTS_OUTPUT = "assistants_output"
    BATCH = "batch"
    BATCH_OUTPUT = "batch_output"
    FINE_TUNE = "fine-tune"
    FINE_TUNE_RESULTS = "fine-tune-results"
    VISION = "vision"
    USER_DATA = "user_data"


@dataclass
class OpenAIFile:
    """File object."""
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str
    status_details: Optional[str]


@dataclass
class FileDeleted:
    """File deletion confirmation."""
    id: str
    object: str
    deleted: bool


@dataclass
class FileList:
    """List of files."""
    object: str
    data: List[OpenAIFile]


@dataclass
class FileContent:
    """File content response."""
    content: bytes
    filename: str


class BaseFiles:
    """Base class for files API."""
    
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
    
    def _parse_file(self, data: Dict[str, Any]) -> OpenAIFile:
        """Parse file from API response."""
        return OpenAIFile(
            id=data.get("id", ""),
            object=data.get("object", "file"),
            bytes=data.get("bytes", 0),
            created_at=data.get("created_at", 0),
            filename=data.get("filename", ""),
            purpose=data.get("purpose", ""),
            status=data.get("status", ""),
            status_details=data.get("status_details"),
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
    
    def _get_filename(self, file: Union[str, Path, BinaryIO, bytes]) -> str:
        """Get filename from various input types."""
        if isinstance(file, str):
            return Path(file).name
        elif isinstance(file, Path):
            return file.name
        else:
            return "file"
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type from filename."""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"


class Files(BaseFiles):
    """
    Synchronous Files API client.
    
    Example:
        >>> client = Files(config)
        >>> file = client.create(
        ...     file=open("data.jsonl", "rb"),
        ...     purpose="fine-tune"
        ... )
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client = httpx.Client(
            timeout=config.timeout * 3,
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
    
    def create(
        self,
        file: Union[str, Path, BinaryIO, bytes],
        purpose: str,
        **kwargs: Any
    ) -> OpenAIFile:
        """
        Upload a file.
        
        Args:
            file: File to upload (path, bytes, or file-like object)
            purpose: Purpose of the file (e.g., "fine-tune", "assistants")
            **kwargs: Additional parameters
            
        Returns:
            Uploaded file object
        """
        url = f"{self.config.base_url}/files"
        
        filename = self._get_filename(file)
        file_content = self._read_file(file)
        mime_type = self._get_mime_type(filename)
        
        files = {
            "file": (filename, file_content, mime_type),
        }
        data = {"purpose": purpose}
        
        response = self._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_file(result)
    
    def list(
        self,
        purpose: Optional[str] = None,
        **kwargs: Any
    ) -> FileList:
        """
        List files.
        
        Args:
            purpose: Filter by purpose
            **kwargs: Additional parameters
            
        Returns:
            List of files
        """
        url = f"{self.config.base_url}/files"
        
        if purpose:
            url = f"{url}?purpose={purpose}"
        
        response = self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        files = [self._parse_file(item) for item in result.get("data", [])]
        
        return FileList(
            object=result.get("object", "list"),
            data=files,
        )
    
    def retrieve(self, file_id: str, **kwargs: Any) -> OpenAIFile:
        """
        Retrieve a file by ID.
        
        Args:
            file_id: ID of the file
            **kwargs: Additional parameters
            
        Returns:
            File object
        """
        url = f"{self.config.base_url}/files/{file_id}"
        
        response = self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_file(result)
    
    def delete(self, file_id: str, **kwargs: Any) -> FileDeleted:
        """
        Delete a file.
        
        Args:
            file_id: ID of the file to delete
            **kwargs: Additional parameters
            
        Returns:
            Deletion confirmation
        """
        url = f"{self.config.base_url}/files/{file_id}"
        
        response = self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return FileDeleted(
            id=result.get("id", ""),
            object=result.get("object", "file"),
            deleted=result.get("deleted", True),
        )
    
    def content(self, file_id: str, **kwargs: Any) -> bytes:
        """
        Retrieve file content.
        
        Args:
            file_id: ID of the file
            **kwargs: Additional parameters
            
        Returns:
            File content as bytes
        """
        url = f"{self.config.base_url}/files/{file_id}/content"
        
        response = self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        return response.content
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class AsyncFiles(BaseFiles):
    """
    Asynchronous Files API client.
    
    Example:
        >>> client = AsyncFiles(config)
        >>> file = await client.create(
        ...     file=open("data.jsonl", "rb"),
        ...     purpose="fine-tune"
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
    
    async def create(
        self,
        file: Union[str, Path, BinaryIO, bytes],
        purpose: str,
        **kwargs: Any
    ) -> OpenAIFile:
        """
        Upload a file asynchronously.
        
        Args:
            file: File to upload
            purpose: Purpose of the file
            **kwargs: Additional parameters
            
        Returns:
            Uploaded file object
        """
        url = f"{self.config.base_url}/files"
        
        filename = self._get_filename(file)
        file_content = self._read_file(file)
        mime_type = self._get_mime_type(filename)
        
        files = {
            "file": (filename, file_content, mime_type),
        }
        data = {"purpose": purpose}
        
        response = await self._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_file(result)
    
    async def list(
        self,
        purpose: Optional[str] = None,
        **kwargs: Any
    ) -> FileList:
        """
        List files asynchronously.
        
        Args:
            purpose: Filter by purpose
            **kwargs: Additional parameters
            
        Returns:
            List of files
        """
        url = f"{self.config.base_url}/files"
        
        if purpose:
            url = f"{url}?purpose={purpose}"
        
        response = await self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        files = [self._parse_file(item) for item in result.get("data", [])]
        
        return FileList(
            object=result.get("object", "list"),
            data=files,
        )
    
    async def retrieve(self, file_id: str, **kwargs: Any) -> OpenAIFile:
        """
        Retrieve a file by ID asynchronously.
        
        Args:
            file_id: ID of the file
            **kwargs: Additional parameters
            
        Returns:
            File object
        """
        url = f"{self.config.base_url}/files/{file_id}"
        
        response = await self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return self._parse_file(result)
    
    async def delete(self, file_id: str, **kwargs: Any) -> FileDeleted:
        """
        Delete a file asynchronously.
        
        Args:
            file_id: ID of the file to delete
            **kwargs: Additional parameters
            
        Returns:
            Deletion confirmation
        """
        url = f"{self.config.base_url}/files/{file_id}"
        
        response = await self._request_with_retry("DELETE", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        result = response.json()
        return FileDeleted(
            id=result.get("id", ""),
            object=result.get("object", "file"),
            deleted=result.get("deleted", True),
        )
    
    async def content(self, file_id: str, **kwargs: Any) -> bytes:
        """
        Retrieve file content asynchronously.
        
        Args:
            file_id: ID of the file
            **kwargs: Additional parameters
            
        Returns:
            File content as bytes
        """
        url = f"{self.config.base_url}/files/{file_id}/content"
        
        response = await self._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._handle_error(response)
        
        return response.content
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
