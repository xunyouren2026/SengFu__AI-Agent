"""
Fine-tuning API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Fine-tuning API
for creating and managing fine-tuned models.

Module path: compat/openai/fine_tuning.py
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


class FineTuningJobStatus(str, Enum):
    """Fine-tuning job status values."""
    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuningJobIntegrationType(str, Enum):
    """Fine-tuning job integration types."""
    WANDB = "wandb"


@dataclass
class FineTuningJobHyperparameters:
    """Fine-tuning job hyperparameters."""
    n_epochs: Optional[Union[int, str]] = None
    batch_size: Optional[Union[int, str]] = None
    learning_rate_multiplier: Optional[Union[float, str]] = None


@dataclass
class FineTuningJobIntegration:
    """Fine-tuning job integration configuration."""
    type: str
    wandb: Dict[str, Any]


@dataclass
class FineTuningJob:
    """Fine-tuning job object."""
    id: str
    object: str
    created_at: int
    finished_at: Optional[int]
    model: str
    fine_tuned_model: Optional[str]
    organization_id: str
    result_files: List[str]
    status: str
    validation_file: Optional[str]
    training_file: str
    hyperparameters: FineTuningJobHyperparameters
    trained_tokens: Optional[int]
    error: Optional[Dict[str, Any]]
    seed: int
    estimated_finish: Optional[int]
    integrations: Optional[List[FineTuningJobIntegration]]
    metadata: Optional[Dict[str, str]]
    method: Optional[Dict[str, Any]]


@dataclass
class FineTuningJobEvent:
    """Fine-tuning job event."""
    id: str
    object: str
    created_at: int
    level: str
    message: str
    data: Optional[Dict[str, Any]]
    type: str


@dataclass
class FineTuningJobCheckpoint:
    """Fine-tuning job checkpoint."""
    id: str
    object: str
    created_at: int
    fine_tuned_model_checkpoint: str
    metrics: Dict[str, Any]
    fine_tuning_job_id: str
    step_number: int


@dataclass
class FineTuningJobList:
    """List of fine-tuning jobs."""
    object: str
    data: List[FineTuningJob]
    has_more: bool


@dataclass
class FineTuningJobEventList:
    """List of fine-tuning job events."""
    object: str
    data: List[FineTuningJobEvent]
    has_more: bool


@dataclass
class FineTuningJobCheckpointList:
    """List of fine-tuning job checkpoints."""
    object: str
    data: List[FineTuningJobCheckpoint]
    has_more: bool


class BaseFineTuning:
    """Base class for fine-tuning API."""
    
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
    
    def _parse_hyperparameters(self, data: Dict[str, Any]) -> FineTuningJobHyperparameters:
        """Parse hyperparameters from API response."""
        return FineTuningJobHyperparameters(
            n_epochs=data.get("n_epochs"),
            batch_size=data.get("batch_size"),
            learning_rate_multiplier=data.get("learning_rate_multiplier"),
        )
    
    def _parse_job(self, data: Dict[str, Any]) -> FineTuningJob:
        """Parse fine-tuning job from API response."""
        hyperparams = self._parse_hyperparameters(data.get("hyperparameters", {}))
        
        integrations = None
        if "integrations" in data and data["integrations"]:
            integrations = []
            for int_data in data["integrations"]:
                integrations.append(FineTuningJobIntegration(
                    type=int_data.get("type", ""),
                    wandb=int_data.get("wandb", {}),
                ))
        
        return FineTuningJob(
            id=data.get("id", ""),
            object=data.get("object", "fine_tuning.job"),
            created_at=data.get("created_at", 0),
            finished_at=data.get("finished_at"),
            model=data.get("model", ""),
            fine_tuned_model=data.get("fine_tuned_model"),
            organization_id=data.get("organization_id", ""),
            result_files=data.get("result_files", []),
            status=data.get("status", ""),
            validation_file=data.get("validation_file"),
            training_file=data.get("training_file", ""),
            hyperparameters=hyperparams,
            trained_tokens=data.get("trained_tokens"),
            error=data.get("error"),
            seed=data.get("seed", 0),
            estimated_finish=data.get("estimated_finish"),
            integrations=integrations,
            metadata=data.get("metadata"),
            method=data.get("method"),
        )
    
    def _parse_event(self, data: Dict[str, Any]) -> FineTuningJobEvent:
        """Parse event from API response."""
        return FineTuningJobEvent(
            id=data.get("id", ""),
            object=data.get("object", "fine_tuning.job.event"),
            created_at=data.get("created_at", 0),
            level=data.get("level", ""),
            message=data.get("message", ""),
            data=data.get("data"),
            type=data.get("type", ""),
        )
    
    def _parse_checkpoint(self, data: Dict[str, Any]) -> FineTuningJobCheckpoint:
        """Parse checkpoint from API response."""
        return FineTuningJobCheckpoint(
            id=data.get("id", ""),
            object=data.get("object", "fine_tuning.job.checkpoint"),
            created_at=data.get("created_at", 0),
            fine_tuned_model_checkpoint=data.get("fine_tuned_model_checkpoint", ""),
            metrics=data.get("metrics", {}),
            fine_tuning_job_id=data.get("fine_tuning_job_id", ""),
            step_number=data.get("step_number", 0),
        )


class FineTuning(BaseFineTuning):
    """
    Synchronous Fine-tuning API client.
    
    Example:
        >>> client = FineTuning(config)
        >>> job = client.jobs.create(
        ...     model="gpt-3.5-turbo",
        ...     training_file="file-abc123"
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
        self._jobs = FineTuningJobs(self)
    
    @property
    def jobs(self) -> FineTuningJobs:
        """Access fine-tuning jobs API."""
        return self._jobs
    
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
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class FineTuningJobs:
    """Fine-tuning jobs API."""
    
    def __init__(self, fine_tuning: FineTuning):
        self._fine_tuning = fine_tuning
    
    def create(
        self,
        model: str,
        training_file: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        validation_file: Optional[str] = None,
        integrations: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        method: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> FineTuningJob:
        """
        Create a fine-tuning job.
        
        Args:
            model: Base model to fine-tune
            training_file: ID of training file
            hyperparameters: Fine-tuning hyperparameters
            suffix: Suffix for the fine-tuned model name
            validation_file: ID of validation file
            integrations: Third-party integrations (e.g., Weights & Biases)
            seed: Random seed for reproducibility
            metadata: Key-value metadata
            method: Fine-tuning method configuration
            **kwargs: Additional parameters
            
        Returns:
            Created FineTuningJob object
        """
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs"
        
        data: Dict[str, Any] = {
            "model": model,
            "training_file": training_file,
        }
        if hyperparameters is not None:
            data["hyperparameters"] = hyperparameters
        if suffix is not None:
            data["suffix"] = suffix
        if validation_file is not None:
            data["validation_file"] = validation_file
        if integrations is not None:
            data["integrations"] = integrations
        if seed is not None:
            data["seed"] = seed
        if metadata is not None:
            data["metadata"] = metadata
        if method is not None:
            data["method"] = method
        
        response = self._fine_tuning._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        return self._fine_tuning._parse_job(result)
    
    def retrieve(self, fine_tuning_job_id: str, **kwargs: Any) -> FineTuningJob:
        """
        Retrieve a fine-tuning job.
        
        Args:
            fine_tuning_job_id: ID of the fine-tuning job
            **kwargs: Additional parameters
            
        Returns:
            FineTuningJob object
        """
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}"
        
        response = self._fine_tuning._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        return self._fine_tuning._parse_job(result)
    
    def list(
        self,
        after: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any
    ) -> FineTuningJobList:
        """
        List fine-tuning jobs.
        
        Args:
            after: Cursor for pagination
            limit: Number of jobs to return
            **kwargs: Additional parameters
            
        Returns:
            List of fine-tuning jobs
        """
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs"
        
        params: Dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = self._fine_tuning._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        jobs = [self._fine_tuning._parse_job(item) for item in result.get("data", [])]
        
        return FineTuningJobList(
            object=result.get("object", "list"),
            data=jobs,
            has_more=result.get("has_more", False),
        )
    
    def cancel(self, fine_tuning_job_id: str, **kwargs: Any) -> FineTuningJob:
        """
        Cancel a fine-tuning job.
        
        Args:
            fine_tuning_job_id: ID of the job to cancel
            **kwargs: Additional parameters
            
        Returns:
            Cancelled FineTuningJob object
        """
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}/cancel"
        
        response = self._fine_tuning._request_with_retry("POST", url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        return self._fine_tuning._parse_job(result)
    
    def list_events(
        self,
        fine_tuning_job_id: str,
        after: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any
    ) -> FineTuningJobEventList:
        """
        List events for a fine-tuning job.
        
        Args:
            fine_tuning_job_id: ID of the fine-tuning job
            after: Cursor for pagination
            limit: Number of events to return
            **kwargs: Additional parameters
            
        Returns:
            List of fine-tuning job events
        """
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}/events"
        
        params: Dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = self._fine_tuning._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        events = [self._fine_tuning._parse_event(item) for item in result.get("data", [])]
        
        return FineTuningJobEventList(
            object=result.get("object", "list"),
            data=events,
            has_more=result.get("has_more", False),
        )
    
    def list_checkpoints(
        self,
        fine_tuning_job_id: str,
        after: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any
    ) -> FineTuningJobCheckpointList:
        """
        List checkpoints for a fine-tuning job.
        
        Args:
            fine_tuning_job_id: ID of the fine-tuning job
            after: Cursor for pagination
            limit: Number of checkpoints to return
            **kwargs: Additional parameters
            
        Returns:
            List of fine-tuning job checkpoints
        """
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints"
        
        params: Dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = self._fine_tuning._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        checkpoints = [self._fine_tuning._parse_checkpoint(item) for item in result.get("data", [])]
        
        return FineTuningJobCheckpointList(
            object=result.get("object", "list"),
            data=checkpoints,
            has_more=result.get("has_more", False),
        )


class AsyncFineTuning(BaseFineTuning):
    """
    Asynchronous Fine-tuning API client.
    
    Example:
        >>> client = AsyncFineTuning(config)
        >>> job = await client.jobs.create(
        ...     model="gpt-3.5-turbo",
        ...     training_file="file-abc123"
        ... )
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client: Optional[httpx.AsyncClient] = None
        self._config = config
        self._jobs = AsyncFineTuningJobs(self)
    
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
    def jobs(self) -> AsyncFineTuningJobs:
        """Access async fine-tuning jobs API."""
        return self._jobs
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class AsyncFineTuningJobs:
    """Async fine-tuning jobs API."""
    
    def __init__(self, fine_tuning: AsyncFineTuning):
        self._fine_tuning = fine_tuning
    
    async def create(
        self,
        model: str,
        training_file: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        validation_file: Optional[str] = None,
        integrations: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        method: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> FineTuningJob:
        """Create a fine-tuning job asynchronously."""
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs"
        
        data: Dict[str, Any] = {
            "model": model,
            "training_file": training_file,
        }
        if hyperparameters is not None:
            data["hyperparameters"] = hyperparameters
        if suffix is not None:
            data["suffix"] = suffix
        if validation_file is not None:
            data["validation_file"] = validation_file
        if integrations is not None:
            data["integrations"] = integrations
        if seed is not None:
            data["seed"] = seed
        if metadata is not None:
            data["metadata"] = metadata
        if method is not None:
            data["method"] = method
        
        response = await self._fine_tuning._request_with_retry("POST", url, data)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        return self._fine_tuning._parse_job(result)
    
    async def retrieve(self, fine_tuning_job_id: str, **kwargs: Any) -> FineTuningJob:
        """Retrieve a fine-tuning job asynchronously."""
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}"
        
        response = await self._fine_tuning._request_with_retry("GET", url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        return self._fine_tuning._parse_job(result)
    
    async def list(
        self,
        after: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any
    ) -> FineTuningJobList:
        """List fine-tuning jobs asynchronously."""
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs"
        
        params: Dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = await self._fine_tuning._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        jobs = [self._fine_tuning._parse_job(item) for item in result.get("data", [])]
        
        return FineTuningJobList(
            object=result.get("object", "list"),
            data=jobs,
            has_more=result.get("has_more", False),
        )
    
    async def cancel(self, fine_tuning_job_id: str, **kwargs: Any) -> FineTuningJob:
        """Cancel a fine-tuning job asynchronously."""
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}/cancel"
        
        response = await self._fine_tuning._request_with_retry("POST", url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        return self._fine_tuning._parse_job(result)
    
    async def list_events(
        self,
        fine_tuning_job_id: str,
        after: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any
    ) -> FineTuningJobEventList:
        """List events for a fine-tuning job asynchronously."""
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}/events"
        
        params: Dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = await self._fine_tuning._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        events = [self._fine_tuning._parse_event(item) for item in result.get("data", [])]
        
        return FineTuningJobEventList(
            object=result.get("object", "list"),
            data=events,
            has_more=result.get("has_more", False),
        )
    
    async def list_checkpoints(
        self,
        fine_tuning_job_id: str,
        after: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any
    ) -> FineTuningJobCheckpointList:
        """List checkpoints for a fine-tuning job asynchronously."""
        url = f"{self._fine_tuning.config.base_url}/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints"
        
        params: Dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query}"
        
        response = await self._fine_tuning._request_with_retry("GET", full_url)
        
        if response.status_code != 200:
            self._fine_tuning._handle_error(response)
        
        result = response.json()
        checkpoints = [self._fine_tuning._parse_checkpoint(item) for item in result.get("data", [])]
        
        return FineTuningJobCheckpointList(
            object=result.get("object", "list"),
            data=checkpoints,
            has_more=result.get("has_more", False),
        )
