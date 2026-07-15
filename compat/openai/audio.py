"""
Audio API - OpenAI Compatible Interface

Provides synchronous and asynchronous access to OpenAI's Audio API
for speech-to-text (transcription), text-to-speech, and audio translation.

Module path: compat/openai/audio.py
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Any, Union, BinaryIO, Iterator, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio
import io

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class AudioResponseFormat(str, Enum):
    """Response format for transcriptions and translations."""
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


class TTSModel(str, Enum):
    """Text-to-speech models."""
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"


class TTSVoice(str, Enum):
    """Text-to-speech voices."""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class WhisperModel(str, Enum):
    """Whisper models for transcription/translation."""
    WHISPER_1 = "whisper-1"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


@dataclass
class TranscriptionSegment:
    """Transcription segment with timestamps."""
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


@dataclass
class TranscriptionWord:
    """Individual word with timestamp."""
    word: str
    start: float
    end: float


@dataclass
class TranscriptionVerbose:
    """Verbose transcription response."""
    task: str
    language: str
    duration: float
    text: str
    segments: Optional[List[TranscriptionSegment]] = None
    words: Optional[List[TranscriptionWord]] = None


@dataclass
class CreateTranscriptionResponse:
    """Transcription creation response."""
    text: str


@dataclass
class CreateTranslationResponse:
    """Translation creation response."""
    text: str


class BaseAudio:
    """Base class for audio API."""
    
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
            return "audio.mp3"


class Audio(BaseAudio):
    """
    Synchronous Audio API client.
    
    Example:
        >>> client = Audio(config)
        >>> # Text to speech
        >>> response = client.speech.create(
        ...     model="tts-1",
        ...     voice="alloy",
        ...     input="Hello, world!"
        ... )
        >>> # Speech to text
        >>> transcription = client.transcriptions.create(
        ...     model="whisper-1",
        ...     file=open("audio.mp3", "rb")
        ... )
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client = httpx.Client(
            timeout=config.timeout * 5,  # Audio operations take longer
            verify=config.verify_ssl,
            proxies=config.proxies,
        )
        self._speech = Speech(self)
        self._transcriptions = Transcriptions(self)
        self._translations = Translations(self)
    
    @property
    def speech(self) -> Speech:
        """Access speech synthesis API."""
        return self._speech
    
    @property
    def transcriptions(self) -> Transcriptions:
        """Access speech-to-text transcription API."""
        return self._transcriptions
    
    @property
    def translations(self) -> Translations:
        """Access speech translation API."""
        return self._translations
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
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
                elif content:
                    headers = self._get_headers()
                    headers["Content-Type"] = "application/json"
                    response = self._client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        content=content,
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
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()


class Speech:
    """Text-to-speech synthesis."""
    
    def __init__(self, audio: Audio):
        self._audio = audio
    
    def create(
        self,
        model: str,
        voice: str,
        input: str,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        **kwargs: Any
    ) -> bytes:
        """
        Generate speech from text.
        
        Args:
            model: TTS model ("tts-1", "tts-1-hd")
            voice: Voice to use ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
            input: Text to convert to speech (max 4096 characters)
            response_format: Audio format ("mp3", "opus", "aac", "flac", "wav", "pcm")
            speed: Speech speed (0.25 to 4.0)
            instructions: Additional instructions for the voice
            **kwargs: Additional parameters
            
        Returns:
            Audio content as bytes
        """
        url = f"{self._audio.config.base_url}/audio/speech"
        
        data = {
            "model": model,
            "voice": voice,
            "input": input,
        }
        if response_format:
            data["response_format"] = response_format
        if speed is not None:
            data["speed"] = speed
        if instructions:
            data["instructions"] = instructions
        
        response = self._audio._request_with_retry("POST", url, json_data=data)
        
        if response.status_code != 200:
            self._audio._handle_error(response)
        
        return response.content
    
    def create_streaming(
        self,
        model: str,
        voice: str,
        input: str,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        chunk_size: int = 8192,
        **kwargs: Any
    ) -> Iterator[bytes]:
        """
        Generate speech from text with streaming response.
        
        Args:
            model: TTS model
            voice: Voice to use
            input: Text to convert
            response_format: Audio format
            speed: Speech speed
            instructions: Additional instructions
            chunk_size: Size of each chunk in bytes
            **kwargs: Additional parameters
            
        Yields:
            Audio content chunks as bytes
        """
        url = f"{self._audio.config.base_url}/audio/speech"
        
        data = {
            "model": model,
            "voice": voice,
            "input": input,
        }
        if response_format:
            data["response_format"] = response_format
        if speed is not None:
            data["speed"] = speed
        if instructions:
            data["instructions"] = instructions
        
        with self._audio._client.stream(
            "POST",
            url,
            headers=self._audio._get_headers(),
            json=data,
        ) as response:
            if response.status_code != 200:
                self._audio._handle_error(response)
            
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                yield chunk


class Transcriptions:
    """Speech-to-text transcription."""
    
    def __init__(self, audio: Audio):
        self._audio = audio
    
    def create(
        self,
        file: Union[str, Path, BinaryIO, bytes],
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """
        Transcribe audio to text.
        
        Args:
            file: Audio file to transcribe
            model: Model to use ("whisper-1", "gpt-4o-transcribe", etc.)
            language: Language code (e.g., "en", "zh")
            prompt: Optional prompt to guide transcription
            response_format: Output format ("json", "text", "srt", "verbose_json", "vtt")
            temperature: Sampling temperature (0 to 1)
            timestamp_granularities: Timestamp detail level ("word", "segment")
            **kwargs: Additional parameters
            
        Returns:
            Transcription text or detailed response dict depending on format
        """
        url = f"{self._audio.config.base_url}/audio/transcriptions"
        
        files = {
            "file": (self._audio._get_filename(file), self._audio._read_file(file)),
        }
        
        data: Dict[str, Any] = {"model": model}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if response_format:
            data["response_format"] = response_format
        if temperature is not None:
            data["temperature"] = temperature
        if timestamp_granularities:
            for i, granularity in enumerate(timestamp_granularities):
                data[f"timestamp_granularities[{i}]"] = granularity
        
        response = self._audio._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._audio._handle_error(response)
        
        if response_format in ("text", "srt", "vtt"):
            return response.text
        
        result = response.json()
        if response_format == "verbose_json":
            return result
        return CreateTranscriptionResponse(text=result.get("text", ""))


class Translations:
    """Speech translation to English."""
    
    def __init__(self, audio: Audio):
        self._audio = audio
    
    def create(
        self,
        file: Union[str, Path, BinaryIO, bytes],
        model: str,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """
        Translate audio to English text.
        
        Args:
            file: Audio file to translate
            model: Model to use
            prompt: Optional prompt to guide translation
            response_format: Output format
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Translation text or detailed response dict
        """
        url = f"{self._audio.config.base_url}/audio/translations"
        
        files = {
            "file": (self._audio._get_filename(file), self._audio._read_file(file)),
        }
        
        data: Dict[str, Any] = {"model": model}
        if prompt:
            data["prompt"] = prompt
        if response_format:
            data["response_format"] = response_format
        if temperature is not None:
            data["temperature"] = temperature
        
        response = self._audio._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._audio._handle_error(response)
        
        if response_format in ("text", "srt", "vtt"):
            return response.text
        
        result = response.json()
        return CreateTranslationResponse(text=result.get("text", ""))


class AsyncAudio(BaseAudio):
    """
    Asynchronous Audio API client.
    
    Example:
        >>> client = AsyncAudio(config)
        >>> # Text to speech
        >>> response = await client.speech.create(
        ...     model="tts-1",
        ...     voice="alloy",
        ...     input="Hello, world!"
        ... )
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        self._client: Optional[httpx.AsyncClient] = None
        self._config = config
        self._speech = AsyncSpeech(self)
        self._transcriptions = AsyncTranscriptions(self)
        self._translations = AsyncTranslations(self)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout * 5,
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
        content: Optional[bytes] = None,
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
                elif content:
                    headers = self._get_headers()
                    headers["Content-Type"] = "application/json"
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        content=content,
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
    
    @property
    def speech(self) -> AsyncSpeech:
        """Access async speech synthesis API."""
        return self._speech
    
    @property
    def transcriptions(self) -> AsyncTranscriptions:
        """Access async speech-to-text transcription API."""
        return self._transcriptions
    
    @property
    def translations(self) -> AsyncTranslations:
        """Access async speech translation API."""
        return self._translations
    
    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class AsyncSpeech:
    """Async text-to-speech synthesis."""
    
    def __init__(self, audio: AsyncAudio):
        self._audio = audio
    
    async def create(
        self,
        model: str,
        voice: str,
        input: str,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        **kwargs: Any
    ) -> bytes:
        """Generate speech from text asynchronously."""
        url = f"{self._audio.config.base_url}/audio/speech"
        
        data = {
            "model": model,
            "voice": voice,
            "input": input,
        }
        if response_format:
            data["response_format"] = response_format
        if speed is not None:
            data["speed"] = speed
        if instructions:
            data["instructions"] = instructions
        
        response = await self._audio._request_with_retry("POST", url, json_data=data)
        
        if response.status_code != 200:
            self._audio._handle_error(response)
        
        return response.content
    
    async def create_streaming(
        self,
        model: str,
        voice: str,
        input: str,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        chunk_size: int = 8192,
        **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """Generate speech from text with async streaming."""
        url = f"{self._audio.config.base_url}/audio/speech"
        
        data = {
            "model": model,
            "voice": voice,
            "input": input,
        }
        if response_format:
            data["response_format"] = response_format
        if speed is not None:
            data["speed"] = speed
        if instructions:
            data["instructions"] = instructions
        
        client = await self._audio._get_client()
        async with client.stream(
            "POST",
            url,
            headers=self._audio._get_headers(),
            json=data,
        ) as response:
            if response.status_code != 200:
                self._audio._handle_error(response)
            
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                yield chunk


class AsyncTranscriptions:
    """Async speech-to-text transcription."""
    
    def __init__(self, audio: AsyncAudio):
        self._audio = audio
    
    async def create(
        self,
        file: Union[str, Path, BinaryIO, bytes],
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Transcribe audio to text asynchronously."""
        url = f"{self._audio.config.base_url}/audio/transcriptions"
        
        files = {
            "file": (self._audio._get_filename(file), self._audio._read_file(file)),
        }
        
        data: Dict[str, Any] = {"model": model}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if response_format:
            data["response_format"] = response_format
        if temperature is not None:
            data["temperature"] = temperature
        if timestamp_granularities:
            for i, granularity in enumerate(timestamp_granularities):
                data[f"timestamp_granularities[{i}]"] = granularity
        
        response = await self._audio._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._audio._handle_error(response)
        
        if response_format in ("text", "srt", "vtt"):
            return response.text
        
        result = response.json()
        if response_format == "verbose_json":
            return result
        return CreateTranscriptionResponse(text=result.get("text", ""))


class AsyncTranslations:
    """Async speech translation to English."""
    
    def __init__(self, audio: AsyncAudio):
        self._audio = audio
    
    async def create(
        self,
        file: Union[str, Path, BinaryIO, bytes],
        model: str,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Translate audio to English text asynchronously."""
        url = f"{self._audio.config.base_url}/audio/translations"
        
        files = {
            "file": (self._audio._get_filename(file), self._audio._read_file(file)),
        }
        
        data: Dict[str, Any] = {"model": model}
        if prompt:
            data["prompt"] = prompt
        if response_format:
            data["response_format"] = response_format
        if temperature is not None:
            data["temperature"] = temperature
        
        response = await self._audio._request_with_retry("POST", url, files=files, data=data)
        
        if response.status_code != 200:
            self._audio._handle_error(response)
        
        if response_format in ("text", "srt", "vtt"):
            return response.text
        
        result = response.json()
        return CreateTranslationResponse(text=result.get("text", ""))
