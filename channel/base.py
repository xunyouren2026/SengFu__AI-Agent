"""
AGI Unified Framework - Channel Base Module

This module defines the abstract base class for all channel adapters and provides
the core interfaces and types for IM channel integration.

Key Components:
- ChannelAdapter: Abstract base class for all channel implementations
- ChannelCapability: Enum for channel capabilities
- ConnectionState: Enum for connection states
- SendResult/ReceiveResult: Result types for send/receive operations
- RetryConfig: Configuration for retry behavior

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    TYPE_CHECKING,
)

# Type variables for generic typing
T = TypeVar("T")
U = TypeVar("U")

if TYPE_CHECKING:
    from .universal_message import UniversalMessage, MessageMetadata
    from .session_manager import Session, SessionContext

logger = logging.getLogger(__name__)


class ChannelCapability(Enum):
    """
    Enum representing the capabilities of a channel adapter.
    
    Each capability indicates what features the channel supports.
    """
    TEXT_MESSAGES = auto()
    """Support for plain text messages"""
    HTML_MESSAGES = auto()
    """Support for HTML formatted messages"""
    MARKDOWN_MESSAGES = auto()
    """Support for Markdown formatted messages"""
    MEDIA_MESSAGES = auto()
    """Support for media messages (images, videos, audio)"""
    FILE_ATTACHMENTS = auto()
    """Support for file attachments"""
    EMOJI_SUPPORT = auto()
    """Support for emoji reactions"""
    THREADING = auto()
    """Support for threaded conversations"""
    CHANNEL_TYPES = auto()
    """Support for different channel types (public/private)"""
    DIRECT_MESSAGES = auto()
    """Support for direct/private messages"""
    GROUPS = auto()
    """Support for group chats"""
    WEBHOOK_MODE = auto()
    """Support for webhook-based event receiving"""
    POLLING_MODE = auto()
    """Support for polling-based event receiving"""
    EDIT_MESSAGES = auto()
    """Support for editing existing messages"""
    DELETE_MESSAGES = auto()
    """Support for deleting messages"""
    PUSH_NOTIFICATIONS = auto()
    """Support for push notifications"""
    TYPING_INDICATOR = auto()
    """Support for typing indicators"""
    READ_RECEIPTS = auto()
    """Support for read receipts"""
    ENCRYPTION = auto()
    """Support for end-to-end encryption"""
    BOT_COMMANDS = auto()
    """Support for bot commands"""
    SLASH_COMMANDS = auto()
    """Support for slash commands"""
    BUTTONS = auto()
    """Support for inline keyboard buttons"""
    MODALS = auto()
    """Support for modal dialogs"""
    INTERACTIVE_MESSAGES = auto()
    """Support for interactive message components"""
    CHANNEL_INFO = auto()
    """Support for retrieving channel information"""
    USER_INFO = auto()
    """Support for retrieving user information"""
    MEMBER_INFO = auto()
    """Support for retrieving member information"""
    RATE_LIMITING = auto()
    """Support for rate limiting"""
    AUTO_RECONNECT = auto()
    """Support for automatic reconnection"""
    MULTI_TENANT = auto()
    """Support for multi-tenant configurations"""


class ConnectionState(Enum):
    """
    Enum representing the state of a channel connection.
    """
    DISCONNECTED = auto()
    """The channel is not connected"""
    CONNECTING = auto()
    """The channel is in the process of connecting"""
    CONNECTED = auto()
    """The channel is connected and ready"""
    AUTHENTICATING = auto()
    """The channel is authenticating"""
    AUTHENTICATED = auto()
    """The channel is authenticated"""
    RECONNECTING = auto()
    """The channel is reconnecting"""
    ERROR = auto()
    """The channel is in an error state"""
    RATE_LIMITED = auto()
    """The channel is rate limited"""
    MAINTENANCE = auto()
    """The channel is under maintenance"""


class MessagePriority(Enum):
    """
    Enum representing the priority level for message delivery.
    """
    LOW = 0
    """Low priority - best effort delivery"""
    NORMAL = 1
    """Normal priority - standard delivery"""
    HIGH = 2
    """High priority - expedited delivery"""
    URGENT = 3
    """Urgent priority - immediate delivery"""
    CRITICAL = 4
    """Critical priority - guaranteed delivery"""


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior in channel operations.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retryable_errors: List of error types that should trigger a retry
        retryable_status_codes: HTTP status codes that should trigger a retry
        timeout: Overall timeout for the operation in seconds
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: List[str] = field(default_factory=lambda: [
        "ConnectionError",
        "TimeoutError",
        "TemporaryFailure",
        "RateLimitError",
    ])
    retryable_status_codes: List[int] = field(default_factory=lambda: [
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    ])
    timeout: float = 30.0
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a given retry attempt.
        
        Args:
            attempt: The current retry attempt number (0-indexed)
            
        Returns:
            The delay in seconds before the next retry
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            import random
            delay *= 0.5 + random.random()
        return delay
    
    def should_retry(self, error: Exception, status_code: Optional[int] = None) -> bool:
        """
        Determine if an operation should be retried based on the error.
        
        Args:
            error: The exception that occurred
            status_code: Optional HTTP status code
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        if status_code and status_code in self.retryable_status_codes:
            return True
        
        error_type = type(error).__name__
        return error_type in self.retryable_errors


@dataclass
class SendResult:
    """
    Result of a message send operation.
    
    Attributes:
        success: Whether the send operation was successful
        message_id: Platform-specific message ID if successful
        timestamp: Timestamp when the message was sent
        error: Error message if the operation failed
        error_code: Platform-specific error code
        retry_count: Number of retries performed
        metadata: Additional metadata about the send operation
    """
    success: bool
    message_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "success": self.success,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "error": self.error,
            "error_code": self.error_code,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class ReceiveResult:
    """
    Result of a message receive operation.
    
    Attributes:
        success: Whether the receive operation was successful
        messages: List of received messages
        raw_payload: Raw platform-specific payload
        webhook_id: Webhook ID if received via webhook
        error: Error message if the operation failed
        metadata: Additional metadata about the receive operation
    """
    success: bool
    messages: List["UniversalMessage"] = field(default_factory=list)
    raw_payload: Optional[Dict[str, Any]] = None
    webhook_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "success": self.success,
            "message_count": len(self.messages),
            "raw_payload": self.raw_payload,
            "webhook_id": self.webhook_id,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ChannelConfig:
    """
    Base configuration for a channel adapter.
    
    This class provides common configuration options that all channel
    adapters should support.
    
    Attributes:
        channel_id: Unique identifier for this channel
        name: Human-readable name for the channel
        enabled: Whether the channel is enabled
        timeout: Default timeout for operations in seconds
        retry_config: Configuration for retry behavior
        webhook_url: URL for receiving webhooks (if applicable)
        polling_interval: Interval for polling in seconds (if applicable)
        max_message_length: Maximum message length allowed
        supported_content_types: List of supported content types
        custom_headers: Custom HTTP headers to include in requests
        proxy_config: Proxy configuration
        rate_limit: Rate limit configuration
    """
    channel_id: str
    name: str = ""
    enabled: bool = True
    timeout: float = 30.0
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    webhook_url: Optional[str] = None
    polling_interval: float = 1.0
    max_message_length: int = 4096
    supported_content_types: List[str] = field(default_factory=list)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    proxy_config: Optional[Dict[str, str]] = None
    rate_limit: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.name:
            self.name = self.channel_id


class ChannelAdapter(ABC):
    """
    Abstract base class for all IM channel adapters.
    
    This class defines the interface that all channel adapters must implement.
    It provides default implementations for common functionality and defines
    abstract methods that must be overridden by specific channel implementations.
    
    Type Parameters:
        T: The type of channel-specific configuration
        U: The type of channel-specific message format
        
    Example:
        ```python
        class MyChannelAdapter(ChannelAdapter[MyChannelConfig, MyMessage]):
            def __init__(self, config: MyChannelConfig):
                super().__init__(config)
                
            async def _send_impl(self, message: UniversalMessage) -> SendResult:
                # Implementation here
                pass
        ```
    """
    
    def __init__(self, config: ChannelConfig) -> None:
        """
        Initialize the channel adapter.
        
        Args:
            config: Configuration for the channel adapter
        """
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._capabilities: set[ChannelCapability] = set()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._message_transformers: List[Callable] = []
        self._middlewares: List[Callable] = []
        self._connected_at: Optional[float] = None
        self._last_activity: Optional[float] = None
        self._error_count: int = 0
        self._total_messages_sent: int = 0
        self._total_messages_received: int = 0
        
        self._initialize_capabilities()
        self._register_default_handlers()
    
    @abstractmethod
    def _initialize_capabilities(self) -> None:
        """
        Initialize the capabilities supported by this channel.
        
        This method should be overridden by subclasses to set up the
        set of ChannelCapability values that the channel supports.
        """
        pass
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        self._event_handlers["message"] = []
        self._event_handlers["error"] = []
        self._event_handlers["state_change"] = []
        self._event_handlers["connect"] = []
        self._event_handlers["disconnect"] = []
        self._event_handlers["rate_limit"] = []
    
    @property
    def config(self) -> ChannelConfig:
        """Get the channel configuration."""
        return self._config
    
    @property
    def state(self) -> ConnectionState:
        """Get the current connection state."""
        return self._state
    
    @property
    def capabilities(self) -> set[ChannelCapability]:
        """Get the set of capabilities supported by this channel."""
        return self._capabilities.copy()
    
    @property
    def is_connected(self) -> bool:
        """Check if the channel is currently connected."""
        return self._state == ConnectionState.CONNECTED
    
    @property
    def connected_at(self) -> Optional[float]:
        """Get the timestamp when the channel was last connected."""
        return self._connected_at
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get statistics about the channel."""
        return {
            "channel_id": self._config.channel_id,
            "state": self._state.name,
            "connected_at": self._connected_at,
            "last_activity": self._last_activity,
            "error_count": self._error_count,
            "total_messages_sent": self._total_messages_sent,
            "total_messages_received": self._total_messages_received,
        }
    
    def has_capability(self, capability: ChannelCapability) -> bool:
        """
        Check if the channel supports a specific capability.
        
        Args:
            capability: The capability to check
            
        Returns:
            True if the capability is supported, False otherwise
        """
        return capability in self._capabilities
    
    def supports_capabilities(self, capabilities: List[ChannelCapability]) -> bool:
        """
        Check if the channel supports all specified capabilities.
        
        Args:
            capabilities: List of capabilities to check
            
        Returns:
            True if all capabilities are supported, False otherwise
        """
        return all(cap in self._capabilities for cap in capabilities)
    
    def add_event_handler(self, event: str, handler: Callable) -> None:
        """
        Add an event handler for a specific event.
        
        Args:
            event: The event name (e.g., "message", "error", "state_change")
            handler: The handler function to call when the event occurs
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def remove_event_handler(self, event: str, handler: Callable) -> None:
        """
        Remove an event handler for a specific event.
        
        Args:
            event: The event name
            handler: The handler function to remove
        """
        if event in self._event_handlers and handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)
    
    async def _emit_event(self, event: str, *args: Any, **kwargs: Any) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event: The event name
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                self._logger.error(f"Error in event handler for {event}: {e}")
    
    def add_message_transformer(self, transformer: Callable) -> None:
        """
        Add a message transformer to the processing pipeline.
        
        Transformers are called in order before sending messages.
        
        Args:
            transformer: A callable that takes a UniversalMessage and returns
                        a transformed UniversalMessage
        """
        self._message_transformers.append(transformer)
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        Add a middleware to the processing pipeline.
        
        Middlewares are called around each operation and can modify
        behavior or add logging/tracing.
        
        Args:
            middleware: A callable that takes the next handler and returns
                       a wrapped handler
        """
        self._middlewares.append(middleware)
    
    async def _apply_transformers(self, message: "UniversalMessage") -> "UniversalMessage":
        """
        Apply all registered transformers to a message.
        
        Args:
            message: The message to transform
            
        Returns:
            The transformed message
        """
        result = message
        for transformer in self._message_transformers:
            if asyncio.iscoroutinefunction(transformer):
                result = await transformer(result)
            else:
                result = transformer(result)
        return result
    
    async def _apply_middlewares(
        self,
        operation: str,
        handler: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Apply all registered middlewares to an operation.
        
        Args:
            operation: The name of the operation being performed
            handler: The handler function to wrap
            *args: Positional arguments to pass to the handler
            **kwargs: Keyword arguments to pass to the handler
            
        Returns:
            The result of the operation
        """
        result = handler
        for middleware in reversed(self._middlewares):
            if asyncio.iscoroutinefunction(middleware):
                result = await middleware(operation, result)
            else:
                result = middleware(operation, result)
        
        if asyncio.iscoroutinefunction(result):
            return await result(*args, **kwargs)
        return result(*args, **kwargs)
    
    async def connect(self) -> bool:
        """
        Establish a connection to the channel.
        
        This method should be overridden by subclasses to implement
        the specific connection logic for each channel.
        
        Returns:
            True if the connection was successful, False otherwise
        """
        try:
            self._set_state(ConnectionState.CONNECTING)
            
            # Call the abstract implementation
            result = await self._connect_impl()
            
            if result:
                self._set_state(ConnectionState.CONNECTED)
                self._connected_at = time.time()
                self._last_activity = self._connected_at
                await self._emit_event("connect")
                self._logger.info(f"Connected to channel {self._config.channel_id}")
            else:
                self._set_state(ConnectionState.ERROR)
                
            return result
            
        except Exception as e:
            self._logger.error(f"Failed to connect to channel: {e}")
            self._set_state(ConnectionState.ERROR)
            self._error_count += 1
            return False
    
    async def disconnect(self) -> None:
        """
        Disconnect from the channel.
        
        This method should be overridden by subclasses to implement
        the specific disconnection logic for each channel.
        """
        try:
            await self._disconnect_impl()
            self._set_state(ConnectionState.DISCONNECTED)
            await self._emit_event("disconnect")
            self._logger.info(f"Disconnected from channel {self._config.channel_id}")
        except Exception as e:
            self._logger.error(f"Error during disconnect: {e}")
            self._set_state(ConnectionState.ERROR)
    
    async def reconnect(self, max_attempts: int = 5) -> bool:
        """
        Attempt to reconnect to the channel.
        
        Args:
            max_attempts: Maximum number of reconnection attempts
            
        Returns:
            True if reconnection was successful, False otherwise
        """
        self._set_state(ConnectionState.RECONNECTING)
        
        for attempt in range(max_attempts):
            try:
                self._logger.info(
                    f"Reconnection attempt {attempt + 1}/{max_attempts} "
                    f"for channel {self._config.channel_id}"
                )
                
                await self.disconnect()
                await asyncio.sleep(self._config.retry_config.get_delay(attempt))
                
                if await self.connect():
                    return True
                    
            except Exception as e:
                self._logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        self._set_state(ConnectionState.ERROR)
        return False
    
    def _set_state(self, new_state: ConnectionState) -> None:
        """
        Set the connection state and emit a state change event.
        
        Args:
            new_state: The new connection state
        """
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._logger.debug(
                f"State change for {self._config.channel_id}: "
                f"{old_state.name} -> {new_state.name}"
            )
            asyncio.create_task(
                self._emit_event("state_change", old_state, new_state)
            )
    
    async def send(
        self,
        message: "UniversalMessage",
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> SendResult:
        """
        Send a message through the channel.
        
        This method applies transformers, handles retries, and calls the
        abstract _send_impl method to perform the actual send operation.
        
        Args:
            message: The message to send
            priority: The priority level for the message
            
        Returns:
            SendResult indicating the outcome of the send operation
        """
        if not self.is_connected:
            return SendResult(
                success=False,
                error="Channel is not connected",
                error_code="NOT_CONNECTED",
            )
        
        try:
            # Apply message transformers
            transformed_message = await self._apply_transformers(message)
            
            # Check message length
            if len(transformed_message.content.text or "") > self._config.max_message_length:
                if not self.has_capability(ChannelCapability.FILE_ATTACHMENTS):
                    return SendResult(
                        success=False,
                        error=f"Message exceeds maximum length of {self._config.max_message_length}",
                        error_code="MESSAGE_TOO_LONG",
                    )
            
            # Attempt to send with retries
            retry_config = self._config.retry_config
            last_error = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        self._send_impl(transformed_message, priority),
                        timeout=retry_config.timeout,
                    )
                    
                    if result.success:
                        self._total_messages_sent += 1
                        self._last_activity = time.time()
                        return result
                    
                    # Non-retryable error
                    if not retry_config.should_retry(
                        Exception(result.error or ""),
                        result.metadata.get("status_code")
                    ):
                        return result
                    
                    last_error = result.error
                    
                except asyncio.TimeoutError:
                    last_error = "Operation timed out"
                except Exception as e:
                    last_error = str(e)
                    
                    if not retry_config.should_retry(e):
                        return SendResult(
                            success=False,
                            error=str(e),
                            error_code=type(e).__name__,
                        )
                
                # Retry if we have attempts left
                if attempt < retry_config.max_retries:
                    delay = retry_config.get_delay(attempt)
                    self._logger.warning(
                        f"Send attempt {attempt + 1} failed for "
                        f"{self._config.channel_id}: {last_error}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
            
            return SendResult(
                success=False,
                error=last_error or "Max retries exceeded",
                error_code="MAX_RETRIES_EXCEEDED",
                retry_count=retry_config.max_retries,
            )
            
        except Exception as e:
            self._logger.error(f"Send error for {self._config.channel_id}: {e}")
            self._error_count += 1
            return SendResult(
                success=False,
                error=str(e),
                error_code=type(e).__name__,
            )
    
    async def receive(
        self,
        payload: Optional[Dict[str, Any]] = None,
    ) -> ReceiveResult:
        """
        Receive messages from the channel.
        
        This method processes incoming payloads and converts them to
        UniversalMessage format.
        
        Args:
            payload: Optional raw payload to process (for webhook handlers)
            
        Returns:
            ReceiveResult containing the processed messages
        """
        try:
            result = await self._receive_impl(payload)
            
            if result.success:
                self._total_messages_received += len(result.messages)
                self._last_activity = time.time()
            
            return result
            
        except Exception as e:
            self._logger.error(f"Receive error for {self._config.channel_id}: {e}")
            self._error_count += 1
            return ReceiveResult(
                success=False,
                error=str(e),
            )
    
    async def start_webhook_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        path: str = "/webhook",
    ) -> None:
        """
        Start a webhook server to receive events.
        
        Provides a default implementation using asyncio that listens
        for incoming HTTP POST requests on the specified path and
        dispatches them to the channel's message handler.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            path: URL path for the webhook endpoint
        """
        self._logger.info(
            "Starting webhook server for channel %s on %s:%d%s",
            self._config.channel_id, host, port, path,
        )
        try:
            import aiohttp
            from aiohttp import web

            async def _webhook_handler(request: web.Request) -> web.Response:
                try:
                    payload = await request.json()
                    self._logger.debug(
                        "Webhook received on %s: %s", path, payload
                    )
                    await self._handle_webhook_payload(payload)
                    return web.json_response({"status": "ok"})
                except Exception as exc:
                    self._logger.error("Webhook handler error: %s", exc)
                    return web.json_response(
                        {"status": "error", "message": str(exc)}, status=500
                    )

            app = web.Application()
            app.router.add_post(path, _webhook_handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            self._webhook_runner = runner
            self._logger.info(
                "Webhook server for channel %s is running on %s:%d",
                self._config.channel_id, host, port,
            )
        except ImportError:
            self._logger.warning(
                "aiohttp is not installed; webhook server for channel %s "
                "cannot start. Install with: pip install aiohttp",
                self._config.channel_id,
            )
        except Exception as e:
            self._logger.error(
                "Failed to start webhook server for channel %s: %s",
                self._config.channel_id, e,
            )
    
    async def _handle_webhook_payload(self, payload: Dict[str, Any]) -> None:
        """Handle an incoming webhook payload. Subclasses may override."""
        self._logger.debug("Handling webhook payload: %s", payload)
    
    async def start_polling(self) -> None:
        """
        Start polling for new messages.
        
        Provides a default polling implementation that periodically
        calls the abstract ``_poll_once`` method at a configurable
        interval. Subclasses should override ``_poll_once`` to define
        the actual polling logic.
        """
        self._logger.info("Starting polling for channel %s", self._config.channel_id)
        self._polling_active = True
        poll_interval = getattr(self._config, "poll_interval", 5.0)
        try:
            while self._polling_active:
                try:
                    await self._poll_once()
                except Exception as exc:
                    self._logger.error("Polling error for channel %s: %s", self._config.channel_id, exc)
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            self._logger.info("Polling cancelled for channel %s", self._config.channel_id)
        finally:
            self._polling_active = False
    
    async def _poll_once(self) -> None:
        """Execute a single polling cycle. Override in subclasses."""
        pass
    
    def stop_polling(self) -> None:
        """
        Stop polling for new messages.
        """
        self._logger.debug(f"Stopping polling for {self._config.channel_id}")
    
    @abstractmethod
    async def _connect_impl(self) -> bool:
        """
        Implementation of the connect method.
        
        This method should be overridden by subclasses to implement
        the specific connection logic.
        
        Returns:
            True if the connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def _disconnect_impl(self) -> None:
        """
        Implementation of the disconnect method.
        
        This method should be overridden by subclasses to implement
        the specific disconnection logic.
        """
        pass
    
    @abstractmethod
    async def _send_impl(
        self,
        message: "UniversalMessage",
        priority: MessagePriority,
    ) -> SendResult:
        """
        Implementation of the send method.
        
        This method should be overridden by subclasses to implement
        the specific message sending logic.
        
        Args:
            message: The message to send
            priority: The priority level for the message
            
        Returns:
            SendResult indicating the outcome of the send operation
        """
        pass
    
    @abstractmethod
    async def _receive_impl(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> ReceiveResult:
        """
        Implementation of the receive method.
        
        This method should be overridden by subclasses to implement
        the specific message receiving logic.
        
        Args:
            payload: Optional raw payload to process
            
        Returns:
            ReceiveResult containing the processed messages
        """
        pass
    
    @abstractmethod
    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a user.
        
        Args:
            user_id: The platform-specific user ID
            
        Returns:
            A dictionary containing user information, or None if not found
        """
        pass
    
    @abstractmethod
    async def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a channel.
        
        Args:
            channel_id: The platform-specific channel ID
            
        Returns:
            A dictionary containing channel information, or None if not found
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the channel.
        
        Returns:
            True if the channel is healthy, False otherwise
        """
        try:
            if not self.is_connected:
                return False
            
            # Call the abstract implementation for channel-specific health check
            result = await self._health_check_impl()
            
            if not result:
                self._set_state(ConnectionState.ERROR)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Health check failed for {self._config.channel_id}: {e}")
            return False
    
    async def _health_check_impl(self) -> bool:
        """
        Implementation of the health check method.
        
        This method should be overridden by subclasses to implement
        channel-specific health checks.
        
        Returns:
            True if the channel is healthy, False otherwise
        """
        # Default implementation - just check connection state
        return self.is_connected
    
    def __repr__(self) -> str:
        """Return a string representation of the adapter."""
        return (
            f"{self.__class__.__name__}("
            f"channel_id={self._config.channel_id!r}, "
            f"state={self._state.name}, "
            f"capabilities={len(self._capabilities)})"
        )
