"""
AGI Unified Framework - Channel Gateway Module

This module provides the unified gateway for managing multiple channel adapters.
It serves as the central entry point for all messaging operations.

Key Components:
- ChannelGateway: Main gateway class for multi-channel management
- GatewayConfig: Configuration for the gateway
- GatewayEvent: Event types for gateway events

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .base import ChannelAdapter, ChannelConfig, SendResult, ReceiveResult
    from .universal_message import UniversalMessage
    from .router import MessageRouter
    from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class GatewayEventType(Enum):
    """Enum for gateway event types."""
    CHANNEL_REGISTERED = auto()
    CHANNEL_UNREGISTERED = auto()
    CHANNEL_CONNECTED = auto()
    CHANNEL_DISCONNECTED = auto()
    CHANNEL_ERROR = auto()
    MESSAGE_SENT = auto()
    MESSAGE_RECEIVED = auto()
    MESSAGE_ROUTED = auto()
    SESSION_STARTED = auto()
    SESSION_ENDED = auto()
    RATE_LIMIT_EXCEEDED = auto()
    HEALTH_CHECK_FAILED = auto()


@dataclass
class GatewayEvent:
    """
    Data class representing a gateway event.
    
    Attributes:
        event_type: Type of the event
        timestamp: Timestamp when the event occurred
        channel_id: Channel ID if applicable
        data: Event-specific data
        error: Error information if applicable
    """
    event_type: GatewayEventType
    timestamp: float = field(default_factory=time.time)
    channel_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "channel_id": self.channel_id,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class GatewayConfig:
    """
    Configuration for the channel gateway.
    
    Attributes:
        gateway_id: Unique identifier for this gateway
        name: Human-readable name
        enable_routing: Whether to enable message routing
        enable_sessions: Whether to enable session management
        enable_metrics: Whether to enable metrics collection
        enable_health_checks: Whether to enable health checks
        health_check_interval: Interval between health checks in seconds
        default_timeout: Default timeout for operations
        max_retry_attempts: Maximum retry attempts for failed operations
        retry_delay: Delay between retries in seconds
        concurrent_limit: Maximum concurrent operations
        webhook_config: Webhook server configuration
        storage_backend: Backend for state persistence
        event_queue_size: Size of the event queue
    """
    gateway_id: str
    name: str = "Channel Gateway"
    enable_routing: bool = True
    enable_sessions: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval: float = 60.0
    default_timeout: float = 30.0
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    concurrent_limit: int = 100
    webhook_config: Optional[Dict[str, Any]] = None
    storage_backend: Optional[str] = None
    event_queue_size: int = 1000


class ChannelRegistry:
    """
    Registry for managing channel adapters.
    
    This class provides thread-safe registration, lookup, and management
    of channel adapters within the gateway.
    """
    
    def __init__(self):
        """Initialize the channel registry."""
        self._channels: Dict[str, "ChannelAdapter"] = {}
        self._channel_types: Dict[str, Type["ChannelAdapter"]] = {}
        self._channel_aliases: Dict[str, str] = {}
        self._lock = asyncio.Lock()
    
    async def register(
        self,
        channel_id: str,
        adapter: "ChannelAdapter",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a channel adapter.
        
        Args:
            channel_id: Unique identifier for the channel
            adapter: The channel adapter instance
            aliases: Optional list of aliases for the channel
        """
        async with self._lock:
            if channel_id in self._channels:
                raise ValueError(f"Channel {channel_id} is already registered")
            
            self._channels[channel_id] = adapter
            self._channel_types[channel_id] = type(adapter)
            
            if aliases:
                for alias in aliases:
                    self._channel_aliases[alias] = channel_id
            
            logger.info(f"Registered channel: {channel_id}")
    
    async def unregister(self, channel_id: str) -> bool:
        """
        Unregister a channel adapter.
        
        Args:
            channel_id: The channel ID to unregister
            
        Returns:
            True if the channel was unregistered, False if not found
        """
        async with self._lock:
            if channel_id not in self._channels:
                return False
            
            # Remove aliases
            aliases_to_remove = [
                alias for alias, cid in self._channel_aliases.items()
                if cid == channel_id
            ]
            for alias in aliases_to_remove:
                del self._channel_aliases[alias]
            
            del self._channels[channel_id]
            del self._channel_types[channel_id]
            
            logger.info(f"Unregistered channel: {channel_id}")
            return True
    
    def get(self, channel_id: str) -> Optional["ChannelAdapter"]:
        """
        Get a channel adapter by ID or alias.
        
        Args:
            channel_id: Channel ID or alias
            
        Returns:
            The channel adapter, or None if not found
        """
        # Check if it's an alias
        actual_id = self._channel_aliases.get(channel_id, channel_id)
        return self._channels.get(actual_id)
    
    def get_all(self) -> List["ChannelAdapter"]:
        """Get all registered channel adapters."""
        return list(self._channels.values())
    
    def get_by_type(self, adapter_type: Type["ChannelAdapter"]) -> List["ChannelAdapter"]:
        """
        Get all channels of a specific type.
        
        Args:
            adapter_type: The type of adapter to filter by
            
        Returns:
            List of matching channel adapters
        """
        return [
            adapter for adapter in self._channels.values()
            if isinstance(adapter, adapter_type)
        ]
    
    def list_channel_ids(self) -> List[str]:
        """Get a list of all registered channel IDs."""
        return list(self._channels.keys())
    
    def __len__(self) -> int:
        """Get the number of registered channels."""
        return len(self._channels)
    
    def __contains__(self, channel_id: str) -> bool:
        """Check if a channel is registered."""
        actual_id = self._channel_aliases.get(channel_id, channel_id)
        return actual_id in self._channels


class ChannelGateway:
    """
    Unified gateway for managing multiple IM channel adapters.
    
    This class serves as the central entry point for all messaging operations,
    providing a unified interface for sending/receiving messages across
    multiple channels.
    
    Features:
    - Multi-channel management
    - Unified message routing
    - Session management integration
    - Metrics collection
    - Health monitoring
    - Event handling
    
    Example:
        ```python
        # Initialize gateway
        gateway = ChannelGateway(GatewayConfig(gateway_id="main"))
        
        # Register channels
        await gateway.register_channel("telegram", TelegramAdapter(config))
        await gateway.register_channel("discord", DiscordAdapter(config))
        
        # Send message to specific channel
        result = await gateway.send_to_channel(
            "telegram",
            UniversalMessage(content=MessageContent.from_text("Hello!"))
        )
        
        # Broadcast to all channels
        results = await gateway.broadcast(
            UniversalMessage(content=MessageContent.from_text("Hello all!"))
        )
        ```
    """
    
    def __init__(self, config: GatewayConfig) -> None:
        """
        Initialize the channel gateway.
        
        Args:
            config: Gateway configuration
        """
        self._config = config
        self._registry = ChannelRegistry()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Component references
        self._router: Optional["MessageRouter"] = None
        self._session_manager: Optional["SessionManager"] = None
        self._metrics_collector: Optional["MetricsCollector"] = None
        
        # State
        self._is_running = False
        self._start_time: Optional[float] = None
        self._event_handlers: Dict[GatewayEventType, List[Callable]] = defaultdict(list)
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.event_queue_size)
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(config.concurrent_limit)
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._event_processor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_broadcasts": 0,
            "total_errors": 0,
        }
    
    @property
    def config(self) -> GatewayConfig:
        """Get the gateway configuration."""
        return self._config
    
    @property
    def is_running(self) -> bool:
        """Check if the gateway is running."""
        return self._is_running
    
    @property
    def uptime(self) -> float:
        """Get the gateway uptime in seconds."""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            **self._stats,
            "uptime": self.uptime,
            "channel_count": len(self._registry),
            "channels": {
                channel_id: adapter.statistics
                for channel_id, adapter in [
                    (cid, self._registry.get(cid))
                    for cid in self._registry.list_channel_ids()
                ]
                if adapter is not None
            },
        }
    
    # ============= Channel Management =============
    
    async def register_channel(
        self,
        channel_id: str,
        adapter: "ChannelAdapter",
        aliases: Optional[List[str]] = None,
        auto_connect: bool = True,
    ) -> None:
        """
        Register a channel adapter with the gateway.
        
        Args:
            channel_id: Unique identifier for the channel
            adapter: The channel adapter instance
            aliases: Optional list of aliases
            auto_connect: Whether to automatically connect the channel
            
        Raises:
            ValueError: If channel_id is already registered
        """
        await self._registry.register(channel_id, adapter, aliases)
        
        # Set up event forwarding from adapter
        adapter.add_event_handler("message", self._on_channel_message)
        adapter.add_event_handler("error", self._on_channel_error)
        adapter.add_event_handler("state_change", self._on_channel_state_change)
        
        if auto_connect:
            try:
                await adapter.connect()
            except Exception as e:
                self._logger.error(f"Failed to auto-connect channel {channel_id}: {e}")
        
        await self._emit_event(GatewayEventType.CHANNEL_REGISTERED, {
            "channel_id": channel_id,
            "channel_type": type(adapter).__name__,
            "aliases": aliases,
        })
    
    async def unregister_channel(
        self,
        channel_id: str,
        disconnect: bool = True,
    ) -> bool:
        """
        Unregister a channel from the gateway.
        
        Args:
            channel_id: The channel ID to unregister
            disconnect: Whether to disconnect before unregistering
            
        Returns:
            True if the channel was unregistered, False if not found
        """
        adapter = self._registry.get(channel_id)
        if not adapter:
            return False
        
        if disconnect:
            try:
                await adapter.disconnect()
            except Exception as e:
                self._logger.error(f"Error disconnecting channel {channel_id}: {e}")
        
        result = await self._registry.unregister(channel_id)
        
        if result:
            await self._emit_event(GatewayEventType.CHANNEL_UNREGISTERED, {
                "channel_id": channel_id,
            })
        
        return result
    
    def get_channel(self, channel_id: str) -> Optional["ChannelAdapter"]:
        """
        Get a registered channel adapter.
        
        Args:
            channel_id: Channel ID or alias
            
        Returns:
            The channel adapter, or None if not found
        """
        return self._registry.get(channel_id)
    
    def list_channels(self) -> List[str]:
        """Get a list of all registered channel IDs."""
        return self._registry.list_channel_ids()
    
    def get_channel_by_type(self, channel_type: Type["ChannelAdapter"]) -> List["ChannelAdapter"]:
        """
        Get all channels of a specific type.
        
        Args:
            channel_type: The type of channel to filter by
            
        Returns:
            List of matching channel adapters
        """
        return self._registry.get_by_type(channel_type)
    
    # ============= Router Integration =============
    
    def set_router(self, router: "MessageRouter") -> None:
        """
        Set the message router for the gateway.
        
        Args:
            router: The message router instance
        """
        self._router = router
        self._logger.info("Message router set")
    
    def set_session_manager(self, session_manager: "SessionManager") -> None:
        """
        Set the session manager for the gateway.
        
        Args:
            session_manager: The session manager instance
        """
        self._session_manager = session_manager
        self._logger.info("Session manager set")
    
    # ============= Message Sending =============
    
    async def send_to_channel(
        self,
        channel_id: str,
        message: "UniversalMessage",
        priority: int = 1,
    ) -> "SendResult":
        """
        Send a message to a specific channel.
        
        Args:
            channel_id: Target channel ID
            message: The message to send
            priority: Message priority
            
        Returns:
            SendResult indicating the outcome
        """
        async with self._semaphore:
            adapter = self._registry.get(channel_id)
            if not adapter:
                self._logger.error(f"Channel not found: {channel_id}")
                return SendResult(
                    success=False,
                    error=f"Channel not found: {channel_id}",
                    error_code="CHANNEL_NOT_FOUND",
                )
            
            # Update message direction
            message.direction = MessageDirection.OUTGOING
            
            try:
                result = await adapter.send(message, priority)
                
                if result.success:
                    self._stats["total_messages_sent"] += 1
                    await self._emit_event(GatewayEventType.MESSAGE_SENT, {
                        "channel_id": channel_id,
                        "message_id": result.message_id,
                        "priority": priority,
                    })
                else:
                    self._stats["total_errors"] += 1
                
                return result
                
            except Exception as e:
                self._logger.error(f"Error sending to channel {channel_id}: {e}")
                self._stats["total_errors"] += 1
                return SendResult(
                    success=False,
                    error=str(e),
                    error_code=type(e).__name__,
                )
    
    async def broadcast(
        self,
        message: "UniversalMessage",
        channel_filter: Optional[Callable[["ChannelAdapter"], bool]] = None,
        parallel: bool = True,
    ) -> Dict[str, "SendResult"]:
        """
        Broadcast a message to multiple channels.
        
        Args:
            message: The message to broadcast
            channel_filter: Optional filter function for channels
            parallel: Whether to send to channels in parallel
            
        Returns:
            Dictionary mapping channel IDs to their send results
        """
        channels = self._registry.get_all()
        
        if channel_filter:
            channels = [ch for ch in channels if channel_filter(ch)]
        
        if not channels:
            return {}
        
        self._stats["total_broadcasts"] += 1
        
        if parallel:
            tasks = [
                self.send_to_channel(
                    channel._config.channel_id,
                    message.clone(),
                )
                for channel in channels
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for channel in channels:
                result = await self.send_to_channel(
                    channel._config.channel_id,
                    message.clone(),
                )
                results.append(result)
        
        return {
            channel._config.channel_id: (
                result if not isinstance(result, Exception) else SendResult(
                    success=False,
                    error=str(result),
                    error_code=type(result).__name__,
                )
            )
            for channel, result in zip(channels, results)
        }
    
    async def send_with_routing(
        self,
        message: "UniversalMessage",
        routing_context: Optional[Dict[str, Any]] = None,
    ) -> Optional["SendResult"]:
        """
        Send a message using the configured router.
        
        Args:
            message: The message to send
            routing_context: Context for routing decisions
            
        Returns:
            SendResult if routing was successful, None otherwise
        """
        if not self._router:
            self._logger.error("No router configured")
            return None
        
        try:
            route = await self._router.route(message, routing_context or {})
            
            if not route:
                self._logger.warning("No route found for message")
                return None
            
            target_channel = route.get("channel_id")
            if not target_channel:
                self._logger.warning("Route has no target channel")
                return None
            
            await self._emit_event(GatewayEventType.MESSAGE_ROUTED, {
                "message_id": message.correlation_id,
                "route": route,
            })
            
            return await self.send_to_channel(target_channel, message)
            
        except Exception as e:
            self._logger.error(f"Routing error: {e}")
            return SendResult(
                success=False,
                error=str(e),
                error_code="ROUTING_ERROR",
            )
    
    # ============= Message Receiving =============
    
    async def receive_from_channel(
        self,
        channel_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "ReceiveResult":
        """
        Receive messages from a specific channel.
        
        Args:
            channel_id: Source channel ID
            payload: Optional webhook payload
            
        Returns:
            ReceiveResult containing the received messages
        """
        adapter = self._registry.get(channel_id)
        if not adapter:
            return ReceiveResult(
                success=False,
                error=f"Channel not found: {channel_id}",
            )
        
        try:
            result = await adapter.receive(payload)
            
            if result.success:
                self._stats["total_messages_received"] += len(result.messages)
                
                for message in result.messages:
                    await self._handle_incoming_message(channel_id, message)
                
                await self._emit_event(GatewayEventType.MESSAGE_RECEIVED, {
                    "channel_id": channel_id,
                    "message_count": len(result.messages),
                })
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error receiving from channel {channel_id}: {e}")
            return ReceiveResult(
                success=False,
                error=str(e),
            )
    
    async def _handle_incoming_message(
        self,
        channel_id: str,
        message: "UniversalMessage",
    ) -> None:
        """
        Handle an incoming message from a channel.
        
        Args:
            channel_id: Source channel ID
            message: The received message
        """
        message.direction = MessageDirection.INCOMING
        
        # Update session if manager is available
        if self._session_manager:
            try:
                session = await self._session_manager.get_or_create_session(
                    session_id=message.metadata.conversation_id if message.metadata else None,
                    channel_id=channel_id,
                    user_id=message.metadata.sender.user_id if message.metadata and message.metadata.sender else None,
                )
                message.session_id = session.session_id
                
                await self._session_manager.update_session_context(
                    session.session_id,
                    {"last_message_id": message.correlation_id}
                )
            except Exception as e:
                self._logger.error(f"Error updating session: {e}")
    
    # ============= Event Handling =============
    
    def add_event_handler(
        self,
        event_type: GatewayEventType,
        handler: Callable,
    ) -> None:
        """
        Add an event handler for gateway events.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(
        self,
        event_type: GatewayEventType,
        handler: Callable,
    ) -> None:
        """
        Remove an event handler.
        
        Args:
            event_type: Type of event
            handler: Handler function to remove
        """
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
    
    async def _emit_event(
        self,
        event_type: GatewayEventType,
        data: Optional[Dict[str, Any]] = None,
        channel_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Emit a gateway event.
        
        Args:
            event_type: Type of event
            data: Event data
            channel_id: Related channel ID
            error: Error information
        """
        event = GatewayEvent(
            event_type=event_type,
            channel_id=channel_id,
            data=data or {},
            error=error,
        )
        
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._logger.warning("Event queue full, dropping event")
        
        # Also call handlers directly for important events
        for handler in self._event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"Error in event handler: {e}")
    
    async def _process_events(self) -> None:
        """Process events from the event queue."""
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                
                for handler in self._event_handlers.get(event.event_type, []):
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        self._logger.error(f"Error processing event: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error(f"Error in event processor: {e}")
    
    # ============= Channel Event Callbacks =============
    
    async def _on_channel_message(
        self,
        channel_id: str,
        message: "UniversalMessage",
    ) -> None:
        """Handle a message event from a channel."""
        await self._handle_incoming_message(channel_id, message)
    
    async def _on_channel_error(
        self,
        channel_id: str,
        error: Exception,
    ) -> None:
        """Handle an error event from a channel."""
        self._stats["total_errors"] += 1
        await self._emit_event(
            GatewayEventType.CHANNEL_ERROR,
            {"error": str(error)},
            channel_id=channel_id,
            error=str(error),
        )
    
    async def _on_channel_state_change(
        self,
        channel_id: str,
        old_state: Any,
        new_state: Any,
    ) -> None:
        """Handle a state change event from a channel."""
        if new_state == ConnectionState.CONNECTED:
            await self._emit_event(
                GatewayEventType.CHANNEL_CONNECTED,
                channel_id=channel_id,
            )
        elif new_state == ConnectionState.DISCONNECTED:
            await self._emit_event(
                GatewayEventType.CHANNEL_DISCONNECTED,
                channel_id=channel_id,
            )
    
    # ============= Health & Lifecycle =============
    
    async def start(self) -> None:
        """Start the gateway and all registered channels."""
        if self._is_running:
            return
        
        self._is_running = True
        self._start_time = time.time()
        
        # Connect all channels
        for channel_id in self._registry.list_channel_ids():
            adapter = self._registry.get(channel_id)
            if adapter and not adapter.is_connected:
                try:
                    await adapter.connect()
                except Exception as e:
                    self._logger.error(f"Failed to connect channel {channel_id}: {e}")
        
        # Start event processor
        self._event_processor_task = asyncio.create_task(self._process_events())
        
        # Start health checker if enabled
        if self._config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self._logger.info(f"Gateway {self._config.gateway_id} started")
    
    async def stop(self) -> None:
        """Stop the gateway and all channels."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._event_processor_task:
            self._event_processor_task.cancel()
            try:
                await self._event_processor_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all channels
        for channel_id in self._registry.list_channel_ids():
            adapter = self._registry.get(channel_id)
            if adapter and adapter.is_connected:
                try:
                    await adapter.disconnect()
                except Exception as e:
                    self._logger.error(f"Error disconnecting channel {channel_id}: {e}")
        
        self._logger.info(f"Gateway {self._config.gateway_id} stopped")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                
                unhealthy_channels = []
                for channel_id in self._registry.list_channel_ids():
                    adapter = self._registry.get(channel_id)
                    if adapter:
                        is_healthy = await adapter.health_check()
                        if not is_healthy:
                            unhealthy_channels.append(channel_id)
                
                if unhealthy_channels:
                    await self._emit_event(
                        GatewayEventType.HEALTH_CHECK_FAILED,
                        {"unhealthy_channels": unhealthy_channels},
                    )
                    
                    # Attempt to reconnect unhealthy channels
                    for channel_id in unhealthy_channels:
                        adapter = self._registry.get(channel_id)
                        if adapter:
                            try:
                                await adapter.reconnect()
                            except Exception as e:
                                self._logger.error(
                                    f"Failed to reconnect channel {channel_id}: {e}"
                                )
                                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in health check loop: {e}")
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Perform health checks on all channels.
        
        Returns:
            Dictionary mapping channel IDs to their health status
        """
        results = {}
        for channel_id in self._registry.list_channel_ids():
            adapter = self._registry.get(channel_id)
            if adapter:
                results[channel_id] = await adapter.health_check()
            else:
                results[channel_id] = False
        return results
    
    # ============= Utility Methods =============
    
    async def restart_channel(self, channel_id: str) -> bool:
        """
        Restart a specific channel.
        
        Args:
            channel_id: The channel ID to restart
            
        Returns:
            True if restart was successful, False otherwise
        """
        adapter = self._registry.get(channel_id)
        if not adapter:
            return False
        
        try:
            if adapter.is_connected:
                await adapter.disconnect()
            await adapter.connect()
            return True
        except Exception as e:
            self._logger.error(f"Failed to restart channel {channel_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive gateway statistics.
        
        Returns:
            Dictionary containing all statistics
        """
        return {
            "gateway": {
                "gateway_id": self._config.gateway_id,
                "name": self._config.name,
                "uptime": self.uptime,
                "is_running": self._is_running,
            },
            "channels": {
                channel_id: adapter.statistics
                for channel_id in self._registry.list_channel_ids()
                if (adapter := self._registry.get(channel_id)) is not None
            },
            "totals": self._stats,
        }
    
    def __repr__(self) -> str:
        """Return a string representation of the gateway."""
        return (
            f"ChannelGateway("
            f"id={self._config.gateway_id!r}, "
            f"channels={len(self._registry)}, "
            f"running={self._is_running})"
        )


# Import at bottom to avoid circular imports
from .base import ConnectionState, SendResult, ReceiveResult
from .universal_message import UniversalMessage, MessageDirection
