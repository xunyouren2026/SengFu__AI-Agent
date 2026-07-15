"""
AGI Unified Framework - Discord Adapter Module

This module provides a complete Discord API adapter for the channel framework.

Features:
- Discord API v10 support
- Gateway connection with shards
- All message types
- Slash commands
- Message components (buttons, select menus)
- Embeds
- Thread management
- Channel management
- Guild management
- Webhooks

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import websockets
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ..base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    RetryConfig,
    SendResult,
)
from ..universal_message import (
    Attachment,
    AttachmentType,
    ChannelIdentity,
    MessageContent,
    MessageDirection,
    MessageMetadata,
    MessageStatus,
    MessageType,
    UniversalMessage,
    UserIdentity,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig(ChannelConfig):
    """
    Configuration for Discord adapter.
    
    Attributes:
        bot_token: Discord bot token
        application_id: Discord application ID
        api_version: Discord API version
        gateway_url: Gateway URL
        intents: Gateway intents
        shard_count: Number of shards
        shard_id: Current shard ID (if sharding)
        proxy_url: Optional proxy URL
        presence: Initial presence status
    """
    bot_token: str = ""
    application_id: str = ""
    api_version: int = 10
    gateway_url: str = "wss://gateway.discord.gg"
    intents: int = 513  # Default intents
    shard_count: int = 1
    shard_id: int = 0
    proxy_url: Optional[str] = None
    presence: Optional[Dict[str, Any]] = None


class DiscordIntents:
    """Discord Gateway Intents."""
    GUILDS = 1 << 0
    GUILD_MEMBERS = 1 << 1
    GUILD_BANS = 1 << 2
    GUILD_EMOJIS_AND_STICKERS = 1 << 3
    GUILD_INTEGRATIONS = 1 << 4
    GUILD_WEBHOOKS = 1 << 5
    GUILD_INVITES = 1 << 6
    GUILD_VOICE_STATES = 1 << 7
    GUILD_PRESENCES = 1 << 8
    GUILD_MESSAGES = 1 << 9
    GUILD_MESSAGE_REACTIONS = 1 << 10
    GUILD_MESSAGE_TYPING = 1 << 11
    DIRECT_MESSAGES = 1 << 12
    DIRECT_MESSAGE_REACTIONS = 1 << 13
    DIRECT_MESSAGE_TYPING = 1 << 14
    MESSAGE_CONTENT = 1 << 15
    GUILD_SCHEDULED_EVENTS = 1 << 16
    AUTO_MODERATION_CONFIGURATION = 1 << 20
    AUTO_MODERATION_EXECUTION = 1 << 21


class DiscordEmbed:
    """Helper class for creating Discord embeds."""
    
    def __init__(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        color: Optional[int] = None,
        timestamp: Optional[str] = None,
    ):
        self.title = title
        self.description = description
        self.url = url
        self.color = color
        self.timestamp = timestamp
        self.author: Optional[Dict[str, Any]] = None
        self.footer: Optional[Dict[str, Any]] = None
        self.image: Optional[Dict[str, Any]] = None
        self.thumbnail: Optional[Dict[str, Any]] = None
        self.fields: List[Dict[str, Any]] = []
    
    def set_author(
        self,
        name: str,
        url: Optional[str] = None,
        icon_url: Optional[str] = None,
    ) -> DiscordEmbed:
        """Set the embed author."""
        self.author = {"name": name}
        if url:
            self.author["url"] = url
        if icon_url:
            self.author["icon_url"] = icon_url
        return self
    
    def set_footer(
        self,
        text: str,
        icon_url: Optional[str] = None,
    ) -> DiscordEmbed:
        """Set the embed footer."""
        self.footer = {"text": text}
        if icon_url:
            self.footer["icon_url"] = icon_url
        return self
    
    def set_image(self, url: str) -> DiscordEmbed:
        """Set the embed image."""
        self.image = {"url": url}
        return self
    
    def set_thumbnail(self, url: str) -> DiscordEmbed:
        """Set the embed thumbnail."""
        self.thumbnail = {"url": url}
        return self
    
    def add_field(
        self,
        name: str,
        value: str,
        inline: bool = False,
    ) -> DiscordEmbed:
        """Add a field to the embed."""
        self.fields.append({
            "name": name,
            "value": value,
            "inline": inline,
        })
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        if self.title:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description
        if self.url:
            result["url"] = self.url
        if self.color:
            result["color"] = self.color
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.author:
            result["author"] = self.author
        if self.footer:
            result["footer"] = self.footer
        if self.image:
            result["image"] = self.image
        if self.thumbnail:
            result["thumbnail"] = self.thumbnail
        if self.fields:
            result["fields"] = self.fields
        return result


class DiscordComponent:
    """Helper class for creating Discord message components."""
    
    @staticmethod
    def action_row(components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an action row."""
        return {"type": 1, "components": components}
    
    @staticmethod
    def button(
        label: str,
        custom_id: Optional[str] = None,
        style: int = 1,
        url: Optional[str] = None,
        disabled: bool = False,
        emoji: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a button component."""
        component = {
            "type": 2,
            "label": label,
            "style": style,
            "disabled": disabled,
        }
        if custom_id:
            component["custom_id"] = custom_id
        if url:
            component["url"] = url
            component["style"] = 5  # Link style
        if emoji:
            component["emoji"] = emoji
        return component
    
    @staticmethod
    def select_menu(
        custom_id: str,
        options: List[Dict[str, Any]],
        placeholder: Optional[str] = None,
        min_values: int = 1,
        max_values: int = 1,
        disabled: bool = False,
    ) -> Dict[str, Any]:
        """Create a select menu component."""
        return {
            "type": 3,
            "custom_id": custom_id,
            "options": options,
            "placeholder": placeholder,
            "min_values": min_values,
            "max_values": max_values,
            "disabled": disabled,
        }
    
    @staticmethod
    def select_option(
        label: str,
        value: str,
        description: Optional[str] = None,
        emoji: Optional[Dict[str, Any]] = None,
        default: bool = False,
    ) -> Dict[str, Any]:
        """Create a select menu option."""
        option = {
            "label": label,
            "value": value,
        }
        if description:
            option["description"] = description
        if emoji:
            option["emoji"] = emoji
        if default:
            option["default"] = True
        return option
    
    @staticmethod
    def text_input(
        custom_id: str,
        label: str,
        style: int = 1,
        placeholder: Optional[str] = None,
        value: Optional[str] = None,
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a text input component."""
        component = {
            "type": 4,
            "custom_id": custom_id,
            "label": label,
            "style": style,
        }
        if placeholder:
            component["placeholder"] = placeholder
        if value:
            component["value"] = value
        component["required"] = required
        if min_length is not None:
            component["min_length"] = min_length
        if max_length is not None:
            component["max_length"] = max_length
        return component


class DiscordAdapter(ChannelAdapter):
    """
    Discord API adapter.
    
    This adapter provides full integration with the Discord API,
    supporting both gateway events and REST API calls.
    
    Features:
    - Gateway connection with intents
    - Message sending and receiving
    - Slash commands
    - Message components
    - Embeds
    - Thread management
    - Webhooks
    
    Example:
        ```python
        # Create adapter
        config = DiscordConfig(
            channel_id="discord_main",
            bot_token="YOUR_BOT_TOKEN",
            application_id="YOUR_APP_ID"
        )
        adapter = DiscordAdapter(config)
        
        # Connect and send message
        await adapter.connect()
        result = await adapter.send_message(
            channel_id="123456789",
            content="Hello from Discord!"
        )
        ```
    """
    
    # Discord API endpoints
    BASE_API = "https://discord.com/api/v10"
    
    # Gateway opcodes
    OPCODE_DISPATCH = 0
    OPCODE_HEARTBEAT = 1
    OPCODE_IDENTIFY = 2
    OPCODE_PRESENCE_UPDATE = 3
    OPCODE_VOICE_STATE_UPDATE = 4
    OPCODE_RESUME = 6
    OPCODE_RECONNECT = 7
    OPCODE_REQUEST_GUILD_MEMBERS = 8
    OPCODE_INVALID_SESSION = 9
    OPCODE_HELLO = 10
    OPCODE_HEARTBEAT_ACK = 11
    
    def __init__(self, config: DiscordConfig) -> None:
        """
        Initialize the Discord adapter.
        
        Args:
            config: Discord configuration
        """
        super().__init__(config)
        
        self._discord_config = config
        self._api_base = f"{self.BASE_API}"
        
        # Gateway state
        self._gateway_url: Optional[str] = None
        self._websocket: Optional[Any] = None
        self._sequence: Optional[int] = None
        self._session_id: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: float = 0.0
        self._last_heartbeat_ack: float = 0.0
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._interaction_handlers: Dict[str, Callable] = {}
        self._slash_command_handlers: Dict[str, Callable] = {}
        
        # Message handlers
        self._message_handlers: Dict[str, Callable] = {}
    
    def _initialize_capabilities(self) -> None:
        """Initialize Discord capabilities."""
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.MARKDOWN_MESSAGES,
            ChannelCapability.MEDIA_MESSAGES,
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.EMOJI_SUPPORT,
            ChannelCapability.THREADING,
            ChannelCapability.CHANNEL_TYPES,
            ChannelCapability.DIRECT_MESSAGES,
            ChannelCapability.GROUPS,
            ChannelCapability.WEBHOOK_MODE,
            ChannelCapability.EDIT_MESSAGES,
            ChannelCapability.DELETE_MESSAGES,
            ChannelCapability.BUTTONS,
            ChannelCapability.SLASH_COMMANDS,
            ChannelCapability.INTERACTIVE_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
            ChannelCapability.USER_INFO,
            ChannelCapability.MEMBER_INFO,
            ChannelCapability.RATE_LIMITING,
            ChannelCapability.AUTO_RECONNECT,
        }
    
    # ============= API Methods =============
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[List[Tuple[str, bytes, str]]] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make a request to the Discord API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            files: Files to upload
            reason: Audit log reason
            
        Returns:
            API response
        """
        import aiohttp
        
        url = f"{self._api_base}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bot {self._discord_config.bot_token}",
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (AGI Framework, 1.0.0)",
        }
        
        timeout = aiohttp.ClientTimeout(total=self._config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if files:
                form_data = aiohttp.FormData()
                for i, (name, content, content_type) in enumerate(files):
                    form_data.add_field(
                        f"file{i}",
                        content,
                        filename=name,
                        content_type=content_type,
                    )
                if data:
                    form_data.add_field("payload_json", json.dumps(data))
                
                async with session.request(method, url, data=form_data, headers=headers) as response:
                    result = await response.json()
            else:
                async with session.request(method, url, json=data, headers=headers) as response:
                    result = await response.json()
            
            if response.status == 204:
                return {}
            
            if response.status >= 400:
                error_code = result.get("code", response.status)
                error_message = result.get("message", "Unknown error")
                raise DiscordError(error_code, error_message, response.status)
            
            return result
    
    # ============= Gateway Connection =============
    
    async def _connect_impl(self) -> bool:
        """Implementation of connect."""
        try:
            # Get gateway URL
            gateway_info = await self._get_gateway()
            self._gateway_url = gateway_info.get("url", self._discord_config.gateway_url)
            
            # Connect to gateway
            await self._connect_gateway()
            
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to Discord: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """Implementation of disconnect."""
        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close websocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
    
    async def _health_check_impl(self) -> bool:
        """Implementation of health check."""
        try:
            await self._make_request("GET", "/users/@me")
            return True
        except Exception:
            return False
    
    async def _get_gateway(self) -> Dict[str, Any]:
        """Get the gateway URL."""
        return await self._make_request("GET", "/gateway")
    
    async def _connect_gateway(self) -> None:
        """Connect to the Discord gateway."""
        url = f"{self._gateway_url}/?v=10&encoding=json"
        
        if self._discord_config.proxy_url:
            self._websocket = await websockets.connect(
                url,
                proxy=self._discord_config.proxy_url,
            )
        else:
            self._websocket = await websockets.connect(url)
        
        # Start receiving messages
        asyncio.create_task(self._gateway_receiver())
    
    async def _gateway_receiver(self) -> None:
        """Receive and process gateway messages."""
        try:
            async for message in self._websocket:
                data = json.loads(message)
                await self._handle_gateway_message(data)
        except websockets.ConnectionClosed:
            self._logger.warning("Gateway connection closed")
            # Attempt to reconnect
            await self._handle_disconnect()
        except Exception as e:
            self._logger.error(f"Gateway receiver error: {e}")
    
    async def _handle_gateway_message(self, data: Dict[str, Any]) -> None:
        """Handle a gateway message."""
        opcode = data.get("op")
        payload = data.get("d", {})
        sequence = data.get("s")
        event_name = data.get("t")
        
        # Update sequence
        if sequence is not None:
            self._sequence = sequence
        
        if opcode == self.OPCODE_HELLO:
            # Start heartbeat
            interval = payload.get("heartbeat_interval", 45000) / 1000
            self._heartbeat_interval = interval
            self._start_heartbeat()
            
            # Send identify
            await self._send_identify()
        
        elif opcode == self.OPCODE_HEARTBEAT:
            # Send heartbeat immediately
            await self._send_heartbeat()
        
        elif opcode == self.OPCODE_HEARTBEAT_ACK:
            self._last_heartbeat_ack = time.time()
        
        elif opcode == self.OPCODE_DISPATCH:
            await self._handle_dispatch(event_name, payload)
        
        elif opcode == self.OPCODE_RECONNECT:
            self._logger.info("Discord requested reconnect")
            await self._handle_disconnect()
        
        elif opcode == self.OPCODE_INVALID_SESSION:
            self._logger.warning("Invalid session, re-identifying")
            await asyncio.sleep(5)
            await self._send_identify()
    
    async def _handle_dispatch(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Handle a dispatch event."""
        if event_name == "READY":
            self._session_id = payload.get("session_id")
            self._logger.info(f"Discord gateway ready, session: {self._session_id}")
            self._set_state(ConnectionState.CONNECTED)
        
        elif event_name == "MESSAGE_CREATE":
            asyncio.create_task(self._handle_message_create(payload))
        
        elif event_name == "MESSAGE_UPDATE":
            asyncio.create_task(self._handle_message_update(payload))
        
        elif event_name == "MESSAGE_DELETE":
            asyncio.create_task(self._handle_message_delete(payload))
        
        elif event_name == "INTERACTION_CREATE":
            asyncio.create_task(self._handle_interaction(payload))
        
        elif event_name == "THREAD_CREATE":
            asyncio.create_task(self._handle_thread_create(payload))
        
        elif event_name == "THREAD_UPDATE":
            asyncio.create_task(self._handle_thread_update(payload))
        
        elif event_name == "THREAD_DELETE":
            asyncio.create_task(self._handle_thread_delete(payload))
        
        # Call custom handlers
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception as e:
                self._logger.error(f"Event handler error for {event_name}: {e}")
    
    async def _send_identify(self) -> None:
        """Send the identify payload."""
        payload = {
            "op": self.OPCODE_IDENTIFY,
            "d": {
                "token": self._discord_config.bot_token,
                "intents": self._discord_config.intents,
                "properties": {
                    "os": "linux",
                    "browser": "AGI Framework",
                    "device": "AGI Framework",
                },
                "shard": [self._discord_config.shard_id, self._discord_config.shard_count],
            },
        }
        
        if self._discord_config.presence:
            payload["d"]["presence"] = self._discord_config.presence
        
        if self._session_id and self._sequence:
            # Resume
            payload = {
                "op": self.OPCODE_RESUME,
                "d": {
                    "token": self._discord_config.bot_token,
                    "session_id": self._session_id,
                    "seq": self._sequence,
                },
            }
        
        await self._websocket.send(json.dumps(payload))
    
    async def _send_heartbeat(self) -> None:
        """Send a heartbeat."""
        payload = {
            "op": self.OPCODE_HEARTBEAT,
            "d": self._sequence,
        }
        await self._websocket.send(json.dumps(payload))
    
    def _start_heartbeat(self) -> None:
        """Start the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            await self._send_heartbeat()
    
    async def _handle_disconnect(self) -> None:
        """Handle gateway disconnection."""
        self._set_state(ConnectionState.RECONNECTING)
        
        try:
            await asyncio.sleep(5)
            await self._connect_gateway()
        except Exception as e:
            self._logger.error(f"Failed to reconnect: {e}")
            self._set_state(ConnectionState.ERROR)
    
    # ============= Event Handling =============
    
    def add_event_handler(self, event_name: str, handler: Callable) -> None:
        """Add an event handler."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
    
    def register_slash_command(self, command_name: str, handler: Callable) -> None:
        """Register a slash command handler."""
        self._slash_command_handlers[command_name] = handler
    
    def register_interaction_handler(self, custom_id: str, handler: Callable) -> None:
        """Register a component interaction handler."""
        self._interaction_handlers[custom_id] = handler
    
    async def _handle_interaction(self, payload: Dict[str, Any]) -> None:
        """Handle an interaction."""
        interaction_type = payload.get("type")
        data = payload.get("data", {})
        custom_id = data.get("custom_id")
        
        # Handle slash commands
        if interaction_type == 2:  # APPLICATION_COMMAND
            command_name = data.get("name")
            if command_name in self._slash_command_handlers:
                handler = self._slash_command_handlers[command_name]
                try:
                    await handler(payload)
                except Exception as e:
                    self._logger.error(f"Slash command handler error: {e}")
        
        # Handle component interactions
        elif interaction_type in (3, 5):  # MESSAGE_COMPONENT, MODAL_SUBMIT
            if custom_id in self._interaction_handlers:
                handler = self._interaction_handlers[custom_id]
                try:
                    await handler(payload)
                except Exception as e:
                    self._logger.error(f"Interaction handler error: {e}")
        
        # Acknowledge interaction
        await self._create_interaction_response(
            payload.get("id"),
            payload.get("token"),
            {"type": 5} if interaction_type in (3, 5) else {"type": 4, "data": {"content": "..."}},
        )
    
    async def _create_interaction_response(
        self,
        interaction_id: str,
        interaction_token: str,
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create an interaction response."""
        endpoint = f"/interactions/{interaction_id}/{interaction_token}/callback"
        return await self._make_request("POST", endpoint, response)
    
    # ============= Message Handling =============
    
    async def _handle_message_create(self, payload: Dict[str, Any]) -> None:
        """Handle a message create event."""
        if payload.get("author", {}).get("bot") and not payload.get("webhook_id"):
            # Ignore bot messages (could add exception handling)
            pass
        
        message = self._convert_to_universal_message(payload)
        if message:
            await self._emit_event("message", message)
    
    async def _handle_message_update(self, payload: Dict[str, Any]) -> None:
        """Handle a message update event."""
        # Convert to universal message
        message = self._convert_to_universal_message(payload)
        if message:
            message.message_type = MessageType.EDITED
            await self._emit_event("message_edit", message)
    
    async def _handle_message_delete(self, payload: Dict[str, Any]) -> None:
        """Handle a message delete event."""
        await self._emit_event("message_delete", {
            "message_id": payload.get("id"),
            "channel_id": payload.get("channel_id"),
            "guild_id": payload.get("guild_id"),
        })
    
    async def _handle_thread_create(self, payload: Dict[str, Any]) -> None:
        """Handle a thread create event."""
        await self._emit_event("thread_create", payload)
    
    async def _handle_thread_update(self, payload: Dict[str, Any]) -> None:
        """Handle a thread update event."""
        await self._emit_event("thread_update", payload)
    
    async def _handle_thread_delete(self, payload: Dict[str, Any]) -> None:
        """Handle a thread delete event."""
        await self._emit_event("thread_delete", payload)
    
    def _convert_to_universal_message(self, data: Dict[str, Any]) -> Optional[UniversalMessage]:
        """Convert a Discord message to UniversalMessage."""
        try:
            # Create user identity
            author = data.get("author", {})
            user_identity = UserIdentity(
                user_id=str(author.get("id", "")),
                username=author.get("username"),
                first_name=author.get("username"),
                discriminator=author.get("discriminator"),
                is_bot=author.get("bot", False),
                avatar_url=self._get_avatar_url(author.get("id"), author.get("avatar")),
            )
            
            # Create channel identity
            channel = data.get("channel", {})
            channel_identity = ChannelIdentity(
                channel_id=str(data.get("channel_id", "")),
                channel_type="dm" if data.get("guild_id") is None else "text",
                channel_name=channel.get("name"),
                metadata={
                    "guild_id": data.get("guild_id"),
                    "thread_id": data.get("thread", {}).get("id"),
                },
            )
            
            # Create metadata
            metadata = MessageMetadata(
                message_id=str(data.get("id", "")),
                conversation_id=str(data.get("channel_id", "")),
                reply_to=data.get("message_reference", {}).get("message_id"),
                channel_identity=channel_identity,
                sender=user_identity,
                edited_at=data.get("edited_timestamp"),
                thread_id=data.get("thread", {}).get("id"),
            )
            
            # Extract mentions
            for mention in data.get("mentions", []):
                metadata.mentions.append(mention.get("id", ""))
            
            # Determine message type
            message_type = MessageType.TEXT
            if data.get("embeds"):
                message_type = MessageType.CARD
            elif data.get("components"):
                message_type = MessageType.INTERACTIVE
            
            # Create content
            content = MessageContent.from_text(data.get("content", ""))
            
            # Extract attachments
            attachments = []
            for att in data.get("attachments", []):
                attachments.append(Attachment(
                    attachment_id=str(att.get("id", "")),
                    attachment_type=AttachmentType.IMAGE if att.get("content_type", "").startswith("image/") else AttachmentType.DOCUMENT,
                    url=att.get("url"),
                    file_id=att.get("id"),
                    file_name=att.get("filename"),
                    file_size=att.get("size"),
                    width=att.get("width"),
                    height=att.get("height"),
                    proxy_url=att.get("proxy_url"),
                ))
            
            # Create message
            return UniversalMessage(
                content=content,
                message_type=message_type,
                direction=MessageDirection.INCOMING,
                metadata=metadata,
                attachments=attachments,
            )
            
        except Exception as e:
            self._logger.error(f"Error converting Discord message: {e}")
            return None
    
    def _get_avatar_url(self, user_id: str, avatar_hash: Optional[str]) -> Optional[str]:
        """Get the avatar URL for a user."""
        if not avatar_hash:
            return None
        
        if avatar_hash.startswith("a_"):
            return f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.gif"
        return f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png"
    
    # ============= Message Sending =============
    
    async def _send_impl(
        self,
        message: UniversalMessage,
        priority: MessagePriority,
    ) -> SendResult:
        """Implementation of send message."""
        try:
            # Get target channel ID
            channel_id = (
                message.metadata.channel_identity.channel_id
                if message.metadata and message.metadata.channel_identity
                else message.get_context("channel_id")
            )
            
            if not channel_id:
                return SendResult(
                    success=False,
                    error="No target channel_id provided",
                    error_code="MISSING_TARGET",
                )
            
            # Build message payload
            payload = self._build_message_payload(message)
            
            # Send message
            result = await self._make_request(
                "POST",
                f"/channels/{channel_id}/messages",
                payload,
            )
            
            return SendResult(
                success=True,
                message_id=result.get("id"),
                timestamp=int(result.get("timestamp", time.time()).replace("Z", "").replace("T", "")) / 1000,
                metadata={"channel_id": channel_id, "message": result},
            )
            
        except DiscordError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=str(e.code),
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                error_code=type(e).__name__,
            )
    
    def _build_message_payload(self, message: UniversalMessage) -> Dict[str, Any]:
        """Build a Discord message payload."""
        payload: Dict[str, Any] = {}
        
        # Content
        text = message.content.text or message.content.get_primary_text()
        if text:
            payload["content"] = text
        
        # Embeds
        embeds = []
        
        # Check for embeds in context
        embed_data = message.get_context("embed")
        if embed_data:
            if isinstance(embed_data, dict):
                embeds.append(embed_data)
            elif isinstance(embed_data, DiscordEmbed):
                embeds.append(embed_data.to_dict())
        
        # Add media attachments as embed images
        for att in message.attachments:
            if att.url:
                if not embeds:
                    embeds.append({})
                embeds[0]["image"] = {"url": att.url}
                if att.thumbnail_url:
                    embeds[0]["thumbnail"] = {"url": att.thumbnail_url}
        
        if embeds:
            payload["embeds"] = embeds
        
        # Components
        components = message.get_context("components")
        if components:
            payload["components"] = components
        
        # Reply
        if message.metadata and message.metadata.reply_to:
            payload["message_reference"] = {
                "message_id": message.metadata.reply_to,
            }
        
        # Allowed mentions
        allowed_mentions = message.get_context("allowed_mentions")
        if allowed_mentions:
            payload["allowed_mentions"] = allowed_mentions
        
        # Flags
        if message.get_context("suppress_embeds"):
            payload["flags"] = 1 << 2  # SUPPRESS_EMBEDS
        
        return payload
    
    async def _receive_impl(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> ReceiveResult:
        """Implementation of receive messages."""
        if payload:
            message = self._convert_to_universal_message(payload)
            return ReceiveResult(
                success=True,
                messages=[message] if message else [],
                raw_payload=payload,
            )
        
        return ReceiveResult(
            success=True,
            messages=[],
        )
    
    # ============= REST API Methods =============
    
    async def send_message(
        self,
        channel_id: str,
        content: str,
        embed: Optional[DiscordEmbed] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        reply_to: Optional[str] = None,
        allowed_mentions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a message to a channel."""
        payload: Dict[str, Any] = {"content": content}
        
        if embed:
            payload["embeds"] = [embed.to_dict()]
        
        if components:
            payload["components"] = components
        
        if reply_to:
            payload["message_reference"] = {"message_id": reply_to}
        
        if allowed_mentions:
            payload["allowed_mentions"] = allowed_mentions
        
        return await self._make_request(
            "POST",
            f"/channels/{channel_id}/messages",
            payload,
        )
    
    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        content: Optional[str] = None,
        embed: Optional[DiscordEmbed] = None,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Edit a message."""
        payload: Dict[str, Any] = {}
        
        if content is not None:
            payload["content"] = content
        
        if embed:
            payload["embeds"] = [embed.to_dict()]
        
        if components is not None:
            payload["components"] = components
        
        return await self._make_request(
            "PATCH",
            f"/channels/{channel_id}/messages/{message_id}",
            payload,
        )
    
    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Delete a message."""
        await self._make_request(
            "DELETE",
            f"/channels/{channel_id}/messages/{message_id}",
            reason=reason,
        )
        return True
    
    async def create_channel(
        self,
        guild_id: str,
        name: str,
        channel_type: int = 0,  # 0 = GUILD_TEXT
        topic: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a channel."""
        payload: Dict[str, Any] = {
            "name": name,
            "type": channel_type,
        }
        
        if topic:
            payload["topic"] = topic
        if parent_id:
            payload["parent_id"] = parent_id
        
        return await self._make_request(
            "POST",
            f"/guilds/{guild_id}/channels",
            payload,
        )
    
    async def create_thread(
        self,
        channel_id: str,
        name: str,
        message_id: Optional[str] = None,
        auto_archive_duration: int = 1440,
    ) -> Dict[str, Any]:
        """Create a thread."""
        payload: Dict[str, Any] = {
            "name": name,
            "auto_archive_duration": auto_archive_duration,
        }
        
        if message_id:
            endpoint = f"/channels/{channel_id}/messages/{message_id}/threads"
        else:
            endpoint = f"/channels/{channel_id}/threads"
        
        return await self._make_request("POST", endpoint, payload)
    
    async def add_thread_member(
        self,
        channel_id: str,
        user_id: str,
    ) -> bool:
        """Add a member to a thread."""
        await self._make_request(
            "PUT",
            f"/channels/{channel_id}/thread-members/{user_id}",
        )
        return True
    
    async def remove_thread_member(
        self,
        channel_id: str,
        user_id: str,
    ) -> bool:
        """Remove a member from a thread."""
        await self._make_request(
            "DELETE",
            f"/channels/{channel_id}/thread-members/{user_id}",
        )
        return True
    
    # ============= Slash Commands =============
    
    async def create_global_command(
        self,
        name: str,
        description: str,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a global slash command."""
        if not self._discord_config.application_id:
            raise ValueError("application_id not configured")
        
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "type": 1,  # CHAT_INPUT
        }
        
        if options:
            payload["options"] = options
        
        return await self._make_request(
            "POST",
            f"/applications/{self._discord_config.application_id}/commands",
            payload,
        )
    
    async def create_guild_command(
        self,
        guild_id: str,
        name: str,
        description: str,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a guild slash command."""
        if not self._discord_config.application_id:
            raise ValueError("application_id not configured")
        
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "type": 1,
        }
        
        if options:
            payload["options"] = options
        
        return await self._make_request(
            "POST",
            f"/applications/{self._discord_config.application_id}/guilds/{guild_id}/commands",
            payload,
        )
    
    async def get_commands(self, guild_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get application commands."""
        if not self._discord_config.application_id:
            raise ValueError("application_id not configured")
        
        endpoint = f"/applications/{self._discord_config.application_id}/commands"
        if guild_id:
            endpoint = f"/applications/{self._discord_config.application_id}/guilds/{guild_id}/commands"
        
        return await self._make_request("GET", endpoint)
    
    # ============= Webhook Methods =============
    
    async def create_webhook(
        self,
        channel_id: str,
        name: str,
        avatar: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Create a webhook."""
        payload: Dict[str, Any] = {"name": name}
        
        files = None
        if avatar:
            files = [("avatar", avatar, "image/png")]
        
        return await self._make_request(
            "POST",
            f"/channels/{channel_id}/webhooks",
            payload if not files else {"name": name},
            files,
        )
    
    async def execute_webhook(
        self,
        webhook_id: str,
        webhook_token: str,
        content: Optional[str] = None,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        wait: bool = False,
    ) -> Dict[str, Any]:
        """Execute a webhook."""
        url = f"{self._api_base}/webhooks/{webhook_id}/{webhook_token}"
        if wait:
            url += "?wait=true"
        
        payload: Dict[str, Any] = {}
        if content:
            payload["content"] = content
        if username:
            payload["username"] = username
        if avatar_url:
            payload["avatar_url"] = avatar_url
        if embeds:
            payload["embeds"] = embeds
        if components:
            payload["components"] = components
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if wait:
                    return await response.json()
                return {}
    
    # ============= Channel & User Info =============
    
    async def get_channel(self, channel_id: str) -> Dict[str, Any]:
        """Get channel information."""
        return await self._make_request("GET", f"/channels/{channel_id}")
    
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user information."""
        return await self._make_request("GET", f"/users/{user_id}")
    
    async def get_member(
        self,
        guild_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Get guild member information."""
        return await self._make_request("GET", f"/guilds/{guild_id}/members/{user_id}")
    
    async def get_guild(self, guild_id: str) -> Dict[str, Any]:
        """Get guild information."""
        return await self._make_request("GET", f"/guilds/{guild_id}")
    
    # ============= Utility =============
    
    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user info (required by base class)."""
        try:
            return await self.get_user(user_id)
        except Exception:
            return None
    
    async def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get channel info (required by base class)."""
        try:
            return await self.get_channel(channel_id)
        except Exception:
            return None


class DiscordError(Exception):
    """Discord API error."""
    
    def __init__(self, code: int, message: str, http_status: int = 400):
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(f"Discord API Error {code}: {message}")


# Type alias for discriminator
DiscordAdapter.discriminator = property(lambda self: None)
