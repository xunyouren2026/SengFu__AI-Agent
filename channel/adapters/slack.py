"""
AGI Unified Framework - Slack Adapter Module

This module provides a complete Slack API adapter for the channel framework.

Features:
- Slack Web API integration
- Events API support
- Interactive components (buttons, select menus)
- Modal dialogs
- Block Kit
- Threading support
- User and channel management

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
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
class SlackConfig(ChannelConfig):
    """
    Configuration for Slack adapter.
    
    Attributes:
        bot_token: Slack bot token (xoxb-...)
        signing_secret: Slack signing secret
        verification_token: Legacy verification token
        app_token: App-level token (xapp-...)
        socket_mode: Whether to use Socket Mode
        bot_id: Bot user ID
        team_id: Workspace team ID
    """
    bot_token: str = ""
    signing_secret: str = ""
    verification_token: str = ""
    app_token: str = ""
    socket_mode: bool = False
    bot_id: Optional[str] = None
    team_id: Optional[str] = None


class SlackBlockKit:
    """Helper class for creating Slack Block Kit messages."""
    
    @staticmethod
    def section(
        text: str,
        text_type: str = "mrkdwn",
        fields: Optional[List[str]] = None,
        accessory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a section block."""
        block: Dict[str, Any] = {"type": "section"}
        
        if text:
            block["text"] = {"type": text_type, "text": text}
        
        if fields:
            block["fields"] = [
                {"type": "mrkdwn", "text": f} for f in fields
            ]
        
        if accessory:
            block["accessory"] = accessory
        
        return block
    
    @staticmethod
    def divider() -> Dict[str, Any]:
        """Create a divider block."""
        return {"type": "divider"}
    
    @staticmethod
    def image(image_url: str, alt_text: str) -> Dict[str, Any]:
        """Create an image block."""
        return {
            "type": "image",
            "image_url": image_url,
            "alt_text": alt_text,
        }
    
    @staticmethod
    def actions(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an actions block."""
        return {"type": "actions", "elements": elements}
    
    @staticmethod
    def context(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a context block."""
        return {"type": "context", "elements": elements}
    
    @staticmethod
    def header(text: str, emoji: bool = True) -> Dict[str, Any]:
        """Create a header block."""
        return {
            "type": "header",
            "text": {"type": "plain_text", "text": text, "emoji": emoji},
        }
    
    @staticmethod
    def input(
        label: str,
        element: Dict[str, Any],
        hint: Optional[str] = None,
        optional: bool = False,
    ) -> Dict[str, Any]:
        """Create an input block."""
        block = {
            "type": "input",
            "label": {"type": "plain_text", "text": label},
            "element": element,
        }
        
        if hint:
            block["hint"] = {"type": "plain_text", "text": hint}
        
        if optional:
            block["optional"] = True
        
        return block


class SlackElement:
    """Helper class for creating Slack elements."""
    
    @staticmethod
    def button(
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: Optional[str] = None,
        url: Optional[str] = None,
        emoji: bool = True,
    ) -> Dict[str, Any]:
        """Create a button element."""
        element: Dict[str, Any] = {
            "type": "button",
            "text": {"type": "plain_text", "text": text, "emoji": emoji},
            "action_id": action_id,
        }
        
        if value:
            element["value"] = value
        if style:
            element["style"] = style
        if url:
            element["url"] = url
        
        return element
    
    @staticmethod
    def select(
        action_id: str,
        placeholder: str,
        options: Optional[List[Dict[str, Any]]] = None,
        option_groups: Optional[List[Dict[str, Any]]] = None,
        initial_option: Optional[Dict[str, Any]] = None,
        min_query_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a select menu element."""
        element: Dict[str, Any] = {
            "type": "select",
            "action_id": action_id,
            "placeholder": {"type": "plain_text", "text": placeholder},
        }
        
        if options:
            element["options"] = options
        if option_groups:
            element["option_groups"] = option_groups
        if initial_option:
            element["initial_option"] = initial_option
        if min_query_length is not None:
            element["min_query_length"] = min_query_length
        
        return element
    
    @staticmethod
    def external_select(
        action_id: str,
        placeholder: str,
        min_query_length: int = 3,
        initial_option: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an external select menu element."""
        return {
            "type": "external_select",
            "action_id": action_id,
            "placeholder": {"type": "plain_text", "text": placeholder},
            "min_query_length": min_query_length,
            "initial_option": initial_option,
        }
    
    @staticmethod
    def users_select(
        action_id: str,
        placeholder: str,
        initial_user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a user select menu element."""
        element: Dict[str, Any] = {
            "type": "users_select",
            "action_id": action_id,
            "placeholder": {"type": "plain_text", "text": placeholder},
        }
        
        if initial_user:
            element["initial_user"] = initial_user
        
        return element
    
    @staticmethod
    def channels_select(
        action_id: str,
        placeholder: str,
        initial_channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a channel select menu element."""
        element: Dict[str, Any] = {
            "type": "channels_select",
            "action_id": action_id,
            "placeholder": {"type": "plain_text", "text": placeholder},
        }
        
        if initial_channel:
            element["initial_channel"] = initial_channel
        
        return element
    
    @staticmethod
    def static_select(
        action_id: str,
        placeholder: str,
        options: List[Dict[str, Any]],
        initial_option: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a static select menu element."""
        element: Dict[str, Any] = {
            "type": "static_select",
            "action_id": action_id,
            "placeholder": {"type": "plain_text", "text": placeholder},
            "options": options,
        }
        
        if initial_option:
            element["initial_option"] = initial_option
        
        return element
    
    @staticmethod
    def option(label: str, value: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a select option."""
        opt: Dict[str, Any] = {
            "text": {"type": "plain_text", "text": label},
            "value": value,
        }
        
        if description:
            opt["description"] = {"type": "plain_text", "text": description}
        
        return opt
    
    @staticmethod
    def checkboxes(action_id: str, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create checkboxes element."""
        return {
            "type": "checkboxes",
            "action_id": action_id,
            "options": options,
        }
    
    @staticmethod
    def radio_buttons(
        action_id: str,
        options: List[Dict[str, Any]],
        initial_option: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create radio buttons element."""
        element: Dict[str, Any] = {
            "type": "radio_buttons",
            "action_id": action_id,
            "options": options,
        }
        
        if initial_option:
            element["initial_option"] = initial_option
        
        return element
    
    @staticmethod
    def plain_text_input(
        action_id: str,
        placeholder: Optional[str] = None,
        initial_value: Optional[str] = None,
        multiline: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a plain text input element."""
        element: Dict[str, Any] = {
            "type": "plain_text_input",
            "action_id": action_id,
        }
        
        if placeholder:
            element["placeholder"] = {"type": "plain_text", "text": placeholder}
        if initial_value:
            element["initial_value"] = initial_value
        if multiline:
            element["multiline"] = True
        if min_length is not None:
            element["min_length"] = min_length
        if max_length is not None:
            element["max_length"] = max_length
        
        return element


class SlackAdapter(ChannelAdapter):
    """
    Slack API adapter.
    
    This adapter provides full integration with the Slack API,
    supporting both Events API and Web API calls.
    
    Features:
    - Message sending with Block Kit
    - Interactive components
    - Modal dialogs
    - Thread support
    - User and channel management
    - Webhook handling
    
    Example:
        ```python
        # Create adapter
        config = SlackConfig(
            channel_id="slack_main",
            bot_token="xoxb-...",
            signing_secret="..."
        )
        adapter = SlackAdapter(config)
        
        # Connect and send message
        await adapter.connect()
        result = await adapter.send_message(
            channel="#general",
            text="Hello from Slack!"
        )
        ```
    """
    
    BASE_API = "https://slack.com/api"
    
    def __init__(self, config: SlackConfig) -> None:
        """
        Initialize the Slack adapter.
        
        Args:
            config: Slack configuration
        """
        super().__init__(config)
        
        self._slack_config = config
        
        # Socket mode
        self._socket_mode_enabled = config.socket_mode
        self._socket_app: Optional[Any] = None
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._action_handlers: Dict[str, Callable] = {}
        self._command_handlers: Dict[str, Callable] = {}
        self._shortcut_handlers: Dict[str, Callable] = {}
    
    def _initialize_capabilities(self) -> None:
        """Initialize Slack capabilities."""
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
            ChannelCapability.PUSH_NOTIFICATIONS,
            ChannelCapability.TYPING_INDICATOR,
            ChannelCapability.BOT_COMMANDS,
            ChannelCapability.SLASH_COMMANDS,
            ChannelCapability.BUTTONS,
            ChannelCapability.MODALS,
            ChannelCapability.INTERACTIVE_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
            ChannelCapability.USER_INFO,
            ChannelCapability.RATE_LIMITING,
        }
    
    # ============= API Methods =============
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[List[tuple]] = None,
    ) -> Dict[str, Any]:
        """
        Make a request to the Slack API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            files: Files to upload
            
        Returns:
            API response
        """
        import aiohttp
        
        url = f"{self.BASE_API}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self._slack_config.bot_token}",
            "Content-Type": "application/json",
        }
        
        timeout = aiohttp.ClientTimeout(total=self._config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if files:
                form_data = aiohttp.FormData()
                for name, content, filename in files:
                    form_data.add_field(name, content, filename=filename)
                if data:
                    form_data.add_field("payload", json.dumps(data))
                
                headers.pop("Content-Type")
                
                async with session.request(method, url, data=form_data, headers=headers) as response:
                    result = await response.json()
            else:
                async with session.request(method, url, json=data, headers=headers) as response:
                    result = await response.json()
            
            if not result.get("ok"):
                error = result.get("error", "Unknown error")
                raise SlackError(error, result)
            
            return result
    
    # ============= Connection =============
    
    async def _connect_impl(self) -> bool:
        """Implementation of connect."""
        try:
            # Verify token and get bot info
            auth_result = await self._make_request("POST", "auth.test")
            self._slack_config.bot_id = auth_result.get("user_id")
            self._slack_config.team_id = auth_result.get("team_id")
            
            self._logger.info(f"Connected to Slack workspace: {auth_result.get('team')}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to Slack: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """Implementation of disconnect."""
        if self._socket_app:
            await self._socket_app.close()
            self._socket_app = None
    
    async def _health_check_impl(self) -> bool:
        """Implementation of health check."""
        try:
            await self._make_request("POST", "auth.test")
            return True
        except Exception:
            return False
    
    # ============= Webhook Verification =============
    
    def verify_signature(
        self,
        timestamp: str,
        body: str,
        signature: str,
    ) -> bool:
        """
        Verify a request signature from Slack.
        
        Args:
            timestamp: Request timestamp
            body: Request body
            signature: Request signature
            
        Returns:
            True if signature is valid
        """
        if not self._slack_config.signing_secret:
            return True
        
        # Check timestamp to prevent replay attacks
        current_time = int(time.time())
        if abs(current_time - int(timestamp)) > 60 * 5:
            return False
        
        # Create the signature base string
        sig_basestring = f"v0:{timestamp}:{body}"
        
        # Create the expected signature
        my_signature = "v0=" + hmac.new(
            self._slack_config.signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
        
        return hmac.compare_digest(my_signature, signature)
    
    # ============= Event Handling =============
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def add_action_handler(self, action_id: str, handler: Callable) -> None:
        """Add an action handler for interactive components."""
        self._action_handlers[action_id] = handler
    
    def add_command_handler(self, command: str, handler: Callable) -> None:
        """Add a slash command handler."""
        self._command_handlers[command] = handler
    
    def add_shortcut_handler(self, callback_id: str, handler: Callable) -> None:
        """Add a shortcut handler."""
        self._shortcut_handlers[callback_id] = handler
    
    async def handle_event_callback(self, payload: Dict[str, Any]) -> None:
        """Handle an event callback."""
        event_type = payload.get("event", {}).get("type")
        if not event_type:
            return
        
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception as e:
                self._logger.error(f"Event handler error for {event_type}: {e}")
    
    async def handle_interaction(self, payload: Dict[str, Any]) -> None:
        """Handle an interactive payload."""
        action_type = payload.get("type")
        
        if action_type == "block_actions":
            actions = payload.get("actions", [])
            for action in actions:
                action_id = action.get("action_id")
                if action_id in self._action_handlers:
                    handler = self._action_handlers[action_id]
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(payload)
                        else:
                            handler(payload)
                    except Exception as e:
                        self._logger.error(f"Action handler error for {action_id}: {e}")
        
        elif action_type == "view_submission":
            view = payload.get("view", {})
            callback_id = view.get("callback_id")
            if callback_id in self._action_handlers:
                handler = self._action_handlers[callback_id]
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(payload)
                    else:
                        handler(payload)
                except Exception as e:
                    self._logger.error(f"View submission handler error: {e}")
        
        elif action_type == "shortcut":
            callback_id = payload.get("callback_id")
            if callback_id in self._shortcut_handlers:
                handler = self._shortcut_handlers[callback_id]
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(payload)
                    else:
                        handler(payload)
                except Exception as e:
                    self._logger.error(f"Shortcut handler error: {e}")
    
    async def handle_slash_command(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a slash command."""
        if command in self._command_handlers:
            handler = self._command_handlers[command]
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(payload)
                else:
                    return handler(payload)
            except Exception as e:
                self._logger.error(f"Command handler error for {command}: {e}")
                return {"text": f"Error: {str(e)}", "response_type": "ephemeral"}
        
        return {"text": "Unknown command", "response_type": "ephemeral"}
    
    # ============= Message Handling =============
    
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
        
        return ReceiveResult(success=True, messages=[])
    
    def _convert_to_universal_message(self, payload: Dict[str, Any]) -> Optional[UniversalMessage]:
        """Convert a Slack event to UniversalMessage."""
        try:
            event = payload.get("event", {})
            
            # Ignore bot messages
            if event.get("subtype") == "bot_message":
                return None
            
            # Create user identity
            user = event.get("user", {})
            user_identity = UserIdentity(
                user_id=str(user.get("id", "")),
                username=user.get("name"),
                display_name=user.get("real_name"),
            )
            
            # Create channel identity
            channel_identity = ChannelIdentity(
                channel_id=str(event.get("channel", "")),
                channel_type="dm" if event.get("channel", "").startswith("D") else "channel",
            )
            
            # Create metadata
            metadata = MessageMetadata(
                message_id=str(event.get("client_msg_id", "")),
                conversation_id=str(event.get("channel", "")),
                reply_to=event.get("thread_ts"),
                channel_identity=channel_identity,
                sender=user_identity,
                thread_id=event.get("thread_ts"),
            )
            
            # Extract mentions
            for mention in event.get("mentions", {}).values():
                metadata.mentions.append(mention.get("user", ""))
            
            # Determine message type
            message_type = MessageType.TEXT
            if event.get("files"):
                message_type = MessageType.FILE
            
            # Create content
            content = MessageContent.from_text(event.get("text", ""))
            
            # Create attachments
            attachments = []
            for file_info in event.get("files", []):
                attachments.append(Attachment(
                    attachment_id=str(file_info.get("id", "")),
                    attachment_type=AttachmentType.IMAGE if file_info.get("mimetype", "").startswith("image/") else AttachmentType.DOCUMENT,
                    url=file_info.get("url_private"),
                    file_name=file_info.get("name"),
                    file_size=file_info.get("size"),
                    mime_type=file_info.get("mimetype"),
                ))
            
            return UniversalMessage(
                content=content,
                message_type=message_type,
                direction=MessageDirection.INCOMING,
                metadata=metadata,
                attachments=attachments,
            )
        except Exception as e:
            self._logger.error(f"Error converting Slack message: {e}")
            return None
    
    # ============= Message Sending =============
    
    async def _send_impl(
        self,
        message: UniversalMessage,
        priority: MessagePriority,
    ) -> SendResult:
        """Implementation of send message."""
        try:
            # Get target channel
            channel = (
                message.metadata.channel_identity.channel_id
                if message.metadata and message.metadata.channel_identity
                else message.get_context("channel")
            )
            
            if not channel:
                return SendResult(
                    success=False,
                    error="No target channel provided",
                    error_code="MISSING_TARGET",
                )
            
            # Build message payload
            payload = self._build_message_payload(message, channel)
            
            # Send message
            result = await self._make_request("POST", "chat.postMessage", payload)
            
            return SendResult(
                success=True,
                message_id=result.get("ts"),
                timestamp=float(result.get("ts", time.time())),
                metadata={"channel": channel, "message": result.get("message", {})},
            )
        except SlackError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=str(e.error_code),
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                error_code=type(e).__name__,
            )
    
    def _build_message_payload(
        self,
        message: UniversalMessage,
        channel: str,
    ) -> Dict[str, Any]:
        """Build a Slack message payload."""
        payload: Dict[str, Any] = {"channel": channel}
        
        # Text content
        text = message.content.text or message.content.get_primary_text()
        
        # Check for blocks in context
        blocks = message.get_context("blocks")
        if blocks:
            payload["blocks"] = blocks
            if not text:
                # Extract text from blocks for notification
                text = self._extract_text_from_blocks(blocks)
        
        # Add text
        if text:
            payload["text"] = text
        
        # Thread
        thread_ts = message.get_context("thread_ts")
        if thread_ts:
            payload["thread_ts"] = thread_ts
        
        # Reply broadcast
        if message.get_context("reply_broadcast"):
            payload["reply_broadcast"] = True
        
        # Unfurl
        if message.get_context("unfurl_links") is False:
            payload["unfurl_links"] = False
        if message.get_context("unfurl_media") is False:
            payload["unfurl_media"] = False
        
        # Metadata
        metadata = message.get_context("metadata")
        if metadata:
            payload["metadata"] = metadata
        
        return payload
    
    def _extract_text_from_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Extract plain text from Block Kit blocks."""
        text_parts = []
        
        for block in blocks:
            block_type = block.get("type")
            
            if block_type == "section":
                text = block.get("text", {})
                if isinstance(text, dict):
                    text_parts.append(text.get("text", ""))
                
                for field in block.get("fields", []):
                    if isinstance(field, dict):
                        text_parts.append(field.get("text", ""))
            
            elif block_type == "header":
                text = block.get("text", {})
                if isinstance(text, dict):
                    text_parts.append(text.get("text", ""))
            
            elif block_type == "context":
                for element in block.get("elements", []):
                    if isinstance(element, dict):
                        text_parts.append(element.get("text", ""))
            
            elif block_type == "actions":
                text_parts.append("[Interactive elements]")
        
        return " ".join(text_parts)
    
    # ============= Message Methods =============
    
    async def send_message(
        self,
        channel: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
        reply_broadcast: bool = False,
        unfurl_links: Optional[bool] = None,
        unfurl_media: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Send a message to a channel."""
        payload: Dict[str, Any] = {
            "channel": channel,
            "text": text,
        }
        
        if blocks:
            payload["blocks"] = blocks
        
        if attachments:
            payload["attachments"] = attachments
        
        if thread_ts:
            payload["thread_ts"] = thread_ts
        
        if reply_broadcast:
            payload["reply_broadcast"] = True
        
        if unfurl_links is not None:
            payload["unfurl_links"] = unfurl_links
        
        if unfurl_media is not None:
            payload["unfurl_media"] = unfurl_media
        
        return await self._make_request("POST", "chat.postMessage", payload)
    
    async def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Update an existing message."""
        payload: Dict[str, Any] = {
            "channel": channel,
            "ts": ts,
            "text": text,
        }
        
        if blocks:
            payload["blocks"] = blocks
        
        return await self._make_request("POST", "chat.update", payload)
    
    async def delete_message(self, channel: str, ts: str) -> bool:
        """Delete a message."""
        await self._make_request("POST", "chat.delete", {
            "channel": channel,
            "ts": ts,
        })
        return True
    
    async def schedule_message(
        self,
        channel: str,
        text: str,
        post_at: Union[str, int],
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Schedule a message."""
        payload: Dict[str, Any] = {
            "channel": channel,
            "text": text,
            "post_at": str(post_at) if isinstance(post_at, int) else post_at,
        }
        
        if blocks:
            payload["blocks"] = blocks
        
        return await self._make_request("POST", "chat.scheduleMessage", payload)
    
    async def scheduled_messages_list(
        self,
        channel: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List scheduled messages."""
        params: Dict[str, Any] = {"limit": limit}
        
        if channel:
            params["channel"] = channel
        if cursor:
            params["cursor"] = cursor
        
        return await self._make_request("POST", "chat.scheduledMessages.list", params)
    
    async def delete_scheduled_message(self, channel: str, scheduled_message_id: str) -> bool:
        """Delete a scheduled message."""
        await self._make_request("POST", "chat.scheduledMessages.delete", {
            "channel": channel,
            "scheduled_message_id": scheduled_message_id,
        })
        return True
    
    # ============= Thread Methods =============
    
    async def reply_in_thread(
        self,
        channel: str,
        thread_ts: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Reply to a message in a thread."""
        return await self.send_message(
            channel=channel,
            text=text,
            blocks=blocks,
            thread_ts=thread_ts,
        )
    
    async def get_thread_replies(
        self,
        channel: str,
        thread_ts: str,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Get replies to a thread."""
        params: Dict[str, Any] = {
            "channel": channel,
            "ts": thread_ts,
            "limit": limit,
        }
        
        if cursor:
            params["cursor"] = cursor
        
        return await self._make_request("POST", "conversations.replies", params)
    
    async def set_thread_archive(
        self,
        channel: str,
        thread_ts: str,
        archive: bool = True,
    ) -> bool:
        """Archive or unarchive a thread."""
        params: Dict[str, Any] = {
            "channel": channel,
            "thread_ts": thread_ts,
            "archive": archive,
        }
        
        await self._make_request("POST", "conversations.setThread", params)
        return True
    
    # ============= Reactions =============
    
    async def add_reaction(self, channel: str, ts: str, emoji: str) -> bool:
        """Add a reaction to a message."""
        await self._make_request("POST", "reactions.add", {
            "channel": channel,
            "timestamp": ts,
            "name": emoji,
        })
        return True
    
    async def remove_reaction(self, channel: str, ts: str, emoji: str) -> bool:
        """Remove a reaction from a message."""
        await self._make_request("POST", "reactions.remove", {
            "channel": channel,
            "timestamp": ts,
            "name": emoji,
        })
        return True
    
    async def get_reactions(
        self,
        channel: str,
        ts: str,
        full: bool = True,
    ) -> Dict[str, Any]:
        """Get reactions to a message."""
        return await self._make_request("POST", "reactions.get", {
            "channel": channel,
            "timestamp": ts,
            "full": full,
        })
    
    # ============= Files =============
    
    async def upload_file(
        self,
        channels: str,
        content: Union[str, bytes],
        filename: str,
        title: Optional[str] = None,
        initial_comment: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a file."""
        if isinstance(content, str):
            content = content.encode()
        
        return await self._make_request(
            "POST",
            "files.upload",
            {
                "channels": channels,
                "filename": filename,
                "title": title or filename,
                "initial_comment": initial_comment,
                "thread_ts": thread_ts,
            },
            [("file", content, filename)],
        )
    
    # ============= Views & Modals =============
    
    async def open_view(
        self,
        trigger_id: str,
        view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Open a modal view."""
        return await self._make_request("POST", "views.open", {
            "trigger_id": trigger_id,
            "view": view,
        })
    
    async def publish_view(
        self,
        user_id: str,
        view: Dict[str, Any],
        hash_value: Optional[str] = None,
    ) -> bool:
        """Publish a Home tab view."""
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "view": view,
        }
        
        if hash_value:
            payload["hash"] = hash_value
        
        await self._make_request("POST", "views.publish", payload)
        return True
    
    async def push_view(
        self,
        view: Dict[str, Any],
    ) -> bool:
        """Push a view onto the stack."""
        await self._make_request("POST", "views.push", {"view": view})
        return True
    
    @staticmethod
    def modal(
        callback_id: str,
        title: str,
        blocks: List[Dict[str, Any]],
        close_text: str = "Cancel",
        submit_text: str = "Submit",
        private_metadata: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a modal view."""
        return {
            "type": "modal",
            "callback_id": callback_id,
            "title": {"type": "plain_text", "text": title},
            "blocks": blocks,
            "close": {"type": "plain_text", "text": close_text},
            "submit": {"type": "plain_text", "text": submit_text},
            "private_metadata": private_metadata,
        }
    
    # ============= Channel & User Methods =============
    
    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get channel information."""
        return await self._make_request("POST", "conversations.info", {
            "channel": channel_id,
        })
    
    async def list_channels(
        self,
        types: Optional[List[str]] = None,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """List channels."""
        params: Dict[str, Any] = {"limit": limit}
        
        if types:
            params["types"] = ",".join(types)
        if cursor:
            params["cursor"] = cursor
        
        return await self._make_request("POST", "conversations.list", params)
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information."""
        return await self._make_request("POST", "users.info", {
            "user": user_id,
        })
    
    async def list_users(
        self,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """List users."""
        params: Dict[str, Any] = {"limit": limit}
        
        if cursor:
            params["cursor"] = cursor
        
        return await self._make_request("POST", "users.list", params)
    
    async def open_dm(self, user_id: str) -> Dict[str, Any]:
        """Open a direct message with a user."""
        return await self._make_request("POST", "conversations.open", {
            "users": user_id,
        })
    
    async def invite_to_channel(
        self,
        channel_id: str,
        users: List[str],
    ) -> Dict[str, Any]:
        """Invite users to a channel."""
        return await self._make_request("POST", "conversations.invite", {
            "channel": channel_id,
            "users": ",".join(users),
        })
    
    # ============= Utility =============
    
    async def post_ephemeral(
        self,
        channel: str,
        user: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Post an ephemeral message to a user."""
        payload: Dict[str, Any] = {
            "channel": channel,
            "user": user,
            "text": text,
        }
        
        if blocks:
            payload["blocks"] = blocks
        
        return await self._make_request("POST", "chat.postEphemeral", payload)
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Look up a user by email."""
        try:
            return await self._make_request("POST", "users.lookupByEmail", {
                "email": email,
            })
        except SlackError:
            return None


class SlackError(Exception):
    """Slack API error."""
    
    def __init__(self, message: str, response: Dict[str, Any]):
        self.message = message
        self.response = response
        self.error_code = response.get("error", "unknown")
        super().__init__(f"Slack API Error: {message}")
