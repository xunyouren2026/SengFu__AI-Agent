"""
AGI Unified Framework - Feishu (Lark) Adapter Module

This module provides a complete Feishu (Lark) Open Platform API adapter
for the channel framework.

Features:
- Feishu Open Platform API integration
- Webhook messages
- Interactive cards
- Event subscriptions
- User and department management
- File handling

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
class FeishuConfig(ChannelConfig):
    """
    Configuration for Feishu adapter.
    
    Attributes:
        app_id: Feishu app ID
        app_secret: Feishu app secret
        verification_token: Verification token for webhook
        encrypt_key: Encryption key for webhook
        tenant_access_token: Cached tenant access token
        token_expires_at: Token expiration time
    """
    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    encrypt_key: str = ""
    tenant_access_token: Optional[str] = None
    token_expires_at: float = 0.0


class FeishuCard:
    """Helper class for creating Feishu interactive cards."""
    
    @staticmethod
    def simple_card(
        header: str,
        content: str,
        action_text: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a simple card."""
        elements = [
            {
                "tag": "markdown",
                "content": content,
            }
        ]
        
        if action_text and action_url:
            elements.append({
                "tag": "action",
                "actions": [
                    {
                        "tag": "a",
                        "text": {"tag": "plain_text", "content": action_text},
                        "href": action_url,
                    }
                ],
            })
        
        return FeishuCard.card(
            header=header,
            elements=elements,
        )
    
    @staticmethod
    def card(
        header: Optional[Dict[str, Any]] = None,
        elements: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a card structure."""
        card: Dict[str, Any] = {}
        
        if header:
            card["header"] = {
                "title": {"tag": "plain_text", "content": header.get("title", "")},
                "template": header.get("template", "blue"),
            }
        
        if elements:
            card["elements"] = elements
        
        if config:
            card["config"] = config
        
        return card


class FeishuElement:
    """Helper class for creating Feishu card elements."""
    
    @staticmethod
    def markdown(content: str) -> Dict[str, Any]:
        """Create a markdown element."""
        return {"tag": "markdown", "content": content}
    
    @staticmethod
    def text(content: str) -> Dict[str, Any]:
        """Create a text element."""
        return {"tag": "plain_text", "content": content}
    
    @staticmethod
    def div(content: str) -> Dict[str, Any]:
        """Create a div (hr) element."""
        return {"tag": "hr"}
    
    @staticmethod
    def image(image_key: str, alt_text: str = "") -> Dict[str, Any]:
        """Create an image element."""
        return {
            "tag": "img",
            "img_key": image_key,
            "alt": {"tag": "plain_text", "content": alt_text},
        }
    
    @staticmethod
    def action(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an action container."""
        return {"tag": "action", "actions": actions}
    
    @staticmethod
    def button(
        text: str,
        value: Optional[str] = None,
        action_id: Optional[str] = None,
        type: str = "default",
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a button element."""
        button: Dict[str, Any] = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
        }
        
        if value:
            button["value"] = value
        
        if action_id:
            button["action_id"] = action_id
        
        if type != "default":
            button["type"] = type
        
        if url:
            button["url"] = url
        
        return button
    
    @staticmethod
    def select_menu(
        placeholder: str,
        options: List[Dict[str, Any]],
        action_id: str,
    ) -> Dict[str, Any]:
        """Create a select menu element."""
        return {
            "tag": "select_static",
            "placeholder": {"tag": "plain_text", "content": placeholder},
            "options": [
                {
                    "text": {"tag": "plain_text", "content": opt.get("label", "")},
                    "value": opt.get("value", ""),
                }
                for opt in options
            ],
            "action_id": action_id,
        }


class FeishuAdapter(ChannelAdapter):
    """
    Feishu (Lark) Open Platform adapter.
    
    This adapter provides integration with the Feishu Open Platform API,
    supporting messages, cards, and events.
    
    Features:
    - Webhook messages
    - Interactive cards
    - Event subscriptions
    - User management
    - Department management
    
    Example:
        ```python
        # Create adapter
        config = FeishuConfig(
            channel_id="feishu_main",
            app_id="cli_xxx",
            app_secret="xxx"
        )
        adapter = FeishuAdapter(config)
        
        # Connect and send message
        await adapter.connect()
        result = await adapter.send_message(
            receive_id="ou_xxx",
            msg_type="text",
            content={"text": "Hello from Feishu!"}
        )
        ```
    """
    
    BASE_API = "https://open.feishu.cn/open-apis"
    
    def __init__(self, config: FeishuConfig) -> None:
        """
        Initialize the Feishu adapter.
        
        Args:
            config: Feishu configuration
        """
        super().__init__(config)
        
        self._feishu_config = config
        
        # Token management
        self._tenant_access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def _initialize_capabilities(self) -> None:
        """Initialize Feishu capabilities."""
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.HTML_MESSAGES,
            ChannelCapability.MARKDOWN_MESSAGES,
            ChannelCapability.MEDIA_MESSAGES,
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.CHANNEL_TYPES,
            ChannelCapability.DIRECT_MESSAGES,
            ChannelCapability.GROUPS,
            ChannelCapability.WEBHOOK_MODE,
            ChannelCapability.BUTTONS,
            ChannelCapability.INTERACTIVE_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
            ChannelCapability.USER_INFO,
            ChannelCapability.RATE_LIMITING,
        }
    
    # ============= Token Management =============
    
    async def _get_access_token(self, force_refresh: bool = False) -> str:
        """Get tenant access token."""
        now = time.time()
        
        if not force_refresh and self._tenant_access_token and now < self._token_expires_at - 60:
            return self._tenant_access_token
        
        # Get new token
        result = await self._make_request(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            {"app_id": self._feishu_config.app_id, "app_secret": self._feishu_config.app_secret},
        )
        
        self._tenant_access_token = result.get("tenant_access_token")
        self._token_expires_at = now + result.get("expire", 7200)
        
        return self._tenant_access_token
    
    # ============= API Methods =============
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[List[Tuple[str, bytes, str]]] = None,
    ) -> Dict[str, Any]:
        """Make a request to the Feishu API."""
        import aiohttp
        
        url = f"{self.BASE_API}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        
        timeout = aiohttp.ClientTimeout(total=self._config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if files:
                form_data = aiohttp.FormData()
                for name, content, filename in files:
                    form_data.add_field(name, content, filename=filename)
                if data:
                    form_data.add_field("file_name", json.dumps(data))
                
                async with session.request(method, url, data=form_data, headers=headers) as response:
                    result = await response.json()
            else:
                async with session.request(method, url, json=data, params=params, headers=headers) as response:
                    result = await response.json()
            
            code = result.get("code", 0)
            if code != 0:
                msg = result.get("msg", "Unknown error")
                raise FeishuError(code, msg)
            
            return result
    
    # ============= Connection =============
    
    async def _connect_impl(self) -> bool:
        """Implementation of connect."""
        try:
            # Verify credentials by getting access token
            token = await self._get_access_token()
            self._logger.info(f"Connected to Feishu, token: {token[:20]}...")
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to Feishu: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """Implementation of disconnect."""
        self._tenant_access_token = None
        self._token_expires_at = 0.0
    
    async def _health_check_impl(self) -> bool:
        """Implementation of health check."""
        try:
            await self._get_access_token()
            return True
        except Exception:
            return False
    
    # ============= Event Handling =============
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def verify_webhook(
        self,
        challenge: str,
        verify_token: str,
    ) -> bool:
        """Verify webhook URL."""
        return verify_token == self._feishu_config.verification_token
    
    async def handle_event(self, event: Dict[str, Any]) -> None:
        """Handle an incoming event."""
        event_type = event.get("event_type")
        
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"Event handler error for {event_type}: {e}")
    
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
        """Convert a Feishu event to UniversalMessage."""
        try:
            event = payload.get("event", {})
            event_type = payload.get("event_type", "")
            
            # Determine message type
            if "message" in event_type:
                message_data = event.get("message", {})
            else:
                return None
            
            # Create user identity
            sender = event.get("sender", {})
            user_identity = UserIdentity(
                user_id=str(sender.get("sender_id", {}).get("open_id", "")),
                sender_type=sender.get("sender_type"),
            )
            
            # Create channel identity
            chat_id = message_data.get("chat_id", "")
            channel_identity = ChannelIdentity(
                channel_id=str(chat_id),
                channel_type="p2p" if message_data.get("chat_type") == "p2p" else "group",
            )
            
            # Create metadata
            metadata = MessageMetadata(
                message_id=str(message_data.get("message_id", "")),
                conversation_id=str(chat_id),
                channel_identity=channel_identity,
                sender=user_identity,
            )
            
            # Determine message type
            msg_type = message_data.get("msg_type", "")
            message_type = MessageType.TEXT
            if msg_type == "image":
                message_type = MessageType.IMAGE
            elif msg_type == "file":
                message_type = MessageType.FILE
            elif msg_type == "audio":
                message_type = MessageType.AUDIO
            elif msg_type == "video":
                message_type = MessageType.VIDEO
            elif msg_type == "post":
                message_type = MessageType.HTML
            
            # Create content
            content = message_data.get("content", "{}")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {"text": content}
            
            text = content.get("text", "") if isinstance(content, dict) else str(content)
            message_content = MessageContent.from_text(text)
            
            return UniversalMessage(
                content=message_content,
                message_type=message_type,
                direction=MessageDirection.INCOMING,
                metadata=metadata,
            )
        except Exception as e:
            self._logger.error(f"Error converting Feishu message: {e}")
            return None
    
    # ============= Message Sending =============
    
    async def _send_impl(
        self,
        message: UniversalMessage,
        priority: MessagePriority,
    ) -> SendResult:
        """Implementation of send message."""
        try:
            # Get target
            receive_id = message.get_context("receive_id")
            chat_id = (
                message.metadata.channel_identity.channel_id
                if message.metadata and message.metadata.channel_identity
                else None
            )
            
            # Determine receive_id_type
            receive_id_type = message.get_context("receive_id_type", "open_id")
            msg_type = message.get_context("msg_type", "text")
            
            # Build content
            content = self._build_message_content(message)
            
            # Send message
            result = await self.send_message(
                receive_id=receive_id or chat_id or "",
                receive_id_type=receive_id_type,
                msg_type=msg_type,
                content=content,
            )
            
            return SendResult(
                success=True,
                message_id=result.get("data", {}).get("message_id"),
                timestamp=time.time(),
                metadata={"result": result},
            )
        except FeishuError as e:
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
    
    def _build_message_content(self, message: UniversalMessage) -> Dict[str, Any]:
        """Build message content for Feishu API."""
        msg_type = message.get_context("msg_type", "text")
        
        if msg_type == "text":
            return {"text": message.content.get_primary_text()}
        elif msg_type == "post":
            return {
                "post": {
                    "zh_cn": {
                        "title": message.content.text or "",
                        "content": [
                            [
                                {"tag": "text", "text": message.content.get_primary_text()}
                            ]
                        ]
                    }
                }
            }
        elif msg_type == "image":
            file_key = message.get_context("image_key")
            return {"image_key": file_key} if file_key else {"image_key": ""}
        elif msg_type == "file":
            file_key = message.get_context("file_key")
            return {"file_key": file_key} if file_key else {"file_key": ""}
        elif msg_type == "interactive":
            card = message.get_context("card")
            return {"card": json.dumps(card)} if card else {"card": "{}"}
        
        return {"text": message.content.get_primary_text()}
    
    # ============= Send Message Methods =============
    
    async def send_message(
        self,
        receive_id: str,
        msg_type: str,
        content: Union[Dict[str, Any], str],
        receive_id_type: str = "open_id",
    ) -> Dict[str, Any]:
        """Send a message to a user or chat."""
        await self._get_access_token()
        
        if isinstance(content, dict):
            content = json.dumps(content)
        
        payload = {
            "receive_id": receive_id,
            "receive_id_type": receive_id_type,
            "msg_type": msg_type,
            "content": content,
        }
        
        return await self._make_request(
            "POST",
            "/im/v1/messages",
            payload,
            params={"receive_id_type": receive_id_type},
        )
    
    async def send_text(
        self,
        receive_id: str,
        text: str,
        receive_id_type: str = "open_id",
    ) -> Dict[str, Any]:
        """Send a text message."""
        return await self.send_message(
            receive_id=receive_id,
            msg_type="text",
            content={"text": text},
            receive_id_type=receive_id_type,
        )
    
    async def send_image(
        self,
        receive_id: str,
        image_key: str,
        receive_id_type: str = "open_id",
    ) -> Dict[str, Any]:
        """Send an image message."""
        return await self.send_message(
            receive_id=receive_id,
            msg_type="image",
            content={"image_key": image_key},
            receive_id_type=receive_id_type,
        )
    
    async def send_card(
        self,
        receive_id: str,
        card: Dict[str, Any],
        receive_id_type: str = "open_id",
    ) -> Dict[str, Any]:
        """Send an interactive card."""
        return await self.send_message(
            receive_id=receive_id,
            msg_type="interactive",
            content={"card": json.dumps(card)},
            receive_id_type=receive_id_type,
        )
    
    async def reply_message(
        self,
        message_id: str,
        msg_type: str,
        content: Union[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """Reply to a message."""
        await self._get_access_token()
        
        if isinstance(content, dict):
            content = json.dumps(content)
        
        payload = {
            "msg_type": msg_type,
            "content": content,
        }
        
        return await self._make_request(
            "PATCH",
            f"/im/v1/messages/{message_id}/reply",
            payload,
        )
    
    async def update_message(
        self,
        message_id: str,
        msg_type: str,
        content: Union[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """Update a message."""
        await self._get_access_token()
        
        if isinstance(content, dict):
            content = json.dumps(content)
        
        payload = {
            "msg_type": msg_type,
            "content": content,
        }
        
        return await self._make_request(
            "PUT",
            f"/im/v1/messages/{message_id}",
            payload,
        )
    
    async def delete_message(self, message_id: str) -> bool:
        """Delete a message."""
        await self._get_access_token()
        await self._make_request("DELETE", f"/im/v1/messages/{message_id}")
        return True
    
    # ============= File Methods =============
    
    async def upload_image(self, image_data: bytes, image_name: str = "image.jpg") -> Dict[str, Any]:
        """Upload an image and get image_key."""
        await self._get_access_token()
        
        return await self._make_request(
            "POST",
            "/im/v1/images",
            None,
            files=[("image", image_data, image_name)],
        )
    
    async def download_image(self, image_key: str) -> Dict[str, Any]:
        """Download an image."""
        await self._get_access_token()
        
        return await self._make_request(
            "GET",
            f"/im/v1/images/{image_key}",
        )
    
    async def upload_file(
        self,
        file_data: bytes,
        file_name: str,
        file_type: str = "stream",
    ) -> Dict[str, Any]:
        """Upload a file and get file_key."""
        await self._get_access_token()
        
        return await self._make_request(
            "POST",
            "/im/v1/files",
            None,
            files=[("file", file_data, file_name)],
        )
    
    # ============= Bot & Chat Methods =============
    
    async def get_bot_info(self) -> Dict[str, Any]:
        """Get bot information."""
        await self._get_access_token()
        
        return await self._make_request("GET", "/bot/v3/info")
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get chat information."""
        await self._get_access_token()
        
        return await self._make_request("GET", f"/im/v1/chats/{chat_id}")
    
    async def get_chat_members(
        self,
        chat_id: str,
        member_id_type: str = "open_id",
    ) -> Dict[str, Any]:
        """Get chat members."""
        await self._get_access_token()
        
        return await self._make_request(
            "GET",
            f"/im/v1/chats/{chat_id}/members",
            params={"member_id_type": member_id_type},
        )
    
    async def create_chat(
        self,
        name: str,
        description: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a chat."""
        await self._get_access_token()
        
        payload: Dict[str, Any] = {"name": name}
        
        if description:
            payload["description"] = description
        
        if user_ids:
            payload["user_ids"] = user_ids
        
        return await self._make_request("POST", "/im/v1/chats", payload)
    
    async def update_chat(
        self,
        chat_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update chat information."""
        await self._get_access_token()
        
        payload: Dict[str, Any] = {}
        
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        
        await self._make_request("PUT", f"/im/v1/chats/{chat_id}", payload)
        return True
    
    async def add_chat_members(
        self,
        chat_id: str,
        user_ids: List[str],
        member_id_type: str = "open_id",
    ) -> bool:
        """Add members to a chat."""
        await self._get_access_token()
        
        payload = {
            "id_list": user_ids,
            "member_id_type": member_id_type,
        }
        
        await self._make_request("POST", f"/im/v1/chats/{chat_id}/members", payload)
        return True
    
    async def remove_chat_members(
        self,
        chat_id: str,
        user_ids: List[str],
        member_id_type: str = "open_id",
    ) -> bool:
        """Remove members from a chat."""
        await self._get_access_token()
        
        payload = {
            "id_list": user_ids,
            "member_id_type": member_id_type,
        }
        
        await self._make_request("DELETE", f"/im/v1/chats/{chat_id}/members", payload)
        return True
    
    # ============= User Methods =============
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information."""
        await self._get_access_token()
        
        return await self._make_request("GET", f"/contact/v3/users/{user_id}")
    
    async def batch_get_user_info(
        self,
        user_ids: List[str],
    ) -> Dict[str, Any]:
        """Batch get user information."""
        await self._get_access_token()
        
        return await self._make_request(
            "POST",
            "/contact/v3/users/batch_get_id",
            {"user_ids": user_ids},
        )
    
    async def get_user_by_email(self, email: str) -> Dict[str, Any]:
        """Get user by email."""
        await self._get_access_token()
        
        return await self._make_request(
            "POST",
            "/contact/v3/users/batch_get_id",
            {"emails": [email]},
        )
    
    # ============= Department Methods =============
    
    async def get_department_info(self, department_id: str) -> Dict[str, Any]:
        """Get department information."""
        await self._get_access_token()
        
        return await self._make_request(
            "GET",
            f"/contact/v3/departments/{department_id}",
        )
    
    async def get_department_users(
        self,
        department_id: str,
        user_id_type: str = "open_id",
    ) -> Dict[str, Any]:
        """Get users in a department."""
        await self._get_access_token()
        
        return await self._make_request(
            "GET",
            f"/contact/v3/users/find_by_department",
            params={
                "department_id": department_id,
                "user_id_type": user_id_type,
            },
        )
    
    # ============= Utility =============
    
    def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get channel info (placeholder)."""
        return {}
    
    def __repr__(self) -> str:
        return f"FeishuAdapter(app_id={self._feishu_config.app_id})"


class FeishuError(Exception):
    """Feishu API error."""
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Feishu API Error {code}: {message}")
