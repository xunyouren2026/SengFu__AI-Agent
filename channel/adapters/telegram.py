"""
AGI Unified Framework - Telegram Adapter Module

This module provides a complete Telegram Bot API adapter for the channel framework.

Features:
- Bot API v5/v6 support
- Webhook and Polling modes
- All message types (text, photo, video, audio, document, sticker, location, etc.)
- Inline keyboards and keyboards
- Callback queries
- Inline queries
- Chat management
- User and chat information
- Media group support
- Edit message functionality

Author: AGI Team
License: Apache 2.0
"""

from __future__ import annotations

import asyncio
import hashlib
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
class TelegramConfig(ChannelConfig):
    """
    Configuration for Telegram adapter.
    
    Attributes:
        bot_token: Telegram Bot API token
        api_base_url: Base URL for Telegram API
        webhook_secret: Secret for webhook verification
        proxy_url: Optional proxy URL
        poll_timeout: Timeout for polling in seconds
        max_connections: Maximum connections for webhook
        allowed_updates: List of update types to receive
        drop_pending_updates: Whether to drop pending updates
    """
    bot_token: str = ""
    api_base_url: str = "https://api.telegram.org"
    webhook_secret: str = ""
    proxy_url: Optional[str] = None
    poll_timeout: float = 60.0
    max_connections: int = 100
    allowed_updates: List[str] = field(default_factory=lambda: [
        "message", "edited_message", "callback_query",
        "inline_query", "chosen_inline_result"
    ])
    drop_pending_updates: bool = False


class TelegramKeyboard:
    """
    Helper class for creating Telegram keyboards.
    """
    
    @staticmethod
    def inline_keyboard(
        buttons: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Create an inline keyboard.
        
        Args:
            buttons: List of button rows, each row is a list of button dicts
            
        Returns:
            Inline keyboard markup dict
        """
        keyboard = []
        for row in buttons:
            button_row = []
            for button in row:
                if "text" in button:
                    btn = {"text": button["text"]}
                    if "url" in button:
                        btn["url"] = button["url"]
                    if "callback_data" in button:
                        btn["callback_data"] = button["callback_data"]
                    if "switch_inline_query" in button:
                        btn["switch_inline_query"] = button["switch_inline_query"]
                    button_row.append(btn)
            keyboard.append(button_row)
        
        return {"inline_keyboard": keyboard}
    
    @staticmethod
    def reply_keyboard(
        buttons: List[List[Dict[str, Any]]],
        resize_keyboard: bool = True,
        one_time_keyboard: bool = False,
        selective: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a reply keyboard.
        
        Args:
            buttons: List of button rows
            resize_keyboard: Whether to resize keyboard
            one_time_keyboard: Whether to hide keyboard after use
            selective: Whether to apply selectively
            
        Returns:
            Reply keyboard markup dict
        """
        keyboard = []
        for row in buttons:
            button_row = []
            for button in row:
                if "text" in button:
                    btn = {"text": button["text"]}
                    if "request_contact" in button:
                        btn["request_contact"] = button["request_contact"]
                    if "request_location" in button:
                        btn["request_location"] = button["request_location"]
                    button_row.append(btn)
            keyboard.append(button_row)
        
        return {
            "keyboard": keyboard,
            "resize_keyboard": resize_keyboard,
            "one_time_keyboard": one_time_keyboard,
            "selective": selective,
        }
    
    @staticmethod
    def remove_keyboard(selective: bool = False) -> Dict[str, Any]:
        """Remove the reply keyboard."""
        return {"remove_keyboard": True, "selective": selective}


class TelegramAdapter(ChannelAdapter):
    """
    Telegram Bot API adapter.
    
    This adapter provides full integration with the Telegram Bot API,
    supporting both webhook and polling modes.
    
    Features:
    - Send all message types
    - Receive all update types
    - Inline keyboards and reply keyboards
    - Callback query handling
    - Inline query handling
    - Chat actions
    - File uploads and downloads
    - Media groups
    
    Example:
        ```python
        # Create adapter
        config = TelegramConfig(
            channel_id="telegram_main",
            bot_token="YOUR_BOT_TOKEN"
        )
        adapter = TelegramAdapter(config)
        
        # Connect and send message
        await adapter.connect()
        result = await adapter.send(
            UniversalMessage(
                content=MessageContent.from_text("Hello from Telegram!")
            ),
            target_chat_id="@my_channel"
        )
        ```
    """
    
    def __init__(self, config: TelegramConfig) -> None:
        """
        Initialize the Telegram adapter.
        
        Args:
            config: Telegram configuration
        """
        super().__init__(config)
        
        self._telegram_config = config
        self._api_base = f"{config.api_base_url}/bot{config.bot_token}"
        
        # Polling state
        self._polling_offset: int = 0
        self._polling_task: Optional[asyncio.Task] = None
        self._stop_polling_event = asyncio.Event()
        
        # Webhook state
        self._webhook_server: Optional[Any] = None
        
        # Callback handlers
        self._callback_handlers: Dict[str, Callable] = {}
        self._inline_handlers: Dict[str, Callable] = {}
        
        # Update handlers
        self._update_handlers: Dict[str, Callable] = {}
    
    def _initialize_capabilities(self) -> None:
        """Initialize Telegram capabilities."""
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.HTML_MESSAGES,
            ChannelCapability.MARKDOWN_MESSAGES,
            ChannelCapability.MEDIA_MESSAGES,
            ChannelCapability.FILE_ATTACHMENTS,
            ChannelCapability.EMOJI_SUPPORT,
            ChannelCapability.THREADING,
            ChannelCapability.CHANNEL_TYPES,
            ChannelCapability.DIRECT_MESSAGES,
            ChannelCapability.GROUPS,
            ChannelCapability.WEBHOOK_MODE,
            ChannelCapability.POLLING_MODE,
            ChannelCapability.EDIT_MESSAGES,
            ChannelCapability.DELETE_MESSAGES,
            ChannelCapability.PUSH_NOTIFICATIONS,
            ChannelCapability.TYPING_INDICATOR,
            ChannelCapability.BOT_COMMANDS,
            ChannelCapability.BUTTONS,
            ChannelCapability.INTERACTIVE_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
            ChannelCapability.USER_INFO,
            ChannelCapability.RATE_LIMITING,
            ChannelCapability.AUTO_RECONNECT,
        }
    
    # ============= API Methods =============
    
    async def _make_request(
        self,
        method: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make a request to the Telegram API.
        
        Args:
            method: API method name
            data: Request data
            files: Files to upload
            timeout: Request timeout
            
        Returns:
            API response
        """
        import aiohttp
        
        url = f"{self._api_base}/{method}"
        
        timeout_value = timeout or self._config.timeout
        
        async with asyncio.timeout(timeout_value):
            if files:
                form_data = aiohttp.FormData()
                for key, value in (data or {}).items():
                    form_data.add_field(key, str(value) if value is not None else "")
                for key, file_info in files.items():
                    if isinstance(file_info, tuple):
                        filename, file_content, content_type = file_info
                        form_data.add_field(
                            key,
                            file_content,
                            filename=filename,
                            content_type=content_type,
                        )
                    else:
                        form_data.add_field(key, str(file_info))
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=form_data) as response:
                        result = await response.json()
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        result = await response.json()
        
        if not result.get("ok"):
            error_code = result.get("error_code", "unknown")
            description = result.get("description", "Unknown error")
            raise TelegramError(error_code, description)
        
        return result.get("result", {})
    
    # ============= Connection =============
    
    async def _connect_impl(self) -> bool:
        """Implementation of connect."""
        try:
            # Get bot info to verify token
            bot_info = await self.get_me()
            self._logger.info(f"Connected to Telegram bot: {bot_info.get('username')}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to Telegram: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """Implementation of disconnect."""
        self.stop_polling()
        
        if self._webhook_server:
            # Close webhook server
            self._webhook_server = None
    
    async def _health_check_impl(self) -> bool:
        """Implementation of health check."""
        try:
            await self.get_me()
            return True
        except Exception:
            return False
    
    # ============= Message Sending =============
    
    async def _send_impl(
        self,
        message: UniversalMessage,
        priority: MessagePriority,
    ) -> SendResult:
        """
        Implementation of send message.
        
        Args:
            message: The message to send
            priority: Message priority
            
        Returns:
            SendResult
        """
        try:
            # Get target chat ID from metadata or context
            target_chat_id = message.metadata.channel_identity.channel_id if message.metadata and message.metadata.channel_identity else None
            reply_to = message.metadata.reply_to if message.metadata else None
            
            if not target_chat_id:
                target_chat_id = message.get_context("chat_id") or message.get_context("target_chat_id")
            
            if not target_chat_id:
                return SendResult(
                    success=False,
                    error="No target chat_id provided",
                    error_code="MISSING_TARGET",
                )
            
            # Prepare message parameters
            params = self._prepare_message_params(message)
            
            if reply_to:
                params["reply_to_message_id"] = reply_to
            
            # Handle reply markup
            if message.keyboard:
                params["reply_markup"] = message.keyboard
            elif message.quick_replies:
                params["reply_markup"] = self._prepare_quick_reply_markup(message.quick_replies)
            
            # Determine send method based on content
            method = self._get_send_method(message)
            
            # Send message
            result = await self._make_request(method, params)
            
            return SendResult(
                success=True,
                message_id=str(result.get("message_id", "")),
                timestamp=result.get("date", time.time()),
                metadata={
                    "chat_id": result.get("chat", {}).get("id"),
                    "message": result,
                },
            )
            
        except TelegramError as e:
            return SendResult(
                success=False,
                error=e.description,
                error_code=str(e.error_code),
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                error_code=type(e).__name__,
            )
    
    def _prepare_message_params(self, message: UniversalMessage) -> Dict[str, Any]:
        """Prepare parameters for sending a message."""
        params: Dict[str, Any] = {}
        
        # Text content
        text = self._get_message_text(message)
        if text:
            params["text"] = text
        
        # Parse mode
        parse_mode = self._get_parse_mode(message)
        if parse_mode:
            params["parse_mode"] = parse_mode
        
        # Disable web page preview
        if message.get_context("disable_web_page_preview"):
            params["disable_web_page_preview"] = True
        
        # Disable notification
        if message.get_context("disable_notification"):
            params["disable_notification"] = True
        
        # Protect content
        if message.get_context("protect_content"):
            params["protect_content"] = True
        
        return params
    
    def _get_message_text(self, message: UniversalMessage) -> Optional[str]:
        """Get the text content for a message."""
        if message.content.html:
            return message.content.html
        elif message.content.markdown:
            return message.content.markdown
        elif message.content.text:
            return message.content.text
        return None
    
    def _get_parse_mode(self, message: UniversalMessage) -> Optional[str]:
        """Determine the parse mode based on content."""
        if message.content.html:
            return "HTML"
        elif message.content.markdown:
            return "Markdown"
        return None
    
    def _get_send_method(self, message: UniversalMessage) -> str:
        """Determine the send method based on message type."""
        msg_type = message.message_type
        
        if msg_type == MessageType.IMAGE:
            return "sendPhoto"
        elif msg_type == MessageType.VIDEO:
            return "sendVideo"
        elif msg_type == MessageType.AUDIO:
            return "sendAudio"
        elif msg_type == MessageType.FILE:
            return "sendDocument"
        elif msg_type == MessageType.STICKER:
            return "sendSticker"
        elif msg_type == MessageType.LOCATION:
            return "sendLocation"
        elif msg_type == MessageType.VOICE:
            return "sendVoice"
        elif msg_type == MessageType.VIDEO_NOTE:
            return "sendVideoNote"
        elif msg_type == MessageType.VENUE:
            return "sendVenue"
        elif msg_type == MessageType.CONTACT:
            return "sendContact"
        
        return "sendMessage"
    
    def _prepare_quick_reply_markup(self, quick_replies: List[str]) -> Dict[str, Any]:
        """Prepare quick reply markup from quick reply list."""
        keyboard = []
        row = []
        
        for i, reply in enumerate(quick_replies):
            try:
                reply_data = json.loads(reply)
                text = reply_data.get("text", reply)
            except (json.JSONDecodeError, TypeError):
                text = reply
            
            row.append({"text": text})
            
            if len(row) >= 3:  # Max 3 buttons per row
                keyboard.append(row)
                row = []
        
        if row:
            keyboard.append(row)
        
        return {"keyboard": keyboard, "resize_keyboard": True}
    
    # ============= Message Receiving =============
    
    async def _receive_impl(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> ReceiveResult:
        """
        Implementation of receive messages.
        
        Args:
            payload: Optional webhook payload
            
        Returns:
            ReceiveResult
        """
        if payload:
            # Process webhook payload
            return await self._process_update(payload)
        else:
            # Polling would be handled by the polling task
            return ReceiveResult(
                success=True,
                messages=[],
            )
    
    async def _process_update(self, update: Dict[str, Any]) -> ReceiveResult:
        """
        Process a Telegram update.
        
        Args:
            update: The update from Telegram
            
        Returns:
            ReceiveResult with messages
        """
        messages = []
        
        # Process based on update type
        if "message" in update:
            msg = await self._process_message(update["message"])
            if msg:
                messages.append(msg)
        
        elif "edited_message" in update:
            msg = await self._process_edited_message(update["edited_message"])
            if msg:
                messages.append(msg)
        
        elif "callback_query" in update:
            await self._process_callback_query(update["callback_query"])
        
        elif "inline_query" in update:
            await self._process_inline_query(update["inline_query"])
        
        elif "chosen_inline_result" in update:
            await self._process_chosen_inline_result(update["chosen_inline_result"])
        
        return ReceiveResult(
            success=True,
            messages=messages,
            raw_payload=update,
            metadata={"update_id": update.get("update_id")},
        )
    
    async def _process_message(self, data: Dict[str, Any]) -> Optional[UniversalMessage]:
        """Process a message update."""
        try:
            # Create user identity
            from_user = data.get("from")
            user_identity = None
            if from_user:
                user_identity = UserIdentity(
                    user_id=str(from_user.get("id", "")),
                    username=from_user.get("username"),
                    first_name=from_user.get("first_name"),
                    last_name=from_user.get("last_name"),
                    is_bot=from_user.get("is_bot", False),
                    language=from_user.get("language_code"),
                )
            
            # Create channel identity
            chat = data.get("chat", {})
            channel_identity = ChannelIdentity(
                channel_id=str(chat.get("id", "")),
                channel_type="private" if chat.get("type") == "private" else "group",
                channel_name=chat.get("title") or chat.get("username"),
            )
            
            # Determine message type
            message_type = self._determine_message_type(data)
            
            # Create content
            content = self._extract_content(data, message_type)
            
            # Create metadata
            metadata = MessageMetadata(
                message_id=str(data.get("message_id", "")),
                conversation_id=str(chat.get("id", "")),
                reply_to=data.get("reply_to_message", {}).get("message_id"),
                channel_identity=channel_identity,
                sender=user_identity,
                edited_at=data.get("edit_date"),
                thread_id=data.get("is_topic_message") and str(data.get("message_thread_id", "")),
            )
            
            # Extract attachments
            attachments = self._extract_attachments(data)
            
            # Extract entities (mentions, hashtags, etc.)
            entities = data.get("entities", []) or data.get("caption_entities", []) or []
            for entity in entities:
                if entity.get("type") == "mention":
                    metadata.mentions.append(entity.get("user", {}).get("id", ""))
                elif entity.get("type") == "hashtag":
                    metadata.hashtags.append(
                        data.get("text", data.get("caption", ""))[
                            entity["offset"]:entity["offset"] + entity["length"]
                        ]
                    )
            
            # Create message
            message = UniversalMessage(
                content=content,
                message_type=message_type,
                direction=MessageDirection.INCOMING,
                metadata=metadata,
                attachments=attachments,
            )
            
            return message
            
        except Exception as e:
            self._logger.error(f"Error processing message: {e}")
            return None
    
    async def _process_edited_message(self, data: Dict[str, Any]) -> Optional[UniversalMessage]:
        """Process an edited message."""
        message = await self._process_message(data)
        if message:
            message.message_type = MessageType.EDITED
            message.metadata.edited_at = data.get("edit_date")
        return message
    
    async def _process_callback_query(self, query: Dict[str, Any]) -> None:
        """Process a callback query."""
        query_id = query.get("id")
        callback_data = query.get("data")
        
        if callback_data and callback_data in self._callback_handlers:
            handler = self._callback_handlers[callback_data]
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(query)
                else:
                    handler(query)
            except Exception as e:
                self._logger.error(f"Callback handler error: {e}")
        
        # Answer callback query
        await self._make_request("answerCallbackQuery", {"callback_query_id": query_id})
    
    async def _process_inline_query(self, query: Dict[str, Any]) -> None:
        """Process an inline query."""
        query_id = query.get("id")
        query_text = query.get("query", "")
        
        # Find matching handler
        for pattern, handler in self._inline_handlers.items():
            if pattern in query_text or pattern == "*":
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(query)
                    else:
                        handler(query)
                except Exception as e:
                    self._logger.error(f"Inline query handler error: {e}")
    
    async def _process_chosen_inline_result(self, result: Dict[str, Any]) -> None:
        """Process a chosen inline result."""
        # This can be used for tracking analytics
        pass
    
    def _determine_message_type(self, data: Dict[str, Any]) -> MessageType:
        """Determine the message type from Telegram data."""
        if data.get("text"):
            return MessageType.TEXT
        elif data.get("photo"):
            return MessageType.IMAGE
        elif data.get("video"):
            return MessageType.VIDEO
        elif data.get("audio"):
            return MessageType.AUDIO
        elif data.get("document"):
            return MessageType.FILE
        elif data.get("sticker"):
            return MessageType.STICKER
        elif data.get("location"):
            return MessageType.LOCATION
        elif data.get("venue"):
            return MessageType.VENUE
        elif data.get("contact"):
            return MessageType.CONTACT
        elif data.get("voice"):
            return MessageType.VOICE
        elif data.get("video_note"):
            return MessageType.VIDEO_NOTE
        elif data.get("new_chat_members"):
            return MessageType.SYSTEM
        elif data.get("left_chat_member"):
            return MessageType.SYSTEM
        elif data.get("new_chat_title"):
            return MessageType.SYSTEM
        elif data.get("pinned_message"):
            return MessageType.NOTIFICATION
        elif data.get("game"):
            return MessageType.INTERACTIVE
        elif data.get("invoice"):
            return MessageType.INTERACTIVE
        elif data.get("successful_payment"):
            return MessageType.NOTIFICATION
        elif data.get("passport_data"):
            return MessageType.FILE
        elif data.get("poll"):
            return MessageType.VOTE
        elif data.get("dice"):
            return MessageType.INTERACTIVE
        elif data.get("animation"):
            return MessageType.ANIMATION
        
        return MessageType.TEXT
    
    def _extract_content(self, data: Dict[str, Any], msg_type: MessageType) -> MessageContent:
        """Extract content from Telegram message data."""
        text = data.get("text")
        caption = data.get("caption")
        
        if msg_type == MessageType.IMAGE and caption:
            text = caption
        elif msg_type == MessageType.VIDEO and caption:
            text = caption
        elif msg_type == MessageType.FILE and caption:
            text = caption
        
        if text:
            return MessageContent.from_text(text)
        
        return MessageContent(text="")
    
    def _extract_attachments(self, data: Dict[str, Any]) -> List[Attachment]:
        """Extract attachments from Telegram message data."""
        attachments = []
        
        # Photo
        if data.get("photo"):
            photos = data["photo"]
            largest = photos[-1] if isinstance(photos, list) else photos
            attachments.append(Attachment(
                attachment_id=str(largest.get("file_id", "")),
                attachment_type=AttachmentType.IMAGE,
                file_id=largest.get("file_id"),
                file_size=largest.get("file_size"),
                width=largest.get("width"),
                height=largest.get("height"),
            ))
        
        # Video
        if data.get("video"):
            video = data["video"]
            attachments.append(Attachment(
                attachment_id=str(video.get("file_id", "")),
                attachment_type=AttachmentType.VIDEO,
                file_id=video.get("file_id"),
                file_size=video.get("file_size"),
                duration=video.get("duration"),
                width=video.get("width"),
                height=video.get("height"),
                mime_type=video.get("mime_type"),
            ))
        
        # Audio
        if data.get("audio"):
            audio = data["audio"]
            attachments.append(Attachment(
                attachment_id=str(audio.get("file_id", "")),
                attachment_type=AttachmentType.AUDIO,
                file_id=audio.get("file_id"),
                file_size=audio.get("file_size"),
                duration=audio.get("duration"),
                mime_type=audio.get("mime_type"),
            ))
        
        # Document
        if data.get("document"):
            doc = data["document"]
            attachments.append(Attachment(
                attachment_id=str(doc.get("file_id", "")),
                attachment_type=AttachmentType.DOCUMENT,
                file_id=doc.get("file_id"),
                file_name=doc.get("file_name"),
                file_size=doc.get("file_size"),
                mime_type=doc.get("mime_type"),
            ))
        
        # Voice
        if data.get("voice"):
            voice = data["voice"]
            attachments.append(Attachment(
                attachment_id=str(voice.get("file_id", "")),
                attachment_type=AttachmentType.VOICE,
                file_id=voice.get("file_id"),
                file_size=voice.get("file_size"),
                duration=voice.get("duration"),
                mime_type=voice.get("mime_type"),
            ))
        
        # Video note
        if data.get("video_note"):
            video_note = data["video_note"]
            attachments.append(Attachment(
                attachment_id=str(video_note.get("file_id", "")),
                attachment_type=AttachmentType.VIDEO,
                file_id=video_note.get("file_id"),
                file_size=video_note.get("file_size"),
                duration=video_note.get("duration"),
                width=video_note.get("length"),
                height=video_note.get("length"),
            ))
        
        # Sticker
        if data.get("sticker"):
            sticker = data["sticker"]
            attachments.append(Attachment(
                attachment_id=str(sticker.get("file_id", "")),
                attachment_type=AttachmentType.STICKER,
                file_id=sticker.get("file_id"),
                width=sticker.get("width"),
                height=sticker.get("height"),
                metadata={
                    "set_name": sticker.get("set_name"),
                    "is_animated": sticker.get("is_animated"),
                    "is_video": sticker.get("is_video"),
                },
            ))
        
        # Location
        if data.get("location"):
            loc = data["location"]
            attachments.append(Attachment(
                attachment_id=f"loc_{loc.get('latitude')}_{loc.get('longitude')}",
                attachment_type=AttachmentType.LOCATION,
                metadata={
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                    "horizontal_accuracy": loc.get("horizontal_accuracy"),
                    "heading": loc.get("heading"),
                    "live_period": loc.get("live_period"),
                },
            ))
        
        # Venue
        if data.get("venue"):
            venue = data["venue"]
            location = venue.get("location", {})
            attachments.append(Attachment(
                attachment_id=f"venue_{venue.get('title', '')}",
                attachment_type=AttachmentType.LOCATION,
                metadata={
                    "title": venue.get("title"),
                    "address": venue.get("address"),
                    "foursquare_id": venue.get("foursquare_id"),
                    "foursquare_type": venue.get("foursquare_type"),
                    "google_place_id": venue.get("google_place_id"),
                    "google_place_type": venue.get("google_place_type"),
                    "latitude": location.get("latitude"),
                    "longitude": location.get("longitude"),
                },
            ))
        
        # Contact
        if data.get("contact"):
            contact = data["contact"]
            attachments.append(Attachment(
                attachment_id=f"contact_{contact.get('user_id', '')}",
                attachment_type=AttachmentType.CONTACT,
                metadata={
                    "phone_number": contact.get("phone_number"),
                    "first_name": contact.get("first_name"),
                    "last_name": contact.get("last_name"),
                    "user_id": contact.get("user_id"),
                    "vcard": contact.get("vcard"),
                },
            ))
        
        return attachments
    
    # ============= Polling =============
    
    async def start_polling(self) -> None:
        """Start polling for updates."""
        if self._polling_task and not self._polling_task.done():
            return
        
        self._stop_polling_event.clear()
        self._polling_task = asyncio.create_task(self._polling_loop())
    
    def stop_polling(self) -> None:
        """Stop polling for updates."""
        self._stop_polling_event.set()
        if self._polling_task:
            self._polling_task.cancel()
    
    async def _polling_loop(self) -> None:
        """Main polling loop."""
        self._logger.info("Starting polling loop")
        
        while not self._stop_polling_event.is_set():
            try:
                updates = await self._get_updates()
                
                for update in updates:
                    asyncio.create_task(self._process_update(update))
                    
                    # Update offset
                    update_id = update.get("update_id", 0)
                    self._polling_offset = max(self._polling_offset, update_id + 1)
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _get_updates(self) -> List[Dict[str, Any]]:
        """Get updates from Telegram."""
        params = {
            "offset": self._polling_offset,
            "timeout": int(self._telegram_config.poll_timeout),
            "allowed_updates": self._telegram_config.allowed_updates,
        }
        
        if self._telegram_config.drop_pending_updates:
            params["offset"] = -1
        
        return await self._make_request("getUpdates", params)
    
    # ============= Webhook =============
    
    async def start_webhook_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        path: str = "/webhook",
    ) -> None:
        """Start a webhook server."""
        from aiohttp import web
        
        app = web.Application()
        
        async def handle_webhook(request):
            try:
                update = await request.json()
                await self._process_update(update)
                return web.Response(text="OK")
            except Exception as e:
                self._logger.error(f"Webhook error: {e}")
                return web.Response(text="ERROR", status=500)
        
        app.router.add_post(path, handle_webhook)
        
        # Set webhook
        webhook_url = f"https://{self._telegram_config.webhook_url}{path}"
        await self._set_webhook(webhook_url)
        
        self._webhook_server = web.AppRunner(app)
        await self._webhook_server.setup()
        site = web.TCPSite(self._webhook_server, host, port)
        await site.start()
        
        self._logger.info(f"Webhook server started on {host}:{port}{path}")
    
    async def _set_webhook(self, url: str) -> None:
        """Set the webhook URL."""
        params = {
            "url": url,
            "max_connections": self._telegram_config.max_connections,
            "allowed_updates": self._telegram_config.allowed_updates,
            "drop_pending_updates": self._telegram_config.drop_pending_updates,
        }
        
        if self._telegram_config.webhook_secret:
            params["secret_token"] = self._telegram_config.webhook_secret
        
        await self._make_request("setWebhook", params)
    
    # ============= Bot API Methods =============
    
    async def get_me(self) -> Dict[str, Any]:
        """Get bot information."""
        return await self._make_request("getMe")
    
    async def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: Optional[str] = None,
        disable_web_page_preview: bool = False,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a text message."""
        params = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_page_preview,
            "disable_notification": disable_notification,
        }
        
        if parse_mode:
            params["parse_mode"] = parse_mode
        
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        return await self._make_request("sendMessage", params)
    
    async def send_photo(
        self,
        chat_id: Union[int, str],
        photo: Union[str, Tuple[str, bytes]],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a photo."""
        params = {
            "chat_id": chat_id,
            "caption": caption,
            "disable_notification": disable_notification,
        }
        
        if parse_mode:
            params["parse_mode"] = parse_mode
        
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        files = {}
        if isinstance(photo, str):
            params["photo"] = photo
        else:
            filename, content = photo
            files["photo"] = (filename, content, "image/jpeg")
        
        return await self._make_request("sendPhoto", params, files if files else None)
    
    async def send_document(
        self,
        chat_id: Union[int, str],
        document: Union[str, Tuple[str, bytes]],
        caption: Optional[str] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a document."""
        params = {
            "chat_id": chat_id,
            "caption": caption,
            "disable_notification": disable_notification,
        }
        
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        files = {}
        if isinstance(document, str):
            params["document"] = document
        else:
            filename, content = document
            files["document"] = (filename, content, "application/octet-stream")
        
        return await self._make_request("sendDocument", params, files if files else None)
    
    async def send_location(
        self,
        chat_id: Union[int, str],
        latitude: float,
        longitude: float,
        horizontal_accuracy: Optional[float] = None,
        heading: Optional[int] = None,
        live_period: Optional[int] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a location."""
        params = {
            "chat_id": chat_id,
            "latitude": latitude,
            "longitude": longitude,
            "disable_notification": disable_notification,
        }
        
        if horizontal_accuracy is not None:
            params["horizontal_accuracy"] = horizontal_accuracy
        if heading is not None:
            params["heading"] = heading
        if live_period is not None:
            params["live_period"] = live_period
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        return await self._make_request("sendLocation", params)
    
    async def send_venue(
        self,
        chat_id: Union[int, str],
        latitude: float,
        longitude: float,
        title: str,
        address: str,
        foursquare_id: Optional[str] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a venue."""
        params = {
            "chat_id": chat_id,
            "latitude": latitude,
            "longitude": longitude,
            "title": title,
            "address": address,
            "disable_notification": disable_notification,
        }
        
        if foursquare_id:
            params["foursquare_id"] = foursquare_id
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        return await self._make_request("sendVenue", params)
    
    async def send_contact(
        self,
        chat_id: Union[int, str],
        phone_number: str,
        first_name: str,
        last_name: Optional[str] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a contact."""
        params = {
            "chat_id": chat_id,
            "phone_number": phone_number,
            "first_name": first_name,
            "disable_notification": disable_notification,
        }
        
        if last_name:
            params["last_name"] = last_name
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        return await self._make_request("sendContact", params)
    
    async def send_sticker(
        self,
        chat_id: Union[int, str],
        sticker: Union[str, Tuple[str, bytes]],
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a sticker."""
        params = {
            "chat_id": chat_id,
            "disable_notification": disable_notification,
        }
        
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        files = {}
        if isinstance(sticker, str):
            params["sticker"] = sticker
        else:
            filename, content = sticker
            files["sticker"] = (filename, content, "image/webp")
        
        return await self._make_request("sendSticker", params, files if files else None)
    
    async def send_video(
        self,
        chat_id: Union[int, str],
        video: Union[str, Tuple[str, bytes]],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        duration: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a video."""
        params = {
            "chat_id": chat_id,
            "caption": caption,
            "disable_notification": disable_notification,
        }
        
        if parse_mode:
            params["parse_mode"] = parse_mode
        if duration is not None:
            params["duration"] = duration
        if width is not None:
            params["width"] = width
        if height is not None:
            params["height"] = height
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        files = {}
        if isinstance(video, str):
            params["video"] = video
        else:
            filename, content = video
            files["video"] = (filename, content, "video/mp4")
        
        return await self._make_request("sendVideo", params, files if files else None)
    
    async def send_audio(
        self,
        chat_id: Union[int, str],
        audio: Union[str, Tuple[str, bytes]],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        duration: Optional[int] = None,
        performer: Optional[str] = None,
        title: Optional[str] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send an audio file."""
        params = {
            "chat_id": chat_id,
            "caption": caption,
            "disable_notification": disable_notification,
        }
        
        if parse_mode:
            params["parse_mode"] = parse_mode
        if duration is not None:
            params["duration"] = duration
        if performer:
            params["performer"] = performer
        if title:
            params["title"] = title
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        files = {}
        if isinstance(audio, str):
            params["audio"] = audio
        else:
            filename, content = audio
            files["audio"] = (filename, content, "audio/mpeg")
        
        return await self._make_request("sendAudio", params, files if files else None)
    
    async def send_voice(
        self,
        chat_id: Union[int, str],
        voice: Union[str, Tuple[str, bytes]],
        caption: Optional[str] = None,
        duration: Optional[int] = None,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a voice message."""
        params = {
            "chat_id": chat_id,
            "caption": caption,
            "disable_notification": disable_notification,
        }
        
        if duration is not None:
            params["duration"] = duration
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        files = {}
        if isinstance(voice, str):
            params["voice"] = voice
        else:
            filename, content = voice
            files["voice"] = (filename, content, "audio/ogg")
        
        return await self._make_request("sendVoice", params, files if files else None)
    
    async def send_media_group(
        self,
        chat_id: Union[int, str],
        media: List[Dict[str, Any]],
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a media group."""
        # Process media for upload
        processed_media = []
        files = {}
        
        for i, item in enumerate(media):
            media_item = {
                "type": item.get("type", "photo"),
                "caption": item.get("caption"),
            }
            
            if "media" in item:
                media_value = item["media"]
                if isinstance(media_value, str):
                    media_item["media"] = media_value
                else:
                    filename, content = media_value
                    files[f"media{i}"] = (filename, content, "image/jpeg")
                    media_item["media"] = f"attach://media{i}"
            
            processed_media.append(media_item)
        
        params = {
            "chat_id": chat_id,
            "media": json.dumps(processed_media),
            "disable_notification": disable_notification,
        }
        
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        
        return await self._make_request("sendMediaGroup", params, files if files else None)
    
    async def send_chat_action(
        self,
        chat_id: Union[int, str],
        action: str,
    ) -> bool:
        """Send a chat action."""
        return await self._make_request("sendChatAction", {
            "chat_id": chat_id,
            "action": action,
        })
    
    async def edit_message_text(
        self,
        chat_id: Optional[Union[int, str]] = None,
        message_id: Optional[int] = None,
        inline_message_id: Optional[str] = None,
        text: str = "",
        parse_mode: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Edit message text."""
        params = {
            "text": text,
        }
        
        if chat_id:
            params["chat_id"] = chat_id
        if message_id:
            params["message_id"] = message_id
        if inline_message_id:
            params["inline_message_id"] = inline_message_id
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        return await self._make_request("editMessageText", params)
    
    async def edit_message_caption(
        self,
        chat_id: Optional[Union[int, str]] = None,
        message_id: Optional[int] = None,
        inline_message_id: Optional[str] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Edit message caption."""
        params = {
            "caption": caption,
        }
        
        if chat_id:
            params["chat_id"] = chat_id
        if message_id:
            params["message_id"] = message_id
        if inline_message_id:
            params["inline_message_id"] = inline_message_id
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        return await self._make_request("editMessageCaption", params)
    
    async def edit_message_reply_markup(
        self,
        chat_id: Optional[Union[int, str]] = None,
        message_id: Optional[int] = None,
        inline_message_id: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Edit message reply markup."""
        params = {}
        
        if chat_id:
            params["chat_id"] = chat_id
        if message_id:
            params["message_id"] = message_id
        if inline_message_id:
            params["inline_message_id"] = inline_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        
        return await self._make_request("editMessageReplyMarkup", params)
    
    async def delete_message(
        self,
        chat_id: Union[int, str],
        message_id: int,
    ) -> bool:
        """Delete a message."""
        return await self._make_request("deleteMessage", {
            "chat_id": chat_id,
            "message_id": message_id,
        })
    
    async def get_file(self, file_id: str) -> Dict[str, Any]:
        """Get file information."""
        return await self._make_request("getFile", {"file_id": file_id})
    
    async def get_file_url(self, file_id: str) -> Optional[str]:
        """Get the download URL for a file."""
        file_info = await self.get_file(file_id)
        file_path = file_info.get("file_path")
        if file_path:
            return f"{self._telegram_config.api_base_url}/file/bot{self._telegram_config.bot_token}/{file_path}"
        return None
    
    # ============= Chat Management =============
    
    async def get_chat(self, chat_id: Union[int, str]) -> Dict[str, Any]:
        """Get chat information."""
        return await self._make_request("getChat", {"chat_id": chat_id})
    
    async def get_chat_member(
        self,
        chat_id: Union[int, str],
        user_id: int,
    ) -> Dict[str, Any]:
        """Get information about a chat member."""
        return await self._make_request("getChatMember", {
            "chat_id": chat_id,
            "user_id": user_id,
        })
    
    async def get_chat_administrators(
        self,
        chat_id: Union[int, str],
    ) -> List[Dict[str, Any]]:
        """Get administrators of a chat."""
        return await self._make_request("getChatAdministrators", {
            "chat_id": chat_id,
        })
    
    async def get_chat_member_count(self, chat_id: Union[int, str]) -> int:
        """Get the member count of a chat."""
        return await self._make_request("getChatMemberCount", {
            "chat_id": chat_id,
        })
    
    async def leave_chat(self, chat_id: Union[int, str]) -> bool:
        """Leave a chat."""
        return await self._make_request("leaveChat", {"chat_id": chat_id})
    
    # ============= User Information =============
    
    async def get_user_profile_photos(
        self,
        user_id: int,
        offset: Optional[int] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get user profile photos."""
        params = {
            "user_id": user_id,
            "limit": limit,
        }
        
        if offset is not None:
            params["offset"] = offset
        
        return await self._make_request("getUserProfilePhotos", params)
    
    # ============= Callback Handling =============
    
    def register_callback_handler(
        self,
        callback_data: str,
        handler: Callable,
    ) -> None:
        """Register a handler for a callback button."""
        self._callback_handlers[callback_data] = handler
    
    def register_inline_handler(
        self,
        pattern: str,
        handler: Callable,
    ) -> None:
        """Register a handler for inline queries."""
        self._inline_handlers[pattern] = handler
    
    # ============= Inline Methods =============
    
    async def answer_inline_query(
        self,
        inline_query_id: str,
        results: List[Dict[str, Any]],
        cache_time: int = 300,
        is_personal: bool = False,
        next_offset: Optional[str] = None,
        switch_pm_text: Optional[str] = None,
        switch_pm_parameter: Optional[str] = None,
    ) -> bool:
        """Answer an inline query."""
        params = {
            "inline_query_id": inline_query_id,
            "results": json.dumps(results),
            "cache_time": cache_time,
            "is_personal": is_personal,
        }
        
        if next_offset is not None:
            params["next_offset"] = next_offset
        if switch_pm_text:
            params["switch_pm_text"] = switch_pm_text
        if switch_pm_parameter:
            params["switch_pm_parameter"] = switch_pm_parameter
        
        return await self._make_request("answerInlineQuery", params)
    
    def create_inline_result_article(
        self,
        result_id: str,
        title: str,
        input_message_content: Dict[str, Any],
        description: Optional[str] = None,
        url: Optional[str] = None,
        thumb_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an inline result article."""
        result = {
            "type": "article",
            "id": result_id,
            "title": title,
            "input_message_content": input_message_content,
        }
        
        if description:
            result["description"] = description
        if url:
            result["url"] = url
        if thumb_url:
            result["thumb_url"] = thumb_url
        
        return result
    
    # ============= Other Methods =============
    
    async def get_updates(
        self,
        offset: Optional[int] = None,
        limit: int = 100,
        timeout: Optional[int] = None,
        allowed_updates: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get updates (manual polling)."""
        params = {
            "limit": limit,
        }
        
        if offset is not None:
            params["offset"] = offset
        if timeout is not None:
            params["timeout"] = timeout
        if allowed_updates:
            params["allowed_updates"] = allowed_updates
        
        return await self._make_request("getUpdates", params)
    
    async def set_game_score(
        self,
        user_id: int,
        score: int,
        chat_id: Optional[Union[int, str]] = None,
        message_id: Optional[int] = None,
        inline_message_id: Optional[str] = None,
        force: bool = False,
        disable_edit_message: bool = False,
    ) -> Dict[str, Any]:
        """Set game score."""
        params = {
            "user_id": user_id,
            "score": score,
            "force": force,
            "disable_edit_message": disable_edit_message,
        }
        
        if chat_id:
            params["chat_id"] = chat_id
        if message_id:
            params["message_id"] = message_id
        if inline_message_id:
            params["inline_message_id"] = inline_message_id
        
        return await self._make_request("setGameScore", params)
    
    async def get_game_high_scores(
        self,
        user_id: int,
        chat_id: Optional[Union[int, str]] = None,
        message_id: Optional[int] = None,
        inline_message_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get game high scores."""
        params = {"user_id": user_id}
        
        if chat_id:
            params["chat_id"] = chat_id
        if message_id:
            params["message_id"] = message_id
        if inline_message_id:
            params["inline_message_id"] = inline_message_id
        
        return await self._make_request("getGameHighScores", params)


class TelegramError(Exception):
    """Telegram API error."""
    
    def __init__(self, error_code: int, description: str):
        self.error_code = error_code
        self.description = description
        super().__init__(f"Telegram API Error {error_code}: {description}")
